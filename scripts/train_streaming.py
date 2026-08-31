#!/usr/bin/env python3
"""Continuous (streaming) training for solar active region segmentation.

Trains an Attention U-Net FOREVER on a growing dataset: a separate downloader
keeps adding frames to the manifest, and this script picks every new frame up
automatically. It starts training as soon as the first frame exists and never
restarts — optimizer state, EMA weights and the LR schedule accumulate across
the whole run, so the model improves continuously from frame #1 and the
longer it runs, the better it gets.

This is the "I want results" mode: pair it with build_arpil_resumable.py
running in the background (scripts/run_forever.sh does this for you).
"""

from __future__ import annotations

import argparse
import json
import random
import signal
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from solar_ar.data import SolarActiveRegionDataset  # noqa: E402
from solar_ar.models import AttentionUNet  # noqa: E402
from solar_ar.runtime import (  # noqa: E402
    amp_settings,
    configure_runtime,
    describe_accelerator,
    preferred_device,
    suggest_batch_size,
)
from solar_ar.training import (  # noqa: E402
    BCEDiceLoss,
    ComboLoss,
    DeepSupervisionLoss,
    FocalTverskyLoss,
    ModelEma,
    compute_metrics_from_probs,
    seed_everything,
)
from solar_ar.tta import tta_predict  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train continuously on a growing manifest.")
    p.add_argument("--manifest", required=True, help="Manifest CSV (re-read whenever it changes)")
    p.add_argument("--channels", nargs="+", default=["aia171"])
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--status-file", default=None, help="Optional human-readable STATUS.md to refresh each epoch")
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=3e-4, help="Base LR (constant after warmup — tuned for long incremental runs)")
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument(
        "--rewarm-epochs", type=int, default=10,
        help="When resuming a checkpoint that recorded its LR, ramp from that LR "
             "up to --lr over this many epochs instead of jumping straight to the "
             "base LR (a jump several-fold above a converged model's final LR can "
             "destroy the weights within a few epochs).",
    )
    p.add_argument("--batch-size", type=int, default=0, help="0 = auto from the memory budget")
    p.add_argument("--val-every", type=int, default=2, help="Validate (and update best.pt) every N epochs")
    p.add_argument(
        "--max-tiles-per-epoch", type=int, default=0,
        help="Randomly train on at most this many tiles per epoch (0 = whole dataset). "
             "Keeps epochs fast and constant as the dataset grows; the full dataset is "
             "still covered across successive epochs. Validation is ALWAYS the full val set.",
    )
    p.add_argument(
        "--val-subset", type=int, default=0,
        help="Validate on a fixed random subset of this size from the val set instead of "
             "the whole val set (0 = full). A full validation on a grown val set with TTA "
             "can take an hour — a 300-tile quick validation takes ~1-2 min and keeps the "
             "best.pt bar fresh. The bar is consistent within a dataset generation; run "
             "evaluate.py on the full val set for the final number.",
    )
    p.add_argument("--max-epochs", type=int, default=0, help="0 = train forever until stopped")
    p.add_argument("--min-samples", type=int, default=4, help="Wait until the manifest has at least this many train samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loss", choices=["bce_dice", "focal_tversky", "combo"], default="bce_dice")
    p.add_argument(
        "--deep-supervision", action="store_true",
        help="Auxiliary losses on intermediate decoder stages: pushes gradient into the "
             "deep layers and improves recall of small/thin structures (active regions "
             "and polarity inversion lines). ~10-15% slower per epoch, usually worth it.",
    )
    p.add_argument("--tta", choices=["none", "flips", "d4"], default="flips")
    p.add_argument("--cpu-budget", type=int, default=0, help="0 = auto (every core minus headroom)")
    p.add_argument("--cpu-headroom", type=int, default=2)
    p.add_argument("--memory-budget-gb", type=float, default=0.0, help="0 = all detected RAM")
    p.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    p.add_argument("--no-torch-compile", dest="torch_compile", action="store_false", default=True)
    p.add_argument(
        "--calibrate-best",
        action="store_true",
        help="After resuming, set the best-Dice bar from the FIRST validation on the "
             "current data instead of the checkpoint's stored value. Use when resuming a "
             "checkpoint whose val Dice was measured on a DIFFERENT dataset/representation "
             "(e.g. a full-disk-thumbnail model resumed on 512 tiles): otherwise the old, "
             "incomparable score stays the bar and best.pt would never update.",
    )
    return p


def build_criterion(name: str, deep_supervision: bool = False):
    if name == "focal_tversky":
        loss = FocalTverskyLoss()
    elif name == "combo":
        loss = ComboLoss()
    else:
        loss = BCEDiceLoss()
    if deep_supervision:
        return DeepSupervisionLoss(loss)
    return loss


def load_last(path: Path, model, optimizer, ema, device) -> tuple[int, float, float | None]:
    """Resume from a checkpoint, tolerating every format this repo has produced.

    New streaming checkpoints store top-level ``epoch``/``best_dice``; the
    classic Trainer checkpoints (e.g. a 50-epoch best.pt from an earlier run)
    store them under ``metrics.epoch``/``best_score``/``metrics.val_dice`` and
    may have no EMA state at all. All of them load here. Also returns the LR
    the checkpoint was trained with (when recorded) so the caller can re-warm
    gently instead of shocking a converged model with the full base LR.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if ema is not None and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"], checkpoint.get("ema_updates", 0))
    epoch = checkpoint.get("epoch")
    if epoch is None:
        epoch = checkpoint.get("metrics", {}).get("epoch", 0)
    best_dice = checkpoint.get("best_dice")
    if best_dice is None:
        best_dice = checkpoint.get("best_score", checkpoint.get("metrics", {}).get("val_dice", -1e9))
    lr = checkpoint.get("learning_rate")
    if lr is None:
        lr = checkpoint.get("metrics", {}).get("learning_rate")
    return int(epoch), float(best_dice), (float(lr) if lr else None)


def save_checkpoint(
    path: Path, model, optimizer, ema, epoch: int, best_dice: float, channels: list[str],
    learning_rate: float | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_dice": best_dice,
        "channels": channels,
    }
    if learning_rate is not None:
        # Stored so a resume can re-warm from exactly this run's current LR.
        payload["learning_rate"] = learning_rate
    if ema is not None:
        payload["ema_state_dict"] = ema.state_dict()
        payload["ema_updates"] = ema.updates
    torch.save(payload, path)


def _manifest_tile_count(manifest: Path) -> int:
    """Number of tile rows in the manifest (cheap: line count minus header)."""
    try:
        with manifest.open() as handle:
            return sum(1 for _ in handle) - 1
    except OSError:
        return -1


def _cgroup_memory() -> tuple[int | None, int | None]:
    """Container memory (usage, limit) in BYTES — None when unavailable.

    Reads cgroup v2 (memory.current / memory.max) or v1 (memory.usage_in_bytes /
    memory.limit_in_bytes). A non-None limit that is absurdly large (>= 1 TiB)
    means "effectively unlimited" and callers should treat it as None.
    """
    usage = limit = None
    pairs = (
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
        ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    )
    for cur_p, lim_p in pairs:
        try:
            usage = int(Path(cur_p).read_text().strip())
        except (OSError, ValueError):
            usage = None
        try:
            raw = Path(lim_p).read_text().strip()
            limit = int(raw) if raw.isdigit() else None
        except OSError:
            limit = None
        if usage is not None:
            break
    if limit is not None and limit >= 1024**4:  # >= 1 TiB -> effectively unlimited
        limit = None
    return usage, limit


def write_status(path: str, epoch: int, n_train: int, n_val: int, line: str, best_dice: float) -> None:
    text = (
        "# Solar AR training — status (continuous mode)\n\n"
        f"_Last updated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}_\n\n"
        f"- Epoch: **{epoch}** (training forever until stopped)\n"
        f"- Dataset: {n_train} train / {n_val} val tiles (grows as frames download)\n"
        f"- Best val Dice: **{best_dice:.4f}**\n"
        f"- Latest: {line}\n"
        f"- Checkpoints: `best.pt` / `last.pt` in this run directory\n"
    )
    Path(path).write_text(text, encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    # Tolerate a single quoted multi-word --channels argument (e.g.
    # --channels "aia94 aia131 ..."), which argparse would otherwise treat
    # as one channel name.
    args.channels = [c for part in args.channels for c in part.split()]
    seed_everything(args.seed)

    device = torch.device(preferred_device())
    plan = configure_runtime(
        cpu_budget=args.cpu_budget,
        memory_budget_gb=args.memory_budget_gb,
        cpu_headroom=args.cpu_headroom,
        use_cuda=device.type == "cuda",
    )
    print(plan.describe(), flush=True)
    print(describe_accelerator(), flush=True)
    autocast_type, amp_enabled, scaler_enabled = amp_settings(device.type)
    print(f"[stream] mixed precision: autocast={autocast_type} enabled={amp_enabled} scaler={scaler_enabled}", flush=True)

    # Multi-GPU detection happens before the batch size: DataParallel trains
    # with a GLOBAL batch that it splits across the GPUs (Kaggle T4x2), while
    # single-GPU runs (Mac/MPS) size the batch for exactly one device.
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    use_dp = num_gpus > 1

    batch_size = args.batch_size or suggest_batch_size(
        args.image_size,
        len(args.channels),
        args.base_channels,
        plan.memory_budget_gb * 0.55,
        data_parallel=use_dp,  # True: GLOBAL batch, DataParallel splits it across the GPUs
    )
    print(f"[stream] batch_size={batch_size} (auto)", flush=True)

    model = AttentionUNet(
        in_channels=len(args.channels),
        out_channels=1,
        base_channels=args.base_channels,
        dropout=args.dropout,
        deep_supervision=args.deep_supervision,
    ).to(device)

    # Multi-GPU (Kaggle's T4x2): nn.DataParallel splits every batch across both
    # GPUs for the forward pass. It is a forward-only wrapper that SHARES the
    # parameter tensors with the raw model, so the optimizer, EMA, gradient
    # clipping, EMA-swap validation and checkpoints all keep operating on
    # `raw_model` below — which keeps state-dict keys clean (no "module."
    # prefix) and every checkpoint portable between this run, the Mac, and the
    # evaluation scripts. Single-GPU runs (Mac/MPS, one-GPU machines) are
    # completely unaffected; any DP setup failure falls back to one GPU.
    raw_model = model
    if use_dp:
        try:
            model = torch.nn.DataParallel(model)
            print(f"[stream] using {num_gpus} GPUs with DataParallel (each batch splits across both)", flush=True)
        except Exception as exc:  # noqa: BLE001 - DP is a speedup, never a requirement
            use_dp = False
            print(f"[stream] DataParallel unavailable ({type(exc).__name__}: {exc}) — continuing on one GPU", flush=True)

    # C-level graph engine with the same smoke-tested fallback as train.py.
    # dynamic=False: shapes are fixed per run (image size, channels, batch
    # size), and static shapes avoid the inductor symbolic-shape codegen bug
    # that breaks torch.compile on MPS ("cannot determine truth value of
    # Relational" in WelfordReduction). Only the final partial batch differs,
    # which costs a couple of one-time re-compiles.
    #
    # SKIPPED ON MPS (Apple Silicon): torch 2.8's inductor-MPS backend is
    # unreliable in practice — each 512^2 graph takes 15-30+ min to compile
    # (one-time re-compiles per batch size), the compile can hang the whole
    # trainer, and the first compiled epoch produced a NaN loss. Eager mode
    # with bfloat16 autocast is the proven, hang-free path on Apple Silicon
    # (CUDA keeps torch.compile — its inductor backend is mature).
    #
    # SKIPPED WITH DataParallel: compiled modules do not mix with the
    # replicate/scatter/gather machinery, so two eager GPUs beat one
    # compiled GPU here.
    compile_enabled = False
    if args.torch_compile and device.type == "mps":
        print("[compile] skipped on MPS (Apple Silicon) — inductor-MPS is unreliable in torch 2.8 (slow/hanging compiles, NaN); using eager + bfloat16 autocast", flush=True)
    elif args.torch_compile and use_dp:
        print("[compile] skipped — torch.compile does not mix with DataParallel; using eager on both GPUs", flush=True)
    elif args.torch_compile:
        # NOTE: this is a one-time cost, paid on EVERY trainer start (including
        # every supervisor restart / cell re-run). It is silent while it runs,
        # so announce it explicitly — on a 4-core box it can take 15-40 min
        # while downloads are competing for CPU.
        print("[compile] compiling model — one-time cost (~1-5 min on 12+ cores; can take 15-40 min on a 4-core box while downloads run) ...", flush=True)
        t_compile = time.time()
        try:
            with torch.no_grad():
                dummy = torch.zeros(2, len(args.channels), min(args.image_size, 64), min(args.image_size, 64), device=device)
                model = torch.compile(model, dynamic=False)
                _ = model(dummy)
            compile_enabled = True
            print(f"[compile] torch.compile active (C-level graph engine; compiled in {time.time() - t_compile:.0f}s — the first real epoch recompiles for 512^2 and is the slow one)", flush=True)
        except Exception as exc:  # noqa: BLE001
            model = AttentionUNet(
                in_channels=len(args.channels), out_channels=1,
                base_channels=args.base_channels, dropout=args.dropout,
                deep_supervision=args.deep_supervision,
            ).to(device)
            raw_model = model  # keep the parameter-level handles on the live model
            print(f"[compile] torch.compile unavailable ({type(exc).__name__}: {exc}) — continuing in eager mode", flush=True)

    # EMA + optimizer on the RAW model: identical tensors in the single-GPU
    # case, and prefix-free state-dict keys in the DataParallel case.
    ema = ModelEma(raw_model, decay=0.999)
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    criterion = build_criterion(args.loss, deep_supervision=args.deep_supervision)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = out_dir / "best.pt", out_dir / "last.pt"
    metrics_path = out_dir / "metrics.jsonl"

    epoch = 0
    best_dice = -1e9
    resume_lr: float | None = None
    if args.resume and last_path.exists():
        try:
            epoch, best_dice, resume_lr = load_last(last_path, raw_model, optimizer, ema, device)
            print(f"[stream] resumed from {last_path} at epoch {epoch} (best_dice={best_dice:.4f})", flush=True)
            if resume_lr is not None:
                print(
                    f"[stream] LR re-warm: ramping {resume_lr:.2e} (checkpoint's LR) -> {args.lr:.2e} "
                    f"over {args.rewarm_epochs} epochs — protects a converged model from an LR shock",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - e.g. architecture mismatch
            epoch, best_dice, resume_lr = 0, -1e9, None
            print(
                f"[stream] WARNING: could not resume from {last_path} "
                f"({type(exc).__name__}: {exc}) — starting fresh. "
                "If this was a checkpoint from a different channel set or "
                "base-channels, it is incompatible with the current model; remove "
                f"{last_path} to silence this warning.",
                flush=True,
            )
    start_epoch = epoch
    # When resuming a model whose stored Dice came from a different dataset
    # (e.g. full-disk thumbnails -> 512 tiles), the old bar is meaningless:
    # calibrate_pending makes the FIRST validation on current data set it.
    calibrate_pending = bool(args.calibrate_best and (epoch > 0 or resume_lr is not None))
    current_lr: float | None = None

    stop = {"flag": False}

    def _signal(signum, frame) -> None:  # noqa: ANN001
        stop["flag"] = True
        print(f"[stream] signal {signum} received — finishing epoch, saving checkpoint, exiting.", flush=True)

    signal.signal(signal.SIGTERM, _signal)
    signal.signal(signal.SIGINT, _signal)

    manifest = Path(args.manifest)
    manifest_mtime: float | None = None
    train_ds = None
    val_loader: DataLoader | None = None
    n_train = n_val = 0
    t_start = time.time()

    print(f"[stream] watching {manifest} — training starts as soon as >= {args.min_samples} samples exist", flush=True)

    last_refresh_time = 0.0
    last_refresh_tiles = -10**9
    while not stop["flag"]:
        # ---- pick up new data: the manifest is rewritten atomically by the
        # downloader after every completed frame, so an mtime change means
        # "new frames are available". But rebuilding the dataset (and its
        # worker pools) every epoch churns RAM inside a RAM-capped container,
        # so we only rebuild when a meaningful amount of new data has arrived
        # (>= 200 tiles ~ one download batch) or 30 minutes have passed.
        mtime = manifest.stat().st_mtime if manifest.exists() else None
        refresh_due = train_ds is None or (
            mtime != manifest_mtime
            and (
                time.time() - last_refresh_time > 1800
                or _manifest_tile_count(manifest) - last_refresh_tiles >= 200
            )
        )
        if refresh_due:
            try:
                train_ds = SolarActiveRegionDataset(
                    manifest_path=manifest, channels=args.channels, split="train",
                    image_size=args.image_size, augment=True, seed=args.seed,
                )
                val_ds = SolarActiveRegionDataset(
                    manifest_path=manifest, channels=args.channels, split="val",
                    image_size=args.image_size, augment=False, seed=args.seed,
                )
            except (FileNotFoundError, ValueError, OSError) as exc:
                if not stop["flag"]:
                    print(f"[stream] waiting for data: manifest not ready yet ({type(exc).__name__}) — polling every 15 s", flush=True)
                    time.sleep(15)
                continue
            n_train, n_val = len(train_ds), len(val_ds)
            if n_train < args.min_samples:
                if not stop["flag"]:
                    print(f"[stream] waiting for data: {n_train} train samples (< {args.min_samples}) — polling every 15 s", flush=True)
                    time.sleep(15)
                continue
            loader_kwargs = {
                "num_workers": min(plan.dataloader_workers, 4),
                "pin_memory": plan.pin_memory,
                # persistent_workers=False is essential here: the loaders are
                # REBUILT when the dataset grows. Persistent workers on a
                # rebuilt loader accumulate zombie worker processes over a
                # long run (~0.7-1 GB RAM each) and eventually OOM the session
                # (observed: Kaggle 30 GB notebook restarted after ~2 h).
                # Workers spawn/tear down per use now: ~2 s overhead, no leak.
                "persistent_workers": False,
            }
            if plan.dataloader_workers > 0:
                loader_kwargs["prefetch_factor"] = 2
            # Quick validation: a fixed random slice of the val set (drawn once per
            # dataset generation) instead of the whole set — a full validation on a
            # grown val set with TTA can take an hour and starve training.
            val_eval_ds = val_ds
            val_eval_n = n_val
            if args.val_subset > 0 and n_val > args.val_subset:
                val_idx = random.Random(args.seed).sample(range(n_val), args.val_subset)
                val_eval_ds = Subset(val_ds, val_idx)
                val_eval_n = args.val_subset
            val_loader = DataLoader(val_eval_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
            # (the train loader is built per epoch, below, so it can sample a
            #  random subset of the dataset when --max-tiles-per-epoch is set)
            manifest_mtime = mtime
            last_refresh_time = time.time()
            last_refresh_tiles = n_train + n_val
            sub_note = f" (validating on {val_eval_n}/{n_val} quick subset)" if val_eval_n < n_val else ""
            print(f"[stream] dataset refreshed: {n_train} train / {n_val} val tiles{sub_note} — new frames now included automatically", flush=True)

        # ---- one epoch of continuous training
        epoch += 1
        if resume_lr is not None and (epoch - start_epoch) <= args.rewarm_epochs:
            # Re-warm from the checkpoint's own LR (safe for converged models).
            t = (epoch - start_epoch) / max(1, args.rewarm_epochs)
            lr = resume_lr + (args.lr - resume_lr) * t
        elif epoch <= args.warmup_epochs:
            lr = args.lr * (epoch / max(1, args.warmup_epochs))
        else:
            lr = args.lr
        for group in optimizer.param_groups:
            group["lr"] = lr

        # ---- pick this epoch's training data
        # With --max-tiles-per-epoch, each epoch trains on a random subset of
        # the dataset (like a very large mini-batch): epochs stay fast and
        # CONSTANT as the dataset grows, and every tile is still seen
        # repeatedly across epochs in random order. Without it, the whole
        # dataset is used. Validation below is always the FULL val set.
        k = args.max_tiles_per_epoch
        if k > 0 and n_train > k:
            indices = random.Random(args.seed + epoch * 7919).sample(range(n_train), k)
            epoch_ds = Subset(train_ds, indices)
            samples_this_epoch = k
        else:
            epoch_ds = train_ds
            samples_this_epoch = n_train
        epoch_loader = DataLoader(epoch_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)

        model.train(True)
        t0 = time.time()
        loss_sum, n_batches = 0.0, 0
        for images, masks in epoch_loader:
            if stop["flag"]:
                break
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=autocast_type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            if scaler_enabled:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(raw_model)
            loss_sum += float(loss.item())
            n_batches += 1

        # Drop the epoch's loader (and its worker processes) before the next
        # one is built — belt-and-braces on top of persistent_workers=False.
        del epoch_loader

        avg_loss = loss_sum / max(n_batches, 1)
        seconds = time.time() - t0
        line = (f"epoch={epoch:04d} loss={avg_loss:.4f} "
                f"samples={samples_this_epoch}/{n_train} lr={lr:.2e} {seconds:.0f}s "
                f"(run time {int((time.time() - t_start) // 60)} min)")

        if epoch % args.val_every == 0 and val_loader is not None:
            # Validate with the EMA weights — those are what best.pt tracks.
            # Quick subset validation runs without TTA (4 views would cost 4x);
            # full validation keeps the configured TTA.
            val_transforms = args.tta if val_eval_n >= n_val else "none"
            backup = ema.copy_to(raw_model)
            model.eval()
            # MICRO-AVERAGE over the WHOLE val set instead of averaging per-tile
            # Dice. A tile with an empty mask (quiet Sun, or a rolling-window
            # all-zero substitute for a just-freed tile) makes per-tile Dice
            # eps/eps = 1.0; a handful of those in a tiny early validation
            # (only ~12 tiles existed at the first validation) averaged to a
            # fake "perfect" 1.0000 that then froze best.pt forever, since
            # every later real score is lower and can never beat 1.0. Pooling
            # sums intersections and areas across all tiles first, so empty
            # tiles contribute nothing and the score reflects real overlap.
            inter = pred_area = target_area = union_area = 0.0
            vb = 0
            val_interrupted = False
            with torch.no_grad():
                for images, masks in val_loader:
                    if stop["flag"]:
                        val_interrupted = True
                        break
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    probs = tta_predict(
                        model, images, transforms=val_transforms,
                        autocast_kwargs={"device_type": autocast_type, "enabled": amp_enabled},
                    )
                    preds = (probs > 0.5).float()
                    inter += float((preds * masks).sum())
                    pred_area += float(preds.sum())
                    target_area += float(masks.sum())
                    union_area += float((preds + masks).clamp(0, 1).sum())
                    vb += 1
            raw_model.load_state_dict(backup)
            if target_area <= 0:
                # No active-region pixels anywhere in this validation (all
                # tiles empty/substituted): nothing real to score, so never
                # let an eps/eps = 1.0 become the best bar.
                val_dice = 0.0
                val_iou = 0.0
                line += " val_dice=n/a (no foreground in val set — not updating best)"
            else:
                val_dice = (2 * inter + 1e-6) / (pred_area + target_area + 1e-6)
                val_iou = (inter + 1e-6) / (union_area + 1e-6)
                quick_note = f" ({val_eval_n} quick)" if val_eval_n < n_val else ""
                line += f" val_dice={val_dice:.4f} val_iou={val_iou:.4f}{quick_note}"
            if not val_interrupted and target_area > 0:
                if calibrate_pending:
                    # First real validation on current data: discard any stale
                    # (incomparable / inflated) bar, adopt THIS score AND save
                    # best.pt with the current good weights immediately so the
                    # on-disk best.pt never lingers on a fake/bad checkpoint.
                    calibrate_pending = False
                    best_dice = val_dice
                    save_checkpoint(best_path, raw_model, optimizer, ema, epoch, best_dice, args.channels, current_lr)
                    line += f"  (best bar calibrated to current data: {val_dice:.4f}; best.pt updated)"
                elif val_dice > best_dice:
                    best_dice = val_dice
                    save_checkpoint(best_path, raw_model, optimizer, ema, epoch, best_dice, args.channels, current_lr)
                    line += "  ★ new best"

        current_lr = lr
        save_checkpoint(last_path, raw_model, optimizer, ema, epoch, best_dice, args.channels, current_lr)
        # ---- Container memory self-recycle: if container memory usage creeps
        # toward the cgroup limit (observed on Kaggle 30 GiB: ~2 GB/h creep from
        # loader/IPC churn), exit CLEANLY with code 42 right after the
        # checkpoint save. run_forever.sh auto-restarts us and we resume from
        # last.pt: a clean recycle costs ~20 s and loses nothing, while the
        # alternative is the OOM killer restarting the whole container.
        # Two guards against false positives (a nested-cgroup read can report
        # a limit smaller than the real container, which would trigger a
        # recycle->restart loop on a FRESH process):
        #   - only after 30+ minutes of healthy uptime in this process,
        #   - against the most lenient plausible cap (cgroup limit vs
        #     /proc/meminfo), at a conservative 85%.
        if time.time() - t_start > 1800:
            mem_used, mem_limit = _cgroup_memory()
            if mem_limit is not None:
                try:
                    meminfo_b = int(Path("/proc/meminfo").readline().split()[1]) * 1024
                    if meminfo_b > mem_limit:
                        mem_limit = meminfo_b
                except (OSError, ValueError):
                    pass
            if epoch % 10 == 0 and mem_used is not None:
                # Diagnostic: what the guard is actually seeing in THIS
                # container (a misread cap shows up here immediately).
                print(
                    f"[mem] {mem_used / 2**30:.1f}"
                    + (f"/{mem_limit / 2**30:.1f}" if mem_limit is not None else "/?")
                    + " GiB",
                    flush=True,
                )
            if mem_used is not None and mem_limit is not None and mem_used > 0.85 * mem_limit:
                print(
                    f"[stream] container memory high ({mem_used / 2**30:.1f} / {mem_limit / 2**30:.1f} GiB) — "
                    "checkpoint saved, self-recycling (exit 42); run_forever restarts me from last.pt",
                    flush=True,
                )
                sys.exit(42)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "loss": avg_loss, "samples": n_train}) + "\n")
        print(f"[stream] {line}", flush=True)
        if args.status_file:
            write_status(args.status_file, epoch, n_train, n_val, line, best_dice)

        if args.max_epochs and epoch >= args.max_epochs:
            print("[stream] reached --max-epochs — stopping.", flush=True)
            break

    save_checkpoint(last_path, raw_model, optimizer, ema, epoch, best_dice, args.channels, current_lr)
    print(f"[stream] stopped at epoch {epoch}; checkpoints saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
