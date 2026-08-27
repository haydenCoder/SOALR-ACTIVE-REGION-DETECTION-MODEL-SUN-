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
import signal
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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
    p.add_argument("--batch-size", type=int, default=0, help="0 = auto from the memory budget")
    p.add_argument("--val-every", type=int, default=2, help="Validate (and update best.pt) every N epochs")
    p.add_argument("--max-epochs", type=int, default=0, help="0 = train forever until stopped")
    p.add_argument("--min-samples", type=int, default=4, help="Wait until the manifest has at least this many train samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loss", choices=["bce_dice", "focal_tversky", "combo"], default="bce_dice")
    p.add_argument("--tta", choices=["none", "flips", "d4"], default="flips")
    p.add_argument("--cpu-budget", type=int, default=0, help="0 = auto (every core minus headroom)")
    p.add_argument("--cpu-headroom", type=int, default=2)
    p.add_argument("--memory-budget-gb", type=float, default=0.0, help="0 = all detected RAM")
    p.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    p.add_argument("--no-torch-compile", dest="torch_compile", action="store_false", default=True)
    return p


def build_criterion(name: str):
    if name == "focal_tversky":
        return FocalTverskyLoss()
    if name == "combo":
        return ComboLoss()
    return BCEDiceLoss()


def load_last(path: Path, model, optimizer, ema, device) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if ema is not None and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"], checkpoint.get("ema_updates", 0))
    return int(checkpoint["epoch"]), float(checkpoint.get("best_dice", -1e9))


def save_checkpoint(path: Path, model, optimizer, ema, epoch: int, best_dice: float, channels: list[str]) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_dice": best_dice,
        "channels": channels,
    }
    if ema is not None:
        payload["ema_state_dict"] = ema.state_dict()
        payload["ema_updates"] = ema.updates
    torch.save(payload, path)


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

    batch_size = args.batch_size or suggest_batch_size(
        args.image_size, len(args.channels), args.base_channels, plan.memory_budget_gb * 0.55
    )
    print(f"[stream] batch_size={batch_size} (auto)", flush=True)

    model = AttentionUNet(
        in_channels=len(args.channels),
        out_channels=1,
        base_channels=args.base_channels,
        dropout=args.dropout,
    ).to(device)

    # C-level graph engine with the same smoke-tested fallback as train.py.
    compile_enabled = False
    if args.torch_compile:
        try:
            with torch.no_grad():
                dummy = torch.zeros(2, len(args.channels), min(args.image_size, 64), min(args.image_size, 64), device=device)
                model = torch.compile(model, dynamic=True)
                _ = model(dummy)
            compile_enabled = True
            print("[compile] torch.compile active (C-level graph engine)", flush=True)
        except Exception as exc:  # noqa: BLE001
            model = AttentionUNet(
                in_channels=len(args.channels), out_channels=1,
                base_channels=args.base_channels, dropout=args.dropout,
            ).to(device)
            print(f"[compile] torch.compile unavailable ({type(exc).__name__}: {exc}) — continuing in eager mode", flush=True)

    ema = ModelEma(model, decay=0.999)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    criterion = build_criterion(args.loss)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = out_dir / "best.pt", out_dir / "last.pt"
    metrics_path = out_dir / "metrics.jsonl"

    epoch = 0
    best_dice = -1e9
    if args.resume and last_path.exists():
        epoch, best_dice = load_last(last_path, model, optimizer, ema, device)
        print(f"[stream] resumed from {last_path} at epoch {epoch} (best_dice={best_dice:.4f})", flush=True)

    stop = {"flag": False}

    def _signal(signum, frame) -> None:  # noqa: ANN001
        stop["flag"] = True
        print(f"[stream] signal {signum} received — finishing epoch, saving checkpoint, exiting.", flush=True)

    signal.signal(signal.SIGTERM, _signal)
    signal.signal(signal.SIGINT, _signal)

    manifest = Path(args.manifest)
    manifest_mtime: float | None = None
    train_loader: DataLoader | None = None
    val_loader: DataLoader | None = None
    n_train = n_val = 0
    t_start = time.time()

    print(f"[stream] watching {manifest} — training starts as soon as >= {args.min_samples} samples exist", flush=True)

    while not stop["flag"]:
        # ---- pick up new data: the manifest is rewritten atomically by the
        # downloader after every completed frame, so an mtime change means
        # "new frames are available" — rebuild the dataset, keep training.
        mtime = manifest.stat().st_mtime if manifest.exists() else None
        if train_loader is None or mtime != manifest_mtime:
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
                "num_workers": plan.dataloader_workers,
                "pin_memory": plan.pin_memory,
                "persistent_workers": plan.dataloader_workers > 0,
            }
            if plan.dataloader_workers > 0:
                loader_kwargs["prefetch_factor"] = 8
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
            manifest_mtime = mtime
            print(f"[stream] dataset refreshed: {n_train} train / {n_val} val tiles — new frames now included automatically", flush=True)

        # ---- one epoch of continuous training
        epoch += 1
        lr = args.lr * (epoch / max(1, args.warmup_epochs)) if epoch <= args.warmup_epochs else args.lr
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train(True)
        t0 = time.time()
        loss_sum, n_batches = 0.0, 0
        for images, masks in train_loader:
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            loss_sum += float(loss.item())
            n_batches += 1

        avg_loss = loss_sum / max(n_batches, 1)
        seconds = time.time() - t0
        line = (f"epoch={epoch:04d} loss={avg_loss:.4f} samples={n_train} lr={lr:.2e} {seconds:.0f}s "
                f"(run time {int((time.time() - t_start) // 60)} min)")

        if epoch % args.val_every == 0 and val_loader is not None:
            # Validate with the EMA weights — those are what best.pt tracks.
            backup = ema.copy_to(model)
            model.eval()
            dice_sum = iou_sum = 0.0
            vb = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    if stop["flag"]:
                        break
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    probs = tta_predict(
                        model, images, transforms=args.tta,
                        autocast_kwargs={"device_type": autocast_type, "enabled": amp_enabled},
                    )
                    d, i = compute_metrics_from_probs(probs, masks)
                    dice_sum += d
                    iou_sum += i
                    vb += 1
            model.load_state_dict(backup)
            val_dice = dice_sum / max(vb, 1)
            val_iou = iou_sum / max(vb, 1)
            line += f" val_dice={val_dice:.4f} val_iou={val_iou:.4f}"
            if val_dice > best_dice:
                best_dice = val_dice
                save_checkpoint(best_path, model, optimizer, ema, epoch, best_dice, args.channels)
                line += "  ★ new best"

        save_checkpoint(last_path, model, optimizer, ema, epoch, best_dice, args.channels)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "loss": avg_loss, "samples": n_train}) + "\n")
        print(f"[stream] {line}", flush=True)
        if args.status_file:
            write_status(args.status_file, epoch, n_train, n_val, line, best_dice)

        if args.max_epochs and epoch >= args.max_epochs:
            print("[stream] reached --max-epochs — stopping.", flush=True)
            break

    save_checkpoint(last_path, model, optimizer, ema, epoch, best_dice, args.channels)
    print(f"[stream] stopped at epoch {epoch}; checkpoints saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
