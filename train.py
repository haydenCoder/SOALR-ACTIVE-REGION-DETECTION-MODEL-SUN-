from __future__ import annotations

import argparse
import json
import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from solar_ar.runtime import (
    DEFAULT_CPU_BUDGET,
    DEFAULT_CPU_HEADROOM,
    DEFAULT_MEMORY_BUDGET_GB,
    DEFAULT_MEMORY_HEADROOM_GB,
    plan_resources,
    suggest_batch_size,
)
from solar_ar.training import Trainer
from solar_ar.tta import TTA_PRESETS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an Attention U-Net for solar active region segmentation.")
    parser.add_argument("--manifest", default=None, help="CSV manifest created by scripts/prepare_uad_manifest.py. If omitted, will auto-download and prepare UAD dataset.")
    parser.add_argument("--channels", nargs="+", default=["171", "195", "284", "304"], help="Channel names to use")
    parser.add_argument("--image-size", type=int, default=256, help="Training crop/resize size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="Dataloader worker processes; -1 (default) derives it from --cpu-budget",
    )
    parser.add_argument("--output-dir", default="runs/attention_unet_uad", help="Directory to store checkpoints and logs")
    parser.add_argument("--base-channels", type=int, default=32, help="Base feature width of the Attention U-Net")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate inside convolutional blocks")
    parser.add_argument(
        "--normalize-mode",
        choices=["percentile", "minmax", "zscore", "solar_physics"],
        default="percentile",
        help="Per-channel normalization strategy",
    )
    parser.add_argument("--amp", action="store_true", default=True, help="Enable mixed precision when CUDA is available")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision")
    parser.add_argument("--patience", type=int, default=0, help="Early-stopping patience in epochs; 0 (default) disables it")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--loss",
        choices=["bce_dice", "focal_tversky", "combo"],
        default="bce_dice",
        help="Segmentation loss function ('combo' blends BCE-Dice with Focal-Tversky)",
    )
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=1,
        help="Number of optimizer accumulation steps. Useful for low-RAM training.",
    )
    parser.add_argument(
        "--hdf5-image-key",
        default=None,
        help=(
            "Dataset/variable name to read from HDF5, netCDF or npz image files "
            "(e.g. 'image'). A 'path#key' suffix in the manifest overrides this."
        ),
    )
    parser.add_argument(
        "--hdf5-mask-key",
        default=None,
        help="Dataset/variable name for mask files (e.g. 'union_with_intersect' for ARPIL masks).",
    )
    parser.add_argument("--torch-num-threads", type=int, default=0, help="Set PyTorch intra-op CPU threads (0 = keep default)")
    parser.add_argument(
        "--torch-num-interop-threads",
        type=int,
        default=0,
        help="Set PyTorch inter-op CPU threads (0 = keep default)",
    )

    hardware = parser.add_argument_group("hardware utilisation")
    hardware.add_argument(
        "--cpu-budget",
        type=int,
        default=DEFAULT_CPU_BUDGET,
        help=(
            "Target CPU cores to saturate. Automatically clamped to the cores "
            f"actually available. Default: {DEFAULT_CPU_BUDGET}."
        ),
    )
    hardware.add_argument(
        "--memory-budget-gb",
        type=float,
        default=DEFAULT_MEMORY_BUDGET_GB,
        help=(
            "Target RAM in GB. Drives the in-RAM sample cache and batch-size "
            f"suggestion; clamped to detected RAM. Default: {DEFAULT_MEMORY_BUDGET_GB}."
        ),
    )
    hardware.add_argument(
        "--cache-fraction",
        type=float,
        default=0.45,
        help="Fraction of the memory budget used to cache decoded samples in RAM.",
    )
    hardware.add_argument(
        "--cpu-headroom",
        type=int,
        default=DEFAULT_CPU_HEADROOM,
        help=(
            "Cores to leave free for the OS when --cpu-budget is auto (<=0). "
            f"Default: {DEFAULT_CPU_HEADROOM}; 0 uses every detected core."
        ),
    )
    hardware.add_argument(
        "--memory-headroom-gb",
        type=float,
        default=DEFAULT_MEMORY_HEADROOM_GB,
        help=(
            "RAM in GB to leave free when --memory-budget-gb is auto (<=0). "
            f"Default: {DEFAULT_MEMORY_HEADROOM_GB}; 0 uses all detected RAM."
        ),
    )
    hardware.add_argument(
        "--auto-batch-size",
        action="store_true",
        default=True,
        help="Pick the largest batch size that fits the memory budget, overriding --batch-size.",
    )
    hardware.add_argument(
        "--no-auto-batch-size",
        dest="auto_batch_size",
        action="store_false",
        help="Disable automatic batch size selection.",
    )
    hardware.add_argument(
        "--no-channels-last",
        dest="channels_last",
        action="store_false",
        help="Disable channels-last memory format (enabled by default; faster convolutions).",
    )

    model_group = parser.add_argument_group("model architecture")
    model_group.add_argument("--model-depth", type=int, default=4, help="Number of encoder/decoder levels")
    model_group.add_argument(
        "--no-residual",
        dest="residual",
        action="store_false",
        help="Disable residual connections inside conv blocks",
    )
    model_group.add_argument(
        "--no-se",
        dest="use_se",
        action="store_false",
        help="Disable squeeze-and-excitation channel attention",
    )
    model_group.add_argument(
        "--norm-groups",
        type=int,
        default=0,
        help="Use GroupNorm with this many groups instead of BatchNorm (recommended for batch sizes < 8)",
    )
    model_group.add_argument(
        "--deep-supervision",
        action="store_true",
        help="Add auxiliary losses on intermediate decoder stages",
    )

    optim = parser.add_argument_group("optimisation")
    optim.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="Exponential moving average decay for model weights, e.g. 0.999. 0 disables EMA.",
    )
    optim.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm; 0 disables clipping")
    optim.add_argument("--warmup-epochs", type=int, default=1, help="Linear LR warmup length in epochs")
    optim.add_argument(
        "--tta",
        choices=sorted(TTA_PRESETS),
        default="none",
        help=(
            "Test-time augmentation used during validation. 'flips' averages 4 "
            "views, 'd4' averages all 8 symmetries for the best accuracy."
        ),
    )
    optim.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint in output-dir")
    return parser


import subprocess


def bootstrap_data() -> str:
    manifest_path = Path("data/processed/uad_manifest.csv")
    if manifest_path.exists():
        return str(manifest_path)

    import shutil
    # Detect free disk space in GB
    _, _, free_bytes = shutil.disk_usage(".")
    free_gb = free_bytes / (1024**3)

    print(f"Detected {free_gb:.1f} GB free disk space.")

    print("Manifest not found. Auto-downloading and preparing data...")
    # 1. Download primary UAD dataset (~15GB)
    if free_gb > 15:
        print("Downloading UAD dataset...")
        subprocess.run(["bash", "scripts/download_uad_dataset.sh"], check=True)
    else:
        print("Not enough disk space for UAD dataset (>15GB required).")

    # The optional sample scripts contain unlabelled demonstration files. They
    # cannot be added to this segmentation manifest safely, so do not download
    # them automatically or spend the user's Colab disk quota on unused data.

    # 2. Prepare manifest
    print("Preparing manifest...")
    subprocess.run([
        sys.executable, "scripts/prepare_uad_manifest.py",
        "--raw-root", "data/raw/Solar_data_UAD",
        "--output", str(manifest_path)
    ], check=True)

    print(f"Data bootstrap complete. Manifest created at {manifest_path}")
    return str(manifest_path)


def main() -> None:
    args = build_parser().parse_args()

    if args.manifest is None:
        args.manifest = bootstrap_data()

    # --torch-num-threads predates --cpu-budget. When given explicitly it is a
    # deliberate override, so it becomes the CPU budget that drives the whole
    # runtime plan (threads, loader workers, interop pool) instead of being
    # silently overwritten by it.
    if args.torch_num_threads > 0:
        args.cpu_budget = args.torch_num_threads
    if args.torch_num_interop_threads > 0:
        torch.set_num_interop_threads(args.torch_num_interop_threads)

    batch_size = args.batch_size
    if args.auto_batch_size:
        plan = plan_resources(
            cpu_budget=args.cpu_budget,
            memory_budget_gb=args.memory_budget_gb,
            cache_fraction=args.cache_fraction,
            use_cuda=torch.cuda.is_available(),
            cpu_headroom=args.cpu_headroom,
            memory_headroom_gb=args.memory_headroom_gb,
        )
        batch_size = suggest_batch_size(
            image_size=args.image_size,
            channels=len(args.channels),
            base_channels=args.base_channels,
            memory_budget_gb=plan.memory_budget_gb * (1.0 - args.cache_fraction),
        )
        print(f"[auto-batch-size] using batch_size={batch_size} (was {args.batch_size})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "train_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    trainer = Trainer(
        manifest_path=args.manifest,
        channels=args.channels,
        image_size=args.image_size,
        batch_size=batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        output_dir=output_dir,
        base_channels=args.base_channels,
        dropout=args.dropout,
        normalize_mode=args.normalize_mode,
        amp=args.amp,
        patience=args.patience,
        seed=args.seed,
        loss_name=args.loss,
        grad_accumulation_steps=args.grad_accumulation_steps,
        hdf5_image_key=args.hdf5_image_key,
        hdf5_mask_key=args.hdf5_mask_key,
        cpu_budget=args.cpu_budget,
        memory_budget_gb=args.memory_budget_gb,
        cache_fraction=args.cache_fraction,
        cpu_headroom=args.cpu_headroom,
        memory_headroom_gb=args.memory_headroom_gb,
        deep_supervision=args.deep_supervision,
        model_depth=args.model_depth,
        residual=args.residual,
        use_se=args.use_se,
        norm_groups=args.norm_groups,
        ema_decay=args.ema_decay,
        grad_clip=args.grad_clip,
        warmup_epochs=args.warmup_epochs,
        tta=args.tta,
        channels_last=args.channels_last,
        resume=args.resume,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
