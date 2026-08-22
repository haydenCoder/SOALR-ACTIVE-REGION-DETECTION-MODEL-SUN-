from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from solar_ar.training import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an Attention U-Net for solar active region segmentation.")
    parser.add_argument("--manifest", required=True, help="CSV manifest created by scripts/prepare_uad_manifest.py")
    parser.add_argument("--channels", nargs="+", default=["171", "195", "284", "304"], help="Channel names to use")
    parser.add_argument("--image-size", type=int, default=256, help="Training crop/resize size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker processes")
    parser.add_argument("--output-dir", default="runs/attention_unet_uad", help="Directory to store checkpoints and logs")
    parser.add_argument("--base-channels", type=int, default=32, help="Base feature width of the Attention U-Net")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate inside convolutional blocks")
    parser.add_argument(
        "--normalize-mode",
        choices=["percentile", "minmax", "zscore", "solar_physics"],
        default="percentile",
        help="Per-channel normalization strategy",
    )
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision when CUDA is available")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience in epochs; use 0 to disable")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--loss",
        choices=["bce_dice", "focal_tversky"],
        default="bce_dice",
        help="Segmentation loss function",
    )
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=1,
        help="Number of optimizer accumulation steps. Useful for 4 GB RAM training.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "train_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    trainer = Trainer(
        manifest_path=args.manifest,
        channels=args.channels,
        image_size=args.image_size,
        batch_size=args.batch_size,
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
    )
    trainer.fit()


if __name__ == "__main__":
    main()
