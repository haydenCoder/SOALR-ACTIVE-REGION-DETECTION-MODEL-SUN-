#!/usr/bin/env python3
"""Evaluate a saved solar active-region segmentation checkpoint on a manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from solar_ar.data import SolarActiveRegionDataset
from solar_ar.models import AttentionUNet
from solar_ar.training import compute_metrics_from_probs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an Attention U-Net checkpoint on a CSV manifest.")
    parser.add_argument("--manifest", required=True, help="CSV manifest to evaluate")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file, normally best.pt")
    parser.add_argument("--channels", nargs="+", default=["171"], help="Channel names represented by the manifest")
    parser.add_argument("--split", default="val", help="Manifest split to evaluate (default: val)")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--normalize-mode", choices=["percentile", "minmax", "zscore", "solar_physics"], default="percentile")
    parser.add_argument("--output", required=True, help="JSON file for the resulting metrics")
    return parser


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SolarActiveRegionDataset(
        manifest_path=args.manifest,
        channels=args.channels,
        split=args.split,
        image_size=args.image_size,
        normalize_mode=args.normalize_mode,
        augment=False,
        cache_budget_bytes=0,
    )
    if not dataset:
        raise RuntimeError(f"No rows with split={args.split!r} were found in {args.manifest}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
    model = AttentionUNet(
        in_channels=len(args.channels),
        out_channels=1,
        base_channels=args.base_channels,
        dropout=args.dropout,
    ).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    total_dice = total_iou = total_loss = 0.0
    batches = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(images)
        # BCE requires float32 here; compute it outside CUDA autocast.
        logits = logits.float()
        masks = masks.float()
        probs = torch.sigmoid(logits)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, masks)
        dice, iou = compute_metrics_from_probs(probs, masks)
        total_dice += dice
        total_iou += iou
        total_loss += float(loss.item())
        batches += 1

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "split": args.split,
        "samples": len(dataset),
        "dice": total_dice / batches,
        "iou": total_iou / batches,
        "bce_loss": total_loss / batches,
        "device": str(device),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved independent-test metrics to {output}")


if __name__ == "__main__":
    main()
