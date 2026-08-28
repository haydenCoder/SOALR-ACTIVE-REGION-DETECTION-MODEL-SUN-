#!/usr/bin/env python3
"""Object-detection metrics from a segmentation checkpoint.

Predicted and ground-truth masks are converted to connected components, then
matched one-to-one using bounding-box IoU.  This reports the conventional
precision/recall/F1 and AP@IoU threshold used by many AR detection papers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from solar_ar.data import SolarActiveRegionDataset
from solar_ar.models import AttentionUNet
from solar_ar.runtime import amp_settings, preferred_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate object-level detection metrics from segmentation masks.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--channels", nargs="+", default=["aia171"])
    parser.add_argument("--split", default="val")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--normalize-mode", choices=["percentile", "minmax", "zscore", "solar_physics"], default="percentile")
    parser.add_argument("--threshold", type=float, default=0.5, help="Pixel probability cutoff for predicted objects")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="Bounding-box IoU needed to match an object")
    parser.add_argument("--min-area", type=int, default=1, help="Ignore connected components smaller than this many pixels")
    parser.add_argument("--output", required=True)
    return parser


def components(mask: np.ndarray, scores: np.ndarray | None, min_area: int) -> list[tuple[tuple[int, int, int, int], float]]:
    """Return ``((x1, y1, x2, y2), score)`` for 8-connected components."""
    labels, count = ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))
    slices = ndimage.find_objects(labels)
    found: list[tuple[tuple[int, int, int, int], float]] = []
    for label_id, region in enumerate(slices, start=1):
        if region is None:
            continue
        ys, xs = region
        region_labels = labels[ys, xs] == label_id
        area = int(region_labels.sum())
        if area < min_area:
            continue
        score = 1.0 if scores is None else float(scores[ys, xs][region_labels].mean())
        found.append(((xs.start, ys.start, xs.stop, ys.stop), score))
    return found


def box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    width = max(0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union else 0.0


def average_precision(predictions: list[tuple[float, bool]], targets: int) -> float:
    if not targets:
        return 0.0
    ordered = sorted(predictions, key=lambda item: item[0], reverse=True)
    true_positive = np.asarray([int(match) for _, match in ordered], dtype=np.float64)
    false_positive = 1.0 - true_positive
    recall = np.cumsum(true_positive) / targets
    precision = np.cumsum(true_positive) / np.maximum(np.cumsum(true_positive + false_positive), 1)
    # Standard interpolated area under the precision-recall curve.
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision) - 1, 0, -1):
        precision[index - 1] = max(precision[index - 1], precision[index])
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    # Tolerate a single quoted multi-word --channels argument (e.g.
    # --channels "aia94 aia131 ..."), which argparse would otherwise treat
    # as one channel name.
    args.channels = [c for part in args.channels for c in part.split()]
    device = torch.device(preferred_device())
    autocast_type, amp_enabled, _ = amp_settings(device.type)
    dataset = SolarActiveRegionDataset(
        manifest_path=args.manifest,
        channels=args.channels,
        split=args.split,
        image_size=args.image_size,
        normalize_mode=args.normalize_mode,
        augment=False,
        cache_budget_bytes=0,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No {args.split!r} rows in {args.manifest}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
    model = AttentionUNet(in_channels=len(args.channels), out_channels=1, base_channels=args.base_channels, dropout=args.dropout).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state_dict"])

    tp = fp = fn = target_count = 0
    scored_predictions: list[tuple[float, bool]] = []
    for images, targets in loader:
        with torch.amp.autocast(device_type=autocast_type, enabled=amp_enabled):
            probabilities = torch.sigmoid(model(images.to(device, non_blocking=True))).float().cpu().numpy()[:, 0]
        target_arrays = targets.numpy()[:, 0]
        for probability, target in zip(probabilities, target_arrays):
            predicted = sorted(components(probability >= args.threshold, probability, args.min_area), key=lambda item: item[1], reverse=True)
            truth = components(target >= 0.5, None, args.min_area)
            target_count += len(truth)
            unused_truth = set(range(len(truth)))
            for box, score in predicted:
                best_index = max(unused_truth, key=lambda index: box_iou(box, truth[index][0]), default=None)
                matched = best_index is not None and box_iou(box, truth[best_index][0]) >= args.iou_threshold
                scored_predictions.append((score, matched))
                if matched:
                    tp += 1
                    unused_truth.remove(best_index)
                else:
                    fp += 1
            fn += len(unused_truth)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = {
        "metric": f"object detection at bounding-box IoU >= {args.iou_threshold}",
        "samples": len(dataset),
        "ground_truth_objects": target_count,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        f"ap_iou_{args.iou_threshold:.2f}": average_precision(scored_predictions, target_count),
        "pixel_threshold": args.threshold,
        "min_component_area_pixels": args.min_area,
        "checkpoint": str(Path(args.checkpoint).resolve()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved object-detection metrics to {output}")


if __name__ == "__main__":
    main()
