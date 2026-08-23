#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic SMARP-like training patches to create a real/synthetic training mix.")
    parser.add_argument("--manifest", default="data/processed/smarp_sample/manifest.csv", help="Real patch manifest")
    parser.add_argument("--output-dir", default="data/processed/smarp_mix_70_30", help="Output directory for mixed manifest and synthetic patches")
    parser.add_argument("--synthetic-fraction", type=float, default=0.30, help="Target synthetic fraction in final training set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def rotate_flip(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    k = rng.randint(0, 3)
    image = np.rot90(image, k)
    mask = np.rot90(mask, k)
    if rng.random() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    if rng.random() < 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def shift_with_zero_fill(arr: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    out = np.zeros_like(arr)
    src_y0 = max(0, -shift_y)
    src_y1 = min(arr.shape[0], arr.shape[0] - shift_y) if shift_y >= 0 else arr.shape[0]
    dst_y0 = max(0, shift_y)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    src_x0 = max(0, -shift_x)
    src_x1 = min(arr.shape[1], arr.shape[1] - shift_x) if shift_x >= 0 else arr.shape[1]
    dst_x0 = max(0, shift_x)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return out


def dilate_or_erode(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    tensor = torch.from_numpy(mask[None, None, ...].astype(np.float32))
    op = rng.choice(["none", "dilate", "erode"])
    if op == "none":
        return mask.astype(np.float32)
    if op == "dilate":
        result = F.max_pool2d(tensor, kernel_size=3, stride=1, padding=1)
    else:
        result = 1.0 - F.max_pool2d(1.0 - tensor, kernel_size=3, stride=1, padding=1)
    return result[0, 0].numpy().astype(np.float32)


def synthesize_pair(img_a: np.ndarray, mask_a: np.ndarray, img_b: np.ndarray, mask_b: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    img_a, mask_a = rotate_flip(img_a, mask_a, rng)
    img_b, mask_b = rotate_flip(img_b, mask_b, rng)

    max_shift = max(1, img_a.shape[0] // 10)
    sy = rng.randint(-max_shift, max_shift)
    sx = rng.randint(-max_shift, max_shift)
    img_b = shift_with_zero_fill(img_b, sy, sx)
    mask_b = shift_with_zero_fill(mask_b, sy, sx)

    blend = rng.uniform(0.55, 0.85)
    img = blend * img_a + (1.0 - blend) * img_b
    img += rng.normalvariate(0.0, 0.03)
    img = img + rng.uniform(-0.08, 0.08)
    img = (img - img.mean()) * rng.uniform(0.9, 1.2) + 0.5
    img = np.clip(img, 0.0, 1.0)

    mask = np.maximum(mask_a, mask_b).astype(np.float32)
    mask = dilate_or_erode(mask, rng)
    mask = (mask > 0.5).astype(np.float32)

    # Give the synthetic magnetogram stronger contrast around active regions.
    emphasis = rng.uniform(0.15, 0.35)
    img = np.clip(img * (1.0 - emphasis) + mask * emphasis, 0.0, 1.0)
    return img.astype(np.float32), mask.astype(np.float32)


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    syn_image_dir = output_dir / "synthetic_images"
    syn_mask_dir = output_dir / "synthetic_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    syn_image_dir.mkdir(parents=True, exist_ok=True)
    syn_mask_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if len(train_rows) < 2:
        raise SystemExit("Need at least two training rows to synthesize mixed examples.")

    real_train_count = len(train_rows)
    synthetic_count = math.ceil(real_train_count * args.synthetic_fraction / max(1e-6, 1.0 - args.synthetic_fraction))

    mixed_rows: list[dict[str, str]] = []
    mixed_rows.extend(train_rows)
    mixed_rows.extend(val_rows)

    for idx in range(synthetic_count):
        row_a, row_b = rng.sample(train_rows, 2)
        img_a = np.load((manifest_path.parent / row_a["image_mag"]).resolve()).astype(np.float32)
        mask_a = np.load((manifest_path.parent / row_a["mask"]).resolve()).astype(np.float32)
        img_b = np.load((manifest_path.parent / row_b["image_mag"]).resolve()).astype(np.float32)
        mask_b = np.load((manifest_path.parent / row_b["mask"]).resolve()).astype(np.float32)

        syn_img, syn_mask = synthesize_pair(img_a, mask_a, img_b, mask_b, rng)
        sample_id = f"synthetic_{idx:04d}"
        img_path = syn_image_dir / f"{sample_id}.npy"
        mask_path = syn_mask_dir / f"{sample_id}.npy"
        np.save(img_path, syn_img)
        np.save(mask_path, syn_mask)
        mixed_rows.append(
            {
                "sample_id": sample_id,
                "split": "train",
                "image_mag": os.path.relpath(img_path, start=output_dir).replace(os.sep, "/"),
                "mask": os.path.relpath(mask_path, start=output_dir).replace(os.sep, "/"),
            }
        )

    # Rewrite paths for real rows so they resolve from the new output directory.
    adjusted_rows: list[dict[str, str]] = []
    for row in mixed_rows:
        new_row = dict(row)
        if not new_row["sample_id"].startswith("synthetic_"):
            real_img = (manifest_path.parent / row["image_mag"]).resolve()
            real_mask = (manifest_path.parent / row["mask"]).resolve()
            new_row["image_mag"] = os.path.relpath(real_img, start=output_dir).replace(os.sep, "/")
            new_row["mask"] = os.path.relpath(real_mask, start=output_dir).replace(os.sep, "/")
        adjusted_rows.append(new_row)

    manifest_out = output_dir / "manifest.csv"
    with manifest_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "image_mag", "mask"])
        writer.writeheader()
        writer.writerows(adjusted_rows)

    final_train = sum(1 for row in adjusted_rows if row["split"] == "train")
    synthetic_train = sum(1 for row in adjusted_rows if row["split"] == "train" and row["sample_id"].startswith("synthetic_"))
    final_fraction = synthetic_train / max(final_train, 1)
    print(f"Real train patches: {real_train_count}")
    print(f"Synthetic train patches generated: {synthetic_train}")
    print(f"Final train synthetic fraction: {final_fraction:.3f}")
    print(f"Output manifest: {manifest_out}")


if __name__ == "__main__":
    main()
