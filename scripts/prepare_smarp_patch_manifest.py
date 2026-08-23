#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

import numpy as np
from astropy.io import fits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create training patches from SMARP magnetogram/bitmap FITS files.")
    parser.add_argument("--raw-root", default="data/raw/smarp_sample", help="Directory containing *.magnetogram.fits and *.bitmap.fits")
    parser.add_argument("--output-dir", default="data/processed/smarp_sample", help="Directory for .npy patches and manifest")
    parser.add_argument("--patch-size", type=int, default=64, help="Patch size in pixels")
    parser.add_argument("--stride", type=int, default=16, help="Sliding-window stride")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--min-mask-fraction", type=float, default=0.01, help="Minimum positive mask coverage to keep a patch")
    parser.add_argument("--keep-empty-every", type=int, default=5, help="Keep every Nth empty patch as background examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def normalize_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"\.(bitmap|magnetogram)$", "", stem)
    return stem


def iter_patch_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def robust_normalize(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.percentile(array, 1))
    hi = float(np.percentile(array, 99))
    array = np.clip(array, lo, hi)
    return (array - lo) / max(hi - lo, 1e-6)


def main() -> None:
    args = build_parser().parse_args()
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    magnetograms = {normalize_stem(p.name): p for p in raw_root.glob("*.magnetogram.fits")}
    bitmaps = {normalize_stem(p.name): p for p in raw_root.glob("*.bitmap.fits")}
    common = sorted(set(magnetograms) & set(bitmaps))
    if not common:
        raise SystemExit(f"No matched SMARP FITS pairs found in {raw_root}")

    random_gen = random.Random(args.seed)
    rows: list[dict[str, str]] = []
    empty_counter = 0

    for key in common:
        mag = robust_normalize(fits.getdata(magnetograms[key]))
        bmp = np.asarray(fits.getdata(bitmaps[key]), dtype=np.int32)
        # In SMARP bitmaps, bit 32 (ON_PATCH) indicates pixels belonging to the active-region patch.
        mask = ((bmp & 32) > 0).astype(np.float32)

        height, width = mag.shape
        ys = iter_patch_starts(height, args.patch_size, args.stride)
        xs = iter_patch_starts(width, args.patch_size, args.stride)

        for y in ys:
            for x in xs:
                mag_patch = mag[y : y + args.patch_size, x : x + args.patch_size]
                mask_patch = mask[y : y + args.patch_size, x : x + args.patch_size]
                mask_fraction = float(mask_patch.mean())

                if mask_fraction < args.min_mask_fraction:
                    empty_counter += 1
                    if args.keep_empty_every <= 0 or empty_counter % args.keep_empty_every != 0:
                        continue

                patch_id = f"{key}_y{y:03d}_x{x:03d}"
                image_path = image_dir / f"{patch_id}.npy"
                mask_path = mask_dir / f"{patch_id}.npy"
                np.save(image_path, mag_patch.astype(np.float32))
                np.save(mask_path, mask_patch.astype(np.float32))
                rows.append(
                    {
                        "sample_id": patch_id,
                        "split": "train",
                        "image_mag": image_path.relative_to(output_dir).as_posix(),
                        "mask": mask_path.relative_to(output_dir).as_posix(),
                    }
                )

    if len(rows) < 2:
        raise SystemExit("Need at least 2 patches to create train/val splits.")

    random_gen.shuffle(rows)
    val_count = max(1, int(len(rows) * args.val_ratio))
    for idx, row in enumerate(rows):
        row["split"] = "val" if idx < val_count else "train"

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "image_mag", "mask"])
        writer.writeheader()
        writer.writerows(rows)

    positives = sum(np.load(mask_dir / f"{row['sample_id']}.npy").mean() > 0 for row in rows)
    print(f"Matched FITS pairs: {len(common)}")
    print(f"Wrote {len(rows)} patches to {output_dir}")
    print(f"Positive patches: {positives} | Train: {len(rows) - val_count} | Val: {val_count}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
