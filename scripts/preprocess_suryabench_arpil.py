#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import h5py  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("h5py is required for ARPIL preprocessing. Install requirements first.") from exc

try:
    from netCDF4 import Dataset as NetCDFDataset  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("netCDF4 is required for core-SDO preprocessing. Install requirements first.") from exc

DEFAULT_CHANNELS = ["aia171", "aia193", "hmi_m"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess local core-SDO + ARPIL data into low-RAM patch training data.")
    parser.add_argument("--core-root", required=True, help="Directory containing core-SDO .nc files")
    parser.add_argument("--mask-root", required=True, help="Directory containing ARPIL .h5 masks")
    parser.add_argument("--output-dir", required=True, help="Directory for generated patches + manifest")
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS, help="netCDF variables to extract")
    parser.add_argument("--mask-key", default="union_with_intersect", help="HDF5 dataset key for ARPIL mask")
    parser.add_argument("--resize", type=int, default=1024, help="Resize full-disk data before patch extraction")
    parser.add_argument("--patch-size", type=int, default=256, help="Patch size after resize")
    parser.add_argument("--stride", type=int, default=256, help="Patch stride after resize")
    parser.add_argument("--min-mask-fraction", type=float, default=0.003, help="Keep patches with at least this positive fraction")
    parser.add_argument("--keep-empty-every", type=int, default=20, help="Keep every Nth empty patch as background")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def timestamp_from_name(path: Path) -> str:
    match = re.search(r"(\d{8}_\d{4})", path.name)
    if not match:
        raise ValueError(f"Could not parse timestamp from filename: {path.name}")
    return match.group(1)


def iter_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def resize_stack(array: np.ndarray, size: int, mode: str) -> np.ndarray:
    tensor = torch.from_numpy(array.astype(np.float32))
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    elif tensor.ndim == 3:
        tensor = tensor[None]
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape={array.shape}")
    out = F.interpolate(tensor, size=(size, size), mode=mode, align_corners=False if mode != "nearest" else None)
    return out.squeeze(0).numpy()


def load_core_sample(nc_path: Path, channels: list[str]) -> np.ndarray:
    with NetCDFDataset(nc_path, "r") as ds:
        arrays = []
        for channel in channels:
            if channel not in ds.variables:
                raise KeyError(f"Missing channel '{channel}' in {nc_path}")
            arrays.append(np.asarray(ds.variables[channel][:], dtype=np.float32))
    return np.stack(arrays, axis=0)


def load_mask(h5_path: Path, key: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as handle:
        if key in handle:
            array = np.asarray(handle[key], dtype=np.float32)
        else:
            keys = list(handle.keys())
            if not keys:
                raise KeyError(f"No datasets found in {h5_path}")
            array = np.asarray(handle[keys[0]], dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    return (array > 0).astype(np.float32)


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    core_root = Path(args.core_root)
    mask_root = Path(args.mask_root)
    output_dir = Path(args.output_dir)
    image_root = output_dir / "images"
    mask_out_root = output_dir / "masks"
    image_root.mkdir(parents=True, exist_ok=True)
    mask_out_root.mkdir(parents=True, exist_ok=True)

    core_files = {timestamp_from_name(path): path for path in core_root.rglob("*.nc")}
    mask_files = {timestamp_from_name(path): path for path in mask_root.rglob("*.h5")}
    common = sorted(set(core_files) & set(mask_files))
    if not common:
        raise SystemExit("No timestamp-aligned .nc/.h5 pairs found. Place local core-SDO and ARPIL files first.")

    rows: list[dict[str, str]] = []
    empty_counter = 0

    for timestamp in common:
        image_stack = load_core_sample(core_files[timestamp], args.channels)
        mask = load_mask(mask_files[timestamp], args.mask_key)

        image_stack = resize_stack(image_stack, args.resize, mode="bilinear")
        mask = resize_stack(mask, args.resize, mode="nearest")
        if mask.ndim == 3:
            mask = mask[0]
        mask = (mask > 0.5).astype(np.float32)

        ys = iter_starts(mask.shape[0], args.patch_size, args.stride)
        xs = iter_starts(mask.shape[1], args.patch_size, args.stride)

        for y in ys:
            for x in xs:
                mask_patch = mask[y : y + args.patch_size, x : x + args.patch_size]
                fraction = float(mask_patch.mean())
                if fraction < args.min_mask_fraction:
                    empty_counter += 1
                    if args.keep_empty_every <= 0 or empty_counter % args.keep_empty_every != 0:
                        continue

                sample_id = f"{timestamp}_y{y:04d}_x{x:04d}"
                row = {"sample_id": sample_id, "split": "train"}
                for channel_index, channel_name in enumerate(args.channels):
                    channel_dir = image_root / channel_name
                    channel_dir.mkdir(parents=True, exist_ok=True)
                    channel_path = channel_dir / f"{sample_id}.npy"
                    np.save(channel_path, image_stack[channel_index, y : y + args.patch_size, x : x + args.patch_size].astype(np.float32))
                    row[f"image_{channel_name}"] = os.path.relpath(channel_path, start=output_dir).replace(os.sep, "/")

                mask_path = mask_out_root / f"{sample_id}.npy"
                np.save(mask_path, mask_patch.astype(np.float32))
                row["mask"] = os.path.relpath(mask_path, start=output_dir).replace(os.sep, "/")
                rows.append(row)

    rng.shuffle(rows)
    val_count = max(1, int(len(rows) * args.val_ratio)) if len(rows) > 1 else 0
    for idx, row in enumerate(rows):
        row["split"] = "val" if idx < val_count else "train"

    manifest_path = output_dir / "manifest.csv"
    fieldnames = ["sample_id", "split", *(f"image_{channel}" for channel in args.channels), "mask"]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Aligned frames: {len(common)}")
    print(f"Generated patches: {len(rows)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
