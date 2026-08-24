#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import random
import tempfile
from pathlib import Path

import boto3
import h5py
import numpy as np
import requests
from botocore import UNSIGNED
from botocore.client import Config
from netCDF4 import Dataset as NetCDFDataset

MASK_SPLIT_URLS = {
    "train": "https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-ar-segmentation/raw/main/train.csv",
    "validation": "https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-ar-segmentation/raw/main/validation.csv",
    "test": "https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-ar-segmentation/raw/main/test.csv",
    "leaky_validation": "https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-ar-segmentation/raw/main/leaky_validation.csv",
}
MASK_BASE_URL = "https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-ar-segmentation/resolve/main/"
CORE_BUCKET = "nasa-surya-bench"
DEFAULT_CHANNELS = ["aia171", "aia193", "hmi_m"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream-download ARPIL masks + core-SDO 3-channel frames, then build 512-sized training tiles."
    )
    parser.add_argument("--output-dir", default="data/processed/arpil_3ch_512", help="Directory for compressed tiles and manifest")
    parser.add_argument("--split", choices=sorted(MASK_SPLIT_URLS), default="train", help="ARPIL split to sample from")
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS, help="core-SDO variables to extract")
    parser.add_argument("--patch-size", type=int, default=512, help="Tile size in pixels")
    parser.add_argument("--stride", type=int, default=512, help="Tile stride in pixels")
    parser.add_argument("--max-frames", type=int, default=48, help="Maximum aligned hourly frames to process")
    parser.add_argument("--start-index", type=int, default=0, help="Skip this many valid CSV rows before processing")
    parser.add_argument("--min-mask-fraction", type=float, default=0.0025, help="Keep tiles with at least this positive fraction")
    parser.add_argument("--keep-empty-every", type=int, default=32, help="Retain every Nth empty tile as background")
    parser.add_argument("--mask-key", default="union_with_intersect", help="HDF5 dataset key for ARPIL masks")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation fraction inside the built manifest")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def iter_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def read_csv_rows(url: str) -> list[dict[str, str]]:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return list(csv.DictReader(response.text.splitlines()))


def download_mask(mask_rel_path: str, destination: Path) -> None:
    url = MASK_BASE_URL + mask_rel_path
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def download_core_s3(s3_client, key: str, destination: Path) -> None:
    s3_client.download_file(CORE_BUCKET, key, str(destination))


def load_core_channels(nc_path: Path, channels: list[str]) -> np.ndarray:
    with NetCDFDataset(nc_path, "r") as ds:
        arrays = []
        for channel in channels:
            if channel not in ds.variables:
                raise KeyError(f"Missing variable '{channel}' in {nc_path.name}")
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


def save_compressed(path: Path, array: np.ndarray) -> None:
    np.savez_compressed(path, array.astype(np.float32))


def core_key_from_mask_path(mask_path: str) -> str:
    return mask_path.replace("data/", "").replace(".h5", ".nc")


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root = output_dir / "images"
    mask_root = output_dir / "masks"
    image_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    for channel in args.channels:
        (image_root / channel).mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(MASK_SPLIT_URLS[args.split])
    rows = [row for row in rows if row.get("present", "0") not in {"0", "0.0", "", None} and row.get("file_path")]
    rows = rows[args.start_index : args.start_index + args.max_frames]
    if not rows:
        raise SystemExit("No ARPIL rows selected. Adjust --split/--start-index/--max-frames.")

    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    manifest_rows: list[dict[str, str]] = []
    empty_counter = 0
    kept_tiles = 0

    with tempfile.TemporaryDirectory(prefix="arpil_stream_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for frame_idx, row in enumerate(rows, start=1):
            mask_rel_path = row["file_path"]
            core_key = core_key_from_mask_path(mask_rel_path)
            timestamp = Path(mask_rel_path).stem
            nc_path = tmpdir_path / f"{timestamp}.nc"
            h5_path = tmpdir_path / f"{timestamp}.h5"

            print(f"[{frame_idx}/{len(rows)}] downloading core={core_key} mask={mask_rel_path}")
            download_core_s3(s3_client, core_key, nc_path)
            download_mask(mask_rel_path, h5_path)

            image_stack = load_core_channels(nc_path, args.channels)
            mask = load_mask(h5_path, args.mask_key)

            if image_stack.shape[1:] != mask.shape:
                raise ValueError(
                    f"Shape mismatch for {timestamp}: image={image_stack.shape[1:]} mask={mask.shape}."
                )

            ys = iter_starts(mask.shape[0], args.patch_size, args.stride)
            xs = iter_starts(mask.shape[1], args.patch_size, args.stride)

            for y in ys:
                for x in xs:
                    mask_patch = mask[y : y + args.patch_size, x : x + args.patch_size]
                    positive_fraction = float(mask_patch.mean())
                    if positive_fraction < args.min_mask_fraction:
                        empty_counter += 1
                        if args.keep_empty_every <= 0 or empty_counter % args.keep_empty_every != 0:
                            continue

                    sample_id = f"{timestamp}_y{y:04d}_x{x:04d}"
                    manifest_row = {"sample_id": sample_id, "split": "train"}
                    for channel_idx, channel_name in enumerate(args.channels):
                        out_path = image_root / channel_name / f"{sample_id}.npz"
                        save_compressed(out_path, image_stack[channel_idx, y : y + args.patch_size, x : x + args.patch_size])
                        manifest_row[f"image_{channel_name}"] = os.path.relpath(out_path, start=output_dir).replace(os.sep, "/")
                    mask_out = mask_root / f"{sample_id}.npz"
                    save_compressed(mask_out, mask_patch)
                    manifest_row["mask"] = os.path.relpath(mask_out, start=output_dir).replace(os.sep, "/")
                    manifest_rows.append(manifest_row)
                    kept_tiles += 1

            nc_path.unlink(missing_ok=True)
            h5_path.unlink(missing_ok=True)

    rng.shuffle(manifest_rows)
    val_count = max(1, int(len(manifest_rows) * args.val_ratio)) if len(manifest_rows) > 1 else 0
    for idx, row in enumerate(manifest_rows):
        row["split"] = "val" if idx < val_count else "train"

    manifest_path = output_dir / "manifest.csv"
    fieldnames = ["sample_id", "split", *(f"image_{channel}" for channel in args.channels), "mask"]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Frames processed: {len(rows)}")
    print(f"Tiles kept: {kept_tiles}")
    print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
