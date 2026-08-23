#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import random
import re
from pathlib import Path

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".npz", ".fits", ".fit", ".fts"}
DEFAULT_CHANNEL_SPECS = ["171=training_images_171", "195=training_images_195", "284=training_images_284", "304=training_images_304"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a training manifest for the UAD solar active region dataset.")
    parser.add_argument("--raw-root", default="data/raw/Solar_data_UAD", help="Root directory containing extracted archives")
    parser.add_argument("--mask-dir", default="Masks", help="Directory under raw-root containing segmentation masks")
    parser.add_argument(
        "--channel-spec",
        nargs="+",
        default=DEFAULT_CHANNEL_SPECS,
        help="Pairs of channel_name=relative_directory, e.g. 171=training_images_171",
    )
    parser.add_argument("--output", default="data/processed/uad_manifest.csv", help="Path to write the CSV manifest")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    return parser


def normalize_key(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"(mask|label|labels|image|images)", "_", stem)
    stem = re.sub(r"(171|195|284|304)", "_", stem)
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem or path.stem.lower()


def index_files(root: Path) -> dict[str, Path]:
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    mapping: dict[str, Path] = {}
    for path in files:
        key = normalize_key(path)
        if key in mapping:
            suffix = "_" + re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
            key = key + suffix
        mapping[key] = path
    return mapping


def main() -> None:
    args = build_parser().parse_args()
    raw_root = Path(args.raw_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    channel_pairs: list[tuple[str, Path]] = []
    for spec in args.channel_spec:
        if "=" not in spec:
            raise ValueError(f"Invalid --channel-spec '{spec}'. Expected name=directory.")
        channel_name, relative_dir = spec.split("=", 1)
        channel_dir = raw_root / relative_dir
        if not channel_dir.exists():
            raise FileNotFoundError(f"Missing channel directory: {channel_dir}")
        channel_pairs.append((channel_name, channel_dir))

    mask_dir = raw_root / args.mask_dir
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask directory: {mask_dir}")

    mask_map = index_files(mask_dir)
    channel_maps = {name: index_files(path) for name, path in channel_pairs}

    common_keys = set(mask_map)
    for channel_name, mapping in channel_maps.items():
        common_keys &= set(mapping)
        print(f"Indexed {len(mapping)} files for channel {channel_name}")
    print(f"Indexed {len(mask_map)} mask files")

    if not common_keys:
        raise RuntimeError(
            "Could not match any mask/image pairs. Inspect extracted filenames or adjust normalize_key() in this script."
        )

    keys = sorted(common_keys)
    random.Random(args.seed).shuffle(keys)
    val_count = max(1, int(len(keys) * args.val_ratio)) if len(keys) > 1 else 0
    val_keys = set(keys[:val_count])

    fieldnames = ["sample_id", "split", *(f"image_{name}" for name, _ in channel_pairs), "mask"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for key in keys:
            row = {
                "sample_id": key,
                "split": "val" if key in val_keys else "train",
                "mask": os.path.relpath(mask_map[key], start=output_path.parent).replace(os.sep, "/"),
            }
            for channel_name, _ in channel_pairs:
                row[f"image_{channel_name}"] = os.path.relpath(channel_maps[channel_name][key], start=output_path.parent).replace(os.sep, "/")
            writer.writerow(row)

    print(f"Wrote {len(keys)} matched samples to {output_path}")
    print(f"Train samples: {len(keys) - len(val_keys)} | Val samples: {len(val_keys)}")


if __name__ == "__main__":
    main()
