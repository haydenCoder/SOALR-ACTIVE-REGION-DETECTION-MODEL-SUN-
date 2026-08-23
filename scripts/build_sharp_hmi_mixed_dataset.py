#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits

DEFAULT_CHANNELS = ["magnetogram", "continuum", "dopplergram"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a 70/30 real/synthetic SDO/HMI dataset from online SHARPs sample files.")
    parser.add_argument("--output-dir", default="data/processed/sharp_hmi_real_synth", help="Output dataset directory")
    parser.add_argument("--patch-size", type=int, default=192, help="Real tile size before loader resize")
    parser.add_argument("--stride", type=int, default=64, help="Tile stride")
    parser.add_argument("--synthetic-fraction", type=float, default=0.30, help="Synthetic fraction inside the train split")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--min-mask-fraction", type=float, default=0.02, help="Minimum positive fraction for keeping a real tile")
    parser.add_argument("--keep-empty-every", type=int, default=12, help="Keep every Nth empty tile")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def clone_sharps_sample(tmpdir: Path) -> Path:
    repo = tmpdir / "SHARPs"
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/mbobra/SHARPs.git", str(repo)], check=True)
    return repo / "files"


def iter_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def normalize_channel(channel: np.ndarray, channel_name: str) -> np.ndarray:
    channel = np.nan_to_num(channel.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    name = channel_name.lower()
    if "mag" in name:
        scale = max(float(np.percentile(np.abs(channel), 99.5)), 1e-6)
        return (0.5 * (np.tanh(channel / scale) + 1.0)).astype(np.float32)
    if "doppler" in name:
        mean = float(np.mean(channel))
        scale = max(float(np.percentile(np.abs(channel - mean), 99.5)), 1e-6)
        return (0.5 * (np.tanh((channel - mean) / scale) + 1.0)).astype(np.float32)
    lo = float(np.percentile(channel, 1.0))
    hi = float(np.percentile(channel, 99.5))
    channel = np.clip(channel, lo, hi)
    return ((channel - lo) / max(hi - lo, 1e-6)).astype(np.float32)


def save_npz(path: Path, arr: np.ndarray) -> None:
    np.savez_compressed(path, arr.astype(np.float32))


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


def aug_real_stack(stack: np.ndarray, mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    k = rng.randint(0, 3)
    stack = np.rot90(stack, k, axes=(1, 2)).copy()
    mask = np.rot90(mask, k).copy()
    if rng.random() < 0.5:
        stack = np.flip(stack, axis=2).copy()
        mask = np.fliplr(mask).copy()
    if rng.random() < 0.5:
        stack = np.flip(stack, axis=1).copy()
        mask = np.flipud(mask).copy()
    return stack, mask


def dilate_or_erode(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    tensor = torch.from_numpy(mask[None, None].astype(np.float32))
    choice = rng.choice(["none", "dilate", "erode"])
    if choice == "none":
        return mask.astype(np.float32)
    if choice == "dilate":
        out = F.max_pool2d(tensor, kernel_size=3, stride=1, padding=1)
    else:
        out = 1.0 - F.max_pool2d(1.0 - tensor, kernel_size=3, stride=1, padding=1)
    return out[0, 0].numpy().astype(np.float32)


def synthesize(stack_a: np.ndarray, mask_a: np.ndarray, stack_b: np.ndarray, mask_b: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    stack_a, mask_a = aug_real_stack(stack_a, mask_a, rng)
    stack_b, mask_b = aug_real_stack(stack_b, mask_b, rng)
    max_shift = max(1, stack_a.shape[-1] // 12)
    sy = rng.randint(-max_shift, max_shift)
    sx = rng.randint(-max_shift, max_shift)
    stack_b = np.stack([shift_with_zero_fill(ch, sy, sx) for ch in stack_b], axis=0)
    mask_b = shift_with_zero_fill(mask_b, sy, sx)
    alpha = rng.uniform(0.55, 0.8)
    stack = alpha * stack_a + (1 - alpha) * stack_b
    stack += rng.uniform(-0.04, 0.04)
    stack = np.clip(stack, 0.0, 1.0)
    mask = np.maximum(mask_a, mask_b)
    mask = dilate_or_erode(mask, rng)
    mask = (mask > 0.5).astype(np.float32)
    return stack.astype(np.float32), mask.astype(np.float32)


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    image_root = output_dir / "images"
    mask_root = output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    for channel in DEFAULT_CHANNELS:
        (image_root / channel).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sharp_online_") as td:
        files_dir = clone_sharps_sample(Path(td))
        source_files = {
            "magnetogram": files_dir / "hmi.sharp_cea_720s.377.20110215_020000_TAI.magnetogram.fits",
            "continuum": files_dir / "hmi.sharp_cea_720s.377.20110215_020000_TAI.continuum.fits",
            "dopplergram": files_dir / "hmi.sharp_cea_720s.377.20110215_020000_TAI.Dopplergram.fits",
            "bitmap": files_dir / "hmi.sharp_cea_720s.377.20110215_020000_TAI.bitmap.fits",
        }

        channels = [normalize_channel(fits.getdata(source_files[name]), name) for name in DEFAULT_CHANNELS]
        stack = np.stack(channels, axis=0)
        bitmap = np.asarray(fits.getdata(source_files["bitmap"]), dtype=np.float32)
        mask_full = ((bitmap.astype(np.int32) & 32) > 0).astype(np.float32)

        real_records: list[dict[str, str]] = []
        provenance: list[dict[str, str]] = []
        real_patches: list[tuple[str, np.ndarray, np.ndarray]] = []
        empty_counter = 0

        ys = iter_starts(mask_full.shape[0], args.patch_size, args.stride)
        xs = iter_starts(mask_full.shape[1], args.patch_size, args.stride)
        for y in ys:
            for x in xs:
                patch_mask = mask_full[y : y + args.patch_size, x : x + args.patch_size]
                frac = float(patch_mask.mean())
                if frac < args.min_mask_fraction:
                    empty_counter += 1
                    if args.keep_empty_every <= 0 or empty_counter % args.keep_empty_every != 0:
                        continue
                patch_stack = stack[:, y : y + args.patch_size, x : x + args.patch_size]
                sample_id = f"sharp377_y{y:03d}_x{x:03d}"
                row = {"sample_id": sample_id, "split": "train"}
                for idx, channel_name in enumerate(DEFAULT_CHANNELS):
                    path = image_root / channel_name / f"{sample_id}.npz"
                    save_npz(path, patch_stack[idx])
                    row[f"image_{channel_name}"] = os.path.relpath(path, start=output_dir).replace(os.sep, "/")
                mask_path = mask_root / f"{sample_id}.npz"
                save_npz(mask_path, patch_mask)
                row["mask"] = os.path.relpath(mask_path, start=output_dir).replace(os.sep, "/")
                real_records.append(row)
                provenance.append({
                    "sample_id": sample_id,
                    "source_type": "real",
                    "source_id_a": source_files['magnetogram'].name,
                    "source_id_b": "",
                    "y": str(y),
                    "x": str(x),
                    "notes": f"mask_fraction={frac:.4f}",
                })
                real_patches.append((sample_id, patch_stack.astype(np.float32), patch_mask.astype(np.float32)))

    rng.shuffle(real_records)
    val_count = max(1, int(len(real_records) * args.val_ratio)) if len(real_records) > 1 else 0
    val_ids = {row["sample_id"] for row in real_records[:val_count]}
    train_records = [dict(row) for row in real_records if row["sample_id"] not in val_ids]
    val_records = [dict(row) for row in real_records if row["sample_id"] in val_ids]
    for row in val_records:
        row["split"] = "val"
    train_patch_lookup = {sid: (stack_arr, mask_arr) for sid, stack_arr, mask_arr in real_patches if sid not in val_ids}

    desired_synth = math.ceil(len(train_records) * args.synthetic_fraction / max(1e-6, 1 - args.synthetic_fraction))
    train_ids = list(train_patch_lookup)
    synth_records: list[dict[str, str]] = []

    if len(train_ids) >= 2:
        used_pairs = set()
        synth_index = 0
        while len(synth_records) < desired_synth:
            a, b = rng.sample(train_ids, 2)
            pair = tuple(sorted((a, b)))
            if pair in used_pairs and len(used_pairs) < (len(train_ids) * (len(train_ids) - 1)) // 2:
                continue
            used_pairs.add(pair)
            stack_a, mask_a = train_patch_lookup[a]
            stack_b, mask_b = train_patch_lookup[b]
            synth_stack, synth_mask = synthesize(stack_a, mask_a, stack_b, mask_b, rng)
            sample_id = f"synthetic_{synth_index:04d}"
            synth_index += 1
            row = {"sample_id": sample_id, "split": "train"}
            for idx, channel_name in enumerate(DEFAULT_CHANNELS):
                path = image_root / channel_name / f"{sample_id}.npz"
                save_npz(path, synth_stack[idx])
                row[f"image_{channel_name}"] = os.path.relpath(path, start=output_dir).replace(os.sep, "/")
            mask_path = mask_root / f"{sample_id}.npz"
            save_npz(mask_path, synth_mask)
            row["mask"] = os.path.relpath(mask_path, start=output_dir).replace(os.sep, "/")
            synth_records.append(row)
            provenance.append({
                "sample_id": sample_id,
                "source_type": "synthetic",
                "source_id_a": a,
                "source_id_b": b,
                "y": "",
                "x": "",
                "notes": "blend+shift+flip+rotate",
            })

    manifest_rows = train_records + synth_records + val_records
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", *(f"image_{c}" for c in DEFAULT_CHANNELS), "mask"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    sources_used_path = output_dir / "sources_used.csv"
    with sources_used_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_name", "channel", "origin_repo", "description"])
        writer.writeheader()
        writer.writerows([
            {"source_name": "hmi.sharp_cea_720s.377.20110215_020000_TAI.magnetogram.fits", "channel": "magnetogram", "origin_repo": "mbobra/SHARPs", "description": "Real HMI LOS magnetogram sample"},
            {"source_name": "hmi.sharp_cea_720s.377.20110215_020000_TAI.continuum.fits", "channel": "continuum", "origin_repo": "mbobra/SHARPs", "description": "Real HMI continuum sample"},
            {"source_name": "hmi.sharp_cea_720s.377.20110215_020000_TAI.Dopplergram.fits", "channel": "dopplergram", "origin_repo": "mbobra/SHARPs", "description": "Real HMI Dopplergram sample"},
            {"source_name": "hmi.sharp_cea_720s.377.20110215_020000_TAI.bitmap.fits", "channel": "mask", "origin_repo": "mbobra/SHARPs", "description": "Real SHARP bitmap mask"},
        ])

    provenance_path = output_dir / "sample_provenance.csv"
    with provenance_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "source_type", "source_id_a", "source_id_b", "y", "x", "notes"])
        writer.writeheader()
        writer.writerows(provenance)

    print(f"Real samples: {len(real_records)}")
    print(f"Train real: {len(train_records)} | Synthetic train: {len(synth_records)} | Val real: {len(val_records)}")
    print(f"Manifest: {manifest_path}")
    print(f"Saved source list: {sources_used_path}")
    print(f"Saved provenance: {provenance_path}")


if __name__ == "__main__":
    main()
