#!/usr/bin/env python3
from __future__ import annotations

"""
Solar ARPIL Plugin
==================
One-file pipeline that can:
1. auto-install dependencies if missing,
2. download real ARPIL + core-SDO data when reachable,
3. convert data to HDF5 training tiles,
4. build a 70% real / 30% synthetic train split,
5. save provenance and selected frame lists,
6. resume training automatically from checkpoints,
7. write verbose logs and periodic preview images.

Default target runtime:
- 15 GB RAM-ish CPU box
- 4 CPU threads
- 3 physically meaningful channels: AIA 171, AIA 193, HMI magnetogram
"""

import argparse
import csv
import importlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter

# CPU-oriented defaults. Users can still override via env or CLI.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

REQUIRED_IMPORTS = {
    "boto3": "boto3",
    "h5py": "h5py",
    "numpy": "numpy",
    "requests": "requests",
    "torch": "torch",
    "astropy": "astropy",
    "netCDF4": "netCDF4",
    "PIL": "Pillow",
}

MASK_SPLIT_URLS = {
    "train": "https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-ar-segmentation/raw/main/train.csv",
    "validation": "https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-ar-segmentation/raw/main/validation.csv",
    "test": "https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-ar-segmentation/raw/main/test.csv",
    "leaky_validation": "https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-ar-segmentation/raw/main/leaky_validation.csv",
}
CORE_INDEX_URLS = {
    "train": "https://huggingface.co/datasets/nasa-ibm-ai4science/core-sdo/resolve/main/train_index_surya_1_0.csv?download=true",
    "validation": "https://huggingface.co/datasets/nasa-ibm-ai4science/core-sdo/raw/main/valid_index_surya_1_0.csv",
    "test": "https://huggingface.co/datasets/nasa-ibm-ai4science/core-sdo/resolve/main/test.csv?download=true",
    "leaky_validation": "https://huggingface.co/datasets/nasa-ibm-ai4science/core-sdo/raw/main/valid_index_surya_1_0.csv",
}
MASK_BASE_URL = "https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-ar-segmentation/resolve/main/"
CORE_BUCKET = "nasa-surya-bench"
DEFAULT_CHANNELS = ["aia171", "aia193", "hmi_m"]
SHARP_SAMPLE_REPO = "https://github.com/mbobra/SHARPs.git"
SHARP_SAMPLE_FILES = {
    "magnetogram": "hmi.sharp_cea_720s.377.20110215_020000_TAI.magnetogram.fits",
    "continuum": "hmi.sharp_cea_720s.377.20110215_020000_TAI.continuum.fits",
    "dopplergram": "hmi.sharp_cea_720s.377.20110215_020000_TAI.Dopplergram.fits",
    "bitmap": "hmi.sharp_cea_720s.377.20110215_020000_TAI.bitmap.fits",
}
FALLBACK_CHANNELS = ["magnetogram", "continuum", "dopplergram"]


# Runtime-populated globals after dependency bootstrap
boto3 = None
h5py = None
np = None
requests = None
torch = None
F = None
fits = None
UNSIGNED = None
Config = None
NetCDFDataset = None
nn = None
DataLoader = None
Dataset = None
Image = None
ImageOps = None


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_dice: float
    val_iou: float
    learning_rate: float
    seconds: float


class Logger:
    def __init__(self, run_dir: Path, verbose: bool = False) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.console_path = run_dir / "console.log"
        self.handle = self.console_path.open("a", encoding="utf-8")
        self.verbose_enabled = verbose

    def log(self, message: str) -> None:
        print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def debug(self, message: str) -> None:
        if self.verbose_enabled:
            self.log(f"[verbose] {message}")

    def close(self) -> None:
        self.handle.close()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def bootstrap_dependencies(requirements_path: Path, auto_install: bool = True) -> None:
    missing = [pkg for module, pkg in REQUIRED_IMPORTS.items() if importlib.util.find_spec(module) is None]
    if not missing:
        return
    if not auto_install:
        raise RuntimeError(f"Missing Python packages: {missing}. Install requirements first.")

    print(f"Missing packages detected: {missing}", flush=True)
    if requirements_path.exists():
        print(f"Installing dependencies from {requirements_path} ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)], check=True)
    else:
        print("requirements.txt not found, installing fallback package list...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", *sorted(set(missing))], check=True)

    importlib.invalidate_caches()


def load_runtime_modules() -> None:
    global boto3, h5py, np, requests, torch, F, fits, UNSIGNED, Config, NetCDFDataset, nn, DataLoader, Dataset, Image, ImageOps
    import boto3 as _boto3
    import h5py as _h5py
    import numpy as _np
    import requests as _requests
    import torch as _torch
    import torch.nn.functional as _F
    from astropy.io import fits as _fits
    from botocore import UNSIGNED as _UNSIGNED
    from botocore.client import Config as _Config
    from netCDF4 import Dataset as _NetCDFDataset
    from PIL import Image as _Image, ImageOps as _ImageOps
    from torch import nn as _nn
    from torch.utils.data import DataLoader as _DataLoader, Dataset as _Dataset

    boto3 = _boto3
    h5py = _h5py
    np = _np
    requests = _requests
    torch = _torch
    F = _F
    fits = _fits
    UNSIGNED = _UNSIGNED
    Config = _Config
    NetCDFDataset = _NetCDFDataset
    nn = _nn
    DataLoader = _DataLoader
    Dataset = _Dataset
    Image = _Image
    ImageOps = _ImageOps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="All-in-one ARPIL/SDO training plugin")
    parser.add_argument("--source-mode", choices=["arpil_sdo", "sharp_sample"], default="arpil_sdo")
    parser.add_argument("--work-dir", default="plugin_runs/arpil_plugin", help="Main output directory")
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS, help="3-channel stack for ARPIL/core-SDO mode")
    parser.add_argument("--mask-split", choices=sorted(MASK_SPLIT_URLS), default="train")
    parser.add_argument("--mask-key", default="union_with_intersect")
    parser.add_argument("--real-ratio", type=float, default=0.70)
    parser.add_argument("--synthetic-ratio", type=float, default=0.30)
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--random-frame-order", action="store_true")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-stride", type=int, default=512)
    parser.add_argument("--min-mask-fraction", type=float, default=0.0025)
    parser.add_argument("--keep-empty-every", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accumulation-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", choices=["auto", "always", "never"], default="auto", help="Checkpoint resume behavior")
    parser.add_argument("--preview-every", type=int, default=5, help="Save preview images every N epochs (0 disables)")
    parser.add_argument("--num-preview-samples", type=int, default=2, help="How many fixed validation samples to render in previews")
    parser.add_argument("--force-rebuild-dataset", action="store_true", help="Rebuild dataset even if manifest already exists")
    parser.add_argument("--no-auto-install-deps", action="store_true", help="Disable automatic dependency installation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_cpu_runtime(cpu_threads: int) -> None:
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)


def download_text(url: str, timeout: int = 300) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def download_binary(url: str, destination: Path, timeout: int = 600) -> None:
    if destination.exists():
        return
    ensure_dir(destination.parent)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def normalize_core_key(path_value: str) -> str:
    path_value = path_value.strip()
    filename = Path(path_value).name
    if "/" in path_value:
        return path_value
    timestamp = Path(filename).stem
    year = timestamp[:4]
    month = timestamp[4:6]
    return f"{year}/{month}/{filename}"


def read_csv_from_text(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(text.splitlines()))


def align_arpil_and_core(mask_rows: list[dict[str, str]], core_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    mask_map: dict[str, dict[str, str]] = {}
    for row in mask_rows:
        if row.get("present", "0") in {"0", "0.0", "", None}:
            continue
        mask_path = row.get("file_path", "")
        if not mask_path:
            continue
        mask_map[Path(mask_path).stem] = row

    core_map: dict[str, dict[str, str]] = {}
    for row in core_rows:
        if row.get("present", "1") in {"0", "0.0", "", None}:
            continue
        key = normalize_core_key(row["path"])
        core_map[Path(key).stem] = {**row, "normalized_key": key}

    timestamps = sorted(set(mask_map) & set(core_map))
    return [{"timestamp": ts, "mask": mask_map[ts], "core": core_map[ts]} for ts in timestamps]


def iter_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def normalize_channel(channel: np.ndarray, channel_name: str) -> np.ndarray:
    channel = np.nan_to_num(channel.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    name = channel_name.lower()
    if name.startswith("aia") or name in {"171", "193", "211", "304", "335", "94", "131", "1600", "1700"}:
        channel = np.maximum(channel, 0.0)
        channel = np.log1p(channel)
        lo = float(np.percentile(channel, 1.0))
        hi = float(np.percentile(channel, 99.5))
        channel = np.clip(channel, lo, hi)
        return ((channel - lo) / max(hi - lo, 1e-6)).astype(np.float32)
    if "hmi" in name or "mag" in name or name in {"m", "bx", "by", "bz"}:
        scale = max(float(np.percentile(np.abs(channel), 99.5)), 1e-6)
        return (0.5 * (np.tanh(channel / scale) + 1.0)).astype(np.float32)
    if "doppler" in name or name.endswith("_v"):
        mean = float(np.mean(channel))
        scale = max(float(np.percentile(np.abs(channel - mean), 99.5)), 1e-6)
        return (0.5 * (np.tanh((channel - mean) / scale) + 1.0)).astype(np.float32)
    lo = float(np.percentile(channel, 1.0))
    hi = float(np.percentile(channel, 99.5))
    channel = np.clip(channel, lo, hi)
    return ((channel - lo) / max(hi - lo, 1e-6)).astype(np.float32)


def read_core_stack(nc_path: Path, channels: list[str]) -> np.ndarray:
    with NetCDFDataset(nc_path, "r") as ds:
        arrays = []
        for channel in channels:
            if channel not in ds.variables:
                raise KeyError(f"Missing variable '{channel}' in {nc_path.name}")
            arrays.append(normalize_channel(np.asarray(ds.variables[channel][:], dtype=np.float32), channel))
    return np.stack(arrays, axis=0)


def read_arpil_mask(h5_path: Path, key: str) -> np.ndarray:
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


def download_core_s3(s3_client, key: str, destination: Path) -> None:
    if destination.exists():
        return
    ensure_dir(destination.parent)
    s3_client.download_file(CORE_BUCKET, key, str(destination))


def write_tile_h5(path: Path, image: np.ndarray, mask: np.ndarray, metadata: dict[str, str | int | float]) -> None:
    ensure_dir(path.parent)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("image", data=image.astype(np.float32), compression="gzip")
        handle.create_dataset("mask", data=mask.astype(np.float32), compression="gzip")
        for key, value in metadata.items():
            handle.attrs[key] = value


def load_tile_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        image = np.asarray(handle["image"], dtype=np.float32)
        mask = np.asarray(handle["mask"], dtype=np.float32)
    return image, mask


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


def rotate_flip(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    k = rng.randint(0, 3)
    image = np.rot90(image, k, axes=(1, 2)).copy()
    mask = np.rot90(mask, k, axes=(1, 2)).copy()
    if rng.random() < 0.5:
        image = np.flip(image, axis=2).copy()
        mask = np.flip(mask, axis=2).copy()
    if rng.random() < 0.5:
        image = np.flip(image, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()
    return image, mask


def dilate_or_erode(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    tensor = torch.from_numpy(mask[None].astype(np.float32))
    mode = rng.choice(["none", "dilate", "erode"])
    if mode == "none":
        return mask.astype(np.float32)
    if mode == "dilate":
        out = F.max_pool2d(tensor, kernel_size=3, stride=1, padding=1)
    else:
        out = 1.0 - F.max_pool2d(1.0 - tensor, kernel_size=3, stride=1, padding=1)
    return out[0].numpy().astype(np.float32)


def synthesize(image_a: np.ndarray, mask_a: np.ndarray, image_b: np.ndarray, mask_b: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    image_a, mask_a = rotate_flip(image_a, mask_a, rng)
    image_b, mask_b = rotate_flip(image_b, mask_b, rng)
    max_shift = max(1, image_a.shape[-1] // 12)
    sy = rng.randint(-max_shift, max_shift)
    sx = rng.randint(-max_shift, max_shift)
    image_b = np.stack([shift_with_zero_fill(ch, sy, sx) for ch in image_b], axis=0)
    mask_b = np.stack([shift_with_zero_fill(ch, sy, sx) for ch in mask_b], axis=0)
    alpha = rng.uniform(0.55, 0.8)
    image = np.clip(alpha * image_a + (1 - alpha) * image_b + rng.uniform(-0.04, 0.04), 0.0, 1.0)
    mask = np.maximum(mask_a, mask_b)
    mask = dilate_or_erode(mask, rng)
    mask = (mask > 0.5).astype(np.float32)
    return image.astype(np.float32), mask.astype(np.float32)


def maybe_skip_dataset_build(dataset_dir: Path, logger: Logger, force_rebuild: bool) -> bool:
    manifest = dataset_dir / "manifest.csv"
    provenance = dataset_dir / "sample_provenance.csv"
    if not force_rebuild and manifest.exists() and provenance.exists():
        logger.log("Existing dataset manifest detected, skipping dataset rebuild.")
        return True
    return False


def build_sharp_sample_dataset(args: argparse.Namespace, dataset_dir: Path, logger: Logger) -> None:
    if maybe_skip_dataset_build(dataset_dir, logger, args.force_rebuild_dataset):
        return

    tiles_dir = ensure_dir(dataset_dir / "tiles")
    provenance_rows: list[dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="sharp_plugin_") as td:
        repo = Path(td) / "SHARPs"
        logger.log("Cloning SHARPs sample repository from GitHub...")
        subprocess.run(["git", "clone", "--depth", "1", SHARP_SAMPLE_REPO, str(repo)], check=True)
        files_dir = repo / "files"
        stack = np.stack(
            [
                normalize_channel(fits.getdata(files_dir / SHARP_SAMPLE_FILES["magnetogram"]), "magnetogram"),
                normalize_channel(fits.getdata(files_dir / SHARP_SAMPLE_FILES["continuum"]), "continuum"),
                normalize_channel(fits.getdata(files_dir / SHARP_SAMPLE_FILES["dopplergram"]), "dopplergram"),
            ],
            axis=0,
        )
        bitmap = np.asarray(fits.getdata(files_dir / SHARP_SAMPLE_FILES["bitmap"]), dtype=np.float32)
        mask_full = (((bitmap.astype(np.int32) & 32) > 0).astype(np.float32))[None]

        real_ids: list[str] = []
        empty_counter = 0
        ys = iter_starts(mask_full.shape[1], args.tile_size, args.tile_stride)
        xs = iter_starts(mask_full.shape[2], args.tile_size, args.tile_stride)
        for y in ys:
            for x in xs:
                patch_mask = mask_full[:, y : y + args.tile_size, x : x + args.tile_size]
                fraction = float(patch_mask.mean())
                if fraction < args.min_mask_fraction:
                    empty_counter += 1
                    if args.keep_empty_every <= 0 or empty_counter % args.keep_empty_every != 0:
                        continue
                patch_image = stack[:, y : y + args.tile_size, x : x + args.tile_size]
                sample_id = f"sharp377_y{y:04d}_x{x:04d}"
                write_tile_h5(
                    tiles_dir / f"{sample_id}.h5",
                    patch_image,
                    patch_mask,
                    {
                        "sample_id": sample_id,
                        "source_type": "real",
                        "source_frame": SHARP_SAMPLE_FILES["magnetogram"],
                        "frame_timestamp": "20110215_020000",
                        "y": y,
                        "x": x,
                        "channels": ",".join(FALLBACK_CHANNELS),
                    },
                )
                real_ids.append(sample_id)
                logger.debug(f"keep_real_tile sample_id={sample_id} mask_fraction={fraction:.5f}")
                provenance_rows.append(
                    {
                        "sample_id": sample_id,
                        "source_type": "real",
                        "source_a": SHARP_SAMPLE_FILES["magnetogram"],
                        "source_b": "",
                        "frame_timestamp": "20110215_020000",
                        "y": str(y),
                        "x": str(x),
                        "notes": f"mask_fraction={fraction:.5f}",
                    }
                )

    build_manifest_and_synthetic(
        dataset_dir=dataset_dir,
        real_ids=real_ids,
        channels=FALLBACK_CHANNELS,
        val_ratio=args.val_ratio,
        synthetic_ratio=args.synthetic_ratio,
        seed=args.seed,
        provenance_rows=provenance_rows,
        logger=logger,
    )

    with (dataset_dir / "sources_used.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_name", "channel", "origin_repo", "description"])
        writer.writeheader()
        writer.writerows(
            [
                {"source_name": SHARP_SAMPLE_FILES["magnetogram"], "channel": "magnetogram", "origin_repo": "mbobra/SHARPs", "description": "Real HMI LOS magnetogram sample"},
                {"source_name": SHARP_SAMPLE_FILES["continuum"], "channel": "continuum", "origin_repo": "mbobra/SHARPs", "description": "Real HMI continuum sample"},
                {"source_name": SHARP_SAMPLE_FILES["dopplergram"], "channel": "dopplergram", "origin_repo": "mbobra/SHARPs", "description": "Real HMI Dopplergram sample"},
                {"source_name": SHARP_SAMPLE_FILES["bitmap"], "channel": "mask", "origin_repo": "mbobra/SHARPs", "description": "Real SHARP bitmap mask"},
            ]
        )


def build_arpil_sdo_dataset(args: argparse.Namespace, dataset_dir: Path, logger: Logger) -> None:
    if maybe_skip_dataset_build(dataset_dir, logger, args.force_rebuild_dataset):
        return

    rng = random.Random(args.seed)
    download_cache = ensure_dir(dataset_dir / "download_cache")
    core_cache = ensure_dir(download_cache / "core")
    mask_cache = ensure_dir(download_cache / "mask")
    tiles_dir = ensure_dir(dataset_dir / "tiles")
    metadata_dir = ensure_dir(dataset_dir / "metadata")

    logger.log(f"Downloading ARPIL split CSV: {args.mask_split}")
    mask_rows = read_csv_from_text(download_text(MASK_SPLIT_URLS[args.mask_split]))
    logger.log(f"Downloading core-SDO index CSV: {args.mask_split}")
    core_rows = read_csv_from_text(download_text(CORE_INDEX_URLS[args.mask_split]))
    aligned = align_arpil_and_core(mask_rows, core_rows)
    if not aligned:
        raise RuntimeError("No aligned ARPIL/core-SDO timestamps were found.")

    if args.random_frame_order:
        rng.shuffle(aligned)
    selected = aligned[args.start_index : args.start_index + args.max_frames]
    logger.debug(f"aligned_frames={len(aligned)} selected_frames={len(selected)}")

    with (metadata_dir / "selected_frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "core_key", "mask_path"])
        writer.writeheader()
        for item in selected:
            writer.writerow({"timestamp": item["timestamp"], "core_key": item["core"]["normalized_key"], "mask_path": item["mask"]["file_path"]})

    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    real_ids: list[str] = []
    provenance_rows: list[dict[str, str]] = []
    empty_counter = 0

    for frame_index, item in enumerate(selected, start=1):
        timestamp = item["timestamp"]
        core_key = item["core"]["normalized_key"]
        mask_rel = item["mask"]["file_path"]
        core_local = core_cache / core_key
        mask_local = mask_cache / Path(mask_rel).name
        logger.log(f"[{frame_index}/{len(selected)}] fetching {timestamp}")
        logger.debug(f"core_s3_key={core_key} -> {core_local}")
        logger.debug(f"mask_url={MASK_BASE_URL + mask_rel} -> {mask_local}")
        download_core_s3(s3_client, core_key, core_local)
        download_binary(MASK_BASE_URL + mask_rel, mask_local)

        stack = read_core_stack(core_local, args.channels)
        mask_full = read_arpil_mask(mask_local, args.mask_key)[None]
        if stack.shape[1:] != mask_full.shape[1:]:
            raise ValueError(f"Shape mismatch for {timestamp}: image={stack.shape[1:]} mask={mask_full.shape[1:]}")

        ys = iter_starts(mask_full.shape[1], args.tile_size, args.tile_stride)
        xs = iter_starts(mask_full.shape[2], args.tile_size, args.tile_stride)
        for y in ys:
            for x in xs:
                patch_mask = mask_full[:, y : y + args.tile_size, x : x + args.tile_size]
                fraction = float(patch_mask.mean())
                if fraction < args.min_mask_fraction:
                    empty_counter += 1
                    if args.keep_empty_every <= 0 or empty_counter % args.keep_empty_every != 0:
                        continue
                patch_image = stack[:, y : y + args.tile_size, x : x + args.tile_size]
                sample_id = f"{timestamp}_y{y:04d}_x{x:04d}"
                write_tile_h5(
                    tiles_dir / f"{sample_id}.h5",
                    patch_image,
                    patch_mask,
                    {
                        "sample_id": sample_id,
                        "source_type": "real",
                        "source_frame": timestamp,
                        "source_core_key": core_key,
                        "source_mask_path": mask_rel,
                        "frame_timestamp": timestamp,
                        "y": y,
                        "x": x,
                        "channels": ",".join(args.channels),
                    },
                )
                real_ids.append(sample_id)
                logger.debug(f"keep_real_tile sample_id={sample_id} mask_fraction={fraction:.5f}")
                provenance_rows.append(
                    {
                        "sample_id": sample_id,
                        "source_type": "real",
                        "source_a": core_key,
                        "source_b": mask_rel,
                        "frame_timestamp": timestamp,
                        "y": str(y),
                        "x": str(x),
                        "notes": f"mask_fraction={fraction:.5f}",
                    }
                )

    build_manifest_and_synthetic(
        dataset_dir=dataset_dir,
        real_ids=real_ids,
        channels=args.channels,
        val_ratio=args.val_ratio,
        synthetic_ratio=args.synthetic_ratio,
        seed=args.seed,
        provenance_rows=provenance_rows,
        logger=logger,
    )

    with (dataset_dir / "sources_used.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_name", "channel", "origin_repo", "description"])
        writer.writeheader()
        for channel in args.channels:
            writer.writerow(
                {
                    "source_name": channel,
                    "channel": channel,
                    "origin_repo": "nasa-ibm-ai4science/core-sdo + surya-bench-ar-segmentation",
                    "description": "Real ARPIL/core-SDO stream",
                }
            )


def build_manifest_and_synthetic(
    dataset_dir: Path,
    real_ids: list[str],
    channels: list[str],
    val_ratio: float,
    synthetic_ratio: float,
    seed: int,
    provenance_rows: list[dict[str, str]],
    logger: Logger,
) -> None:
    rng = random.Random(seed)
    tiles_dir = dataset_dir / "tiles"
    real_ids = sorted(real_ids)
    rng.shuffle(real_ids)
    val_count = max(1, int(len(real_ids) * val_ratio)) if len(real_ids) > 1 else 0
    val_ids = set(real_ids[:val_count])
    train_real_ids = [sid for sid in real_ids if sid not in val_ids]
    val_real_ids = [sid for sid in real_ids if sid in val_ids]

    max_unique_pairs = max(0, (len(train_real_ids) * (len(train_real_ids) - 1)) // 2)
    desired_synth = math.ceil(len(train_real_ids) * synthetic_ratio / max(1e-6, 1.0 - synthetic_ratio))
    if desired_synth > max_unique_pairs:
        logger.log(f"Requested {desired_synth} synthetic samples, but only {max_unique_pairs} unique train pairs exist. Capping synthetic count.")
        desired_synth = max_unique_pairs

    used_pairs: set[tuple[str, str]] = set()
    synth_ids: list[str] = []
    while len(synth_ids) < desired_synth:
        a, b = rng.sample(train_real_ids, 2)
        pair = tuple(sorted((a, b)))
        if pair in used_pairs:
            continue
        used_pairs.add(pair)
        image_a, mask_a = load_tile_h5(tiles_dir / f"{a}.h5")
        image_b, mask_b = load_tile_h5(tiles_dir / f"{b}.h5")
        image, mask = synthesize(image_a, mask_a, image_b, mask_b, rng)
        sample_id = f"synthetic_{len(synth_ids):05d}"
        write_tile_h5(
            tiles_dir / f"{sample_id}.h5",
            image,
            mask,
            {
                "sample_id": sample_id,
                "source_type": "synthetic",
                "source_a": a,
                "source_b": b,
                "channels": ",".join(channels),
            },
        )
        synth_ids.append(sample_id)
        logger.debug(f"create_synthetic sample_id={sample_id} source_a={a} source_b={b}")
        provenance_rows.append(
            {
                "sample_id": sample_id,
                "source_type": "synthetic",
                "source_a": a,
                "source_b": b,
                "frame_timestamp": "",
                "y": "",
                "x": "",
                "notes": "blend+shift+flip+rotate+mask_morph",
            }
        )

    manifest_path = dataset_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["sample_id", "split", "tile_path", "source_type"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sid in train_real_ids:
            writer.writerow({"sample_id": sid, "split": "train", "tile_path": f"tiles/{sid}.h5", "source_type": "real"})
        for sid in synth_ids:
            writer.writerow({"sample_id": sid, "split": "train", "tile_path": f"tiles/{sid}.h5", "source_type": "synthetic"})
        for sid in val_real_ids:
            writer.writerow({"sample_id": sid, "split": "val", "tile_path": f"tiles/{sid}.h5", "source_type": "real"})

    provenance_path = dataset_dir / "sample_provenance.csv"
    with provenance_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["sample_id", "source_type", "source_a", "source_b", "frame_timestamp", "y", "x", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(provenance_rows)

    unique_count = len({row["sample_id"] for row in provenance_rows})
    logger.log(f"Real samples: {len(real_ids)}")
    logger.log(f"Train real: {len(train_real_ids)} | Synthetic train: {len(synth_ids)} | Val real: {len(val_real_ids)}")
    logger.log(f"Unique sample IDs: {unique_count}")
    logger.log(f"Manifest: {manifest_path}")
    logger.log(f"Saved provenance: {provenance_path}")


# Bootstrap runtime modules before torch/nn dependent class definitions.
_PLUGIN_REQUIREMENTS = Path(__file__).resolve().parent / "requirements.txt"
bootstrap_dependencies(_PLUGIN_REQUIREMENTS, auto_install=os.getenv("SOLAR_PLUGIN_NO_AUTO_INSTALL", "0") != "1")
load_runtime_modules()

class H5TileDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, augment: bool, seed: int = 42) -> None:
        self.manifest_path = manifest_path
        self.split = split
        self.augment = augment
        self.random = random.Random(seed)
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows found for split='{split}' in {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image, mask = load_tile_h5(self.manifest_path.parent / row["tile_path"])
        image_t = torch.from_numpy(image)
        mask_t = torch.from_numpy(mask)
        if self.augment:
            if self.random.random() < 0.5:
                image_t = torch.flip(image_t, dims=[2])
                mask_t = torch.flip(mask_t, dims=[2])
            if self.random.random() < 0.5:
                image_t = torch.flip(image_t, dims=[1])
                mask_t = torch.flip(mask_t, dims=[1])
            k = self.random.randint(0, 3)
            if k:
                image_t = torch.rot90(image_t, k, dims=[1, 2])
                mask_t = torch.rot90(mask_t, k, dims=[1, 2])
        return image_t.contiguous(), mask_t.contiguous()


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return skip, self.pool(skip)


class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_projection = nn.Sequential(nn.Conv2d(gate_channels, inter_channels, 1), nn.BatchNorm2d(inter_channels))
        self.skip_projection = nn.Sequential(nn.Conv2d(skip_channels, inter_channels, 1), nn.BatchNorm2d(inter_channels))
        self.psi = nn.Sequential(nn.Conv2d(inter_channels, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        attention = self.relu(self.gate_projection(gate) + self.skip_projection(skip))
        return skip * self.psi(attention)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.attention = AttentionGate(out_channels, skip_channels, max(out_channels // 2, 1))
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.attention(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32, dropout: float = 0.05) -> None:
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.enc1 = EncoderBlock(in_channels, widths[0], dropout)
        self.enc2 = EncoderBlock(widths[0], widths[1], dropout)
        self.enc3 = EncoderBlock(widths[1], widths[2], dropout)
        self.enc4 = EncoderBlock(widths[2], widths[3], dropout)
        self.bridge = ConvBlock(widths[3], widths[3] * 2, dropout)
        self.dec4 = DecoderBlock(widths[3] * 2, widths[3], widths[3], dropout)
        self.dec3 = DecoderBlock(widths[3], widths[2], widths[2], dropout)
        self.dec2 = DecoderBlock(widths[2], widths[1], widths[1], dropout)
        self.dec1 = DecoderBlock(widths[1], widths[0], widths[0], dropout)
        self.head = nn.Conv2d(widths[0], out_channels, 1)

    def forward(self, x):
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        x = self.bridge(x)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.head(x)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        numerator = 2 * (probs * targets).sum(dim=(1, 2, 3)) + 1e-6
        denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-6
        dice_loss = 1 - (numerator / denominator).mean()
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


@torch.no_grad()
def segmentation_metrics(logits, targets):
    preds = (torch.sigmoid(logits) > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    pred_area = preds.sum(dim=(1, 2, 3))
    target_area = targets.sum(dim=(1, 2, 3))
    union = pred_area + target_area - intersection
    dice = ((2 * intersection + 1e-6) / (pred_area + target_area + 1e-6)).mean().item()
    iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
    return dice, iou


def update_progress_markdown(run_dir: Path, total_epochs: int, last: EpochMetrics | None, best: EpochMetrics | None) -> None:
    lines = [
        "# Training Progress",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Live files",
        f"- `{run_dir / 'console.log'}`",
        f"- `{run_dir / 'metrics.jsonl'}`",
        f"- `{run_dir / 'best.pt'}`",
        f"- `{run_dir / 'last.pt'}`",
        f"- `{run_dir / 'previews'}`",
    ]
    if last is not None:
        lines += [
            "",
            "## Current",
            f"- Last epoch: {last.epoch} / {total_epochs}",
            f"- Last val Dice: {last.val_dice:.4f}",
            f"- Last val IoU: {last.val_iou:.4f}",
            f"- Last val loss: {last.val_loss:.4f}",
        ]
    if best is not None:
        lines += [
            "",
            "## Best so far",
            f"- Best epoch: {best.epoch}",
            f"- Best val Dice: {best.val_dice:.4f}",
            f"- Best val IoU: {best.val_iou:.4f}",
        ]
    (run_dir / "PROGRESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_preview_image(image: np.ndarray, mask: np.ndarray, pred: np.ndarray, labels: list[str]) -> "Image.Image":
    def gray_panel(arr: np.ndarray) -> "Image.Image":
        arr = np.clip(arr, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8), mode="L").convert("RGB")

    def overlay_panel(base: np.ndarray, overlay: np.ndarray, color: tuple[int, int, int]) -> "Image.Image":
        img = gray_panel(base)
        pix = np.asarray(img).copy().astype(np.float32)
        idx = overlay.astype(bool)
        pix[idx] = 0.45 * pix[idx] + 0.55 * np.array(color, dtype=np.float32)
        return Image.fromarray(np.clip(pix, 0, 255).astype(np.uint8))

    panels = [gray_panel(image[i]) for i in range(min(3, image.shape[0]))]
    base = image[0]
    panels.append(overlay_panel(base, mask[0], (255, 80, 80)))
    panels.append(overlay_panel(base, pred[0], (80, 255, 120)))
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * len(panels), height), "black")
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i * width, 0))
    return canvas


def save_previews(model, val_rows: list[dict[str, str]], dataset_dir: Path, run_dir: Path, device, epoch: int, num_samples: int, logger: Logger) -> None:
    previews_dir = ensure_dir(run_dir / "previews")
    for row in val_rows[:num_samples]:
        image, mask = load_tile_h5(dataset_dir / row["tile_path"])
        tensor = torch.from_numpy(image[None]).to(device)
        with torch.no_grad():
            logits = model(tensor)
            pred = (torch.sigmoid(logits)[0] > 0.5).float().cpu().numpy()
        preview = make_preview_image(image, mask, pred, [])
        preview_path = previews_dir / f"epoch_{epoch:03d}_{row['sample_id']}.png"
        preview.save(preview_path)
    logger.debug(f"saved_previews epoch={epoch} count={min(num_samples, len(val_rows))}")


def maybe_load_checkpoint(model, optimizer, scheduler, checkpoint_path: Path, metrics_path: Path, resume_mode: str, logger: Logger):
    start_epoch = 1
    best_metrics = None
    if metrics_path.exists():
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        if rows:
            best_row = max(rows, key=lambda r: r["val_dice"])
            best_metrics = EpochMetrics(**best_row)
    exists = checkpoint_path.exists()
    if resume_mode == "never":
        return start_epoch, best_metrics
    if resume_mode == "always" and not exists:
        raise FileNotFoundError(f"Requested resume=always but checkpoint not found: {checkpoint_path}")
    if resume_mode in {"auto", "always"} and exists:
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        last_metrics = state.get("metrics")
        if last_metrics:
            start_epoch = int(last_metrics["epoch"]) + 1
            logger.log(f"Resuming from checkpoint at epoch {last_metrics['epoch']}")
    return start_epoch, best_metrics


def train_model(args: argparse.Namespace, dataset_dir: Path, run_dir: Path, logger: Logger) -> None:
    manifest_path = dataset_dir / "manifest.csv"
    train_dataset = H5TileDataset(manifest_path, split="train", augment=True, seed=args.seed)
    val_dataset = H5TileDataset(manifest_path, split="val", augment=False, seed=args.seed)
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        val_rows = [row for row in csv.DictReader(handle) if row["split"] == "val"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Device: {device}")
    logger.log(f"CPU threads: {args.cpu_threads}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    in_channels = 3
    model = AttentionUNet(in_channels=in_channels, out_channels=1, base_channels=args.base_channels, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
    criterion = BCEDiceLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    metrics_path = run_dir / "metrics.jsonl"
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    start_epoch, best_metrics = maybe_load_checkpoint(model, optimizer, scheduler, last_path, metrics_path, args.resume, logger)

    for epoch in range(start_epoch, args.epochs + 1):
        start = perf_counter()
        model.train(True)
        optimizer.zero_grad(set_to_none=True)
        train_loss_total = 0.0
        train_batches = 0
        for batch_index, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(device)
            masks = masks.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss / args.grad_accumulation_steps).backward()
            if (batch_index % args.grad_accumulation_steps == 0) or (batch_index == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_loss_total += float(loss.item())
            train_batches += 1

        model.train(False)
        val_loss_total = 0.0
        val_dice_total = 0.0
        val_iou_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                    logits = model(images)
                    loss = criterion(logits, masks)
                dice, iou = segmentation_metrics(logits, masks)
                val_loss_total += float(loss.item())
                val_dice_total += dice
                val_iou_total += iou
                val_batches += 1

        scheduler.step()
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss_total / max(train_batches, 1),
            val_loss=val_loss_total / max(val_batches, 1),
            val_dice=val_dice_total / max(val_batches, 1),
            val_iou=val_iou_total / max(val_batches, 1),
            learning_rate=optimizer.param_groups[0]["lr"],
            seconds=perf_counter() - start,
        )
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(metrics)) + "\n")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics": asdict(metrics),
                "channels": args.channels if args.source_mode == "arpil_sdo" else FALLBACK_CHANNELS,
            },
            last_path,
        )
        if best_metrics is None or metrics.val_dice > best_metrics.val_dice:
            best_metrics = metrics
            shutil.copy2(last_path, best_path)

        logger.log(
            f"epoch={epoch:03d} train_loss={metrics.train_loss:.4f} val_loss={metrics.val_loss:.4f} "
            f"val_dice={metrics.val_dice:.4f} val_iou={metrics.val_iou:.4f} lr={metrics.learning_rate:.6f}"
        )
        logger.debug(f"epoch_seconds={metrics.seconds:.2f}")
        update_progress_markdown(run_dir, args.epochs, metrics, best_metrics)
        if args.preview_every > 0 and (epoch % args.preview_every == 0 or epoch == args.epochs):
            save_previews(model, val_rows, dataset_dir, run_dir, device, epoch, args.num_preview_samples, logger)


def audit_dataset(dataset_dir: Path) -> dict[str, int]:
    manifest_path = dataset_dir / "manifest.csv"
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Duplicate sample IDs detected in manifest.")
    train_count = sum(row["split"] == "train" for row in rows)
    val_count = sum(row["split"] == "val" for row in rows)
    synth_count = sum(row["source_type"] == "synthetic" and row["split"] == "train" for row in rows)
    return {"total": len(rows), "train": train_count, "val": val_count, "synthetic_train": synth_count}


def main() -> None:
    args = build_parser().parse_args()
    if not math.isclose(args.real_ratio + args.synthetic_ratio, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError("real-ratio and synthetic-ratio must sum to 1.0")

    if args.no_auto_install_deps and any(importlib.util.find_spec(m) is None for m in REQUIRED_IMPORTS):
        raise RuntimeError("Required dependencies are missing and --no-auto-install-deps was requested.")
    seed_everything(args.seed)
    set_cpu_runtime(args.cpu_threads)

    work_dir = ensure_dir(Path(args.work_dir))
    dataset_dir = ensure_dir(work_dir / "dataset")
    run_dir = ensure_dir(work_dir / "run")
    logger = Logger(run_dir, verbose=args.verbose)

    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    logger.log("Starting solar ARPIL plugin pipeline...")
    logger.log(f"Mode: {args.source_mode}")
    logger.log(f"Work directory: {work_dir}")

    try:
        if args.source_mode == "arpil_sdo":
            build_arpil_sdo_dataset(args, dataset_dir, logger)
        else:
            build_sharp_sample_dataset(args, dataset_dir, logger)

        audit = audit_dataset(dataset_dir)
        logger.log(
            f"Dataset audit: total={audit['total']} train={audit['train']} val={audit['val']} synthetic_train={audit['synthetic_train']}"
        )
        train_model(args, dataset_dir, run_dir, logger)
        logger.log("Training finished.")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
