from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".npz", ".fits", ".fit", ".fts"}

try:
    from astropy.io import fits  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fits = None


@dataclass
class SampleRecord:
    sample_id: str
    split: str
    image_paths: list[Path]
    mask_path: Path


class SolarActiveRegionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        manifest_path: str | Path,
        channels: Sequence[str],
        split: str,
        image_size: int = 256,
        mask_threshold: float = 0.5,
        normalize_mode: str = "percentile",
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.channels = list(channels)
        self.split = split
        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.normalize_mode = normalize_mode
        self.augment = augment
        self.random = random.Random(seed)
        self.records = self._load_records()

        if not self.records:
            raise ValueError(f"No samples found for split='{split}' in {self.manifest_path}.")

    def _load_records(self) -> list[SampleRecord]:
        base_dir = self.manifest_path.parent
        records: list[SampleRecord] = []
        with self.manifest_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            expected_columns = {"sample_id", "split", "mask", *(f"image_{channel}" for channel in self.channels)}
            missing = expected_columns.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Manifest {self.manifest_path} is missing columns: {sorted(missing)}")

            for row in reader:
                if row["split"].lower() != self.split.lower():
                    continue
                image_paths = [base_dir / row[f"image_{channel}"] for channel in self.channels]
                mask_path = base_dir / row["mask"]
                records.append(
                    SampleRecord(
                        sample_id=row["sample_id"],
                        split=row["split"],
                        image_paths=image_paths,
                        mask_path=mask_path,
                    )
                )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        channels = [self._load_image(path) for path in record.image_paths]
        image = np.stack(channels, axis=0).astype(np.float32)
        mask = self._load_image(record.mask_path).astype(np.float32)

        image = self._normalize(image)
        mask = (mask > self.mask_threshold).astype(np.float32)[None, ...]

        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)

        image_tensor = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0)

        if self.augment:
            image_tensor, mask_tensor = self._augment(image_tensor, mask_tensor)

        return image_tensor, mask_tensor

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        normalized = np.empty_like(image, dtype=np.float32)
        for channel_index in range(image.shape[0]):
            channel = image[channel_index]
            channel_name = self.channels[channel_index].lower()
            if self.normalize_mode == "zscore":
                mean = float(channel.mean())
                std = float(channel.std())
                normalized[channel_index] = (channel - mean) / max(std, 1e-6)
            elif self.normalize_mode == "minmax":
                min_val = float(channel.min())
                max_val = float(channel.max())
                normalized[channel_index] = (channel - min_val) / max(max_val - min_val, 1e-6)
            elif self.normalize_mode == "solar_physics":
                normalized[channel_index] = self._normalize_solar_physics(channel, channel_name)
            else:
                lo = float(np.percentile(channel, 1.0))
                hi = float(np.percentile(channel, 99.0))
                clipped = np.clip(channel, lo, hi)
                normalized[channel_index] = (clipped - lo) / max(hi - lo, 1e-6)
        return normalized

    @staticmethod
    def _normalize_solar_physics(channel: np.ndarray, channel_name: str) -> np.ndarray:
        channel = np.nan_to_num(channel.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # EUV AIA channels have a highly skewed photon-count distribution, so log compression is helpful.
        if channel_name.startswith("aia") or channel_name in {"171", "193", "191", "211", "304", "335", "94", "131", "1600", "1700"}:
            channel = np.maximum(channel, 0.0)
            channel = np.log1p(channel)
            lo = float(np.percentile(channel, 1.0))
            hi = float(np.percentile(channel, 99.5))
            channel = np.clip(channel, lo, hi)
            return (channel - lo) / max(hi - lo, 1e-6)

        # HMI magnetograms are signed; use symmetric scaling so polarity information is preserved.
        if "hmi" in channel_name or "mag" in channel_name or channel_name in {"m", "bz", "bx", "by"}:
            scale = float(np.percentile(np.abs(channel), 99.5))
            scale = max(scale, 1e-6)
            return 0.5 * (np.tanh(channel / scale) + 1.0)

        lo = float(np.percentile(channel, 1.0))
        hi = float(np.percentile(channel, 99.0))
        channel = np.clip(channel, lo, hi)
        return (channel - lo) / max(hi - lo, 1e-6)

    def _augment(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
        if self.random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
        rotations = self.random.randint(0, 3)
        if rotations:
            image = torch.rot90(image, rotations, dims=[1, 2])
            mask = torch.rot90(mask, rotations, dims=[1, 2])
        return image.contiguous(), mask.contiguous()

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Expected file does not exist: {path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {path}")

        if suffix == ".npy":
            array = np.load(path)
        elif suffix == ".npz":
            archive = np.load(path)
            if len(archive.files) != 1:
                raise ValueError(f"Expected a single-array npz file, got keys={archive.files} for {path}")
            array = archive[archive.files[0]]
        elif suffix in {".fits", ".fit", ".fts"}:
            if fits is None:
                raise ImportError(
                    "FITS input requires astropy. Install dependencies from requirements.txt or use Docker."
                )
            array = fits.getdata(path)
        else:
            with Image.open(path) as image:
                array = np.asarray(image.convert("F"), dtype=np.float32)

        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 3:
            array = array[..., 0]
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D image for {path}, got shape={array.shape}")
        return array
