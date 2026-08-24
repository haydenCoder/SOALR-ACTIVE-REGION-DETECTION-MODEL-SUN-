from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from solar_ar.arrayio import SUPPORTED_EXTENSIONS, load_array

# Re-exported for callers that historically imported it from this module.
__all__ = ["SampleRecord", "SolarActiveRegionDataset", "SUPPORTED_EXTENSIONS"]


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
        hdf5_image_key: str | None = None,
        hdf5_mask_key: str | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.channels = list(channels)
        self.split = split
        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.normalize_mode = normalize_mode
        self.augment = augment
        self.random = random.Random(seed)
        # Dataset/variable names used for container formats (HDF5, netCDF, npz).
        # A ``path#key`` suffix in the manifest overrides these per file.
        self.hdf5_image_key = hdf5_image_key
        self.hdf5_mask_key = hdf5_mask_key
        self.records = self._load_records()

        if not self.records:
            raise ValueError(f"No samples found for split='{split}' in {self.manifest_path}.")

    def _resolve(self, base_dir: Path, value: str) -> Path:
        """Resolve a manifest cell to a path, keeping any ``#dataset`` suffix.

        Relative paths are interpreted against the manifest's own directory so
        a dataset folder stays portable; absolute paths are left untouched.
        """
        value = (value or "").strip()
        if not value:
            raise ValueError(f"Empty path cell in manifest {self.manifest_path}")
        file_part, separator, key_part = value.partition("#")
        path = Path(file_part)
        if not path.is_absolute():
            path = base_dir / path
        return Path(f"{path}{separator}{key_part}") if separator else path

    def _load_records(self) -> list[SampleRecord]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        base_dir = self.manifest_path.parent
        records: list[SampleRecord] = []
        seen_splits: set[str] = set()

        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            expected_columns = {"sample_id", "split", "mask", *(f"image_{channel}" for channel in self.channels)}
            missing = expected_columns.difference(fieldnames)
            if missing:
                raise ValueError(
                    f"Manifest {self.manifest_path} is missing columns: {sorted(missing)}. "
                    f"Found columns: {fieldnames}. "
                    f"Check that --channels matches the image_* columns in the manifest."
                )

            for row in reader:
                row_split = (row.get("split") or "").strip()
                seen_splits.add(row_split)
                if row_split.lower() != self.split.lower():
                    continue
                records.append(
                    SampleRecord(
                        sample_id=row["sample_id"],
                        split=row_split,
                        image_paths=[
                            self._resolve(base_dir, row[f"image_{channel}"]) for channel in self.channels
                        ],
                        mask_path=self._resolve(base_dir, row["mask"]),
                    )
                )

        if not records and seen_splits:
            raise ValueError(
                f"No rows with split='{self.split}' in {self.manifest_path}. "
                f"Splits present in the manifest: {sorted(seen_splits)}."
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        channels = [self._load_image(path, self.hdf5_image_key) for path in record.image_paths]
        image = np.stack(channels, axis=0).astype(np.float32)
        mask = self._load_image(record.mask_path, self.hdf5_mask_key).astype(np.float32)

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
    def _load_image(path: Path, key: str | None = None) -> np.ndarray:
        """Load one 2-D plane from any supported container.

        Delegates to :func:`solar_ar.arrayio.load_array`, which handles HDF5,
        netCDF, FITS, npy/npz and ordinary images, and resolves ``path#dataset``
        specs for the formats that hold named datasets.
        """
        return load_array(path, key=key, as_2d=True)
