"""Regression tests for the multi-format array loader and dataset.

Run with:  python3 -m pytest tests/ -q
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_ar.arrayio import (  # noqa: E402
    ArrayLoadError,
    file_extension,
    hdf5_dataset_names,
    load_array,
    read_hdf5_tile,
    split_path_spec,
    squeeze_to_2d,
    write_hdf5_tile,
)
from solar_ar.data import SolarActiveRegionDataset  # noqa: E402

h5py = pytest.importorskip("h5py")


# --------------------------------------------------------------------------
# HDF5 support: the format the loader previously rejected outright.
# --------------------------------------------------------------------------


def test_loads_flat_hdf5(tmp_path: Path) -> None:
    path = tmp_path / "mask.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("union_with_intersect", data=np.ones((8, 8), np.uint8))

    array = load_array(path)
    assert array.shape == (8, 8)
    assert array.dtype == np.float32


def test_loads_hdf5_nested_in_group(tmp_path: Path) -> None:
    """A mask nested in a group used to raise 'could not convert string to float'."""
    path = tmp_path / "nested.h5"
    with h5py.File(path, "w") as handle:
        handle.create_group("masks").create_dataset("union_with_intersect", data=np.ones((8, 8)))

    assert load_array(path, key="union_with_intersect").shape == (8, 8)
    assert load_array(path).shape == (8, 8)  # auto-detected
    assert hdf5_dataset_names(path) == ["masks/union_with_intersect"]


def test_hdf5_inline_key_spec(tmp_path: Path) -> None:
    """`file.h5#dataset` selects a dataset straight from a manifest cell."""
    path = tmp_path / "multi.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("a", data=np.zeros((4, 4)))
        handle.create_dataset("b", data=np.ones((4, 4)))

    assert load_array(f"{path}#b").mean() == 1.0
    assert load_array(f"{path}#a").mean() == 0.0


def test_unknown_hdf5_key_falls_back(tmp_path: Path) -> None:
    """An unexpected layout warns and degrades instead of crashing the run."""
    path = tmp_path / "odd.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("some_unexpected_name", data=np.ones((5, 6)))

    assert load_array(path, key="union_with_intersect").shape == (5, 6)


def test_empty_hdf5_raises_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w"):
        pass
    with pytest.raises(ArrayLoadError, match="No HDF5 datasets"):
        load_array(path)


# --------------------------------------------------------------------------
# Shape handling: the silent (1,H,W) -> (1,H) corruption.
# --------------------------------------------------------------------------


def test_leading_singleton_cube_is_squeezed_not_sliced(tmp_path: Path) -> None:
    path = tmp_path / "cube.npy"
    np.save(path, np.random.rand(1, 16, 16).astype(np.float32))
    # The old implementation did array[..., 0] and produced (1, 16).
    assert load_array(path).shape == (16, 16)


def test_trailing_singleton_channel_is_squeezed() -> None:
    assert squeeze_to_2d(np.zeros((16, 16, 1))).shape == (16, 16)


def test_multi_frame_cube_takes_first_frame() -> None:
    stack = np.stack([np.full((4, 4), i, dtype=np.float32) for i in range(3)])
    reduced = squeeze_to_2d(stack)
    assert reduced.shape == (4, 4)
    assert reduced.mean() == 0.0  # first frame


def test_one_dimensional_input_rejected() -> None:
    with pytest.raises(ArrayLoadError):
        squeeze_to_2d(np.zeros(10))


# --------------------------------------------------------------------------
# Other formats and path handling.
# --------------------------------------------------------------------------


def test_npz_and_npy_round_trip(tmp_path: Path) -> None:
    np.save(tmp_path / "a.npy", np.ones((3, 3), np.float32))
    np.savez(tmp_path / "b.npz", np.ones((3, 3), np.float32))
    assert load_array(tmp_path / "a.npy").shape == (3, 3)
    assert load_array(tmp_path / "b.npz").shape == (3, 3)


def test_compound_extension_detection() -> None:
    assert file_extension(Path("frame.fits.gz")) == ".fits"
    assert file_extension(Path("frame.h5")) == ".h5"
    assert file_extension(Path("noext")) == ""


def test_split_path_spec() -> None:
    path, key = split_path_spec("dir/file.h5#group/name")
    assert path == Path("dir/file.h5") and key == "group/name"
    assert split_path_spec("dir/file.h5") == (Path("dir/file.h5"), None)


def test_unsupported_extension_message(tmp_path: Path) -> None:
    bogus = tmp_path / "data.xyz"
    bogus.write_text("nope")
    with pytest.raises(ArrayLoadError, match="Unsupported file extension"):
        load_array(bogus)


def test_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_array(tmp_path / "absent.h5")


def test_hdf5_tile_round_trip(tmp_path: Path) -> None:
    image = np.random.rand(3, 8, 8).astype(np.float32)
    mask = (np.random.rand(8, 8) > 0.5).astype(np.float32)
    path = write_hdf5_tile(tmp_path / "tile.h5", image, mask, {"sample_id": "x1"})

    loaded_image, loaded_mask = read_hdf5_tile(path)
    assert np.allclose(loaded_image, image)
    assert np.allclose(loaded_mask, mask)
    assert not list(tmp_path.glob("*.tmp"))  # no leftover temp files


# --------------------------------------------------------------------------
# Dataset integration: an end-to-end HDF5 manifest.
# --------------------------------------------------------------------------


def _write_hdf5_manifest(tmp_path: Path, rows: int = 4) -> Path:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "image_aia171", "mask"])
        writer.writeheader()
        for index in range(rows):
            image_path = tmp_path / f"img_{index}.h5"
            mask_path = tmp_path / f"mask_{index}.h5"
            with h5py.File(image_path, "w") as f:
                f.create_dataset("image", data=np.random.rand(1, 32, 32))
            with h5py.File(mask_path, "w") as f:
                f.create_group("masks").create_dataset(
                    "union_with_intersect", data=(np.random.rand(32, 32) > 0.5).astype(np.uint8)
                )
            writer.writerow(
                {
                    "sample_id": f"s{index}",
                    "split": "train" if index < rows - 1 else "val",
                    "image_aia171": image_path.name,
                    "mask": mask_path.name,
                }
            )
    return manifest


def test_dataset_reads_hdf5_manifest(tmp_path: Path) -> None:
    manifest = _write_hdf5_manifest(tmp_path)
    dataset = SolarActiveRegionDataset(
        manifest_path=manifest,
        channels=["aia171"],
        split="train",
        image_size=32,
        hdf5_image_key="image",
        hdf5_mask_key="union_with_intersect",
    )
    assert len(dataset) == 3

    image, mask = dataset[0]
    assert image.shape == (1, 32, 32)
    assert mask.shape == (1, 32, 32)
    assert set(np.unique(mask.numpy())) <= {0.0, 1.0}


def test_dataset_reports_available_splits(tmp_path: Path) -> None:
    manifest = _write_hdf5_manifest(tmp_path)
    with pytest.raises(ValueError, match="Splits present"):
        SolarActiveRegionDataset(
            manifest_path=manifest, channels=["aia171"], split="test", image_size=32
        )


def test_dataset_reports_missing_columns(tmp_path: Path) -> None:
    manifest = _write_hdf5_manifest(tmp_path)
    with pytest.raises(ValueError, match="missing columns"):
        SolarActiveRegionDataset(
            manifest_path=manifest, channels=["not_a_channel"], split="train", image_size=32
        )
