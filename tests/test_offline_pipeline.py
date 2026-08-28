"""Offline end-to-end audit: the FULL pipeline with the network stubbed out.

This is the "does everything actually work" test. It builds synthetic SDO core
frames (netCDF with all 13 channels) and ARPIL masks (h5), fakes the three
network touchpoints (mask index CSV, mask archive, S3 core frames), then runs
the REAL tile builder main() and the REAL streaming trainer on the produced
manifest — including the resume path. No network, no GPU required.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pytest
from netCDF4 import Dataset as NetCDFDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

os.environ.setdefault("SOLAR_PLUGIN_NO_AUTO_INSTALL", "1")

CHANNELS = [
    "aia94", "aia131", "aia1600", "aia171", "aia193", "aia211",
    "aia304", "aia335", "hmi_m", "hmi_bx", "hmi_by", "hmi_bz", "hmi_v",
]
STAMPS = ["20100513_0100", "20100513_0200", "20100513_0300", "20100513_0400"]
FRAME_SIZE = 640  # -> 2x2 tiles of 512 with stride 512 (overlapping by 128 px)


def _make_image_and_mask(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Smooth gradient (compresses well) + two bright 'active region' blobs."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:FRAME_SIZE, 0:FRAME_SIZE]
    noise = rng.normal(0, 2.0, (FRAME_SIZE, FRAME_SIZE)).astype(np.float32)
    image = (y * 0.5 + x * 0.25).astype(np.float32) + noise
    mask = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
    for (y0, x0, y1, x1) in [(50, 50, 250, 250), (450, 450, 600, 600)]:
        image[y0:y1, x0:x1] += 800.0
        mask[y0:y1, x0:x1] = 1.0
    return image, mask


def _write_core_nc(path: Path, seed: int) -> None:
    """One synthetic SDO core frame: every channel variable, same shape."""
    image, _ = _make_image_and_mask(seed)
    with NetCDFDataset(path, "w") as ds:
        ds.createDimension("y", FRAME_SIZE)
        ds.createDimension("x", FRAME_SIZE)
        for channel in CHANNELS:
            variable = ds.createVariable(channel, "f4", ("y", "x"))
            # Give each channel its own scale so the tiles are distinguishable.
            variable[:] = image * (1.0 + 0.1 * CHANNELS.index(channel))


def _write_mask_h5(path: Path, seed: int) -> None:
    _, mask = _make_image_and_mask(seed)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("union_with_intersect", data=mask)


def _builder_argv(output_dir: Path) -> list[str]:
    return [
        "build_arpil_resumable.py",
        "--output-dir", str(output_dir),
        "--split", "train",
        "--channels", *CHANNELS,
        "--patch-size", "512",
        "--stride", "512",
        "--min-mask-fraction", "0.0005",
        "--keep-empty-every", "64",
        "--max-frames", "4",
        "--download-workers", "2",
        "--min-free-disk-gb", "0",
        "--seed", "42",
    ]


@pytest.fixture()
def offline_env(tmp_path: Path):
    """Synthetic S3 frames + a real mask archive tar.gz + the CSV, stubbed in."""
    import build_arpil_resumable as builder
    import build_arpil_3ch_tiles as upstream

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    mask_src = tmp_path / "masks"
    mask_src.mkdir()
    for index, stamp in enumerate(STAMPS):
        _write_core_nc(frames_dir / f"{stamp}.nc", seed=index)
        _write_mask_h5(mask_src / f"{stamp}.h5", seed=index)

    # The mask archive: a real tar.gz with the upstream-style member layout.
    archive = tmp_path / "data.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for stamp in STAMPS:
            tar.add(mask_src / f"{stamp}.h5", arcname=f"data/2010/05/{stamp}.h5")

    rows = [{"file_path": f"data/2010/05/{stamp}.h5", "present": "1"} for stamp in STAMPS]

    def fake_read_csv_rows(url: str) -> list[dict[str, str]]:
        return [dict(row) for row in rows]

    def fake_download_core_s3(s3_client, key: str, destination: Path, attempts: int = 3) -> None:
        source = frames_dir / f"{Path(key).name}"
        assert source.exists(), f"unexpected S3 key {key}"
        destination.write_bytes(source.read_bytes())

    builder.read_csv_rows = fake_read_csv_rows
    builder.download_core_s3 = fake_download_core_s3
    try:
        yield {"builder": builder, "archive": archive, "frames_dir": frames_dir}
    finally:
        # restore: module-level patching must not leak into other tests
        builder.read_csv_rows = upstream.read_csv_rows
        builder.download_core_s3 = upstream.download_core_s3


def _run_builder(builder, output_dir: Path, archive: Path) -> Path:
    """Pre-seed the mask archive cache, run the REAL builder main(), return manifest."""
    cache_dir = output_dir / "download_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "data.tar.gz").write_bytes(archive.read_bytes())
    old_argv = sys.argv
    sys.argv = _builder_argv(output_dir)
    try:
        builder.main()
    finally:
        sys.argv = old_argv
    manifest = output_dir / "manifest.csv"
    assert manifest.exists(), "builder did not publish manifest.csv"
    return manifest


def test_full_offline_pipeline(offline_env, tmp_path, capsys):
    """Builder main() end-to-end on synthetic data: tiles + manifest are valid."""
    builder = offline_env["builder"]
    output_dir = tmp_path / "out"
    with capsys.disabled():
        manifest = _run_builder(builder, output_dir, offline_env["archive"])

    with manifest.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "manifest is empty"
    # Every tile of every frame carries all 13 channels + a mask, all on disk.
    for row in rows:
        for channel in CHANNELS:
            assert (output_dir / row[f"image_{channel}"]).exists(), f"missing image_{channel}"
        assert (output_dir / row["mask"]).exists()
        assert row["split"] in {"train", "val"}
    # Frame-level val split: exactly one frame went to val (15% of 4 -> 1),
    # and no frame leaks between train and val.
    val_frames = {row["sample_id"].rsplit("_y", 1)[0] for row in rows if row["split"] == "val"}
    train_frames = {row["sample_id"].rsplit("_y", 1)[0] for row in rows if row["split"] == "train"}
    assert val_frames and train_frames and not (val_frames & train_frames)
    # Tiles are loadable, right shape, and at least one carries mask signal.
    tile = np.load(output_dir / rows[0]["image_aia171"])["arr_0"]
    assert tile.shape == (512, 512)
    assert any(np.load(output_dir / r["mask"])["arr_0"].sum() > 0 for r in rows)


def test_streaming_trainer_end_to_end(offline_env, tmp_path):
    """Train 2 epochs on the offline manifest, then prove resume works."""
    builder = offline_env["builder"]
    out_root = tmp_path / "pipeline"
    train_dir = out_root / "train"
    manifest = _run_builder(builder, out_root, offline_env["archive"])

    script = REPO_ROOT / "scripts" / "train_streaming.py"
    base = [
        sys.executable, str(script),
        "--manifest", str(manifest),
        "--channels", *CHANNELS,
        "--image-size", "512",
        "--base-channels", "8",
        "--max-epochs", "2",
        "--val-every", "1",
        "--val-subset", "4",
        "--max-tiles-per-epoch", "8",
        "--tta", "none",
        "--no-torch-compile",
        "--cpu-headroom", "0",
        "--memory-budget-gb", "0",
        "--output-dir", str(train_dir),
    ]
    first = subprocess.run(base, capture_output=True, text=True, cwd=REPO_ROOT, timeout=900)
    assert first.returncode == 0, f"trainer failed:\n{first.stdout[-3000:]}\n{first.stderr[-3000:]}"
    assert "batch_size=" in first.stdout
    assert (train_dir / "last.pt").exists(), "no checkpoint saved"
    assert (train_dir / "best.pt").exists(), "no best checkpoint saved"
    metrics = (train_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(metrics) >= 2, "expected one metrics line per epoch"

    # Resume: 2 more epochs from the saved checkpoint.
    resume_argv = list(base)
    i = resume_argv.index("--max-epochs")
    resume_argv[i + 1] = "4"
    resumed = subprocess.run(resume_argv, capture_output=True, text=True, cwd=REPO_ROOT, timeout=900)
    assert resumed.returncode == 0, f"resume run failed:\n{resumed.stdout[-3000:]}\n{resumed.stderr[-3000:]}"
    assert "resumed from" in resumed.stdout, f"trainer did not resume:\n{resumed.stdout[-2000:]}"
