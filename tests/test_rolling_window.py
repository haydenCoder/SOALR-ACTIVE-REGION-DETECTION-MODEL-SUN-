"""Rolling-window rotation + missing-tile loader fallback (offline tests)."""

from __future__ import annotations

import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_arpil_resumable as builder  # noqa: E402

CHANNELS = ["aia171", "hmi_m"]
FIELDNAMES = ["sample_id", "split", "image_aia171", "image_hmi_m", "mask"]
TILES_PER_FRAME = 2


def _write_frame(root: Path, stamp: str, age_seconds: float) -> None:
    """Create one frame's fragment record + real tile files, with an aged mtime."""
    frag_dir = root / "frame_manifests"
    frag_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(TILES_PER_FRAME):
        row = {"sample_id": f"{stamp}_y{i:04d}_x{i:04d}", "split": "train"}
        for channel in CHANNELS:
            p = root / "images" / channel / f"{stamp}_t{i}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, np.zeros((8, 8), dtype=np.float16))
            row[f"image_{channel}"] = os.path.relpath(p, root)
        mp = root / "masks" / f"{stamp}_t{i}.npz"
        mp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(mp, np.zeros((8, 8), dtype=np.float16))
        row["mask"] = os.path.relpath(mp, root)
        rows.append(row)
    fp = frag_dir / f"{stamp}.csv"
    with fp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    ts = time.time() - age_seconds
    os.utime(fp, (ts, ts))


def _expected_val_frames(fragments: dict, val_ratio: float, seed: int) -> set:
    # Mirrors write_combined_manifest's frame-level val assignment.
    frame_ids = sorted(f for f, rows in fragments.items() if rows)
    shuffled = frame_ids[:]
    random.Random(seed).shuffle(shuffled)
    return set(shuffled[: max(1, int(len(shuffled) * val_ratio))]) if len(shuffled) > 1 else set()


@pytest.fixture()
def rolling_env(tmp_path: Path):
    root = tmp_path / "data"
    stamps = [f"20100513_{i:04d}" for i in range(10)]
    for i, stamp in enumerate(stamps):
        _write_frame(root, stamp, age_seconds=(10 - i) * 100)  # stamp 0 oldest
    fragments = builder.load_fragment_rows(root / "frame_manifests")
    builder.write_combined_manifest(root, fragments, FIELDNAMES, val_ratio=0.2, seed=42)
    return {"root": root, "stamps": stamps, "fragments": fragments}


def test_rotation_retires_oldest_and_protects_val(rolling_env):
    root, stamps, fragments = rolling_env["root"], rolling_env["stamps"], rolling_env["fragments"]
    val_frames = _expected_val_frames(fragments, val_ratio=0.2, seed=42)
    assert len(val_frames) == 2

    builder.rotate_window(root, fragments, FIELDNAMES, val_ratio=0.2, seed=42,
                          window=6, min_lifetime_hours=0, grace_hours=0, max_tile_gb=0)

    retired_dir = root / "frame_manifests_retired"
    retired = {p.stem for p in retired_dir.glob("*.csv")}
    live = {p.stem for p in (root / "frame_manifests").glob("*.csv")}
    # 10 - 6 = 4 retired, except val frames that sit in the oldest 4 stay live.
    assert len(live) + len(retired) == 10
    assert len(retired) >= 3
    # Validation frames must survive rotation (they are the scoreboard).
    assert not (retired & val_frames)
    # Manifest was rewritten: only live frames remain, all val frames present.
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest_frames = {r["sample_id"].rsplit("_y", 1)[0] for r in csv.DictReader(handle)}
    assert manifest_frames == live
    assert val_frames <= manifest_frames
    # grace_hours=0 -> retired tiles are already freed, live tiles intact.
    for stamp in retired:
        assert not (root / "images" / "aia171" / f"{stamp}_t0.npz").exists()
    for stamp in live:
        assert (root / "images" / "aia171" / f"{stamp}_t0.npz").exists()


def test_retired_names_are_permanent_record(rolling_env):
    """Retired frames must count as 'completed' so they are never re-downloaded."""
    root, stamps, fragments = rolling_env["root"], rolling_env["stamps"], rolling_env["fragments"]
    builder.rotate_window(root, fragments, FIELDNAMES, val_ratio=0.2, seed=42,
                          window=6, min_lifetime_hours=0, grace_hours=0, max_tile_gb=0)
    retired_dir = root / "frame_manifests_retired"
    completed = set(fragments) | set(builder.load_fragment_rows(retired_dir))
    assert completed == set(stamps)  # every frame name is recorded, live or retired


def test_grace_gap_keeps_retired_tiles_for_training(rolling_env):
    root, _, fragments = rolling_env["root"], None, rolling_env["fragments"]
    builder.rotate_window(root, fragments, FIELDNAMES, val_ratio=0.2, seed=42,
                          window=6, min_lifetime_hours=0, grace_hours=1.0, max_tile_gb=0)
    retired = list((root / "frame_manifests_retired").glob("*.csv"))
    assert retired, "expected retired frames"
    for fp in retired:
        # Just retired -> within the grace gap -> tiles must still be on disk.
        assert (root / "images" / "aia171" / f"{fp.stem}_t0.npz").exists()


def test_min_lifetime_protects_young_frames_when_budgets_hold(rolling_env):
    """With no budget violated, young frames must train for min_lifetime_hours first."""
    root, stamps, fragments = rolling_env["root"], rolling_env["stamps"], rolling_env["fragments"]
    # No budget pressure (window == frame count, no byte budget); require 6 h
    # -> nothing may be retired, all frames are < 1 h old.
    builder.rotate_window(root, fragments, FIELDNAMES, val_ratio=0.2, seed=42,
                          window=10, min_lifetime_hours=6.0, grace_hours=0, max_tile_gb=0)
    assert not list((root / "frame_manifests_retired").glob("*.csv"))
    assert len(list((root / "frame_manifests").glob("*.csv"))) == 10


def test_byte_budget_forces_retirement_of_young_frames(tmp_path: Path):
    """The hard byte budget is the real disk guarantee: it overrides the
    minimum lifetime (the disk is the hard wall) and retires in measured bytes."""
    root = tmp_path / "data"
    stamps = [f"20100513_{i:04d}" for i in range(10)]
    for i, stamp in enumerate(stamps):
        _write_frame(root, stamp, age_seconds=(10 - i) * 60)  # all young (< 1 min)
    fragments = builder.load_fragment_rows(root / "frame_manifests")
    builder.write_combined_manifest(root, fragments, FIELDNAMES, val_ratio=0.2, seed=42)

    total_tiles = sum(
        f.stat().st_size
        for d in (root / "images", root / "masks") for f in d.rglob("*.npz")
    )
    assert total_tiles > 0
    budget_gb = total_tiles / (1024 ** 3) * 0.4  # budget = 40% of current tiles

    builder.rotate_window(root, fragments, FIELDNAMES, val_ratio=0.2, seed=42,
                          window=100,  # no count pressure
                          min_lifetime_hours=100.0,  # nobody is lifetime-eligible
                          grace_hours=0, max_tile_gb=budget_gb)

    retired = {p.stem for p in (root / "frame_manifests_retired").glob("*.csv")}
    assert retired, "byte budget must force retirement even for young frames"
    # Oldest-first, val frames protected.
    val_frames = _expected_val_frames({s: [{}] for s in stamps}, val_ratio=0.2, seed=42)
    assert not (retired & val_frames)
    # Measured bytes of the LIVE frames are now under the budget.
    remaining = 0
    live_dir = root / "frame_manifests"
    for fp in live_dir.glob("*.csv"):
        with fp.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for key, value in row.items():
                    if key == "mask" or key.startswith("image_"):
                        remaining += (root / value).stat().st_size
    assert remaining <= budget_gb * (1024 ** 3)


def _build_two_frame_dataset(root: Path) -> tuple[Path, list[str]]:
    """Two frames x 2 tiles x 1 channel, real npz files + manifest."""
    frag_dir = root / "frame_manifests"
    frag_dir.mkdir(parents=True, exist_ok=True)
    stamps = ["20100513_0000", "20100513_0001"]
    rows_all = []
    for stamp in stamps:
        for i in range(2):
            row = {"sample_id": f"{stamp}_y{i:04d}_x{i:04d}", "split": "train"}
            p = root / "images" / "aia171" / f"{stamp}_t{i}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, np.random.default_rng(i).random((16, 16)).astype(np.float16))
            row["image_aia171"] = os.path.relpath(p, root)
            mp = root / "masks" / f"{stamp}_t{i}.npz"
            mp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mp, np.ones((16, 16), dtype=np.float16))
            row["mask"] = os.path.relpath(mp, root)
            rows_all.append(row)
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "image_aia171", "mask"])
        writer.writeheader()
        writer.writerows(rows_all)
    return manifest, stamps


def test_loader_survives_rotated_out_tile(tmp_path: Path):
    """A missing tile (just rotated out) must yield a background sample, not a crash."""
    from solar_ar.data import SolarActiveRegionDataset

    root = tmp_path / "data"
    manifest, stamps = _build_two_frame_dataset(root)
    ds = SolarActiveRegionDataset(manifest, ["aia171"], "train", image_size=32)
    assert len(ds.records) == 4
    # Remove one tile file mid-dataset (simulates a rotation race).
    victim = root / "images" / "aia171" / f"{stamps[1]}_t1.npz"
    victim.unlink()
    img, mask = ds[3]  # the record that referenced the victim tile
    assert img.shape == (1, 32, 32) and mask.shape == (1, 32, 32)
    assert torch_finite(img) and torch_finite(mask)
    # ...while the healthy record still loads real data.
    img_ok, _ = ds[0]
    assert img_ok.shape == (1, 32, 32)


def torch_finite(t) -> bool:
    import torch
    return bool(torch.isfinite(t).all().item())
