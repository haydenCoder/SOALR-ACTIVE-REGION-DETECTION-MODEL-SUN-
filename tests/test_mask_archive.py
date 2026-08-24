"""Tests for ARPIL mask retrieval from the packaged data.tar.gz archive.

Upstream (nasa-ibm-ai4science/surya-bench-ar-segmentation) publishes the CSV
splits plus a single ``data.tar.gz``; the per-frame ``data/<year>/<month>/
<stamp>.h5`` paths named in the CSV "file_path" column are NOT individually
downloadable. These tests pin the archive-based access path so a regression back
to per-file HTTP fetches (which 404) is caught.
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("SOLAR_PLUGIN_NO_AUTO_INSTALL", "1")
plugin = pytest.importorskip("solar_arpil_plugin")


def _build_archive(path: Path, names: list[str]) -> None:
    """Write a tar.gz mirroring the upstream layout."""
    with tarfile.open(path, "w:gz") as tar:
        for name in names:
            payload = f"payload::{Path(name).name}".encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


@pytest.fixture()
def archive(tmp_path: Path) -> Path:
    target = tmp_path / "data.tar.gz"
    _build_archive(
        target,
        [
            "data/2011/01/20110115_0000.h5",
            "data/2011/01/20110115_0100.h5",
            "data/2012/03/20120301_1200.h5",
            "data/README.txt",
        ],
    )
    return target


def test_archive_url_points_at_the_published_tarball():
    assert plugin.MASK_ARCHIVE_URL.startswith(plugin.MASK_BASE_URL)
    assert "data.tar.gz" in plugin.MASK_ARCHIVE_URL


def test_index_finds_only_h5_members(archive: Path, tmp_path: Path):
    ma = plugin.MaskArchive(archive, tmp_path / "out")
    names = ma.available_basenames()
    assert names == {
        "20110115_0000.h5",
        "20110115_0100.h5",
        "20120301_1200.h5",
    }
    assert not any(n.endswith(".txt") for n in names)


def test_extract_returns_readable_file_for_csv_style_path(archive: Path, tmp_path: Path):
    ma = plugin.MaskArchive(archive, tmp_path / "out")
    # Exactly the value the CSV "file_path" column carries.
    got = ma.extract("data/2011/01/20110115_0000.h5")
    assert got.exists() and got.stat().st_size > 0
    assert got.read_bytes() == b"payload::20110115_0000.h5"


def test_extract_is_idempotent_and_cached(archive: Path, tmp_path: Path):
    ma = plugin.MaskArchive(archive, tmp_path / "out")
    first = ma.extract("data/2012/03/20120301_1200.h5")
    mtime = first.stat().st_mtime_ns
    second = ma.extract("data/2012/03/20120301_1200.h5")
    assert first == second
    # Second call must reuse the cached file rather than re-extracting.
    assert second.stat().st_mtime_ns == mtime


def test_extract_leaves_no_part_files(archive: Path, tmp_path: Path):
    out = tmp_path / "out"
    ma = plugin.MaskArchive(archive, out)
    ma.extract("data/2011/01/20110115_0100.h5")
    assert list(out.glob("*.part")) == []


def test_missing_member_raises_actionable_error(archive: Path, tmp_path: Path):
    ma = plugin.MaskArchive(archive, tmp_path / "out")
    with pytest.raises(RuntimeError, match="not present"):
        ma.extract("data/1999/01/19990101_0000.h5")


def test_download_skipped_when_archive_already_present(archive: Path, tmp_path: Path, monkeypatch):
    called = {"n": 0}

    def _fail(*_args, **_kwargs):
        called["n"] += 1
        raise AssertionError("download_binary must not run for a cached archive")

    monkeypatch.setattr(plugin, "download_binary", _fail)
    ma = plugin.MaskArchive(archive, tmp_path / "out")
    ma.ensure_downloaded()
    assert called["n"] == 0


def test_download_invoked_once_when_archive_absent(tmp_path: Path, monkeypatch):
    target = tmp_path / "missing.tar.gz"
    calls: list[str] = []

    def _fake(url: str, destination: Path, **_kwargs):
        calls.append(url)
        _build_archive(destination, ["data/2011/01/20110115_0000.h5"])

    monkeypatch.setattr(plugin, "download_binary", _fake)
    ma = plugin.MaskArchive(target, tmp_path / "out")
    ma.ensure_downloaded()
    assert calls == [plugin.MASK_ARCHIVE_URL]
    # A second call sees the now-present file and must not re-download.
    ma.ensure_downloaded()
    assert len(calls) == 1
