"""Regression tests for the plugin's download and index-alignment logic."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("SOLAR_PLUGIN_NO_AUTO_INSTALL", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

plugin = pytest.importorskip("solar_arpil_plugin")


# --------------------------------------------------------------------------
# `present` column handling.
# --------------------------------------------------------------------------


def test_rows_without_present_column_are_kept() -> None:
    """A missing column used to be read as "absent" and discarded every row."""
    assert plugin.is_present({"file_path": "data/2011/01/20110101_0000.h5"}) is True


@pytest.mark.parametrize("value", ["0", "0.0", "", "false", "No", "nan", "none"])
def test_absent_markers_are_filtered(value: str) -> None:
    assert plugin.is_present({"present": value}) is False


@pytest.mark.parametrize("value", ["1", "1.0", "true", "yes", "T"])
def test_present_markers_are_kept(value: str) -> None:
    assert plugin.is_present({"present": value}) is True


# --------------------------------------------------------------------------
# Timestamp alignment between the two index CSVs.
# --------------------------------------------------------------------------


def test_timestamp_key_extraction() -> None:
    assert plugin.timestamp_key("data/2011/01/20110101_0000.h5") == "20110101_0000"
    assert plugin.timestamp_key("2011/01/20110101_0000.nc") == "20110101_0000"
    assert plugin.timestamp_key("20110101T000000.h5") == "20110101_000000"


def test_align_matches_h5_masks_to_nc_frames() -> None:
    masks = [{"file_path": "data/2011/01/20110101_0000.h5", "present": "1"}]
    cores = [{"path": "2011/01/20110101_0000.nc"}]
    aligned = plugin.align_arpil_and_core(masks, cores)
    assert len(aligned) == 1
    assert aligned[0]["timestamp"] == "20110101_0000"


def test_align_raises_actionable_error_when_nothing_matches() -> None:
    masks = [{"file_path": "data/2011/01/20110101_0000.h5"}]
    cores = [{"path": "2019/05/20190505_1200.nc"}]
    with pytest.raises(RuntimeError, match="No timestamps are common"):
        plugin.align_arpil_and_core(masks, cores)


# --------------------------------------------------------------------------
# Git LFS pointers and CSV validation.
# --------------------------------------------------------------------------


LFS_POINTER = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123\nsize 4096\n"


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None


def test_lfs_pointer_is_retried_via_resolve_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_get(url: str, **kwargs: object) -> _FakeResponse:
        seen.append(url)
        if "/raw/" in url:
            return _FakeResponse(LFS_POINTER)
        return _FakeResponse("file_path,present\ndata/x.h5,1\n")

    monkeypatch.setattr(plugin.requests, "get", fake_get)
    text = plugin.download_text("https://example.org/datasets/d/raw/main/train.csv")

    assert "file_path" in text
    assert any("/resolve/" in url for url in seen)


def test_empty_csv_raises_rather_than_returning_nothing() -> None:
    with pytest.raises(RuntimeError, match="zero rows"):
        plugin.read_csv_from_text("file_path,present\n")


def test_read_csv_parses_rows() -> None:
    rows = plugin.read_csv_from_text("file_path,present\ndata/x.h5,1\n")
    assert rows == [{"file_path": "data/x.h5", "present": "1"}]


# --------------------------------------------------------------------------
# Retry behaviour.
# --------------------------------------------------------------------------


def test_transient_failures_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def flaky_get(url: str, **kwargs: object) -> _FakeResponse:
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("connection reset")
        return _FakeResponse("ok")

    monkeypatch.setattr(plugin.requests, "get", flaky_get)
    monkeypatch.setattr(plugin.time, "sleep", lambda _seconds: None)

    assert plugin._request_with_retries("https://example.org/x", timeout=5).text == "ok"
    assert calls["n"] == 3


def test_persistent_failure_raises_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_fail(url: str, **kwargs: object) -> _FakeResponse:
        raise OSError("network unreachable")

    monkeypatch.setattr(plugin.requests, "get", always_fail)
    monkeypatch.setattr(plugin.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="after .* attempts"):
        plugin._request_with_retries("https://example.org/x", timeout=5)


# --------------------------------------------------------------------------
# Mask reading through the shared loader.
# --------------------------------------------------------------------------


def test_read_arpil_mask_handles_nested_groups(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    numpy = pytest.importorskip("numpy")

    path = tmp_path / "mask.h5"
    with h5py.File(path, "w") as handle:
        data = numpy.zeros((1, 8, 8), numpy.uint8)
        data[0, 2:5, 2:5] = 1
        handle.create_group("masks").create_dataset("union_with_intersect", data=data)

    mask = plugin.read_arpil_mask(path, "union_with_intersect")
    assert mask.shape == (8, 8)
    assert mask.sum() == 9.0


# --------------------------------------------------------------------------
# Dependency bootstrap messaging.
# --------------------------------------------------------------------------


def test_bootstrap_without_auto_install_names_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plugin, "missing_dependencies", lambda: ["h5py", "netCDF4"])
    with pytest.raises(RuntimeError) as excinfo:
        plugin.bootstrap_dependencies(Path("requirements.txt"), auto_install=False)

    message = str(excinfo.value)
    assert "h5py" in message and "netCDF4" in message
    assert "pip install" in message
