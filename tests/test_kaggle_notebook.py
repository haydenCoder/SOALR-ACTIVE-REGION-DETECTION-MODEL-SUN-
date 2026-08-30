r"""Regression tests for the Kaggle launcher notebook.

These guard against the two failures that left the Kaggle run at 100% GPU for
hours with no ``best.pt``:

1. The code cell must be multi-line with a real ``%%bash`` magic on the first
   line. A previous build flattened the whole cell to a single line
   (``%%bashset -xexport ...``), which IPython parses as an unknown magic
   ``%%bashset`` and fails instantly — meaning NOTHING in the cell ever runs.

2. The ``VAR=value ... VAR=value bash scripts/run_forever.sh`` launch prefix
   must be one clean backslash-continued chain with NO comment (``#``) line in
   the middle. A comment line on a ``\``-continued chain makes bash treat the
   rest of that logical line as a comment, silently dropping every env var
   before it (so the trainer falls back to desktop defaults — 1 channel,
   ``MIN_FREE_GB=40`` — and downloads never start on a 20 GB Kaggle disk).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "kaggle" / "start_kaggle.ipynb"

# The env vars the Kaggle rolling preset MUST propagate to run_forever.sh.
REQUIRED_PRESET_VARS = [
    "CHANNELS",
    "SOLAR_PYTHON",
    "BASE_CHANNELS",
    "DEEP_SUPERVISION",
    "ROLLING",
    "ROLLING_WINDOW",
    "ROLLING_MAX_TILE_GB",
    "ROLLING_MIN_LIFETIME_HOURS",
    "TILE_GRACE_HOURS",
    "LR",
    "TORCH_COMPILE",
    "MIN_FREE_GB",
    "MAX_TOTAL_FRAMES",
    "FRAMES_PER_CYCLE",
    "DOWNLOAD_WORKERS",
    "CPU_HEADROOM",
    "TILES_PER_EPOCH",
    "VAL_EPOCH",
    "VAL_SUBSET",
]


def _code_cell_source() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, "notebook has no code cell"
    return "".join(code_cells[0]["source"])


def test_notebook_is_valid_json() -> None:
    json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_bash_magic_is_alone_on_first_line() -> None:
    src = _code_cell_source()
    assert src.count("\n") > 10, "cell must be multi-line (was flattened to one line?)"
    first = src.splitlines()[0]
    assert first.strip() == "%%bash", (
        f"first line must be exactly '%%bash', got {first!r} — a flattened cell "
        "is parsed as the unknown magic '%%bashset' and the whole cell fails to run."
    )


def test_launch_prefix_is_a_clean_continuation_chain() -> None:
    lines = _code_cell_source().splitlines()
    # The prefix starts at the CHANNELS= assignment (column 0) and ends at the
    # launch command.
    start = next(i for i, line in enumerate(lines) if line.startswith('CHANNELS="aia'))
    end = next(i for i, line in enumerate(lines) if line.strip() == "bash scripts/run_forever.sh")
    assert end > start, "could not find the `bash scripts/run_forever.sh` launch line"

    assignments = lines[start:end]
    # Every assignment line ends in a backslash so it belongs to the chain.
    not_continued = [line for line in assignments if not line.rstrip().endswith("\\")]
    assert not not_continued, f"prefix lines missing a trailing backslash: {not_continued}"
    # A comment anywhere inside the chain would comment out the rest of the
    # (already backslash-joined) logical line and silently drop env vars.
    comments = [line for line in assignments if line.lstrip().startswith("#")]
    assert not comments, f"comment line(s) inside the env launch chain drop env vars: {comments}"


@pytest.mark.parametrize("var", REQUIRED_PRESET_VARS)
def test_every_preset_var_is_present(var: str) -> None:
    src = _code_cell_source()
    assert f"{var}=" in src, f"preset env var {var} is not set in the launch chain"


def test_kaggle_disk_reserve_is_small() -> None:
    # The desktop default (40 GB) is fatal on Kaggle's ~20 GB working disk:
    # the downloader sees free < MIN_FREE_GB and pauses forever. The preset
    # must set a small reserve.
    src = _code_cell_source()
    assert "MIN_FREE_GB=4" in src, "Kaggle preset must set MIN_FREE_GB=4"


def test_torch_compile_disabled_on_kaggle() -> None:
    # torch.compile is a silent 15-40 min one-time compile on a 4-core box and
    # re-pays on every restart; the preset disables it.
    src = _code_cell_source()
    assert "TORCH_COMPILE=0" in src
