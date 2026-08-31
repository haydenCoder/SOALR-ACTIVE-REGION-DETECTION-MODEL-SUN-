# Kaggle "runs at 100% GPU for hours, no output, no `best.pt`" — Audit

**Branch audited:** `arena/01a04247-soalr-active-region-detection`
**Entry point:** `kaggle/start_kaggle.ipynb` (single `%%bash` cell) → `scripts/run_forever.sh` → `scripts/build_arpil_resumable.py` (downloader) + `scripts/train_streaming.py` (trainer).

## Bottom line

The Python training pipeline is **healthy** — the full end-to-end builder + streaming trainer + rolling-window rotation passes on synthetic data (offline test, 127 tests green). The failure is entirely in the **launcher layer** (the Kaggle cell and its bash wiring). There are **three** independent defects, any one of which explains the symptom:

| # | Defect | Effect on Kaggle |
|---|--------|------------------|
| 1 | The `%%bash` code cell was flattened to **one single line with no newlines** (`%%bashset -xexport HOME=…`). Introduced by the "fix" commit `e726fdf`. | IPython/Jupyter reads the first line as a cell magic named **`%%bashset`** → `UsageError: Cell magic '%%bashset' not found.` The cell aborts instantly and **nothing runs at all**. Re-running it therefore never kills an older background run — a previously-started trainer keeps burning GPU at ~100% while the foreground cell produces nothing. |
| 2 | (pre-existing, what commit `e726fdf` was *trying* to fix) A `# comment` line placed **inside** the `VAR=value \ … \ bash run_forever.sh` backslash-continuation launch chain. | Bash joins backslash-continued lines *before* parsing comments, so a `#` comments out the **rest of that logical line** — every `VAR=value` before the `#` is silently dropped (proven: `CHANNELS`/`ROLLING` arrive empty in the child). |
| 3 | Dropped vars fall back to **desktop defaults**: `CHANNELS=aia171` (1 ch), `ROLLING=0`, `MIN_FREE_GB=40`, `DOWNLOAD_WORKERS=16`, `TORCH_COMPILE=1`. On Kaggle's ~14–19 GB free disk the downloader sees `free < MIN_FREE_GB(40)` and **pauses downloads forever**; additionally `detect_hardware` clobbered `MAX_TOTAL_FRAMES=0` (rolling "forever") to `disk_cap=0`. | The streaming trainer then **polls for an empty manifest every 15 s forever** — no epochs, no `last.pt`, no `best.pt`, no training output. (`best.pt` is only written at the 10th epoch *and* on a Dice improvement; `last.pt` is written only at the end of each epoch — so an epoch that never finishes produces neither.) |

All three were reproduced/verified in this audit (IPython magic parsing, bash comment-drops vars, and the trainer's wait-loop in `train_streaming.py`).

## What was fixed

1. **`kaggle/start_kaggle.ipynb`** — rebuilt the code cell with proper newlines:
   - `%%bash` is alone on the first line; every physical line is a real line.
   - The 19 preset environment variables form **one clean backslash-continued chain** with **no comment lines inside it** (all explanatory notes moved above it). Verified each var now reaches `run_forever.sh`.
   - Added a `/kaggle/working` mirror of `best.pt`/`last.pt`/`metrics.jsonl` (in addition to `/kaggle/output`) so a model is visible from the Output tab whichever path Kaggle captures.
2. **`scripts/run_forever.sh`**
   - `detect_hardware`: in **rolling mode** (`ROLLING=1`) the static disk-space frame cap is **no longer** applied, so `MAX_TOTAL_FRAMES=0` ("download new frames forever") is honoured and the rolling tile-byte budget (`ROLLING_MAX_TILE_GB`) — enforced per-frame by the builder — does the bounding instead. (Non-rolling desktop behaviour unchanged.)
   - Added a **config sanity gate** (`config_sanity`) that runs in the first second and **fails loudly** if a non-macOS box is handed desktop-only defaults (`MIN_FREE_GB≥30` on a small disk, or `DOWNLOAD_WORKERS>8` on a small disk, or empty `CHANNELS`), printing the full resolved config. This turns the previous "hours of silence" into an immediate, unmistakable message.
3. **`tests/test_kaggle_notebook.py`** (new, 24 tests) — regression-locks: cell is multi-line with `%%bash` first; the launch prefix is a clean continuation chain (no stray `#`, every line `\`-continued); every required preset var is present; `MIN_FREE_GB=4` and `TORCH_COMPILE=0` for Kaggle.

## How to verify a healthy run (what you should see)

Within the first few minutes of the cell, in the cell output and in `~/solar_results/arpil/run_forever.log`:

- `RESOLVED CONFIG: channels=[aia94 … hmi_v] … rolling=1 min_free=4GB … torch_compile=0 …` then `Config sanity gate passed.`
- `torch 2.x | CUDA available: True | GPUs: 2`
- `[stream] using 2 GPUs with DataParallel …`
- `Pre-flight: … Preflight summary: 3 ok …`
- `[stream] dataset refreshed: N train / M val tiles …`
- Per epoch: `[stream] epoch=0001 loss=… …`; at epoch **10** the first `val_dice` appears and `best.pt` is written to `~/solar_results/arpil/continuous/` (mirrored to `/kaggle/output` and `/kaggle/working`).

If instead you see `✖ FATAL CONFIG: MIN_FREE_GB=40 …`, the preset env vars did not propagate (re-pull/re-run the fixed cell) — it now fails immediately rather than after hours.

## Deploying the fix to Kaggle

The Kaggle cell re-clones `arena/01a04247-…` at runtime. These fixes must land on that branch (or the cell's `git clone -b …` line must point at the branch containing them) before the next Kaggle session picks them up.
