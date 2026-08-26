#!/usr/bin/env bash
# =============================================================================
# run_forever.sh
# -----------------------------------------------------------------------------
# A fire-and-forget training loop for your Mac (or any machine). It:
#
#   1. Detects your hardware (CPU cores, RAM, free disk) and uses it sensibly:
#        - training uses (cores - 2) and (RAM - headroom) automatically,
#        - patch size is chosen from RAM (512 if >=12 GB, else 256),
#        - how much data to download is chosen from free disk.
#   2. Downloads ARPIL tiles a little at a time  (resumes where it left off),
#   3. Trains an Attention U-Net on everything downloaded so far,
#   4. Evaluates it (pixel Dice/IoU  AND  object-detection F1),
#   5. Optionally evaluates on the OFFICIAL 2020-2024 test split,
#   6. Cleans up junk (never your data or checkpoints),
#   7. Repeats forever, adding more frames each pass.
#
# Usage:
#     bash scripts/run_forever.sh
#
# - Stop it any time with Ctrl-C. Re-run it later and it picks up exactly
#   where it stopped (every step is resumable).
# =============================================================================

# We do NOT use `set -e`: a transient failure (e.g. a lost network request)
# should not kill the whole "forever" loop. Each step is checked individually.
set -uo pipefail

# ---- CONFIG (edit these to taste) -----------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$HOME/solar_data/arpil"          # train tiles + manifest live here
RESULTS_DIR="$HOME/solar_results/arpil"    # checkpoints + metrics live here
TEST_DIR="$HOME/solar_data/arpil_test"     # OFFICIAL test-split tiles (optional)

CHANNELS="aia171"          # wavelength(s) to train on (ARPIL manifest column name)
EPOCHS=30                  # epochs to train each cycle
FRAMES_PER_CYCLE=200       # NEW frames to add each cycle (may be auto-reduced)
MAX_TOTAL_FRAMES=2000      # stop downloading beyond this (0 = no cap)
MAX_CYCLES=100000          # effectively "forever"
MIN_FREE_GB=40             # refuse to download when free disk drops below this

# Official test-split evaluation (paper-comparable number). Turn on when you
# want the 2020-2024 test metric; it builds test tiles then evaluates the
# latest checkpoint against them every cycle.
EVAL_TEST_SPLIT=0          # 1 = build + evaluate the official test split
TEST_FRAMES=200            # how many official test frames to evaluate on

# ARPIL tile build options (match what the official workflow uses).
STRIDE=512
MIN_MASK_FRACTION=0.0005
KEEP_EMPTY_EVERY=64
# -----------------------------------------------------------------------------

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"  # make Homebrew tools visible
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/run_forever.log"
PY="$REPO_DIR/.venv/bin/python"

log() { printf '%s\n' "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$LOG"; }

# --- helpers ---------------------------------------------------------------

free_gb() {
    # macOS/BSD `df` prints a 1024-byte-block free count on line 2.
    df -k "$DATA_DIR" 2>/dev/null | awk 'NR==2 {print int($4/1024/1024)}'
}

completed_frames() {
    "$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["completed_frames"])' \
        "$1/progress.json" 2>/dev/null || echo 0
}

setup() {
    if [[ ! -x "$PY" ]]; then
        log "Creating Python venv at $REPO_DIR/.venv ..."
        python3 -m venv "$REPO_DIR/.venv"
        "$PY" -m pip install --upgrade pip >/dev/null
        "$PY" -m pip install -r "$REPO_DIR/requirements.txt" >>"$LOG" 2>&1
        log "Dependencies installed."
    fi
    log "Python: $("$PY" --version 2>&1)"
    "$PY" -c 'import torch; print("torch", torch.__version__, "| mps:", torch.backends.mps.is_available(), "| cuda:", torch.cuda.is_available())' 2>&1 | tee -a "$LOG"
}

# Detect CPU cores, RAM and derive the patch size + CPU budget to use.
# Training itself already applies (cores - 2) and RAM headroom internally
# (see src/solar_ar/runtime.py); here we decide the *download* shape.
detect_hardware() {
    local hw
    hw="$("$PY" - <<'PY'
import os
cpus = os.cpu_count() or 1
try:
    ram_gb = round(os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 1073741824, 1)
except (AttributeError, ValueError, OSError):
    ram_gb = 8.0
patch = 512 if ram_gb >= 12.0 else 256
print(f"cpus={cpus}")
print(f"ram_gb={ram_gb}")
print(f"cpu_budget={max(1, cpus - 2)}")
print(f"patch={patch}")
PY
)"
    eval "$hw"
    PATCH_SIZE="$patch"

    # Cap total frames by free disk, reserving MIN_FREE_GB. Each source frame
    # yields ~3 kept 512x512 tiles (~4 MB after compression), so ~4 MB/frame.
    local disk_free
    disk_free="$(free_gb)" || disk_free=0
    if [[ "$disk_free" -gt 0 ]]; then
        local disk_cap
        disk_cap=$(( (disk_free - MIN_FREE_GB) * 1024 / 4 ))
        [[ "$disk_cap" -lt 0 ]] && disk_cap=0
        if [[ "$MAX_TOTAL_FRAMES" -eq 0 || "$disk_cap" -lt "$MAX_TOTAL_FRAMES" ]]; then
            MAX_TOTAL_FRAMES="$disk_cap"
        fi
    fi

    log "Hardware: $cpus cores | ${ram_gb} GB RAM | ~$disk_free GB free disk"
    log "Decisions: train CPU budget=$cpu_budget | patch size=$PATCH_SIZE | max total frames=$MAX_TOTAL_FRAMES"
}

# Run one step. Log its output; return success/failure without aborting.
run() {
    local desc="$1"; shift
    log "▶ $desc"
    if "$@" >>"$LOG" 2>&1; then
        log "✔ $desc"
        return 0
    else
        log "✖ $desc FAILED — will retry on the next cycle (see $LOG)"
        return 1
    fi
}

cleanup() {
    # Delete only safe junk. NEVER delete tiles or checkpoints — those are your
    # trained model and its training data.
    find "$REPO_DIR" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
    log "Disk: $(free_gb) GB free"
}

# --- main loop --------------------------------------------------------------

setup
detect_hardware
CYCLE=1

while [[ $CYCLE -le $MAX_CYCLES ]]; do
    log "==================== Cycle $CYCLE ===================="
    FRAMES=$(completed_frames "$DATA_DIR")

    # 1) Download more tiles (skip once we hit the cap, or when disk is tight).
    if [[ $MAX_TOTAL_FRAMES -gt 0 && $FRAMES -lt $MAX_TOTAL_FRAMES ]]; then
        if [[ $(free_gb) -lt $MIN_FREE_GB ]]; then
            log "Disk low ($(free_gb) GB < ${MIN_FREE_GB} GB). Skipping download this cycle; training on existing data."
        else
            run "build ARPIL tiles (have $FRAMES frames)" \
                "$PY" "$REPO_DIR/scripts/build_arpil_resumable.py" \
                --output-dir "$DATA_DIR" \
                --channels "$CHANNELS" \
                --patch-size "$PATCH_SIZE" --stride "$STRIDE" \
                --min-mask-fraction "$MIN_MASK_FRACTION" --keep-empty-every "$KEEP_EMPTY_EVERY" \
                --max-frames "$FRAMES_PER_CYCLE" --sampling random \
                --min-free-disk-gb "$MIN_FREE_GB"
            FRAMES=$(completed_frames "$DATA_DIR")
            log "Now have $FRAMES frames."
        fi
    else
        log "At download cap ($MAX_TOTAL_FRAMES frames) — training only."
    fi

    # Guard: don't try to train until there is at least a real dataset.
    if [[ ! -f "$DATA_DIR/manifest.csv" ]]; then
        log "No manifest yet — nothing to train on. Waiting and retrying."
        sleep 30
        CYCLE=$((CYCLE+1))
        continue
    fi

    # 2) Train (fresh run each cycle on the growing dataset — robust and simple).
    OUT="$RESULTS_DIR/cycle_$CYCLE"
    run "train (cycle $CYCLE, $EPOCHS epochs)" \
        "$PY" "$REPO_DIR/train.py" \
        --manifest "$DATA_DIR/manifest.csv" \
        --channels "$CHANNELS" \
        --image-size "$PATCH_SIZE" \
        --epochs "$EPOCHS" --patience 0 \
        --output-dir "$OUT"

    # 3) Evaluate on the validation split (pixel + object metrics).
    if [[ -f "$OUT/best.pt" ]]; then
        run "evaluate pixel Dice/IoU" \
            "$PY" "$REPO_DIR/evaluate.py" \
            --manifest "$DATA_DIR/manifest.csv" --checkpoint "$OUT/best.pt" \
            --channels "$CHANNELS" --split val --image-size "$PATCH_SIZE" --batch-size 8 \
            --output "$OUT/pixel_metrics.json"
        run "evaluate object-detection F1" \
            "$PY" "$REPO_DIR/evaluate_detection.py" \
            --manifest "$DATA_DIR/manifest.csv" --checkpoint "$OUT/best.pt" \
            --channels "$CHANNELS" --split val --image-size "$PATCH_SIZE" --batch-size 5 \
            --iou-threshold 0.5 --output "$OUT/detection_metrics.json"
    else
        log "No best.pt produced this cycle — skipping evaluation."
    fi

    # 4) Optional: official 2020-2024 test-split evaluation (paper-comparable).
    if [[ "$EVAL_TEST_SPLIT" == "1" ]]; then
        if [[ $(free_gb) -ge $MIN_FREE_GB ]]; then
            run "build official test tiles" \
                "$PY" "$REPO_DIR/scripts/build_arpil_resumable.py" \
                --output-dir "$TEST_DIR" --split test \
                --channels "$CHANNELS" \
                --patch-size "$PATCH_SIZE" --stride "$STRIDE" \
                --min-mask-fraction "$MIN_MASK_FRACTION" --keep-empty-every "$KEEP_EMPTY_EVERY" \
                --max-frames "$TEST_FRAMES" --sampling random --val-ratio 1.0 \
                --min-free-disk-gb "$MIN_FREE_GB" \
                --selection-file "$RESULTS_DIR/arpil_test_selection.json"
        fi
        # The test manifest is written with every row as "val" (val-ratio 1.0),
        # so --split val evaluates the whole test set.
        if [[ -f "$OUT/best.pt" && -f "$TEST_DIR/manifest.csv" ]]; then
            run "evaluate OFFICIAL TEST pixel Dice/IoU" \
                "$PY" "$REPO_DIR/evaluate.py" \
                --manifest "$TEST_DIR/manifest.csv" --checkpoint "$OUT/best.pt" \
                --channels "$CHANNELS" --split val --image-size "$PATCH_SIZE" --batch-size 8 \
                --output "$OUT/test_pixel_metrics.json"
            run "evaluate OFFICIAL TEST object-detection F1" \
                "$PY" "$REPO_DIR/evaluate_detection.py" \
                --manifest "$TEST_DIR/manifest.csv" --checkpoint "$OUT/best.pt" \
                --channels "$CHANNELS" --split val --image-size "$PATCH_SIZE" --batch-size 5 \
                --iou-threshold 0.5 --output "$OUT/test_detection_metrics.json"
        fi
    fi

    # 5) Cleanup junk and report disk usage.
    cleanup

    CYCLE=$((CYCLE+1))
done
