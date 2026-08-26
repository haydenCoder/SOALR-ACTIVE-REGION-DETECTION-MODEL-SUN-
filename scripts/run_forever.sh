#!/usr/bin/env bash
# =============================================================================
# run_forever.sh
# -----------------------------------------------------------------------------
# A fire-and-forget training loop for your Mac (or any machine). It:
#
#   1. Downloads ARPIL tiles a little at a time  (resumes where it left off),
#   2. Trains an Attention U-Net on everything downloaded so far,
#   3. Evaluates it (pixel Dice/IoU  AND  object-detection F1),
#   4. Cleans up junk (never your data or checkpoints),
#   5. Repeats forever, adding more frames each pass.
#
# Usage:
#     bash scripts/run_forever.sh
#
# - Stop it any time with Ctrl-C.
# - Re-run it later and it picks up exactly where it stopped, because every
#   step in this project is resumable (downloads, training, everything).
# - The data lives on your own SSD, so nothing is lost on reboot.
# =============================================================================

# We do NOT use `set -e`: a transient failure (e.g. a lost network request)
# should not kill the whole "forever" loop. Each step is checked individually.
set -uo pipefail

# ---- CONFIG (edit these to taste) -----------------------------------------
# Paths are auto-detected relative to this script's location in the repo.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$HOME/solar_data/arpil"          # tiles + manifest live here
RESULTS_DIR="$HOME/solar_results/arpil"    # checkpoints + metrics live here

CHANNELS="aia171"          # wavelength(s) to train on (ARPIL manifest column name)
IMAGE_SIZE=512             # patch size fed to the model
EPOCHS=30                  # epochs to train each cycle
FRAMES_PER_CYCLE=200       # NEW frames to add each cycle
MAX_TOTAL_FRAMES=2000      # stop downloading beyond this (see README notes)
MAX_CYCLES=100000          # effectively "forever"
MIN_FREE_GB=40             # refuse to download when free disk drops below this

# ARPIL tile build options (match what the official workflow uses)
PATCH_SIZE=512
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
    # Number of source frames already built (from the resumable builder's state).
    "$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["completed_frames"])' \
        "$DATA_DIR/progress.json" 2>/dev/null || echo 0
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
    # Confirm the accelerator the code will use (cuda > mps > cpu).
    "$PY" -c 'import torch; print("torch", torch.__version__, "| mps:", torch.backends.mps.is_available(), "| cuda:", torch.cuda.is_available())' 2>&1 | tee -a "$LOG"
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
    # trained model and its training data. The real disk saver is the
    # MAX_TOTAL_FRAMES cap plus MIN_FREE_GB guard above.
    find "$REPO_DIR" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
    log "Disk: $(free_gb) GB free | data=$("$PY" -c 'import os,sys;print(round(sum(os.path.getsize(os.path.join(r,f)) for r,d,fs in os.walk(sys.argv[1]) for f in fs)/1e9,2))' "$DATA_DIR" 2>/dev/null || echo '?') GB | results=$("$PY" -c 'import os,sys;print(round(sum(os.path.getsize(os.path.join(r,f)) for r,d,fs in os.walk(sys.argv[1]) for f in fs)/1e9,2))' "$RESULTS_DIR" 2>/dev/null || echo '?') GB"
}

# --- main loop --------------------------------------------------------------

setup
CYCLE=1

while [[ $CYCLE -le $MAX_CYCLES ]]; do
    log "==================== Cycle $CYCLE ===================="
    FRAMES=$(completed_frames)

    # 1) Download more tiles (skip once we hit the cap, or when disk is tight).
    if [[ $FRAMES -lt $MAX_TOTAL_FRAMES ]]; then
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
            FRAMES=$(completed_frames)
            log "Now have $FRAMES frames."
        fi
    else
        log "Reached $MAX_TOTAL_FRAMES frames — download cap hit, training only."
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
        --image-size "$IMAGE_SIZE" \
        --epochs "$EPOCHS" --patience 0 \
        --output-dir "$OUT"

    # 3) Evaluate (pixel segmentation metrics + object-detection F1).
    if [[ -f "$OUT/best.pt" ]]; then
        run "evaluate pixel Dice/IoU" \
            "$PY" "$REPO_DIR/evaluate.py" \
            --manifest "$DATA_DIR/manifest.csv" --checkpoint "$OUT/best.pt" \
            --channels "$CHANNELS" --split val --image-size "$IMAGE_SIZE" --batch-size 8 \
            --output "$OUT/pixel_metrics.json"
        run "evaluate object-detection F1" \
            "$PY" "$REPO_DIR/evaluate_detection.py" \
            --manifest "$DATA_DIR/manifest.csv" --checkpoint "$OUT/best.pt" \
            --channels "$CHANNELS" --split val --image-size "$IMAGE_SIZE" --batch-size 5 \
            --iou-threshold 0.5 --output "$OUT/detection_metrics.json"
    else
        log "No best.pt produced this cycle — skipping evaluation."
    fi

    # 4) Cleanup junk and report disk usage.
    cleanup

    CYCLE=$((CYCLE+1))
done
