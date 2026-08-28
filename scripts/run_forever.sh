#!/usr/bin/env bash
# =============================================================================
# run_forever.sh — multi-day unattended training loop
# -----------------------------------------------------------------------------
# ONE COMMAND to leave running overnight or for several days:
#
#     bash scripts/run_forever.sh
#
# On macOS it re-launches itself under `caffeinate -is` so the machine will
# NOT sleep or shut down while training (keep it plugged into AC power).
# On Linux with systemd it inhibits idle sleep the same way.
#
# CONTINUOUS mode (default, "I want results"):
#   A background downloader pulls new frames forever (resumable, never
#   redownloads) WHILE the streaming trainer (scripts/train_streaming.py)
#   trains from the very FIRST frame and never restarts. Every newly
#   downloaded frame is picked up within one epoch, and optimizer/EMA/LR state
#   accumulates across the whole run — one model that only gets better.
#
# Set CONTINUOUS=0 for the classic batched loop instead:
#   download N frames -> train on everything so far -> evaluate -> repeat.
#
# In both modes it:
#   1. Detects your hardware and uses maximum power: every core minus a
#      cooling headroom (2 cores idle on macOS by default — 8 of 10 cores
#      train — override with CPU_HEADROOM=N) and all RAM (Apple Silicon also
#      uses the MPS GPU),
#        - patch size is chosen from RAM (512 if >=12 GB, else 256),
#        - how much data to download is chosen from free disk.
#   2. Downloads ARPIL tiles a little at a time (resumable, never redownloads).
#   3. Trains an Attention U-Net (continuous, or per-cycle in batched mode).
#   4. Evaluates it (pixel Dice/IoU AND object-detection F1).
#   5. Logs EVERYTHING, timestamped, to ~/solar_results/arpil/run_forever.log:
#        - session banner (machine, GPU, config, log location)
#        - pre-flight checks of every data source URL (see below)
#        - a resource watchdog line every 5 minutes (load / RAM / disk)
#        - per-step timing, full subprocess output, per-cycle metric summary
#      plus a human-readable snapshot in ~/solar_results/arpil/STATUS.md.
#
# Useful extras:
#     bash scripts/run_forever.sh --once     # run exactly one cycle, then exit
#     tail -f ~/solar_results/arpil/run_forever.log    # follow the log
#     cat   ~/solar_results/arpil/STATUS.md            # latest state at a glance
#     kill "$(cat ~/solar_results/arpil/run_forever.lock)"   # stop cleanly
#     CPU_HEADROOM=6 bash scripts/run_forever.sh       # cooler/slower (more idle cores)
#
# Stop it any time with Ctrl-C. Re-run it later and it picks up exactly
# where it stopped (every step is resumable). A lock file prevents two
# instances from running at once.
# =============================================================================

# We do NOT use `set -e`: a transient failure (e.g. a lost network request)
# should not kill the whole "forever" loop. Each step is checked individually.
set -uo pipefail

# ---- CONFIG (edit these to taste) ------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$HOME/solar_data/arpil"          # train tiles + manifest live here
RESULTS_DIR="$HOME/solar_results/arpil"    # checkpoints + metrics live here
TEST_DIR="$HOME/solar_data/arpil_test"     # OFFICIAL test-split tiles (optional)

CHANNELS="${CHANNELS:-aia171}"   # wavelength(s) to train on. The plasma-physics
                                  # stack "aia171 aia193 hmi_m" (cool loops + hot
                                  # plasma + magnetic field) is the strongest input
                                  # for a from-scratch model — active regions are
                                  # defined by the field. The full 13-channel core
                                  # stack (all 8 AIA EUV + all 5 HMI field
                                  # channels) is the maximum-info option. Changing
                                  # CHANNELS means the tiles must be rebuilt (wipe
                                  # $DATA_DIR and the old checkpoints).
BASE_CHANNELS="${BASE_CHANNELS:-32}"  # model width: 32 (≈8M params, default) or 16
                                       # (≈2M, ~2x faster epochs — for short windows)
DEEP_SUPERVISION="${DEEP_SUPERVISION:-0}"  # 1 = auxiliary losses on decoder stages
                                           # (better small/thin structure recall,
                                           # ~10-15% slower per epoch — usually worth it)
# MAXIMUM-SPEED preset (M4 Mac, all 13 channels, no headroom):
#   CHANNELS="aia94 aia131 aia1600 aia171 aia193 aia211 aia304 aia335 hmi_m hmi_bx hmi_by hmi_bz hmi_v" \
#   CPU_HEADROOM=0 TILES_PER_EPOCH=200 VAL_EPOCH=5 VAL_SUBSET=500 \
#   bash scripts/run_forever.sh
# (expect a hot machine and spinning fans — that is the trade-off of 10/10 cores)
# BEST-MODEL preset (best quality per image size for ~6k tiles on MPS):
#   ...plus BASE_CHANNELS=48 DEEP_SUPERVISION=1  (~18M params + deep supervision)
# KAGGLE preset (T4 GPU, 20 GB working disk, 12-hour sessions — just re-run
# the same notebook cell after each session kill; data + checkpoints resume):
#   export HOME=/kaggle/work
#   CHANNELS="aia94 aia131 aia1600 aia171 aia193 aia211 aia304 aia335 hmi_m hmi_bx hmi_by hmi_bz hmi_v" \
#   MIN_FREE_GB=4 MAX_TOTAL_FRAMES=1500 DOWNLOAD_WORKERS=4 \
#   CPU_HEADROOM=2 TILES_PER_EPOCH=200 VAL_EPOCH=10 VAL_SUBSET=300 \
#   bash scripts/run_forever.sh
# (fewer download workers: each in-flight frame holds a ~570 MB temp file, and
# Kaggle's whole working disk is ~20 GB; 4 workers ≈ 2.4 GB in flight)
MASK_SPLIT="${MASK_SPLIT:-train}"  # ARPIL split to mine: train|validation|test|leaky_validation
# Two-phase training, tuned for short deadlines (e.g. a 3-day window):
#   - while data is still being collected: FAST passes (EPOCHS), so each cycle
#     ends quickly and the next batch of frames starts downloading sooner
#   - once MAX_TOTAL_FRAMES is reached: DEEP passes (EPOCHS_FINAL) on the
#     complete dataset — this is where the final model quality comes from
EPOCHS=15                  # fast passes during the collection phase
EPOCHS_FINAL=60            # deep passes once the download cap is reached
FRAMES_PER_CYCLE="${FRAMES_PER_CYCLE:-200}"   # NEW frames to add each cycle (may be auto-reduced)
MAX_TOTAL_FRAMES="${MAX_TOTAL_FRAMES:-2000}"  # stop downloading beyond this (0 = no cap)
# Tiles sampled PER EPOCH (0 = use the whole dataset every epoch). With
# subsampling + constant LR, this number does NOT change how fast the model
# learns — MPS processes tiles at a fixed rate regardless of the chunk size.
# It only sets the epoch GRANULARITY: smaller = more, shorter epochs (finer
# random shuffling, faster progress ticks).
#
# VALIDATION: a FULL validation on the grown val set (with TTA) takes ~1 hour
# and used to eat ~80% of the wall clock. So the streaming trainer validates
# on a fixed random VAL_SUBSET slice WITHOUT TTA (~1-2 min), every VAL_EPOCH
# short epochs. best.pt tracks that consistent quick bar; for the FINAL
# official number, run evaluate.py on the full val set at the end.
TILES_PER_EPOCH="${TILES_PER_EPOCH:-100}"
VAL_EPOCH="${VAL_EPOCH:-10}"
VAL_SUBSET="${VAL_SUBSET:-300}"
MAX_CYCLES=100000          # effectively "forever"
MIN_FREE_GB="${MIN_FREE_GB:-40}"    # refuse to download when free disk drops below this
                                  # (small machines such as Kaggle's 20 GB working
                                  # disk need MIN_FREE_GB=4 — see the KAGGLE preset)
CYCLE_PAUSE=60             # seconds between cycles (lets the disk settle)
WATCH_EVERY=300            # resource watchdog interval, seconds

# HOW training is organised:
#   CONTINUOUS=1 (default) — "I want results" mode: the downloader runs in the
#     background forever while the STREAMING trainer (scripts/train_streaming.py)
#     trains from the very FIRST frame and never restarts — optimizer state,
#     EMA and LR accumulate across the whole run, and every newly downloaded
#     frame is picked up within one epoch. One model that only gets better.
#   CONTINUOUS=0 — classic batched cycles (download 200 frames -> train -> repeat).
CONTINUOUS="${CONTINUOUS:-1}"
# Use 1 when resuming a checkpoint whose stored Dice was measured on a
# DIFFERENT dataset/representation (e.g. your 50-epoch Drive model trained on
# full-disk thumbnails, now resumed on 512 tiles). Without it, the old score
# (0.9535) stays the "best" bar forever and best.pt would never update.
CALIBRATE_BEST="${CALIBRATE_BEST:-0}"

# Official test-split evaluation (paper-comparable number). Turn on when you
# want the 2020-2024 test metric; it builds test tiles then evaluates the
# latest checkpoint against them every cycle.
EVAL_TEST_SPLIT=0          # 1 = build + evaluate the official test split
TEST_FRAMES=200            # how many official test frames to evaluate on

# ARPIL tile build options (match what the official workflow uses).
STRIDE=512
MIN_MASK_FRACTION=0.0005
KEEP_EMPTY_EVERY=64
# Parallel S3 downloaders — max-speed default. 16 frames in flight at once
# (~9.6 GB of temp .nc), and each file is itself pulled as 16 concurrent 16 MB
# parts, so the link is saturated end to end. Raise on very fast fiber
# (DOWNLOAD_WORKERS=32); don't go near 100 (~57 GB in flight, S3 throttles).
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-16}"

# ROLLING mode (for small persistent disks, e.g. Kaggle's 20 GiB): instead of
# stopping at MAX_TOTAL_FRAMES, keep downloading NEW frames forever and rotate
# the on-disk tile window — after each batch the oldest frames are retired
# (their names are recorded permanently and never re-downloaded) and their
# tiles are freed after a safety gap, making room for new frames. The model
# trains on a continuously rotating set of frames it has never seen.
ROLLING="${ROLLING:-0}"
ROLLING_WINDOW="${ROLLING_WINDOW:-1100}"   # frames kept on disk (rolling mode)
LR="${LR:-}"                     # optional learning rate for the streaming trainer
                                  # (rolling runs use a lower LR, e.g. 1e-4, so
                                  # updates overwrite less of the old knowledge)

# CPU headroom = cores left idle for the OS + to keep the machine cool.
# Default: 2 cores idle on macOS (8 of 10 cores train — maximum speed that
# still leaves the machine usable), 2 on Linux. Override with CPU_HEADROOM=N:
#   CPU_HEADROOM=1 -> 9 training cores (faster, machine feels busy)
#   CPU_HEADROOM=4 -> 6 training cores (cooler/slower)
if [[ "$(uname -s)" == "Darwin" ]]; then
    CPU_HEADROOM="${CPU_HEADROOM:-2}"
else
    CPU_HEADROOM="${CPU_HEADROOM:-2}"
fi
# -----------------------------------------------------------------------------

[[ "${1:-}" == "--once" ]] && MAX_CYCLES=1

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"  # make Homebrew tools visible
mkdir -p "$RESULTS_DIR" "$DATA_DIR" "$TEST_DIR"
LOG="$RESULTS_DIR/run_forever.log"
STATUS="$RESULTS_DIR/STATUS.md"
LOCK="$RESULTS_DIR/run_forever.lock"
PY="${SOLAR_PYTHON:-$REPO_DIR/.venv/bin/python}"   # SOLAR_PYTHON: use an existing
                                  # interpreter (e.g. Kaggle's session Python,
                                  # where creating a venv is broken); default
                                  # is the repo's own .venv (Mac / other).

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$LOG"; }
hr()  { printf '[%s] %s\n' "$(ts)" "════════════════════════════════════════════════════════════════" | tee -a "$LOG"; }
dur() { local s=$(( $2 - $1 )); printf '%dh %02dm %02ds' $(( s / 3600 )) $(( (s % 3600) / 60 )) $(( s % 60 )); }

# --- macOS: make sure the machine will NOT sleep/turn off --------------------
if [[ "$(uname -s)" == "Darwin" && -z "${SOLAR_CAFFEINATED:-}" && -n "$(command -v caffeinate)" ]]; then
    log "macOS detected — re-launching under 'caffeinate -is' so the machine"
    log "keeps running while you sleep. Keep it plugged into AC power."
    export SOLAR_CAFFEINATED=1
    exec caffeinate -is bash "$0" "$@"
fi

# --- single-instance lock ----------------------------------------------------
if [[ -f "$LOCK" ]]; then
    OLD_PID="$(cat "$LOCK" 2>/dev/null || echo "")"
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        log "Another run_forever is already running (pid $OLD_PID) — exiting to avoid a double run."
        exit 1
    fi
    log "Removing stale lock file (pid ${OLD_PID:-?} no longer running)."
fi
echo $$ > "$LOCK"

WATCH_PID=""
INHIBIT_PID=""
DOWNLOADER_PID=""
on_exit() {
    local rc=$?
    [[ -n "${SHUTDOWN_MSG:-}" ]] && log "$SHUTDOWN_MSG"
    # Stop the background downloader (and its current python child) first.
    if [[ -n "$DOWNLOADER_PID" ]]; then
        pkill -P "$DOWNLOADER_PID" 2>/dev/null
        kill "$DOWNLOADER_PID" 2>/dev/null
    fi
    [[ -n "$WATCH_PID" ]]   && kill "$WATCH_PID" 2>/dev/null
    [[ -n "$INHIBIT_PID" ]] && kill "$INHIBIT_PID" 2>/dev/null
    rm -f "$LOCK"
    log "Session ended (exit $rc). Re-run 'bash scripts/run_forever.sh' to resume where it stopped (data + checkpoints kept)."
}
on_signal() {
    SHUTDOWN_MSG="Stop signal received — shutting down cleanly (all completed work is saved)."
    exit 130
}
trap on_signal INT TERM
trap on_exit EXIT

# --- helpers -----------------------------------------------------------------

free_gb() {
    # Free space for the data dir, in whole GB. Walks up to the nearest
    # existing ancestor in case the dir has not been created yet, and works
    # on both GNU (Linux) and BSD (macOS) df output.
    local target="${1:-$DATA_DIR}"
    while [[ -n "$target" && ! -e "$target" ]]; do target="$(dirname "$target")"; done
    [[ -z "$target" || "$target" == "." ]] && target="/"
    df -k "$target" 2>/dev/null | awk 'NR==2 {print int($4/1024/1024)}'
}

completed_frames() {
    "$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["completed_frames"])' \
        "$1/progress.json" 2>/dev/null || echo 0
}

# Compact machine stats for the watchdog line. Cross-platform (Linux / macOS).
sys_stats() {
    # Best-effort; must NEVER crash (older/newer macOS changed vm_stat's
    # format over the years) — worst case it prints "?".
    "$PY" - <<'PY' 2>/dev/null || echo "stats-unavailable"
import os, re, subprocess
total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
try:
    load = open("/proc/loadavg").read().split()[0]
except Exception:
    try:
        load = subprocess.check_output(["sysctl", "-n", "vm.loadavg"]).decode().lstrip("[]{} ").split()[0]
    except Exception:
        load = "?"
used = None
try:
    mi = dict(l.split(":", 1) for l in open("/proc/meminfo") if ":" in l)
    used = total - int(mi["MemAvailable"].strip().split()[0]) / 2**20
except Exception:
    try:
        v = subprocess.check_output(["vm_stat"]).decode()
        m = re.search(r"page size of (\d+)", v) or re.search(r"page size[:\s]+(\d+)", v)
        if m:
            page = int(m.group(1))
            def pages(pat):
                mm = re.search(pat + r":\s*(\d+)", v)
                return int(mm.group(1)) if mm else 0
            used = (pages(r"Pages active") + pages(r"Pages wired count")
                    + pages(r"Pages occupied by compressor") + pages(r"Compressed")) * page / 2**30
    except Exception:
        pass
ram = f"{used:.1f}/{total:.1f}GB" if used is not None else f"?/{total:.1f}GB"
print(f"load={load} ram={ram}")
PY
}

# Linux: inhibit idle sleep for the session (no-op elsewhere).
if [[ "$(uname -s)" == "Linux" && -n "$(command -v systemd-inhibit)" ]]; then
    systemd-inhibit --what=idle --who="solar-arpil" \
        --why="Solar active-region training (run_forever.sh)" sleep infinity >/dev/null 2>&1 &
    INHIBIT_PID=$!
    sleep 1
    if kill -0 "$INHIBIT_PID" 2>/dev/null; then
        log "Sleep inhibited for this session (systemd-inhibit, pid $INHIBIT_PID)."
    else
        INHIBIT_PID=""
        log "Sleep inhibition unavailable (no systemd bus) — the machine may idle-sleep on its own schedule."
    fi
fi

# Pick the best available Python 3: prefer a Homebrew build (3.10+) over the
# macOS system Python 3.9, which is EOL and only gets older, older wheels.
pick_python() {
    local cand
    for cand in python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
        if command -v "$cand" >/dev/null 2>&1; then
            echo "$cand"
            return 0
        fi
    done
    return 1
}

# True iff the venv actually has every module the pipeline imports.
deps_ok() {
    "$PY" -c 'import torch, numpy, h5py, netCDF4, astropy, PIL, boto3, requests, scipy, tqdm' >/dev/null 2>&1
}

# Install (or reinstall) dependencies into the venv, VERIFY the result, and
# put any failure where you can see it: the console AND the log.
install_deps() {
    local t0
    local pipout="$RESULTS_DIR/.pip_install.log"
    t0=$(date +%s)
    if ! "$PY" -m pip --version >>"$LOG" 2>&1; then
        log "venv has no pip — bootstrapping with ensurepip ..."
        "$PY" -m ensurepip --upgrade >>"$LOG" 2>&1
    fi
    "$PY" -m pip install --upgrade pip >"$pipout" 2>&1
    if "$PY" -m pip install -r "$REPO_DIR/requirements.txt" >>"$pipout" 2>&1; then
        cat "$pipout" >>"$LOG"
        if deps_ok; then
            log "✔ Dependencies installed and verified in $(dur $t0 $(date +%s))."
            rm -f "$pipout"
            return 0
        fi
        log "✖ pip install finished but the import check failed — last 15 lines of pip output:"
    else
        log "✖ Dependency install FAILED. Last 15 lines of pip output:"
    fi
    tail -n 15 "$pipout" | sed 's/^/      | /' | tee -a "$LOG"
    log "  The next cycle will retry automatically. If it keeps failing:"
    log "    1) rm -rf \"$REPO_DIR/.venv\" and re-run, or"
    log "    2) install a newer Python first (macOS system Python 3.9 is EOL):"
    log "       brew install python@3.11    (then re-run this script)"
    return 1
}

setup() {
    # Only create a venv when we're actually using the repo venv. With
    # SOLAR_PYTHON set (Kaggle), the session interpreter is used as-is.
    if [[ "$PY" == "$REPO_DIR/.venv/bin/python" && ! -x "$PY" ]]; then
        log "Creating Python venv at $REPO_DIR/.venv ..."
        local pybin
        pybin="$(pick_python || echo python3)"
        log "Python: $(command -v "$pybin" 2>/dev/null || echo "$pybin")  ($("$pybin" --version 2>&1))"
        if ! "$pybin" -m venv "$REPO_DIR/.venv" >>"$LOG" 2>&1; then
            log "✖ venv creation failed — last 10 lines of $LOG:"
            tail -n 10 "$LOG" | sed 's/^/      | /' | tee -a "$LOG"
        fi
    fi
    if ! deps_ok; then
        log "Python dependencies missing or broken — installing into $PY ..."
        install_deps || log "(install not complete yet — each cycle retries automatically)"
    fi
    log "Python: $("$PY" --version 2>&1)"
    if deps_ok; then
        "$PY" -c 'import torch; print("  torch", torch.__version__, "| mps:", torch.backends.mps.is_available(), "| cuda:", torch.cuda.is_available())' >>"$LOG" 2>&1
    fi
}

# Detect CPU cores, RAM and derive the patch size to use. Training is launched
# with --cpu-headroom "$CPU_HEADROOM" --memory-headroom-gb 0 so it grabs every
# core MINUS the cooling headroom and all RAM (see src/solar_ar/runtime.py);
# here we decide the *download* shape.
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
            if [[ "$disk_cap" -eq 0 ]]; then
                log "Disk limit: not enough free space for more frames (reserve ${MIN_FREE_GB} GB) — downloads will be skipped until space frees up."
            else
                log "Disk limit: MAX_TOTAL_FRAMES reduced to $disk_cap to respect the ${MIN_FREE_GB} GB reserve."
            fi
        fi
    fi

    GPU_INFO="$("$PY" -c '
import torch
if torch.cuda.is_available():
    print("CUDA " + torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    print("Apple Metal (MPS) - will be used for training automatically")
else:
    print("none (CPU only)")
' 2>/dev/null || echo 'unknown (torch not ready yet)')"
}

session_banner() {
    local disk_free; disk_free="$(free_gb)"
    hr
    log "SOLAR ACTIVE-REGION TRAINING — UNATTENDED LOOP"
    log "Host     : $(hostname 2>/dev/null || uname -n)  ($(uname -srmo 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g'))"
    log "Started  : $(date '+%Y-%m-%d %H:%M:%S %Z')"
    log "Hardware : $cpus cores (training uses all minus $CPU_HEADROOM cooling headroom) | ${ram_gb} GB RAM (all used) | ~${disk_free} GB free disk"
    log "GPU      : $GPU_INFO"
    log "Data dir : $DATA_DIR"
    log "Results  : $RESULTS_DIR"
    log "Config   : channels=$CHANNELS split=$MASK_SPLIT epochs=$EPOCHS frames/cycle=$FRAMES_PER_CYCLE max_frames=$MAX_TOTAL_FRAMES patch=$PATCH_SIZE min_free=${MIN_FREE_GB}GB cycles=${MAX_CYCLES} rolling=$ROLLING window=$ROLLING_WINDOW lr=${LR:-default}"
    log "Log      : $LOG   (watch: tail -f $LOG)"
    log "Status   : $STATUS"
    hr
}

# Pre-flight: verify every data source BEFORE committing to a multi-day run.
# Runs on the user's machine, where the real network applies.
preflight() {
    log "── Pre-flight: checking data sources ──────────────────────────────"
    "$PY" - "$MASK_SPLIT" <<'PY' 2>&1 | tee -a "$LOG" || log "Preflight script error — continuing; the loop will retry each source."
import csv, re, sys
split = sys.argv[1]
ok = warn = fail = 0
def good(m):  global ok;   ok += 1;   print(f"  [ OK ] {m}")
def mid(m):   global warn; warn += 1; print(f"  [WARN] {m}")
def bad(m):   global fail; fail += 1; print(f"  [FAIL] {m}")

import requests
BASE = "https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-ar-segmentation"
sample = []
# 1) Mask index CSV: reachable? how many frames? which date range?
try:
    r = requests.get(f"{BASE}/raw/main/{split}.csv", timeout=90)
    r.raise_for_status()
    if r.text.lstrip().startswith("version https://git-lfs.github.com/spec/v1"):
        bad(f"mask index {split}.csv returned a Git LFS pointer instead of CSV content")
        r = requests.get(f"{BASE}/resolve/main/{split}.csv?download=true", timeout=90)
        r.raise_for_status()
    rows = list(csv.DictReader(r.text.splitlines()))
    ABSENT = {"", "0", "0.0", "false", "no", "nan", "none"}
    present = [x for x in rows
               if x.get("present") is None or str(x.get("present")).strip().lower() not in ABSENT
               if x.get("file_path")]
    stamps = sorted({m.group(1) for x in present
                     if (m := re.search(r"(\d{8}[_T]?\d{4})", x["file_path"].split("/")[-1]))})
    if present and stamps:
        good(f"mask index {split}.csv: {len(present)} available frames, {stamps[0]} .. {stamps[-1]}")
        step = max(1, len(present) // 5)
        sample = [present[i] for i in range(0, len(present), step)][:5]
    else:
        bad(f"mask index {split}.csv: no 'present' frames found (try MASK_SPLIT=validation)")
except Exception as e:
    bad(f"mask index {BASE}/raw/main/{split}.csv unreachable ({e.__class__.__name__}: {e})")

# 2) Mask archive: exists and roughly the right size?
try:
    r = requests.head(f"{BASE}/resolve/main/data.tar.gz", allow_redirects=True, timeout=90)
    if r.status_code == 200:
        gb = int(r.headers.get("content-length", 0)) / 2**30
        good(f"mask archive data.tar.gz present ({gb:.2f} GB, one-time download)")
    else:
        bad(f"mask archive data.tar.gz HTTP {r.status_code}")
except Exception as e:
    mid(f"mask archive HEAD check failed ({e.__class__.__name__}) — will retry at download time")

# 3) Core S3 frames: do they exist for THIS split? (HEAD only, ~570 MB never downloaded)
if sample:
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.client import Config
        client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        found, examples = 0, []
        for row in sample:
            key = row["file_path"].replace("data/", "").replace(".h5", ".nc")
            try:
                client.head_object(Bucket="nasa-surya-bench", Key=key)
                found += 1
            except Exception:
                examples.append(key)
        if found == len(sample):
            good(f"core S3 frames: {found}/{len(sample)} probe frames exist in s3://nasa-surya-bench — downloads will work")
        elif found:
            mid(f"core S3 frames: only {found}/{len(sample)} probe frames exist — downloads will be partial (e.g. missing: {examples[:2]})")
        else:
            bad(f"core S3 frames: 0/{len(sample)} probe frames exist (e.g. {examples[:2]}) — try MASK_SPLIT=validation (it overlaps the core-SDO index)")
    except Exception as e:
        mid(f"core S3 probe failed ({e.__class__.__name__}: {e}) — network may be restricted here; will retry at download time")
else:
    mid("core S3 probe skipped (no mask frames to probe)")

print(f"  Preflight summary: {ok} ok, {warn} warnings, {fail} failures")
PY
    log "────────────────────────────────────────────────────────────────────"
}

# Run one step. Log its output; return success/failure without aborting.
run() {
    local desc="$1"; shift
    local t0 t1
    log "▶ $desc"
    t0=$(date +%s)
    if "$@" >>"$LOG" 2>&1; then
        t1=$(date +%s)
        log "✔ $desc — finished in $(dur $t0 $t1)"
        return 0
    else
        local rc=$?
        t1=$(date +%s)
        log "✖ $desc FAILED (exit $rc) after $(dur $t0 $t1) — full output above in $LOG; will retry on the next cycle"
        return 1
    fi
}

cleanup() {
    # Delete only safe junk. NEVER delete tiles or checkpoints — those are your
    # trained model and its training data.
    find "$REPO_DIR" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
    log "Disk: $(free_gb) GB free"
}

# Log the key numbers from a metrics JSON so you can skim the log without
# opening each file. Reads a handful of known keys; extra keys are ignored.
summarize_json() {
    local label="$1" file="$2"
    [[ -f "$file" ]] || { log "  $label: (no file)"; return; }
    "$PY" - "$label" "$file" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
label, path = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path))
except Exception as e:
    print(f"  {label}: unreadable ({e})")
    sys.exit(0)
keys = ["dice", "iou", "bce_loss", "precision", "recall", "f1", "ap", "ap50",
        "tp", "fp", "fn", "samples"]
parts = []
for k in keys:
    if k in d:
        v = d[k]
        parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
print(f"  {label}: " + ", ".join(parts))
PY
}

# Append the last few epoch lines of a cycle to the main log (training curve
# visible without opening the per-cycle files).
log_training_curve() {
    local metrics="$1"
    [[ -f "$metrics" ]] || return 0
    log "  Training curve (last epochs of this cycle):"
    "$PY" - "$metrics" <<'PY' 2>/dev/null | tee -a "$LOG"
import json, sys
lines = open(sys.argv[1]).read().splitlines()[-5:]
for line in lines:
    m = json.loads(line)
    print(f"    epoch {m['epoch']:3d}  train_loss={m['train_loss']:.4f}  val_loss={m['val_loss']:.4f}  val_dice={m['val_dice']:.4f}  val_iou={m['val_iou']:.4f}  lr={m['learning_rate']:.2e}")
PY
}

# Human-readable snapshot, refreshed at the end of every cycle.
write_status() {
    local frames="$1"
    {
        echo "# Solar AR training — status"
        echo
        echo "_Last updated: $(date '+%Y-%m-%d %H:%M:%S %Z') (cycle $CYCLE)_"
        echo
        echo "- Frames downloaded: **$frames** / $MAX_TOTAL_FRAMES"
        echo "- Latest checkpoint: \`$RESULTS_DIR/cycle_$CYCLE/best.pt\`"
        if [[ -f "$RESULTS_DIR/cycle_$CYCLE/pixel_metrics.json" ]]; then
            echo "- Val pixel:  \`$(cat "$RESULTS_DIR/cycle_$CYCLE/pixel_metrics.json" | tr -d '\n')\`"
        fi
        if [[ -f "$RESULTS_DIR/cycle_$CYCLE/detection_metrics.json" ]]; then
            echo "- Val objects: \`$(cat "$RESULTS_DIR/cycle_$CYCLE/detection_metrics.json" | tr -d '\n')\`"
        fi
        echo "- Disk free: $(free_gb) GB"
        echo
        echo "Full log: \`$LOG\`"
    } > "$STATUS"
}

start_watchdog() {
    # A resource line every WATCH_EVERY seconds. Counts 1-second sleeps (not
    # one big sleep) so the background subshell dies within ~1s when the
    # session ends — a long `sleep 300` child would otherwise outlive the
    # script, linger, and hold open any stdout pipe the user is reading.
    (
        count=0
        while :; do
            sleep 1
            count=$((count + 1))
            if [[ "$count" -ge "$WATCH_EVERY" ]]; then
                count=0
                log "[watch] $(sys_stats) | disk_free=$(free_gb)GB"
            fi
        done
    ) &
    WATCH_PID=$!
}

# Background downloader for CONTINUOUS mode: keeps pulling new frames forever
# (each invocation resumes where the last stopped) until the frame cap or the
# disk reserve says to pause. Every completed frame is republished to the
# manifest, which the streaming trainer picks up automatically.
downloader_loop() {
    while :; do
        FRAMES=$(completed_frames "$DATA_DIR")
        if [[ $MAX_TOTAL_FRAMES -gt 0 && $FRAMES -ge $MAX_TOTAL_FRAMES ]]; then
            log "Downloader: cap reached ($FRAMES frames) — all planned data collected, downloading complete."
            break
        fi
        if [[ $(free_gb) -lt $MIN_FREE_GB ]]; then
            log "Downloader: disk below the ${MIN_FREE_GB} GB reserve — pausing downloads for 10 min."
            sleep 600
            continue
        fi
        log "Downloader: batch start ($FRAMES frames on disk, adding up to $FRAMES_PER_CYCLE more)"
        ROLLING_ARG=""
        [[ "$ROLLING" == "1" ]] && ROLLING_ARG="--rolling --rolling-window $ROLLING_WINDOW"
        if ! "$PY" "$REPO_DIR/scripts/build_arpil_resumable.py" \
            --output-dir "$DATA_DIR" \
            --split "$MASK_SPLIT" \
            --channels "$CHANNELS" \
            --patch-size "$PATCH_SIZE" --stride "$STRIDE" \
            --min-mask-fraction "$MIN_MASK_FRACTION" --keep-empty-every "$KEEP_EMPTY_EVERY" \
            --max-frames "$FRAMES_PER_CYCLE" --sampling random \
            --download-workers "$DOWNLOAD_WORKERS" \
            $ROLLING_ARG \
            --min-free-disk-gb "$MIN_FREE_GB"; then
            # A crash (e.g. segfault in a native library) leaves the batch's
            # temp .nc files orphaned — they would sit in $TMPDIR for days.
            # Only ONE builder runs at a time (this loop is sequential), so it
            # is safe to sweep the prefix now.
            log "Downloader CRASHED (exit $?) — cleaning orphaned temp files, retrying with the next batch."
            rm -rf "${TMPDIR:-/tmp}"/arpil_resume_* 2>/dev/null
            sleep 10
        fi
    done
    log "Downloader: finished."
}

# --- main ---------------------------------------------------------------------

setup
detect_hardware
session_banner
preflight
start_watchdog

# ===========================================================================
# CONTINUOUS mode (default): download and train IN PARALLEL, forever.
# Training starts from the very first frame and never restarts — the streaming
# trainer accumulates optimizer/EMA/LR state across the whole run and picks up
# each newly downloaded frame within one epoch. "I want results" mode.
# ===========================================================================
if [[ "$CONTINUOUS" == "1" ]]; then
    log "CONTINUOUS mode: background downloader + streaming trainer (Ctrl-C stops both cleanly)."
    downloader_loop &
    DOWNLOADER_PID=$!
    CALIBRATE_ARG=""
    [[ "$CALIBRATE_BEST" == "1" ]] && CALIBRATE_ARG="--calibrate-best"
    DEEP_SUPERVISION_ARG=""
    [[ "$DEEP_SUPERVISION" == "1" ]] && DEEP_SUPERVISION_ARG="--deep-supervision"
    LR_ARG=""
    [[ -n "$LR" ]] && LR_ARG="--lr $LR"
    "$PY" "$REPO_DIR/scripts/train_streaming.py" \
        --manifest "$DATA_DIR/manifest.csv" \
        --channels "$CHANNELS" \
        --image-size "$PATCH_SIZE" \
        --base-channels "$BASE_CHANNELS" \
        $DEEP_SUPERVISION_ARG \
        $LR_ARG \
        --output-dir "$RESULTS_DIR/continuous" \
        --val-every "$VAL_EPOCH" \
        --max-tiles-per-epoch "$TILES_PER_EPOCH" \
        --val-subset "$VAL_SUBSET" \
        --status-file "$STATUS" \
        --cpu-headroom "$CPU_HEADROOM" --memory-budget-gb 0 \
        $CALIBRATE_ARG
    TRAINER_RC=$?
    log "Streaming trainer exited (code $TRAINER_RC) — stopping the downloader."
    kill "$DOWNLOADER_PID" 2>/dev/null
    exit "$TRAINER_RC"
fi

log "Entering the batched training loop (Ctrl-C to stop cleanly; re-run any time to resume)."

CYCLE=1
while [[ $CYCLE -le $MAX_CYCLES ]]; do
    CYCLE_START=$(date +%s)
    hr
    log "CYCLE $CYCLE started — $(date '+%Y-%m-%d %H:%M:%S %Z')"
    log "────────────────────────────────────────────────────────────────────"

    # Auto-repair: if the environment is broken (failed install, partial
    # download, wiped venv), fix it BEFORE spending a cycle on doomed steps.
    if [[ ! -x "$PY" ]] || ! deps_ok; then
        log "Python environment missing or broken — repairing before this cycle ..."
        if [[ "$PY" == "$REPO_DIR/.venv/bin/python" && ! -x "$PY" ]]; then
            pybin="$(pick_python || echo python3)"
            "$pybin" -m venv "$REPO_DIR/.venv" >>"$LOG" 2>&1
        fi
        if install_deps; then
            log "✔ Environment repaired — continuing with this cycle."
        else
            sleep 30
            CYCLE=$((CYCLE+1))
            continue
        fi
    fi

    FRAMES=$(completed_frames "$DATA_DIR")
    log "Frames on disk: $FRAMES / $MAX_TOTAL_FRAMES"

    # 1) Download more tiles (skip once we hit the cap, or when disk is tight).
    if [[ $MAX_TOTAL_FRAMES -gt 0 && $FRAMES -lt $MAX_TOTAL_FRAMES ]]; then
        if [[ $(free_gb) -lt $MIN_FREE_GB ]]; then
            log "Disk low ($(free_gb) GB < ${MIN_FREE_GB} GB). Skipping download this cycle; training on existing data."
        else
            run "build ARPIL tiles (split=$MASK_SPLIT, have $FRAMES frames, adding up to $FRAMES_PER_CYCLE)" \
                "$PY" "$REPO_DIR/scripts/build_arpil_resumable.py" \
                --output-dir "$DATA_DIR" \
                --split "$MASK_SPLIT" \
                --channels "$CHANNELS" \
                --patch-size "$PATCH_SIZE" --stride "$STRIDE" \
                --min-mask-fraction "$MIN_MASK_FRACTION" --keep-empty-every "$KEEP_EMPTY_EVERY" \
                --max-frames "$FRAMES_PER_CYCLE" --sampling random \
                --download-workers "$DOWNLOAD_WORKERS" \
                --min-free-disk-gb "$MIN_FREE_GB"
            FRAMES=$(completed_frames "$DATA_DIR")
            log "Now have $FRAMES frames."
        fi
    else
        if [[ $MAX_TOTAL_FRAMES -eq 0 ]]; then
            log "Download paused (free disk is below the ${MIN_FREE_GB} GB reserve) — training on existing data."
        else
            log "At download cap ($MAX_TOTAL_FRAMES frames) — training only from now on."
        fi
    fi

    # Guard: don't try to train until there is at least a real dataset.
    if [[ ! -f "$DATA_DIR/manifest.csv" ]]; then
        log "No manifest yet — nothing to train on. Waiting 30 s and retrying (next cycle will retry the download)."
        sleep 30
        CYCLE=$((CYCLE+1))
        continue
    fi

    # 2) Train (fresh run each cycle on the growing dataset — robust and simple).
    #    Maximum power: every core minus the cooling headroom and all RAM; on
    #    a Mac train.py picks the Apple Metal (MPS) GPU automatically.
    #    Two-phase: fast passes while collecting, deep passes once complete.
    OUT="$RESULTS_DIR/cycle_$CYCLE"
    if [[ $MAX_TOTAL_FRAMES -gt 0 && $FRAMES -ge $MAX_TOTAL_FRAMES ]]; then
        CYCLE_EPOCHS=$EPOCHS_FINAL
        PHASE="FINAL (dataset complete — deep $EPOCHS_FINAL-epoch training on all $FRAMES frames)"
    else
        CYCLE_EPOCHS=$EPOCHS
        PHASE="collection (fast $EPOCHS-epoch pass; more frames next cycle)"
    fi
    log "Training phase: $PHASE"
    run "train (cycle $CYCLE, $CYCLE_EPOCHS epochs, $FRAMES frames of data)" \
        "$PY" "$REPO_DIR/train.py" \
        --manifest "$DATA_DIR/manifest.csv" \
        --channels "$CHANNELS" \
        --image-size "$PATCH_SIZE" \
        --base-channels "$BASE_CHANNELS" \
        --epochs "$CYCLE_EPOCHS" --patience 0 \
        --cpu-headroom "$CPU_HEADROOM" --memory-headroom-gb 0 \
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
        log_training_curve "$OUT/metrics.jsonl"
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
                --download-workers "$DOWNLOAD_WORKERS" \
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

    # 6) Summary of this cycle's results + wall-clock time.
    if [[ -d "$OUT" ]]; then
        log "── Cycle $CYCLE results ────────────────────────────────────────"
        summarize_json "val pixel Dice/IoU" "$OUT/pixel_metrics.json"
        summarize_json "val object-detection" "$OUT/detection_metrics.json"
        if [[ "$EVAL_TEST_SPLIT" == "1" ]]; then
            summarize_json "TEST pixel Dice/IoU" "$OUT/test_pixel_metrics.json"
            summarize_json "TEST object-detection" "$OUT/test_detection_metrics.json"
        fi
    fi
    write_status "$FRAMES"
    CYCLE_END=$(date +%s)
    log "Cycle $CYCLE done in $(dur $CYCLE_START $CYCLE_END) — $FRAMES frames total."
    log "────────────────────────────────────────────────────────────────────"

    if [[ $CYCLE -lt $MAX_CYCLES ]]; then
        log "Next cycle in ${CYCLE_PAUSE} s (auto-continuing; Ctrl-C to stop)."
        sleep "$CYCLE_PAUSE"
    fi

    CYCLE=$((CYCLE+1))
done

log "Reached MAX_CYCLES=$MAX_CYCLES — the loop is done (data and checkpoints are kept)."
