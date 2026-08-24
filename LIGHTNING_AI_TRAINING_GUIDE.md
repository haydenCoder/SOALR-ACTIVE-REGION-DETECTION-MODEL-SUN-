# Lightning AI Training Guide

This repository includes a **single-file plugin**:

- `solar_arpil_plugin.py`

It now supports:
- automatic dependency installation,
- automatic real-data download when reachable,
- conversion to **HDF5 `.h5` training tiles**,
- **70% real / 30% synthetic** train split creation,
- duplicate-sample protection,
- saved provenance for every sample,
- automatic checkpoint resume,
- periodic preview image generation,
- verbose logging.

---

## 1) Clone the branch in Lightning AI

Open a Lightning AI Studio, then in its terminal:

```bash
git clone https://github.com/haydenCoder/SOALR-ACTIVE-REGION-DETECTION-MODEL-SUN-.git
cd SOALR-ACTIVE-REGION-DETECTION-MODEL-SUN-
git fetch origin arena/01a03190-soalr-active-region-detection
git checkout arena/01a03190-soalr-active-region-detection
```

### Already have an older copy in the Studio?

If the Studio already contains a checkout from before the mask fix, **replace it
before running** — the old `solar_arpil_plugin.py` builds per-frame mask URLs
that 404, so it will fail on every ARPIL frame. Refresh in place:

```bash
cd SOALR-ACTIVE-REGION-DETECTION-MODEL-SUN-
git fetch origin arena/01a03190-soalr-active-region-detection
git checkout arena/01a03190-soalr-active-region-detection
git reset --hard origin/arena/01a03190-soalr-active-region-detection
```

`reset --hard` discards local edits to tracked files. If you changed anything you
want to keep, `git stash` first. Untracked files — including `plugin_runs/` and
any downloaded data — are left alone, so checkpoints and caches survive.

If the old copy was downloaded as a zip rather than cloned, it has no git
history to update; delete the directory and clone fresh instead:

```bash
rm -rf SOALR-ACTIVE-REGION-DETECTION-MODEL-SUN-
```

Then confirm the fix is actually present in the working copy:

```bash
grep -c MASK_ARCHIVE_URL solar_arpil_plugin.py    # must print 2, not 0
```

Verify you are on the right commit and that the code is healthy before training:

```bash
git log --oneline -1
pip install -r requirements.txt
python -m pytest tests/ -q          # expect: 108 passed
```

## 1b) Where the data actually comes from

Two separate sources, with different access methods. This matters because they
fail in different ways:

| Data | Host | Access | Size |
| --- | --- | --- | --- |
| ARPIL masks (`.h5`, `[2,4096,4096]`) | Hugging Face `surya-bench-ar-segmentation` | one `data.tar.gz`, extracted on demand | ~1.31 GB total |
| Core SDO frames (`.nc`, 13 channels) | AWS S3 `s3://nasa-surya-bench` (unsigned) | per-frame `boto3` download | ~570 MB **per frame** |

**The masks are not individually downloadable.** Upstream publishes only the CSV
splits plus a single `data.tar.gz`; the `data/<year>/<month>/<stamp>.h5` paths in
the CSV `file_path` column exist *inside* that archive. The plugin downloads the
archive once into `<work-dir>/dataset/download_cache/` and extracts members as
needed. Budget the 1.3 GB and the one-time index pass on first run.

Frames are the expensive part: at ~570 MB each, `--max-frames 48` streams roughly
27 GB. They are fetched to a temp/cache location, tiled, and the `.nc` is not
kept, so peak disk stays modest — but the network transfer is real. Start with
`--max-frames 4` to confirm the path works before committing to a long run.

Check reachability from the Studio first; if either is blocked, use the
SHARP fallback in section 3 instead:

```bash
curl -sI "https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-ar-segmentation/resolve/main/train.csv" | head -1
python -c "import boto3;from botocore import UNSIGNED;from botocore.config import Config;print(boto3.client('s3',config=Config(signature_version=UNSIGNED)).list_objects_v2(Bucket='nasa-surya-bench',MaxKeys=3).get('KeyCount'))"
```

---

## 2) Main training command

You do **not** have to install dependencies manually if you use the plugin normally.
It auto-installs missing dependencies from `requirements.txt`.

This is the recommended run for your target setup:
- ~15 GB RAM
- 4 CPU threads
- 3 channels: `aia171`, `aia193`, `hmi_m`
- 70% real / 30% synthetic
- HDF5 tiles
- 100 epochs
- D4 test-time augmentation at validation
- verbose logging
- previews every 5 epochs
- automatic resume if `last.pt` already exists

```bash
python3 solar_arpil_plugin.py \
  --source-mode arpil_sdo \
  --work-dir plugin_runs/arpil_plugin \
  --channels aia171 aia193 hmi_m \
  --mask-split train \
  --real-ratio 0.70 \
  --synthetic-ratio 0.30 \
  --max-frames 48 \
  --tile-size 512 \
  --tile-stride 512 \
  --min-mask-fraction 0.0025 \
  --keep-empty-every 32 \
  --val-ratio 0.15 \
  --cpu-threads 4 \
  --memory-budget-gb 15 \
  --epochs 100 \
  --batch-size 2 \
  --grad-accumulation-steps 2 \
  --tta d4 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --base-channels 32 \
  --dropout 0.05 \
  --resume auto \
  --preview-every 5 \
  --num-preview-samples 2 \
  --seed 42 \
  --verbose \
  2>&1 | tee plugin_runs/arpil_plugin_terminal.log
```

---

## 2b) Reading the real/synthetic log

The run prints the mix twice. After the dataset is built:

```
--- Download summary ---
Frames requested : 48
Frames downloaded: 48
Frames failed    : 0
Download complete: yes
Real tiles built : 812
--- Dataset composition ---
Real-image download: COMPLETE (48/48 frames)
Train REAL          : 690 (69.9%, target 70.0%)
Train SYNTHETIC     : 297 (30.1%, target 30.0%)
Train total         : 987
Val real            : 122
```

and again right before epoch 1, which is the line that also shows up on a
resumed run that reused an existing dataset:

```
Training mix: 690 real (69.9%) + 297 synthetic (30.1%) = 987 samples
```

The same numbers are written to `<work-dir>/dataset/dataset_composition.json`.

### When the download does not finish

Frame downloads are ~570 MB each, so a long run will usually hit at least one
transient error. A failed frame no longer aborts the run: it is logged, recorded
in `<work-dir>/dataset/metadata/failed_frames.csv`, and skipped.

```
[17/48] FRAME FAILED 2011-03-02 12:00:00 -> EndpointConnectionError: ...
  continuing; this frame's share will be covered by synthetic samples
...
Download complete: NO (incomplete)
NOTE: training is starting on a PARTIAL real set. Synthetic samples were
generated from the real tiles that did download.
```

Because synthetic samples are blended from real tiles, the 70/30 ratio is
maintained against whatever real data *did* arrive — so training still starts,
just on a smaller real base. Resumed runs re-print the warning:

```
WARNING: real-image download was INCOMPLETE (31/48 frames).
```

To pick up the missing frames later, re-run the same command with
`--force-rebuild-dataset`. If *every* frame fails the run stops with an
explanatory error rather than training on synthetic-only data, since there would
be no real signal to blend from.

---

## 3) Fallback mode if ARPIL/core-SDO downloads are blocked

If Lightning can reach GitHub but not Hugging Face / AWS, use the SHARP fallback mode:

```bash
python3 solar_arpil_plugin.py \
  --source-mode sharp_sample \
  --work-dir plugin_runs/arpil_plugin \
  --real-ratio 0.70 \
  --synthetic-ratio 0.30 \
  --tile-size 512 \
  --tile-stride 256 \
  --cpu-threads 4 \
  --memory-budget-gb 15 \
  --epochs 100 \
  --batch-size 2 \
  --grad-accumulation-steps 2 \
  --tta d4 \
  --base-channels 32 \
  --dropout 0.05 \
  --resume auto \
  --preview-every 5 \
  --num-preview-samples 2 \
  --seed 42 \
  --verbose \
  2>&1 | tee plugin_runs/arpil_plugin_terminal.log
```

---

## 4) Optional manual environment setup

If you prefer to create the environment yourself:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Then run the same command as above.

---

## 5) What verbose mode logs

`--verbose` adds extra logging for:
- aligned frame counts,
- selected download targets,
- kept real tiles,
- synthetic tile creation,
- epoch timing,
- preview-save events.

Use it while testing and for long jobs.

---

## 6) Resume training

Resume behavior is controlled by:

```bash
--resume auto
```

Options:
- `auto` → resume if `last.pt` exists, otherwise start fresh
- `always` → require a checkpoint and resume from it
- `never` → start from scratch

Examples:

```bash
python3 solar_arpil_plugin.py --resume always
python3 solar_arpil_plugin.py --resume never
```

---

## 7) Preview image generation

The plugin saves preview images to show prediction progress on fixed validation tiles.

Relevant flags:

```bash
--preview-every 5
--num-preview-samples 2
```

Preview files are written to:

```bash
plugin_runs/arpil_plugin/run/previews/
```

---

## 8) Where outputs are saved

### Run logs and checkpoints
- `plugin_runs/arpil_plugin/run/console.log`
- `plugin_runs/arpil_plugin/run/metrics.jsonl`
- `plugin_runs/arpil_plugin/run/PROGRESS.md`
- `plugin_runs/arpil_plugin/run/best.pt`
- `plugin_runs/arpil_plugin/run/last.pt`
- `plugin_runs/arpil_plugin/run/run_config.json`
- `plugin_runs/arpil_plugin/run/previews/*.png`

### Dataset and provenance
- `plugin_runs/arpil_plugin/dataset/manifest.csv`
- `plugin_runs/arpil_plugin/dataset/sample_provenance.csv`
- `plugin_runs/arpil_plugin/dataset/sources_used.csv`
- `plugin_runs/arpil_plugin/dataset/tiles/*.h5`
- `plugin_runs/arpil_plugin/dataset/metadata/selected_frames.csv` (ARPIL/core-SDO mode)

---

## 9) No duplicate images guarantee

The plugin protects against duplicate dataset samples in several ways:

1. **Real tiles** use deterministic IDs based on frame timestamp and tile coordinates.
2. The manifest is audited before training.
3. Training stops if duplicate `sample_id`s are found.
4. **Synthetic samples** are built from unique real-tile pairs.
5. If the requested synthetic count exceeds the number of unique real-tile pairs, the plugin **caps** the synthetic count instead of repeating pairs.
6. If an existing dataset already exists and you do not pass `--force-rebuild-dataset`, the plugin reuses the existing manifest instead of rebuilding duplicate data.

Important note: training will still revisit the same dataset across epochs, which is normal machine-learning behavior. The guarantee here is about **dataset duplication**, not about never seeing a sample in later epochs.

---

## 10) Force rebuild vs reuse dataset

By default, if the dataset manifest already exists, the plugin will reuse it.
This helps avoid rebuilding and redownloading the same tiles.

To rebuild everything from scratch:

```bash
python3 solar_arpil_plugin.py --force-rebuild-dataset
```

---

## 11) Real-data smoke test before a big run

Quick test command:

```bash
python3 solar_arpil_plugin.py \
  --source-mode sharp_sample \
  --work-dir plugin_runs/smoke \
  --epochs 2 \
  --tile-size 192 \
  --tile-stride 128 \
  --batch-size 2 \
  --grad-accumulation-steps 1 \
  --base-channels 16 \
  --cpu-threads 2 \
  --preview-every 1 \
  --num-preview-samples 1 \
  --verbose
```

---

## 12) Audit result

Every non-README download URL in the repo was re-verified against the live hosts.

| Endpoint | Verdict |
| --- | --- |
| `core-sdo/resolve/main/{train,val,test}.csv` | correct, present upstream |
| ARPIL mask splits `{train,validation,test,leaky_validation}.csv` | correct, present upstream |
| ARPIL per-frame `resolve/main/data/<y>/<m>/<stamp>.h5` | **broken — fixed** (see below) |
| S3 `nasa-surya-bench` unsigned, key `<YYYY>/<MM>/<file>.nc` | bucket confirmed via AWS Open Data; key layout not reachable from the audit sandbox — validate with the `list_objects_v2` probe in section 1b |
| `github.com/mbobra/SHARPs.git` | HTTP 200 |
| `zenodo.org/records/7950721` (UAD helper script) | not reachable from the audit sandbox; unused by the main pipeline |

**Bug found and fixed.** Both `solar_arpil_plugin.py` and
`scripts/build_arpil_3ch_tiles.py` built mask URLs as
`<repo>/resolve/main/` + the CSV `file_path`. The AR-segmentation repo root
contains only `.gitattributes`, `README.md`, `data.tar.gz`, and the four CSVs —
there is no `data/` tree — so *every* mask fetch returned 404 and no
`arpil_sdo` run could ever have produced a real dataset. Both call sites now
download `data.tar.gz` once, index it in a single streaming pass, and extract
members on demand. `tests/test_mask_archive.py` covers indexing, extraction,
caching, `.part` cleanup, missing-member errors, and download-skip behaviour.

**Formats confirmed against the upstream dataset cards:** ARPIL masks are HDF5
`[2, 4096, 4096]` at 1-hour cadence (May 2010 - Dec 2024); core-SDO frames are
netCDF float32 `[13, 4096, 4096]` at 12-minute cadence. The plugin's default
channels `aia171 aia193 hmi_m` are all in the 13-variable set.

Verification run on this branch:

```
python -m pytest tests/ -q                      # 108 passed
python -m pyflakes $(git ls-files '*.py')       # clean
python3 solar_arpil_plugin.py --source-mode sharp_sample ... --epochs 1
                                                # exit 0, val_dice=0.4801
```
