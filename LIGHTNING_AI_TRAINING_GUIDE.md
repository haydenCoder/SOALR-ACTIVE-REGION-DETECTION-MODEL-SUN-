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

```bash
git clone https://github.com/haydenCoder/SOALR-ACTIVE-REGION-DETECTION-MODEL-SUN-.git
cd SOALR-ACTIVE-REGION-DETECTION-MODEL-SUN-
git fetch origin arena/01a02a48-soalr-active-region-detection
git checkout arena/01a02a48-soalr-active-region-detection
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
  --epochs 100 \
  --batch-size 2 \
  --grad-accumulation-steps 2 \
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
  --epochs 100 \
  --batch-size 2 \
  --grad-accumulation-steps 2 \
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

The plugin was audited with:
- Python syntax compilation,
- dependency bootstrap check,
- end-to-end smoke test in `sharp_sample` mode,
- duplicate-ID audit,
- resume/log/progress/preview/provenance file generation checks.

Smoke test confirmed:
- dataset built successfully,
- provenance files written,
- previews saved,
- checkpoints saved,
- progress file updated,
- no duplicate sample IDs.
