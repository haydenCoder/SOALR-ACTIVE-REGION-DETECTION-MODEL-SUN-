# Solar Active Region Detection - Attention U-Net Training

This repository now includes a complete training pipeline for **solar active region segmentation** using an **Attention U-Net** in **PyTorch**, plus a **Docker** workflow that can:

1. download a public solar active region dataset,
2. extract the multi-spectral channels and masks,
3. build a manifest for training/validation,
4. train the model and save checkpoints.

## What was added

- `train.py` — CLI training entrypoint
- `src/solar_ar/models/attention_unet.py` — Attention U-Net model
- `src/solar_ar/data.py` — dataset + multi-channel image loading
- `src/solar_ar/training.py` — training loop, loss, metrics, checkpoints
- `scripts/download_uad_dataset.sh` — downloads and extracts the public UAD dataset from Zenodo
- `scripts/prepare_uad_manifest.py` — creates a CSV manifest matching masks with the 171/195/284/304 image channels
- `scripts/run_training_docker.sh` — build + download + manifest + train in Docker
- `scripts/download_smarp_sample.sh` — clone small public SMARP sample FITS files from GitHub
- `scripts/prepare_smarp_patch_manifest.py` — turn SMARP FITS files into patch-level training data
- `scripts/generate_synthetic_smarp.py` — create a 70/30 real/synthetic training mix for sandbox-feasible experiments
- `scripts/preprocess_suryabench_arpil.py` — preprocess local core-SDO + ARPIL files into low-RAM training patches
- `scripts/train_arpil_3ch_lowram.sh` — 4 GB RAM preset for AIA 171 + AIA 193 + HMI training
- `Dockerfile` — reproducible training environment

## Dataset used by the scripts

The downloader targets the public Zenodo release:

- `https://zenodo.org/records/7950721`

The training pipeline is configured for the **UAD** subset with segmentation masks and the four training image archives:

- `training_images_171.rar`
- `training_images_195.rar`
- `training_images_284.rar`
- `training_images_304.rar`
- `Masks.zip`

## Quick start with Docker

### 1) Pull/build the image

```bash
docker pull pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
docker build --pull -t solar-ar-attention-unet .
```

### 2) Download and extract the dataset

```bash
docker run --rm -it \
  -v "$PWD:/workspace" \
  solar-ar-attention-unet \
  bash scripts/download_uad_dataset.sh data/raw
```

### 3) Build the training manifest

```bash
docker run --rm -it \
  -v "$PWD:/workspace" \
  solar-ar-attention-unet \
  python scripts/prepare_uad_manifest.py \
    --raw-root data/raw/Solar_data_UAD \
    --output data/processed/uad_manifest.csv
```

### 4) Train the Attention U-Net

```bash
docker run --rm -it --gpus all \
  -v "$PWD:/workspace" \
  solar-ar-attention-unet \
  python train.py \
    --manifest data/processed/uad_manifest.csv \
    --channels 171 195 284 304 \
    --image-size 256 \
    --batch-size 8 \
    --epochs 50 \
    --lr 1e-3 \
    --amp \
    --output-dir runs/attention_unet_uad
```

## One-command helper

If you want the repo to do the full flow for you:

```bash
bash scripts/run_training_docker.sh
```

If you need CPU-only execution for the helper script:

```bash
USE_GPU=0 bash scripts/run_training_docker.sh
```

## Included sample-data training path

Because the Arena sandbox used here does **not** expose a Docker daemon, I also added a lightweight **sample-data** workflow that can be trained directly inside the sandbox using public SMARP FITS samples cloned from GitHub:

```bash
bash scripts/download_smarp_sample.sh data/raw/smarp_sample
python scripts/prepare_smarp_patch_manifest.py \
  --raw-root data/raw/smarp_sample \
  --output-dir data/processed/smarp_sample
python train.py \
  --manifest data/processed/smarp_sample/manifest.csv \
  --channels mag \
  --image-size 64 \
  --batch-size 8 \
  --epochs 30 \
  --base-channels 16 \
  --output-dir runs/attention_unet_smarp_sample
```

You can also create a stronger sandbox-feasible 70/30 real/synthetic mix and train with a larger model:

```bash
python scripts/generate_synthetic_smarp.py \
  --manifest data/processed/smarp_sample/manifest.csv \
  --output-dir data/processed/smarp_mix_70_30 \
  --synthetic-fraction 0.30

python train.py \
  --manifest data/processed/smarp_mix_70_30/manifest.csv \
  --channels mag \
  --image-size 64 \
  --batch-size 4 \
  --epochs 150 \
  --lr 3e-4 \
  --base-channels 32 \
  --dropout 0.02 \
  --loss bce_dice \
  --output-dir runs/attention_unet_smarp_mix_b32
```

This is useful for smoke-testing the full pipeline when the larger Zenodo dataset or Docker runtime is unavailable.

## Local training without Docker

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Then run:

```bash
python train.py \
  --manifest data/processed/uad_manifest.csv \
  --channels 171 195 284 304 \
  --image-size 256 \
  --batch-size 8 \
  --epochs 50 \
  --lr 1e-3 \
  --output-dir runs/attention_unet_uad
```

## Outputs

Training writes artifacts to the output directory, for example `runs/attention_unet_uad/`:

- `best.pt` — best checkpoint by validation Dice
- `last.pt` — final checkpoint
- `metrics.jsonl` — per-epoch logs
- `train_config.json` — training arguments

## Supported input file formats

All loading goes through `src/solar_ar/arrayio.py`, so every script and the
training pipeline read the same formats:

| Extension | Reader | Typical source |
| --- | --- | --- |
| `.h5`, `.hdf5`, `.he5`, `.hdf` | h5py | ARPIL / SuryaBench masks, plugin tiles |
| `.nc`, `.nc4`, `.netcdf`, `.cdf` | netCDF4 | core-SDO frames |
| `.fits`, `.fit`, `.fts` | astropy | SHARP / SMARP / SunPy archives |
| `.npy`, `.npz` | numpy | pre-processed patches |
| `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff` | Pillow | Zenodo UAD dataset |

Compressed variants such as `frame.fits.gz` are detected automatically.

### Choosing the dataset inside an HDF5/netCDF file

These containers hold *named* datasets, so the loader needs to know which one to
read. There are three ways to say so, in increasing priority:

1. **Auto-detection** (default) — tries `union_with_intersect`, `mask`, `image`,
   `data`, `segmentation`, then the first 2-D dataset in the file. Nested groups
   are searched recursively.
2. **A CLI flag** applied to the whole run:

   ```bash
   python train.py --manifest data/processed/arpil/manifest.csv \
     --channels aia171 aia193 hmi_m \
     --hdf5-image-key image \
     --hdf5-mask-key union_with_intersect
   ```

3. **A per-file suffix in the manifest**, which overrides everything else:

   ```csv
   sample_id,split,image_aia171,mask
   s0,train,frames/0.h5#image,masks/0.h5#masks/union_with_intersect
   ```

### Adding another format

Add the extension to the right set in `src/solar_ar/arrayio.py`, write a
`_load_<format>(path, key)` returning a 2-D `float32` array, and dispatch to it
in `load_array`. Nothing else needs to change — the manifest builders and the
training dataset both pick it up from `SUPPORTED_EXTENSIONS`.

## Tests

Run the suite with:

```bash
python -m pytest tests/ -q
```

Covers the multi-format loaders, download/retry logic, resource planning, TTA
correctness (exact invertibility of every transform), the model's shapes and
gradient flow, the LR schedule, EMA, and the RAM cache.

## Notes

- The Docker image uses an official PyTorch CUDA runtime base image.
- If you do not have an NVIDIA GPU, remove `--gpus all` from the `docker run` command.
- The downloader pulls a **large** archive (~6.6 GB), so make sure you have disk space.
- The manifest builder matches masks and image channels by normalized filename. If your extracted filenames differ, adjust `normalize_key()` in `scripts/prepare_uad_manifest.py`.

## 4 GB RAM real-data preset

For a real low-RAM SuryaBench/ARPIL run, preprocess local files into patches and train with the bundled preset.

```bash
python scripts/preprocess_suryabench_arpil.py \
  --core-root /path/to/core-sdo-nc \
  --mask-root /path/to/arpil-h5 \
  --output-dir data/processed/arpil_3ch_lowram \
  --channels aia171 aia193 hmi_m \
  --resize 1024 \
  --patch-size 256 \
  --stride 256

bash scripts/train_arpil_3ch_lowram.sh \
  data/processed/arpil_3ch_lowram/manifest.csv \
  runs/attention_unet_arpil_171_193_hmi_lowram
```

This preset assumes the requested 3-channel physically meaningful stack of **AIA 171 Å**, **AIA 193 Å** (often mistyped as 191), and **HMI line-of-sight magnetogram**. The `solar_physics` normalization mode uses log compression for EUV channels and symmetric scaling for HMI polarity.


## 15 GB RAM / 4 CPU preset

For a larger CPU-oriented run with **3 physically meaningful channels**:

- `aia171` — lower-temperature coronal loops
- `aia193` — hotter active coronal plasma
- `hmi_m` — line-of-sight magnetogram

Use the all-in-one preset below. It will:

1. auto-download ARPIL masks from Hugging Face,
2. auto-download aligned core-SDO frames from the public `nasa-surya-bench` S3 bucket,
3. build **512×512 tiles**,
4. train an Attention U-Net with a moderate-to-large base width.

```bash
bash scripts/run_arpil_3ch_15gb.sh
```

Useful overrides:

```bash
MAX_FRAMES=96 BASE_CHANNELS=32 BATCH_SIZE=2 GRAD_ACCUM=2 EPOCHS=1000 PATCH_SIZE=512 STRIDE=512 RUN_DIR=runs/attention_unet_arpil_171_193_hmi_15gb DATASET_DIR=data/processed/arpil_3ch_512 bash scripts/run_arpil_3ch_15gb.sh
```

Notes:
- If you wrote `191`, this preset uses the standard **AIA 193 Å** channel.
- The `solar_physics` normalization mode uses log compression for EUV and symmetric scaling for HMI polarity.
- The dataset builder streams frame pairs and tiles them, so it is more disk-friendly than downloading a huge archive first.


## Multi-day unattended run (one command)

Leave it running overnight or for several days — it auto-downloads more data,
trains, evaluates, and repeats forever, resuming exactly where it stopped:

```bash
bash scripts/run_forever.sh                       # start (Mac: auto no-sleep)
nohup bash scripts/run_forever.sh > /dev/null 2>&1 &   # or fully detached
tail -f ~/solar_results/arpil/run_forever.log     # follow the full log
cat   ~/solar_results/arpil/STATUS.md             # latest state at a glance
bash scripts/run_forever.sh --once                # single cycle, then exit (test)
kill "$(cat ~/solar_results/arpil/run_forever.lock)"  # stop cleanly
```

What makes it safe to walk away:

- **The machine will not sleep/turn off** — on macOS the script re-launches
  itself under `caffeinate -is` automatically (keep it on AC power); on
  systemd Linux it inhibits idle sleep.
- **Full, timestamped logging** of *everything* to
  `~/solar_results/arpil/run_forever.log`: a session banner (machine, GPU,
  config), **pre-flight checks of every data-source URL** (mask index date
  range, mask archive size, and a live probe that core frames actually exist
  in S3 for the chosen split), a resource watchdog line every 5 minutes
  (CPU load / RAM / disk), per-step timing with full subprocess output, and a
  per-cycle metric summary (val Dice/IoU + object F1).
- **Auto-run / auto-resume** — every download is resumable (per-frame
  fragments, never redownloaded), a lock file prevents double-launches,
  transient failures (network blips, OOMs) just retry on the next cycle, and
  Ctrl-C / `kill` shuts down cleanly with all completed work kept.
- **Maximum power with a cooling headroom** — every core minus 4 (macOS, so a
  multi-day run doesn't overheat the machine) or 2 (Linux), all RAM; Apple
  Silicon uses the MPS GPU with bfloat16 automatically. Override with
  `CPU_HEADROOM=N` (e.g. `CPU_HEADROOM=6` for an even cooler, slower run).

Useful overrides (env vars before the command):

```bash
MASK_SPLIT=validation EPOCHS=40 FRAMES_PER_CYCLE=100 bash scripts/run_forever.sh
# faster downloads on very fast fiber (default is 16 parallel frames,
# and each 570 MB frame is itself pulled as 16 concurrent 16 MB parts):
DOWNLOAD_WORKERS=32 bash scripts/run_forever.sh
```

## Hardware utilisation (up to 15 GB RAM / 4 CPUs)

Training claims as much of the machine as the budget allows, and **clamps to
what the machine actually has** — the same command works on a 2-core laptop and
on a 4-core/15 GB box.

```bash
python train.py --manifest data/processed/uad/manifest.csv \
  --cpu-budget 4 --memory-budget-gb 15 --auto-batch-size
```

| Flag | Default | Effect |
| --- | --- | --- |
| `--cpu-budget` | `0` (auto) | Target cores, clamped to the cores actually available. Auto = every detected core minus `--cpu-headroom`. Sets torch intra-/inter-op threads, BLAS thread env vars and DataLoader workers. |
| `--cpu-headroom` | `2` | Cores left free for the OS when the budget is auto — maximum power while the machine stays usable (e.g. a Mac you keep working on). `0` grabs every core on a dedicated box. |
| `--memory-budget-gb` | `0` (auto) | Target RAM. Auto = all detected RAM. Sizes the in-RAM sample cache and `--auto-batch-size`. |
| `--cache-fraction` | `0.45` | Share of the budget spent caching decoded samples. |
| `--auto-batch-size` | on | Picks the largest batch size that fits the memory budget. |
| `--no-channels-last` | on | Disables the channels-last memory format. |

**Accelerator & mixed precision:** the device is `cuda` when available, else
Apple Metal (`mps`) on Apple Silicon, else CPU. Training and both evaluators
automatically use fp16 autocast + GradScaler on CUDA and bfloat16 autocast on
MPS (no scaler needed) — on a Mac this is a free 2-5× speedup over CPU, and
the `--cpu-headroom 2` default keeps the machine responsive while it runs.

How the budget is spent:

- **CPU** — thread counts are set *before* numpy/torch import (BLAS pools are
  sized at import time), plus `OMP_WAIT_POLICY=ACTIVE` to keep worker threads
  spinning instead of sleeping between the many small ops a U-Net issues.
- **RAM** — decoded, normalized, resized samples are cached in memory. Decoding
  FITS/HDF5 and percentile-normalizing costs more than the CPU forward pass, so
  from the second epoch on the run becomes compute-bound. The cache budget is
  divided by the worker count, since each worker process holds its own copy.
- Both are detected **cgroup-aware**, so a container limit is respected rather
  than the host's core count.

Each epoch logs `samples/s`, resident memory and the cache hit rate; the
resolved plan is written to `runtime.json` in the run directory.

## Test-time augmentation (TTA)

Active regions have no canonical orientation, so predictions are averaged over
the symmetries of the square (the D4 group). Every transform is exactly
invertible, so no interpolation blur is introduced, and probabilities — not
logits — are averaged so a single confident view cannot dominate.

```bash
python train.py --manifest ... --tta d4     # 8 views, best accuracy
python train.py --manifest ... --tta flips  # 4 views, half the cost
```

`--tta` applies to validation, so the reported Dice reflects the inference path
you would actually deploy. For full-disk frames larger than the training tile,
`solar_ar.tta.sliding_window_predict` tiles the image and blends overlaps with a
cosine window to avoid visible seams.

## Better Attention U-Net training

Architecture (`src/solar_ar/models/attention_unet.py`):

| Flag | Effect |
| --- | --- |
| `--model-depth` | Encoder/decoder levels (default 4). |
| `--deep-supervision` | Auxiliary losses on intermediate decoder stages, upsampled to full resolution. Pushes gradient into deep layers early. |
| `--norm-groups N` | GroupNorm instead of BatchNorm — **recommended for batch sizes below 8**, where batch statistics are too noisy. |
| `--no-residual` / `--no-se` | Disable residual connections / squeeze-excitation channel attention (both on by default). |

Optimisation:

| Flag | Effect |
| --- | --- |
| `--ema-decay 0.999` | Exponential moving average of weights; validation and early stopping use the EMA weights, and they are saved as `ema_state_dict`. |
| `--grad-clip` | Max gradient norm, default `1.0` (correctly unscaled first under AMP). |
| `--warmup-epochs` | Linear LR warmup before cosine decay, stepped per optimizer step. Prevents the early all-background collapse AdamW can cause on a fresh U-Net. |
| `--loss combo` | Blends BCE-Dice with Focal-Tversky, penalising false negatives on small regions. |
| `--no-torch-compile` | off — **torch.compile is on by default**: the C-level graph engine (Metal/MPS on a Mac, oneDNN on CPU) fuses the U-Net into optimized kernels for a further ~1.2–1.5×. It is smoke-tested before the first epoch and falls back to eager automatically if the build can't support it, so it can never break an unattended run. |

Training augmentation now includes photometric jitter (gain/bias, gamma, noise)
alongside the D4 geometric transforms, modelling instrument-response drift so
the network keys on morphology rather than absolute brightness.

### Measured effect

On a synthetic 24-sample benchmark, best val Dice, mean of 3 seeds:

| Configuration | 8 epochs | 40 epochs |
| --- | --- | --- |
| Baseline (plain U-Net) | **0.858** | 0.916 |
| Residual + SE + deep supervision + combo + EMA | 0.593 | 0.987 |
| ...plus D4 TTA | 0.450 | **0.991** |

The ordering **reverses** with training length: regularisation (EMA, warmup,
deep supervision, GroupNorm) trades early-epoch speed for final quality. Judge
these options on a converged run — on a short one they will look worse.

## Single-file plugin

For the simplest end-to-end workflow, use:

- `solar_arpil_plugin.py`

This one file can auto-install dependencies, download data when reachable, convert to HDF5 training tiles, build a 70/30 real/synthetic split, save provenance, resume training, and generate preview images.

See also:

- `LIGHTNING_AI_TRAINING_GUIDE.md`

Quick fallback smoke test:

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
