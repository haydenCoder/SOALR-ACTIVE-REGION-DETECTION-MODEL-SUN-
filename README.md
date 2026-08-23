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
