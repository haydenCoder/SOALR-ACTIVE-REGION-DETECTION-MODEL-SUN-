# Walkthrough: how this code works, and why

A plain-language essay for the codebase: what it does, **why this dataset**
was chosen, **what counts as the official score**, **why the old Zenodo (UAD)
dataset scored 0.9 easily** while this one is harder, and how to actually
operate the system day to day.

---

## 1. What this project is

A model that looks at a full-disk image of the Sun (13 physics channels from
the SDO satellite: 8 EUV wavelengths + 5 magnetic-field components) and
paints a mask over every **active region** — the magnetically complex areas
where flares and coronal mass ejections originate. The network is an
Attention U-Net (encoder-decoder with skip connections + attention gates),
trained end-to-end from raw satellite data to segmentation masks.

The end product is a single file, `best.pt` (~400 MB): load it, feed it a
solar image, get a mask of the active regions.

---

## 2. Why *this* dataset (surya-bench ARPIL), and why the old Zenodo one was "easy"

### The current dataset

The run downloads from two public NASA sources:

- **Images**: `s3://nasa-surya-bench` — real SDO/AIA+HMI full-disk frames,
  2010–2019 (train) and 2020–2024 (official test), ~570 MB per frame.
- **Masks**: the `surya-bench-ar-segmentation` benchmark (Hugging Face) —
  active-region segmentation masks produced by the published **ARPIL**
  method, one `data.tar.gz` plus per-split CSV indices
  (`train.csv`, `validation.csv`, `test.csv`, `leaky_validation.csv`).

Why this one:

1. **It is a benchmark, not just a dataset.** There is an official, held-out
   test split (2020–2024) that no training run sees. That makes our score
   *paper-comparable* — the number means something to other people, not just
   to us.
2. **Physically complete input.** All 13 SDO channels: active regions are
   defined by magnetic field (HMI) *and* hot plasma (AIA), so the model gets
   the full physical picture instead of one wavelength.
3. **Large and real.** ~70k available frames of actual satellite data, not
   synthetic or heavily preprocessed crops.
4. **The masks are fine-grained.** ARPIL masks include the thin internal
   structure (polarity inversion lines, filament channels), not just a blob
   outline. That is what makes this task *hard* — and scientifically useful.

### Why the old Zenodo dataset (UAD) gave you 0.9 easily

The original dataset in this repo is the **UAD** release
(Zenodo record 7950721: "MLMT-CNN Solar Active Regions — Bounding Box and
Segmentation Annotations for Deep-Learning Application"), used via
`scripts/download_uad_dataset.sh` at **256×256** with 4 EUV channels
(171/195/284/304) for 50 epochs. Your 0.9 came from that setup. The gap to
the current ~0.74 is **not** that the model got worse — the tasks are
different, and several things about UAD inflate scores:

1. **Coarser, box-based masks.** UAD is a bounding-box + segmentation
   annotation set built for DL applications — region-level, blob-like masks.
   Dice is a pixel-overlap metric: a blob mask is easy to overlap. ARPIL
   masks add the thin internal structure where partial/blurry predictions
   lose real pixel overlap. Same model quality, lower metric.
2. **Small images, AR-centric framing.** 256×256 images (typically around
   the region of interest) let the active region fill a large fraction of
   the frame — a model that "paints where the bright stuff is" scores very
   well. Our 512×512 full-disk patches contain small, sparse structures on a
   mostly quiet disk, where the background has to be *right* too.
3. **50 full epochs on a small set = heavy repetition + likely temporal
   correlation.** Each UAD image was seen ~50 times, and its validation set
   comes from the same solar period as its training set — the Sun changes
   slowly, so test frames closely resemble training frames (the benchmark
   here even ships a split literally named `leaky_validation` to acknowledge
   that temporally-correlated splits leak). Our design deliberately does the
   opposite: each rolling frame is seen only ~2–4 times, and the val set is
   held out at the frame level.
4. **Different metric mix** (bbox IoU + segmentation). A 0.9 in one metric
   on one dataset is not comparable to a 0.74 Dice on a finer task.

**Bottom line:** don't compare the two numbers directly. The fair, honest
number for this project is the **official test-split score** (section 4).

---

## 3. The pipeline, end to end

```
 S3 (SDO frames)          Hugging Face (ARPIL masks)
        \                          /
         \                        /
   scripts/build_arpil_resumable.py        <- background downloader
     570 MB .nc frame  -->  512x512 npz tiles + manifest.csv
                              (13 channel tiles + 1 mask per tile)
                                        |
   scripts/train_streaming.py  <--------+     <- foreground trainer
     continuous training, rolling window, EMA, checkpoints (best.pt/last.pt)
                                        |
   evaluate.py / evaluate_detection.py <+     <- official scoring
```

- **`scripts/run_forever.sh`** — the conductor. One command. It starts the
  background downloader and the foreground trainer, watches resources, logs
  everything, and (new) *supervises* the trainer: if it dies for any reason
  it is restarted from `last.pt` automatically.
- **`scripts/build_arpil_resumable.py`** — turns raw frames into training
  tiles. Each source frame is a 570 MB netCDF; it is cut into 512×512 tiles
  (13 channel tiles + 1 mask tile per tile, saved as compressed `.npz`),
  recorded in a per-frame manifest fragment, and the raw file is deleted.
  Completed frame names are recorded **permanently** (retired frames included),
  so nothing is ever downloaded twice. In `ROLLING` mode it keeps a rotating
  window of the newest frames on a small disk.
- **`scripts/train_streaming.py`** — the trainer. Trains continuously on a
  growing/rotating dataset: 200-tile epochs, learning-rate warmup, EMA
  (exponential moving average of the weights — the model that gets saved is
  a smoothed, more stable version of the raw weights), a quick validation
  every 10 epochs, `best.pt` saved whenever val Dice improves, `last.pt`
  every epoch (the resume point), and a memory self-recycle so a RAM creep
  can never kill the run.
- **`src/solar_ar/`** — the library: `models/attention_unet.py` (the
  network), `data.py` (dataset: loads tiles, normalizes per-channel,
  augments, survives missing tiles), `training.py` (losses: BCE+Dice /
  focal-Tversky / combo; EMA; metrics), `runtime.py` (GPU/CPU detection,
  thread planning, batch-size auto-fit), `tta.py` (test-time augmentation:
  flip/rotation ensemble at inference).
- **`evaluate.py`** — pixel-level official scoring (Dice/IoU) on any
  manifest split. **`evaluate_detection.py`** — object-level official
  scoring (connected components → bounding boxes → precision/recall/F1 and
  AP at IoU ≥ 0.5): does the model find the *regions*, not just the pixels?

### The knobs (environment variables on `run_forever.sh`)

| Knob | Meaning |
|---|---|
| `CHANNELS` | which of the 13 channels the model sees |
| `BASE_CHANNELS` | model width (16/32/48/64) — quality vs speed |
| `DEEP_SUPERVISION` | 1 = auxiliary losses on decoder stages (helps thin structure) |
| `ROLLING` / `ROLLING_WINDOW` / `ROLLING_MAX_TILE_GB` | rotating window for small disks |
| `ROLLING_MIN_LIFETIME_HOURS` | each frame must train ~2-4 passes before it may leave |
| `LR` | learning rate (lower = less "overwriting" in a rolling run) |
| `MAX_TOTAL_FRAMES` / `MIN_FREE_GB` | static-mode dataset size / disk safety |
| `TILES_PER_EPOCH` / `VAL_EPOCH` / `VAL_SUBSET` | epoch size / validation cadence / quick-val size |
| `DOWNLOAD_WORKERS` | parallel frame downloads (each file = 12 parallel parts) |
| `CPU_HEADROOM` | cores left idle for the OS (0 = maximum power) |

---

## 4. Scoring: what is official and what is a proxy

**`Best val Dice` (what STATUS.md shows) is a training-time proxy, not the
official number.** It is pixel Dice measured every 10 epochs on a 300-tile
random slice of the held-out val frames, *without* test-time augmentation.
It is deliberately fast (1–2 min) so it can run constantly. It's a good
compass — it tracks real quality — but it is not the number you report.

**The official score** is what `evaluate.py` + `evaluate_detection.py`
produce on the **official 2020–2024 test split** (frames the model has never
seen, from solar years outside training):

- **Pixel metrics** (evaluate.py): `dice` and `iou` — pixel-overlap quality
  of the mask. Dice is the headline number for segmentation.
- **Object metrics** (evaluate_detection.py): the predicted mask is
  converted to connected components (bounding boxes), matched to the truth
  boxes at IoU ≥ 0.5, giving `precision`, `recall`, `f1` and
  `ap_iou_0.50` (average precision) — i.e., *does the model locate the
  active regions as objects*, the way a forecaster would use it.

`run_forever.sh` has this built in: set `EVAL_TEST_SPLIT=1` (and
`TEST_FRAMES=200`) and it builds the official test tiles and runs both
evaluations on `best.pt` every cycle, writing `test_pixel_metrics.json` and
`test_detection_metrics.json`. Expect the official number to sit *slightly
above* the quick val bar (TTA + a stable test set), and that is the number
to compare against published surya-bench results.

---

## 5. Why it is built the way it is

- **Streaming instead of batched epochs.** Data arrives slowly (570 MB per
  frame over the internet); training on 200 tiles at a time from frame #1
  wastes zero time, and optimizer/EMA/LR state accumulates across the whole
  run — one model that only gets better, instead of a throwaway model per
  download batch.
- **Frame-level val split.** All tiles from a source frame go to the same
  split, so a val frame can never share pixels with training — no
  within-frame leakage.
- **Rolling window + permanent name record.** On a small disk (Kaggle 20 GB)
  you cannot keep 2000 frames. So: train on the newest window, but *remember
  every frame name forever* (no re-downloads) and require each frame to earn
  ~2–4 training passes before it leaves. Diversity without storage.
- **float16 tile storage.** The pipeline re-normalizes every tile at load
  time, so half-precision storage is lossless for training and halves the
  disk footprint (the difference between fitting 200 or 400 frames on Kaggle).
- **EMA.** The saved model is a running average of the weights — it smooths
  out the last few chaotic epochs and is reliably worth a fraction of a
  Dice point, and it makes val scores less jumpy.
- **Self-healing.** Cloud notebooks die (12 h caps, OOM, restarts). So:
  every epoch saves `last.pt`; the trainer self-recycles cleanly before an
  OOM; the run script supervises and restarts the trainer automatically;
  every frame name is a permanent record. Re-running one cell is always
  "resume", never "start over".

---

## 6. Operating manual (the only two commands you need)

**Kaggle (T4×2):** one cell (see `kaggle/start_kaggle.ipynb`). Stop/12 h
cap → re-run the same cell → it resumes from `last.pt`. Insurance: at ~1 h
and before the 12 h cap, click **Save Version** and download `last.pt` from
the Output tab.

**Mac (MPS, max power):** the one-liner in the chat history — kill any old
run, `git pull`, and start `scripts/run_forever.sh` with the 13-channel
max-power environment.

**Reading health:** `cat .../solar_results/arpil/STATUS.md` (epoch,
dataset size, best val Dice), `tail -f .../run_forever.log` (everything),
and the `[rolling] ... frames on disk` lines (window rotating as designed).
