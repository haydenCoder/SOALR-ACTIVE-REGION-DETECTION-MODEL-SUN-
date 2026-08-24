from __future__ import annotations

import json
import os
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from solar_ar.data import SolarActiveRegionDataset
from solar_ar.models import AttentionUNet
from solar_ar.runtime import configure_runtime, process_memory_gb
from solar_ar.tta import tta_predict


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_dice: float
    val_iou: float
    learning_rate: float
    seconds: float
    samples_per_second: float = 0.0
    process_memory_gb: float = 0.0
    cache_hit_rate: float = 0.0


class ModelEma:
    """Exponential moving average of the model weights.

    The EMA weights sit closer to the centre of the loss basin than the final
    SGD iterate, which on noisy segmentation data is reliably worth a fraction
    of a Dice point and makes epoch-to-epoch validation far less jumpy. The
    decay is warmed up so the average is not anchored to the random init.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.updates = 0
        self.shadow = {
            name: param.detach().clone().float()
            for name, param in model.state_dict().items()
            if param.dtype.is_floating_point
        }
        self.buffers = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if not param.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        # Ramp the decay in: early on, trust recent weights more.
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for name, param in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].mul_(decay).add_(param.detach().float(), alpha=1 - decay)
            else:
                self.buffers[name] = param.detach().clone()

    def state_dict(self) -> dict:
        return {**{k: v.clone() for k, v in self.shadow.items()}, **self.buffers}

    def copy_to(self, model: nn.Module) -> dict:
        """Load EMA weights into ``model``, returning the previous state."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.state_dict(), strict=False)
        return backup


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        numerator = 2 * (probs * targets).sum(dim=(1, 2, 3)) + 1e-6
        denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-6
        dice_loss = 1 - (numerator / denominator).mean()
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        true_pos = (probs * targets).sum(dim=(1, 2, 3))
        false_neg = ((1 - probs) * targets).sum(dim=(1, 2, 3))
        false_pos = (probs * (1 - targets)).sum(dim=(1, 2, 3))
        tversky = (true_pos + 1e-6) / (true_pos + self.alpha * false_neg + self.beta * false_pos + 1e-6)
        return ((1 - tversky) ** self.gamma).mean()


class ComboLoss(nn.Module):
    """BCE-Dice plus Focal-Tversky.

    BCE-Dice gives well-calibrated probabilities and a stable gradient; the
    Tversky term (alpha < beta) explicitly penalises false negatives, which
    counteracts the strong background bias of active-region masks. Combining
    them is more robust across datasets than either alone.
    """

    def __init__(self, tversky_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_dice = BCEDiceLoss()
        self.focal_tversky = FocalTverskyLoss()
        self.tversky_weight = tversky_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (1 - self.tversky_weight) * self.bce_dice(logits, targets) + (
            self.tversky_weight * self.focal_tversky(logits, targets)
        )


class DeepSupervisionLoss(nn.Module):
    """Applies a base loss to the main output and every auxiliary head.

    Auxiliary heads get geometrically decaying weights, so the final head
    dominates the objective while the deeper stages still receive direct
    gradient. Accepts a bare tensor too, so the same criterion works whether or
    not deep supervision is active.
    """

    def __init__(self, base_loss: nn.Module, weight_decay_factor: float = 0.5) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.weight_decay_factor = weight_decay_factor

    def forward(self, outputs: torch.Tensor | list[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return self.base_loss(outputs, targets)

        total = torch.zeros((), device=targets.device, dtype=torch.float32)
        total_weight = 0.0
        for index, output in enumerate(outputs):
            weight = self.weight_decay_factor ** index
            total = total + weight * self.base_loss(output, targets)
            total_weight += weight
        return total / total_weight


def build_loss(loss_name: str, deep_supervision: bool = False) -> nn.Module:
    if loss_name == "focal_tversky":
        base: nn.Module = FocalTverskyLoss()
    elif loss_name == "bce_dice":
        base = BCEDiceLoss()
    elif loss_name == "combo":
        base = ComboLoss()
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    return DeepSupervisionLoss(base) if deep_supervision else base


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
    learning_rate: float,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine decay with a linear warmup, stepped per optimizer step.

    Warmup matters here because AdamW's second-moment estimate is unreliable for
    the first few hundred steps; starting at the full LR on a freshly
    initialised U-Net often collapses the mask to all-background, from which
    the Dice term cannot recover.
    """
    total_steps = max(1, epochs * max(1, steps_per_epoch))
    warmup_steps = min(max(0, warmup_epochs) * max(1, steps_per_epoch), max(0, total_steps - 1))
    minimum_factor = 0.05

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return minimum_factor + (1.0 - minimum_factor) * (step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_factor + (1.0 - minimum_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def compute_segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    return compute_metrics_from_probs(torch.sigmoid(logits), targets)


@torch.no_grad()
def compute_metrics_from_probs(
    probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> tuple[float, float]:
    """Dice and IoU from probabilities (so TTA output can be scored directly)."""
    preds = (probs > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    pred_area = preds.sum(dim=(1, 2, 3))
    target_area = targets.sum(dim=(1, 2, 3))
    union = pred_area + target_area - intersection

    dice = ((2 * intersection + 1e-6) / (pred_area + target_area + 1e-6)).mean().item()
    iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
    return dice, iou


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(
        self,
        manifest_path: str | Path,
        channels: list[str],
        image_size: int,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        num_workers: int,
        output_dir: str | Path,
        base_channels: int,
        dropout: float,
        normalize_mode: str,
        amp: bool,
        patience: int,
        seed: int,
        loss_name: str,
        grad_accumulation_steps: int,
        hdf5_image_key: str | None = None,
        hdf5_mask_key: str | None = None,
        cpu_budget: int | None = None,
        memory_budget_gb: float | None = None,
        cache_fraction: float = 0.45,
        deep_supervision: bool = False,
        model_depth: int = 4,
        residual: bool = True,
        use_se: bool = True,
        norm_groups: int = 0,
        ema_decay: float = 0.0,
        grad_clip: float = 1.0,
        warmup_epochs: int = 1,
        tta: str = "none",
        channels_last: bool = True,
    ) -> None:
        seed_everything(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Detect and claim the CPU/RAM budget before building anything heavy.
        self.resource_plan = configure_runtime(
            cpu_budget=cpu_budget,
            memory_budget_gb=memory_budget_gb,
            dataloader_workers=num_workers if num_workers and num_workers > 0 else None,
            cache_fraction=cache_fraction,
            use_cuda=self.device.type == "cuda",
        )
        print(self.resource_plan.describe())
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.best_checkpoint_path = self.output_dir / "best.pt"
        self.last_checkpoint_path = self.output_dir / "last.pt"
        self.channels = channels
        self.epochs = epochs
        self.amp_enabled = amp and self.device.type == "cuda"
        self.patience = patience
        self.loss_name = loss_name
        self.grad_accumulation_steps = max(1, grad_accumulation_steps)
        self.deep_supervision = deep_supervision
        self.grad_clip = grad_clip
        self.tta = tta
        # channels_last suits convolutions on both CUDA tensor cores and CPU
        # oneDNN kernels; harmless where unsupported.
        self.channels_last = channels_last

        # Split the cache budget between the two datasets in proportion to a
        # typical 80/20 split, so the train set (read every epoch, augmented)
        # gets the bulk of it.
        #
        # Each DataLoader worker forks its own copy of the dataset and therefore
        # its own cache, so the per-dataset budget must be divided by the number
        # of processes that will hold one. Without this the RSS would be
        # workers x the intended budget and the box would swap or OOM.
        cache_replicas = max(1, self.resource_plan.dataloader_workers)
        total_cache = self.resource_plan.cache_budget_bytes // cache_replicas
        train_cache = int(total_cache * 0.8)
        val_cache = total_cache - train_cache

        train_dataset = SolarActiveRegionDataset(
            manifest_path=manifest_path,
            channels=channels,
            split="train",
            image_size=image_size,
            normalize_mode=normalize_mode,
            augment=True,
            seed=seed,
            hdf5_image_key=hdf5_image_key,
            hdf5_mask_key=hdf5_mask_key,
            cache_budget_bytes=train_cache,
        )
        val_dataset = SolarActiveRegionDataset(
            manifest_path=manifest_path,
            channels=channels,
            split="val",
            image_size=image_size,
            normalize_mode=normalize_mode,
            augment=False,
            seed=seed,
            hdf5_image_key=hdf5_image_key,
            hdf5_mask_key=hdf5_mask_key,
            cache_budget_bytes=val_cache,
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        loader_kwargs = {
            "num_workers": self.resource_plan.dataloader_workers,
            "pin_memory": self.resource_plan.pin_memory,
            "persistent_workers": self.resource_plan.persistent_workers,
        }
        if self.resource_plan.dataloader_workers > 0:
            loader_kwargs["prefetch_factor"] = self.resource_plan.prefetch_factor

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            **loader_kwargs,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        self.model = AttentionUNet(
            in_channels=len(channels),
            out_channels=1,
            base_channels=base_channels,
            dropout=dropout,
            depth=model_depth,
            residual=residual,
            use_se=use_se,
            norm_groups=norm_groups,
            deep_supervision=deep_supervision,
        ).to(self.device)
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        self.ema = ModelEma(self.model, decay=ema_decay) if ema_decay > 0 else None
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        steps_per_epoch = max(1, math.ceil(len(self.train_loader) / self.grad_accumulation_steps))
        self.scheduler = build_scheduler(
            self.optimizer,
            epochs=max(epochs, 1),
            steps_per_epoch=steps_per_epoch,
            learning_rate=learning_rate,
            warmup_epochs=warmup_epochs,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.criterion = build_loss(loss_name, deep_supervision=deep_supervision)

        self._write_run_metadata()

    def _write_run_metadata(self) -> None:
        """Record the resolved resource plan alongside the checkpoints."""
        payload = {
            "resource_plan": self.resource_plan.__dict__,
            "device": str(self.device),
            "amp_enabled": self.amp_enabled,
            "deep_supervision": self.deep_supervision,
            "ema": self.ema is not None,
            "tta": self.tta,
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }
        with (self.output_dir / "runtime.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

    def fit(self) -> None:
        best_score = -math.inf
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            start = perf_counter()
            train_loss = self._run_epoch(epoch, training=True)
            train_seconds = perf_counter() - start

            # Validate with the EMA weights when available: they are what we
            # would actually deploy, so early stopping should track them.
            backup = self.ema.copy_to(self.model) if self.ema is not None else None
            try:
                val_loss, val_dice, val_iou = self._run_epoch(epoch, training=False)
            finally:
                if backup is not None:
                    self.model.load_state_dict(backup)

            elapsed = perf_counter() - start
            samples = max(1, len(self.train_loader.dataset))

            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_dice=val_dice,
                val_iou=val_iou,
                learning_rate=self.optimizer.param_groups[0]["lr"],
                seconds=elapsed,
                samples_per_second=samples / max(train_seconds, 1e-6),
                process_memory_gb=process_memory_gb(),
                cache_hit_rate=self._cache_hit_rate(),
            )
            self._append_metrics(metrics)
            self._save_checkpoint(self.last_checkpoint_path, metrics)

            if val_dice > best_score:
                best_score = val_dice
                epochs_without_improvement = 0
                self._save_checkpoint(self.best_checkpoint_path, metrics)
            else:
                epochs_without_improvement += 1

            print(
                f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_dice={val_dice:.4f} val_iou={val_iou:.4f} lr={metrics.learning_rate:.6f} "
                f"samples/s={metrics.samples_per_second:.2f} rss={metrics.process_memory_gb:.2f}GB "
                f"cache_hit={metrics.cache_hit_rate:.2f}"
            )

            if self.patience > 0 and epochs_without_improvement >= self.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    def _run_epoch(self, epoch: int, training: bool) -> float | tuple[float, float, float]:
        loader = self.train_loader if training else self.val_loader
        self.model.train(training)
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        batches = 0

        progress = tqdm(loader, desc=("train" if training else "val") + f" epoch {epoch}", leave=False, disable=os.getenv("DISABLE_TQDM", "0") == "1")
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        for batch_index, (images, masks) in enumerate(progress, start=1):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            if self.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            if not training:
                # Validation optionally goes through TTA so the reported Dice
                # reflects the inference path that will actually be deployed.
                probs = tta_predict(
                    self.model,
                    images,
                    transforms=self.tta,
                    autocast_kwargs={"device_type": "cuda", "enabled": self.amp_enabled}
                    if self.device.type == "cuda"
                    else {"device_type": "cpu", "enabled": False},
                )
                loss = F.binary_cross_entropy(probs.clamp(1e-6, 1 - 1e-6), masks)
                dice, iou = compute_metrics_from_probs(probs, masks)
                total_dice += dice
                total_iou += iou
            else:
                with torch.amp.autocast(device_type="cuda", enabled=self.amp_enabled):
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)

                scaled_loss = loss / self.grad_accumulation_steps
                self.scaler.scale(scaled_loss).backward()
                should_step = (batch_index % self.grad_accumulation_steps == 0) or (batch_index == len(loader))
                if should_step:
                    if self.grad_clip > 0:
                        # Unscale before clipping, otherwise the threshold would
                        # apply to AMP-scaled gradients and do nothing useful.
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    # Per-step LR schedule (warmup + cosine).
                    self.scheduler.step()
                    if self.ema is not None:
                        self.ema.update(self.model)

            total_loss += float(loss.item())
            batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        average_loss = total_loss / max(batches, 1)
        if training:
            return average_loss
        return average_loss, total_dice / max(batches, 1), total_iou / max(batches, 1)

    def _cache_hit_rate(self) -> float:
        """Training-set cache hit rate, or -1 when it cannot be observed.

        With ``num_workers > 0`` each worker process owns its own copy of the
        dataset, so the parent's counters stay at zero however well the caches
        are performing. Report -1 rather than a misleading 0.00.
        """
        if self.resource_plan.dataloader_workers > 0:
            return -1.0
        return float(self.train_dataset.cache_stats()["hit_rate"])

    def _append_metrics(self, metrics: EpochMetrics) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(metrics)) + "\n")

    def _save_checkpoint(self, path: Path, metrics: EpochMetrics) -> None:
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "channels": self.channels,
            "metrics": asdict(metrics),
            "tta": self.tta,
            "deep_supervision": self.deep_supervision,
        }
        if self.ema is not None:
            # The EMA weights are the ones that were validated, so store them
            # separately rather than overwriting the raw weights (which are
            # still needed to resume training).
            payload["ema_state_dict"] = self.ema.state_dict()
        torch.save(payload, path)
