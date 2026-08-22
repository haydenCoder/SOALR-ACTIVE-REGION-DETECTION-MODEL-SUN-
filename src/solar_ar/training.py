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
from torch.utils.data import DataLoader
from tqdm import tqdm

from solar_ar.data import SolarActiveRegionDataset
from solar_ar.models import AttentionUNet


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_dice: float
    val_iou: float
    learning_rate: float
    seconds: float


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


def build_loss(loss_name: str) -> nn.Module:
    if loss_name == "focal_tversky":
        return FocalTverskyLoss()
    if loss_name == "bce_dice":
        return BCEDiceLoss()
    raise ValueError(f"Unsupported loss: {loss_name}")


@torch.no_grad()
def compute_segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    preds = (torch.sigmoid(logits) > 0.5).float()
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
    ) -> None:
        seed_everything(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        train_dataset = SolarActiveRegionDataset(
            manifest_path=manifest_path,
            channels=channels,
            split="train",
            image_size=image_size,
            normalize_mode=normalize_mode,
            augment=True,
            seed=seed,
        )
        val_dataset = SolarActiveRegionDataset(
            manifest_path=manifest_path,
            channels=channels,
            split="val",
            image_size=image_size,
            normalize_mode=normalize_mode,
            augment=False,
            seed=seed,
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

        self.model = AttentionUNet(
            in_channels=len(channels),
            out_channels=1,
            base_channels=base_channels,
            dropout=dropout,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(epochs, 1),
            eta_min=learning_rate * 0.05,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.criterion = build_loss(loss_name)

    def fit(self) -> None:
        best_score = -math.inf
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            start = perf_counter()
            train_loss = self._run_epoch(epoch, training=True)
            val_loss, val_dice, val_iou = self._run_epoch(epoch, training=False)
            self.scheduler.step()
            elapsed = perf_counter() - start

            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_dice=val_dice,
                val_iou=val_iou,
                learning_rate=self.optimizer.param_groups[0]["lr"],
                seconds=elapsed,
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
                f"val_dice={val_dice:.4f} val_iou={val_iou:.4f} lr={metrics.learning_rate:.6f}"
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

            with torch.set_grad_enabled(training):
                with torch.amp.autocast(device_type="cuda", enabled=self.amp_enabled):
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)

                if training:
                    scaled_loss = loss / self.grad_accumulation_steps
                    self.scaler.scale(scaled_loss).backward()
                    should_step = (batch_index % self.grad_accumulation_steps == 0) or (batch_index == len(loader))
                    if should_step:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                else:
                    dice, iou = compute_segmentation_metrics(logits, masks)
                    total_dice += dice
                    total_iou += iou

            total_loss += float(loss.item())
            batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        average_loss = total_loss / max(batches, 1)
        if training:
            return average_loss
        return average_loss, total_dice / max(batches, 1), total_iou / max(batches, 1)

    def _append_metrics(self, metrics: EpochMetrics) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(metrics)) + "\n")

    def _save_checkpoint(self, path: Path, metrics: EpochMetrics) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "channels": self.channels,
                "metrics": asdict(metrics),
            },
            path,
        )
