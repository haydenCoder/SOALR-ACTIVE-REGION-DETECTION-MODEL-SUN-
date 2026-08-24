from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

torch = pytest.importorskip("torch")

from solar_ar.models import AttentionUNet  # noqa: E402
from solar_ar.training import (  # noqa: E402
    ComboLoss,
    DeepSupervisionLoss,
    ModelEma,
    build_loss,
    build_scheduler,
    compute_metrics_from_probs,
)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_output_shape_matches_input_at_every_depth(depth):
    model = AttentionUNet(in_channels=3, base_channels=4, depth=depth).eval()
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 1, 32, 32)


def test_non_power_of_two_input_still_round_trips():
    """Odd sizes must survive the pooling/upsampling round trip."""
    model = AttentionUNet(in_channels=1, base_channels=4, depth=3).eval()
    out = model(torch.randn(1, 1, 50, 34))
    assert out.shape == (1, 1, 50, 34)


def test_deep_supervision_returns_aligned_heads_in_training_mode():
    model = AttentionUNet(in_channels=3, base_channels=4, depth=4, deep_supervision=True)
    model.train()
    outputs = model(torch.randn(2, 3, 32, 32))
    assert isinstance(outputs, list) and len(outputs) == 4
    # Every auxiliary head must be upsampled to the full input resolution.
    for head in outputs:
        assert head.shape == (2, 1, 32, 32)


def test_deep_supervision_is_disabled_at_inference():
    model = AttentionUNet(in_channels=3, base_channels=4, deep_supervision=True).eval()
    assert isinstance(model(torch.randn(1, 3, 32, 32)), torch.Tensor)


def test_groupnorm_variant_works_with_batch_size_one():
    """BatchNorm cannot train on a single sample; GroupNorm is why it is offered."""
    model = AttentionUNet(in_channels=3, base_channels=8, norm_groups=4)
    model.train()
    out = model(torch.randn(1, 3, 32, 32))
    out.sum().backward()
    assert out.shape == (1, 1, 32, 32)


def test_gradients_reach_the_earliest_encoder_layer():
    """Guards against a silently detached graph in the attention gates."""
    model = AttentionUNet(in_channels=3, base_channels=4, depth=3)
    model.train()
    model(torch.randn(2, 3, 32, 32)).sum().backward()
    first = model.encoders[0].conv.conv1.weight
    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert first.grad.abs().sum() > 0


def test_residual_and_se_toggles_change_parameter_count():
    plain = AttentionUNet(in_channels=3, base_channels=8, residual=False, use_se=False)
    fancy = AttentionUNet(in_channels=3, base_channels=8, residual=True, use_se=True)
    assert sum(p.numel() for p in fancy.parameters()) > sum(p.numel() for p in plain.parameters())


def test_invalid_depth_is_rejected():
    with pytest.raises(ValueError):
        AttentionUNet(depth=1)


def test_predict_helper_returns_probabilities():
    model = AttentionUNet(in_channels=3, base_channels=4, depth=2)
    probs = model.predict(torch.randn(1, 3, 16, 16))
    assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0


def test_checkpoint_round_trips_between_deep_supervision_settings():
    """Aux heads are always built, so a checkpoint stays loadable either way."""
    trained = AttentionUNet(in_channels=3, base_channels=4, depth=3, deep_supervision=True)
    deployed = AttentionUNet(in_channels=3, base_channels=4, depth=3, deep_supervision=False)
    deployed.load_state_dict(trained.state_dict())


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["bce_dice", "focal_tversky", "combo"])
def test_losses_are_lower_for_a_correct_prediction(name):
    loss_fn = build_loss(name)
    targets = torch.zeros(2, 1, 16, 16)
    targets[:, :, 4:12, 4:12] = 1.0
    good = torch.where(targets > 0, 6.0, -6.0)
    assert float(loss_fn(good, targets)) < float(loss_fn(-good, targets))


def test_unknown_loss_is_rejected():
    with pytest.raises(ValueError):
        build_loss("not_a_loss")


def test_deep_supervision_loss_weights_the_main_head_most():
    base = torch.nn.BCEWithLogitsLoss()
    criterion = DeepSupervisionLoss(base)
    targets = torch.zeros(1, 1, 8, 8)
    good, bad = torch.full((1, 1, 8, 8), -6.0), torch.full((1, 1, 8, 8), 6.0)
    # Same set of outputs, but with the good prediction in the main slot.
    main_good = criterion([good, bad], targets)
    main_bad = criterion([bad, good], targets)
    assert float(main_good) < float(main_bad)


def test_deep_supervision_loss_accepts_a_bare_tensor():
    criterion = DeepSupervisionLoss(torch.nn.BCEWithLogitsLoss())
    value = criterion(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
    assert torch.isfinite(value)


def test_build_loss_wraps_only_when_deep_supervision_is_on():
    assert isinstance(build_loss("combo", deep_supervision=True), DeepSupervisionLoss)
    assert isinstance(build_loss("combo", deep_supervision=False), ComboLoss)


def test_losses_are_finite_for_an_empty_mask():
    """All-background tiles are common and must not produce NaNs."""
    targets = torch.zeros(2, 1, 8, 8)
    for name in ("bce_dice", "focal_tversky", "combo"):
        value = build_loss(name)(torch.randn(2, 1, 8, 8), targets)
        assert torch.isfinite(value), name


# --------------------------------------------------------------------------
# Metrics, scheduler, EMA
# --------------------------------------------------------------------------


def test_metrics_are_perfect_for_an_exact_match():
    targets = torch.zeros(1, 1, 8, 8)
    targets[:, :, 2:6, 2:6] = 1.0
    dice, iou = compute_metrics_from_probs(targets.clone(), targets)
    assert dice == pytest.approx(1.0, abs=1e-4)
    assert iou == pytest.approx(1.0, abs=1e-4)


def test_metrics_are_zero_for_a_disjoint_prediction():
    targets = torch.zeros(1, 1, 8, 8)
    targets[:, :, 0:4, 0:4] = 1.0
    probs = torch.zeros(1, 1, 8, 8)
    probs[:, :, 4:8, 4:8] = 1.0
    dice, iou = compute_metrics_from_probs(probs, targets)
    assert dice == pytest.approx(0.0, abs=1e-4)
    assert iou == pytest.approx(0.0, abs=1e-4)


def test_scheduler_warms_up_then_decays():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, epochs=10, steps_per_epoch=10, learning_rate=1e-3, warmup_epochs=2)

    start = optimizer.param_groups[0]["lr"]
    for _ in range(20):  # end of warmup
        scheduler.step()
    peak = optimizer.param_groups[0]["lr"]
    for _ in range(80):  # end of training
        scheduler.step()
    end = optimizer.param_groups[0]["lr"]

    assert start < peak, "learning rate should rise during warmup"
    assert peak == pytest.approx(1e-3, rel=1e-3), "warmup should reach the base learning rate"
    assert end < peak, "learning rate should decay after warmup"
    assert end > 0, "cosine floor should stay positive"


def test_scheduler_without_warmup_starts_at_the_base_rate():
    optimizer = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=1e-3)
    build_scheduler(optimizer, epochs=5, steps_per_epoch=4, learning_rate=1e-3, warmup_epochs=0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3, rel=1e-3)


def test_ema_tracks_the_model_without_matching_it_immediately():
    model = torch.nn.Linear(4, 4)
    ema = ModelEma(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model)
    assert not torch.allclose(ema.shadow["weight"], model.weight)

    for _ in range(200):
        ema.update(model)
    # After many updates at a fixed target, the average converges to it.
    assert torch.allclose(ema.shadow["weight"], model.weight, atol=1e-2)


def test_ema_copy_to_restores_the_original_weights():
    model = torch.nn.Linear(4, 4)
    ema = ModelEma(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)

    backup = ema.copy_to(model)
    assert not torch.allclose(model.weight, torch.full_like(model.weight, 3.0))
    model.load_state_dict(backup)
    assert torch.allclose(model.weight, torch.full_like(model.weight, 3.0))


def test_ema_handles_integer_buffers():
    """BatchNorm's num_batches_tracked is integral and must not be averaged."""
    model = torch.nn.BatchNorm2d(4)
    model.train()
    model(torch.randn(2, 4, 8, 8))
    ema = ModelEma(model, decay=0.9)
    ema.update(model)
    assert ema.state_dict()["num_batches_tracked"].dtype == torch.int64


# --------------------------------------------------------------------------
# Dataset RAM cache
# --------------------------------------------------------------------------


def _tiny_manifest(directory: Path, samples: int = 4) -> Path:
    import csv

    import numpy as np

    rows = []
    for index in range(samples):
        paths = {}
        for channel in ("aia171", "hmi_m"):
            path = directory / f"{channel}_{index}.npy"
            np.save(path, np.random.rand(8, 8).astype("float32"))
            paths[channel] = path.name
        mask_path = directory / f"mask_{index}.npy"
        np.save(mask_path, (np.random.rand(8, 8) > 0.5).astype("float32"))
        rows.append(
            {
                "sample_id": f"s{index}",
                "split": "train",
                "image_aia171": paths["aia171"],
                "image_hmi_m": paths["hmi_m"],
                "mask": mask_path.name,
            }
        )

    manifest = directory / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _dataset(manifest: Path, **kwargs):
    from solar_ar.data import SolarActiveRegionDataset

    return SolarActiveRegionDataset(
        manifest_path=manifest,
        channels=["aia171", "hmi_m"],
        split="train",
        image_size=8,
        **kwargs,
    )


def test_cache_serves_repeat_reads(tmp_path):
    dataset = _dataset(_tiny_manifest(tmp_path), cache_budget_bytes=10 * 1024 * 1024)
    for _ in range(2):
        for index in range(len(dataset)):
            dataset[index]

    stats = dataset.cache_stats()
    assert stats["cached_samples"] == len(dataset)
    assert stats["hit_rate"] == pytest.approx(0.5, abs=1e-6)


def test_cache_is_disabled_with_a_zero_budget(tmp_path):
    dataset = _dataset(_tiny_manifest(tmp_path), cache_budget_bytes=0)
    dataset[0]
    dataset[0]
    assert dataset.cache_stats()["cached_samples"] == 0


def test_cache_stops_at_its_budget(tmp_path):
    """A tiny budget must cache a few samples, not overshoot."""
    dataset = _dataset(_tiny_manifest(tmp_path, samples=8), cache_budget_bytes=1024)
    for index in range(len(dataset)):
        dataset[index]
    stats = dataset.cache_stats()
    assert stats["cache_bytes"] <= 1024
    assert stats["cached_samples"] < 8


def test_cached_tensors_are_not_mutated_by_augmentation(tmp_path):
    """Augmentation must clone, or the cache would degrade every epoch."""
    dataset = _dataset(
        _tiny_manifest(tmp_path),
        cache_budget_bytes=10 * 1024 * 1024,
        augment=True,
        seed=1,
    )
    dataset[0]
    pristine = dataset._cache[0][0].clone()
    for _ in range(10):
        dataset[0]
    assert torch.equal(dataset._cache[0][0], pristine)


def test_caching_does_not_change_unaugmented_output(tmp_path):
    manifest = _tiny_manifest(tmp_path)
    uncached = _dataset(manifest, cache_budget_bytes=0)
    cached = _dataset(manifest, cache_budget_bytes=10 * 1024 * 1024)
    for index in range(len(uncached)):
        image_a, mask_a = uncached[index]
        image_b, mask_b = cached[index]
        assert torch.equal(image_a, image_b)
        assert torch.equal(mask_a, mask_b)


def test_augmentation_still_varies_when_cached(tmp_path):
    dataset = _dataset(
        _tiny_manifest(tmp_path),
        cache_budget_bytes=10 * 1024 * 1024,
        augment=True,
        seed=3,
    )
    views = {dataset[0][0].numpy().tobytes() for _ in range(30)}
    assert len(views) > 1, "cache must not freeze augmentation to a single view"


def test_intensity_augmentation_leaves_the_mask_binary(tmp_path):
    dataset = _dataset(_tiny_manifest(tmp_path), augment=True, intensity_augment=True, seed=5)
    for _ in range(20):
        _, mask = dataset[0]
        assert torch.all((mask == 0) | (mask == 1))
