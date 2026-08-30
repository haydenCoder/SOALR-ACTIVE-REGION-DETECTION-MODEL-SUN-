from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

torch = pytest.importorskip("torch")

from solar_ar.runtime import (  # noqa: E402
    DEFAULT_CPU_BUDGET,
    DEFAULT_CPU_HEADROOM,
    DEFAULT_MEMORY_BUDGET_GB,
    amp_settings,
    detect_cpu_count,
    detect_memory_gb,
    plan_resources,
    process_memory_gb,
    suggest_batch_size,
)
from solar_ar.tta import (  # noqa: E402
    TTA_PRESETS,
    _TRANSFORMS,
    resolve_transforms,
    sliding_window_predict,
    tta_predict,
)


# --------------------------------------------------------------------------
# Resource planning
# --------------------------------------------------------------------------


def test_detection_returns_positive_values():
    assert detect_cpu_count() >= 1
    assert detect_memory_gb() > 0


def test_plan_never_exceeds_detected_hardware():
    """A 4-CPU/15-GB request must scale down on a smaller machine."""
    plan = plan_resources(cpu_budget=DEFAULT_CPU_BUDGET, memory_budget_gb=DEFAULT_MEMORY_BUDGET_GB)
    assert plan.cpu_budget <= plan.cpu_count_detected
    assert plan.memory_budget_gb <= plan.memory_detected_gb
    assert plan.torch_threads >= 1
    assert plan.dataloader_workers >= 0


def test_plan_respects_a_smaller_explicit_budget():
    """The budget is a ceiling, not a target to inflate towards."""
    plan = plan_resources(cpu_budget=1, memory_budget_gb=0.5)
    assert plan.cpu_budget == 1
    assert plan.memory_budget_gb <= 0.5


def test_zero_and_none_budgets_fall_back_to_defaults():
    for value in (None, 0, -1):
        plan = plan_resources(cpu_budget=value, memory_budget_gb=value)
        assert plan.cpu_budget >= 1
        assert plan.memory_budget_gb > 0


def test_workers_never_oversubscribe_the_cpu_budget():
    plan = plan_resources(cpu_budget=2, memory_budget_gb=1, dataloader_workers=64)
    assert plan.dataloader_workers <= plan.cpu_budget


def test_prefetch_and_persistence_only_apply_with_workers():
    plan = plan_resources(cpu_budget=4, memory_budget_gb=4, dataloader_workers=0)
    assert plan.dataloader_workers == 0
    assert plan.persistent_workers is False


def test_cache_budget_is_a_fraction_of_the_memory_budget():
    plan = plan_resources(cpu_budget=2, memory_budget_gb=2.0, cache_fraction=0.5)
    assert 0 < plan.cache_budget_bytes <= 2.0 * 0.5 * (1024 ** 3) + 1


def test_describe_mentions_cpu_and_ram():
    text = plan_resources().describe()
    assert "CPU" in text and "RAM" in text


def test_suggest_batch_size_shrinks_as_images_grow():
    small = suggest_batch_size(128, 3, 32, 15.0)
    large = suggest_batch_size(1024, 3, 32, 15.0)
    assert small > large >= 1


def test_process_memory_is_reported():
    assert process_memory_gb() >= 0


def test_auto_cpu_budget_is_maximum_power_minus_headroom():
    """Auto mode grabs every detected core except the OS headroom."""
    detected = detect_cpu_count()
    plan = plan_resources(cpu_budget=DEFAULT_CPU_BUDGET)
    assert plan.cpu_budget == max(1, detected - DEFAULT_CPU_HEADROOM)


def test_explicit_cpu_budget_bypasses_the_headroom():
    """A deliberate --cpu-budget request is honoured exactly (clamped only by the hardware)."""
    detected = detect_cpu_count()
    plan = plan_resources(cpu_budget=detected)
    assert plan.cpu_budget == detected


def test_prefetch_is_maxed_out_with_workers():
    plan = plan_resources(cpu_budget=4, memory_budget_gb=4, dataloader_workers=2)
    assert plan.prefetch_factor == 8


# --------------------------------------------------------------------------
# Mixed-precision selection (CUDA fp16+scaler / MPS bfloat16 / CPU fp32)
# --------------------------------------------------------------------------


def test_amp_settings_cuda_uses_fp16_and_scaler():
    assert amp_settings("cuda") == ("cuda", True, True)


def test_amp_settings_cpu_is_plain_fp32():
    assert amp_settings("cpu") == ("cpu", False, False)


def test_amp_settings_mps_never_uses_a_grad_scaler():
    """MPS runs bfloat16 autocast: no scaler needed, and none exists for MPS."""
    autocast_type, amp_enabled, scaler_enabled = amp_settings("mps")
    assert autocast_type in {"mps", "cpu"}
    assert not scaler_enabled
    if autocast_type == "mps":
        assert amp_enabled


def test_amp_settings_disabled_flag_is_always_inert():
    for device in ("cpu", "cuda", "mps"):
        assert amp_settings(device, enabled=False) == ("cpu", False, False)


def test_amp_settings_always_yield_a_constructible_autocast():
    """Whatever the probe picks, this torch build must accept the device type."""
    for device in ("cpu", "cuda", "mps"):
        autocast_type, amp_enabled, _ = amp_settings(device)
        with torch.amp.autocast(device_type=autocast_type, enabled=amp_enabled):
            pass


# --------------------------------------------------------------------------
# TTA
# --------------------------------------------------------------------------


class _Equivariant(torch.nn.Module):
    """Channel mean: commutes with every flip and rotation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1, keepdim=True)


def test_every_transform_has_an_exact_inverse():
    x = torch.randn(2, 3, 16, 16)
    for name, (forward, inverse) in _TRANSFORMS.items():
        assert torch.equal(inverse(forward(x)), x), f"{name} is not exactly invertible"


def test_d4_preset_has_eight_unique_transforms():
    assert len(TTA_PRESETS["d4"]) == 8
    assert len(set(TTA_PRESETS["d4"])) == 8


def test_identity_tta_equals_a_plain_forward_pass():
    model = torch.nn.Conv2d(3, 1, 3, padding=1).eval()
    x = torch.randn(2, 3, 16, 16)
    expected = torch.sigmoid(model(x))
    assert torch.allclose(tta_predict(model, x, "none"), expected, atol=1e-6)


def test_tta_is_a_noop_on_an_equivariant_model():
    """If the model already respects D4 symmetry, averaging changes nothing."""
    model = _Equivariant().eval()
    x = torch.randn(2, 3, 16, 16)
    expected = torch.sigmoid(model(x))
    assert torch.allclose(tta_predict(model, x, "d4"), expected, atol=1e-6)


def test_tta_output_is_a_probability_with_the_right_shape():
    model = torch.nn.Conv2d(3, 1, 3, padding=1).eval()
    x = torch.randn(2, 3, 16, 16)
    for preset in TTA_PRESETS:
        probs = tta_predict(model, x, preset)
        assert probs.shape == (2, 1, 16, 16)
        assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0


def test_tta_averaging_reduces_extremes():
    """Averaging distinct views must pull predictions off the rails."""
    torch.manual_seed(0)
    model = torch.nn.Conv2d(3, 1, 3, padding=1).eval()
    x = torch.randn(2, 3, 16, 16)
    single = tta_predict(model, x, "none")
    averaged = tta_predict(model, x, "d4")
    assert averaged.max() <= single.max() + 1e-6
    assert averaged.min() >= single.min() - 1e-6


def test_max_merge_is_at_least_the_mean():
    model = torch.nn.Conv2d(3, 1, 3, padding=1).eval()
    x = torch.randn(2, 3, 16, 16)
    mean = tta_predict(model, x, "d4", merge="mean")
    maximum = tta_predict(model, x, "d4", merge="max")
    assert torch.all(maximum >= mean - 1e-6)


def test_return_logits_round_trips_through_sigmoid():
    model = torch.nn.Conv2d(3, 1, 3, padding=1).eval()
    x = torch.randn(2, 3, 16, 16)
    probs = tta_predict(model, x, "flips")
    logits = tta_predict(model, x, "flips", return_logits=True)
    assert torch.allclose(torch.sigmoid(logits), probs, atol=1e-4)


def test_unknown_preset_and_transform_are_rejected():
    with pytest.raises(ValueError):
        resolve_transforms("does_not_exist")
    with pytest.raises(ValueError):
        resolve_transforms(["hflip", "nonsense"])
    with pytest.raises(ValueError):
        resolve_transforms([])


def test_invalid_merge_mode_is_rejected():
    model = torch.nn.Conv2d(3, 1, 3, padding=1).eval()
    with pytest.raises(ValueError):
        tta_predict(model, torch.randn(1, 3, 8, 8), "none", merge="median")


def test_explicit_transform_list_is_accepted():
    model = torch.nn.Conv2d(3, 1, 3, padding=1).eval()
    probs = tta_predict(model, torch.randn(1, 3, 8, 8), ["identity", "rot90"])
    assert probs.shape == (1, 1, 8, 8)


# --------------------------------------------------------------------------
# Sliding-window inference
# --------------------------------------------------------------------------


def test_sliding_window_covers_a_non_square_frame():
    model = _Equivariant().eval()
    image = torch.randn(3, 40, 72)
    output = sliding_window_predict(model, image, tile_size=16, overlap=0.5)
    assert output.shape == (1, 40, 72)
    assert torch.isfinite(output).all()


def test_sliding_window_handles_images_smaller_than_the_tile():
    model = _Equivariant().eval()
    output = sliding_window_predict(model, torch.randn(3, 8, 8), tile_size=32)
    assert output.shape == (1, 8, 8)


def test_sliding_window_reconstructs_a_constant_prediction():
    """Blending weights must sum correctly, or seams would show up."""

    class Constant(torch.nn.Module):
        def forward(self, x):
            # logit 0 -> probability 0.5 everywhere
            return torch.zeros(x.shape[0], 1, *x.shape[-2:])

    output = sliding_window_predict(Constant().eval(), torch.randn(3, 48, 48), tile_size=16, overlap=0.5)
    assert torch.allclose(output, torch.full_like(output, 0.5), atol=1e-4)


def test_sliding_window_rejects_bad_input():
    model = _Equivariant().eval()
    with pytest.raises(ValueError):
        sliding_window_predict(model, torch.randn(1, 3, 16, 16), tile_size=8)
    with pytest.raises(ValueError):
        sliding_window_predict(model, torch.randn(3, 16, 16), tile_size=8, overlap=1.0)
