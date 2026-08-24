"""Test-time augmentation for segmentation.

Active regions have no canonical orientation on the solar disk, so a prediction
should not change when the tile is flipped or rotated. Averaging the model's
output over the 8 symmetries of the square (the dihedral group D4) both exploits
that invariance and cancels part of the model's own noise, typically buying a
point or two of Dice for zero extra training.

Each augmentation is applied to the input, the prediction is mapped back to the
original orientation, and the probabilities are averaged. Every transform here
is its own exact inverse or has an exact inverse, so no interpolation blur is
introduced.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import torch

#: Named D4 symmetries: (forward, inverse). Rotations use k and -k.
_TRANSFORMS: dict[str, tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]] = {
    "identity": (lambda t: t, lambda t: t),
    "hflip": (lambda t: torch.flip(t, dims=[-1]), lambda t: torch.flip(t, dims=[-1])),
    "vflip": (lambda t: torch.flip(t, dims=[-2]), lambda t: torch.flip(t, dims=[-2])),
    "hvflip": (
        lambda t: torch.flip(t, dims=[-2, -1]),
        lambda t: torch.flip(t, dims=[-2, -1]),
    ),
    "rot90": (
        lambda t: torch.rot90(t, 1, dims=[-2, -1]),
        lambda t: torch.rot90(t, -1, dims=[-2, -1]),
    ),
    "rot180": (
        lambda t: torch.rot90(t, 2, dims=[-2, -1]),
        lambda t: torch.rot90(t, -2, dims=[-2, -1]),
    ),
    "rot270": (
        lambda t: torch.rot90(t, 3, dims=[-2, -1]),
        lambda t: torch.rot90(t, -3, dims=[-2, -1]),
    ),
    "transpose": (
        lambda t: t.transpose(-2, -1),
        lambda t: t.transpose(-2, -1),
    ),
}

#: Presets, cheapest first. "none" disables TTA entirely.
TTA_PRESETS: dict[str, tuple[str, ...]] = {
    "none": ("identity",),
    "flips": ("identity", "hflip", "vflip", "hvflip"),
    "d4": (
        "identity",
        "hflip",
        "vflip",
        "hvflip",
        "rot90",
        "rot180",
        "rot270",
        "transpose",
    ),
}


def resolve_transforms(spec: str | Sequence[str]) -> tuple[str, ...]:
    """Turn a preset name or explicit transform list into validated names."""
    if isinstance(spec, str):
        if spec not in TTA_PRESETS:
            raise ValueError(
                f"Unknown TTA preset {spec!r}. Choose from {sorted(TTA_PRESETS)} "
                f"or pass a list of {sorted(_TRANSFORMS)}."
            )
        return TTA_PRESETS[spec]

    names = tuple(spec)
    unknown = [name for name in names if name not in _TRANSFORMS]
    if unknown:
        raise ValueError(f"Unknown TTA transforms: {unknown}. Available: {sorted(_TRANSFORMS)}")
    if not names:
        raise ValueError("TTA transform list must not be empty")
    return names


@torch.no_grad()
def tta_predict(
    model: torch.nn.Module,
    images: torch.Tensor,
    transforms: str | Sequence[str] = "d4",
    *,
    merge: str = "mean",
    return_logits: bool = False,
    autocast_kwargs: dict | None = None,
) -> torch.Tensor:
    """Predict with test-time augmentation, averaging in probability space.

    Parameters
    ----------
    model:
        Segmentation model returning raw logits.
    images:
        Input batch, shaped ``(N, C, H, W)``.
    transforms:
        Preset name (``none``/``flips``/``d4``) or explicit transform names.
    merge:
        ``mean`` (default, lowest variance) or ``max`` (higher recall, useful
        when small active regions are being missed).
    return_logits:
        Convert the merged probability back to a logit. Handy for reusing a
        loss that expects logits; slightly lossy at saturation.

    Returns
    -------
    Probabilities in ``[0, 1]`` with the same shape as the model output.
    """
    names = resolve_transforms(transforms)
    if merge not in {"mean", "max"}:
        raise ValueError(f"Unsupported merge mode: {merge!r}. Use 'mean' or 'max'.")

    autocast_kwargs = autocast_kwargs or {"enabled": False, "device_type": "cpu"}
    accumulated: torch.Tensor | None = None

    for name in names:
        forward, inverse = _TRANSFORMS[name]
        augmented = forward(images).contiguous()

        with torch.amp.autocast(**autocast_kwargs):
            logits = model(augmented)

        # Average probabilities, not logits: logits are unbounded and a single
        # confident view would dominate the mean.
        probs = torch.sigmoid(logits.float())
        restored = inverse(probs)

        if accumulated is None:
            accumulated = restored
        elif merge == "mean":
            accumulated = accumulated + restored
        else:
            accumulated = torch.maximum(accumulated, restored)

    assert accumulated is not None  # resolve_transforms guarantees >= 1 entry
    if merge == "mean":
        accumulated = accumulated / len(names)

    if return_logits:
        clamped = accumulated.clamp(1e-6, 1 - 1e-6)
        return torch.log(clamped / (1 - clamped))
    return accumulated


@torch.no_grad()
def sliding_window_predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    tile_size: int,
    overlap: float = 0.25,
    transforms: str | Sequence[str] = "none",
    batch_size: int = 4,
) -> torch.Tensor:
    """Predict on a full frame larger than the training tile size.

    Tiles are blended with a cosine window so seams do not appear at tile
    boundaries, where predictions are least reliable. Optionally applies TTA per
    tile.

    Parameters
    ----------
    image:
        A single image shaped ``(C, H, W)``.
    overlap:
        Fraction of the tile that neighbouring tiles share (0 to <1).
    """
    if image.dim() != 3:
        raise ValueError(f"Expected a single (C, H, W) image, got shape {tuple(image.shape)}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    channels, height, width = image.shape
    tile = min(tile_size, height, width)
    stride = max(1, int(tile * (1.0 - overlap)))

    device = image.device
    accumulator = torch.zeros((1, height, width), dtype=torch.float32, device=device)
    weights = torch.zeros((1, height, width), dtype=torch.float32, device=device)
    window = _cosine_window(tile, device)

    positions = [
        (y, x)
        for y in _tile_starts(height, tile, stride)
        for x in _tile_starts(width, tile, stride)
    ]

    for index in range(0, len(positions), batch_size):
        chunk = positions[index : index + batch_size]
        patches = torch.stack([image[:, y : y + tile, x : x + tile] for y, x in chunk])
        probs = tta_predict(model, patches, transforms=transforms)

        for (y, x), prob in zip(chunk, probs):
            accumulator[:, y : y + tile, x : x + tile] += prob[0] * window
            weights[:, y : y + tile, x : x + tile] += window

    return accumulator / weights.clamp_min(1e-8)


def _tile_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _cosine_window(size: int, device: torch.device) -> torch.Tensor:
    """2-D raised-cosine window that tapers to a small positive value at edges."""
    ramp = torch.hann_window(size, periodic=False, device=device).clamp_min(1e-3)
    return torch.outer(ramp, ramp)


def available_transforms() -> Iterable[str]:
    return sorted(_TRANSFORMS)
