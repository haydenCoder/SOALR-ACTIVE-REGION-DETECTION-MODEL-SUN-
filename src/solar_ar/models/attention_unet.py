from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SqueezeExcite(nn.Module):
    """Channel attention (Squeeze-and-Excitation).

    Recalibrates channels using global context. For a multi-instrument stack
    (AIA 171/193 + HMI) this lets the network weight whichever channel carries
    the evidence for a given region, which a plain conv cannot do because its
    receptive field is local.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, optionally residual with SE channel attention.

    ``residual=True`` adds an identity path so gradients reach the encoder
    directly; this is what makes the deeper/wider configurations trainable
    without the loss stalling early.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        residual: bool = True,
        use_se: bool = True,
        norm_groups: int = 0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _make_norm(out_channels, norm_groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _make_norm(out_channels, norm_groups)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.se = SqueezeExcite(out_channels) if use_se else nn.Identity()

        self.residual = residual
        if residual and in_channels != out_channels:
            # 1x1 projection so the identity path matches the output width.
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                _make_norm(out_channels, norm_groups),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x) if self.residual else None

        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.se(out)

        if identity is not None:
            out = out + identity
        out = self.activation(out)
        return self.dropout(out)


def _make_norm(channels: int, norm_groups: int) -> nn.Module:
    """BatchNorm by default; GroupNorm when batches are tiny.

    With the batch size of 2 that a 15 GB CPU box allows at 512x512, BatchNorm's
    batch statistics are extremely noisy. GroupNorm is batch-independent and
    trains far more stably in that regime.
    """
    if norm_groups > 0:
        return nn.GroupNorm(num_groups=min(norm_groups, channels), num_channels=channels)
    return nn.BatchNorm2d(channels)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **block_kwargs) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, **block_kwargs)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return skip, self.pool(skip)


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al.).

    Suppresses skip-connection features in regions the decoder is not attending
    to, which matters here because active regions cover a small fraction of the
    disk and the skips would otherwise flood the decoder with quiet-Sun signal.
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_projection = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_projection = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        attention = self.relu(self.gate_projection(gate) + self.skip_projection(skip))
        return skip * self.psi(attention)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        **block_kwargs,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionGate(
            gate_channels=out_channels,
            skip_channels=skip_channels,
            inter_channels=max(out_channels // 2, 1),
        )
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, **block_kwargs)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        gated_skip = self.attention(x, skip)
        return self.conv(torch.cat([gated_skip, x], dim=1))


class AttentionUNet(nn.Module):
    """Attention U-Net with residual+SE blocks and optional deep supervision.

    Improvements over the plain version:

    * **Residual + SE blocks** for stable gradients and cross-channel weighting.
    * **Deep supervision** -- auxiliary heads on the decoder stages. Training
      them alongside the final head pushes usable gradient into the deep layers
      early, which speeds convergence and improves small-region recall. The
      auxiliary heads cost nothing at inference (``deep_supervision`` outputs
      are only returned in training mode).
    * **GroupNorm option** for the small batches a CPU box forces.
    * **Configurable depth** so the same class covers a fast 4-level model and a
      larger 5-level one.

    ``forward`` returns a single logits tensor in eval mode, and a list of
    ``[final, aux3, aux2, ...]`` in training mode when deep supervision is on.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.1,
        depth: int = 4,
        residual: bool = True,
        use_se: bool = True,
        norm_groups: int = 0,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError(f"depth must be at least 2, got {depth}")

        self.depth = depth
        self.deep_supervision = deep_supervision
        block_kwargs = {
            "dropout": dropout,
            "residual": residual,
            "use_se": use_se,
            "norm_groups": norm_groups,
        }

        widths = [base_channels * (2 ** level) for level in range(depth)]

        self.encoders = nn.ModuleList()
        previous = in_channels
        for width in widths:
            self.encoders.append(EncoderBlock(previous, width, **block_kwargs))
            previous = width

        self.bridge = ConvBlock(widths[-1], widths[-1] * 2, **block_kwargs)

        self.decoders = nn.ModuleList()
        previous = widths[-1] * 2
        for width in reversed(widths):
            self.decoders.append(DecoderBlock(previous, width, width, **block_kwargs))
            previous = width

        self.head = nn.Conv2d(widths[0], out_channels, kernel_size=1)

        # Auxiliary heads for every decoder stage except the last (which is the
        # main head). Built unconditionally so a checkpoint stays loadable
        # whether or not deep supervision was enabled during training.
        # Decoder i outputs reversed(widths)[i], so the aux heads (all stages
        # except the final one) take the widths in reversed order, skipping the
        # narrowest which belongs to the main head.
        self.aux_heads = nn.ModuleList(
            [nn.Conv2d(width, out_channels, kernel_size=1) for width in reversed(widths[1:])]
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """He initialization, matched to the ReLU activations."""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        input_size = x.shape[-2:]

        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        x = self.bridge(x)

        outputs: list[torch.Tensor] = []
        for index, decoder in enumerate(self.decoders):
            x = decoder(x, skips[-(index + 1)])
            # Collect intermediate stages for deep supervision (not the last).
            if self.deep_supervision and self.training and index < len(self.decoders) - 1:
                aux = self.aux_heads[index](x)
                outputs.append(F.interpolate(aux, size=input_size, mode="bilinear", align_corners=False))

        final = self.head(x)
        if self.deep_supervision and self.training:
            return [final, *outputs]
        return final

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience inference helper returning probabilities."""
        self.eval()
        return torch.sigmoid(self(x))
