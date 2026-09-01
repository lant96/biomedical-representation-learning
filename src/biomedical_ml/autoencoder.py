"""A small autoencoder for the SelectKBest-selected GSE138458 feature space.

The encoder/decoder are trained unsupervised — the loss is pure reconstruction
error, the label never appears in it. That is the whole point of using an
autoencoder rather than a classifier's hidden layer here: the latent space it
learns isn't shaped by the label, so asking "does it separate SLE from
control anyway?" (Day 4) is a real question, not a foregone conclusion.

The input space is deliberately the same one the classical baselines use —
:func:`biomedical_ml.models.build_feature_pipeline`'s ~2000 selected, scaled
probes — so a comparison between the two isn't confounded by the AE also
having a different, unaudited feature set to work with.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AutoencoderConfig:
    """Architecture hyperparameters, bundled so a checkpoint can store them as-is."""

    input_dim: int
    latent_dim: int = 32
    hidden_dims: tuple[int, ...] = (256, 64)
    dropout: float = 0.2


class Autoencoder(nn.Module):
    """Symmetric encoder/decoder MLP with a bottleneck of ``latent_dim``.

    The decoder mirrors the encoder's hidden sizes in reverse. Neither the
    bottleneck output nor the final reconstruction carries an activation: the
    latent code should be free to take any sign or scale, and the
    reconstruction targets standardized (zero-mean, unit-variance) expression
    values, which a squashing activation like ReLU or sigmoid would clip.
    """

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = _mlp(
            [config.input_dim, *config.hidden_dims, config.latent_dim], dropout=config.dropout
        )
        self.decoder = _mlp(
            [config.latent_dim, *reversed(config.hidden_dims), config.input_dim],
            dropout=config.dropout,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def _mlp(dims: list[int], *, dropout: float) -> nn.Sequential:
    """A ``Linear -> ReLU -> Dropout`` stack over consecutive sizes in ``dims``.

    The final ``Linear`` (from ``dims[-2]`` to ``dims[-1]``) is left bare —
    no activation, no dropout — since it produces either the bottleneck code
    or the reconstruction, neither of which should be squashed or zeroed out.
    """
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)
