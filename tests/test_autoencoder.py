"""Tests for the autoencoder architecture."""

from __future__ import annotations

import torch
from torch import nn

from biomedical_ml.autoencoder import Autoencoder, AutoencoderConfig
from biomedical_ml.config import set_seed


def test_forward_pass_reconstructs_to_input_shape():
    config = AutoencoderConfig(input_dim=20, latent_dim=4, hidden_dims=(8,))
    model = Autoencoder(config)
    x = torch.randn(6, 20)

    reconstruction = model(x)

    assert reconstruction.shape == x.shape


def test_encode_produces_latent_dim_output():
    config = AutoencoderConfig(input_dim=20, latent_dim=4, hidden_dims=(8,))
    model = Autoencoder(config)
    x = torch.randn(6, 20)

    latent = model.encode(x)

    assert latent.shape == (6, 4)


def test_decode_maps_latent_back_to_input_dim():
    config = AutoencoderConfig(input_dim=20, latent_dim=4, hidden_dims=(8,))
    model = Autoencoder(config)
    z = torch.randn(6, 4)

    reconstruction = model.decode(z)

    assert reconstruction.shape == (6, 20)


def test_works_with_no_hidden_layers():
    # A direct input -> latent -> input linear autoencoder is a degenerate but
    # valid configuration; the _mlp helper must not assume at least one hidden size.
    config = AutoencoderConfig(input_dim=10, latent_dim=3, hidden_dims=())
    model = Autoencoder(config)
    x = torch.randn(5, 10)

    reconstruction = model(x)

    assert reconstruction.shape == x.shape
    assert len(model.encoder) == 1  # a single bare Linear, no activation/dropout
    assert len(model.decoder) == 1


def test_final_encoder_and_decoder_layers_have_no_activation():
    config = AutoencoderConfig(input_dim=20, latent_dim=4, hidden_dims=(8,))
    model = Autoencoder(config)

    assert isinstance(model.encoder[-1], nn.Linear)
    assert isinstance(model.decoder[-1], nn.Linear)


def test_dropout_probability_is_applied_to_hidden_layers():
    config = AutoencoderConfig(input_dim=20, latent_dim=4, hidden_dims=(8, 8), dropout=0.35)
    model = Autoencoder(config)

    dropouts = [m for m in model.encoder if isinstance(m, nn.Dropout)]
    assert dropouts, "expected at least one Dropout layer in the encoder"
    assert all(d.p == 0.35 for d in dropouts)


def test_eval_mode_disables_dropout_stochasticity():
    config = AutoencoderConfig(input_dim=20, latent_dim=4, hidden_dims=(8,), dropout=0.5)
    model = Autoencoder(config)
    model.eval()
    x = torch.randn(4, 20)

    with torch.no_grad():
        first = model(x)
        second = model(x)

    torch.testing.assert_close(first, second)


def test_same_seed_gives_identical_initial_weights():
    config = AutoencoderConfig(input_dim=20, latent_dim=4, hidden_dims=(8,))

    set_seed(0)
    first = Autoencoder(config)
    set_seed(0)
    second = Autoencoder(config)

    for p1, p2 in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(p1, p2)


def test_model_holds_the_config_it_was_built_with():
    config = AutoencoderConfig(input_dim=20, latent_dim=4, hidden_dims=(8,), dropout=0.3)
    model = Autoencoder(config)

    assert model.config == config
