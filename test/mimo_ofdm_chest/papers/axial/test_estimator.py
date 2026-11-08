# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shape, interpolation, compilation, and training tests for axial estimation."""

from __future__ import annotations

import pytest
import torch

from sionna.bench.mimo_ofdm_chest.benchmarking import build_resource_grid
from sionna.bench.mimo_ofdm_chest.papers.axial.asset import load_model, train_model
from sionna.bench.mimo_ofdm_chest.papers.axial.estimator import (
    AxialTransformerChannelEstimator,
    AxialTransformerChannelNet,
)
from test.mimo_ofdm_chest.system_variants import (
    assert_compiled_estimator_matches_eager,
    system_variant_config,
)


_SYSTEM_VARIANTS = ("baseline", "alternate", "wide")


def _state(resource_grid, **architecture) -> dict[str, torch.Tensor]:
    model = AxialTransformerChannelNet(
        int(resource_grid.num_effective_subcarriers),
        int(resource_grid.num_ofdm_symbols),
        **architecture,
    )
    return model.state_dict()


def _architecture() -> dict[str, int]:
    return {"d_model": 8, "num_heads": 2, "num_blocks": 1, "ffn_expansion": 2}


def test_untrained_network_matches_bilinear_pilot_image() -> None:
    """A zero output convolution leaves the upsampled LS image unchanged."""
    model = AxialTransformerChannelNet(12, 8, d_model=8, num_heads=2, num_blocks=1)
    pilots = torch.randn(2, 2, 3, 2)
    output = model(pilots, torch.tensor(0.2))
    expected = torch.nn.functional.interpolate(
        pilots,
        size=(12, 8),
        mode="bilinear",
        align_corners=False,
    )
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("system_variant", _SYSTEM_VARIANTS)
def test_estimator_output_matches_resource_grid(system_variant: str) -> None:
    """Inference returns a finite full-grid channel and nonnegative error variance."""
    cfg = system_variant_config(system_variant)
    resource_grid = build_resource_grid(cfg)
    architecture = _architecture()
    estimator = AxialTransformerChannelEstimator(
        resource_grid,
        [_state(resource_grid, **architecture)],
        precision=cfg.precision,
        device=cfg.device,
        **architecture,
    )
    received = torch.randn(
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.complex64,
    )
    channel_estimate, err_var = estimator(received, torch.tensor(0.1))
    assert channel_estimate.shape == (
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ue,
        cfg.num_streams_per_tx,
        cfg.num_ofdm_symbols,
        int(resource_grid.num_effective_subcarriers),
    )
    assert err_var.shape == channel_estimate.shape
    assert channel_estimate.dtype == torch.complex64
    assert bool(torch.isfinite(channel_estimate).all())
    assert bool(torch.isfinite(err_var).all())
    assert bool(torch.all(err_var >= 0.0))


@pytest.mark.compile
def test_estimator_matches_compiled_inference() -> None:
    """Eager and compiled inference agree for one mixed-SNR state."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    architecture = _architecture()
    estimator = AxialTransformerChannelEstimator(
        resource_grid,
        [_state(resource_grid, **architecture)],
        precision=cfg.precision,
        device=cfg.device,
        **architecture,
    )
    received = torch.randn(
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.complex64,
    )
    assert_compiled_estimator_matches_eager(
        estimator,
        received,
        torch.tensor(0.1, dtype=torch.float32),
        fullgraph=False,
    )


def test_load_model_reports_missing_asset(tmp_path) -> None:
    """Loading does not train when the external asset is absent."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    with pytest.raises(FileNotFoundError, match="prepare-assets"):
        load_model(
            resource_grid,
            cfg,
            asset_root=tmp_path,
            train_batches=1,
            validation_batches=1,
            epochs=1,
            **_architecture(),
        )


@pytest.mark.training
def test_train_model_reloads_a_finite_checkpoint(tmp_path) -> None:
    """One UMi epoch writes a checkpoint that the estimator can load."""
    cfg = system_variant_config("baseline", snr_db_start=5.0, snr_db_stop=5.0)
    resource_grid = build_resource_grid(cfg)
    architecture = _architecture()
    states, noise_variances, manifests = train_model(
        resource_grid,
        cfg,
        train_batches=1,
        train_batch_size=1,
        validation_batches=1,
        validation_batch_size=1,
        epochs=1,
        optimizer_batch_size=4,
        patience_epochs=0,
        seed=3,
        asset_root=tmp_path,
        **architecture,
    )
    assert noise_variances is None
    assert len(states) == 1
    assert manifests[0].is_file()
    estimator = AxialTransformerChannelEstimator(
        resource_grid,
        states,
        precision=cfg.precision,
        device=cfg.device,
        **architecture,
    )
    received = torch.randn(
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.complex64,
    )
    channel_estimate, err_var = estimator(received, torch.tensor(0.1))
    assert bool(torch.isfinite(channel_estimate).all())
    assert bool(torch.isfinite(err_var).all())
