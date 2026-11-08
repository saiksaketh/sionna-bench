# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full-grid channel estimator with axial attention on the OFDM grid."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Optional

import torch
from sionna.phy import Precision
from sionna.phy.object import Object
from sionna.phy.ofdm import BaseChannelEstimator, LSChannelEstimator, ResourceGrid
from sionna.phy.ofdm.channel_estimation import _PilotGroupMixin
from sionna.phy.utils import flatten_last_dims


DEFAULT_D_MODEL = 32
DEFAULT_NUM_HEADS = 4
DEFAULT_NUM_BLOCKS = 2
DEFAULT_FFN_EXPANSION = 2
DEFAULT_ERR_VAR_SCALE = 1.0

__all__ = [
    "DEFAULT_D_MODEL",
    "DEFAULT_ERR_VAR_SCALE",
    "DEFAULT_FFN_EXPANSION",
    "DEFAULT_NUM_BLOCKS",
    "DEFAULT_NUM_HEADS",
    "AxialTransformerBlock",
    "AxialTransformerChannelEstimator",
    "AxialTransformerChannelNet",
    "PilotImageIndex",
    "build_axial_channel_net",
    "examples_from_channel",
    "restore_pilot_observations",
]


class AxialTransformerBlock(torch.nn.Module):
    """Attention along OFDM time, then along subcarriers, then a 1x1 MLP."""

    def __init__(self, d_model: int, num_heads: int, ffn_expansion: int) -> None:
        super().__init__()
        self.time_norm = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.time_attention = torch.nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.freq_norm = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.freq_attention = torch.nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        hidden = d_model * ffn_expansion
        self.ffn_norm = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = torch.nn.Sequential(
            torch.nn.Conv2d(d_model, hidden, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden, d_model, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply one axial block.

        :param features: ``[batch, subcarriers, symbols, d_model]``.
        """
        batch, freq, time, channels = features.shape
        time_tokens = self.time_norm(features).reshape(batch * freq, time, channels)
        time_update, _ = self.time_attention(
            time_tokens,
            time_tokens,
            time_tokens,
            need_weights=False,
        )
        features = features + time_update.reshape(batch, freq, time, channels)

        freq_tokens = (
            self.freq_norm(features).permute(0, 2, 1, 3).reshape(batch * time, freq, channels)
        )
        freq_update, _ = self.freq_attention(
            freq_tokens,
            freq_tokens,
            freq_tokens,
            need_weights=False,
        )
        features = features + freq_update.reshape(batch, time, freq, channels).permute(0, 2, 1, 3)

        ffn_input = self.ffn_norm(features).permute(0, 3, 1, 2)
        ffn_update = self.ffn(ffn_input).permute(0, 2, 3, 1)
        return features + ffn_update


class AxialTransformerChannelNet(torch.nn.Module):
    """Map a low-resolution LS pilot image to a full OFDM channel grid.

    Bilinear interpolation lifts the pilot image onto the full
    subcarrier-by-symbol grid. Axial blocks then mix every subcarrier across
    OFDM symbols and every OFDM symbol across subcarriers. The output
    convolution is zero-initialized, so an untrained network reproduces the
    bilinearly interpolated LS estimate and training learns a residual.

    :param num_subcarriers: Full effective-subcarrier count.
    :param num_symbols: OFDM symbol count.
    :param d_model: Width of the axial residual stream.
    :param num_heads: Heads in each axial attention module.
    :param num_blocks: Number of time-then-frequency blocks.
    :param ffn_expansion: Hidden-width multiplier of the 1x1 feed-forward layer.

    :input pilot_ri: ``[batch, 2, pilot_subcarriers, pilot_symbols]``, float.
        Real and imaginary parts of the LS pilot image.
    :input noise_variance: Positive noise variance, scalar or ``[batch]``.

    :output: ``[batch, 2, num_subcarriers, num_symbols]``, float.
        Real and imaginary parts of the full channel image.
    """

    def __init__(
        self,
        num_subcarriers: int,
        num_symbols: int,
        *,
        d_model: int = DEFAULT_D_MODEL,
        num_heads: int = DEFAULT_NUM_HEADS,
        num_blocks: int = DEFAULT_NUM_BLOCKS,
        ffn_expansion: int = DEFAULT_FFN_EXPANSION,
    ) -> None:
        super().__init__()
        if num_subcarriers < 1 or num_symbols < 1:
            raise ValueError("Grid dimensions must be positive.")
        if d_model < 1 or num_blocks < 1 or ffn_expansion < 1 or num_heads < 1:
            raise ValueError("Architecture widths must be positive.")
        if d_model % num_heads != 0:
            raise ValueError("`d_model` must be divisible by `num_heads`.")

        self.num_subcarriers = int(num_subcarriers)
        self.num_symbols = int(num_symbols)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_blocks = int(num_blocks)
        self.ffn_expansion = int(ffn_expansion)

        freq = torch.linspace(0, 1, self.num_subcarriers)
        time = torch.linspace(0, 1, self.num_symbols)
        grid_f, grid_t = torch.meshgrid(freq, time, indexing="ij")
        self.register_buffer(
            "_pos",
            torch.stack([grid_f, grid_t], dim=-1),
            persistent=False,
        )
        self.input_conv = torch.nn.Conv2d(3, d_model, kernel_size=3, padding=1)
        self.pos_projection = torch.nn.Linear(2, d_model)
        self.blocks = torch.nn.ModuleList(
            [
                AxialTransformerBlock(d_model, num_heads, ffn_expansion)
                for _ in range(num_blocks)
            ]
        )
        self.output_norm = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.output_conv = torch.nn.Conv2d(d_model, 2, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.output_conv.weight)
        torch.nn.init.zeros_(self.output_conv.bias)

    def forward(
        self,
        pilot_ri: torch.Tensor,
        noise_variance: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the full real/imag channel image."""
        upsampled = torch.nn.functional.interpolate(
            pilot_ri,
            size=(self.num_subcarriers, self.num_symbols),
            mode="bilinear",
            align_corners=False,
        )
        log_noise = torch.log10(noise_variance.to(dtype=upsampled.dtype).reshape(-1) + 1e-8)
        if log_noise.numel() == 1:
            log_noise = log_noise.expand(upsampled.shape[0])
        elif log_noise.numel() != upsampled.shape[0]:
            log_noise = log_noise.mean().expand(upsampled.shape[0])
        noise_plane = log_noise.view(-1, 1, 1, 1).expand(
            -1,
            1,
            self.num_subcarriers,
            self.num_symbols,
        )
        features = self.input_conv(torch.cat([upsampled, noise_plane], dim=1))
        features = features.permute(0, 2, 3, 1)
        features = features + self.pos_projection(self._pos.to(dtype=features.dtype))
        for block in self.blocks:
            features = block(features)
        features = self.output_norm(features).permute(0, 3, 1, 2)
        return upsampled + self.output_conv(features)


def restore_pilot_observations(
    full_grid: torch.Tensor,
    pilots: torch.Tensor,
    subcarrier_image: torch.Tensor,
    symbol_image: torch.Tensor,
    *,
    groups_outer: bool,
) -> torch.Tensor:
    """Copy observed pilot values back onto their resource elements.

    The axial residual is free on data resource elements. Pilot resource
    elements keep the least-squares observation, so the residual cannot
    change a resource element the estimator has already observed.

    ``full_grid`` and ``pilots`` are ``[examples, 2, ...]``. Training
    concatenates transmit-stream groups as the outer axis. Inference flattens
    the group axis as the inner axis. ``subcarrier_image`` and
    ``symbol_image`` have shape ``[groups, pilot_subcarriers, pilot_symbols]``.
    """
    groups = int(subcarrier_image.shape[0])
    if full_grid.shape[0] % groups != 0 or pilots.shape[0] != full_grid.shape[0]:
        raise ValueError("Pilot observations do not match the full-grid batch.")
    per_group = full_grid.shape[0] // groups
    restored = full_grid.clone()
    if groups_outer:
        grid = restored.reshape(groups, per_group, *full_grid.shape[1:])
        observed = pilots.reshape(groups, per_group, *pilots.shape[1:])
    else:
        grid = restored.reshape(per_group, groups, *full_grid.shape[1:])
        observed = pilots.reshape(per_group, groups, *pilots.shape[1:])
    for group in range(groups):
        subcarriers = subcarrier_image[group, :, 0].to(device=full_grid.device, dtype=torch.long)
        symbols = symbol_image[group, 0, :].to(device=full_grid.device, dtype=torch.long)
        if groups_outer:
            grid[group, :, :, subcarriers[:, None], symbols[None, :]] = observed[group]
        else:
            grid[:, group, :, subcarriers[:, None], symbols[None, :]] = observed[:, group]
    return restored


def build_axial_channel_net(
    num_subcarriers: int,
    num_symbols: int,
    *,
    d_model: int = DEFAULT_D_MODEL,
    num_heads: int = DEFAULT_NUM_HEADS,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    ffn_expansion: int = DEFAULT_FFN_EXPANSION,
) -> AxialTransformerChannelNet:
    """Build the axial channel network for one resource-grid geometry."""
    return AxialTransformerChannelNet(
        num_subcarriers,
        num_symbols,
        d_model=d_model,
        num_heads=num_heads,
        num_blocks=num_blocks,
        ffn_expansion=ffn_expansion,
    )


def initialize_pilot_image(module, resource_grid: ResourceGrid) -> None:
    """Register the active rectangular pilot image used by the axial net."""
    module._init_pilot_groups(
        module._pilot_pattern,
        group_over=("subcarrier", "symbol"),
        persistent=False,
    )
    group_subcarriers = module._pilot_group_varying_coord("subcarrier")
    group_symbols = module._pilot_group_varying_coord("symbol")
    group_tx_indices = module._pilot_group_constant_coord("tx")
    group_stream_indices = module._pilot_group_constant_coord("stream")
    num_stream_groups = int(module._pilot_pattern.num_tx) * int(
        module._pilot_pattern.num_streams_per_tx
    )
    if module._pilot_group_num_groups != num_stream_groups:
        raise ValueError("Axial estimation expects one active pilot grid per stream.")

    unique_subcarriers = torch.unique(group_subcarriers[0], sorted=True)
    unique_symbols = torch.unique(group_symbols[0], sorted=True)
    num_pilot_subcarriers = int(unique_subcarriers.numel())
    num_pilot_ofdm_symbols = int(unique_symbols.numel())
    if num_pilot_subcarriers * num_pilot_ofdm_symbols != module._pilot_group_len:
        raise ValueError("Active pilots must form a rectangular time-frequency image.")

    subcarrier_image = group_subcarriers.reshape(
        module._pilot_group_num_groups,
        num_pilot_subcarriers,
        num_pilot_ofdm_symbols,
    )
    symbol_image = group_symbols.reshape(
        module._pilot_group_num_groups,
        num_pilot_subcarriers,
        num_pilot_ofdm_symbols,
    )
    if not bool(torch.all(subcarrier_image == subcarrier_image[:, :, :1])):
        raise ValueError("Pilot-image rows must keep a fixed subcarrier.")
    if not bool(torch.all(symbol_image == symbol_image[:, :1, :])):
        raise ValueError("Pilot-image columns must keep a fixed OFDM symbol.")

    pilot_values = module._gather_pilot_groups(
        module._pilot_pattern.pilots.to(module.device)
    ).to(dtype=module.cdtype, device=module.device)
    if not bool(torch.all(pilot_values.abs() > 0.0)):
        raise ValueError("Pilot groups must contain only nonzero pilot symbols.")

    module._num_pilot_subcarriers = num_pilot_subcarriers
    module._num_pilot_ofdm_symbols = num_pilot_ofdm_symbols
    module._num_effective_subcarriers = int(resource_grid.num_effective_subcarriers)
    module._num_ofdm_symbols = int(resource_grid.num_ofdm_symbols)
    module._fft_size = int(resource_grid.fft_size)
    effective_index = torch.as_tensor(resource_grid.effective_subcarrier_ind).to(
        device=module.device,
        dtype=torch.long,
    ).reshape(-1)
    layout_bytes = bytearray()
    layout_bytes.extend(
        module._pilot_group_coords.detach().cpu().contiguous().numpy().tobytes()
    )
    layout_bytes.extend(pilot_values.detach().cpu().contiguous().numpy().tobytes())
    module._pilot_layout_sha256 = hashlib.sha256(layout_bytes).hexdigest()
    module.register_buffer("_group_subcarrier_image", subcarrier_image, persistent=False)
    module.register_buffer("_group_symbol_image", symbol_image, persistent=False)
    module.register_buffer("_group_tx_indices", group_tx_indices, persistent=False)
    module.register_buffer("_group_stream_indices", group_stream_indices, persistent=False)
    module.register_buffer(
        "_group_pilot_power",
        pilot_values.abs().square().to(dtype=module.dtype),
        persistent=False,
    )
    module.register_buffer(
        "_effective_subcarrier_index",
        effective_index,
        persistent=False,
    )


def examples_from_channel(
    module,
    frequency_response: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split one OFDM channel batch into pilot images, targets, and powers.

    :param frequency_response: ``[batch, num_rx, num_rx_ant, num_tx, num_tx_ant,
        num_ofdm_symbols, num_subcarriers]``, complex. The last axis is either
        the FFT size or the effective-subcarrier count.
    :returns: ``(pilots, targets, powers)`` with shapes
        ``[examples, 2, pilot_subcarriers, pilot_symbols]``,
        ``[examples, 2, num_effective_subcarriers, num_ofdm_symbols]``, and
        ``[examples, 1, pilot_subcarriers, pilot_symbols]``.
    """
    if frequency_response.shape[-1] == module._fft_size:
        channel = frequency_response.index_select(-1, module._effective_subcarrier_index)
    elif frequency_response.shape[-1] == module._num_effective_subcarriers:
        channel = frequency_response
    else:
        raise ValueError(
            "Channel frequency response must end in the FFT size or the "
            "effective-subcarrier count."
        )
    if channel.shape[4] < int(module._group_stream_indices.max().item()) + 1:
        raise ValueError(
            "Axial training indexes the transmit-antenna axis with the stream index."
        )

    pilot_parts = []
    target_parts = []
    power_parts = []
    num_pilot_subcarriers = module._num_pilot_subcarriers
    num_pilot_symbols = module._num_pilot_ofdm_symbols
    for group_index in range(module._pilot_group_num_groups):
        tx_index = int(module._group_tx_indices[group_index].item())
        stream_index = int(module._group_stream_indices[group_index].item())
        symbols = module._group_symbol_image[group_index].reshape(-1)
        subcarriers = module._group_subcarrier_image[group_index].reshape(-1)
        values = channel[:, :, :, tx_index, stream_index]
        pilot_channel = values[..., symbols, subcarriers].reshape(
            -1,
            num_pilot_subcarriers,
            num_pilot_symbols,
        )
        target = values.reshape(
            -1,
            module._num_ofdm_symbols,
            module._num_effective_subcarriers,
        ).transpose(-2, -1)
        power = (
            module._group_pilot_power[group_index]
            .reshape(num_pilot_subcarriers, num_pilot_symbols)
            .to(dtype=torch.float32)
        )
        pilot_parts.append(torch.stack([pilot_channel.real, pilot_channel.imag], dim=1))
        target_parts.append(torch.stack([target.real, target.imag], dim=1))
        power_parts.append(power.expand(pilot_channel.shape[0], 1, *power.shape))
    pilots = torch.cat(pilot_parts, dim=0).to(dtype=torch.float32)
    targets = torch.cat(target_parts, dim=0).to(dtype=torch.float32)
    powers = torch.cat(power_parts, dim=0).to(dtype=torch.float32)
    return pilots, targets, powers


class PilotImageIndex(Object, _PilotGroupMixin):
    """Pilot-image coordinates for offline axial training."""

    def __init__(
        self,
        resource_grid: ResourceGrid,
        *,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)
        self._pilot_pattern = resource_grid.pilot_pattern
        initialize_pilot_image(self, resource_grid)


class AxialTransformerChannelEstimator(BaseChannelEstimator, _PilotGroupMixin):
    r"""Channel estimator that refines interpolated LS pilots with axial attention.

    Least-squares estimates at the active pilot resource elements are arranged
    as a real/imaginary image, bilinearly lifted onto the full OFDM
    time-frequency grid, and refined by axial attention. Attention along OFDM
    symbols is separate from attention along subcarriers, matching the grid
    structure of the channel. One weight set is shared across receive antennas
    and transmit streams.

    ``model_noise_variances`` selects an SNR-specific weight set. With no
    variances, the single supplied state is used at every SNR. The network
    also receives ``log10`` of the noise variance as an input channel.

    Detector-facing error variances are the bilinearly upsampled LS pilot
    error variances scaled by ``err_var_scale``. That scale is a benchmark
    interface parameter and is not part of the axial residual.

    :param resource_grid: Sionna resource grid.
    :param model_states: Axial network state dictionaries. Every state must
        match ``resource_grid`` and the architecture arguments.
    :param model_noise_variances: Positive AWGN variance for each state, shape
        ``[num_models]``. Required when more than one state is supplied.
    :param d_model: Axial feature width.
    :param num_heads: Attention heads in each axial block.
    :param num_blocks: Number of axial blocks.
    :param ffn_expansion: Feed-forward width multiplier.
    :param err_var_scale: Nonnegative scale on the detector-facing error variance.
    :param anchor_pilots: Copy least-squares pilot observations onto the
        estimated grid before returning it.
    :param precision: Sionna precision. If `None`, the global precision is used.
    :param device: Torch device. If `None`, the global Sionna device is used.
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        model_states: Sequence[Mapping[str, torch.Tensor]],
        model_noise_variances: Optional[torch.Tensor] = None,
        *,
        d_model: int = DEFAULT_D_MODEL,
        num_heads: int = DEFAULT_NUM_HEADS,
        num_blocks: int = DEFAULT_NUM_BLOCKS,
        ffn_expansion: int = DEFAULT_FFN_EXPANSION,
        err_var_scale: float = DEFAULT_ERR_VAR_SCALE,
        anchor_pilots: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        if len(model_states) == 0:
            raise ValueError("At least one axial state is required.")
        if err_var_scale < 0.0:
            raise ValueError("`err_var_scale` must be non-negative.")
        super().__init__(
            resource_grid=resource_grid,
            precision=precision,
            device=device,
        )
        self._d_model = int(d_model)
        self._num_heads = int(num_heads)
        self._num_blocks = int(num_blocks)
        self._ffn_expansion = int(ffn_expansion)
        self._err_var_scale = float(err_var_scale)
        self._anchor_pilots = bool(anchor_pilots)
        initialize_pilot_image(self, resource_grid)
        self._models = torch.nn.ModuleList(
            self._build_model(state) for state in model_states
        )
        if model_noise_variances is None:
            if len(model_states) != 1:
                raise ValueError("Multiple axial states require `model_noise_variances`.")
            noise_variances = torch.empty(0, dtype=self.dtype, device=self.device)
        else:
            noise_variances = torch.as_tensor(
                model_noise_variances,
                dtype=self.dtype,
                device=self.device,
            ).flatten()
            if int(noise_variances.numel()) != len(model_states):
                raise ValueError("`model_noise_variances` must have one entry per state.")
            if not bool(torch.all(noise_variances > 0.0)):
                raise ValueError("`model_noise_variances` must be positive.")
        self.register_buffer("_model_noise_variances", noise_variances)

    def _build_model(self, state: Mapping[str, torch.Tensor]) -> AxialTransformerChannelNet:
        """Build one axial network and load a prepared state."""
        model = build_axial_channel_net(
            self._num_effective_subcarriers,
            self._num_ofdm_symbols,
            d_model=self._d_model,
            num_heads=self._num_heads,
            num_blocks=self._num_blocks,
            ffn_expansion=self._ffn_expansion,
        ).to(device=self.device)
        model.load_state_dict(state)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    @torch.compiler.disable
    def _select_model_index(self, no: torch.Tensor) -> int:
        """Select one prepared SNR-specific state outside compiled inference."""
        noise_variance = no.to(
            dtype=self._model_noise_variances.dtype,
            device=self._model_noise_variances.device,
        ).reshape(-1).mean()
        log_distance = torch.abs(
            torch.log(noise_variance) - torch.log(self._model_noise_variances)
        )
        model_index = int(torch.argmin(log_distance).item())
        if not bool(log_distance[model_index] < 1e-3):
            raise ValueError(
                "No SNR-specific axial network is available for the current noise variance."
            )
        return model_index

    def _gather_y_pilots(self, y: torch.Tensor) -> torch.Tensor:
        """Gather Sionna compact pilot observations from ``y``."""
        y_eff = self._remove_nulled_scs(y)
        y_eff_flat = flatten_last_dims(y_eff)
        pilot_ind = self._pilot_ind.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        pilot_ind = pilot_ind.expand(
            y_eff_flat.shape[0],
            y_eff_flat.shape[1],
            y_eff_flat.shape[2],
            -1,
            -1,
            -1,
        )
        return torch.gather(
            y_eff_flat.unsqueeze(3)
            .unsqueeze(4)
            .expand(-1, -1, -1, pilot_ind.shape[3], pilot_ind.shape[4], -1),
            -1,
            pilot_ind,
        )

    def call(
        self,
        y: torch.Tensor,
        no: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate the full effective resource-grid channel.

        :input y: ``[batch, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]``,
            complex. Received OFDM resource grid.
        :input no: Float noise variance with a Sionna-compatible broadcast shape.

        :output h_hat: ``[batch, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            num_ofdm_symbols, num_effective_subcarriers]``, complex.
        :output err_var: Same shape as ``h_hat``, float.
        """
        y_pilots = self._gather_y_pilots(y)
        h_ls, ls_err = LSChannelEstimator.estimate_at_pilot_locations(self, y_pilots, no)
        grouped_h_ls = self._gather_pilot_groups(h_ls)
        grouped_ls_err = self._gather_pilot_groups(ls_err)
        leading_shape = grouped_h_ls.shape[:-1]
        h_images = grouped_h_ls.reshape(
            *leading_shape,
            self._num_pilot_subcarriers,
            self._num_pilot_ofdm_symbols,
        )
        err_images = grouped_ls_err.reshape(
            *leading_shape,
            self._num_pilot_subcarriers,
            self._num_pilot_ofdm_symbols,
        )
        model_inputs = torch.stack([h_images.real, h_images.imag], dim=-3).reshape(
            -1,
            2,
            self._num_pilot_subcarriers,
            self._num_pilot_ofdm_symbols,
        ).to(dtype=torch.float32)
        if self._model_noise_variances.numel() == 0:
            model = self._models[0]
        else:
            model = self._models[self._select_model_index(no)]
        with torch.no_grad():
            model_outputs = model(model_inputs, no)
        if self._anchor_pilots:
            model_outputs = restore_pilot_observations(
                model_outputs,
                model_inputs,
                self._group_subcarrier_image,
                self._group_symbol_image,
                groups_outer=False,
            )
        full_images = torch.complex(
            model_outputs[:, 0].to(dtype=self.dtype),
            model_outputs[:, 1].to(dtype=self.dtype),
        )
        full_images = full_images.reshape(
            *leading_shape,
            self._num_effective_subcarriers,
            self._num_ofdm_symbols,
        ).transpose(-2, -1)

        err_inputs = err_images.reshape(
            -1,
            1,
            self._num_pilot_subcarriers,
            self._num_pilot_ofdm_symbols,
        )
        err_outputs = torch.nn.functional.interpolate(
            err_inputs,
            size=(self._num_effective_subcarriers, self._num_ofdm_symbols),
            mode="bilinear",
            align_corners=False,
        )
        err_outputs = self._err_var_scale * err_outputs.reshape(
            *leading_shape,
            self._num_effective_subcarriers,
            self._num_ofdm_symbols,
        ).transpose(-2, -1)
        output_shape = (
            *y_pilots.shape[:3],
            int(self._pilot_pattern.num_tx),
            int(self._pilot_pattern.num_streams_per_tx),
            self._num_ofdm_symbols,
            self._num_effective_subcarriers,
        )
        h_hat = full_images.reshape(output_shape)
        err_var = torch.clamp(err_outputs.reshape(output_shape), min=0.0)
        return h_hat, err_var
