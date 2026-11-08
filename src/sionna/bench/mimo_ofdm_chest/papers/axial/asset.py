# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train and load axial-attention channel-estimation weights."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from sionna.phy.channel import GenerateOFDMChannel, gen_single_sector_topology
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.utils import ebnodb2no

from sionna.bench.mimo_ofdm_chest.asset_store import (
    build_asset_manifest,
    external_asset_dir,
    load_asset_manifest,
    paper_asset_dir,
)
from sionna.bench.mimo_ofdm_chest.benchmarking import UMiChannelSource, set_random_seed
from sionna.bench.mimo_ofdm_chest.configs import SystemConfig
from sionna.bench.mimo_ofdm_chest.persistence import canonical_json, write_json_atomic, write_npz_atomic

from .estimator import (
    DEFAULT_D_MODEL,
    DEFAULT_FFN_EXPANSION,
    DEFAULT_NUM_BLOCKS,
    DEFAULT_NUM_HEADS,
    PilotImageIndex,
    build_axial_channel_net,
    examples_from_channel,
)


_ASSET_TYPE = "axial-channel"
_PAPER_SLUG = "axial"
CANONICAL_TRAIN_BATCHES = 8
CANONICAL_TRAIN_BATCH_SIZE = 1
CANONICAL_VALIDATION_BATCHES = 2
CANONICAL_EPOCHS = 5
CANONICAL_OPTIMIZER_BATCH_SIZE = 4
CANONICAL_LEARNING_RATE = 1e-3
CANONICAL_PATIENCE_EPOCHS = 3
CANONICAL_RELATIVE_MIN_DELTA = 1e-4

__all__ = [
    "load_model",
    "train_model",
    "training_settings",
]


def training_settings(smoke: bool) -> dict[str, Any]:
    """Return the smoke or canonical axial training budget."""
    if smoke:
        return {
            "train_batches": 1,
            "train_batch_size": 1,
            "validation_batches": 1,
            "validation_batch_size": 1,
            "epochs": 1,
            "optimizer_batch_size": 4,
            "learning_rate": CANONICAL_LEARNING_RATE,
            "patience_epochs": 0,
            "relative_min_delta": 0.0,
            "snr_specific_weights": False,
        }
    return {
        "train_batches": CANONICAL_TRAIN_BATCHES,
        "train_batch_size": CANONICAL_TRAIN_BATCH_SIZE,
        "validation_batches": CANONICAL_VALIDATION_BATCHES,
        "validation_batch_size": CANONICAL_TRAIN_BATCH_SIZE,
        "epochs": CANONICAL_EPOCHS,
        "optimizer_batch_size": CANONICAL_OPTIMIZER_BATCH_SIZE,
        "learning_rate": CANONICAL_LEARNING_RATE,
        "patience_epochs": CANONICAL_PATIENCE_EPOCHS,
        "relative_min_delta": CANONICAL_RELATIVE_MIN_DELTA,
        "snr_specific_weights": False,
    }


def load_model(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    train_batches: int = CANONICAL_TRAIN_BATCHES,
    train_batch_size: int = CANONICAL_TRAIN_BATCH_SIZE,
    validation_batches: int = CANONICAL_VALIDATION_BATCHES,
    validation_batch_size: Optional[int] = CANONICAL_TRAIN_BATCH_SIZE,
    epochs: int = CANONICAL_EPOCHS,
    optimizer_batch_size: int = CANONICAL_OPTIMIZER_BATCH_SIZE,
    learning_rate: float = CANONICAL_LEARNING_RATE,
    patience_epochs: int = CANONICAL_PATIENCE_EPOCHS,
    relative_min_delta: float = CANONICAL_RELATIVE_MIN_DELTA,
    snr_specific_weights: bool = False,
    training_snr_dbs: Optional[list[float]] = None,
    d_model: int = DEFAULT_D_MODEL,
    num_heads: int = DEFAULT_NUM_HEADS,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    ffn_expansion: int = DEFAULT_FFN_EXPANSION,
    seed: int = 42,
    asset_root: Optional[Union[str, Path]] = None,
) -> tuple[list[dict[str, torch.Tensor]], Optional[torch.Tensor], list[Path]]:
    """Load prepared axial weights without training or writing.

    :returns: ``(model_states, model_noise_variances, manifest_paths)``.
        ``model_noise_variances`` is `None` for a mixed-SNR checkpoint.
    """
    contexts, noise_variances = _model_contexts(
        resource_grid,
        cfg,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        epochs=epochs,
        optimizer_batch_size=optimizer_batch_size,
        learning_rate=learning_rate,
        patience_epochs=patience_epochs,
        relative_min_delta=relative_min_delta,
        snr_specific_weights=snr_specific_weights,
        training_snr_dbs=training_snr_dbs,
        d_model=d_model,
        num_heads=num_heads,
        num_blocks=num_blocks,
        ffn_expansion=ffn_expansion,
        seed=seed,
        asset_root=asset_root,
    )
    states = []
    manifest_paths = []
    for context in contexts:
        state, manifest_path = _load_state(context)
        states.append(state)
        manifest_paths.append(manifest_path)
    return states, noise_variances, manifest_paths


def train_model(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    train_batches: int = CANONICAL_TRAIN_BATCHES,
    train_batch_size: int = CANONICAL_TRAIN_BATCH_SIZE,
    validation_batches: int = CANONICAL_VALIDATION_BATCHES,
    validation_batch_size: Optional[int] = CANONICAL_TRAIN_BATCH_SIZE,
    epochs: int = CANONICAL_EPOCHS,
    optimizer_batch_size: int = CANONICAL_OPTIMIZER_BATCH_SIZE,
    learning_rate: float = CANONICAL_LEARNING_RATE,
    patience_epochs: int = CANONICAL_PATIENCE_EPOCHS,
    relative_min_delta: float = CANONICAL_RELATIVE_MIN_DELTA,
    snr_specific_weights: bool = False,
    training_snr_dbs: Optional[list[float]] = None,
    d_model: int = DEFAULT_D_MODEL,
    num_heads: int = DEFAULT_NUM_HEADS,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    ffn_expansion: int = DEFAULT_FFN_EXPANSION,
    seed: int = 42,
    asset_root: Union[str, Path],
    overwrite: bool = False,
    resume: bool = False,
) -> tuple[list[dict[str, torch.Tensor]], Optional[torch.Tensor], list[Path]]:
    """Train axial weights on the benchmark UMi channel and reload them.

    Each example is one receive-antenna and transmit-stream channel. The loss
    is the normalized mean-square error of the full channel grid. Every epoch
    appends ``history.jsonl`` and writes a weight checkpoint under a sibling
    ``<artifact>-run`` directory. ``resume=True`` continues from ``latest.pt``.
    Observation noise is applied to the clean pilot image at the LS variance
    ``no / |pilot|^2``. A mixed-SNR run stores one checkpoint. An SNR-specific
    run stores one checkpoint per requested SNR.

    :param asset_root: Writable external root. Training never writes into the
        package tree.
    :param overwrite: Replace an existing asset with the same identity.
    """
    if asset_root is None:
        raise ValueError("`train_model` requires an explicit external `asset_root`.")
    contexts, _ = _model_contexts(
        resource_grid,
        cfg,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        epochs=epochs,
        optimizer_batch_size=optimizer_batch_size,
        learning_rate=learning_rate,
        patience_epochs=patience_epochs,
        relative_min_delta=relative_min_delta,
        snr_specific_weights=snr_specific_weights,
        training_snr_dbs=training_snr_dbs,
        d_model=d_model,
        num_heads=num_heads,
        num_blocks=num_blocks,
        ffn_expansion=ffn_expansion,
        seed=seed,
        asset_root=asset_root,
    )
    for context in contexts:
        if context["asset_dir"].exists() and not overwrite:
            raise FileExistsError(
                f"Axial asset already exists at {context['asset_dir']}. "
                "Pass overwrite=True to replace it."
            )
    layout = PilotImageIndex(
        resource_grid,
        precision=cfg.precision,
        device=cfg.device,
    )
    for context in contexts:
        _train_context(
            resource_grid,
            cfg,
            layout,
            context,
            epochs=epochs,
            optimizer_batch_size=optimizer_batch_size,
            learning_rate=learning_rate,
            patience_epochs=patience_epochs,
            relative_min_delta=relative_min_delta,
            seed=seed,
            overwrite=overwrite,
            resume=resume,
        )
    return load_model(
        resource_grid,
        cfg,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        epochs=epochs,
        optimizer_batch_size=optimizer_batch_size,
        learning_rate=learning_rate,
        patience_epochs=patience_epochs,
        relative_min_delta=relative_min_delta,
        snr_specific_weights=snr_specific_weights,
        training_snr_dbs=training_snr_dbs,
        d_model=d_model,
        num_heads=num_heads,
        num_blocks=num_blocks,
        ffn_expansion=ffn_expansion,
        seed=seed,
        asset_root=asset_root,
    )


def _training_snr_list(
    cfg: SystemConfig,
    training_snr_dbs: Optional[list[float]],
) -> list[float]:
    if training_snr_dbs is None:
        return [float(snr) for snr in cfg.snr_db_range()]
    return [float(snr) for snr in training_snr_dbs]


def _recipe(
    *,
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: int,
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    patience_epochs: int,
    relative_min_delta: float,
    snr_dbs: list[float],
    seed: int,
    precision: str,
) -> dict[str, Any]:
    return {
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "loss": "nmse",
        "optimizer": "adam",
        "optimizer_batch_size": int(optimizer_batch_size),
        "patience_epochs": int(patience_epochs),
        "precision": precision,
        "relative_min_delta": float(relative_min_delta),
        "seed": int(seed),
        "snr_db": [float(snr) for snr in snr_dbs],
        "train_batch_size": int(train_batch_size),
        "train_batches": int(train_batches),
        "validation_batch_size": int(validation_batch_size),
        "validation_batches": int(validation_batches),
    }


def _artifact_id(architecture: dict[str, int], mode: str, recipe: dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(recipe).encode("utf-8")).hexdigest()[:12]
    return (
        f"d{architecture['d_model']}-h{architecture['num_heads']}-"
        f"b{architecture['num_blocks']}-ffn{architecture['ffn_expansion']}-"
        f"{mode}-{digest}"
    )


def _snr_token(snr_db: float) -> str:
    sign = "m" if snr_db < 0.0 else ""
    return sign + f"{abs(snr_db):.1f}".replace(".", "p")


def _model_contexts(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: Optional[int],
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    patience_epochs: int,
    relative_min_delta: float,
    snr_specific_weights: bool,
    training_snr_dbs: Optional[list[float]],
    d_model: int,
    num_heads: int,
    num_blocks: int,
    ffn_expansion: int,
    seed: int,
    asset_root: Optional[Union[str, Path]],
) -> tuple[list[dict[str, Any]], Optional[torch.Tensor]]:
    validation_batch_size = (
        train_batch_size if validation_batch_size is None else validation_batch_size
    )
    layout = PilotImageIndex(
        resource_grid,
        precision=cfg.precision,
        device=cfg.device,
    )
    architecture = {
        "d_model": int(d_model),
        "ffn_expansion": int(ffn_expansion),
        "num_blocks": int(num_blocks),
        "num_heads": int(num_heads),
    }
    compatibility = {
        "format": "axial-ls-bilinear-residual",
        **architecture,
        "num_effective_subcarriers": int(layout._num_effective_subcarriers),
        "num_ofdm_symbols": int(layout._num_ofdm_symbols),
        "num_pilot_ofdm_symbols": int(layout._num_pilot_ofdm_symbols),
        "num_pilot_subcarriers": int(layout._num_pilot_subcarriers),
        "num_streams_per_tx": int(resource_grid.num_streams_per_tx),
        "num_tx": int(resource_grid.num_tx),
        "pilot_layout_sha256": layout._pilot_layout_sha256,
    }
    snr_dbs = _training_snr_list(cfg, training_snr_dbs)
    groups = [[snr] for snr in snr_dbs] if snr_specific_weights else [snr_dbs]

    def context_for(group: list[float]) -> dict[str, Any]:
        mode = "mixed" if len(group) != 1 or not snr_specific_weights else f"snr-{_snr_token(group[0])}"
        if snr_specific_weights:
            mode = f"snr-{_snr_token(group[0])}"
        recipe = _recipe(
            train_batches=train_batches,
            train_batch_size=train_batch_size,
            validation_batches=validation_batches,
            validation_batch_size=validation_batch_size,
            epochs=epochs,
            optimizer_batch_size=optimizer_batch_size,
            learning_rate=learning_rate,
            patience_epochs=patience_epochs,
            relative_min_delta=relative_min_delta,
            snr_dbs=group,
            seed=seed,
            precision=cfg.precision,
        )
        artifact_id = _artifact_id(architecture, mode, recipe)
        if asset_root is None:
            asset_dir = paper_asset_dir(cfg, _ASSET_TYPE, artifact_id, __file__)
        else:
            asset_dir = external_asset_dir(
                cfg,
                _PAPER_SLUG,
                _ASSET_TYPE,
                artifact_id,
                asset_root=asset_root,
            )
        return {
            "asset_dir": asset_dir,
            "scenario_id": cfg.scenario_id,
            "compatibility": compatibility,
            "recipe": recipe,
            "snr_dbs": group,
            "architecture": architecture,
        }

    contexts = [context_for(group) for group in groups]
    if not snr_specific_weights:
        return contexts, None
    snr_tensor = torch.tensor(
        snr_dbs,
        dtype=torch.float64 if cfg.precision == "double" else torch.float32,
        device=resource_grid.device,
    )
    noise_variances = ebnodb2no(
        snr_tensor,
        cfg.num_bits_per_symbol,
        cfg.coderate,
        resource_grid,
        precision=cfg.precision,
        device=cfg.device,
    ).detach().cpu()
    return contexts, noise_variances


def _missing_asset_message(asset_dir: Path) -> str:
    root = asset_dir.parents[3]
    return (
        f"Axial channel weights were not found at {asset_dir}. Train them with "
        "`python -m sionna.bench.mimo_ofdm_chest.papers.axial.run_benchmark "
        f"--prepare-assets --asset-root {root}` using the same architecture, "
        "SNR mode, and training budget, then evaluate with "
        "`python -m sionna.bench.mimo_ofdm_chest.papers.axial.run_benchmark "
        f"--asset-root {root}`."
    )


def _load_state(context: dict[str, Any]) -> tuple[dict[str, torch.Tensor], Path]:
    asset_dir = Path(context["asset_dir"])
    manifest_path = asset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(_missing_asset_message(asset_dir))
    load_asset_manifest(
        manifest_path,
        asset_dir=asset_dir,
        expected_asset_type=_ASSET_TYPE,
        expected_artifact_id=asset_dir.name,
        expected_compatibility=context["compatibility"],
    )
    payload = asset_dir / "weights.npz"
    with np.load(payload) as archive:
        state = {
            key: torch.from_numpy(np.array(archive[key]))
            for key in archive.files
        }
    return state, manifest_path


def _state_to_numpy(state: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        key: value.detach().cpu().contiguous().numpy()
        for key, value in state.items()
    }


def _write_asset(
    context: dict[str, Any],
    state: dict[str, torch.Tensor],
    history: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    asset_dir = Path(context["asset_dir"])
    if asset_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Axial asset already exists at {asset_dir}. Pass overwrite=True to replace it."
        )
    asset_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{asset_dir.name}.", dir=asset_dir.parent))
    try:
        weights_path = temporary_dir / "weights.npz"
        write_npz_atomic(weights_path, **_state_to_numpy(state))
        manifest = build_asset_manifest(
            cfg=_config_for_manifest(context),
            asset_type=_ASSET_TYPE,
            artifact_id=asset_dir.name,
            compatibility=context["compatibility"],
            generation_recipe=context["recipe"],
            payload_paths=[weights_path],
            metadata=history,
        )
        write_json_atomic(temporary_dir / "manifest.json", manifest)
        if asset_dir.exists():
            shutil.rmtree(asset_dir)
        os.replace(temporary_dir, asset_dir)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


class _ManifestConfig:
    """Minimal config view used by ``build_asset_manifest``."""

    def __init__(self, scenario_id: str) -> None:
        self.scenario_id = scenario_id


def _config_for_manifest(context: dict[str, Any]):
    """Use the system scenario id, not a parent directory of the asset root."""
    return _ManifestConfig(context["scenario_id"])


def _sample_channel(
    sampler: GenerateOFDMChannel,
    channel_model,
    cfg: SystemConfig,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    set_random_seed(seed)
    topology = gen_single_sector_topology(
        batch_size=batch_size,
        num_ut=cfg.num_ue,
        scenario="umi",
        min_ut_velocity=cfg.ue_speed,
        max_ut_velocity=cfg.ue_speed,
        precision=cfg.precision,
        device=cfg.device,
    )
    channel_model.set_topology(*topology)
    return sampler(batch_size)


def _noise_variance(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    snr_db: float,
) -> float:
    variance = ebnodb2no(
        torch.tensor(float(snr_db), dtype=torch.float32),
        cfg.num_bits_per_symbol,
        cfg.coderate,
        resource_grid,
        precision=cfg.precision,
        device=cfg.device,
    )
    return float(variance.reshape(-1)[0].item())


def _add_pilot_noise(
    pilots: torch.Tensor,
    powers: torch.Tensor,
    noise_variance: float,
    generator: torch.Generator,
) -> torch.Tensor:
    noise_std = torch.sqrt(
        torch.tensor(noise_variance, dtype=pilots.dtype, device=pilots.device) / (2.0 * powers)
    )
    noise = torch.randn(
        pilots.shape,
        generator=generator,
        device=pilots.device,
        dtype=pilots.dtype,
    )
    return pilots + noise_std * noise


def _run_directory(asset_dir: Path) -> Path:
    """Directory of epoch checkpoints and the training history."""
    return asset_dir.parent / f"{asset_dir.name}-run"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _channel_errors(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-example squared error, channel power, and L1 mass."""
    squared_error = (prediction - target).square().sum(dim=(1, 2, 3))
    power = target.square().sum(dim=(1, 2, 3))
    absolute_error = (prediction - target).abs().sum(dim=(1, 2, 3))
    return squared_error, power, absolute_error


def _run_epoch(
    model: torch.nn.Module,
    *,
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    layout: PilotImageIndex,
    sampler: GenerateOFDMChannel,
    channel_model,
    snr_dbs: list[float],
    num_batches: int,
    batch_size: int,
    optimizer_batch_size: int,
    seed: int,
    optimizer: Optional[torch.optim.Optimizer],
) -> tuple[float, float]:
    """Return epoch NMSE and mean L1."""
    training = optimizer is not None
    model.train(training)
    squared_error_sum = 0.0
    power_sum = 0.0
    absolute_error_sum = 0.0
    value_count = 0
    device = torch.device(cfg.device)
    for batch_index in range(num_batches):
        snr_db = snr_dbs[batch_index % len(snr_dbs)]
        frequency_response = _sample_channel(
            sampler,
            channel_model,
            cfg,
            batch_size,
            seed + batch_index,
        )
        pilots, targets, powers = examples_from_channel(layout, frequency_response)
        noise_variance = _noise_variance(resource_grid, cfg, snr_db)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + 100003 * (batch_index + 1))
        pilots = _add_pilot_noise(pilots, powers, noise_variance, generator)
        noise_tensor = torch.tensor(noise_variance, dtype=torch.float32, device=device)
        for start in range(0, int(pilots.shape[0]), optimizer_batch_size):
            stop = start + optimizer_batch_size
            pilot_batch = pilots[start:stop].to(device=device, dtype=torch.float32)
            target_batch = targets[start:stop].to(device=device, dtype=torch.float32)
            if training:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(pilot_batch, noise_tensor)
                squared_error, power, absolute_error = _channel_errors(
                    prediction, target_batch
                )
                loss = squared_error.sum() / power.sum().clamp_min(1e-12)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    prediction = model(pilot_batch, noise_tensor)
                    squared_error, power, absolute_error = _channel_errors(
                        prediction, target_batch
                    )
            squared_error_sum += float(squared_error.detach().sum().item())
            power_sum += float(power.detach().sum().item())
            absolute_error_sum += float(absolute_error.detach().sum().item())
            value_count += int(target_batch.numel())
    nmse = squared_error_sum / max(power_sum, 1e-12)
    mean_l1 = absolute_error_sum / max(value_count, 1)
    return nmse, mean_l1


def _train_context(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    layout: PilotImageIndex,
    context: dict[str, Any],
    *,
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    patience_epochs: int,
    relative_min_delta: float,
    seed: int,
    overwrite: bool,
    resume: bool,
) -> None:
    device = torch.device(cfg.device)
    architecture = context["architecture"]
    fork_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        model = build_axial_channel_net(
            layout._num_effective_subcarriers,
            layout._num_ofdm_symbols,
            d_model=architecture["d_model"],
            num_heads=architecture["num_heads"],
            num_blocks=architecture["num_blocks"],
            ffn_expansion=architecture["ffn_expansion"],
        ).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    channel_source = UMiChannelSource(cfg, resource_grid)
    sampler = GenerateOFDMChannel(
        channel_source.channel_model,
        resource_grid,
        normalize_channel=False,
        precision=cfg.precision,
        device=cfg.device,
    )
    run_dir = _run_directory(Path(context["asset_dir"]))
    latest_path = run_dir / "latest.pt"
    if overwrite and run_dir.exists() and not resume:
        shutil.rmtree(run_dir)
    if latest_path.exists() and not resume and not overwrite:
        raise FileExistsError(
            f"Training checkpoints already exist at {run_dir}. "
            "Pass resume=True to continue, or overwrite=True to start over."
        )
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    best_val = float("inf")
    best_epoch = 0
    start_epoch = 0
    stale_epochs = 0
    if resume and latest_path.is_file():
        payload = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        best_state = payload["best_model"]
        best_val = float(payload["best_val_nmse"])
        best_epoch = int(payload["best_epoch"])
        start_epoch = int(payload["epoch"])
        stale_epochs = int(payload.get("stale_epochs", 0))
        print(f"[axial] resumed {latest_path} at epoch {start_epoch}/{epochs}")
    elif resume:
        print(f"[axial] no checkpoint at {latest_path}; starting from scratch")
    started = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[axial] training {context['asset_dir'].name} "
        f"snrs={context['snr_dbs']} device={cfg.device}"
    )
    print(f"[axial] track with: tail -f {run_dir / 'history.jsonl'}")
    for epoch_index in range(start_epoch, int(epochs)):
        epoch_seed = seed + 1000 * (epoch_index + 1)
        train_nmse, train_l1 = _run_epoch(
            model,
            resource_grid=resource_grid,
            cfg=cfg,
            layout=layout,
            sampler=sampler,
            channel_model=channel_source.channel_model,
            snr_dbs=context["snr_dbs"],
            num_batches=int(context["recipe"]["train_batches"]),
            batch_size=int(context["recipe"]["train_batch_size"]),
            optimizer_batch_size=int(optimizer_batch_size),
            seed=epoch_seed,
            optimizer=optimizer,
        )
        val_nmse, val_l1 = _run_epoch(
            model,
            resource_grid=resource_grid,
            cfg=cfg,
            layout=layout,
            sampler=sampler,
            channel_model=channel_source.channel_model,
            snr_dbs=context["snr_dbs"],
            num_batches=int(context["recipe"]["validation_batches"]),
            batch_size=int(context["recipe"]["validation_batch_size"]),
            optimizer_batch_size=int(optimizer_batch_size),
            seed=seed + 500000,
            optimizer=None,
        )
        improved = val_nmse < best_val * (1.0 - float(relative_min_delta))
        model_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        if improved:
            best_val = float(val_nmse)
            best_epoch = epoch_index + 1
            stale_epochs = 0
            best_state = model_state
        else:
            stale_epochs += 1
        epoch_name = f"epoch-{epoch_index + 1:04d}.pt"
        checkpoint = {
            "epoch": epoch_index + 1,
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "best_model": best_state,
            "best_val_nmse": best_val,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "train_nmse": train_nmse,
            "val_nmse": val_nmse,
        }
        _save_checkpoint(run_dir / epoch_name, checkpoint)
        _save_checkpoint(latest_path, checkpoint)
        if improved:
            _save_checkpoint(run_dir / "best.pt", checkpoint)
        record = {
            "epoch": epoch_index + 1,
            "epochs": int(epochs),
            "train_nmse": train_nmse,
            "val_nmse": val_nmse,
            "train_l1": train_l1,
            "val_l1": val_l1,
            "best_epoch": best_epoch,
            "best_val_nmse": best_val,
            "elapsed_seconds": time.time() - started,
            "checkpoint": epoch_name,
        }
        _append_jsonl(run_dir / "history.jsonl", record)
        _write_json(run_dir / "progress.json", {**record, "status": "running"})
        print(
            f"[axial] epoch {epoch_index + 1}/{epochs} "
            f"train_nmse={train_nmse:.6e} val_nmse={val_nmse:.6e} "
            f"best_val_nmse={best_val:.6e}@{best_epoch} "
            f"checkpoint={run_dir / epoch_name}"
        )
        if int(patience_epochs) > 0 and stale_epochs >= int(patience_epochs):
            print(f"[axial] early stop after {stale_epochs} epochs without improvement")
            break
    _write_asset(
        context,
        best_state,
        {
            "best_epoch": int(best_epoch),
            "best_val_nmse": float(best_val),
            "elapsed_seconds": float(time.time() - started),
            "run_dir": run_dir.as_posix(),
        },
        overwrite=overwrite,
    )
    _write_json(
        run_dir / "progress.json",
        {
            "status": "finished",
            "best_epoch": int(best_epoch),
            "best_val_nmse": float(best_val),
            "elapsed_seconds": float(time.time() - started),
            "asset_dir": Path(context["asset_dir"]).as_posix(),
        },
    )
