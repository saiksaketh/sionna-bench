# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train and benchmark the axial-attention channel estimator.

Train a mixed-SNR checkpoint, then evaluate it on the shared MIMO-OFDM task::

    python -m sionna.bench.mimo_ofdm_chest.papers.axial.run_benchmark \
        --prepare-assets --asset-root generated-assets --smoke

    python -m sionna.bench.mimo_ofdm_chest.papers.axial.run_benchmark \
        --asset-root generated-assets --smoke
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from sionna.bench.mimo_ofdm_chest.benchmarking import (
    build_resource_grid,
    evaluate_channel_estimator,
)
from sionna.bench.mimo_ofdm_chest.benchmarking.result_store import (
    prepare_paper_result_dir,
    record_result,
    resolve_constructor_arguments,
)
from sionna.bench.mimo_ofdm_chest.configs import SystemConfig, smoke_config, umi_3gpp_config

from .asset import load_model, train_model, training_settings
from .estimator import (
    DEFAULT_D_MODEL,
    DEFAULT_ERR_VAR_SCALE,
    DEFAULT_FFN_EXPANSION,
    DEFAULT_NUM_BLOCKS,
    DEFAULT_NUM_HEADS,
    AxialTransformerChannelEstimator,
)


PAPER_SLUG = "axial"
_RECORDED_ARGUMENTS = (
    "d_model",
    "num_heads",
    "num_blocks",
    "ffn_expansion",
    "err_var_scale",
    "anchor_pilots",
)


def prepare_assets(
    cfg: SystemConfig,
    *,
    asset_root: Union[str, Path],
    settings: dict[str, Any],
    architecture: dict[str, int],
    training_snr_dbs: Optional[list[float]],
    model_seed: int,
    overwrite_assets: bool,
    resume: bool,
) -> list[Path]:
    """Train the axial checkpoint bank and return its manifest paths."""
    resource_grid = build_resource_grid(cfg)
    _, _, manifest_paths = train_model(
        resource_grid,
        cfg,
        training_snr_dbs=training_snr_dbs,
        seed=model_seed,
        asset_root=asset_root,
        overwrite=overwrite_assets,
        resume=resume,
        **settings,
        **architecture,
    )
    for manifest_path in manifest_paths:
        print(f"[{PAPER_SLUG}] asset ready at {manifest_path}")
    return manifest_paths


def run_benchmark(
    cfg: SystemConfig,
    *,
    result_name: str,
    settings: dict[str, Any],
    architecture: dict[str, int],
    training_snr_dbs: Optional[list[float]],
    model_seed: int,
    err_var_scale: float,
    smoke: bool,
    asset_root: Optional[Union[str, Path]],
) -> dict[str, np.ndarray]:
    """Load axial weights, evaluate the estimator, and record the result."""
    constructor_arguments = resolve_constructor_arguments(
        AxialTransformerChannelEstimator,
        _RECORDED_ARGUMENTS,
        {
            **architecture,
            "err_var_scale": err_var_scale,
            "anchor_pilots": bool(settings["anchor_pilots"]),
        },
    )
    resource_grid = build_resource_grid(cfg)
    model_states, model_noise_variances, manifest_paths = load_model(
        resource_grid,
        cfg,
        training_snr_dbs=training_snr_dbs,
        seed=model_seed,
        asset_root=asset_root,
        **settings,
        **architecture,
    )
    output_dir = Path(prepare_paper_result_dir(cfg, PAPER_SLUG, smoke=smoke))
    output_path = output_dir / f"{result_name}.npz"
    estimator_kwargs = {
        "model_states": model_states,
        "err_var_scale": err_var_scale,
        "anchor_pilots": bool(settings["anchor_pilots"]),
        **architecture,
    }
    if model_noise_variances is not None:
        estimator_kwargs["model_noise_variances"] = model_noise_variances
    result = evaluate_channel_estimator(
        AxialTransformerChannelEstimator,
        cfg,
        output_path,
        estimator_kwargs=estimator_kwargs,
    )
    record_result(
        cfg,
        PAPER_SLUG,
        AxialTransformerChannelEstimator,
        result_name=result_name,
        constructor_arguments=constructor_arguments,
        asset_manifests=manifest_paths,
        smoke=smoke,
    )
    print(f"[{PAPER_SLUG}] result saved to {output_path}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    benchmark = parser.add_argument_group("benchmark configuration")
    benchmark.add_argument("--smoke", action="store_true")
    benchmark.add_argument("--output-dir", default=None)
    benchmark.add_argument("--device", default=None)
    benchmark.add_argument("--precision", choices=("single", "double"), default=None)
    benchmark.add_argument("--batch-size", type=int, default=None)
    benchmark.add_argument("--num-mc-batches", type=int, default=None)

    assets = parser.add_argument_group("asset handling")
    assets.add_argument("--asset-root", type=Path, default=None)
    assets.add_argument("--prepare-assets", action="store_true")
    assets.add_argument("--overwrite-assets", action="store_true")
    assets.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the latest epoch checkpoint in the run directory.",
    )
    assets.add_argument("--train-batches", type=int, default=None)
    assets.add_argument("--train-batch-size", type=int, default=None)
    assets.add_argument("--validation-batches", type=int, default=None)
    assets.add_argument("--validation-batch-size", type=int, default=None)
    assets.add_argument("--epochs", type=int, default=None)
    assets.add_argument("--optimizer-batch-size", type=int, default=None)
    assets.add_argument("--learning-rate", type=float, default=None)
    assets.add_argument("--patience-epochs", type=int, default=None)
    assets.add_argument("--relative-min-delta", type=float, default=None)
    assets.add_argument("--model-seed", type=int, default=None)
    assets.add_argument(
        "--equal-snr-loss",
        action="store_true",
        help="Give every SNR the same normalized NMSE weight and clip gradients.",
    )
    assets.add_argument(
        "--anchor-pilots",
        action="store_true",
        help="Keep least-squares values on the pilot resource elements.",
    )
    assets.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Finetune from a best.pt or latest.pt checkpoint instead of random weights.",
    )
    assets.add_argument("--snr-specific-weights", action="store_true")
    assets.add_argument("--training-snr-db", type=float, action="append", default=None)

    estimator = parser.add_argument_group("estimator configuration")
    estimator.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    estimator.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    estimator.add_argument("--num-blocks", type=int, default=DEFAULT_NUM_BLOCKS)
    estimator.add_argument("--ffn-expansion", type=int, default=DEFAULT_FFN_EXPANSION)
    estimator.add_argument("--err-var-scale", type=float, default=DEFAULT_ERR_VAR_SCALE)
    estimator.add_argument("--estimator-suffix", default="")
    return parser.parse_args()


def _system_config_from_args(args: argparse.Namespace) -> SystemConfig:
    cfg = smoke_config if args.smoke else umi_3gpp_config
    overrides = {
        name: value
        for name, value in {
            "output_dir": args.output_dir,
            "device": args.device,
            "precision": args.precision,
            "batch_size": args.batch_size,
            "num_mc_batches": args.num_mc_batches,
        }.items()
        if value is not None
    }
    return replace(cfg, **overrides)


def _settings_from_args(args: argparse.Namespace) -> dict[str, Any]:
    defaults = training_settings(args.smoke)
    settings = {
        name: defaults[name] if getattr(args, name) is None else getattr(args, name)
        for name in (
            "train_batches",
            "train_batch_size",
            "validation_batches",
            "validation_batch_size",
            "epochs",
            "optimizer_batch_size",
            "learning_rate",
            "patience_epochs",
            "relative_min_delta",
        )
    }
    settings["snr_specific_weights"] = bool(args.snr_specific_weights)
    settings["equal_snr_loss"] = bool(args.equal_snr_loss)
    settings["anchor_pilots"] = bool(args.anchor_pilots)
    settings["init_from"] = args.init_from
    return settings


def main() -> None:
    """Prepare axial weights or evaluate them on the channel-estimation task."""
    args = _parse_args()
    cfg = _system_config_from_args(args)
    if args.prepare_assets and args.asset_root is None:
        raise ValueError("`--prepare-assets` requires `--asset-root`.")
    architecture = {
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_blocks": args.num_blocks,
        "ffn_expansion": args.ffn_expansion,
    }
    settings = _settings_from_args(args)
    model_seed = cfg.seed if args.model_seed is None else args.model_seed
    asset_root = args.asset_root
    if args.smoke and asset_root is None:
        asset_root = Path("assets/smoke")
    if args.prepare_assets:
        prepare_assets(
            cfg,
            asset_root=asset_root,
            settings=settings,
            architecture=architecture,
            training_snr_dbs=args.training_snr_db,
            model_seed=model_seed,
            overwrite_assets=args.overwrite_assets,
            resume=args.resume,
        )
        return
    run_benchmark(
        cfg,
        result_name=f"{AxialTransformerChannelEstimator.__name__}{args.estimator_suffix}",
        settings=settings,
        architecture=architecture,
        training_snr_dbs=args.training_snr_db,
        model_seed=model_seed,
        err_var_scale=args.err_var_scale,
        smoke=args.smoke,
        asset_root=asset_root,
    )


if __name__ == "__main__":
    main()
