#!/usr/bin/env python3
"""Plot BER and BLER for every scenario inventory under a result directory.

The layout is the shared MIMO-OFDM chest contract: a directory holds
``results.json`` plus the ``.npz`` files it lists. Each payload stores
``ebno_dbs``, ``ber``, ``bler``, and optional confidence-interval arrays.

Running this file with no arguments walks the directory that contains it,
including nested scenario folders, and writes ``ber_bler.png`` beside each
``results.json``. Pass another directory to plot a different result tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _positive(values: np.ndarray) -> np.ndarray:
    """Keep strictly positive rates so a log axis can draw them."""
    array = np.asarray(values, dtype=np.float64)
    return np.where(array > 0.0, array, np.nan)


def _load_curves(scenario_dir: Path, inventory: dict) -> list[dict]:
    curves = []
    for name, entry in inventory.get("results", {}).items():
        filename = entry.get("file")
        if not filename:
            print(f"skip {name}: results.json entry has no file", file=sys.stderr)
            continue
        payload_path = scenario_dir / filename
        if not payload_path.is_file():
            print(f"skip {name}: missing {payload_path}", file=sys.stderr)
            continue
        with np.load(payload_path) as payload:
            curve = {
                "name": name,
                "ebno_dbs": np.asarray(payload["ebno_dbs"], dtype=np.float64),
                "ber": _positive(payload["ber"]),
                "bler": _positive(payload["bler"]),
            }
            for metric in ("ber", "bler"):
                lower_key = f"{metric}_lower_confidence_interval"
                upper_key = f"{metric}_upper_confidence_interval"
                if lower_key in payload.files and upper_key in payload.files:
                    curve[lower_key] = _positive(payload[lower_key])
                    curve[upper_key] = _positive(payload[upper_key])
        curves.append(curve)
    return curves


def _plot_metric(axis, curves: list[dict], metric: str, ylabel: str) -> None:
    for curve in curves:
        ebno = curve["ebno_dbs"]
        rate = curve[metric]
        (line,) = axis.semilogy(ebno, rate, marker="o", linewidth=1.5, label=curve["name"])
        lower_key = f"{metric}_lower_confidence_interval"
        upper_key = f"{metric}_upper_confidence_interval"
        if lower_key in curve and upper_key in curve:
            axis.fill_between(
                ebno,
                curve[lower_key],
                curve[upper_key],
                color=line.get_color(),
                alpha=0.18,
                linewidth=0,
            )
    axis.set_xlabel("Eb/N0 (dB)")
    axis.set_ylabel(ylabel)
    axis.grid(True, which="both", linestyle=":", linewidth=0.6)
    axis.legend(loc="best", fontsize=8)


def plot_scenario(scenario_dir: Path) -> Path | None:
    inventory_path = scenario_dir / "results.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    curves = _load_curves(scenario_dir, inventory)
    if not curves:
        print(f"no curves in {inventory_path}", file=sys.stderr)
        return None

    scenario_id = inventory.get("scenario_id", scenario_dir.name)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    _plot_metric(axes[0], curves, "ber", "BER")
    _plot_metric(axes[1], curves, "bler", "BLER")
    figure.suptitle(scenario_id)
    figure.tight_layout()

    output_path = scenario_dir / "ber_bler.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to search for results.json files. Defaults to this script's directory.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    inventories = sorted(root.rglob("results.json"))
    if not inventories:
        raise SystemExit(f"no results.json under {root}")

    written = []
    for inventory_path in inventories:
        output_path = plot_scenario(inventory_path.parent)
        if output_path is not None:
            written.append(output_path)
            print(output_path)
    if not written:
        raise SystemExit(f"no plottable results under {root}")


if __name__ == "__main__":
    main()
