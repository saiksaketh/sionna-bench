#!/usr/bin/env python3
"""Compare BER, BLER, and NMSE across stored MIMO-OFDM channel estimators.

The script reads the shared result layout under this directory. Each paper
has ``<paper>/<scenario>/results.json`` plus the NPZ files named there.
Smoke and reduced runs are skipped.

With no estimator arguments it plots the axial estimator against the built-in
linear least-squares estimator and the built-in LMMSE estimator. Pass
``paper/result`` names, or use ``--list``, to choose any other stored run.
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


METRICS = ("ber", "bler", "nmse")
METRIC_LABELS = {"ber": "BER", "bler": "BLER", "nmse": "NMSE"}
SKIPPED_PARTS = {"smoke", "reduced", "comparisons"}
DEFAULT_SELECTION = (
    "axial/AxialTransformerChannelEstimator",
    "builtin/LS (linear)",
    "builtin/LMMSE",
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "h")


def _positive(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(array) & (array > 0.0), array, np.nan)


def discover(root: Path, scenario: str) -> list[dict]:
    runs = []
    for inventory_path in sorted(root.rglob("results.json")):
        scenario_dir = inventory_path.parent
        relative = scenario_dir.relative_to(root)
        if set(relative.parts) & SKIPPED_PARTS:
            continue
        if relative.name != scenario:
            continue
        paper = relative.parts[0]
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        for name, entry in inventory.get("results", {}).items():
            filename = entry.get("file") if isinstance(entry, dict) else None
            if not filename:
                continue
            payload_path = scenario_dir / filename
            if not payload_path.is_file():
                print(f"skip {paper}/{name}: missing {payload_path}", file=sys.stderr)
                continue
            runs.append(
                {
                    "key": f"{paper}/{name}",
                    "paper": paper,
                    "name": name,
                    "label": f"{paper} · {name}",
                    "path": payload_path,
                    "scenario": inventory.get("scenario_id", scenario),
                    "fingerprint": inventory.get("config_fingerprint"),
                }
            )
    return runs


def _resolve_one(token: str, runs: list[dict]) -> dict:
    exact_key = [run for run in runs if run["key"] == token]
    if len(exact_key) == 1:
        return exact_key[0]
    exact_name = [run for run in runs if run["name"] == token]
    if len(exact_name) == 1:
        return exact_name[0]
    folded = token.casefold()
    folded_hits = [
        run
        for run in runs
        if run["key"].casefold() == folded or run["name"].casefold() == folded
    ]
    if len(folded_hits) == 1:
        return folded_hits[0]
    substring_hits = [
        run
        for run in runs
        if folded in run["key"].casefold() or folded in run["name"].casefold()
    ]
    if len(substring_hits) == 1:
        return substring_hits[0]
    choices = substring_hits or exact_name or exact_key
    if not choices:
        known = "\n".join(f"  {run['key']}" for run in runs)
        raise SystemExit(f"unknown estimator {token!r}. Known runs:\n{known}")
    lines = "\n".join(f"  {run['key']}" for run in choices)
    raise SystemExit(f"{token!r} matches more than one run:\n{lines}")


def load_curve(run: dict) -> dict:
    with np.load(run["path"]) as payload:
        curve = {
            "label": run["label"],
            "ebno_dbs": np.asarray(payload["ebno_dbs"], dtype=np.float64),
            "fingerprint": run["fingerprint"],
        }
        for metric in METRICS:
            curve[metric] = _positive(payload[metric])
            lower_key = f"{metric}_lower_confidence_interval"
            upper_key = f"{metric}_upper_confidence_interval"
            if lower_key in payload.files and upper_key in payload.files:
                curve[lower_key] = _positive(payload[lower_key])
                curve[upper_key] = _positive(payload[upper_key])
    return curve


def plot_curves(curves: list[dict], metrics: list[str], scenario: str, output_path: Path) -> None:
    figure, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(5.1 * len(metrics), 4.6),
        sharex=True,
        squeeze=False,
    )
    for axis, metric in zip(axes[0], metrics):
        for index, curve in enumerate(curves):
            (line,) = axis.semilogy(
                curve["ebno_dbs"],
                curve[metric],
                marker=MARKERS[index % len(MARKERS)],
                linewidth=1.5,
                label=curve["label"],
            )
            lower_key = f"{metric}_lower_confidence_interval"
            upper_key = f"{metric}_upper_confidence_interval"
            if lower_key in curve and upper_key in curve:
                axis.fill_between(
                    curve["ebno_dbs"],
                    curve[lower_key],
                    curve[upper_key],
                    color=line.get_color(),
                    alpha=0.15,
                    linewidth=0,
                )
        axis.set_xlabel("Eb/N0 (dB)")
        axis.set_ylabel(METRIC_LABELS[metric])
        axis.grid(True, which="both", linestyle=":", linewidth=0.6)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(3, len(curves)),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.suptitle(scenario)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "estimators",
        nargs="*",
        help=(
            "Runs to plot, as paper/result or a unique result name. "
            "Defaults to axial, built-in linear LS, and built-in LMMSE."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Benchmark output root. Defaults to this script's directory.",
    )
    parser.add_argument("--scenario", default="3gpp-umi")
    parser.add_argument("--list", action="store_true", help="Print selectable runs and exit.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRICS,
        default=list(METRICS),
        help="Metrics to draw. Defaults to ber, bler, and nmse.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Image path. Defaults to comparisons/<scenario>/ber_bler_nmse.png under --root.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    runs = discover(root, args.scenario)
    if not runs:
        raise SystemExit(f"no results for scenario {args.scenario!r} under {root}")
    if args.list:
        current_paper = None
        for run in runs:
            if run["paper"] != current_paper:
                current_paper = run["paper"]
                print(f"[{current_paper}]")
            print(f"  {run['key']}")
        return

    selected_tokens = args.estimators or list(DEFAULT_SELECTION)
    selected = [_resolve_one(token, runs) for token in selected_tokens]
    fingerprints = {run["fingerprint"] for run in selected}
    if len(fingerprints) > 1:
        print(
            "warning: selected runs do not share one config fingerprint",
            file=sys.stderr,
        )
    curves = [load_curve(run) for run in selected]
    output_path = args.output
    if output_path is None:
        output_path = root / "comparisons" / args.scenario / "ber_bler_nmse.png"
    plot_curves(curves, list(args.metrics), args.scenario, output_path)
    print(output_path)
    for run in selected:
        print(f"  {run['key']}")


if __name__ == "__main__":
    main()
