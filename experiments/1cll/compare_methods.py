"""Build one seed-aligned comparison report for the selected methods."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import common


METHOD_DIRECTORIES = {
    "best_k_of_n": "best_k_of_n",
    "o3": "o3",
    "random_pfode": "random_pfode",
    "public_best_k_of_n": "public_best_k_of_n",
}
METHOD_LABELS = {
    "best_k_of_n": "best_k_of_n",
    "o3": "o3",
    "random_pfode": "random_pfode",
    "public_best_k_of_n": "public_best_k_of_n",
}


def _read_manifest(run_id: str) -> dict[str, Any]:
    path = common.comparison_run_root(run_id) / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    seeds = manifest.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"Run manifest has no replicate seed list: {path}")
    return manifest


def _aggregate_path(method: str, run_id: str) -> Path:
    return common.output_root(METHOD_DIRECTORIES[method], run_id) / "aggregate.csv"


def _load_aggregate(method: str, run_id: str, seeds: list[int]) -> dict[int, dict[str, Any]]:
    path = _aggregate_path(method, run_id)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {method} aggregate report: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Aggregate report is empty: {path}")

    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed_key = "run_seed" if "run_seed" in row else "seed"
        try:
            seed = int(row[seed_key])
            row["N"] = int(row["N"])
            row["K"] = int(row["K"])
            row["mean_of_K"] = float(row["mean_of_K"])
            row["max_of_K"] = float(row["max_of_K"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid row in {path}: {row}") from exc
        if seed in result:
            raise ValueError(f"Duplicate seed {seed} in {path}")
        result[seed] = row

    expected = set(seeds)
    actual = set(result)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"Seed mismatch in {path}; missing={missing}, unexpected={extra}"
        )
    return result


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_stats(rows: list[dict[str, Any]], method: str, metric: str) -> dict[str, float]:
    values = [float(row[f"{method}_{metric}"]) for row in rows]
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "best": max(values),
    }


def write_report(
    *,
    budget: str,
    run_id: str,
    methods: list[str],
) -> dict[str, Any]:
    """Write seed-aligned CSV/JSON reports and return their metadata."""

    common.configure_budget(budget)
    manifest = _read_manifest(run_id)
    seeds = [int(seed) for seed in manifest["seeds"]]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Run manifest contains duplicate seeds: {seeds}")
    if not methods:
        raise ValueError("At least one method is required")
    methods = list(dict.fromkeys(methods))
    unknown = sorted(set(methods) - set(METHOD_DIRECTORIES))
    if unknown:
        raise ValueError(f"Unknown comparison method(s): {unknown}")

    aggregates = {
        method: _load_aggregate(method, run_id, seeds) for method in methods
    }
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        first = aggregates[methods[0]][seed]
        row: dict[str, Any] = {"seed": seed, "N": first["N"], "K": first["K"]}
        for method in methods:
            current = aggregates[method][seed]
            if (current["N"], current["K"]) != (row["N"], row["K"]):
                raise ValueError(f"N/K mismatch for seed {seed} in method {method}")
            row[f"{method}_mean_of_K"] = current["mean_of_K"]
            row[f"{method}_max_of_K"] = current["max_of_K"]
        for metric in ("mean_of_K", "max_of_K"):
            metric_values = {method: row[f"{method}_{metric}"] for method in methods}
            row[f"best_{metric}_method"] = max(
                methods, key=lambda method: metric_values[method]
            )
        rows.append(row)

    summary_root = common.comparison_run_root(run_id)
    summary_root.mkdir(parents=True, exist_ok=True)
    comparison_csv = summary_root / "comparison_summary.csv"
    fieldnames = ["seed", "N", "K"]
    fieldnames += [f"{method}_{metric}" for method in methods for metric in ("mean_of_K", "max_of_K")]
    fieldnames += ["best_mean_of_K_method", "best_max_of_K_method"]
    _write_csv(comparison_csv, fieldnames, rows)

    stats_rows = []
    stats: dict[str, Any] = {}
    for method in methods:
        mean_stats = _metric_stats(rows, method, "mean_of_K")
        max_stats = _metric_stats(rows, method, "max_of_K")
        stats[method] = {
            "mean_of_K": mean_stats,
            "max_of_K": max_stats,
        }
        stats_rows.append(
            {
                "method": METHOD_LABELS[method],
                "replicates": len(rows),
                "mean_of_K_mean": mean_stats["mean"],
                "mean_of_K_std": mean_stats["std"],
                "mean_of_K_best": mean_stats["best"],
                "max_of_K_mean": max_stats["mean"],
                "max_of_K_std": max_stats["std"],
                "max_of_K_best": max_stats["best"],
            }
        )
    stats_csv = summary_root / "comparison_stats.csv"
    _write_csv(stats_csv, list(stats_rows[0]), stats_rows)

    json_path = summary_root / "comparison_summary.json"
    report = {
        "target": manifest.get("target", "1cll"),
        "budget": budget,
        "run_id": run_id,
        "replicates": len(rows),
        "seeds": seeds,
        "methods": methods,
        "comparison_summary_csv": str(comparison_csv),
        "comparison_stats_csv": str(stats_csv),
        "comparison_summary_json": str(json_path),
        "stats": stats,
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", required=True, choices=("n20_k2", "n50_k5", "n100_k10"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHOD_DIRECTORIES),
        default=["best_k_of_n", "o3", "random_pfode"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = write_report(budget=args.budget, run_id=args.run_id, methods=args.methods)
    print(f"Combined comparison report: {report['comparison_summary_csv']}", flush=True)
    print(f"Method statistics: {report['comparison_stats_csv']}", flush=True)


if __name__ == "__main__":
    main()
