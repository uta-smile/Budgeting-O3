from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import yaml

from .adapter import load_adapter
from .run_metadata import collect_run_metadata
from .o3 import run_o3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the O3 Boltz-2 experiment sweep")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--adapter",
        default="adapters.boltz2_pfode:create",
        help="module:function adapter factory",
    )
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="Override the first run seed. Replicates continue from this value.",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=None,
        help="Difference between successive replicate seeds (default: 1).",
    )
    parser.add_argument(
        "--seed-list",
        nargs="+",
        type=int,
        default=None,
        help="Explicit replicate seed list; its length must equal --replicates.",
    )
    parser.add_argument("--run-id", default=None, help="Optional run folder name. Defaults to the current timestamp.")
    parser.add_argument("--only", nargs="*", help="run only these budget names")
    parser.add_argument(
        "--method",
        choices=("o3",),
        default="o3",
        help="The root runner is reserved for the custom O3 Boltz backend. "
        "Use experiments/1cll/k10_n100/run.py for Best K-of-N.",
    )
    return parser.parse_args()


def _find_project_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "o3_boltz"
        ).is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find the repository root above configuration {config_path}"
    )


def main() -> None:
    args = _parse_args()
    print("O3 runner: PDB-output/TM-score fix enabled", flush=True)
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    project_root = _find_project_root(config_path)
    os.chdir(project_root)
    config["project_root"] = str(project_root)
    sequence = "".join(str(config["target"]["sequence"]).split()).upper()
    config["target"]["sequence"] = sequence
    target_name = str(config["target"]["name"]).lower()
    config["target"]["name"] = target_name
    if target_name == "1cll":
        if len(sequence) != 144:
            raise ValueError(f"1CLL target sequence must contain 144 residues, got {len(sequence)}")
    output_root = Path(config.get("output_dir", "outputs"))
    if not output_root.is_absolute():
        output_root = project_root / output_root
    replicates = int(config.get("replicates", 1) if args.replicates is None else args.replicates)
    if replicates < 1:
        raise ValueError("replicates must be at least 1")
    if args.seed_list is not None:
        if len(args.seed_list) != replicates:
            raise ValueError(
                f"Expected {replicates} values in --seed-list, got {len(args.seed_list)}"
            )
        if len(set(args.seed_list)) != len(args.seed_list):
            raise ValueError("--seed-list values must be unique")
        run_seeds = [int(seed) for seed in args.seed_list]
        seed_start = None
        seed_step = None
        seed_mode = "explicit_list"
    else:
        seed_start = int(config.get("seed", 0) if args.seed_start is None else args.seed_start)
        seed_step = int(config.get("seed_step", 1) if args.seed_step is None else args.seed_step)
        if seed_step == 0:
            raise ValueError("seed_step must not be 0")
        run_seeds = [seed_start + replicate * seed_step for replicate in range(replicates)]
        seed_mode = "arithmetic_schedule"
    print(f"[o3] seed mode: {seed_mode}", flush=True)
    print(f"[o3] shared replicate seeds: {run_seeds}", flush=True)

    adapter = load_adapter(args.adapter, config)
    if (
        type(adapter).__name__ != "Boltz2PFODEAdapter"
        or type(adapter).__module__ != "adapters.boltz2_pfode"
    ):
        raise RuntimeError(
            "The root O3 runner requires adapters.boltz2_pfode:Boltz2PFODEAdapter; "
            "Best K-of-N must use experiments/1cll/k10_n100/run.py."
        )
    provenance = collect_run_metadata(config=config, adapter=adapter)
    adapter_latent_dim = getattr(adapter, "latent_dim", None)
    configured_latent_dim = config.get("latent_dim")
    if adapter_latent_dim is not None:
        adapter_latent_dim = int(adapter_latent_dim)
        if configured_latent_dim not in (None, "auto"):
            configured_latent_dim = int(configured_latent_dim)
            if configured_latent_dim != adapter_latent_dim:
                print(
                    "Warning: overriding latent_dim="
                    f"{configured_latent_dim} with the generator-derived value "
                    f"{adapter_latent_dim}.",
                    flush=True,
                )
        config["latent_dim"] = adapter_latent_dim
        print(f"Generator latent_dim={adapter_latent_dim}", flush=True)
    elif configured_latent_dim in (None, "auto"):
        raise ValueError(
            "The adapter must expose latent_dim when the configuration uses latent_dim: auto"
        )
    else:
        config["latent_dim"] = int(configured_latent_dim)

    wanted = set(args.only or [])
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    method_name = "o3"
    output_layout = str(config.get("output_layout", "target_method"))
    if output_layout == "method_only":
        method_root = output_root / method_name
    elif output_layout == "target_method":
        method_root = output_root / target_name / method_name
    else:
        raise ValueError("output_layout must be 'target_method' or 'method_only'")
    summaries_by_budget: dict[str, list[dict]] = {}
    for budget in config["budgets"]:
        budget_name = str(budget.get("name", f"n{budget['N']}_k{budget['K']}"))
        if wanted and budget_name not in wanted:
            continue
        for run_seed in run_seeds:
            run_dir = method_root / budget_name / "runs" / run_id / f"seed_{run_seed:04d}"
            print(f"[{budget_name} method={args.method} seed={run_seed}] starting", flush=True)
            summary = run_o3(
                adapter=adapter,
                config=config,
                budget=budget,
                run_seed=run_seed,
                output_dir=run_dir,
            )
            summaries_by_budget.setdefault(budget_name, []).append(summary)
            print(
                f"[{budget_name} method={args.method} seed={run_seed}] "
                f"max-of-K={summary['max_of_K']:.4f} "
                f"mean-of-K={summary['mean_of_K']:.4f}",
                flush=True,
            )

    if not summaries_by_budget:
        print("Finished. No matching budget names were found.", flush=True)
        return

    for budget_name, summaries in summaries_by_budget.items():
        summary_root = method_root / budget_name / "runs" / run_id
        summary_root.mkdir(parents=True, exist_ok=True)
        summary_path = summary_root / "sweep_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
            writer.writeheader()
            writer.writerows(summaries)
        aggregate_path = summary_root / "aggregate.csv"
        with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
            fields = ["seed", "N", "K", "mean_of_K", "max_of_K"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for summary in summaries:
                writer.writerow({field: summary.get(field) for field in fields})

        metadata = {
            "method": args.method,
            "method_directory": method_name,
            "target": target_name,
            "budget": budget_name,
            "run_id": run_id,
            "config_path": str(config_path),
            "replicates": replicates,
            "seed_start": seed_start,
            "seed_step": seed_step,
            "seed_mode": seed_mode,
            "seeds": run_seeds,
            "provenance": provenance,
        }
        with (summary_root / "run_metadata.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(metadata, handle, indent=2)
        print(f"Finished. Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
