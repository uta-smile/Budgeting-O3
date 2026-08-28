from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import yaml

from .adapter import load_adapter
from .baseline import run_best_k_of_n
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
    parser.add_argument("--run-id", default=None, help="Optional run folder name. Defaults to the current timestamp.")
    parser.add_argument("--only", nargs="*", help="run only these budget names")
    parser.add_argument(
        "--method",
        choices=("o3", "best-k-of-n"),
        default="o3",
        help="Use O3 optimization or the random Best K-of-N baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print("O3 runner: PDB-output/TM-score fix enabled", flush=True)
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    project_root = config_path.parent.parent
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

    adapter = load_adapter(args.adapter, config)
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
    method_name = "best_k_of_n" if args.method == "best-k-of-n" else "o3"
    method_root = output_root / target_name / method_name
    summaries_by_budget: dict[str, list[dict]] = {}
    for budget in config["budgets"]:
        budget_name = str(budget.get("name", f"n{budget['N']}_k{budget['K']}"))
        if wanted and budget_name not in wanted:
            continue
        for replicate in range(replicates):
            run_seed = int(config.get("seed", 0)) + replicate
            run_dir = method_root / budget_name / "runs" / run_id / f"seed_{run_seed:04d}"
            print(f"[{budget_name} method={args.method} seed={run_seed}] starting", flush=True)
            runner = run_best_k_of_n if args.method == "best-k-of-n" else run_o3
            summary = runner(
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

        metadata = {
            "method": args.method,
            "method_directory": method_name,
            "target": target_name,
            "budget": budget_name,
            "run_id": run_id,
            "config_path": str(config_path),
            "replicates": replicates,
            "seed_start": int(config.get("seed", 0)),
            "seeds": [
                int(config.get("seed", 0)) + replicate
                for replicate in range(replicates)
            ],
        }
        with (summary_root / "run_metadata.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(metadata, handle, indent=2)
        print(f"Finished. Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
