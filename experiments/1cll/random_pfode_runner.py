"""Run the same-decoder random PF-ODE diagnostic for 1CLL."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

import common
from o3_boltz.adapter import load_adapter
from o3_boltz.random_baseline import run_random_pfode


def run(
    replicates: int,
    run_id: str,
    config_path: Path,
    budget_name: str,
    *,
    seed_start: int = common.DEFAULT_REPLICATE_SEED_START,
    seed_step: int = common.DEFAULT_REPLICATE_SEED_STEP,
    seeds: list[int] | None = None,
    resume: bool = False,
    write_run_reports: bool = True,
) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    config["project_root"] = str(common.REPO_ROOT)
    config["target"]["sequence"] = "".join(str(config["target"]["sequence"]).split()).upper()
    adapter = load_adapter("adapters.boltz2_pfode:create", config)
    if (
        type(adapter).__name__ != "Boltz2PFODEAdapter"
        or type(adapter).__module__ != "adapters.boltz2_pfode"
    ):
        raise RuntimeError("The random PF-ODE diagnostic requires the custom O3 adapter")
    config["latent_dim"] = int(adapter.latent_dim)
    budget = next(
        item for item in config["budgets"] if str(item.get("name")) == budget_name
    )
    run_seeds = common.resolve_replicate_seeds(
        replicates, seeds=seeds, seed_start=seed_start, seed_step=seed_step
    )
    metadata_seed_start = None if seeds is not None else seed_start
    metadata_seed_step = None if seeds is not None else seed_step
    print(f"[random-pfode] shared replicate seeds: {run_seeds}", flush=True)
    root = common.output_root("random_pfode", run_id)
    summaries = []
    for run_seed in run_seeds:
        summaries.append(
            run_random_pfode(
                adapter=adapter,
                config=config,
                budget=budget,
                run_seed=run_seed,
                output_dir=root / f"seed_{run_seed:04d}",
                resume=resume,
            )
        )
    if not write_run_reports:
        return {"method": "random_pfode", "replicates": summaries}
    return finalize_run(
        replicates=replicates,
        run_id=run_id,
        budget_name=budget_name,
        run_seeds=run_seeds,
        summaries=summaries,
        seed_start=metadata_seed_start,
        seed_step=metadata_seed_step,
    )


def finalize_run(
    *,
    replicates: int,
    run_id: str,
    budget_name: str,
    run_seeds: list[int],
    summaries: list[dict] | None = None,
    seed_start: int | None = None,
    seed_step: int | None = None,
) -> dict:
    """Write random-PF-ODE run reports after all seed workers complete."""

    root = common.output_root("random_pfode", run_id)
    if summaries is None:
        summaries = []
        for seed in run_seeds:
            path = root / f"seed_{seed:04d}" / "summary.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed random PF-ODE summary: {path}")
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    if len(summaries) != replicates:
        raise ValueError(
            f"Expected {replicates} random PF-ODE summaries, got {len(summaries)}"
        )
    fields = ["seed", "N", "K", "mean_of_K", "max_of_K"]
    with (root / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fields})
    with (root / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fields})
    (root / "run_metadata.json").write_text(
        json.dumps(
            {
                "method": "random_pfode",
                "budget": budget_name,
                "run_id": run_id,
                "replicates": replicates,
                "seed_mode": "explicit_list" if seed_start is None else "arithmetic_schedule",
                "seed_start": seed_start,
                "seed_step": seed_step,
                "seeds": run_seeds,
                "latent_sampler": "standard_normal_Z",
                "generator_sampling": "deterministic_pf_ode",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"method": "random_pfode", "replicates": summaries}
