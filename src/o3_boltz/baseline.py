from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .adapter import GeneratorOracle


@dataclass
class BaselineEvaluation:
    index: int
    stage: str
    score: float
    structure: str
    latent_file: str
    budget: str
    seed: int


def run_best_k_of_n(
    *,
    adapter: GeneratorOracle,
    config: Mapping[str, Any],
    budget: Mapping[str, Any],
    run_seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Run random Best K-of-N sampling with the same generator and oracle as O3."""

    n = int(budget["N"])
    k = int(budget["K"])
    if not (0 < k <= n):
        raise ValueError(f"Require 0 < K <= N, got N={n}, K={k}")
    latent_dim = int(config["latent_dim"])
    budget_name = str(budget.get("name", f"n{n}_k{k}"))
    baseline_settings = config.get("best_k_of_n", {})
    deterministic = bool(baseline_settings.get("deterministic", False))
    rng = np.random.default_rng(run_seed)
    random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    structure_dir = output_dir / "random"
    structure_dir.mkdir(exist_ok=True)
    latent_dir = output_dir / "latents"
    latent_dir.mkdir(exist_ok=True)
    evaluations: list[BaselineEvaluation] = []

    for index in range(n):
        latent = rng.normal(size=latent_dim)
        latent_path = latent_dir / f"latent_{index:04d}.npy"
        structure_path = structure_dir / f"sample_{index:04d}.pdb"
        np.save(latent_path, latent)
        print(
            f"[{budget_name} best-k-of-n seed={run_seed}] "
            f"generating {index + 1}/{n}",
            flush=True,
        )
        written_path = adapter.generate(
            latent=latent,
            output_path=structure_path,
            config=config,
            metadata={
                "budget": budget_name,
                "seed": run_seed,
                "stage": "random_baseline",
                "evaluation_index": index,
                "deterministic": deterministic,
            },
        )
        final_path = Path(written_path) if written_path is not None else structure_path
        if not final_path.exists():
            raise FileNotFoundError(f"The adapter did not write a structure at {final_path}")
        score = float(adapter.score(final_path, config))
        if not np.isfinite(score):
            raise ValueError(f"Oracle returned a non-finite score for {final_path}")
        evaluations.append(
            BaselineEvaluation(
                index=index,
                stage="random_baseline",
                score=score,
                structure=str(final_path),
                latent_file=str(latent_path),
                budget=budget_name,
                seed=run_seed,
            )
        )
        print(
            f"[{budget_name} best-k-of-n seed={run_seed}] "
            f"completed {index + 1}/{n} | score={score:.4f}",
            flush=True,
        )

    ranked = sorted(evaluations, key=lambda item: item.score, reverse=True)
    returned = ranked[:k]
    records = [asdict(item) for item in evaluations]
    with (output_dir / "evaluations.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    with (output_dir / "evaluations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / "returned_candidates.json").open("w", encoding="utf-8") as handle:
        json.dump([asdict(item) for item in returned], handle, indent=2)

    return {
        "method": "best_k_of_n",
        "budget": budget_name,
        "N": n,
        "K": k,
        "latent_dim": latent_dim,
        "generator_atom_count": getattr(adapter, "atom_count", None),
        "generator_atom_slots": getattr(adapter, "atom_slots", None),
        "seed": run_seed,
        "oracle_evaluations": len(evaluations),
        "generator_sampling": ("deterministic_pf_ode" if deterministic else "stochastic_boltz2"),
        "score_std": float(np.std([item.score for item in evaluations])),
        "max_of_K": max(item.score for item in returned),
        "mean_of_K": float(np.mean([item.score for item in returned])),
        "best_structure": returned[0].structure,
        "output_dir": str(output_dir),
    }
