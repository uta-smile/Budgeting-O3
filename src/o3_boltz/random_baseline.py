"""Same-decoder random baseline for diagnosing O3 Bayesian optimization."""

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
from .run_metadata import collect_run_metadata


@dataclass
class RandomEvaluation:
    index: int
    stage: str
    score: float
    structure: str
    latent_file: str
    budget: str
    seed: int


def run_random_pfode(
    *,
    adapter: GeneratorOracle,
    config: Mapping[str, Any],
    budget: Mapping[str, Any],
    run_seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Spend all N calls on independent standard-normal latents in Z."""

    n = int(budget["N"])
    k = int(budget["K"])
    if not (0 < k <= n):
        raise ValueError(f"Require 0 < K <= N, got N={n}, K={k}")
    latent_dim = int(config["latent_dim"])
    budget_name = str(budget.get("name", f"n{n}_k{k}"))
    rng = np.random.default_rng(run_seed)
    random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    structure_dir = output_dir / "random"
    latent_dir = output_dir / "latents"
    structure_dir.mkdir(exist_ok=True)
    latent_dir.mkdir(exist_ok=True)
    evaluations: list[RandomEvaluation] = []

    print(
        f"[{budget_name} random-pfode seed={run_seed}] diagnostic protocol: "
        f"{n} independent random Z samples -> select best {k}",
        flush=True,
    )

    for index in range(n):
        latent = rng.normal(size=latent_dim)
        latent_path = latent_dir / f"latent_{index:04d}.npy"
        structure_path = structure_dir / f"sample_{index:04d}.pdb"
        np.save(latent_path, latent)
        print(
            f"[{budget_name} random-pfode seed={run_seed}] "
            f"generating {index + 1}/{n} random Z samples",
            flush=True,
        )
        written_path = adapter.generate(
            latent=latent,
            output_path=structure_path,
            config=config,
            metadata={
                "budget": budget_name,
                "seed": run_seed,
                "stage": "random_pfode",
                "evaluation_index": index,
                "deterministic": True,
            },
        )
        final_path = Path(written_path) if written_path is not None else structure_path
        if not final_path.exists():
            raise FileNotFoundError(f"The adapter did not write a structure at {final_path}")
        score = float(adapter.score(final_path, config))
        if not np.isfinite(score):
            raise ValueError(f"Oracle returned a non-finite score for {final_path}")
        evaluations.append(
            RandomEvaluation(
                index=index,
                stage="random_pfode",
                score=score,
                structure=str(final_path),
                latent_file=str(latent_path),
                budget=budget_name,
                seed=run_seed,
            )
        )
        print(
            f"[{budget_name} random-pfode seed={run_seed}] "
            f"completed {index + 1}/{n} | score={score:.4f}",
            flush=True,
        )

    ranked = sorted(evaluations, key=lambda item: item.score, reverse=True)
    returned = ranked[:k]
    all_scores = np.asarray([item.score for item in evaluations], dtype=np.float64)
    selected_scores = np.asarray([item.score for item in returned], dtype=np.float64)
    records = [asdict(item) for item in evaluations]
    with (output_dir / "evaluations.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    with (output_dir / "evaluations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / "returned_candidates.json").open("w", encoding="utf-8") as handle:
        json.dump([asdict(item) for item in returned], handle, indent=2)

    summary = {
        "method": "random_pfode",
        "budget": budget_name,
        "N": n,
        "K": k,
        "seed": run_seed,
        "oracle_evaluations": len(evaluations),
        "generator_sampling": "deterministic_pf_ode",
        "latent_sampler": "standard_normal_Z",
        "selection_metric": "oracle_tm_score",
        "total_mean": float(np.mean(all_scores)),
        "mean_all": float(np.mean(all_scores)),
        "max_of_K": float(np.max(selected_scores)),
        "top_k_mean": float(np.mean(selected_scores)),
        "mean_of_K": float(np.mean(selected_scores)),
        "best_structure": returned[0].structure,
        "output_dir": str(output_dir),
    }
    summary.update(collect_run_metadata(config=config, adapter=adapter))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / "provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "backend": "custom_boltz2_o3",
                "method": "random_pfode",
                "generator": summary.get("generator"),
                "msa_cache": summary.get("msa_cache"),
                "seed": run_seed,
                "N": n,
                "K": k,
                "latent_sampler": "standard_normal_Z",
                "selection_metric": "oracle_tm_score",
            },
            handle,
            indent=2,
        )
    print(
        f"[{budget_name} random-pfode seed={run_seed}] diagnostic complete: "
        f"{n} random Z samples -> best {k}; mean-of-K={summary['mean_of_K']:.4f}",
        flush=True,
    )
    return summary
