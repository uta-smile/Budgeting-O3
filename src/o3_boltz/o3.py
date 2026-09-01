from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood

from .adapter import GeneratorOracle
from .chart import SURROGATE_CHART_VERSION, SurrogateChart
from .run_metadata import collect_run_metadata


@dataclass
class Evaluation:
    index: int
    stage: str
    score: float
    structure: str
    latent_file: str
    budget: str
    seed: int
    u: list[float] | None


from .chart import hypersphere_vertices, hypersphere_weights, map_u_to_latent


def _validate_budget(budget: Mapping[str, Any]) -> tuple[int, int, int, int]:
    n = int(budget["N"])
    k = int(budget["K"])
    m = int(budget["M"])
    d = int(budget["d"])
    if not (0 < k <= n):
        raise ValueError(f"Require 0 < K <= N, got N={n}, K={k}")
    if not (d >= 2 and d <= m):
        raise ValueError(f"Require 2 <= d <= M, got M={m}, d={d}")
    if not (m < n and n - m >= 2):
        raise ValueError("Require M < N and at least two post-seed initial GP evaluations")
    return n, k, m, d


def _fit_and_acquire(train_u: np.ndarray, train_scores: np.ndarray) -> np.ndarray:
    train_x = torch.as_tensor(train_u, dtype=torch.double)
    train_y = torch.as_tensor(train_scores[:, None], dtype=torch.double)
    model = SingleTaskGP(
        train_x,
        train_y,
        mean_module=ConstantMean(),
        covar_module=ScaleKernel(RBFKernel(ard_num_dims=train_u.shape[1])),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=float(np.max(train_scores)),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
    )
    dim = train_u.shape[1]
    bounds = torch.stack(
        [torch.zeros(dim, dtype=torch.double), torch.ones(dim, dtype=torch.double)]
    )
    candidate, _ = optimize_acqf(
        acq_function=acquisition,
        bounds=bounds,
        q=1,
        num_restarts=min(10, max(2, train_u.shape[0] // 2)),
        raw_samples=128,
        options={"batch_limit": 5, "maxiter": 100},
    )
    point = candidate.detach().cpu().numpy()[0]
    if not np.all(np.isfinite(point)):
        raise RuntimeError("BoTorch acquisition optimization returned a non-finite point")
    if np.any(point < -1.0e-8) or np.any(point > 1.0 + 1.0e-8):
        raise RuntimeError("BoTorch acquisition optimization returned a point outside [0, 1]")
    return np.clip(point, 0.0, 1.0)


def run_o3(
    *,
    adapter: GeneratorOracle,
    config: Mapping[str, Any],
    budget: Mapping[str, Any],
    run_seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Run one budget/seed pair and write all artifacts below output_dir."""

    n, k, m, d = _validate_budget(budget)
    latent_dim = int(config["latent_dim"])
    budget_name = str(budget.get("name", f"n{n}_k{k}"))
    rng = np.random.default_rng(run_seed)
    random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_dir = output_dir / "phase1"
    bo_dir = output_dir / "bo"
    seed_dir.mkdir(exist_ok=True)
    bo_dir.mkdir(exist_ok=True)

    evaluations: list[Evaluation] = []
    phase1_latents = np.empty((m, latent_dim), dtype=np.float64)
    phase1_scores = np.empty(m, dtype=np.float64)

    def print_progress(status: str, score: float | None = None) -> None:
        completed = len(evaluations)
        width = 24
        filled = int(width * completed / n)
        bar = "=" * filled + ">" * (completed < n) + " " * max(0, width - filled - (completed < n))
        score_text = "" if score is None else f" | score={score:.4f}"
        print(
            f"[{budget_name} seed={run_seed}] [{bar}] {completed}/{n} {status}{score_text}",
            flush=True,
        )

    def evaluate(
        latent: np.ndarray,
        stage: str,
        structure_path: Path,
        u: np.ndarray | None,
    ) -> float:
        structure_path.parent.mkdir(parents=True, exist_ok=True)
        latent_path = output_dir / "latents" / f"latent_{len(evaluations):04d}.npy"
        latent_path.parent.mkdir(exist_ok=True)
        np.save(latent_path, latent)
        print_progress(f"generating {stage}")
        written_path = adapter.generate(
            latent=latent,
            output_path=structure_path,
            config=config,
            metadata={
                "budget": budget_name,
                "seed": run_seed,
                "stage": stage,
                "evaluation_index": len(evaluations),
                "u": None if u is None else u.tolist(),
                "deterministic": True,
            },
        )
        final_path = Path(written_path) if written_path is not None else structure_path
        if not final_path.exists():
            raise FileNotFoundError(
                f"The adapter did not write a structure at {final_path}. "
                "The generator must create the requested PDB/mmCIF file."
            )
        score = float(adapter.score(final_path, config))
        if not np.isfinite(score):
            raise ValueError(f"Oracle returned a non-finite score for {final_path}")
        evaluations.append(
            Evaluation(
                index=len(evaluations),
                stage=stage,
                score=score,
                structure=str(final_path),
                latent_file=str(latent_path),
                budget=budget_name,
                seed=run_seed,
                u=None if u is None else u.tolist(),
            )
        )
        print_progress("completed", score)
        return score

    for i in range(m):
        latent = rng.normal(size=latent_dim)
        phase1_latents[i] = latent
        phase1_scores[i] = evaluate(
            latent,
            "phase1_random",
            seed_dir / f"sample_{i:04d}.pdb",
            None,
        )

    selected = np.argsort(phase1_scores)[-d:][::-1]
    seed_latents = phase1_latents[selected].copy()
    seed_scores = phase1_scores[selected].copy()
    np.savez_compressed(
        output_dir / "phase1_latents.npz",
        latents=phase1_latents,
        scores=phase1_scores,
        selected_indices=selected,
        selected_latents=seed_latents,
        selected_scores=seed_scores,
    )

    # The selected phase-1 structures are already scored. Project each seed
    # through the reference chart's inverse map, reuse those scores, and spend
    # two fresh calls on random points. This exactly accounts for N calls.
    chart = SurrogateChart(seed_latents)
    train_u = np.asarray(chart.from_z_to_u(seed_latents), dtype=np.float64)
    train_scores = seed_scores.copy()

    for i in range(2):
        u = rng.uniform(0.0, 1.0, size=d - 1)
        latent = map_u_to_latent(u, seed_latents)
        score = evaluate(latent, "bo_initial_random", bo_dir / f"initial_{i:02d}.pdb", u)
        train_u = np.vstack([train_u, u])
        train_scores = np.append(train_scores, score)

    for round_index in range(n - m - 2):
        print_progress(f"fitting BO round {round_index + 1}/{n - m - 2}")
        u = _fit_and_acquire(train_u, train_scores)
        latent = chart.from_u_to_z(u)
        score = evaluate(
            latent,
            "bo_acquisition",
            bo_dir / f"round_{round_index:04d}.pdb",
            u,
        )
        train_u = np.vstack([train_u, u])
        train_scores = np.append(train_scores, score)

    if len(evaluations) != n:
        raise AssertionError(f"Budget accounting error: expected {n}, got {len(evaluations)}")

    new_scores = np.asarray([item.score for item in evaluations[m:]], dtype=np.float64)

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
        "budget": budget_name,
        "N": n,
        "K": k,
        "M": m,
        "d": d,
        "k": k,
        "D": latent_dim,
        "latent_dim": latent_dim,
        "chart_version": SURROGATE_CHART_VERSION,
        "o3_chart": "knothe_rosenblatt_positive_unit_hypersphere",
        "generator_atom_count": getattr(adapter, "atom_count", None),
        "generator_atom_slots": getattr(adapter, "atom_slots", None),
        "seed": run_seed,
        "oracle_evaluations": len(evaluations),
        "generator_sampling": "deterministic_pf_ode",
        "selection_metric": "oracle_tm_score",
        "total_mean": float(np.mean(all_scores)),
        "mean_all": float(np.mean(all_scores)),
        "phase1_max": float(np.max(phase1_scores)),
        "new_points_max": float(np.max(new_scores)),
        "new_points_improvement": float(np.max(new_scores) - np.max(phase1_scores)),
        "max_of_K": float(np.max(selected_scores)),
        "top_k_mean": float(np.mean(selected_scores)),
        "mean_of_K": float(np.mean(selected_scores)),
        "best_structure": returned[0].structure,
        "output_dir": str(output_dir),
    }
    summary.update(collect_run_metadata(config=config, adapter=adapter))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
