from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
from warnings import WarningMessage

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.constraints import Interval
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


def _o3_gp_warning_handler(warning: WarningMessage) -> bool:
    """Accept recoverable GP optimizer warnings but not optimizer timeouts.

    BoTorch 0.18 treats SciPy line-search warnings such as
    ``ABNORMAL_TERMINATION_IN_LNSRCH`` as retry-worthy. On this small,
    deterministic oracle problem, the returned model can still be finite and
    usable for acquisition; retrying the same fit five times can otherwise
    abort an otherwise valid O3 run. Timeouts remain unresolved so they still
    trigger the normal failure path.
    """

    if not issubclass(warning.category, OptimizationWarning):
        return False
    message = str(warning.message)
    if "Optimization timed out" in message:
        return False
    print(f"[O3] accepting recoverable GP optimizer warning: {message}", flush=True)
    return True


def _fit_and_acquire(train_u: np.ndarray, train_scores: np.ndarray) -> np.ndarray:
    train_x = torch.as_tensor(train_u, dtype=torch.double)
    train_y = torch.as_tensor(train_scores[:, None], dtype=torch.double)
    model = SingleTaskGP(
        train_x,
        train_y,
        mean_module=ConstantMean(),
        # U lives in a unit cube.  Without an upper bound, the optimizer can
        # make an ARD length scale effectively infinite, producing an almost
        # rank-one covariance matrix that cannot be Cholesky-factorized after
        # the chart points accumulate.  Ten is already much larger than the
        # cube's diameter and is therefore a numerical guard, not a material
        # restriction on the surrogate's useful length scales.
        covar_module=ScaleKernel(
            RBFKernel(
                ard_num_dims=train_u.shape[1],
                lengthscale_constraint=Interval(1.0e-2, 10.0),
            )
        ),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, warning_handler=_o3_gp_warning_handler)

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


def _load_resume_state(
    *,
    adapter: GeneratorOracle,
    config: Mapping[str, Any],
    budget_name: str,
    run_seed: int,
    output_dir: Path,
    m: int,
    d: int,
    latent_dim: int,
    bo_rounds: int,
    expected_phase1_latents: np.ndarray,
    initial_u_points: np.ndarray,
) -> dict[str, Any] | None:
    """Rebuild the in-memory O3 state from a contiguous saved prefix.

    Latents are checked against the points implied by ``run_seed``.  This is
    what makes ``--resume`` a continuation of the same experiment rather than
    a new random trajectory that happens to share an output directory.
    """

    archive_path = output_dir / "phase1_latents.npz"
    evaluations: list[Evaluation] = []
    phase1_latents = np.empty((m, latent_dim), dtype=np.float64)
    phase1_scores = np.empty(m, dtype=np.float64)

    def add_existing(
        *,
        index: int,
        stage: str,
        latent_path: Path,
        structure_path: Path,
        u: np.ndarray | None,
        expected_latent: np.ndarray | None = None,
    ) -> float:
        if not latent_path.exists() or not structure_path.exists():
            raise FileNotFoundError(
                f"Cannot resume {output_dir}: missing {latent_path} or {structure_path}"
            )
        latent = np.asarray(np.load(latent_path), dtype=np.float64)
        if latent.shape != (latent_dim,) or not np.all(np.isfinite(latent)):
            raise ValueError(f"Cannot resume {output_dir}: invalid latent in {latent_path}")
        if expected_latent is not None and not np.array_equal(latent, expected_latent):
            raise ValueError(
                f"Cannot resume {output_dir}: {latent_path} does not match seed {run_seed}"
            )
        value = float(adapter.score(structure_path, config))
        if not np.isfinite(value):
            raise ValueError(f"Cannot resume {output_dir}: non-finite score for {structure_path}")
        evaluations.append(
            Evaluation(
                index=index,
                stage=stage,
                score=value,
                structure=str(structure_path),
                latent_file=str(latent_path),
                budget=budget_name,
                seed=run_seed,
                u=None if u is None else u.tolist(),
            )
        )
        return value

    phase1_completed = 0
    for index in range(m):
        latent_path = output_dir / "latents" / f"latent_{index:04d}.npy"
        structure_path = output_dir / "phase1" / f"sample_{index:04d}.pdb"
        if structure_path.exists() and not latent_path.exists():
            raise FileNotFoundError(
                f"Cannot resume {output_dir}: structure exists without {latent_path}"
            )
        if not (latent_path.exists() and structure_path.exists()):
            break
        phase1_latents[index] = expected_phase1_latents[index]
        phase1_scores[index] = add_existing(
            index=index,
            stage="phase1_random",
            latent_path=latent_path,
            structure_path=structure_path,
            u=None,
            expected_latent=expected_phase1_latents[index],
        )
        phase1_completed += 1

    batch_size = int(config.get("inference_batch_size", 1))
    if phase1_completed < m and batch_size > 1:
        # Replay the original whole batch after an interrupted multi-file write.
        # Generating only its suffix would change the numerical batch shape.
        phase1_completed -= phase1_completed % batch_size
        evaluations = evaluations[:phase1_completed]
    if phase1_completed == 0 and not archive_path.exists():
        return None
    if archive_path.exists() and phase1_completed != m:
        raise ValueError(
            f"Cannot resume {output_dir}: phase1_latents.npz exists but only "
            f"{phase1_completed}/{m} phase-1 structures are complete"
        )
    if phase1_completed < m:
        print(
            f"[{budget_name} seed={run_seed}] resuming from "
            f"{phase1_completed}/{m + 2 + bo_rounds} saved evaluations",
            flush=True,
        )
        return {
            "evaluations": evaluations,
            "phase1_latents": phase1_latents,
            "phase1_scores": phase1_scores,
            "phase1_completed": phase1_completed,
            "phase2_completed": 0,
            "bo_rounds_completed": 0,
        }

    selected = np.argsort(phase1_scores)[-d:][::-1]
    seed_latents = phase1_latents[selected].copy()
    seed_scores = phase1_scores[selected].copy()

    if archive_path.exists():
        with np.load(archive_path) as archive:
            archived_latents = np.asarray(archive["latents"], dtype=np.float64)
            archived_selected = np.asarray(archive["selected_indices"], dtype=np.int64)
        if not np.array_equal(archived_latents, phase1_latents):
            raise ValueError(f"Cannot resume {output_dir}: phase-1 archive latent mismatch")
        if not np.array_equal(archived_selected, selected):
            raise ValueError(f"Cannot resume {output_dir}: phase-1 archive selection mismatch")

    chart = SurrogateChart(seed_latents)
    train_u = np.asarray(chart.from_z_to_u(seed_latents), dtype=np.float64)
    train_scores = seed_scores.copy()
    phase2_completed = 0
    for index in range(2):
        latent_path = output_dir / "latents" / f"latent_{m + index:04d}.npy"
        u_path = output_dir / "u" / f"u_{m + index:04d}.npy"
        structure_path = output_dir / "bo" / f"initial_{index:02d}.pdb"
        if structure_path.exists() and not latent_path.exists():
            raise FileNotFoundError(
                f"Cannot resume {output_dir}: structure exists without {latent_path}"
            )
        if not (latent_path.exists() and structure_path.exists()):
            break
        if not u_path.exists():
            raise FileNotFoundError(
                f"Cannot resume {output_dir}: completed U evaluation is missing {u_path}"
            )
        u = np.asarray(np.load(u_path), dtype=np.float64)
        if not np.array_equal(u, initial_u_points[index]):
            raise ValueError(
                f"Cannot resume {output_dir}: {u_path} does not match seed {run_seed}"
            )
        expected_latent = np.asarray(chart.from_u_to_z(u), dtype=np.float64)
        score = add_existing(
            index=m + index,
            stage="bo_initial_random",
            latent_path=latent_path,
            structure_path=structure_path,
            u=u,
            expected_latent=expected_latent,
        )
        train_u = np.vstack([train_u, u])
        train_scores = np.append(train_scores, score)
        phase2_completed += 1

    if batch_size > 1 and phase2_completed == 1:
        phase2_completed = 0
        evaluations = evaluations[:m]
        train_u = np.asarray(chart.from_z_to_u(seed_latents), dtype=np.float64)
        train_scores = seed_scores.copy()

    bo_rounds_completed = 0
    for round_index in range(bo_rounds):
        latent_path = output_dir / "latents" / f"latent_{m + 2 + round_index:04d}.npy"
        u_path = output_dir / "u" / f"u_{m + 2 + round_index:04d}.npy"
        structure_path = output_dir / "bo" / f"round_{round_index:04d}.pdb"
        if structure_path.exists() and not latent_path.exists():
            raise FileNotFoundError(
                f"Cannot resume {output_dir}: structure exists without {latent_path}"
            )
        if not (latent_path.exists() and structure_path.exists()):
            break
        if not u_path.exists():
            raise FileNotFoundError(
                f"Cannot resume {output_dir}: completed BO evaluation is missing {u_path}"
            )
        latent = np.asarray(np.load(latent_path), dtype=np.float64)
        u = np.asarray(np.load(u_path), dtype=np.float64)
        if u.shape != (d - 1,) or not np.all(np.isfinite(u)):
            raise ValueError(f"Cannot resume {output_dir}: invalid U point in {u_path}")
        expected_latent = np.asarray(chart.from_u_to_z(u), dtype=np.float64)
        score = add_existing(
            index=m + 2 + round_index,
            stage="bo_acquisition",
            latent_path=latent_path,
            structure_path=structure_path,
            u=u,
            expected_latent=expected_latent,
        )
        train_u = np.vstack([train_u, u])
        train_scores = np.append(train_scores, score)
        bo_rounds_completed += 1

    if phase2_completed < 2 and bo_rounds_completed:
        raise ValueError(f"Cannot resume {output_dir}: BO rounds exist before both initial U samples")
    print(
        f"[{budget_name} seed={run_seed}] resuming from "
        f"{len(evaluations)}/{m + 2 + bo_rounds} saved evaluations "
        f"({bo_rounds_completed} BO rounds complete)",
        flush=True,
    )
    return {
        "evaluations": evaluations,
        "phase1_latents": phase1_latents,
        "phase1_scores": phase1_scores,
        "phase1_completed": phase1_completed,
        "seed_latents": seed_latents,
        "seed_scores": seed_scores,
        "chart": chart,
        "train_u": train_u,
        "train_scores": train_scores,
        "phase2_completed": phase2_completed,
        "bo_rounds_completed": bo_rounds_completed,
    }


def run_o3(
    *,
    adapter: GeneratorOracle,
    config: Mapping[str, Any],
    budget: Mapping[str, Any],
    run_seed: int,
    output_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Run one budget/seed pair and write all artifacts below output_dir."""

    started = time.perf_counter()
    timings = {"generation_seconds": 0.0, "scoring_seconds": 0.0, "acquisition_seconds": 0.0}
    batch_size = int(config.get("inference_batch_size", 1))
    if batch_size < 1:
        raise ValueError("inference_batch_size must be positive")
    if batch_size > 1 and not callable(getattr(adapter, "generate_batch", None)):
        raise ValueError("The selected adapter does not support generate_batch")
    n, k, m, d = _validate_budget(budget)
    latent_dim = int(config["latent_dim"])
    budget_name = str(budget.get("name", f"n{n}_k{k}"))
    bo_rounds = n - m - 2
    rng = np.random.default_rng(run_seed)
    # Materialize the seed-derived random points up front.  Besides making the
    # protocol explicit, this lets a resumed process validate its saved prefix
    # and continue with exactly the same two random U points.
    expected_phase1_latents = np.stack(
        [rng.normal(size=latent_dim) for _ in range(m)], axis=0
    )
    initial_u_points = np.stack(
        [rng.uniform(0.0, 1.0, size=d - 1) for _ in range(2)], axis=0
    )
    random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_dir = output_dir / "phase1"
    bo_dir = output_dir / "bo"
    seed_dir.mkdir(exist_ok=True)
    bo_dir.mkdir(exist_ok=True)

    settings_path = output_dir / "inference_settings.json"
    settings = {"batch_size": batch_size, "bo_batch_size": 1}
    if resume:
        previous = json.loads(settings_path.read_text()) if settings_path.exists() else {"batch_size": 1, "bo_batch_size": 1}
        if previous != settings:
            raise ValueError("Cannot resume with a different inference batch size")
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    summary_path = output_dir / "summary.json"
    if resume and summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        expected_summary_values = {
            "budget": budget_name,
            "N": n,
            "K": k,
            "M": m,
            "d": d,
            "seed": run_seed,
            "latent_dim": latent_dim,
        }
        mismatches = {
            key: (summary.get(key), expected)
            for key, expected in expected_summary_values.items()
            if summary.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"Cannot resume {output_dir}: completed summary does not match "
                f"this run ({mismatches})"
            )
        print(f"[{budget_name} seed={run_seed}] resume: summary already complete", flush=True)
        return summary

    evaluations: list[Evaluation] = []
    phase1_latents = np.empty((m, latent_dim), dtype=np.float64)
    phase1_scores = np.empty(m, dtype=np.float64)
    resumed = (
        _load_resume_state(
            adapter=adapter,
            config=config,
            budget_name=budget_name,
            run_seed=run_seed,
            output_dir=output_dir,
            m=m,
            d=d,
            latent_dim=latent_dim,
            bo_rounds=bo_rounds,
            expected_phase1_latents=expected_phase1_latents,
            initial_u_points=initial_u_points,
        )
        if resume
        else None
    )

    print(
        f"[{budget_name} seed={run_seed}] O3 protocol: "
        f"{m} random Z samples -> select best {d} seeds -> "
        f"2 random U samples -> {bo_rounds} BO samples",
        flush=True,
    )
    print(
        f"[{budget_name} seed={run_seed}] Total = {m} + 2 + {bo_rounds} = "
        f"{n} oracle evaluations",
        flush=True,
    )

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

    def evaluate_many(items) -> list[float]:
        # Each item is (latent, stage, requested path, U). Save all inputs before
        # generation so a completed structure always has its corresponding Z/U.
        metadata = []
        latent_paths = []
        for offset, (latent, stage, structure_path, u) in enumerate(items):
            index = len(evaluations) + offset
            structure_path.parent.mkdir(parents=True, exist_ok=True)
            latent_path = output_dir / "latents" / f"latent_{index:04d}.npy"
            latent_path.parent.mkdir(exist_ok=True)
            np.save(latent_path, latent)
            latent_paths.append(latent_path)
            if u is not None:
                u_path = output_dir / "u" / f"u_{index:04d}.npy"
                u_path.parent.mkdir(exist_ok=True)
                np.save(u_path, u)
            metadata.append({
                "budget": budget_name, "seed": run_seed, "stage": stage,
                "evaluation_index": index,
                "u": None if u is None else u.tolist(), "deterministic": True,
            })
        print_progress(f"generating {items[0][1]} batch={len(items)}")
        tick = time.perf_counter()
        if len(items) == 1:
            paths = [adapter.generate(latent=items[0][0], output_path=items[0][2],
                                      config=config, metadata=metadata[0])]
        else:
            paths = adapter.generate_batch(
                latents=np.stack([item[0] for item in items]),
                output_paths=[item[2] for item in items], config=config, metadata=metadata,
            )
        timings["generation_seconds"] += time.perf_counter() - tick
        if len(paths) != len(items):
            raise RuntimeError("Adapter returned an incorrect batch size")
        scores = []
        for item, written, latent_path in zip(items, paths, latent_paths):
            latent, stage, structure_path, u = item
            final_path = Path(written) if written is not None else structure_path
            if not final_path.exists():
                raise FileNotFoundError(f"The adapter did not write a structure at {final_path}")
            tick = time.perf_counter()
            score = float(adapter.score(final_path, config))
            timings["scoring_seconds"] += time.perf_counter() - tick
            if not np.isfinite(score):
                raise ValueError(f"Oracle returned a non-finite score for {final_path}")
            evaluations.append(Evaluation(
                index=len(evaluations), stage=stage, score=score, structure=str(final_path),
                latent_file=str(latent_path), budget=budget_name, seed=run_seed,
                u=None if u is None else u.tolist(),
            ))
            scores.append(score)
            print_progress("completed", score)
        return scores

    def evaluate(latent, stage, structure_path, u) -> float:
        return evaluate_many([(latent, stage, structure_path, u)])[0]

    if resumed is None:
        phase1_completed = 0
        phase2_completed = 0
        bo_rounds_completed = 0
    else:
        evaluations = resumed["evaluations"]
        phase1_latents = resumed["phase1_latents"]
        phase1_scores = resumed["phase1_scores"]
        phase1_completed = int(resumed["phase1_completed"])
        phase2_completed = int(resumed["phase2_completed"])
        bo_rounds_completed = int(resumed["bo_rounds_completed"])

    for start in range(phase1_completed, m, batch_size):
        end = min(start + batch_size, m)
        phase1_latents[start:end] = expected_phase1_latents[start:end]
        phase1_scores[start:end] = evaluate_many([
            (expected_phase1_latents[i], "phase1_random", seed_dir / f"sample_{i:04d}.pdb", None)
            for i in range(start, end)
        ])

    selected = np.argsort(phase1_scores)[-d:][::-1]
    seed_latents = phase1_latents[selected].copy()
    seed_scores = phase1_scores[selected].copy()
    selected_score_text = ", ".join(
        f"{int(index)}:{float(score):.4f}"
        for index, score in zip(selected, seed_scores)
    )
    print(
        f"[{budget_name} seed={run_seed}] phase 1 complete: selected best "
        f"{d} seeds from {m} random Z samples by oracle TM-score "
        f"(index:score={selected_score_text})",
        flush=True,
    )
    np.savez_compressed(
        output_dir / "phase1_latents.npz",
        latents=phase1_latents,
        scores=phase1_scores,
        selected_indices=selected,
        selected_latents=seed_latents,
        selected_scores=seed_scores,
    )

    if resumed is None or phase1_completed < m:
        chart = SurrogateChart(seed_latents)
        train_u = np.asarray(chart.from_z_to_u(seed_latents), dtype=np.float64)
        train_scores = seed_scores.copy()
        phase2_completed = 0
        bo_rounds_completed = 0
    else:
        seed_latents = resumed["seed_latents"]
        seed_scores = resumed["seed_scores"]
        chart = resumed["chart"]
        train_u = resumed["train_u"]
        train_scores = resumed["train_scores"]

    # The selected phase-1 structures are already scored. Project each seed
    # through the reference chart's inverse map, reuse those scores, and spend
    # two fresh calls on random points. This exactly accounts for N calls.
    print(
        f"[{budget_name} seed={run_seed}] phase 2: constructing U from the "
        f"selected {d} seeds and evaluating 2 random U samples "
        f"(U dimension={d - 1})",
        flush=True,
    )
    for start in range(phase2_completed, 2, batch_size):
        points = initial_u_points[start:min(start + batch_size, 2)]
        scores = evaluate_many([
            (map_u_to_latent(u, seed_latents), "bo_initial_random", bo_dir / f"initial_{start + offset:02d}.pdb", u)
            for offset, u in enumerate(points)
        ])
        train_u = np.vstack([train_u, points])
        train_scores = np.append(train_scores, scores)

    print(
        f"[{budget_name} seed={run_seed}] phase 2 complete: "
        f"{m} random Z + 2 random U = {m + 2}/{n} evaluations; "
        f"starting phase 3 BO for {bo_rounds} rounds",
        flush=True,
    )
    for round_index in range(bo_rounds_completed, bo_rounds):
        print_progress(f"fitting BO round {round_index + 1}/{bo_rounds}")
        # Make every acquisition round reproducible in isolation so restarting
        # the process does not change BoTorch's Sobol/raw-sample sequence.
        acquisition_seed = (int(run_seed) * 1_000_003 + round_index + 1) % (2**63 - 1)
        torch.manual_seed(acquisition_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(acquisition_seed)
        tick = time.perf_counter()
        u = _fit_and_acquire(train_u, train_scores)
        timings["acquisition_seconds"] += time.perf_counter() - tick
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
    print(
        f"[{budget_name} seed={run_seed}] O3 protocol complete: "
        f"{m} + 2 + {bo_rounds} = {len(evaluations)} oracle evaluations",
        flush=True,
    )

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
        "inference_batch_size": batch_size,
        "bo_batch_size": 1,
        "timings_this_session": {**timings, "wall_seconds": time.perf_counter() - started},
        "o3_chart": "knothe_rosenblatt_positive_unit_hypersphere",
        "generator_atom_count": getattr(adapter, "atom_count", None),
        "generator_atom_slots": getattr(adapter, "atom_slots", None),
        "seed": run_seed,
        "oracle_evaluations": len(evaluations),
        "generator_sampling": "deterministic_pf_ode",
        "selection_metric": "oracle_tm_score",
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
    with (output_dir / "provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "backend": "custom_boltz2_o3",
                "method": "o3",
                "inference_batch_size": batch_size,
                "bo_batch_size": 1,
                "generator": summary.get("generator"),
                "msa_cache": summary.get("msa_cache"),
                "seed": run_seed,
                "N": n,
                "K": k,
                "M": m,
                "d": d,
                "selection_metric": "oracle_tm_score",
            },
            handle,
            indent=2,
        )
    return summary
