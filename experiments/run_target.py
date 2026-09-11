"""Run generic single-chain stochastic, PF-ODE, and O3 target experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _path in (REPO_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from o3_boltz.adapter import load_adapter  # noqa: E402
from o3_boltz.o3 import run_o3  # noqa: E402
from o3_boltz.random_baseline import run_random_pfode  # noqa: E402

from experiments.target_utils import load_target_config  # noqa: E402


METHODS = ("paired", "stochastic", "pfode", "o3", "all")
DEFAULT_SEED_START = 20250117
DEFAULT_SEED_STEP = 1009


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--d", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--reference-chain", default="A")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--seed-step", type=int, default=DEFAULT_SEED_STEP)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.N < 1 or args.K < 1 or args.K > args.N:
        raise ValueError(f"Require 1 <= K <= N, got N={args.N}, K={args.K}")
    if args.replicates < 1:
        raise ValueError("replicates must be at least 1")
    if args.seed_step == 0:
        raise ValueError("seed-step must not be 0")
    has_m = args.M is not None
    has_d = args.d is not None
    if args.method in {"o3", "all"} and not (has_m and has_d):
        raise ValueError("--method o3/all requires explicit --M and --d; values are never guessed")
    if args.method not in {"o3", "all"} and (has_m or has_d):
        raise ValueError("--M and --d are only valid for --method o3 or --method all")
    if args.method in {"o3", "all"}:
        if args.M < 1 or args.d < 2 or args.d > args.M or args.M >= args.N or args.N - args.M < 2:
            raise ValueError(
                f"Require 2 <= d <= M < N and N-M >= 2, got N={args.N}, M={args.M}, d={args.d}"
            )


def _seeds(args: argparse.Namespace) -> list[int]:
    values = [args.seed_start + index * args.seed_step for index in range(args.replicates)]
    if any(seed < 0 or seed >= 2**32 for seed in values):
        raise ValueError("replicate seeds must be unsigned 32-bit integers")
    if len(set(values)) != len(values):
        raise ValueError("replicate seeds must be unique")
    return values


def _budget(args: argparse.Namespace) -> dict[str, int | str]:
    result: dict[str, int | str] = {"name": f"n{args.N}_k{args.K}", "N": args.N, "K": args.K}
    if args.method in {"o3", "all"}:
        result.update({"M": args.M, "d": args.d})
    return result


def _method_dirs(method: str) -> list[str]:
    return {
        "paired": ["stochastic", "pf_ode"],
        "stochastic": ["stochastic"],
        "pfode": ["pf_ode"],
        "o3": ["o3"],
        "all": ["stochastic", "pf_ode", "o3"],
    }[method]


def _load_evaluations(root: Path, directory: str, seed: int) -> list[dict[str, Any]]:
    path = root / directory / f"seed_{seed}" / "evaluations.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation report: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a list in {path}")
    return records


def _write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    fasta_path: Path,
    reference_path: Path,
    seeds: list[int],
    latent_dim: int,
) -> None:
    boltz = config["boltz2"]
    metadata: dict[str, Any] = {
        "target": str(args.target).upper(),
        "fasta": str(fasta_path),
        "reference": str(reference_path),
        "reference_chain": args.reference_chain,
        "sequence_length": len(config["target"]["sequence"]),
        "latent_dim": latent_dim,
        "N": args.N,
        "K": args.K,
        "sampling_steps": boltz["sampling_steps"],
        "recycling_steps": boltz["recycling_steps"],
        "step_scale": boltz["step_scale"],
        "stochastic_gamma_0": boltz["stochastic_gamma_0"],
        "msa": "empty",
        "explicit_latent": boltz["explicit_latent"],
        "replicate_seeds": seeds,
    }
    if args.method in {"o3", "all"}:
        metadata.update({"M": args.M, "d": args.d})
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_paired_reports(root: Path, target: str, summaries: dict[str, list[dict[str, Any]]]) -> None:
    stochastic = summaries["stochastic"]
    pfode = summaries["pf_ode"]
    if len(stochastic) != len(pfode):
        raise RuntimeError("Paired methods returned different replicate counts")
    score_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for stoch_summary, pf_summary in zip(stochastic, pfode):
        seed = int(stoch_summary["seed"])
        if seed != int(pf_summary["seed"]):
            raise RuntimeError("Paired methods are not aligned to the same replicate seed")
        stoch_dir = root / "stochastic" / f"seed_{seed}"
        pf_dir = root / "pf_ode" / f"seed_{seed}"
        stoch_latent_dir = stoch_dir / "latents"
        pf_latent_dir = pf_dir / "latents"
        stoch_eval = {
            int(row["index"]): row
            for row in _load_evaluations(root, "stochastic", seed)
        }
        pf_eval = {
            int(row["index"]): row
            for row in _load_evaluations(root, "pf_ode", seed)
        }
        if set(stoch_eval) != set(pf_eval):
            raise RuntimeError(f"Paired evaluation indices differ for seed {seed}")
        deltas: list[float] = []
        for index in sorted(stoch_eval):
            stoch_latent = np.load(stoch_latent_dir / f"latent_{index:04d}.npy", allow_pickle=False)
            pf_latent = np.load(pf_latent_dir / f"latent_{index:04d}.npy", allow_pickle=False)
            if not np.array_equal(stoch_latent, pf_latent):
                raise RuntimeError(
                    f"Paired latent mismatch for target {target}, seed {seed}, index {index}"
                )
            stoch_score = float(stoch_eval[index]["score"])
            pf_score = float(pf_eval[index]["score"])
            delta = pf_score - stoch_score
            deltas.append(delta)
            score_rows.append(
                {
                    "target": target.upper(),
                    "seed": seed,
                    "index": index,
                    "tm_stochastic": stoch_score,
                    "tm_pfode": pf_score,
                    "delta_tm": delta,
                }
            )
        summary_rows.append(
            {
                "target": target.upper(),
                "seed": seed,
                "stochastic_total_mean": float(stoch_summary["total_mean"]),
                "pfode_total_mean": float(pf_summary["total_mean"]),
                "mean_paired_delta": float(np.mean(deltas)),
                "stochastic_mean_of_K": float(stoch_summary["mean_of_K"]),
                "pfode_mean_of_K": float(pf_summary["mean_of_K"]),
                "stochastic_max_of_K": float(stoch_summary["max_of_K"]),
                "pfode_max_of_K": float(pf_summary["max_of_K"]),
            }
        )
    fields = (
        list(score_rows[0])
        if score_rows
        else ["target", "seed", "index", "tm_stochastic", "tm_pfode", "delta_tm"]
    )
    with (root / "paired_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(score_rows)
    fields = list(summary_rows[0]) if summary_rows else ["target", "seed"]
    with (root / "paired_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def run(args: argparse.Namespace) -> Path:
    _validate_args(args)
    seeds = _seeds(args)
    config, fasta_path, reference_path = load_target_config(
        repo_root=REPO_ROOT, target=args.target, reference_chain=args.reference_chain
    )
    adapter = load_adapter("adapters.boltz2_pfode:create", config)
    config["latent_dim"] = int(adapter.latent_dim)
    budget = _budget(args)
    target_slug = str(args.target).strip().lower()
    root = REPO_ROOT / "outputs" / target_slug / str(budget["name"]) / "runs" / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    _write_metadata(
        root / "run_metadata.json",
        args=args,
        config=config,
        fasta_path=fasta_path,
        reference_path=reference_path,
        seeds=seeds,
        latent_dim=int(adapter.latent_dim),
    )
    shared_root = root / "shared_latents"
    summaries: dict[str, list[dict[str, Any]]] = {
        directory: [] for directory in _method_dirs(args.method)
    }
    for seed in seeds:
        shared_path = shared_root / f"seed_{seed}.npy"
        if args.method in {"paired", "stochastic", "all"}:
            summaries.setdefault("stochastic", []).append(
                run_random_pfode(
                    adapter=adapter,
                    config=config,
                    budget=budget,
                    run_seed=seed,
                    output_dir=root / "stochastic" / f"seed_{seed}",
                    resume=args.resume,
                    method="best_k_of_n",
                    shared_latent_path=shared_path,
                )
            )
        if args.method in {"paired", "pfode", "all"}:
            summaries.setdefault("pf_ode", []).append(
                run_random_pfode(
                    adapter=adapter,
                    config=config,
                    budget=budget,
                    run_seed=seed,
                    output_dir=root / "pf_ode" / f"seed_{seed}",
                    resume=args.resume,
                    method="random_pfode",
                    shared_latent_path=shared_path,
                )
            )
        if args.method in {"o3", "all"}:
            summaries.setdefault("o3", []).append(
                run_o3(
                    adapter=adapter,
                    config=config,
                    budget=budget,
                    run_seed=seed,
                    output_dir=root / "o3" / f"seed_{seed}",
                    resume=args.resume,
                )
            )
    if args.method == "paired":
        _write_paired_reports(root, args.target, summaries)
    elif args.method == "all":
        _write_paired_reports(root, args.target, summaries)
    return root


def main(argv: list[str] | None = None) -> None:
    root = run(parse_args(argv))
    print(f"Run complete: {root}", flush=True)


if __name__ == "__main__":
    main()
