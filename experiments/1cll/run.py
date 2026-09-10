"""Canonical command for the 1CLL K=10, N=100 comparison."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import common
from common import (
    BUNDLE,
    DEFAULT_REPLICATE_SEED_START,
    DEFAULT_REPLICATE_SEED_STEP,
    REPO_ROOT,
    SUPPORTED_REPLICATES,
    configure_budget,
    random_replicate_seeds,
    resolve_replicate_seeds,
    shared_replicate_seeds,
)
from compare_methods import write_report as write_comparison_report
from public_runner import run as run_public
from random_pfode_runner import run as run_random_pfode


def comparison_methods(method: str) -> list[str]:
    return {
        "best-k-of-n": ["best_k_of_n"],
        "o3": ["o3"],
        "random-pfode": ["random_pfode"],
        "matched-stochastic": ["best_k_of_n"],
        "public-best-k-of-n": ["public_best_k_of_n"],
        "paired": ["best_k_of_n", "random_pfode"],
        "both": ["best_k_of_n", "o3"],
        "all": ["best_k_of_n", "o3", "random_pfode"],
    }[method]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=("public-best-k-of-n", "best-k-of-n", "o3", "random-pfode", "matched-stochastic", "paired", "both", "all"),
        default="both",
    )
    parser.add_argument(
        "--budget",
        "--only",
        dest="budget",
        choices=("n20_k2", "n50_k5", "n100_k10"),
        default="n100_k10",
        help="Run one supported budget (the --only spelling is kept for run_experiment.sh).",
    )
    parser.add_argument(
        "--replicates", type=int, choices=SUPPORTED_REPLICATES, default=5
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_REPLICATE_SEED_START)
    parser.add_argument("--seed-step", type=int, default=DEFAULT_REPLICATE_SEED_STEP)
    parser.add_argument(
        "--seed-list",
        nargs="+",
        type=int,
        default=None,
        help="Reuse an explicit replicate seed list printed by an earlier run",
    )
    seed_mode = parser.add_mutually_exclusive_group()
    seed_mode.add_argument(
        "--random-seeds",
        dest="seed_mode",
        action="store_const",
        const="random",
        help="Generate fresh replicate seeds (the default when no seed source is given).",
    )
    seed_mode.add_argument(
        "--fixed-seed-schedule",
        dest="seed_mode",
        action="store_const",
        const="fixed",
        help="Use the reproducible --seed-start/--seed-step arithmetic schedule.",
    )
    parser.add_argument(
        "--seeds-from-baseline-run",
        metavar="RUN_ID",
        default=None,
        help="Load the recorded replicate seeds from a completed Best K-of-N run.",
    )
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Opt into persistent baseline inference and batched O3 initialization (try 2 or 4).")
    parser.add_argument(
        "--gpus",
        default=None,
        help="Linux replicate workers: comma-separated GPU indices or 'auto'.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--comparison-report",
        action="store_true",
        help="Write the seed-aligned comparison report after the selected methods finish.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run backend and sampler verification only")
    return parser.parse_args()


def load_baseline_seeds(run_id: str, budget: str, replicates: int) -> list[int]:
    """Load and validate the seed list recorded by a completed baseline run."""

    candidates = [
        common.output_root("best_k_of_n", run_id) / "run_metadata.json",
        common.output_root("public_best_k_of_n", run_id) / "provenance.json",
        common.output_root("best_k_of_n", run_id) / "provenance.json",  # Legacy runs.
    ]
    baseline_metadata = next((path for path in candidates if path.is_file()), candidates[0])
    if not baseline_metadata.is_file():
        raise FileNotFoundError(
            f"Best K-of-N run metadata does not exist: {baseline_metadata}"
        )
    recorded = json.loads(baseline_metadata.read_text(encoding="utf-8"))
    if recorded.get("budget") != budget:
        raise ValueError(
            f"Baseline run uses budget {recorded.get('budget')!r}, not {budget!r}"
        )
    recorded_seeds = recorded.get("seeds")
    if not isinstance(recorded_seeds, list):
        raise ValueError(f"Baseline run has no recorded seed list: {baseline_metadata}")
    return resolve_replicate_seeds(replicates, seeds=recorded_seeds)


def load_comparison_seeds(run_id: str, budget: str, replicates: int) -> list[int]:
    """Load seeds saved before generation for an interrupted comparison."""

    manifest_path = common.comparison_run_root(run_id) / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run manifest does not exist: {manifest_path}")
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if recorded.get("budget") != budget:
        raise ValueError(
            f"Run manifest uses budget {recorded.get('budget')!r}, not {budget!r}"
        )
    recorded_seeds = recorded.get("seeds")
    if not isinstance(recorded_seeds, list):
        raise ValueError(f"Run manifest has no recorded seed list: {manifest_path}")
    return resolve_replicate_seeds(replicates, seeds=recorded_seeds)


def run_o3(
    replicates: int,
    run_id: str,
    config_path: Path,
    budget: str,
    seeds: list[int],
    resume: bool = False,
    batch_size: int = 1,
    worker_only: bool = False,
) -> None:
    uv = os.environ.get("BOLTZ_PUBLIC_UV") or shutil.which("uv") or "uv"
    command = [
        uv, "run", "--project", str(REPO_ROOT), "python", "-m", "o3_boltz.cli",
        "--config", str(config_path),
        "--replicates", str(replicates),
        "--run-id", run_id,
        "--only", budget,
        "--seed-list", *(str(seed) for seed in seeds),
    ]
    command += ["--batch-size", str(batch_size)]
    if resume:
        command.append("--resume")
    if worker_only:
        command.append("--worker-only")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    batch_size = getattr(args, "batch_size", None)
    gpu_spec = getattr(args, "gpus", None)
    if batch_size is not None and batch_size < 1:
        raise ValueError("--batch-size must be positive")
    os.environ.setdefault("UV_CACHE_DIR", str(REPO_ROOT / ".uv-cache"))
    configure_budget(args.budget)
    if args.smoke:
        subprocess.run(
            [sys.executable, str(BUNDLE / "verify.py"), "--gpu"],
            cwd=REPO_ROOT,
            check=True,
        )
        return
    run_id = (
        args.run_id
        or args.seeds_from_baseline_run
        or f"{args.budget}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    selected_seed_sources = sum(
        (
            args.seed_mode is not None,
            args.seed_list is not None,
            args.seeds_from_baseline_run is not None,
        )
    )
    if selected_seed_sources > 1:
        raise ValueError(
            "Use only one seed mode/source: random/fixed, --seed-list, or "
            "--seeds-from-baseline-run"
        )
    if args.seeds_from_baseline_run is not None:
        shared_seeds = load_baseline_seeds(
            args.seeds_from_baseline_run, args.budget, args.replicates
        )
        seed_mode = f"baseline_run:{args.seeds_from_baseline_run}"
    elif args.seed_list is not None:
        shared_seeds = resolve_replicate_seeds(args.replicates, seeds=args.seed_list)
        seed_mode = "explicit_list"
    elif args.resume and args.run_id is not None and (
        common.comparison_run_root(run_id) / "run_manifest.json"
    ).is_file():
        shared_seeds = load_comparison_seeds(run_id, args.budget, args.replicates)
        seed_mode = "resume_manifest"
    elif args.resume and args.run_id is not None and (
        common.output_root("best_k_of_n", run_id) / "provenance.json"
    ).is_file():
        # Compatibility for runs created before comparison manifests existed.
        shared_seeds = load_baseline_seeds(run_id, args.budget, args.replicates)
        seed_mode = "resume_baseline_provenance"
    elif args.seed_mode != "fixed":
        shared_seeds = random_replicate_seeds(args.replicates)
        seed_mode = "fresh_os_random"
    else:
        shared_seeds = shared_replicate_seeds(
            args.replicates, seed_start=args.seed_start, seed_step=args.seed_step
        )
        seed_mode = "arithmetic_schedule"
    gpu_ids = None
    if gpu_spec is not None:
        import multi_gpu

        gpu_ids = multi_gpu.resolve_gpu_ids(gpu_spec)
    print(f"Seed mode: {seed_mode}", flush=True)
    print(f"Shared replicate seeds for all selected methods: {shared_seeds}", flush=True)
    manifest_path = common.comparison_run_root(run_id) / "run_manifest.json"
    manifest = {
        "target": "1cll",
        "budget": args.budget,
        "run_id": run_id,
        "method_selection": args.method,
        "step_scale": 1.0,
        "batch_size": batch_size,
        "requested_gpus": gpu_ids,
        "replicates": args.replicates,
        "seed_mode": seed_mode,
        "seeds": shared_seeds,
    }
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        immutable_fields = ("target", "budget", "run_id", "replicates", "seeds", "batch_size", "step_scale")
        mismatches = {
            key: (existing_manifest.get(key), manifest.get(key))
            for key in immutable_fields
            if existing_manifest.get(key) != manifest.get(key)
        }
        if mismatches:
            raise ValueError(
                f"Run ID {run_id!r} already has incompatible metadata: {mismatches}"
            )
    else:
        common.write_json(manifest_path, manifest)
    print(
        f"Run manifest: {manifest_path}",
        flush=True,
    )

    if gpu_ids is not None:
        parallel_execution = multi_gpu.launch_workers(
            method=args.method,
            budget=args.budget,
            run_id=run_id,
            seeds=shared_seeds,
            gpu_ids=gpu_ids,
            batch_size=batch_size,
            resume=args.resume,
        )
        multi_gpu.finalize_reports(
            method=args.method,
            budget=args.budget,
            run_id=run_id,
            seeds=shared_seeds,
            batch_size=batch_size,
            parallel_execution=parallel_execution,
        )
        if args.comparison_report:
            report = write_comparison_report(
                budget=args.budget,
                run_id=run_id,
                methods=comparison_methods(args.method),
            )
            print(
                f"Combined comparison report: {report['comparison_summary_csv']}",
                flush=True,
            )
        common.write_json(
            manifest_path.parent / "execution_timings_this_session.json",
            {
                "seconds_by_method": {},
                "multi_gpu_wall_seconds": parallel_execution["wall_seconds"],
                "worker_seconds_by_method": {
                    str(worker["gpu"]): worker["seconds_by_method"]
                    for worker in parallel_execution["workers"]
                },
                "includes_model_startup": True,
                "resume": args.resume,
                "batch_size": batch_size,
                "gpus": gpu_ids,
            },
        )
        print(
            f"Multi-GPU run complete. Schedule: "
            f"{common.comparison_run_root(run_id) / 'gpu_schedule_this_session.json'}",
            flush=True,
        )
        return

    session_timings = {}

    def timed(name, function, *positional, **keywords):
        tick = time.perf_counter()
        function(*positional, **keywords)
        session_timings[name] = time.perf_counter() - tick
        common.write_json(manifest_path.parent / "execution_timings_this_session.json", {
            "seconds_by_method": session_timings, "includes_model_startup": True,
            "resume": args.resume, "batch_size": batch_size,
        })

    if args.method == "public-best-k-of-n":
        timed("public_best_k_of_n", run_public,
            args.replicates,
            run_id,
            resume=args.resume,
            seed_start=args.seed_start,
            seed_step=args.seed_step,
            seeds=shared_seeds,
            **({"batch_size": batch_size} if batch_size is not None else {}),
        )
    if args.method in {"o3", "both", "all"}:
        timed("o3", run_o3,
            args.replicates,
            run_id,
            REPO_ROOT / "configs" / "1cll.yaml",
            args.budget,
            shared_seeds,
            resume=args.resume,
            batch_size=batch_size or 1,
        )
    if args.method in {"best-k-of-n", "matched-stochastic", "paired", "both", "all"}:
        timed("best_k_of_n", run_random_pfode,
            args.replicates, run_id, REPO_ROOT / "configs" / "1cll.yaml", args.budget,
            seeds=shared_seeds, resume=args.resume, method="best_k_of_n",
        )
    if args.method in {"random-pfode", "paired", "all"}:
        timed("random_pfode", run_random_pfode,
            args.replicates,
            run_id,
            REPO_ROOT / "configs" / "1cll.yaml",
            args.budget,
            seeds=shared_seeds,
            resume=args.resume,
        )
    if args.comparison_report:
        report = write_comparison_report(
            budget=args.budget,
            run_id=run_id,
            methods=comparison_methods(args.method),
        )
        print(
            f"Combined comparison report: {report['comparison_summary_csv']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
