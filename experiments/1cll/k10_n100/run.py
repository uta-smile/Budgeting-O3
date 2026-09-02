"""Canonical command for the 1CLL K=10, N=100 comparison."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from common import (
    BUNDLE,
    DEFAULT_REPLICATE_SEED_START,
    DEFAULT_REPLICATE_SEED_STEP,
    REPO_ROOT,
    configure_budget,
    random_replicate_seeds,
    resolve_replicate_seeds,
    shared_replicate_seeds,
)
from public_runner import run as run_public
from random_pfode_runner import run as run_random_pfode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=("best-k-of-n", "o3", "random-pfode", "both", "all"),
        default="both",
    )
    parser.add_argument("--budget", choices=("n20_k2", "n50_k5", "n100_k10"), default="n100_k10")
    parser.add_argument("--replicates", type=int, choices=(1, 3, 5), default=5)
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
    parser.add_argument(
        "--random-seeds",
        action="store_true",
        help="Generate a fresh unique seed for each replicate and share the list across methods",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run backend and sampler verification only")
    return parser.parse_args()


def run_o3(
    replicates: int,
    run_id: str,
    config_path: Path,
    budget: str,
    seeds: list[int],
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
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("UV_CACHE_DIR", str(REPO_ROOT / ".uv-cache"))
    configure_budget(args.budget)
    if args.random_seeds and args.seed_list is not None:
        raise ValueError("Use either --random-seeds or --seed-list, not both")
    if args.seed_list is not None:
        shared_seeds = resolve_replicate_seeds(args.replicates, seeds=args.seed_list)
        seed_mode = "explicit_list"
    elif args.random_seeds:
        shared_seeds = random_replicate_seeds(args.replicates)
        seed_mode = "fresh_os_random"
    else:
        shared_seeds = shared_replicate_seeds(
            args.replicates, seed_start=args.seed_start, seed_step=args.seed_step
        )
        seed_mode = "arithmetic_schedule"
    print(f"Seed mode: {seed_mode}", flush=True)
    print(f"Shared replicate seeds for both methods: {shared_seeds}", flush=True)
    run_id = args.run_id or f"{args.budget}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.smoke:
        subprocess.run([sys.executable, str(BUNDLE / "verify.py"), "--gpu"], cwd=REPO_ROOT, check=True)
        return
    if args.method in {"best-k-of-n", "both", "all"}:
        run_public(
            args.replicates,
            run_id,
            resume=args.resume,
            seed_start=args.seed_start,
            seed_step=args.seed_step,
            seeds=shared_seeds,
        )
    if args.method in {"o3", "both", "all"}:
        config_name = "o3.yaml" if args.budget == "n100_k10" else f"o3_{args.budget}.yaml"
        run_o3(
            args.replicates,
            run_id,
            BUNDLE / config_name,
            args.budget,
            shared_seeds,
        )
    if args.method in {"random-pfode", "all"}:
        config_name = "o3.yaml" if args.budget == "n100_k10" else f"o3_{args.budget}.yaml"
        run_random_pfode(
            args.replicates,
            run_id,
            BUNDLE / config_name,
            args.budget,
            seeds=shared_seeds,
        )


if __name__ == "__main__":
    main()
