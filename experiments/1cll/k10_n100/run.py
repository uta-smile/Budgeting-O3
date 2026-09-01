"""Canonical command for the 1CLL K=10, N=100 comparison."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from common import BUNDLE, REPO_ROOT, configure_budget, validate_frozen_msa
from public_runner import run as run_public


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("best-k-of-n", "o3", "both"), default="both")
    parser.add_argument("--budget", choices=("n20_k2", "n50_k5", "n100_k10"), default="n100_k10")
    parser.add_argument("--replicates", type=int, choices=(1, 5), default=5)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run backend and sampler verification only")
    return parser.parse_args()


def run_o3(replicates: int, run_id: str, config_path: Path, budget: str) -> None:
    uv = os.environ.get("BOLTZ_PUBLIC_UV") or shutil.which("uv") or "uv"
    command = [
        uv, "run", "--project", str(REPO_ROOT), "python", "-m", "o3_boltz.cli",
        "--config", str(config_path),
        "--replicates", str(replicates),
        "--run-id", run_id,
        "--only", budget,
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    configure_budget(args.budget)
    validate_frozen_msa()
    run_id = args.run_id or f"{args.budget}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.smoke:
        subprocess.run([sys.executable, str(BUNDLE / "verify.py"), "--gpu"], cwd=REPO_ROOT, check=True)
        return
    if args.method in {"best-k-of-n", "both"}:
        run_public(args.replicates, run_id, resume=args.resume)
    if args.method in {"o3", "both"}:
        config_name = "o3.yaml" if args.budget == "n100_k10" else f"o3_{args.budget}.yaml"
        run_o3(args.replicates, run_id, BUNDLE / config_name, args.budget)


if __name__ == "__main__":
    main()
