"""Internal worker: run a fixed replicate-seed shard on one visible GPU."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import common
import public_runner
import random_pfode_runner
import run as canonical_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=("public-best-k-of-n", "best-k-of-n", "o3", "random-pfode", "matched-stochastic", "paired", "both", "all"),
        required=True,
    )
    parser.add_argument("--budget", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed-list", nargs="+", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != args.gpu_id:
        raise RuntimeError(
            f"Worker GPU isolation mismatch: expected {args.gpu_id}, got {visible!r}"
        )
    common.configure_budget(args.budget)
    timings: dict[str, float] = {}

    def timed(name, function, *positional, **keywords):
        started = time.perf_counter()
        function(*positional, **keywords)
        timings[name] = time.perf_counter() - started

    if args.method == "public-best-k-of-n":
        started = time.perf_counter()
        for seed in args.seed_list:
            public_runner.run_replicate(
                args.run_id,
                seed,
                resume=args.resume,
                batch_size=args.batch_size,
            )
        timings["public_best_k_of_n"] = time.perf_counter() - started

    if args.method in {"o3", "both", "all"}:
        timed(
            "o3",
            canonical_run.run_o3,
            len(args.seed_list),
            args.run_id,
            common.REPO_ROOT / "configs" / "1cll.yaml",
            args.budget,
            args.seed_list,
            resume=args.resume,
            batch_size=args.batch_size or 1,
            worker_only=True,
        )

    if args.method in {"best-k-of-n", "matched-stochastic", "paired", "both", "all"}:
        timed("best_k_of_n", random_pfode_runner.run,
            len(args.seed_list), args.run_id, common.REPO_ROOT / "configs" / "1cll.yaml",
            args.budget, seeds=args.seed_list, resume=args.resume,
            write_run_reports=False, method="best_k_of_n",
        )
    if args.method in {"random-pfode", "paired", "all"}:
        timed(
            "random_pfode",
            random_pfode_runner.run,
            len(args.seed_list),
            args.run_id,
            common.REPO_ROOT / "configs" / "1cll.yaml",
            args.budget,
            seeds=args.seed_list,
            resume=args.resume,
            write_run_reports=False,
        )

    args.marker.parent.mkdir(parents=True, exist_ok=True)
    args.marker.write_text(
        json.dumps(
            {
                "worker_index": args.worker_index,
                "gpu": args.gpu_id,
                "visible_device": "cuda:0",
                "seeds": args.seed_list,
                "seconds_by_method": timings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
