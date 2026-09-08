"""Replicate-level, one-process-per-GPU scheduling for the 1CLL benchmark."""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import common
import public_runner
import random_pfode_runner


def resolve_gpu_ids(spec: str) -> list[str]:
    """Resolve ``0,1,...`` or ``auto`` without initializing CUDA in the parent."""

    value = spec.strip()
    if not value:
        raise ValueError("--gpus must be 'auto' or a comma-separated GPU index list")
    if value.lower() == "auto":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible and visible.strip() not in {"", "-1"}:
            value = visible
        else:
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                raise RuntimeError(
                    "Could not discover GPUs with nvidia-smi; pass --gpus 0,1,..."
                ) from exc
            value = ",".join(line.strip() for line in result.stdout.splitlines())

    gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not gpu_ids or any(not item.isdigit() for item in gpu_ids):
        raise ValueError("--gpus accepts numeric GPU indices, for example --gpus 0,1,2,3")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"--gpus contains duplicate indices: {gpu_ids}")
    if os.name == "nt" and len(gpu_ids) > 1:
        raise RuntimeError(
            "Parallel GPU workers are enabled only on Linux; use one GPU on Windows"
        )
    return gpu_ids


def assign_seeds(gpu_ids: list[str], seeds: list[int]) -> list[dict[str, Any]]:
    """Assign every replicate seed to exactly one GPU in stable round-robin order."""

    if not gpu_ids:
        raise ValueError("At least one GPU is required")
    active_gpu_ids = gpu_ids[: len(seeds)]
    return [
        {
            "worker_index": worker_index,
            "gpu": gpu,
            "seeds": seeds[worker_index :: len(active_gpu_ids)],
        }
        for worker_index, gpu in enumerate(active_gpu_ids)
    ]


def _tail(path: Path, lines: int = 40) -> str:
    if not path.is_file():
        return "(worker log was not created)"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def _terminate_process_tree(process: subprocess.Popen) -> None:
    """Stop a worker and its nested uv/Boltz child on Linux."""

    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass


def _wait_after_termination(process: subprocess.Popen) -> None:
    """Escalate to a hard stop if a failed worker's process group lingers."""

    try:
        process.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=15)


def launch_workers(
    *,
    method: str,
    budget: str,
    run_id: str,
    seeds: list[int],
    gpu_ids: list[str],
    batch_size: int | None,
    resume: bool,
) -> dict[str, Any]:
    """Launch isolated workers and fail the whole session if any worker fails."""

    started = time.perf_counter()
    assignments = assign_seeds(gpu_ids, seeds)
    cache_prepare_seconds = 0.0
    if method in {"best-k-of-n", "both", "all"}:
        print("Preparing the shared public-Boltz cache once before workers start...", flush=True)
        cache_started = time.perf_counter()
        public_runner.prepare_public_assets()
        cache_prepare_seconds = time.perf_counter() - cache_started
    session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    session_dir = common.comparison_run_root(run_id) / "workers" / session_name
    session_dir.mkdir(parents=True, exist_ok=False)
    worker_script = common.BUNDLE / "multi_gpu_worker.py"
    processes: list[dict[str, Any]] = []

    print("Multi-GPU replicate assignment:", flush=True)
    for assignment in assignments:
        gpu = assignment["gpu"]
        worker_index = assignment["worker_index"]
        worker_seeds = assignment["seeds"]
        log_path = session_dir / f"worker_{worker_index}_gpu_{gpu}.log"
        marker_path = session_dir / f"worker_{worker_index}_gpu_{gpu}.json"
        command = [
            sys.executable,
            str(worker_script),
            "--method",
            method,
            "--budget",
            budget,
            "--run-id",
            run_id,
            "--worker-index",
            str(worker_index),
            "--gpu-id",
            gpu,
            "--marker",
            str(marker_path),
            "--seed-list",
            *(str(seed) for seed in worker_seeds),
        ]
        if batch_size is not None:
            command += ["--batch-size", str(batch_size)]
        if resume:
            command.append("--resume")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"
        log_handle = log_path.open("a", encoding="utf-8")
        log_handle.write(
            f"\n=== {datetime.now().isoformat()} GPU {gpu} seeds {worker_seeds} ===\n"
        )
        log_handle.flush()
        print(f"  GPU {gpu}: seeds={worker_seeds} | log={log_path}", flush=True)
        process = subprocess.Popen(
            command,
            cwd=common.REPO_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
        )
        processes.append(
            {
                "process": process,
                "log_handle": log_handle,
                "log_path": log_path,
                "marker_path": marker_path,
                "assignment": assignment,
            }
        )

    failure: dict[str, Any] | None = None
    try:
        remaining = set(range(len(processes)))
        while remaining:
            for index in tuple(remaining):
                returncode = processes[index]["process"].poll()
                if returncode is None:
                    continue
                remaining.remove(index)
                if returncode != 0 and failure is None:
                    failure = processes[index] | {"returncode": returncode}
            if failure is not None:
                for index in remaining:
                    _terminate_process_tree(processes[index]["process"])
                for index in remaining:
                    _wait_after_termination(processes[index]["process"])
                break
            if remaining:
                time.sleep(0.5)
    except KeyboardInterrupt:
        for item in processes:
            _terminate_process_tree(item["process"])
        for item in processes:
            _wait_after_termination(item["process"])
        raise
    finally:
        for item in processes:
            item["log_handle"].close()

    if failure is not None:
        assignment = failure["assignment"]
        raise RuntimeError(
            f"GPU worker {assignment['worker_index']} on GPU {assignment['gpu']} "
            f"failed with exit code {failure['returncode']}. Log: {failure['log_path']}\n"
            f"{_tail(failure['log_path'])}"
        )

    worker_results = []
    for item in processes:
        marker_path = item["marker_path"]
        if not marker_path.is_file():
            raise RuntimeError(f"GPU worker exited without completion marker: {marker_path}")
        worker_results.append(json.loads(marker_path.read_text(encoding="utf-8")))

    report = {
        "execution_mode": "replicate_parallel",
        "method": method,
        "budget": budget,
        "run_id": run_id,
        "batch_size": batch_size,
        "resume": resume,
        "requested_gpus": gpu_ids,
        "worker_count": len(assignments),
        "assignments": assignments,
        "workers": worker_results,
        "public_cache_prepare_seconds": cache_prepare_seconds,
        "wall_seconds": time.perf_counter() - started,
        "session_dir": str(session_dir),
    }
    common.write_json(session_dir / "schedule.json", report)
    common.write_json(
        common.comparison_run_root(run_id) / "gpu_schedule_this_session.json", report
    )
    return report


def _load_summaries(method: str, run_id: str, seeds: list[int]) -> list[dict[str, Any]]:
    root = common.output_root(method, run_id)
    summaries = []
    for seed in seeds:
        directory = (
            f"replicate_{seed:03d}" if method == "best_k_of_n" else f"seed_{seed:04d}"
        )
        path = root / directory / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed {method} summary: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if int(summary.get("seed", -1)) != seed:
            raise ValueError(f"Summary seed mismatch in {path}")
        summaries.append(summary)
    return summaries


def _write_o3_reports(
    *,
    run_id: str,
    budget: str,
    seeds: list[int],
    summaries: list[dict[str, Any]],
    parallel_execution: dict[str, Any],
) -> None:
    root = common.output_root("o3", run_id)
    with (root / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    fields = ["seed", "N", "K", "mean_of_K", "max_of_K"]
    common.write_csv(
        root / "aggregate.csv",
        [{field: summary.get(field) for field in fields} for summary in summaries],
    )
    first = summaries[0]
    common.write_json(
        root / "run_metadata.json",
        {
            "method": "o3",
            "method_directory": "o3",
            "target": "1cll",
            "budget": budget,
            "run_id": run_id,
            "config_path": str(common.REPO_ROOT / "configs" / "1cll.yaml"),
            "replicates": len(seeds),
            "seed_start": None,
            "seed_step": None,
            "seed_mode": "explicit_list",
            "seeds": seeds,
            "provenance": {
                key: first.get(key) for key in ("generator", "msa_cache", "runtime")
            },
            "parallel_execution": parallel_execution,
        },
    )


def finalize_reports(
    *,
    method: str,
    budget: str,
    run_id: str,
    seeds: list[int],
    batch_size: int | None,
    parallel_execution: dict[str, Any],
) -> None:
    """Create shared reports only after every worker has completed."""

    if method in {"best-k-of-n", "both", "all"}:
        summaries = _load_summaries("best_k_of_n", run_id, seeds)
        public_runner.finalize_run(
            replicates=len(seeds),
            run_id=run_id,
            run_seeds=seeds,
            summaries=summaries,
            batch_size=batch_size,
        )
        provenance_path = common.output_root("best_k_of_n", run_id) / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        common.write_json(
            provenance_path,
            provenance | {"parallel_execution": parallel_execution},
        )
    if method in {"o3", "both", "all"}:
        summaries = _load_summaries("o3", run_id, seeds)
        _write_o3_reports(
            run_id=run_id,
            budget=budget,
            seeds=seeds,
            summaries=summaries,
            parallel_execution=parallel_execution,
        )
    if method in {"random-pfode", "all"}:
        summaries = _load_summaries("random_pfode", run_id, seeds)
        random_pfode_runner.finalize_run(
            replicates=len(seeds),
            run_id=run_id,
            budget_name=budget,
            run_seeds=seeds,
            summaries=summaries,
        )
        metadata_path = common.output_root("random_pfode", run_id) / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        common.write_json(
            metadata_path,
            metadata | {"parallel_execution": parallel_execution},
        )
