"""Paper-faithful single-sequence public Boltz-2 Best K-of-N runner."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import common
from common import (
    BUNDLE,
    REPO_ROOT,
    convert_cif_to_pdb,
    find_prediction,
    input_yaml_path,
    output_root,
    provenance,
    read_csv,
    score_structure,
    sha256_file,
    write_csv,
    write_json,
)

PUBLIC_PROJECT = Path(
    os.environ.get("BOLTZ_PUBLIC_PROJECT", str(BUNDLE / "public_boltz"))
).expanduser()
PUBLIC_CACHE = Path(
    os.environ.get("BOLTZ_PUBLIC_CACHE", str(BUNDLE / "cache" / "public_boltz"))
).expanduser()
# Use the exact same checked-in Boltz YAML as O3.  Keeping a second ignored
# copy in the bundle made fresh checkouts depend on an untracked file and made
# it possible for the two methods to generate different constructs.
PUBLIC_INPUT = input_yaml_path()


def _public_env() -> dict[str, str]:
    # The caller commonly has the repository .venv active, while uv must use
    # this bundle's isolated public-Boltz environment. Removing VIRTUAL_ENV
    # avoids a misleading mismatch warning without using --active.
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    return env


def _uv() -> str:
    candidates = [
        os.environ.get("BOLTZ_PUBLIC_UV"),
        shutil.which("uv"),
        str(REPO_ROOT / ".venv" / "Scripts" / "uv.exe"),
    ]
    uv = next((candidate for candidate in candidates if candidate and Path(candidate).is_file()), None)
    if uv is None:
        raise RuntimeError("uv is required to run the isolated public Boltz environment")
    return uv


def public_installation_info() -> dict[str, Any]:
    code = (
        "import boltz, importlib.metadata, json, torch; "
        "print(json.dumps({'version': importlib.metadata.version('boltz'), 'module': boltz.__file__, "
        "'cuda_available': bool(torch.cuda.is_available()), "
        "'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
    )
    result = subprocess.run(
        [_uv(), "run", "--project", str(PUBLIC_PROJECT), "python", "-c", code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=_public_env(),
    )
    info = json.loads(result.stdout.strip().splitlines()[-1])
    if info["version"] != "2.2.1":
        raise RuntimeError(f"Expected public boltz==2.2.1, got {info}")
    if "vendor" in info["module"].replace("\\", "/"):
        raise RuntimeError(f"Public baseline resolved to vendored Boltz: {info}")
    if not info["cuda_available"]:
        raise RuntimeError(f"Public Boltz environment cannot see CUDA: {info}")
    return info


def prepare_public_assets() -> None:
    """Populate the shared stock-Boltz cache before parallel workers start."""

    code = (
        "import sys; from pathlib import Path; "
        "from boltz.main import download_boltz2; "
        "cache = Path(sys.argv[1]); cache.mkdir(parents=True, exist_ok=True); "
        "download_boltz2(cache)"
    )
    subprocess.run(
        [
            _uv(),
            "run",
            "--project",
            str(PUBLIC_PROJECT),
            "python",
            "-c",
            code,
            str(PUBLIC_CACHE),
        ],
        cwd=REPO_ROOT,
        check=True,
        env=_public_env(),
    )


@lru_cache(maxsize=1)
def public_checkpoint_info() -> dict[str, Any]:
    """Return provenance for the checkpoint used by the stock CLI."""

    checkpoint = PUBLIC_CACHE / "boltz2_conf.ckpt"
    return {
        "path": str(checkpoint),
        "exists": checkpoint.is_file(),
        "size_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
        "sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
    }


def _run_public_predict(sample_dir: Path, sample_seed: int) -> Path:
    completed = [
        path
        for path in sample_dir.rglob("*.cif")
        if "predictions" in path.parts
    ]
    if completed:
        return find_prediction(sample_dir)

    boltz_out = sample_dir / "boltz"
    if boltz_out.exists() and any(boltz_out.iterdir()) and not list(boltz_out.rglob("*.cif")):
        retry_index = 1
        while True:
            candidate = sample_dir / f"boltz_retry{retry_index}"
            if not candidate.exists() or not any(candidate.iterdir()) or list(candidate.rglob("*.cif")):
                boltz_out = candidate
                break
            retry_index += 1
    boltz_out.mkdir(parents=True, exist_ok=True)
    command = [
        _uv(), "run", "--project", str(PUBLIC_PROJECT),
    ]
    requested_precision = os.environ.get("BOLTZ_PUBLIC_PRECISION")
    if requested_precision is None:
        command += ["boltz", "predict"]
    else:
        command += ["python", str(BUNDLE / "public_entry.py"), "predict"]
    command += [
        str(PUBLIC_INPUT),
        "--out_dir", str(boltz_out),
        "--cache", str(PUBLIC_CACHE),
        "--seed", str(sample_seed),
        "--step_scale", "1.0",
        "--no_kernels",
        "--output_format", "mmcif",
    ]
    log_path = sample_dir / "boltz.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        try:
            subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=_public_env(),
            )
        except subprocess.CalledProcessError:
            # Boltz 2.2.1 can fail while writing its optional confidence
            # sidecar in single-sequence mode after the prediction CIF has
            # already been written. The CIF is the required oracle input;
            # only propagate failures that produced no structure.
            if not list(boltz_out.rglob("*.cif")):
                log.flush()
                failure_log = log_path.read_text(encoding="utf-8")
                if (
                    "torch._C._LinAlgError" not in failure_log
                    or "linalg.svd" not in failure_log
                    or requested_precision in {"32", "32-true"}
                ):
                    raise

                # The official CLI forces FP16. On older GPUs this can make
                # the CUDA SVD in Boltz's rigid-alignment step fail even
                # though the same model is valid in FP32. Retry the identical
                # seed with the installed public package and only precision
                # changed; do not silently change seeds or sampler settings.
                fallback_out = sample_dir / "boltz_float32"
                fallback_out.mkdir(parents=True, exist_ok=True)
                fallback_command = [
                    _uv(), "run", "--project", str(PUBLIC_PROJECT),
                    "python", str(BUNDLE / "public_entry.py"), "predict",
                    str(PUBLIC_INPUT),
                    "--out_dir", str(fallback_out),
                    "--cache", str(PUBLIC_CACHE),
                    "--seed", str(sample_seed),
                    "--step_scale", "1.0",
                    "--no_kernels",
                    "--output_format", "mmcif",
                ]
                log.write(
                    "\n[warning] Stock FP16 Boltz failed in CUDA linalg.svd; "
                    "retrying the same seed in FP32.\n$ "
                    + " ".join(fallback_command)
                    + "\n"
                )
                subprocess.run(
                    fallback_command,
                    cwd=REPO_ROOT,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**_public_env(), "BOLTZ_PUBLIC_PRECISION": "32"},
                )
                return find_prediction(fallback_out)
            log.write("\n[warning] Boltz exited nonzero after writing a prediction CIF; continuing.\n")
    return find_prediction(boltz_out)


def _batch_jobs(replicate_dir: Path, run_seed: int, batch_size: int) -> list[dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return [
        {"start": start, "count": min(batch_size, common.N - start),
         "seed": common.sample_seed(run_seed, start),
         "output_dir": str(replicate_dir / "batches" / f"batch_{start:04d}")}
        for start in range(0, common.N, batch_size)
    ]


def _read_batch(job: dict[str, Any]) -> dict[str, Any] | None:
    marker = Path(job["output_dir"]) / "complete.json"
    if not marker.exists():
        return None
    saved = json.loads(marker.read_text())
    if any(saved.get(key) != value for key, value in job.items()):
        raise ValueError(f"Incompatible batch metadata: {marker}")
    paths = saved.get("structures", [])
    if len(paths) != job["count"] or len(set(paths)) != len(paths):
        raise ValueError(f"Invalid batch structure count: {marker}")
    if not all(Path(path).is_file() and Path(path).stat().st_size for path in paths):
        return None
    return saved


def _run_public_batches(replicate_dir: Path, run_seed: int, batch_size: int) -> dict[int, dict[str, Any]]:
    jobs = _batch_jobs(replicate_dir, run_seed, batch_size)
    pending = [job for job in jobs if _read_batch(job) is None]
    if pending:
        request_path = replicate_dir / "batch_request.json"
        write_json(request_path, {
            "input": str(PUBLIC_INPUT), "cache": str(PUBLIC_CACHE),
            "work_dir": str(replicate_dir / "public_work"), "jobs": pending,
        })
        command = [_uv(), "run", "--project", str(PUBLIC_PROJECT), "python",
                   str(BUNDLE / "public_batch_entry.py"), str(request_path)]
        print(f"[best-k-of-n] persistent inference: {len(pending)} batches, limit={batch_size}", flush=True)
        log_path = replicate_dir / "boltz_batches.log"
        print(f"[best-k-of-n] inference log: {log_path}", flush=True)
        try:
            with log_path.open("a", encoding="utf-8") as log:
                subprocess.run(command, cwd=REPO_ROOT, check=True, stdout=log,
                               stderr=subprocess.STDOUT, env=_public_env())
        except subprocess.CalledProcessError as exc:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-25:])
            raise RuntimeError(f"Public inference failed. Saved log: {log_path}\n{tail}") from exc
    generated = {}
    for job in jobs:
        saved = _read_batch(job)
        if saved is None:
            raise RuntimeError(f"Public batch did not finish: {job['output_dir']}")
        for rank, structure in enumerate(saved["structures"]):
            generated[job["start"] + rank] = {
                "cif": Path(structure), "batch_seed": job["seed"],
                "batch_start": job["start"], "batch_count": job["count"],
                "batch_confidence_rank": rank, "precision": saved["precision"],
            }
    return generated


def _write_replicate_summary(replicate_dir: Path, rows: list[dict[str, Any]], run_seed: int, info: dict[str, Any]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row["tm_score"]), reverse=True)
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
        row["selected"] = rank <= common.K
    write_csv(replicate_dir / "evaluations.csv", ordered)
    returned = ordered[:common.K]
    write_json(replicate_dir / "returned_candidates.json", returned)
    selected = [float(row["tm_score"]) for row in returned]
    precisions = sorted({str(row.get("inference_precision", "unknown")) for row in rows})
    first_sample_seed = common.sample_seed(run_seed, 0)
    input_provenance = provenance(
        "public_boltz",
        msa_path=None,
        msa_sha256=None,
        input_yaml=str(PUBLIC_INPUT),
        input_yaml_sha256=sha256_file(PUBLIC_INPUT),
        use_msa_server=False,
        msa_server_url=None,
    )
    summary = {
        "method": "best_k_of_n",
        "N": common.N,
        "K": common.K,
        "seed": run_seed,
        "first_sample_seed": first_sample_seed,
        "oracle_evaluations": len(ordered),
        "max_of_K": max(selected),
        "mean_of_K": sum(selected) / len(selected),
        "top_k_mean": sum(selected) / len(selected),
        "best_structure": returned[0]["structure"],
        "generator": {
            "package": "boltz",
            "version": info["version"],
            "module": info["module"],
            "sampling": "official_stochastic_boltz2",
            "step_scale": 1.0,
            "gamma_0": 0.8,
            "no_kernels": True,
            "inference_precisions": precisions,
            "checkpoint": public_checkpoint_info(),
        },
        "msa": input_provenance,
    }
    write_json(replicate_dir / "summary.json", summary)
    write_json(replicate_dir / "provenance.json", summary["msa"] | {
        "backend": "public_boltz",
        "generator": summary["generator"],
        "seed": run_seed,
        "first_sample_seed": first_sample_seed,
        "N": common.N,
        "K": common.K,
    })
    return summary


def run_replicate(run_id: str, run_seed: int, resume: bool = False, batch_size: int | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    if not PUBLIC_INPUT.is_file():
        raise FileNotFoundError(f"Missing public Boltz single-sequence input: {PUBLIC_INPUT}")
    info = public_installation_info()
    replicate_dir = output_root("best_k_of_n", run_id) / f"replicate_{run_seed:03d}"
    evaluations_path = replicate_dir / "evaluations.csv"
    if replicate_dir.exists() and evaluations_path.exists() and not resume:
        raise RuntimeError(f"{replicate_dir} already exists; pass --resume or choose another --run-id")
    replicate_dir.mkdir(parents=True, exist_ok=True)
    rows_by_index = {int(row["sample_index"]): row for row in read_csv(evaluations_path)}
    settings = {"batch_size": batch_size, "step_scale": 1.0,
                "seed_policy": "per_sample" if batch_size is None else "per_batch_first_sample_seed"}
    settings_path = replicate_dir / "inference_settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError("Baseline settings changed (including step scale); use a fresh run ID")
    elif rows_by_index or any(replicate_dir.rglob("*.cif")):
        raise ValueError("Use a fresh run ID for persistent baseline inference")
    write_json(settings_path, settings)
    generated = _run_public_batches(replicate_dir, run_seed, batch_size) if batch_size is not None else None
    rows: list[dict[str, Any]] = []
    scoring_seconds = 0.0
    for index in range(common.N):
        sample_seed_for_index = common.sample_seed(run_seed, index)
        sample_dir = replicate_dir / "samples" / f"sample_{index:04d}"
        existing = rows_by_index.get(index)
        structure = Path(existing["structure"]) if existing else sample_dir / f"sample_{index:04d}.pdb"
        if existing and structure.exists():
            existing.setdefault("inference_precision", "not_recorded")
            rows.append(existing)
            continue
        cif_path = generated[index]["cif"] if generated is not None else _run_public_predict(sample_dir, sample_seed_for_index)
        structure.parent.mkdir(parents=True, exist_ok=True)
        tick = time.perf_counter()
        convert_cif_to_pdb(cif_path, structure)
        score = score_structure(structure)
        scoring_seconds += time.perf_counter() - tick
        if not math.isfinite(score):
            raise ValueError(f"Non-finite TM-score for {structure}")
        requested_precision = os.environ.get("BOLTZ_PUBLIC_PRECISION")
        actual_precision = (
            "32"
            if "boltz_float32" in cif_path.parts
            else requested_precision or "16-mixed"
        )
        row = {
            "sample_index": index,
            "sample_seed": sample_seed_for_index,
            "run_seed": run_seed,
            "tm_score": score,
            "structure": str(structure),
            "inference_precision": actual_precision,
        }
        if generated is not None:
            item = generated[index]
            row["sample_seed"] = ""  # No independent per-sample RNG stream in a batch.
            row.update({key: item[key] for key in ("batch_seed", "batch_start", "batch_count", "batch_confidence_rank")})
            row["inference_precision"] = item["precision"]
        rows.append(row)
        write_csv(evaluations_path, rows)
    summary = _write_replicate_summary(replicate_dir, rows, run_seed, info)
    summary["inference"] = settings
    summary["timings_this_session"] = {
        "wall_seconds": time.perf_counter() - started, "conversion_scoring_seconds": scoring_seconds,
    }
    write_json(replicate_dir / "summary.json", summary)
    provenance_path = replicate_dir / "provenance.json"
    write_json(provenance_path, json.loads(provenance_path.read_text()) | {"inference": settings})
    return summary


def run(
    replicates: int,
    run_id: str,
    resume: bool = False,
    *,
    seed_start: int = common.DEFAULT_REPLICATE_SEED_START,
    seed_step: int = common.DEFAULT_REPLICATE_SEED_STEP,
    seeds: list[int] | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    if replicates not in common.SUPPORTED_REPLICATES:
        choices = ", ".join(str(value) for value in common.SUPPORTED_REPLICATES)
        raise ValueError(f"replicates must be one of: {choices}")
    run_seeds = common.resolve_replicate_seeds(
        replicates, seeds=seeds, seed_start=seed_start, seed_step=seed_step
    )
    metadata_seed_start = None if seeds is not None else seed_start
    metadata_seed_step = None if seeds is not None else seed_step
    print(f"[best-k-of-n] shared replicate seeds: {run_seeds}", flush=True)
    summaries = [run_replicate(run_id, seed, resume=resume, batch_size=batch_size) for seed in run_seeds]
    return finalize_run(
        replicates=replicates,
        run_id=run_id,
        run_seeds=run_seeds,
        summaries=summaries,
        seed_start=metadata_seed_start,
        seed_step=metadata_seed_step,
        batch_size=batch_size,
    )


def finalize_run(
    *,
    replicates: int,
    run_id: str,
    run_seeds: list[int],
    summaries: list[dict[str, Any]] | None = None,
    seed_start: int | None = None,
    seed_step: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Write run-level reports after sequential or parallel replicates finish."""

    if summaries is None:
        summaries = []
        for seed in run_seeds:
            path = output_root("best_k_of_n", run_id) / f"replicate_{seed:03d}" / "summary.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed baseline summary: {path}")
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    if len(summaries) != replicates:
        raise ValueError(
            f"Expected {replicates} baseline summaries, got {len(summaries)}"
        )
    run_dir = output_root("best_k_of_n", run_id)
    aggregate_rows = []
    for summary in summaries:
        aggregate_rows.append({
            "run_seed": summary["seed"],
            "N": summary["N"],
            "K": summary["K"],
            "mean_of_K": summary["mean_of_K"],
            "max_of_K": summary["max_of_K"],
        })
    write_csv(run_dir / "aggregate.csv", aggregate_rows)
    write_json(run_dir / "provenance.json", provenance(
        "public_boltz",
        msa_path=None,
        msa_sha256=None,
        input_yaml=str(PUBLIC_INPUT),
        input_yaml_sha256=sha256_file(PUBLIC_INPUT),
        use_msa_server=False,
        msa_server_url=None,
        budget=common.ACTIVE_BUDGET,
        replicates=replicates,
        seed_mode="explicit_list" if seed_start is None else "arithmetic_schedule",
        seed_start=seed_start,
        seed_step=seed_step,
        seeds=run_seeds,
        sample_seed_function=("common.sample_seed(run_seed, sample_index)" if batch_size is None
                              else "common.sample_seed(run_seed, batch_start_index)"),
        inference_batch_size=batch_size,
        step_scale=1.0,
        execution_mode="per_sample_cli" if batch_size is None else "persistent_batches",
        checkpoint=public_checkpoint_info(),
    ))
    return {"method": "best_k_of_n", "replicates": summaries, "aggregate": aggregate_rows}
