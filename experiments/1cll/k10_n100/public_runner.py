"""Paper-faithful single-sequence public Boltz-2 Best K-of-N runner."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import common
from common import (
    BUNDLE,
    REPO_ROOT,
    convert_cif_to_pdb,
    find_prediction,
    output_root,
    provenance,
    read_csv,
    score_structure,
    sha256_file,
    write_csv,
    write_json,
)

PUBLIC_PROJECT = BUNDLE / "public_boltz"
PUBLIC_CACHE = BUNDLE / "cache" / "public_boltz"
PUBLIC_INPUT = BUNDLE / "inputs" / "1cll_single_sequence.yaml"


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


def _run_public_predict(sample_dir: Path, sample_seed: int) -> Path:
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
        _uv(), "run", "--project", str(PUBLIC_PROJECT), "boltz", "predict",
        str(PUBLIC_INPUT),
        "--out_dir", str(boltz_out),
        "--cache", str(PUBLIC_CACHE),
        "--seed", str(sample_seed),
        "--no_kernels",
        "--output_format", "mmcif",
    ]
    log_path = sample_dir / "boltz.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
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
                raise
            log.write("\n[warning] Boltz exited nonzero after writing a prediction CIF; continuing.\n")
    return find_prediction(boltz_out)


def _write_replicate_summary(replicate_dir: Path, rows: list[dict[str, Any]], run_seed: int, info: dict[str, Any]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row["tm_score"]), reverse=True)
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
        row["selected"] = rank <= common.K
    write_csv(replicate_dir / "evaluations.csv", ordered)
    returned = ordered[:common.K]
    write_json(replicate_dir / "returned_candidates.json", returned)
    selected = [float(row["tm_score"]) for row in returned]
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
        "generator": {"package": "boltz", "version": info["version"], "module": info["module"], "sampling": "official_stochastic_boltz2"},
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


def run_replicate(run_id: str, run_seed: int, resume: bool = False) -> dict[str, Any]:
    if not PUBLIC_INPUT.is_file():
        raise FileNotFoundError(f"Missing public Boltz single-sequence input: {PUBLIC_INPUT}")
    info = public_installation_info()
    replicate_dir = output_root("best_k_of_n", run_id) / f"replicate_{run_seed:03d}"
    evaluations_path = replicate_dir / "evaluations.csv"
    if replicate_dir.exists() and evaluations_path.exists() and not resume:
        raise RuntimeError(f"{replicate_dir} already exists; pass --resume or choose another --run-id")
    replicate_dir.mkdir(parents=True, exist_ok=True)
    rows_by_index = {int(row["sample_index"]): row for row in read_csv(evaluations_path)}
    rows: list[dict[str, Any]] = []
    for index in range(common.N):
        sample_seed_for_index = common.sample_seed(run_seed, index)
        sample_dir = replicate_dir / "samples" / f"sample_{index:04d}"
        existing = rows_by_index.get(index)
        structure = Path(existing["structure"]) if existing else sample_dir / f"sample_{index:04d}.pdb"
        if existing and structure.exists():
            rows.append(existing)
            continue
        boltz_root = sample_dir / "boltz"
        cif_path = find_prediction(boltz_root) if list(boltz_root.rglob("*.cif")) else _run_public_predict(sample_dir, sample_seed_for_index)
        convert_cif_to_pdb(cif_path, structure)
        score = score_structure(structure)
        row = {
            "sample_index": index,
            "sample_seed": sample_seed_for_index,
            "run_seed": run_seed,
            "tm_score": score,
            "structure": str(structure),
        }
        rows.append(row)
        write_csv(evaluations_path, rows)
    return _write_replicate_summary(replicate_dir, rows, run_seed, info)


def run(
    replicates: int,
    run_id: str,
    resume: bool = False,
    *,
    seed_start: int = common.DEFAULT_REPLICATE_SEED_START,
    seed_step: int = common.DEFAULT_REPLICATE_SEED_STEP,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    if replicates not in {1, 3, 5}:
        raise ValueError("replicates must be 1, 3, or 5")
    run_seeds = common.resolve_replicate_seeds(
        replicates, seeds=seeds, seed_start=seed_start, seed_step=seed_step
    )
    metadata_seed_start = None if seeds is not None else seed_start
    metadata_seed_step = None if seeds is not None else seed_step
    print(f"[best-k-of-n] shared replicate seeds: {run_seeds}", flush=True)
    summaries = [run_replicate(run_id, seed, resume=resume) for seed in run_seeds]
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
        seed_mode="explicit_list" if seeds is not None else "arithmetic_schedule",
        seed_start=metadata_seed_start,
        seed_step=metadata_seed_step,
        seeds=run_seeds,
        sample_seed_function="common.sample_seed(run_seed, sample_index)",
    ))
    return {"method": "best_k_of_n", "replicates": summaries, "aggregate": aggregate_rows}
