"""Shared paths, scoring, serialization, and provenance for the 1CLL bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

BUNDLE = Path(__file__).resolve().parent
REPO_ROOT = BUNDLE.parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SEQUENCE = (
    "LTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGNGTIDFPEFLTMMARKMK"
    "DTDSEEEIREAFRVFDKDGNGYISAAELRHVMTNLGEKLTDEEVDEMIREADIDGDGQVNYEEFVQMMTA"
)
_BUDGETS = {
    "n20_k2": {"N": 20, "K": 2, "folder": "k2_n20", "input": "1cll_n20.yaml"},
    "n50_k5": {"N": 50, "K": 5, "folder": "k5_n50", "input": "1cll_n50.yaml"},
    "n100_k10": {"N": 100, "K": 10, "folder": "k10_n100", "input": "1cll.yaml"},
}
ACTIVE_BUDGET = "n100_k10"
N = _BUDGETS[ACTIVE_BUDGET]["N"]
K = _BUDGETS[ACTIVE_BUDGET]["K"]

# Shared by both methods in the canonical comparison. The nontrivial,
# prime-spaced schedule is reproducible and avoids the first few integer
# seeds being a special case.
DEFAULT_REPLICATE_SEED_START = 20250117
DEFAULT_REPLICATE_SEED_STEP = 1009


def configure_budget(name: str) -> None:
    global ACTIVE_BUDGET, N, K
    if name not in _BUDGETS:
        raise ValueError(f"Unknown budget {name!r}; choose from {sorted(_BUDGETS)}")
    ACTIVE_BUDGET = name
    N = int(_BUDGETS[name]["N"])
    K = int(_BUDGETS[name]["K"])


def shared_replicate_seeds(
    replicates: int,
    *,
    seed_start: int = DEFAULT_REPLICATE_SEED_START,
    seed_step: int = DEFAULT_REPLICATE_SEED_STEP,
) -> list[int]:
    """Return the reproducible replicate seeds shared by both methods."""
    if replicates < 1:
        raise ValueError("replicates must be at least 1")
    if seed_step == 0:
        raise ValueError("seed_step must not be 0")
    return [int(seed_start) + index * int(seed_step) for index in range(replicates)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_msa() -> str:
    path = msa_path()
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen MSA: {path}")
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().strip()
    if header != "key,sequence":
        raise ValueError(f"Unexpected frozen MSA format in {path}: {header!r}")
    expected = path.with_name(path.name + ".sha256").read_text(encoding="utf-8").split()[0]
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Frozen MSA checksum mismatch: expected {expected}, got {actual}")
    return actual


def sample_seed(run_seed: int, sample_index: int) -> int:
    return run_seed * N + sample_index


def msa_path() -> Path:
    return BUNDLE / "inputs" / "1CLL_0.csv"


def input_yaml_path() -> Path:
    return BUNDLE / "inputs" / str(_BUDGETS[ACTIVE_BUDGET]["input"])


def reference_path() -> Path:
    return REPO_ROOT / "data" / "1CLL.pdb"


def output_root(method: str, run_id: str) -> Path:
    folder = str(_BUDGETS[ACTIVE_BUDGET]["folder"])
    return REPO_ROOT / "outputs" / "1cll" / folder / method / "runs" / run_id


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = [dict(row) for row in rows]
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def score_structure(path: Path) -> float:
    from o3_boltz.tmscore import TMScoreOracle

    oracle = TMScoreOracle(reference_path(), "A")
    return float(oracle.score(path, "A"))


def convert_cif_to_pdb(cif_path: Path, pdb_path: Path) -> Path:
    import gemmi

    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    gemmi.read_structure(str(cif_path)).write_pdb(str(pdb_path))
    return pdb_path


def find_prediction(root: Path) -> Path:
    predictions = sorted(
        path for path in root.rglob("*.cif") if "predictions" in path.parts
    )
    if not predictions:
        raise FileNotFoundError(f"No prediction CIF found below {root}")
    return predictions[0]


def provenance(backend: str, **extra: Any) -> dict[str, Any]:
    return {
        "backend": backend,
        "sequence_length": len(SEQUENCE),
        "msa_path": str(msa_path()),
        "msa_sha256": sha256_file(msa_path()),
        "reference_pdb": str(reference_path()),
        "reference_chain": "A",
        **extra,
    }
