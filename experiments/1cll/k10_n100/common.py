"""Shared paths, scoring, serialization, and provenance for the 1CLL bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import random
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
UINT32_SEED_MODULUS = 2**32 - 1


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


def random_replicate_seeds(replicates: int) -> list[int]:
    """Return unique fresh 31-bit seeds from the operating system RNG."""
    if replicates < 1:
        raise ValueError("replicates must be at least 1")
    return random.SystemRandom().sample(range(1, 2**31), replicates)


def resolve_replicate_seeds(
    replicates: int,
    *,
    seeds: Iterable[int] | None = None,
    seed_start: int = DEFAULT_REPLICATE_SEED_START,
    seed_step: int = DEFAULT_REPLICATE_SEED_STEP,
) -> list[int]:
    """Validate an explicit seed list or use the shared arithmetic schedule."""
    if seeds is not None:
        resolved = [int(seed) for seed in seeds]
        if len(resolved) != replicates:
            raise ValueError(
                f"Expected {replicates} explicit replicate seeds, got {len(resolved)}"
            )
        if len(set(resolved)) != len(resolved):
            raise ValueError("replicate seeds must be unique")
        return resolved
    return shared_replicate_seeds(
        replicates, seed_start=seed_start, seed_step=seed_step
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_seed(run_seed: int, sample_index: int) -> int:
    """Derive a deterministic public-Boltz seed in NumPy's valid range."""
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    # Public Boltz passes this value through NumPy/Lightning, which accepts
    # only unsigned 32-bit seeds. The multiplicative mix keeps samples from
    # different replicate seeds separated without overflowing that range.
    return (int(run_seed) * 1_000_003 + int(sample_index) + 1) % UINT32_SEED_MODULUS


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
        "conditioning": "single_sequence",
        "msa_path": None,
        "msa_sha256": None,
        "use_msa_server": False,
        "reference_pdb": str(reference_path()),
        "reference_chain": "A",
        **extra,
    }
