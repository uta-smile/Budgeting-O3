"""Static, source-audit, fixture, and optional GPU verification for the bundle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from common import BUNDLE, REPO_ROOT, SEQUENCE, input_yaml_path, msa_path, sha256_file, validate_frozen_msa

ALLOWED_VENDOR_CHANGES = json.loads(
    (BUNDLE / "vendor_patch_allowlist.json").read_text(encoding="utf-8")
)


def check_static() -> None:
    input_text = input_yaml_path().read_text(encoding="utf-8")
    if SEQUENCE not in input_text or len(SEQUENCE) != 144:
        raise AssertionError("1CLL input sequence is not the expected 144-residue sequence")
    validate_frozen_msa()
    public_project = (BUNDLE / "public_boltz" / "pyproject.toml").read_text(encoding="utf-8")
    if '"boltz==2.2.1"' not in public_project or "vendor/boltz" in public_project:
        raise AssertionError("Public environment is not pinned to public boltz==2.2.1")
    adapter_text = (REPO_ROOT / "adapters" / "boltz2_pfode.py").read_text(encoding="utf-8")
    for required in ("initial_atom_coords", "deterministic", "gamma_0", "Boltz2.load_from_checkpoint"):
        if required not in adapter_text:
            raise AssertionError(f"O3 adapter is missing required custom behavior: {required}")


def check_fixture() -> dict[str, float]:
    import csv

    fixture = BUNDLE / "inputs" / "notebook_1CLL_TMscore_results.csv"
    with fixture.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100:
        raise AssertionError(f"Expected 100 notebook fixture scores, got {len(rows)}")
    scores = sorted((float(row["TM_score"]) for row in rows), reverse=True)
    mean_top10 = sum(scores[:10]) / 10
    max_top10 = scores[0]
    expected = {"mean_of_K": 0.6788870863920918, "max_of_K": 0.7958910827539285}
    if abs(mean_top10 - expected["mean_of_K"]) > 1e-12 or abs(max_top10 - expected["max_of_K"]) > 1e-12:
        raise AssertionError(f"Notebook fixture mismatch: {mean_top10=}, {max_top10=}")
    return {"mean_of_K": mean_top10, "max_of_K": max_top10}


def public_info() -> dict[str, Any]:
    from public_runner import public_installation_info

    return public_installation_info()


def public_smoke() -> dict[str, object]:
    from public_runner import _run_public_predict
    from common import convert_cif_to_pdb, score_structure

    info = public_info()
    with tempfile.TemporaryDirectory(prefix="k10_n100_public_smoke_") as temp:
        sample_dir = Path(temp) / "sample_0000"
        cif_path = _run_public_predict(sample_dir, 0)
        pdb_path = convert_cif_to_pdb(cif_path, sample_dir / "sample_0000.pdb")
        score = score_structure(pdb_path)
    return {**info, "sample_seed": 0, "tm_score": score}


def audit_vendor() -> dict[str, list[str]]:
    try:
        info = public_info()
        public_root = Path(info["module"]).resolve().parent
    except (RuntimeError, subprocess.CalledProcessError):
        cache_roots = []
        archive_root = REPO_ROOT / ".uv-cache" / "archive-v0"
        if archive_root.is_dir():
            for archive in archive_root.iterdir():
                candidate = archive / "Lib" / "site-packages" / "boltz"
                if candidate.is_dir() and (candidate.parent / "boltz-2.2.1.dist-info").is_dir():
                    cache_roots.append(candidate)
        cache_roots.sort()
        if not cache_roots:
            raise
        public_root = cache_roots[0]
    vendor_root = REPO_ROOT / "vendor" / "boltz" / "src" / "boltz"
    changed: list[str] = []
    unexpected: list[str] = []
    for vendor_file in sorted(vendor_root.rglob("*.py")):
        relative = vendor_file.relative_to(vendor_root).as_posix()
        public_file = public_root / relative
        if not public_file.exists():
            unexpected.append(relative)
            continue
        vendor_text = vendor_file.read_text(encoding="utf-8").replace("\r\n", "\n")
        public_text = public_file.read_text(encoding="utf-8").replace("\r\n", "\n")
        if vendor_text != public_text:
            changed.append(relative)
            if relative not in ALLOWED_VENDOR_CHANGES:
                unexpected.append(relative)
    if unexpected:
        raise AssertionError(f"Unexpected vendored changes: {unexpected}")
    return {"changed": changed, "allowed": sorted(ALLOWED_VENDOR_CHANGES)}


def gpu_smoke() -> dict[str, object]:
    from adapters.boltz2_pfode import create
    import numpy as np
    import yaml

    config = yaml.safe_load((BUNDLE / "o3.yaml").read_text(encoding="utf-8"))
    # Boltz resolves the MSA path in the input YAML relative to cwd.
    os.chdir(REPO_ROOT)
    config["project_root"] = str(REPO_ROOT)
    config["target"]["reference_pdb"] = "data/1CLL.pdb"
    adapter = create(config)
    if type(adapter).__name__ != "Boltz2PFODEAdapter" or type(adapter).__module__ != "adapters.boltz2_pfode":
        raise AssertionError("O3 smoke did not use the custom adapter")
    if adapter.latent_dim != 3552 or adapter.atom_slots != 1184:
        raise AssertionError(f"Unexpected O3 latent shape: {adapter.latent_dim=}, {adapter.atom_slots=}")
    latent = np.random.default_rng(123).normal(size=adapter.latent_dim)

    def coordinates(path: Path) -> np.ndarray:
        values = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith(("ATOM  ", "HETATM")):
                values.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        return np.asarray(values, dtype=np.float64)

    with tempfile.TemporaryDirectory(prefix="k10_n100_smoke_") as temp:
        root = Path(temp)
        first = adapter.generate(latent, root / "first.pdb", config, {"deterministic": True})
        second = adapter.generate(latent, root / "second.pdb", config, {"deterministic": True})
        other = adapter.generate(-latent, root / "other.pdb", config, {"deterministic": True})
        first_coords = coordinates(first)
        second_coords = coordinates(second)
        other_coords = coordinates(other)
        if first_coords.shape != (adapter.atom_count, 3):
            raise AssertionError(f"Unexpected O3 smoke atom count: {first_coords.shape}")
        if not np.allclose(first_coords, second_coords, rtol=0.0, atol=1e-5):
            raise AssertionError("Repeated O3 PF-ODE sampling was not numerically stable")
        if np.allclose(first_coords, other_coords, rtol=0.0, atol=1e-5):
            raise AssertionError("O3 ignored the supplied latent")
    return {"custom_adapter": type(adapter).__name__, "latent_dim": adapter.latent_dim, "atom_slots": adapter.atom_slots}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-vendor", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    check_static()
    fixture = check_fixture()
    result: dict[str, object] = {"fixture": fixture, "msa_sha256": sha256_file(msa_path())}
    if args.audit_vendor:
        result["vendor_audit"] = audit_vendor()
    if args.gpu:
        result["public"] = public_smoke()
        result["gpu_smoke"] = gpu_smoke()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
