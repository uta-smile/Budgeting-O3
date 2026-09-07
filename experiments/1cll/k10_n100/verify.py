"""Static, source-audit, fixture, and optional GPU verification for the bundle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from common import BUNDLE, REPO_ROOT, SEQUENCE, input_yaml_path

ALLOWED_VENDOR_CHANGES = json.loads(
    (BUNDLE / "vendor_patch_allowlist.json").read_text(encoding="utf-8")
)


def check_static() -> None:
    input_text = input_yaml_path().read_text(encoding="utf-8")
    if SEQUENCE not in input_text or len(SEQUENCE) != 144:
        raise AssertionError("1CLL input sequence is not the expected 144-residue sequence")
    input_config = yaml.safe_load(input_text)
    protein = input_config["sequences"][0]["protein"]
    if protein.get("id") != "A" or protein.get("sequence") != SEQUENCE:
        raise AssertionError("Canonical Boltz YAML must contain the exact 1CLL chain A sequence")
    single_input = input_yaml_path().read_text(encoding="utf-8")
    if "msa: empty" not in single_input:
        raise AssertionError("data/1cll_boltz_input.yaml must explicitly select single-sequence mode")
    if "msa:" in single_input.replace("msa: empty", ""):
        raise AssertionError("data/1cll_boltz_input.yaml supplies an MSA despite single-sequence setup")
    public_project = (BUNDLE / "public_boltz" / "pyproject.toml").read_text(encoding="utf-8")
    if '"boltz==2.2.1"' not in public_project or "vendor/boltz" in public_project:
        raise AssertionError("Public environment is not pinned to public boltz==2.2.1")
    public_runner = (BUNDLE / "public_runner.py").read_text(encoding="utf-8")
    if "PUBLIC_INPUT = input_yaml_path()" not in public_runner:
        raise AssertionError("Public baseline is not using the canonical Boltz input YAML")
    if '"--use_msa_server"' in public_runner:
        raise AssertionError("Public baseline must remain in single-sequence mode")
    public_input = input_yaml_path().read_text(encoding="utf-8")
    if "msa: empty" not in public_input:
        raise AssertionError("Public baseline must use Boltz's explicit empty-MSA marker")
    notebook = (BUNDLE / "notebook" / "Boltz2_1CLL_TMscore_Benchmark.ipynb").read_text(encoding="utf-8")
    if '"USE_MSA_SERVER = True\\n"' in notebook:
        raise AssertionError("Benchmark notebook must not enable the MSA server")
    for forbidden_override in (
        '"--step_scale"',
        '"--recycling_steps"',
        '"--sampling_steps"',
        '"--diffusion_samples"',
        '"--max_parallel_samples"',
    ):
        if forbidden_override in public_runner:
            raise AssertionError(
                f"Public baseline overrides a Boltz-2 prediction default: {forbidden_override}"
            )
    adapter_text = (REPO_ROOT / "adapters" / "boltz2_pfode.py").read_text(encoding="utf-8")
    for required in ("initial_atom_coords", "deterministic", "gamma_0", "Boltz2.load_from_checkpoint"):
        if required not in adapter_text:
            raise AssertionError(f"O3 adapter is missing required custom behavior: {required}")
    config_path = REPO_ROOT / "configs" / "1cll.yaml"
    o3_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if o3_config["target"]["sequence"] != protein["sequence"]:
        raise AssertionError("Configuration and canonical Boltz YAML sequences differ")
    if o3_config.get("latent_dim") != "auto" or int(o3_config["boltz2"]["atom_slots"]) != 1184:
        raise AssertionError("1CLL must derive D=3552 from 1,184 sampler slots")
    if float(o3_config["boltz2"]["step_scale"]) != 1.0:
        raise AssertionError("configs/1cll.yaml must use step_scale=1.0 for PF-ODE")
    if bool(o3_config["boltz2"].get("use_msa_server", False)):
        raise AssertionError("configs/1cll.yaml must not enable the MSA server")
    if o3_config["boltz2"].get("inference_precision") not in {"auto", "bf16-mixed", "32"}:
        raise AssertionError("configs/1cll.yaml has an unsupported inference precision")
    if o3_config.get("output_layout") != "target_budget_method":
        raise AssertionError("O3 output layout is not grouped with the baseline by budget")
    expected_budgets = {
        "n20_k2": (20, 2, 10, 5),
        "n50_k5": (50, 5, 25, 7),
        "n100_k10": (100, 10, 50, 5),
    }
    actual_budgets = {
        item["name"]: (item["N"], item["K"], item["M"], item["d"])
        for item in o3_config["budgets"]
    }
    if actual_budgets != expected_budgets:
        raise AssertionError(f"Unexpected supported budget table: {actual_budgets}")


def public_info() -> dict[str, Any]:
    from public_runner import public_installation_info

    return public_installation_info()


def public_smoke() -> dict[str, object]:
    from public_runner import _run_public_predict, public_checkpoint_info
    from common import convert_cif_to_pdb, score_structure

    info = public_info()
    with tempfile.TemporaryDirectory(prefix="k10_n100_public_smoke_") as temp:
        sample_dir = Path(temp) / "sample_0000"
        cif_path = _run_public_predict(sample_dir, 0)
        pdb_path = convert_cif_to_pdb(cif_path, sample_dir / "sample_0000.pdb")
        score = score_structure(pdb_path)
    return {
        **info,
        "sample_seed": 0,
        "tm_score": score,
        "checkpoint": public_checkpoint_info(),
    }


def audit_vendor() -> dict[str, list[str]]:
    # This is a source comparison, so it must not require CUDA initialization
    # or an online ``uv run``.  Inspect the isolated environment directly.
    from public_runner import PUBLIC_PROJECT

    candidates = [
        *PUBLIC_PROJECT.glob(".venv/lib/python*/site-packages/boltz"),
        PUBLIC_PROJECT / ".venv" / "Lib" / "site-packages" / "boltz",
    ]
    archive_root = REPO_ROOT / ".uv-cache" / "archive-v0"
    if archive_root.is_dir():
        candidates.extend(archive_root.glob("*/Lib/site-packages/boltz"))
        candidates.extend(archive_root.glob("*/lib/python*/site-packages/boltz"))
    public_roots = sorted(
        candidate
        for candidate in candidates
        if candidate.is_dir()
        and (candidate.parent / "boltz-2.2.1.dist-info").is_dir()
    )
    if not public_roots:
        raise FileNotFoundError(
            "Public boltz==2.2.1 is not installed; initialize public_boltz first"
        )
    public_root = public_roots[0]
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
    from common import sha256_file
    import numpy as np
    import yaml

    config = yaml.safe_load((REPO_ROOT / "configs" / "1cll.yaml").read_text(encoding="utf-8"))
    # Boltz resolves the input YAML relative to cwd.
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
    checkpoint_path = Path(adapter.checkpoint_path)
    return {
        "custom_adapter": type(adapter).__name__,
        "latent_dim": adapter.latent_dim,
        "atom_slots": adapter.atom_slots,
        "checkpoint": {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-vendor", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    check_static()
    result: dict[str, object] = {"conditioning": "single_sequence"}
    if args.audit_vendor:
        result["vendor_audit"] = audit_vendor()
    if args.gpu:
        public_result = public_smoke()
        custom_result = gpu_smoke()
        result["public"] = public_result
        result["gpu_smoke"] = custom_result
        public_checkpoint = public_result["checkpoint"]
        custom_checkpoint = custom_result["checkpoint"]
        if public_checkpoint["sha256"] != custom_checkpoint["sha256"]:
            raise AssertionError(
                "Public and custom runners resolved different Boltz-2 checkpoints: "
                f"{public_checkpoint} != {custom_checkpoint}"
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
