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
    for input_name in ("1cll.yaml", "1cll_n20.yaml", "1cll_n50.yaml"):
        input_text = (BUNDLE / "inputs" / input_name).read_text(encoding="utf-8")
        if "msa: empty" not in input_text:
            raise AssertionError(f"{input_name} must explicitly select single-sequence mode")
        if "msa:" in input_text.replace("msa: empty", ""):
            raise AssertionError(f"{input_name} supplies an MSA despite single-sequence setup")
    public_project = (BUNDLE / "public_boltz" / "pyproject.toml").read_text(encoding="utf-8")
    if '"boltz==2.2.1"' not in public_project or "vendor/boltz" in public_project:
        raise AssertionError("Public environment is not pinned to public boltz==2.2.1")
    public_runner = (BUNDLE / "public_runner.py").read_text(encoding="utf-8")
    if 'PUBLIC_INPUT = BUNDLE / "inputs" / "1cll_single_sequence.yaml"' not in public_runner:
        raise AssertionError("Public baseline is not using the single-sequence input")
    if '"--use_msa_server"' in public_runner:
        raise AssertionError("Public baseline must remain in single-sequence mode")
    single_input = (BUNDLE / "inputs" / "1cll_single_sequence.yaml").read_text(encoding="utf-8")
    if "msa: empty" not in single_input:
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
    for config_name in ("o3.yaml", "o3_n20_k2.yaml", "o3_n50_k5.yaml"):
        o3_config = yaml.safe_load((BUNDLE / config_name).read_text(encoding="utf-8"))
        if float(o3_config["boltz2"]["step_scale"]) != 1.0:
            raise AssertionError(f"{config_name} must use step_scale=1.0 for PF-ODE")
        if bool(o3_config["boltz2"].get("use_msa_server", False)):
            raise AssertionError(f"{config_name} must not enable the MSA server")
        if any(key in o3_config["boltz2"] for key in ("msa_server_url", "subsample_msa", "num_subsampled_msa")):
            raise AssertionError(f"{config_name} contains unnecessary MSA configuration")


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
    return {"custom_adapter": type(adapter).__name__, "latent_dim": adapter.latent_dim, "atom_slots": adapter.atom_slots}


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
        result["public"] = public_smoke()
        result["gpu_smoke"] = gpu_smoke()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
