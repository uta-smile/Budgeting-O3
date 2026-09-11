"""Generic single-chain target loading for the Boltz-2 experiments.

This module deliberately stops at constructing the runtime configuration.  It
does not contain sampling, optimization, or scoring logic.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


_TARGET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _target_parts(target: str) -> tuple[str, str]:
    value = str(target).strip()
    if not value or not _TARGET_NAME.fullmatch(value):
        raise ValueError(
            f"Invalid target {target!r}; use a target ID such as 30JR"
        )
    return value.upper(), value.lower()


def resolve_target_files(repo_root: Path, target: str) -> tuple[Path, Path]:
    """Resolve ``data/targets/<PDB_ID>/<PDB_ID>.{fasta,cif}``.

    The reference structure is returned separately so callers cannot
    accidentally pass it as Boltz input.
    """

    target_id, _ = _target_parts(target)
    target_dir = Path(repo_root) / "data" / "targets" / target_id
    fasta_path = target_dir / f"{target_id}.fasta"
    reference_path = target_dir / f"{target_id}.cif"
    missing = [path for path in (fasta_path, reference_path) if not path.is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Target {target_id} requires these files: {missing_text}"
        )
    return fasta_path, reference_path


def read_single_fasta(path: Path) -> str:
    """Read one FASTA protein record and return its normalized sequence."""

    records: list[str] = []
    sequence_lines: list[str] | None = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if sequence_lines is not None:
                records.append("".join(sequence_lines))
            sequence_lines = []
            continue
        if sequence_lines is None:
            raise ValueError(f"FASTA sequence data appears before a header: {path}")
        sequence_lines.append("".join(line.split()))

    if sequence_lines is not None:
        records.append("".join(sequence_lines))
    if len(records) != 1:
        if len(records) > 1:
            raise ValueError(
                "The simple target loader supports exactly one protein chain.\n"
                "Use an explicit Boltz YAML for multichain or ligand-containing inputs."
            )
        raise ValueError(f"FASTA file contains no protein record: {path}")
    sequence = records[0].upper()
    if not sequence:
        raise ValueError(f"FASTA protein record is empty: {path}")
    return sequence


def write_boltz_input(sequence: str, output_path: Path) -> Path:
    """Write the generated single-chain, sequence-only Boltz YAML."""

    normalized = "".join(str(sequence).split()).upper()
    if not normalized:
        raise ValueError("Cannot write Boltz input for an empty sequence")
    payload = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": normalized,
                    "msa": "empty",
                }
            }
        ],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    return output_path


def build_runtime_config(
    *,
    repo_root: Path,
    target: str,
    sequence: str,
    reference_path: Path,
    generated_boltz_yaml: Path,
    reference_chain: str = "A",
) -> dict[str, Any]:
    """Build the common runtime mapping consumed by all execution paths."""

    target_id, target_slug = _target_parts(target)
    normalized = "".join(str(sequence).split()).upper()
    if not normalized:
        raise ValueError("Target sequence must not be empty")
    root = Path(repo_root).resolve()
    return {
        "project_root": str(root),
        "target": {
            "name": target_slug,
            "pdb_id": target_id,
            "sequence": normalized,
            "reference_pdb": str(Path(reference_path).resolve()),
            "reference_chain": reference_chain,
            "generated_chain": "A",
        },
        "latent_dim": "auto",
        "boltz2": {
            "input_yaml": str(Path(generated_boltz_yaml).resolve()),
            "processed_dir": str(root / ".cache" / "targets" / target_slug / "processed"),
            "use_msa_server": False,
            "recycling_steps": 3,
            "sampling_steps": 200,
            "step_scale": 1.0,
            "stochastic_gamma_0": 0.8,
            "explicit_latent": True,
            "subsample_msa": False,
            "num_subsampled_msa": 1024,
            "inference_precision": "auto",
            "atom_slots": "auto",
            "no_kernels": True,
            "deterministic": True,
        },
        "output_dir": "outputs",
    }


def load_target_config(
    *, repo_root: Path, target: str, reference_chain: str = "A"
) -> tuple[dict[str, Any], Path, Path]:
    """Resolve target files, generate YAML, and return ``(config, fasta, cif)``."""

    target_id, target_slug = _target_parts(target)
    fasta_path, reference_path = resolve_target_files(repo_root, target_id)
    sequence = read_single_fasta(fasta_path)
    generated_yaml = (
        Path(repo_root) / ".cache" / "targets" / target_slug / "boltz_input.yaml"
    )
    write_boltz_input(sequence, generated_yaml)
    config = build_runtime_config(
        repo_root=repo_root,
        target=target_id,
        sequence=sequence,
        reference_path=reference_path,
        generated_boltz_yaml=generated_yaml,
        reference_chain=reference_chain,
    )
    return config, fasta_path, reference_path
