"""Reproducibility metadata shared by the O3 and baseline runners."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping


def _sha256_tree(root: Path) -> str | None:
    if not root.exists():
        return None
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_run_metadata(
    *,
    config: Mapping[str, Any],
    adapter: Any,
) -> dict[str, Any]:
    """Collect serializable generator, cache, and runtime provenance."""

    boltz_config = config.get("boltz2", {})
    processed_dir = getattr(adapter, "processed_dir", None)
    processed_path = Path(processed_dir) if processed_dir is not None else None
    msa_root = processed_path / "processed" / "msa" if processed_path else None
    manifest_path = processed_path / "processed" / "manifest.json" if processed_path else None

    msa_digest = None
    if msa_root is not None:
        msa_digest = _sha256_tree(msa_root)
    manifest_digest = None
    # Boltz uses an integer sentinel (-1) for explicit single-sequence mode,
    # but auto-generated MSAs are recorded as string file identifiers such as
    # ``1cll_boltz_input_0``. Preserve the manifest values instead of forcing
    # both forms through int().
    msa_ids: list[str | int] | None = None
    if manifest_path is not None and manifest_path.exists():
        manifest_digest = _sha256_file(manifest_path)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = manifest.get("records", [])
            values = {
                chain["msa_id"]
                for record in records
                for chain in record.get("chains", [])
                if "msa_id" in chain
            }
            msa_ids = sorted(values, key=str)
        except (OSError, ValueError, TypeError, KeyError):
            msa_ids = None

    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    except ImportError:
        torch_version = None
        cuda_version = None
        cuda_available = False
        gpu_name = None

    checkpoint = getattr(adapter, "checkpoint_path", None)
    checkpoint_info: dict[str, Any] | None = None
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        checkpoint_info = {
            "path": str(checkpoint_path),
            "exists": checkpoint_path.exists(),
            "size_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else None,
        }

    return {
        "generator": {
            "adapter": type(adapter).__name__,
            "latent_dim": getattr(adapter, "latent_dim", None),
            "atom_count": getattr(adapter, "atom_count", None),
            "atom_slots": getattr(adapter, "atom_slots", None),
            "recycling_steps": getattr(adapter, "recycling_steps", None),
            "sampling_steps": getattr(adapter, "sampling_steps", None),
            "step_scale": getattr(adapter, "step_scale", None),
            "stochastic_gamma_0": getattr(adapter, "stochastic_gamma_0", None),
            "explicit_latent": getattr(adapter, "explicit_latent", None),
            "subsample_msa": getattr(adapter, "subsample_msa", None),
            "num_subsampled_msa": getattr(adapter, "num_subsampled_msa", None),
            "inference_precision": getattr(adapter, "inference_precision", None),
            "no_kernels": getattr(adapter, "no_kernels", boltz_config.get("no_kernels")),
            "checkpoint": checkpoint_info,
        },
        "msa_cache": {
            "input_msa_path": getattr(adapter, "input_msa_path", None),
            "input_msa_sha256": getattr(adapter, "input_msa_sha256", None),
            "processed_dir": str(processed_path) if processed_path else None,
            "processing_fingerprint": getattr(adapter, "processing_fingerprint", None),
            "msa_sha256": msa_digest,
            "manifest_sha256": manifest_digest,
            "msa_ids": msa_ids,
            "has_msa": bool(msa_ids)
            and any(str(msa_id) not in {"-1", "empty"} for msa_id in msa_ids),
            "use_msa_server": getattr(adapter, "use_msa_server", boltz_config.get("use_msa_server")),
            "server_url": getattr(adapter, "msa_server_url", boltz_config.get("msa_server_url")),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch_version,
            "cuda": cuda_version,
            "cuda_available": cuda_available,
            "gpu": gpu_name,
        },
    }
