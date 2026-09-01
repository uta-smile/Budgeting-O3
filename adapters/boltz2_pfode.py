"""Deterministic Boltz-2 latent generator used by the O3 experiment.

This module is intentionally a thin research adapter around the vendored
Boltz source. The vendor patch adds two opt-in arguments to ``Boltz2.forward``:
an initial coordinate tensor and a deterministic sampling flag. The ordinary
Boltz command remains unchanged.
"""

from __future__ import annotations

import os
import hashlib
import tarfile
import urllib.request
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from o3_boltz.tmscore import TMScoreOracle


def _path(value: str | os.PathLike[str], project_root: Path) -> Path:
    resolved = Path(value).expanduser()
    if not resolved.is_absolute():
        resolved = project_root / resolved
    return resolved


def _download(urls: list[str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in urls:
        try:
            print(f"Downloading {destination} ...", flush=True)
            urllib.request.urlretrieve(url, str(destination))  # noqa: S310
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Could not download {destination}") from last_error


def _ensure_assets(cache_dir: Path) -> Path:
    """Fetch only the Boltz-2 assets needed for structure generation."""

    from boltz.main import BOLTZ2_URL_WITH_FALLBACK, MOL_URL

    cache_dir.mkdir(parents=True, exist_ok=True)
    mol_dir = cache_dir / "mols"
    mol_tar = cache_dir / "mols.tar"
    if not mol_dir.exists():
        if not mol_tar.exists():
            _download([MOL_URL], mol_tar)
        print(f"Extracting {mol_tar} ...", flush=True)
        with tarfile.open(mol_tar, "r") as archive:
            archive.extractall(cache_dir)  # noqa: S202

    checkpoint = cache_dir / "boltz2_conf.ckpt"
    if not checkpoint.exists():
        _download(BOLTZ2_URL_WITH_FALLBACK, checkpoint)
    return checkpoint


def _processed_cache_path(
    base_dir: Path,
    input_yaml: Path,
    *,
    use_msa_server: bool,
    msa_server_url: str,
    max_msa_seqs: int,
) -> tuple[Path, str]:
    """Choose a cache that is unique to the preprocessing inputs/options.

    Boltz's ``process_inputs`` skips records already present in a directory,
    without checking whether the MSA options changed. A cache fingerprint
    prevents a previous single-sequence or different-server run from being
    silently reused for this benchmark.
    """

    digest = hashlib.sha256()
    digest.update(input_yaml.read_bytes())
    digest.update(f"use_msa_server={use_msa_server}\n".encode("utf-8"))
    digest.update(f"msa_server_url={msa_server_url}\n".encode("utf-8"))
    digest.update(f"max_msa_seqs={max_msa_seqs}\n".encode("utf-8"))
    fingerprint = digest.hexdigest()
    return base_dir / f"o3cache_{fingerprint[:16]}", fingerprint


class Boltz2PFODEAdapter:
    """One cached Boltz-2 model plus the TM-score oracle."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        target = config["target"]
        project_root = Path(str(config["project_root"])).resolve()
        boltz_config = dict(config.get("boltz2", {}))

        cache_value = os.environ.get("BOLTZ_CACHE", boltz_config.get("cache_dir", "~/.boltz"))
        self.cache_dir = _path(cache_value, project_root)
        self.input_yaml = _path(
            boltz_config.get("input_yaml", "data/1cll_boltz_input.yaml"),
            project_root,
        )
        self.requested_processed_dir = _path(
            boltz_config.get("processed_dir", "data/boltz2_1cll"),
            project_root,
        )
        self.recycling_steps = int(boltz_config.get("recycling_steps", 3))
        self.sampling_steps = int(boltz_config.get("sampling_steps", 200))
        self.step_scale = float(boltz_config.get("step_scale", 1.5))
        # Boltz-2's ordinary sampler uses EDM churn gamma_0=0.8. O3 changes
        # this to zero at call time for PF-ODE sampling; keep the two modes
        # separate instead of silently using PF-ODE parameters for the
        # Best-k-of-N baseline.
        self.stochastic_gamma_0 = float(boltz_config.get("stochastic_gamma_0", 0.8))
        if self.stochastic_gamma_0 < 0.0:
            raise ValueError("boltz2.stochastic_gamma_0 must be non-negative")
        self.explicit_latent = bool(boltz_config.get("explicit_latent", True))
        # Match an unflagged upstream ``boltz predict`` invocation. The
        # ``--subsample_msa`` Click option is an is_flag with no explicit
        # default, so Click passes False when the flag is omitted (despite the
        # stale function signature/help text claiming a True default).
        self.subsample_msa = bool(boltz_config.get("subsample_msa", False))
        self.num_subsampled_msa = int(boltz_config.get("num_subsampled_msa", 1024))
        if self.num_subsampled_msa <= 0:
            raise ValueError("boltz2.num_subsampled_msa must be positive")
        self.deterministic = bool(boltz_config.get("deterministic", True))
        self.use_msa_server = bool(boltz_config.get("use_msa_server", False))
        self.msa_server_url = str(
            boltz_config.get("msa_server_url", "https://api.colabfold.com")
        )
        self.max_msa_seqs = int(boltz_config.get("max_msa_seqs", 8192))
        if self.max_msa_seqs <= 0:
            raise ValueError("boltz2.max_msa_seqs must be positive")
        self.no_kernels = bool(boltz_config.get("no_kernels", False))
        # The upstream Boltz CLI constructs a Lightning Trainer with
        # precision="bf16-mixed" for Boltz-2. The adapter calls forward()
        # directly, so it must reproduce that autocast context explicitly.
        self.inference_precision = str(
            boltz_config.get("inference_precision", "bf16-mixed")
        )
        if self.inference_precision not in {"bf16-mixed", "32"}:
            raise ValueError(
                "boltz2.inference_precision must be 'bf16-mixed' or '32'"
            )
        configured_atom_slots = boltz_config.get("atom_slots")
        self.configured_atom_slots = (
            None if configured_atom_slots in (None, "auto") else int(configured_atom_slots)
        )
        if self.configured_atom_slots is not None:
            if self.configured_atom_slots <= 0 or self.configured_atom_slots % 32 != 0:
                raise ValueError("boltz2.atom_slots must be a positive multiple of 32")

        reference = _path(str(target["reference_pdb"]), project_root)
        self.oracle = TMScoreOracle(reference, target.get("reference_chain"))
        self.sequence = str(target["sequence"])
        self.last_model_metrics: dict[str, float] = {}

        self._load_model(config)

    def _load_model(self, config: Mapping[str, Any]) -> None:
        try:
            import torch
            from boltz.data.module.inferencev2 import PredictionDataset, collate
            from boltz.data.types import Manifest
            from boltz.main import (
                Boltz2DiffusionParams,
                BoltzSteeringParams,
                MSAModuleArgs,
                PairformerArgsV2,
                process_inputs,
            )
            from boltz.model.models.boltz2 import Boltz2
        except ImportError as exc:
            raise RuntimeError(
                "Boltz-2 dependencies are unavailable. Run this project through "
                "run_experiment.sh so uv installs the vendored CUDA package."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "The O3 Boltz-2 generator requires a CUDA GPU. Run it on the lab GPU node."
            )
        # Match the upstream CLI's matmul setting before constructing the
        # model. This is process-global, just as it is in boltz.main.predict.
        torch.set_float32_matmul_precision("highest")
        if not self.input_yaml.exists():
            raise FileNotFoundError(f"Missing Boltz input YAML: {self.input_yaml}")

        checkpoint = _ensure_assets(self.cache_dir)
        configured_checkpoint = config.get("boltz2", {}).get("checkpoint")
        if configured_checkpoint:
            checkpoint = _path(str(configured_checkpoint), Path(str(config["project_root"])))
        self.checkpoint_path = checkpoint

        self.processed_dir, self.processing_fingerprint = _processed_cache_path(
            self.requested_processed_dir,
            self.input_yaml,
            use_msa_server=self.use_msa_server,
            msa_server_url=self.msa_server_url,
            max_msa_seqs=self.max_msa_seqs,
        )

        process_inputs(
            data=[self.input_yaml],
            out_dir=self.processed_dir,
            ccd_path=self.cache_dir / "ccd.pkl",
            mol_dir=self.cache_dir / "mols",
            msa_server_url=self.msa_server_url,
            msa_pairing_strategy="greedy",
            max_msa_seqs=self.max_msa_seqs,
            use_msa_server=self.use_msa_server,
            boltz2=True,
            preprocessing_threads=1,
        )

        processed = self.processed_dir / "processed"
        manifest = Manifest.load(processed / "manifest.json")
        if len(manifest.records) != 1:
            raise ValueError(
                f"Expected exactly one processed input, found {len(manifest.records)}"
            )

        dataset = PredictionDataset(
            manifest=manifest,
            target_dir=processed / "structures",
            msa_dir=processed / "msa",
            mol_dir=self.cache_dir / "mols",
            constraints_dir=(processed / "constraints")
            if (processed / "constraints").exists()
            else None,
            template_dir=(processed / "templates")
            if (processed / "templates").exists()
            else None,
            extra_mols_dir=(processed / "mols")
            if (processed / "mols").exists()
            else None,
            max_atoms=self.configured_atom_slots,
        )
        self.features = collate([dataset[0]])
        self.record = manifest.records[0]
        self.structure_path = processed / "structures" / f"{self.record.id}.npz"

        self.device = torch.device("cuda")
        for key, value in self.features.items():
            if isinstance(value, torch.Tensor):
                self.features[key] = value.to(self.device)

        atom_mask = self.features["atom_pad_mask"]
        atom_count = int(atom_mask[0].sum().item())
        atom_slots = int(atom_mask.shape[1])
        if self.configured_atom_slots is not None and atom_slots != self.configured_atom_slots:
            raise ValueError(
                "Boltz featurization did not produce the requested sampler width: "
                f"{atom_slots} != {self.configured_atom_slots}"
            )
        self.atom_count = atom_count
        self.atom_slots = atom_slots
        self.latent_dim = atom_slots * 3
        print(
            f"Boltz atom tensor: {atom_count} real atoms, "
            f"{atom_slots} padded sampler slots.",
            flush=True,
        )

        diffusion_params = Boltz2DiffusionParams()
        # Load the ordinary stochastic Boltz-2 setting. O3's deterministic
        # PF-ODE override is applied per generation in generate(), because the
        # same model instance serves both O3 and Best-k-of-N modes.
        diffusion_params.gamma_0 = self.stochastic_gamma_0
        diffusion_params.step_scale = self.step_scale
        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = False
        steering_args.physical_guidance_update = False
        steering_args.contact_guidance_update = False
        msa_args = MSAModuleArgs(
            subsample_msa=self.subsample_msa,
            num_subsampled_msa=self.num_subsampled_msa,
            use_paired_feature=True,
        )
        predict_args = {
            "recycling_steps": self.recycling_steps,
            "sampling_steps": self.sampling_steps,
            "diffusion_samples": 1,
            "max_parallel_samples": 1,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        }
        self.model = Boltz2.load_from_checkpoint(
            checkpoint,
            strict=True,
            predict_args=predict_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            use_kernels=not self.no_kernels,
            pairformer_args=asdict(PairformerArgsV2()),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args),
        )
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        latent: np.ndarray,
        output_path: Path,
        config: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> Path:
        del config
        import torch
        from boltz.data.write.pdb import to_pdb

        latent_array = np.asarray(latent, dtype=np.float32)
        if latent_array.shape != (self.latent_dim,):
            raise ValueError(
                f"Expected latent shape {(self.latent_dim,)}, got {latent_array.shape}"
            )
        deterministic = bool(metadata.get("deterministic", self.deterministic))
        sampling_mode = "deterministic PF-ODE" if deterministic else "stochastic Boltz-2"
        active_gamma_0 = 0.0 if deterministic else self.stochastic_gamma_0
        latent_source = "explicit z" if self.explicit_latent else "internal torch noise"
        print(
            f"[Boltz-2] sampling={sampling_mode} | deterministic={deterministic} "
            f"| latent={latent_source} | gamma_0={active_gamma_0:g}",
            flush=True,
        )
        initial_coords = (
            torch.from_numpy(latent_array.reshape(1, self.atom_slots, 3)).to(self.device)
            if self.explicit_latent
            else None
        )

        # The paper's PF-ODE conversion is deterministic because it removes
        # EDM churn. The ordinary baseline retains Boltz-2's stochastic churn
        # and random SE(3) augmentation. The model is shared across calls, so
        # set the mode-specific parameter immediately around the forward pass.
        diffusion_module = getattr(self.model, "structure_module", None)
        if diffusion_module is None or not hasattr(diffusion_module, "gamma_0"):
            raise RuntimeError(
                "Boltz-2 model does not expose structure_module.gamma_0; "
                "cannot enforce separate PF-ODE and stochastic sampling modes"
            )
        previous_gamma_0 = diffusion_module.gamma_0
        diffusion_module.gamma_0 = 0.0 if deterministic else self.stochastic_gamma_0

        try:
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if self.inference_precision == "bf16-mixed"
                else nullcontext()
            )
            with torch.inference_mode(), autocast_context:
                result = self.model(
                    self.features,
                    recycling_steps=self.recycling_steps,
                    num_sampling_steps=self.sampling_steps,
                    diffusion_samples=1,
                    max_parallel_samples=1,
                    run_confidence_sequentially=True,
                    initial_atom_coords=initial_coords,
                    deterministic=deterministic,
                )
                self.last_model_metrics = self._extract_model_metrics(result, torch)
        finally:
            diffusion_module.gamma_0 = previous_gamma_0

        model_coords = result["sample_atom_coords"][0]
        pad_mask = self.features["atom_pad_mask"][0].bool()
        coord_unpad = model_coords[pad_mask].detach().cpu().numpy()
        self._write_pdb(output_path, coord_unpad, to_pdb)
        return output_path

    @staticmethod
    def _extract_model_metrics(result: Mapping[str, Any], torch) -> dict[str, float]:
        """Return scalar Boltz confidence outputs for optional diagnostics."""

        metrics: dict[str, float] = {}
        for name in ("ptm", "complex_plddt", "complex_pde"):
            value = result.get(name)
            if value is None or not torch.is_tensor(value):
                continue
            flattened = value.detach().float().reshape(-1)
            if flattened.numel() == 1 and bool(torch.isfinite(flattened).item()):
                metrics[name] = float(flattened.item())
        return metrics

    def _write_pdb(self, output_path: Path, coordinates: np.ndarray, to_pdb) -> None:
        from boltz.data.types import Coords, Interface, StructureV2

        structure = StructureV2.load(self.structure_path).remove_invalid_chains()
        if len(coordinates) != len(structure.atoms):
            raise ValueError(
                "Generated coordinate count does not match the processed structure: "
                f"{len(coordinates)} != {len(structure.atoms)}"
            )

        atoms = structure.atoms.copy()
        atoms["coords"] = coordinates
        atoms["is_present"] = True
        residues = structure.residues.copy()
        residues["is_present"] = True
        coordinate_records = np.array([(x,) for x in coordinates], dtype=Coords)
        generated = replace(
            structure,
            atoms=atoms,
            residues=residues,
            interfaces=np.array([], dtype=Interface),
            coords=coordinate_records,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pdb_text = to_pdb(generated, boltz2=True)
        if "\nATOM  " not in f"\n{pdb_text}" and "\nHETATM" not in f"\n{pdb_text}":
            raise ValueError("Boltz PDB writer produced no atom records")
        output_path.write_text(pdb_text, encoding="utf-8")

    def score(self, structure_path: Path, config: Mapping[str, Any]) -> float:
        return self.oracle.score(
            structure_path,
            config["target"].get("generated_chain"),
        )


def create(config: Mapping[str, Any]) -> Boltz2PFODEAdapter:
    """Create the built-in deterministic Boltz-2 adapter."""

    return Boltz2PFODEAdapter(config)
