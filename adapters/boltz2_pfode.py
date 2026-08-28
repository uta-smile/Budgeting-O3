"""Deterministic Boltz-2 latent generator used by the O3 experiment.

This module is intentionally a thin research adapter around the vendored
Boltz source. The vendor patch adds two opt-in arguments to ``Boltz2.forward``:
an initial coordinate tensor and a deterministic sampling flag. The ordinary
Boltz command remains unchanged.
"""

from __future__ import annotations

import os
import tarfile
import urllib.request
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
        self.processed_dir = _path(
            boltz_config.get("processed_dir", "data/boltz2_1cll"),
            project_root,
        )
        self.recycling_steps = int(boltz_config.get("recycling_steps", 3))
        self.sampling_steps = int(boltz_config.get("sampling_steps", 200))
        self.step_scale = float(boltz_config.get("step_scale", 1.5))
        self.deterministic = bool(boltz_config.get("deterministic", True))
        self.use_msa_server = bool(boltz_config.get("use_msa_server", False))
        self.msa_server_url = str(
            boltz_config.get("msa_server_url", "https://api.colabfold.com")
        )
        self.no_kernels = bool(boltz_config.get("no_kernels", False))
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
        if not self.input_yaml.exists():
            raise FileNotFoundError(f"Missing Boltz input YAML: {self.input_yaml}")

        checkpoint = _ensure_assets(self.cache_dir)
        configured_checkpoint = config.get("boltz2", {}).get("checkpoint")
        if configured_checkpoint:
            checkpoint = _path(str(configured_checkpoint), Path(str(config["project_root"])))

        process_inputs(
            data=[self.input_yaml],
            out_dir=self.processed_dir,
            ccd_path=self.cache_dir / "ccd.pkl",
            mol_dir=self.cache_dir / "mols",
            msa_server_url=self.msa_server_url,
            msa_pairing_strategy="greedy",
            max_msa_seqs=int(config.get("boltz2", {}).get("max_msa_seqs", 8192)),
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
        # The paper's PF-ODE conversion explicitly disables EDM churn.
        diffusion_params.gamma_0 = 0.0
        diffusion_params.step_scale = self.step_scale
        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = False
        steering_args.physical_guidance_update = False
        steering_args.contact_guidance_update = False
        msa_args = MSAModuleArgs(
            subsample_msa=False,
            num_subsampled_msa=1024,
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
        initial_coords = (
            torch.from_numpy(latent_array.reshape(1, self.atom_slots, 3)).to(self.device)
            if deterministic
            else None
        )

        with torch.inference_mode():
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

        model_coords = result["sample_atom_coords"][0]
        pad_mask = self.features["atom_pad_mask"][0].bool()
        coord_unpad = model_coords[pad_mask].detach().cpu().numpy()
        self._write_pdb(output_path, coord_unpad, to_pdb)
        return output_path

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
