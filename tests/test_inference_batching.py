"""CPU coverage for public scheduling without downloading a checkpoint."""
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

BUNDLE = Path(__file__).parents[1] / "experiments" / "1cll" / "k10_n100"
sys.path.insert(0, str(BUNDLE))
import common
import public_runner


def test_public_batch_plan_tail_and_completed_batch_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "N", 10)
    jobs = public_runner._batch_jobs(tmp_path, 11, 4)
    assert [job["count"] for job in jobs] == [4, 4, 2]
    assert [job["seed"] for job in jobs] == [common.sample_seed(11, i) for i in (0, 4, 8)]
    for job in jobs:
        directory = Path(job["output_dir"])
        directory.mkdir(parents=True)
        paths = []
        for i in range(job["count"]):
            path = directory / f"model_{i}.cif"
            path.write_text("data_fixture")
            paths.append(str(path))
        (directory / "complete.json").write_text(json.dumps({
            **job, "structures": paths, "precision": "32",
        }))
    monkeypatch.setattr(public_runner, "_uv", lambda: pytest.fail("completed batches must not launch a process"))
    generated = public_runner._run_public_batches(tmp_path, 11, 4)
    assert len(generated) == 10
    assert generated[9]["batch_count"] == 2
    with pytest.raises(ValueError, match="Incompatible batch"):
        public_runner._run_public_batches(tmp_path, 11, 3)


def test_public_worker_keeps_one_model_and_trainer_with_bounded_batches(tmp_path, monkeypatch):
    # This is a CPU integration test; avoid probing the lab driver's NVML.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import BasePredictionWriter
    import public_batch_entry

    calls = []
    transfers = []

    @dataclass(frozen=True)
    class Record:
        name: str

    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.predict_args = {}
        def predict_step(self, batch, batch_idx):
            assert batch["record"] == [Record("target")]
            count = self.predict_args["diffusion_samples"]
            assert self.predict_args["max_parallel_samples"] == count
            calls.append((count, torch.initial_seed(), batch["features"].clone()))
            return {"exception": False, "count": count, "plddt": torch.ones(count)}

    class Writer(BasePredictionWriter):
        def __init__(self):
            super().__init__(write_interval="batch")
        def write_on_batch_end(self, trainer, model, prediction, batch_indices, batch, batch_idx, dataloader_idx):
            assert "plddt" not in prediction
            for rank in range(prediction["count"]):
                (self.output_dir / f"target_model_{rank}.cif").write_text("data_fixture")

    class Data(pl.LightningDataModule):
        def setup(self, stage):
            assert stage == "predict"
        def predict_dataloader(self):
            return [{"features": torch.tensor([1., 2.]), "record": [Record("target")]}]
        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            transfers.append(device.type)
            batch["features"] = batch["features"].to(device)
            return batch

    module = ModuleType("boltz.main")
    module.Trainer = pl.Trainer
    module.BoltzWriter = Writer
    models = []
    def cli_main(args, standalone_mode):
        model = Model()
        models.append(model)
        trainer = module.Trainer(accelerator="cpu", devices=1, logger=False,
                                 enable_checkpointing=False, enable_progress_bar=False,
                                 callbacks=[module.BoltzWriter()])
        trainer.predict(model, datamodule=Data(), return_predictions=False)
    module.cli = SimpleNamespace(main=cli_main)
    import boltz
    monkeypatch.setitem(sys.modules, "boltz.main", module)
    monkeypatch.setattr(boltz, "main", module, raising=False)
    jobs = [{"start": i * 4, "count": count, "seed": 30 + i,
             "output_dir": str(tmp_path / f"batch_{i}")}
            for i, count in enumerate([4, 4, 2])]
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"jobs": jobs, "input": "input.yaml",
                                  "cache": "cache", "work_dir": "work"}))
    monkeypatch.setattr(sys, "argv", ["worker", str(request)])
    monkeypatch.setenv("BOLTZ_PUBLIC_PRECISION", "32")
    public_batch_entry.main()
    assert len(models) == 1
    assert transfers == ["cpu", "cpu", "cpu"]
    assert [(count, seed) for count, seed, _ in calls] == [(4, 30), (4, 31), (2, 32)]
    assert all(torch.equal(batch, torch.tensor([1., 2.])) for _, _, batch in calls)
    assert all(public_runner._read_batch(job) is not None for job in jobs)


def test_adapter_batches_distinct_latents_and_removes_padding(monkeypatch, tmp_path):
    from adapters.boltz2_pfode import Boltz2PFODEAdapter
    import numpy as np

    adapter = object.__new__(Boltz2PFODEAdapter)
    adapter.latent_dim, adapter.atom_slots = 12, 4
    adapter.deterministic, adapter.explicit_latent = True, True
    adapter.stochastic_gamma_0 = 0.8
    adapter.device = torch.device("cpu")
    adapter.inference_precision = "32"
    adapter.recycling_steps, adapter.sampling_steps = 3, 200
    adapter.features = {"atom_pad_mask": torch.tensor([[1, 1, 1, 0]])}
    diffusion = SimpleNamespace(gamma_0=0.8)
    class Model:
        structure_module = diffusion
        def __call__(self, features, **kwargs):
            assert diffusion.gamma_0 == 0.0
            assert kwargs["diffusion_samples"] == kwargs["max_parallel_samples"] == 2
            assert kwargs["deterministic"] is True
            return {"sample_atom_coords": kwargs["initial_atom_coords"]}
    adapter.model = Model()
    written = {}
    adapter._write_pdb = lambda path, coords, writer: written.update({path: coords.copy()})
    latents = np.arange(24).reshape(2, 12)
    paths = [tmp_path / "a.pdb", tmp_path / "b.pdb"]
    assert adapter.generate_batch(latents, paths, {}, [{"deterministic": True}] * 2) == paths
    np.testing.assert_array_equal(written[paths[0]], latents[0].reshape(4, 3)[:3])
    np.testing.assert_array_equal(written[paths[1]], latents[1].reshape(4, 3)[:3])
    assert diffusion.gamma_0 == 0.8
    with pytest.raises(ValueError, match="Cannot mix"):
        adapter.generate_batch(latents, paths, {}, [{"deterministic": True}, {"deterministic": False}])


def test_public_batch_failure_surfaces_saved_error(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setattr(common, "N", 2)
    monkeypatch.setattr(public_runner, "_uv", lambda: "uv")

    def fail(command, **kwargs):
        kwargs["stdout"].write("ValueError: frozen record fixture\n")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(public_runner.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="frozen record fixture") as error:
        public_runner._run_public_batches(tmp_path, 11, 2)
    assert str(tmp_path / "boltz_batches.log") in str(error.value)
