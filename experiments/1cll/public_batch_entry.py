"""Persistent, bounded inference using the installed public Boltz sampler.

Only scheduling, precision and output writing are adapted. The checkpoint,
model forward, stochastic sampler and featurizer come from public Boltz 2.2.1.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path


def main() -> None:
    import boltz.main as boltz_main
    from pytorch_lightning import Callback, seed_everything
    from torch.utils.data import DataLoader, Dataset

    request = json.loads(Path(sys.argv[1]).read_text())
    jobs = request["jobs"]
    precision = os.environ.get("BOLTZ_PUBLIC_PRECISION", "16-mixed")
    if precision not in {"16-mixed", "bf16-mixed", "32", "32-true"}:
        raise ValueError("Unsupported BOLTZ_PUBLIC_PRECISION")

    class RepeatedInput(Dataset):
        def __init__(self, batch):
            self.batch = batch

        def __len__(self):
            return len(jobs)

        def __getitem__(self, index):
            # Isolate any in-place model/Lightning changes between batches.
            return copy.deepcopy(self.batch)

    writer_base = boltz_main.BoltzWriter

    class StructureWriter(writer_base):
        def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices,
                               batch, batch_idx, dataloader_idx):
            job = jobs[batch_idx]
            if prediction.get("exception", False):
                raise RuntimeError("Public Boltz failed to generate this batch; see the inference log")
            self.output_dir = Path(job["output_dir"]) / "predictions"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Optional public confidence sidecars can fail for single chains.
            # Keep confidence_score for the stock writer's model-index mapping;
            # omit pLDDT/B-factors and sidecars, which TM-align does not use.
            prediction = dict(prediction)
            prediction.pop("plddt", None)
            super().write_on_batch_end(trainer, pl_module, prediction, batch_indices,
                                       batch, batch_idx, dataloader_idx)
            paths = sorted(self.output_dir.rglob("*.cif"),
                           key=lambda path: int(path.stem.rsplit("_model_", 1)[1]))
            if len(paths) != job["count"]:
                raise RuntimeError(f"Expected {job['count']} structures, found {len(paths)}")
            # A batch is reusable only after every structure was written.
            completed = {**job, "structures": [str(path) for path in paths],
                         "precision": precision,
                         "generation_seconds": time.perf_counter() - scheduler.started}
            marker = Path(job["output_dir"]) / "complete.json"
            temporary = marker.with_suffix(".tmp")
            temporary.write_text(json.dumps(completed, indent=2))
            temporary.replace(marker)

    class Schedule(Callback):
        def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx,
                                   dataloader_idx=0):
            job = jobs[batch_idx]
            seed_everything(job["seed"], workers=True)
            pl_module.predict_args["diffusion_samples"] = job["count"]
            # Set multiplicity equal to the parallel limit. This also avoids
            # the public sampler's non-ceiling chunk-count calculation.
            pl_module.predict_args["max_parallel_samples"] = job["count"]
            self.started = time.perf_counter()

    scheduler = Schedule()

    class PersistentTrainer(boltz_main.Trainer):
        def __init__(self, *args, **kwargs):
            kwargs["precision"] = precision
            kwargs["callbacks"] = [scheduler, *kwargs.get("callbacks", [])]
            super().__init__(*args, **kwargs)

        def predict(self, model, datamodule, **kwargs):
            datamodule.setup("predict")
            source = list(datamodule.predict_dataloader())
            if len(source) != 1:
                raise ValueError("Persistent benchmark requires exactly one target batch")
            loader = DataLoader(RepeatedInput(source[0]), batch_size=None, num_workers=0)
            # Keep Boltz's data module attached: its transfer hook moves feature
            # tensors while leaving frozen Record metadata on the CPU. Passing
            # only a loader invokes Lightning's generic dataclass transfer.
            original_loader = datamodule.predict_dataloader
            datamodule.predict_dataloader = lambda: loader
            try:
                return super().predict(model, datamodule=datamodule, **kwargs)
            finally:
                datamodule.predict_dataloader = original_loader

    boltz_main.Trainer = PersistentTrainer
    boltz_main.BoltzWriter = StructureWriter
    boltz_main.cli.main(args=[
        "predict", request["input"], "--out_dir", request["work_dir"],
        "--cache", request["cache"], "--seed", str(jobs[0]["seed"]),
        "--no_kernels", "--output_format", "mmcif", "--override",
        "--step_scale", "1.0",
    ], standalone_mode=False)


if __name__ == "__main__":
    main()
