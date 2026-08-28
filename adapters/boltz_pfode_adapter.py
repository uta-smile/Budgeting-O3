from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from o3_boltz.tmscore import TMScoreOracle


class FunctionAdapter:
    """Bridge a lab-provided deterministic Boltz generator to the O3 runner.

    The imported function must have this shape:

        generate(
            *, latent, output_path, sequence, config, metadata
        ) -> path | None

    It must use the lab's probability-flow Boltz-2 implementation and write
    one PDB or mmCIF structure to ``output_path``.
    """

    def __init__(self, generator, config: Mapping[str, Any]):
        self.generator = generator
        target = config["target"]
        reference = Path(str(target["reference_pdb"]))
        if not reference.is_absolute():
            reference = Path(str(config.get("project_root", Path.cwd()))) / reference
        self.oracle = TMScoreOracle(reference, target.get("reference_chain"))
        self.sequence = str(target["sequence"])

    def generate(
        self,
        latent: np.ndarray,
        output_path: Path,
        config: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> Path | None:
        return self.generator(
            latent=latent,
            output_path=output_path,
            sequence=self.sequence,
            config=config,
            metadata=metadata,
        )

    def score(self, structure_path: Path, config: Mapping[str, Any]) -> float:
        return self.oracle.score(
            structure_path,
            config["target"].get("generated_chain"),
        )


def _load_function(spec: str):
    try:
        module_name, function_name = spec.split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError(
            f"Invalid O3_BOLTZ_GENERATOR_SPEC={spec!r}; use module:function"
        ) from exc
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def create(config: Mapping[str, Any]) -> FunctionAdapter:
    spec = os.environ.get("O3_BOLTZ_GENERATOR_SPEC")
    if not spec:
        raise RuntimeError(
            "The public Boltz CLI does not expose the deterministic PF-ODE latent "
            "interface needed by O3. Set O3_BOLTZ_GENERATOR_SPEC to the lab adapter, "
            "for example: export O3_BOLTZ_GENERATOR_SPEC=my_boltz_adapter:generate"
        )
    return FunctionAdapter(_load_function(spec), config)
