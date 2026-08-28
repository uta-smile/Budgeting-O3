from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np


class GeneratorOracle(Protocol):
    """The small interface required from the lab's deterministic Boltz adapter."""

    def generate(
        self,
        latent: np.ndarray,
        output_path: Path,
        config: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> Path | None:
        """Decode one latent and write a PDB/mmCIF structure."""

    def score(self, structure_path: Path, config: Mapping[str, Any]) -> float:
        """Return the scalar oracle reward for one generated structure."""


def load_adapter(spec: str, config: Mapping[str, Any]) -> GeneratorOracle:
    """Load ``module:function`` and call it as an adapter factory."""

    try:
        module_name, function_name = spec.split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError(
            f"Invalid adapter spec {spec!r}; use the form module:function"
        ) from exc

    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    adapter = factory(config)
    if not hasattr(adapter, "generate") or not hasattr(adapter, "score"):
        raise TypeError(
            f"Adapter {spec!r} must return an object with generate() and score() methods"
        )
    return adapter

