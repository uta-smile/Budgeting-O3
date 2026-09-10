"""Persist and validate the common, unscaled Gaussian inputs to both decoders."""

from pathlib import Path
import os
import tempfile

import numpy as np


def load_shared_latents(path: Path, seed: int, n: int, latent_dim: int) -> np.ndarray:
    expected = np.random.default_rng(seed).normal(size=(n, latent_dim))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # Publish only a complete bank, including when workers start together.
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
            temporary = Path(handle.name)
            np.save(handle, expected)
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            temporary.unlink()
    latents = np.load(path, allow_pickle=False)
    if latents.dtype != expected.dtype or not np.array_equal(latents, expected):
        raise ValueError(f"Shared latent bank does not match seed/shape/distribution: {path}")
    return latents
