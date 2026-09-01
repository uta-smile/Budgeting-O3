"""Knothe--Rosenblatt surrogate chart used by the O3 method.

The implementation mirrors the public ``latent-spaces-by-example`` chart:
uniform coordinates are reflected before inverse-Beta stick breaking, and
latent points are formed with a row-major seed matrix ``(d, D)``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import betainc, betaincinv


SURROGATE_CHART_VERSION = "latent-spaces-by-example-kr-0.1.2"
_ENDPOINT_EPS = 1.0e-15


def _as_batch(values: np.ndarray, width: int, name: str) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        if array.shape != (width,):
            raise ValueError(f"{name} must have shape ({width},) or (N, {width})")
        return array[None, :], True
    if array.ndim == 2 and array.shape[1] == width:
        return array, False
    raise ValueError(f"{name} must have shape ({width},) or (N, {width})")


def _validate_cube(u: np.ndarray, tolerance: float) -> np.ndarray:
    if not np.all(np.isfinite(u)):
        raise ValueError("u must contain only finite values")
    if np.any(u < -tolerance) or np.any(u > 1.0 + tolerance):
        raise ValueError("u must contain values in [0, 1]")
    return np.clip(u, 0.0, 1.0)


def _from_u_to_w(u: np.ndarray, tolerance: float) -> np.ndarray:
    """Apply the public library's vectorized KR forward map."""

    u = _validate_cube(u, tolerance)
    d = u.shape[1] + 1

    # The public implementation uses the reflected cube coordinate before
    # inverse-Beta sampling. Clipping is required because scipy returns NaN
    # for some inverse-Beta endpoint combinations.
    reflected = np.clip(1.0 - u, _ENDPOINT_EPS, 1.0 - _ENDPOINT_EPS)
    beta_b = (d + 1.0 - np.arange(2, d + 1, dtype=np.float64)) / 2.0
    sticks = betaincinv(0.5, beta_b[None, :], reflected)

    squared = np.empty((u.shape[0], d), dtype=np.float64)
    remaining = np.ones(u.shape[0], dtype=np.float64)
    for index in range(d - 1):
        stick = sticks[:, index]
        squared[:, index] = remaining * stick
        remaining *= 1.0 - stick
    squared[:, -1] = remaining

    squared = np.clip(squared, 0.0, 1.0)
    weights = np.sqrt(squared)
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    if not np.all(np.isfinite(weights)) or np.any(norms == 0.0):
        raise ValueError("Knothe--Rosenblatt chart produced invalid weights")
    return weights / norms


def _from_w_to_u(w: np.ndarray, tolerance: float) -> np.ndarray:
    """Apply the public library's inverse KR map."""

    weights = _validate_weights(w, tolerance)
    squared = weights**2
    suffix_mass = np.cumsum(squared[:, ::-1], axis=1)[:, ::-1]
    numerator = squared[:, :-1]
    # Match the public implementation's fixed numerical floor.  This is
    # deliberately independent of the chart validation tolerance: the latter
    # controls whether an input is accepted, while this floor controls the
    # inverse-Beta calculation at near-zero suffix masses.
    conditional = numerator / np.clip(suffix_mass[:, :-1], 1.0e-30, None)

    d = w.shape[1]
    beta_b = (d + 1.0 - np.arange(2, d + 1, dtype=np.float64)) / 2.0
    u = 1.0 - betainc(0.5, beta_b[None, :], np.clip(conditional, 0.0, 1.0))
    return np.clip(u, 0.0, 1.0)


def _validate_weights(w: np.ndarray, tolerance: float) -> np.ndarray:
    if not np.all(np.isfinite(w)):
        raise ValueError("w must contain only finite values")
    if np.any(w < -tolerance):
        raise ValueError("w must lie in the nonnegative orthant")
    norms = np.linalg.norm(w, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(np.abs(norms - 1.0) > tolerance):
        raise ValueError("w must have unit norm")
    # The public chart validates the sphere/orthant but preserves the input
    # values for the subsequent transport.  Do not silently renormalize here.
    return w


class SurrogateChart:
    """Identity-transport KR surrogate chart over selected seed latents."""

    version = SURROGATE_CHART_VERSION

    def __init__(
        self,
        seed_latents: np.ndarray,
        *,
        tolerance: float = 1.0e-12,
    ) -> None:
        seeds = np.asarray(seed_latents, dtype=np.float64)
        if seeds.ndim != 2 or seeds.shape[0] < 2:
            raise ValueError("seed_latents must have shape (d, D) with d >= 2")
        if not np.all(np.isfinite(seeds)):
            raise ValueError("seed_latents must contain only finite values")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        self.seed_latents = seeds.copy()
        self.d, self.latent_dim = seeds.shape
        self.tolerance = float(tolerance)
        self._seed_latents_pseudoinverse = np.linalg.pinv(self.seed_latents)

    def from_u_to_w(self, u: np.ndarray) -> np.ndarray:
        values, was_single = _as_batch(u, self.d - 1, "u")
        weights = _from_u_to_w(values, self.tolerance)
        return weights[0] if was_single else weights

    def from_w_to_u(self, w: np.ndarray) -> np.ndarray:
        values, was_single = _as_batch(w, self.d, "w")
        coordinates = _from_w_to_u(values, self.tolerance)
        return coordinates[0] if was_single else coordinates

    def from_u_to_z(self, u: np.ndarray) -> np.ndarray:
        values, was_single = _as_batch(u, self.d - 1, "u")
        latent = _from_u_to_w(values, self.tolerance) @ self.seed_latents
        return latent[0] if was_single else latent

    def from_w_to_z(self, w: np.ndarray) -> np.ndarray:
        values, was_single = _as_batch(w, self.d, "w")
        latent = _validate_weights(values, self.tolerance) @ self.seed_latents
        return latent[0] if was_single else latent

    def from_z_to_w(self, z: np.ndarray) -> np.ndarray:
        values, was_single = _as_batch(z, self.latent_dim, "z")
        if not np.all(np.isfinite(values)):
            raise ValueError("z must contain only finite values")
        projected = values @ self._seed_latents_pseudoinverse
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        weights = projected / np.maximum(norms, self.tolerance)
        weights = np.clip(weights, 0.0, None)
        positive_norms = np.linalg.norm(weights, axis=1, keepdims=True)
        weights = weights / np.maximum(positive_norms, self.tolerance)
        return weights[0] if was_single else weights

    def from_z_to_u(self, z: np.ndarray) -> np.ndarray:
        weights = self.from_z_to_w(z)
        return self.from_w_to_u(weights)


def hypersphere_weights(u: np.ndarray) -> np.ndarray:
    """Compatibility wrapper for the KR map without seed latents."""

    values = np.asarray(u, dtype=np.float64)
    if values.ndim == 1:
        return _from_u_to_w(values[None, :], 1.0e-12)[0]
    if values.ndim == 2:
        return _from_u_to_w(values, 1.0e-12)
    raise ValueError("u must have shape (d - 1,) or (N, d - 1)")


def hypersphere_vertices(d: int) -> np.ndarray:
    """Return U coordinates for the canonical positive-sphere vertices."""

    if d < 2:
        raise ValueError("d must be at least 2")
    return SurrogateChart(np.eye(d, dtype=np.float64)).from_z_to_u(
        np.eye(d, dtype=np.float64)
    )


def map_u_to_latent(u: np.ndarray, seed_latents: np.ndarray) -> np.ndarray:
    """Compatibility wrapper for ``SurrogateChart.from_u_to_z``."""

    return SurrogateChart(seed_latents).from_u_to_z(u)
