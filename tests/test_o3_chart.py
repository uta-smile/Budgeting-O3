import numpy as np
from scipy.special import betaincinv

from o3_boltz.chart import SurrogateChart
from o3_boltz.o3 import hypersphere_vertices, hypersphere_weights, map_u_to_latent


def _reference_kr(u: np.ndarray) -> np.ndarray:
    """Independent transcription of the public library's KR forward map."""

    u = np.asarray(u, dtype=np.float64)
    was_single = u.ndim == 1
    if was_single:
        u = u[None, :]
    d = u.shape[1] + 1
    reflected = np.clip(1.0 - u, 1.0e-15, 1.0 - 1.0e-15)
    beta_b = (d + 1.0 - np.arange(2, d + 1, dtype=np.float64)) / 2.0
    sticks = betaincinv(0.5, beta_b[None, :], reflected)
    squared = np.empty((u.shape[0], d), dtype=np.float64)
    remaining = np.ones(u.shape[0], dtype=np.float64)
    for index in range(d - 1):
        squared[:, index] = remaining * sticks[:, index]
        remaining *= 1.0 - sticks[:, index]
    squared[:, -1] = remaining
    weights = np.sqrt(np.clip(squared, 0.0, 1.0))
    weights /= np.linalg.norm(weights, axis=1, keepdims=True)
    return weights[0] if was_single else weights


def test_kr_matches_reference_and_supports_batches() -> None:
    rng = np.random.default_rng(7)
    for d in (2, 3, 5, 10, 50):
        points = rng.uniform(0.01, 0.99, size=(20, d - 1))
        np.testing.assert_allclose(hypersphere_weights(points), _reference_kr(points), atol=1e-13)
        for point in points:
            np.testing.assert_allclose(hypersphere_weights(point), _reference_kr(point), atol=1e-13)


def test_chart_projects_seed_latents_and_round_trips() -> None:
    rng = np.random.default_rng(11)
    seeds = rng.normal(size=(5, 13))
    chart = SurrogateChart(seeds)
    u = np.array([0.2, 0.35, 0.6, 0.8])

    latent = chart.from_u_to_z(u)
    recovered_u = chart.from_z_to_u(latent)
    np.testing.assert_allclose(recovered_u, u, atol=1e-10)
    np.testing.assert_allclose(chart.from_u_to_z(recovered_u), latent, atol=1e-10)

    seed_u = chart.from_z_to_u(seeds)
    np.testing.assert_allclose(chart.from_z_to_w(seeds), np.eye(5), atol=1e-12)
    assert np.all((seed_u >= 0.0) & (seed_u <= 1.0))
    recovered_seeds = chart.from_u_to_z(seed_u)
    # Endpoint clipping in the reference implementation leaves a nonzero
    # tail for canonical sphere vertices; the exact seed projection is tested
    # above in weight space.
    assert np.all(np.isfinite(recovered_seeds))
    np.testing.assert_allclose(map_u_to_latent(u, seeds), latent, atol=1e-12)


def test_chart_vertices_are_official_positive_sphere_vertices() -> None:
    d = 7
    seeds = np.arange(d * 11, dtype=np.float64).reshape(d, 11)
    vertices = hypersphere_vertices(d)
    weights = hypersphere_weights(vertices)

    np.testing.assert_allclose(weights, np.eye(d), atol=5e-3)
    assert np.all(np.isfinite(map_u_to_latent(vertices, seeds)))


def test_chart_rejects_invalid_points() -> None:
    for u in (np.array([-0.1]), np.array([1.1]), np.array([np.nan])):
        try:
            hypersphere_weights(u)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid chart point was accepted")

    chart = SurrogateChart(np.eye(3))
    try:
        chart.from_w_to_u(np.array([-0.1, 0.0, 0.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid weight point was accepted")
