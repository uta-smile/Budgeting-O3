import numpy as np

from o3_boltz.o3 import hypersphere_vertices, hypersphere_weights, map_u_to_latent


def test_kr_weights_are_nonnegative_and_unit_norm() -> None:
    rng = np.random.default_rng(7)
    for d in (2, 3, 5, 10, 50):
        for u in rng.uniform(size=(20, d - 1)):
            weights = hypersphere_weights(u)
            assert weights.shape == (d,)
            assert np.all(weights >= 0.0)
            np.testing.assert_allclose(np.linalg.norm(weights), 1.0, atol=1e-12)


def test_chart_vertices_recover_each_seed() -> None:
    d = 7
    seeds = np.arange(d * 11, dtype=np.float64).reshape(d, 11)
    vertices = hypersphere_vertices(d)

    for index, u in enumerate(vertices):
        weights = hypersphere_weights(u)
        np.testing.assert_allclose(weights, np.eye(d)[index], atol=1e-12)
        np.testing.assert_allclose(map_u_to_latent(u, seeds), seeds[index], atol=1e-12)


def test_chart_rejects_points_outside_unit_cube() -> None:
    for u in (np.array([-0.1]), np.array([1.1]), np.array([np.nan])):
        try:
            hypersphere_weights(u)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid chart point was accepted")
