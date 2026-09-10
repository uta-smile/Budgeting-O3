from types import SimpleNamespace

import numpy as np
import pytest
import torch

from o3_boltz.random_baseline import run_random_pfode
from o3_boltz.shared_latents import load_shared_latents


def test_bank_is_exact_standard_normal_and_rejects_corruption(tmp_path):
    path = tmp_path / "seed_20250117.npy"
    expected = np.random.default_rng(20250117).normal(size=(100, 3552))
    np.testing.assert_array_equal(load_shared_latents(path, 20250117, 100, 3552), expected)
    np.testing.assert_array_equal(load_shared_latents(path, 20250117, 100, 3552), expected)
    np.save(path, expected * 2)
    with pytest.raises(ValueError, match="Shared latent bank"):
        load_shared_latents(path, 20250117, 100, 3552)


def test_controlled_methods_receive_identical_saved_latents(tmp_path):
    class Adapter:
        def __init__(self):
            self.latents, self.metadata = [], []

        def generate(self, latent, output_path, config, metadata):
            self.latents.append(latent.copy())
            self.metadata.append(metadata)
            output_path.write_text(str(float(latent[0])))
            return output_path

        def score(self, path, config):
            return float(path.read_text())

    adapters = []
    for method in ("best_k_of_n", "random_pfode"):
        adapter = Adapter()
        adapters.append(adapter)
        summary = run_random_pfode(
            adapter=adapter, config={"latent_dim": 12}, budget={"N": 4, "K": 2},
            run_seed=17, output_dir=tmp_path / method, method=method,
            shared_latent_path=tmp_path / "shared_latents" / "seed_17.npy",
        )
        assert summary["latent_source"] == "shared_standard_normal_bank"
        assert summary["latent_seed"] == 17
        assert summary["latent_dim"] == 12
    assert np.array_equal(adapters[0].latents, adapters[1].latents)
    for index in range(4):
        assert adapters[0].metadata[index]["deterministic"] is False
        assert adapters[1].metadata[index]["deterministic"] is True
        assert np.array_equal(
            np.load(tmp_path / "best_k_of_n" / "latents" / f"latent_{index:04d}.npy"),
            np.load(tmp_path / "random_pfode" / "latents" / f"latent_{index:04d}.npy"),
        )


def test_adapter_passes_same_unscaled_coordinates_and_distinct_modes(tmp_path):
    from adapters.boltz2_pfode import Boltz2PFODEAdapter

    adapter = object.__new__(Boltz2PFODEAdapter)
    adapter.latent_dim, adapter.atom_slots = 12, 4
    adapter.deterministic, adapter.explicit_latent = True, True
    adapter.stochastic_gamma_0 = 0.8
    adapter.device = torch.device("cpu")
    adapter.inference_precision = "32"
    adapter.recycling_steps, adapter.sampling_steps = 3, 200
    adapter.features = {"atom_pad_mask": torch.tensor([[1, 1, 1, 0]])}
    diffusion = SimpleNamespace(gamma_0=0.8)
    calls = []

    class Model:
        structure_module = diffusion

        def __call__(self, features, **kwargs):
            calls.append((kwargs["initial_atom_coords"].clone(),
                          kwargs["deterministic"], diffusion.gamma_0))
            return {"sample_atom_coords": kwargs["initial_atom_coords"]}

    adapter.model = Model()
    adapter._write_pdb = lambda *args: None
    latent = np.random.default_rng(17).normal(size=12)
    for deterministic in (False, True):
        adapter.generate(latent, tmp_path / "sample.pdb", {}, {"deterministic": deterministic})
    assert torch.equal(calls[0][0], calls[1][0])
    np.testing.assert_array_equal(calls[0][0].numpy().reshape(-1), latent.astype(np.float32))
    assert calls[0][1:] == (False, 0.8)
    assert calls[1][1:] == (True, 0.0)
    assert diffusion.gamma_0 == 0.8
