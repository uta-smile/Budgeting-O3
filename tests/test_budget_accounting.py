import json
from pathlib import Path

import numpy as np
import yaml

from o3_boltz import baseline, o3


def test_1cll_main_protocol_uses_explicit_single_sequence_input() -> None:
    root = Path(__file__).parents[1]
    config_path = root / "configs" / "1cll.yaml"
    input_path = root / "data" / "1cll_boltz_input.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_config = yaml.safe_load(input_path.read_text(encoding="utf-8"))
    assert config["boltz2"]["use_msa_server"] is False
    assert input_config["sequences"][0]["protein"]["msa"] == "empty"


class FakeAdapter:
    latent_dim = 8
    atom_count = 3
    atom_slots = 4
    processed_dir = None
    use_msa_server = False
    msa_server_url = ""
    no_kernels = True

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.latents: list[np.ndarray] = []
        self.scores: dict[Path, float] = {}
        self.last_model_metrics: dict[str, float] = {}

    def generate(self, latent, output_path, config, metadata):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("ATOM\n", encoding="utf-8")
        self.calls.append(dict(metadata))
        self.latents.append(np.asarray(latent, dtype=np.float64).copy())
        self.scores[output_path] = float(latent[0])
        self.last_model_metrics = {"ptm": float(latent[1])}
        return output_path

    def score(self, structure_path, config):
        return self.scores[structure_path]


def test_o3_spends_exactly_n_calls_and_uses_reference_initialization(tmp_path, monkeypatch) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(
        o3,
        "_fit_and_acquire",
        lambda train_u, train_scores: np.full(train_u.shape[1], 0.5),
    )

    summary = o3.run_o3(
        adapter=adapter,
        config={"latent_dim": adapter.latent_dim, "boltz2": {}},
        budget={"name": "test", "N": 8, "K": 2, "M": 4, "d": 2},
        run_seed=3,
        output_dir=tmp_path / "o3",
    )

    assert len(adapter.calls) == 8
    assert summary["oracle_evaluations"] == 8
    assert all(call["deterministic"] for call in adapter.calls)
    assert [call["stage"] for call in adapter.calls[:4]] == ["phase1_random"] * 4
    assert [call["stage"] for call in adapter.calls[4:6]] == ["bo_initial_random"] * 2
    assert [call["stage"] for call in adapter.calls[6:]] == ["bo_acquisition"] * 2
    assert summary["D"] == adapter.latent_dim
    assert summary["k"] == 2


def test_best_k_of_n_spends_exactly_n_stochastic_calls(tmp_path) -> None:
    adapter = FakeAdapter()
    summary = baseline.run_best_k_of_n(
        adapter=adapter,
        config={
            "latent_dim": adapter.latent_dim,
            "boltz2": {},
            "best_k_of_n": {"deterministic": False},
        },
        budget={"name": "test", "N": 7, "K": 2, "M": 0, "d": 0},
        run_seed=4,
        output_dir=tmp_path / "baseline",
    )

    assert len(adapter.calls) == 7
    assert summary["oracle_evaluations"] == 7
    assert all(not call["deterministic"] for call in adapter.calls)
    assert summary["k"] == 2
    expected = np.random.default_rng(4).normal(size=(7, adapter.latent_dim))
    np.testing.assert_allclose(np.asarray(adapter.latents), expected)


def test_best_k_of_n_can_explicitly_rank_by_model_ptm(tmp_path) -> None:
    adapter = FakeAdapter()
    summary = baseline.run_best_k_of_n(
        adapter=adapter,
        config={
            "latent_dim": adapter.latent_dim,
            "boltz2": {},
            "best_k_of_n": {
                "deterministic": False,
                "selection_metric": "model_ptm",
            },
        },
        budget={"name": "test", "N": 7, "K": 2, "M": 0, "d": 0},
        run_seed=4,
        output_dir=tmp_path / "baseline_ptm",
    )

    assert summary["selection_metric"] == "model_ptm"
    returned = (tmp_path / "baseline_ptm" / "returned_candidates.json").read_text(
        encoding="utf-8"
    )
    records = json.loads(returned)
    expected = np.random.default_rng(4).normal(size=(7, adapter.latent_dim))
    expected_indices = np.argsort(expected[:, 1])[-2:][::-1]
    assert [record["index"] for record in records] == expected_indices.tolist()


def test_o3_and_baseline_share_seeded_phase1_latents(tmp_path, monkeypatch) -> None:
    baseline_adapter = FakeAdapter()
    baseline.run_best_k_of_n(
        adapter=baseline_adapter,
        config={"latent_dim": baseline_adapter.latent_dim, "boltz2": {}},
        budget={"name": "test", "N": 6, "K": 2, "M": 0, "d": 0},
        run_seed=9,
        output_dir=tmp_path / "baseline",
    )

    o3_adapter = FakeAdapter()
    monkeypatch.setattr(
        o3,
        "_fit_and_acquire",
        lambda train_u, train_scores: np.full(train_u.shape[1], 0.5),
    )
    o3.run_o3(
        adapter=o3_adapter,
        config={"latent_dim": o3_adapter.latent_dim, "boltz2": {}},
        budget={"name": "test", "N": 6, "K": 2, "M": 4, "d": 2},
        run_seed=9,
        output_dir=tmp_path / "o3",
    )

    np.testing.assert_allclose(
        np.asarray(o3_adapter.latents[:4]),
        np.asarray(baseline_adapter.latents[:4]),
    )
    assert all(call["deterministic"] for call in o3_adapter.calls)
