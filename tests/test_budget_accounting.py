from pathlib import Path

import numpy as np
import yaml

from o3_boltz import o3


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

    def generate(self, latent, output_path, config, metadata):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("ATOM\n", encoding="utf-8")
        self.calls.append(dict(metadata))
        self.latents.append(np.asarray(latent, dtype=np.float64).copy())
        self.scores[output_path] = float(latent[0])
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


def test_o3_logs_n20_protocol(capsys, tmp_path, monkeypatch) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(
        o3,
        "_fit_and_acquire",
        lambda train_u, train_scores: np.full(train_u.shape[1], 0.5),
    )

    o3.run_o3(
        adapter=adapter,
        config={"latent_dim": adapter.latent_dim, "boltz2": {}},
        budget={"name": "n20_k2", "N": 20, "K": 2, "M": 10, "d": 5},
        run_seed=0,
        output_dir=tmp_path / "o3_n20",
    )

    log = capsys.readouterr().out
    assert (
        "10 random Z samples -> select best 5 seeds -> "
        "2 random U samples -> 8 BO samples"
    ) in log
    assert "Total = 10 + 2 + 8 = 20 oracle evaluations" in log
    assert "phase 1 complete: selected best 5 seeds from 10 random Z samples" in log
    assert "phase 2 complete: 10 random Z + 2 random U = 12/20 evaluations" in log
    assert "O3 protocol complete: 10 + 2 + 8 = 20 oracle evaluations" in log


def test_random_pfode_spends_n_deterministic_random_z_calls(tmp_path) -> None:
    from o3_boltz.random_baseline import run_random_pfode

    adapter = FakeAdapter()
    summary = run_random_pfode(
        adapter=adapter,
        config={"latent_dim": adapter.latent_dim, "boltz2": {}},
        budget={"name": "n20_k2", "N": 20, "K": 2},
        run_seed=4,
        output_dir=tmp_path / "random_pfode",
    )

    assert len(adapter.calls) == 20
    assert summary["method"] == "random_pfode"
    assert summary["oracle_evaluations"] == 20
    assert summary["generator_sampling"] == "deterministic_pf_ode"
    assert summary["latent_sampler"] == "standard_normal_Z"
    assert all(call["deterministic"] for call in adapter.calls)
    expected = np.random.default_rng(4).normal(size=(20, adapter.latent_dim))
    np.testing.assert_allclose(np.asarray(adapter.latents), expected)
