from pathlib import Path
from warnings import WarningMessage

import numpy as np
import yaml
from botorch.exceptions.warnings import OptimizationWarning

from o3_boltz import o3
from o3_boltz.cli import _budget_method_root
from o3_boltz.tmscore import TMScoreOracle


def test_1cll_main_protocol_uses_explicit_single_sequence_input() -> None:
    root = Path(__file__).parents[1]
    config_path = root / "configs" / "1cll.yaml"
    input_path = root / "data" / "1cll_boltz_input.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_config = yaml.safe_load(input_path.read_text(encoding="utf-8"))
    assert config["boltz2"]["use_msa_server"] is False
    assert input_config["sequences"][0]["protein"]["msa"] == "empty"
    assert [budget["name"] for budget in config["budgets"]] == [
        "n20_k2",
        "n50_k5",
        "n100_k10",
    ]


def test_canonical_output_layout_groups_methods_under_one_budget(tmp_path) -> None:
    budget = {"name": "n100_k10", "N": 100, "K": 10}
    root = _budget_method_root(
        output_root=tmp_path,
        target_name="1cll",
        method_name="o3",
        budget=budget,
        output_layout="target_budget_method",
    )
    assert root == tmp_path / "1cll" / "k10_n100" / "o3"


def test_tm_score_uses_the_requested_reference_chain() -> None:
    reference = Path(__file__).parents[1] / "data" / "1CLL.pdb"
    oracle = TMScoreOracle(reference, "A")
    assert abs(oracle.score(reference, "A") - 1.0) < 1.0e-12
    try:
        TMScoreOracle(reference, "missing")
    except ValueError as exc:
        assert "Requested chain" in str(exc)
    else:
        raise AssertionError("missing reference chain silently fell back to another chain")


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


def _warning(message: str) -> WarningMessage:
    return WarningMessage(message, OptimizationWarning, __file__, 1)


def test_o3_accepts_recoverable_gp_warning_but_rejects_timeout() -> None:
    assert o3._o3_gp_warning_handler(_warning("ABNORMAL_TERMINATION_IN_LNSRCH"))
    assert not o3._o3_gp_warning_handler(_warning("Optimization timed out after 60 seconds"))


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


def test_random_pfode_resume_reuses_seeded_prefix(tmp_path) -> None:
    from o3_boltz.random_baseline import run_random_pfode

    class RestartSafeAdapter(FakeAdapter):
        def __init__(self, stop_after=None):
            super().__init__()
            self.stop_after = stop_after

        def generate(self, latent, output_path, config, metadata):
            if self.stop_after is not None and len(self.calls) == self.stop_after:
                raise RuntimeError("simulated interruption")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(str(float(latent[0])), encoding="utf-8")
            self.calls.append(dict(metadata))
            self.latents.append(np.asarray(latent, dtype=np.float64).copy())
            return output_path

        def score(self, structure_path, config):
            return float(Path(structure_path).read_text(encoding="utf-8"))

    output = tmp_path / "random_resume"
    args = {
        "config": {"latent_dim": FakeAdapter.latent_dim, "boltz2": {}},
        "budget": {"name": "test", "N": 8, "K": 2},
        "run_seed": 31,
        "output_dir": output,
    }
    interrupted = RestartSafeAdapter(stop_after=3)
    try:
        run_random_pfode(adapter=interrupted, **args)
    except RuntimeError as exc:
        assert str(exc) == "simulated interruption"
    else:
        raise AssertionError("simulated interruption did not occur")

    resumed = RestartSafeAdapter()
    summary = run_random_pfode(adapter=resumed, resume=True, **args)
    assert summary["oracle_evaluations"] == 8
    assert len(resumed.calls) == 5
    expected = np.random.default_rng(31).normal(size=(8, FakeAdapter.latent_dim))
    for index in range(8):
        np.testing.assert_array_equal(
            np.load(output / "latents" / f"latent_{index:04d}.npy"),
            expected[index],
        )


def test_o3_resume_preserves_seeded_latent_sequence(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        o3,
        "_fit_and_acquire",
        lambda train_u, train_scores: np.full(train_u.shape[1], 0.5),
    )
    config = {"latent_dim": FakeAdapter.latent_dim, "boltz2": {}}
    budget = {"name": "test", "N": 8, "K": 2, "M": 4, "d": 2}

    full_adapter = FakeAdapter()
    full_dir = tmp_path / "full"
    full_summary = o3.run_o3(
        adapter=full_adapter,
        config=config,
        budget=budget,
        run_seed=17,
        output_dir=full_dir,
    )

    class InterruptingAdapter(FakeAdapter):
        def __init__(self, stop_after: int) -> None:
            super().__init__()
            self.stop_after: int | None = stop_after

        def generate(self, latent, output_path, config, metadata):
            if self.stop_after is not None and len(self.calls) == self.stop_after:
                raise RuntimeError("simulated interruption")
            return super().generate(latent, output_path, config, metadata)

    for stop_after in (2, 5):
        resumed_adapter = InterruptingAdapter(stop_after)
        resumed_dir = tmp_path / f"resumed_{stop_after}"
        try:
            o3.run_o3(
                adapter=resumed_adapter,
                config=config,
                budget=budget,
                run_seed=17,
                output_dir=resumed_dir,
            )
        except RuntimeError as exc:
            assert str(exc) == "simulated interruption"
        else:
            raise AssertionError("simulated interruption did not occur")

        resumed_adapter.stop_after = None
        resumed_summary = o3.run_o3(
            adapter=resumed_adapter,
            config=config,
            budget=budget,
            run_seed=17,
            output_dir=resumed_dir,
            resume=True,
        )
        assert resumed_summary["max_of_K"] == full_summary["max_of_K"]
        for index in range(8):
            np.testing.assert_array_equal(
                np.load(resumed_dir / "latents" / f"latent_{index:04d}.npy"),
                np.load(full_dir / "latents" / f"latent_{index:04d}.npy"),
            )


class BatchFakeAdapter(FakeAdapter):
    def __init__(self):
        super().__init__()
        self.batch_lengths = []

    def generate_batch(self, latents, output_paths, config, metadata):
        self.batch_lengths.append(len(latents))
        return [self.generate(z, path, config, item)
                for z, path, item in zip(latents, output_paths, metadata)]


def test_batching_preserves_latents_budget_and_sequential_bo(tmp_path, monkeypatch):
    acquired_counts = []

    def acquire(train_u, train_scores):
        acquired_counts.append(len(train_scores))
        return np.full(train_u.shape[1], 0.5)

    monkeypatch.setattr(o3, "_fit_and_acquire", acquire)
    budget = {"name": "test", "N": 10, "K": 2, "M": 5, "d": 2}
    sequential, batched = FakeAdapter(), BatchFakeAdapter()
    for adapter, batch_size, folder in [(sequential, 1, "single"), (batched, 3, "batch")]:
        summary = o3.run_o3(
            adapter=adapter, config={"latent_dim": 8, "inference_batch_size": batch_size},
            budget=budget, run_seed=7, output_dir=tmp_path / folder,
        )
        assert summary["oracle_evaluations"] == 10
        assert summary["bo_batch_size"] == 1
    assert batched.batch_lengths == [3, 2, 2]
    assert acquired_counts == [4, 5, 6, 4, 5, 6]
    np.testing.assert_array_equal(sequential.latents, batched.latents)
    assert sequential.calls == batched.calls


def test_batch_resume_replays_partial_batch_and_rejects_size_change(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(o3, "_fit_and_acquire", lambda x, y: np.full(x.shape[1], 0.5))
    adapter = BatchFakeAdapter()
    original_generate = adapter.generate_batch
    count = 0

    def interrupted(latents, output_paths, config, metadata):
        nonlocal count
        count += 1
        if count == 2:
            adapter.generate(latents[0], output_paths[0], config, metadata[0])
            raise RuntimeError("interrupted")
        return original_generate(latents, output_paths, config, metadata)

    adapter.generate_batch = interrupted
    config = {"latent_dim": 8, "inference_batch_size": 3}
    args = dict(adapter=adapter, config=config,
                budget={"name": "test", "N": 10, "K": 2, "M": 5, "d": 2},
                run_seed=7, output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="interrupted"):
        o3.run_o3(**args)
    adapter.generate_batch = original_generate
    summary = o3.run_o3(**args, resume=True)
    assert summary["oracle_evaluations"] == 10
    assert adapter.batch_lengths == [3, 2, 2]
    with pytest.raises(ValueError, match="different inference batch size"):
        o3.run_o3(**{**args, "config": {**config, "inference_batch_size": 2}}, resume=True)
