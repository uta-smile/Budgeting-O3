from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).parents[1]
BUNDLE = ROOT / "experiments" / "1cll" / "k10_n100"
sys.path.insert(0, str(BUNDLE))

import common  # noqa: E402
from common import configure_budget, output_root, sample_seed, validate_frozen_msa  # noqa: E402
from public_runner import run_replicate  # noqa: E402
from verify import check_fixture  # noqa: E402


def test_frozen_msa_and_notebook_fixture() -> None:
    assert len(validate_frozen_msa()) == 64
    result = check_fixture()
    assert result["mean_of_K"] == 0.6788870863920917
    assert result["max_of_K"] == 0.7958910827539285


def test_replicates_use_disjoint_notebook_seed_blocks() -> None:
    assert [sample_seed(0, i) for i in (0, 99)] == [0, 99]
    assert [sample_seed(1, i) for i in (0, 99)] == [common.N, 2 * common.N - 1]
    assert sample_seed(4, 0) not in {sample_seed(0, i) for i in range(common.N)}


def test_public_runner_resumes_without_regenerating(tmp_path: Path) -> None:
    output = tmp_path / "best_k_of_n"
    rows = []

    def fake_output_root(method: str, run_id: str) -> Path:
        assert method == "best_k_of_n"
        return output / run_id

    def fake_predict(sample_dir: Path, seed: int) -> Path:
        path = sample_dir / "fake.cif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data_fake\n", encoding="utf-8")
        return path

    def fake_convert(cif: Path, pdb: Path) -> Path:
        pdb.parent.mkdir(parents=True, exist_ok=True)
        pdb.write_text("ATOM\n", encoding="utf-8")
        return pdb

    def fake_score(path: Path) -> float:
        return float(int(path.stem.rsplit("_", 1)[-1])) / 100.0

    with patch("public_runner.output_root", fake_output_root), patch(
        "public_runner.public_installation_info",
        return_value={"version": "2.2.1", "module": "public/boltz/__init__.py"},
    ), patch("public_runner._run_public_predict", side_effect=fake_predict) as predict, patch(
        "public_runner.convert_cif_to_pdb", side_effect=fake_convert
    ), patch("public_runner.score_structure", side_effect=fake_score):
        run_replicate("resume_test", 0, resume=False)
        assert predict.call_count == 100
        predict.reset_mock()
        run_replicate("resume_test", 0, resume=True)
        predict.assert_not_called()


def test_custom_boltz_adapter_is_rejected_by_legacy_baseline() -> None:
    from o3_boltz import baseline

    adapter_type = type("Boltz2PFODEAdapter", (), {})
    adapter = adapter_type()
    try:
        baseline.run_best_k_of_n(
            adapter=adapter,
            config={"latent_dim": 1},
            budget={"N": 1, "K": 1},
            run_seed=0,
            output_dir=ROOT / "tmp-test-output",
        )
    except RuntimeError as error:
        assert "reserved for O3" in str(error)
    else:
        raise AssertionError("legacy baseline accepted the custom O3 adapter")


def test_small_budget_configs_are_distinct() -> None:
    configure_budget("n20_k2")
    assert (common.N, output_root("best_k_of_n", "run").parts[-5:]) == (
        20,
        ("1cll", "k2_n20", "best_k_of_n", "runs", "run"),
    )
    configure_budget("n50_k5")
    assert (common.N, output_root("o3", "run").parts[-5:]) == (
        50,
        ("1cll", "k5_n50", "o3", "runs", "run"),
    )
    configure_budget("n100_k10")


if __name__ == "__main__":
    test_frozen_msa_and_notebook_fixture()
    test_replicates_use_disjoint_notebook_seed_blocks()
    with TemporaryDirectory(prefix="k10_n100_test_") as temp:
        test_public_runner_resumes_without_regenerating(Path(temp))
    test_custom_boltz_adapter_is_rejected_by_legacy_baseline()
    test_small_budget_configs_are_distinct()
    print("bundle tests passed")
