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
from common import (  # noqa: E402
    configure_budget,
    output_root,
    sample_seed,
    shared_replicate_seeds,
)
from public_runner import run_replicate  # noqa: E402
from verify import check_static  # noqa: E402


def test_single_sequence_setup() -> None:
    check_static()


def test_replicates_use_disjoint_notebook_seed_blocks() -> None:
    assert [sample_seed(0, i) for i in (0, 99)] == [1, 100]
    assert all(0 <= sample_seed(20250117, i) <= 2**32 - 1 for i in range(100))
    assert sample_seed(4, 0) not in {sample_seed(0, i) for i in range(common.N)}


def test_shared_replicate_seed_schedule_is_reproducible() -> None:
    assert shared_replicate_seeds(5) == [20250117, 20251126, 20252135, 20253144, 20254153]
    assert shared_replicate_seeds(3, seed_start=17, seed_step=19) == [17, 36, 55]


def test_random_replicate_seeds_are_unique_valid_31_bit_values() -> None:
    seeds = common.random_replicate_seeds(5)
    assert len(seeds) == 5
    assert len(set(seeds)) == 5
    assert all(0 < seed < 2**31 for seed in seeds)


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


def test_o3_configs_use_unit_pfode_step_scale() -> None:
    import yaml

    for name in ("o3.yaml", "o3_n20_k2.yaml", "o3_n50_k5.yaml"):
        config = yaml.safe_load((BUNDLE / name).read_text(encoding="utf-8"))
        assert config["boltz2"]["step_scale"] == 1.0
        assert config["seed"] == 20250117
        assert config["seed_step"] == 1009


if __name__ == "__main__":
    test_single_sequence_setup()
    test_replicates_use_disjoint_notebook_seed_blocks()
    test_shared_replicate_seed_schedule_is_reproducible()
    test_random_replicate_seeds_are_unique_valid_31_bit_values()
    with TemporaryDirectory(prefix="k10_n100_test_") as temp:
        test_public_runner_resumes_without_regenerating(Path(temp))
    test_small_budget_configs_are_distinct()
    print("bundle tests passed")
