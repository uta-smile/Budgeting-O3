from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).parents[1]
BUNDLE = ROOT / "experiments" / "1cll"
sys.path.insert(0, str(BUNDLE))

import common  # noqa: E402
import multi_gpu  # noqa: E402
import multi_gpu_worker  # noqa: E402
import public_runner  # noqa: E402
import run as bundle_run  # noqa: E402
from common import (  # noqa: E402
    configure_budget,
    output_root,
    sample_seed,
    shared_replicate_seeds,
)
from public_runner import run_replicate  # noqa: E402
from run import load_baseline_seeds, load_comparison_seeds  # noqa: E402
from verify import check_static  # noqa: E402


def test_single_sequence_setup() -> None:
    check_static()
    assert public_runner.PUBLIC_INPUT.resolve() == common.input_yaml_path().resolve()


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


def test_ten_replicates_are_supported_by_the_canonical_launcher() -> None:
    assert common.SUPPORTED_REPLICATES == (1, 3, 5, 10)
    with patch.object(sys, "argv", ["run.py", "--replicates", "10"]):
        assert bundle_run.parse_args().replicates == 10


def test_gpu_seed_assignment_is_stable_and_uses_each_seed_once() -> None:
    assert multi_gpu.resolve_gpu_ids("0,2,3") == ["0", "2", "3"]
    assignments = multi_gpu.assign_seeds(
        ["0", "1", "2", "3"], list(range(10))
    )
    assert [item["seeds"] for item in assignments] == [
        [0, 4, 8],
        [1, 5, 9],
        [2, 6],
        [3, 7],
    ]
    assert sorted(seed for item in assignments for seed in item["seeds"]) == list(
        range(10)
    )


def test_gpu_auto_discovery_uses_nvidia_smi(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    discovered = SimpleNamespace(stdout="0\n1\n2\n", returncode=0)
    with patch("multi_gpu.subprocess.run", return_value=discovered) as run_process:
        assert multi_gpu.resolve_gpu_ids("auto") == ["0", "1", "2"]
    assert "--query-gpu=index" in run_process.call_args.args[0]


def test_public_assets_are_prepared_in_the_isolated_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", "/wrong/project/.venv")
    with patch("public_runner._uv", return_value="uv"), patch(
        "public_runner.subprocess.run"
    ) as run_process:
        public_runner.prepare_public_assets()

    command = run_process.call_args.args[0]
    assert command[:4] == [
        "uv",
        "run",
        "--project",
        str(public_runner.PUBLIC_PROJECT),
    ]
    assert command[-1] == str(public_runner.PUBLIC_CACHE)
    assert run_process.call_args.kwargs["check"] is True
    assert "VIRTUAL_ENV" not in run_process.call_args.kwargs["env"]


def test_replicate_seed_validation_rejects_out_of_range_values() -> None:
    for seeds in ([-1], [2**32]):
        try:
            common.resolve_replicate_seeds(1, seeds=seeds)
        except ValueError as exc:
            assert "unsigned 32-bit" in str(exc)
        else:
            raise AssertionError("out-of-range seed was accepted")


def test_can_import_exact_seeds_from_a_baseline_run(tmp_path: Path) -> None:
    root = tmp_path / "baseline"
    root.mkdir()
    (root / "provenance.json").write_text(
        '{"budget": "n100_k10", "seeds": [11, 22, 33]}', encoding="utf-8"
    )
    with patch("common.output_root", return_value=root):
        assert load_baseline_seeds("run01", "n100_k10", 3) == [11, 22, 33]


def test_comparison_manifest_path_is_method_independent() -> None:
    common.configure_budget("n100_k10")
    path = common.comparison_run_root("run01")
    assert path.parts[-5:] == ("outputs", "1cll", "k10_n100", "runs", "run01")


def test_resume_can_restore_seeds_from_early_run_manifest(tmp_path: Path) -> None:
    root = tmp_path / "comparison"
    root.mkdir()
    (root / "run_manifest.json").write_text(
        '{"budget": "n100_k10", "seeds": [44, 55, 66]}', encoding="utf-8"
    )
    with patch("common.comparison_run_root", return_value=root):
        assert load_comparison_seeds("run01", "n100_k10", 3) == [44, 55, 66]


def _runner_args(**overrides):
    values = {
        "method": "best-k-of-n",
        "budget": "n100_k10",
        "replicates": 3,
        "run_id": "run01",
        "seed_start": common.DEFAULT_REPLICATE_SEED_START,
        "seed_step": common.DEFAULT_REPLICATE_SEED_STEP,
        "seed_list": None,
        "seed_mode": None,
        "seeds_from_baseline_run": None,
        "resume": False,
        "smoke": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_runner_records_default_random_seeds_before_generation(tmp_path: Path) -> None:
    manifest_root = tmp_path / "comparison"
    with patch("run.parse_args", return_value=_runner_args()), patch(
        "run.random_replicate_seeds", return_value=[101, 202, 303]
    ), patch("common.comparison_run_root", return_value=manifest_root), patch(
        "run.run_public"
    ) as run_public:
        bundle_run.main()

    manifest = json.loads((manifest_root / "run_manifest.json").read_text())
    assert manifest["seed_mode"] == "fresh_os_random"
    assert manifest["seeds"] == [101, 202, 303]
    assert run_public.call_args.kwargs["seeds"] == [101, 202, 303]


def test_runner_routes_gpu_runs_through_parallel_workers(tmp_path: Path) -> None:
    manifest_root = tmp_path / "comparison"
    report = {
        "wall_seconds": 12.0,
        "workers": [
            {"gpu": "0", "seconds_by_method": {"o3": 10.0}},
            {"gpu": "1", "seconds_by_method": {"o3": 9.0}},
        ],
    }
    args = _runner_args(method="all", gpus="0,1", batch_size=4)
    with patch("run.parse_args", return_value=args), patch(
        "run.random_replicate_seeds", return_value=[101, 202, 303]
    ), patch("common.comparison_run_root", return_value=manifest_root), patch(
        "multi_gpu.launch_workers", return_value=report
    ) as launch, patch("multi_gpu.finalize_reports") as finalize, patch(
        "run.run_public"
    ) as run_public, patch("run.run_o3") as run_o3, patch(
        "run.run_random_pfode"
    ) as run_random:
        bundle_run.main()

    assert launch.call_args.kwargs["gpu_ids"] == ["0", "1"]
    assert launch.call_args.kwargs["seeds"] == [101, 202, 303]
    assert launch.call_args.kwargs["batch_size"] == 4
    finalize.assert_called_once()
    run_public.assert_not_called()
    run_o3.assert_not_called()
    run_random.assert_not_called()


def test_gpu_worker_dispatches_seed_shard_without_shared_reports(
    tmp_path: Path, monkeypatch
) -> None:
    marker = tmp_path / "complete.json"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multi_gpu_worker.py",
            "--method",
            "all",
            "--budget",
            "n100_k10",
            "--run-id",
            "run01",
            "--worker-index",
            "0",
            "--gpu-id",
            "2",
            "--marker",
            str(marker),
            "--seed-list",
            "11",
            "22",
            "--batch-size",
            "4",
        ],
    )
    with patch("multi_gpu_worker.public_runner.run_replicate") as public, patch(
        "multi_gpu_worker.canonical_run.run_o3"
    ) as o3, patch("multi_gpu_worker.random_pfode_runner.run") as random_pfode:
        multi_gpu_worker.main()

    assert [call.args[1] for call in public.call_args_list] == [11, 22]
    assert all(call.kwargs["batch_size"] == 4 for call in public.call_args_list)
    assert o3.call_args.kwargs["worker_only"] is True
    assert random_pfode.call_args.kwargs["write_run_reports"] is False
    assert json.loads(marker.read_text())["seeds"] == [11, 22]


def test_smoke_does_not_create_seed_manifest() -> None:
    with patch("run.parse_args", return_value=_runner_args(smoke=True)), patch(
        "run.subprocess.run"
    ) as run_process, patch("run.random_replicate_seeds") as random_seeds, patch(
        "common.comparison_run_root"
    ) as comparison_root:
        bundle_run.main()

    random_seeds.assert_not_called()
    comparison_root.assert_not_called()
    assert run_process.call_args.args[0][-1] == "--gpu"


def test_runner_resume_restores_manifest_seeds(tmp_path: Path) -> None:
    manifest_root = tmp_path / "comparison"
    manifest_root.mkdir()
    (manifest_root / "run_manifest.json").write_text(
        '{"target":"1cll","budget":"n100_k10","run_id":"run01",'
        '"method_selection":"both","replicates":3,"seed_mode":"fresh_os_random",'
        '"seeds":[101,202,303]}',
        encoding="utf-8",
    )
    args = _runner_args(method="o3", resume=True)
    with patch("run.parse_args", return_value=args), patch(
        "common.comparison_run_root", return_value=manifest_root
    ), patch("run.run_o3") as run_o3:
        bundle_run.main()

    assert run_o3.call_args.args[4] == [101, 202, 303]
    assert run_o3.call_args.kwargs["resume"] is True


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
    ), patch(
        "public_runner.public_checkpoint_info",
        return_value={"path": "fake.ckpt", "exists": True, "size_bytes": 1, "sha256": "abc"},
    ), patch("public_runner._run_public_predict", side_effect=fake_predict) as predict, patch(
        "public_runner.convert_cif_to_pdb", side_effect=fake_convert
    ), patch("public_runner.score_structure", side_effect=fake_score):
        run_replicate("resume_test", 0, resume=False)
        assert predict.call_count == 100
        predict.reset_mock()
        run_replicate("resume_test", 0, resume=True)
        predict.assert_not_called()


def test_public_predict_reuses_completed_retry_directory(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample_0000"
    prediction = (
        sample_dir
        / "boltz_retry1"
        / "boltz_results_input"
        / "predictions"
        / "input"
        / "input_model_0.cif"
    )
    prediction.parent.mkdir(parents=True)
    prediction.write_text("data_input\n", encoding="utf-8")
    with patch("public_runner.subprocess.run") as run_process:
        assert public_runner._run_public_predict(sample_dir, 123) == prediction
    run_process.assert_not_called()


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

    config = yaml.safe_load((ROOT / "configs" / "1cll.yaml").read_text(encoding="utf-8"))
    assert config["boltz2"]["step_scale"] == 1.0
    assert config["boltz2"]["inference_precision"] == "auto"
    assert config["output_layout"] == "target_budget_method"
    assert config["seed"] == 0
    assert config["seed_step"] == 1

    # The stock baseline omits --step_scale, so public Boltz-2 keeps 1.5;
    # deterministic O3 explicitly uses the unit PF-ODE Euler step above.
    from boltz.main import Boltz2DiffusionParams

    assert Boltz2DiffusionParams().step_scale == 1.5


if __name__ == "__main__":
    test_single_sequence_setup()
    test_replicates_use_disjoint_notebook_seed_blocks()
    test_shared_replicate_seed_schedule_is_reproducible()
    test_random_replicate_seeds_are_unique_valid_31_bit_values()
    with TemporaryDirectory(prefix="k10_n100_test_") as temp:
        test_public_runner_resumes_without_regenerating(Path(temp))
    test_small_budget_configs_are_distinct()
    print("bundle tests passed")
