"""Audit saved O3 trajectories; optionally decode seed round trips on a GPU.

GPU mode performs additional diagnostic evaluations outside the experiment
budget. It never modifies the original run or its scores.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from common import REPO_ROOT
from o3_boltz.chart import SurrogateChart
from o3_boltz.tmscore import TMScoreOracle, _load_structure
from tmtools.io import get_residue_data


def coordinates(path):
    return get_residue_data(TMScoreOracle._select_chain(_load_structure(path), "A"))[0]


def rmsd(a, b):
    if a.shape != b.shape:
        raise ValueError("Cannot compare structures with different C-alpha counts")
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, v = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ v)
    return float(np.sqrt(np.mean(np.sum((a @ u @ correction @ v - b) ** 2, axis=1))))


def local_path(root, record, field):
    # Relocate saved absolute paths using their path relative to this replicate.
    original = Path(record[field])
    index = original.parts.index(root.name)
    return root.joinpath(*original.parts[index + 1:])


def audit(root, oracle):
    records = json.loads((root / "evaluations.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    with np.load(root / "phase1_latents.npz") as archive:
        seeds = archive["selected_latents"].copy()
    chart = SurrogateChart(seeds)
    recovered = chart.from_u_to_z(chart.from_z_to_u(seeds))
    top = sorted(records, key=lambda row: row["score"], reverse=True)[:summary["K"]]
    xyz = [coordinates(local_path(root, row, "structure")) for row in top]
    distances = [rmsd(a, b) for a, b in itertools.combinations(xyz, 2)]
    bo = [row for row in records if row["stage"] == "bo_acquisition"]
    points = np.array([row["u"] for row in bo])
    latents = np.array([np.load(local_path(root, row, "latent_file")) for row in bo])
    errors = [float(np.max(np.abs(chart.from_u_to_z(row["u"]) -
              np.load(local_path(root, row, "latent_file"))))) for row in records if row["u"] is not None]
    return {
        "seed": summary["seed"], "evaluations": len(records),
        "stages": {stage: sum(row["stage"] == stage for row in records)
                   for stage in sorted({row["stage"] for row in records})},
        "seed_roundtrip_max_relative_l2": float(np.max(np.linalg.norm(recovered-seeds, axis=1)/np.linalg.norm(seeds, axis=1))),
        "saved_latent_max_abs_error": max(errors, default=0),
        "bo_count": len(bo), "bo_unique_u": len(np.unique(points, axis=0)),
        "bo_unique_z": len(np.unique(latents, axis=0)),
        "top_k_score_range": top[0]["score"]-top[-1]["score"],
        "top_k_ca_rmsd_median_angstrom": float(np.median(distances)) if distances else None,
        "top_k_pairs_below_0_01_angstrom": sum(d < 0.01 for d in distances),
        "top_k_pair_count": len(distances),
        "rescored_max_error": max(abs(oracle.score(local_path(root, row, "structure"), "A")-row["score"]) for row in top),
        "top_k_max_seed_weights": [float(chart.from_u_to_w(row["u"]).max())
                                   if row["u"] is not None else 1.0 for row in top],
    }


def gpu_check(root, output):
    import yaml
    from o3_boltz.adapter import load_adapter
    config = yaml.safe_load((REPO_ROOT / "configs/1cll.yaml").read_text())
    config["project_root"] = str(REPO_ROOT)
    config["boltz2"].setdefault("cache_dir", str(REPO_ROOT / ".boltz"))
    adapter = load_adapter("adapters.boltz2_pfode:create", config)
    config["latent_dim"] = adapter.latent_dim
    with np.load(root / "phase1_latents.npz") as archive:
        seeds, scores = archive["selected_latents"].copy(), archive["selected_scores"].copy()
    chart = SurrogateChart(seeds)
    reconstructed = chart.from_u_to_z(chart.from_z_to_u(seeds))
    results = []
    for index, (seed, mapped) in enumerate(zip(seeds, reconstructed)):
        paths, values = [], []
        for label, latent in (("original", seed), ("repeat", seed), ("roundtrip", mapped)):
            path = output / f"seed_{index}_{label}.pdb"
            adapter.generate(latent, path, config, {"deterministic": True})
            paths.append(path)
            values.append(float(adapter.score(path, config)))
        row = {"index": index, "saved_score": float(scores[index]),
               "original_score": values[0], "repeat_score": values[1], "roundtrip_score": values[2],
               "repeat_ca_rmsd": rmsd(coordinates(paths[0]), coordinates(paths[1])),
               "roundtrip_ca_rmsd": rmsd(coordinates(paths[0]), coordinates(paths[2]))}
        results.append(row)
        (output / "gpu_check.json").write_text(json.dumps({"precision": adapter.inference_precision,
            "step_scale": adapter.step_scale, "samples": results,
            "note": "Sequential current-decoder checks; historical batch/precision differences may affect saved-score comparisons."}, indent=2))
        print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="O3 method run directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="New diagnostic directory")
    parser.add_argument("--gpu-seed", type=int, help="Additionally decode original/repeat/roundtrip for all selected latents of this replicate")
    args = parser.parse_args()
    roots = sorted(p.parent for p in args.run_dir.glob("seed_*/summary.json"))
    if not roots:
        parser.error("No completed O3 replicates found")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    oracle = TMScoreOracle(REPO_ROOT / "data/1CLL.pdb", "A")
    results = [audit(root, oracle) for root in roots]
    (args.output_dir / "audit.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    if args.gpu_seed is not None:
        root = args.run_dir / f"seed_{args.gpu_seed:04d}"
        gpu_check(root, args.output_dir)


if __name__ == "__main__":
    main()
