"""Compare saved O3 latents at batch size one and a requested batch size."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from tmtools.io import get_residue_data

from common import REPO_ROOT
from o3_boltz.adapter import load_adapter
from o3_boltz.tmscore import _load_structure, TMScoreOracle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 2 or args.limit < 2:
        parser.error("batch size and limit must both be at least two")
    paths = sorted(args.latent_dir.glob("latent_*.npy"))[:args.limit]
    if len(paths) < 2:
        parser.error("latent directory must contain at least two saved latent_*.npy files")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((REPO_ROOT / "configs" / "1cll.yaml").read_text())
    config["project_root"] = str(REPO_ROOT)
    config["boltz2"].setdefault("cache_dir", str(REPO_ROOT / ".boltz"))
    adapter = load_adapter("adapters.boltz2_pfode:create", config)
    config["latent_dim"] = adapter.latent_dim
    latents = np.stack([np.load(path) for path in paths])
    outputs = {}
    timings = {}
    for label, size in (("single", 1), ("batch", args.batch_size)):
        outputs[label] = [args.output_dir / label / f"sample_{i:04d}.pdb" for i in range(len(paths))]
        tick = time.perf_counter()
        for start in range(0, len(paths), size):
            end = min(start + size, len(paths))
            adapter.generate_batch(latents[start:end], outputs[label][start:end], config,
                                   [{"deterministic": True}] * (end - start))
        timings[label] = time.perf_counter() - tick
    rows = []
    for index, latent_path in enumerate(paths):
        single, batch = outputs["single"][index], outputs["batch"][index]
        a = get_residue_data(TMScoreOracle._select_chain(_load_structure(single), "A"))[0]
        b = get_residue_data(TMScoreOracle._select_chain(_load_structure(batch), "A"))[0]
        rows.append({"latent": str(latent_path), "single_tm_score": adapter.score(single, config),
                     "batch_tm_score": adapter.score(batch, config),
                     "ca_coordinate_max_abs_delta_angstrom": float(np.max(np.abs(a - b)))})
    report = {
        "batch_size": args.batch_size, "samples": rows, "generation_seconds": timings,
        "precision": adapter.inference_precision, "atom_slots": adapter.atom_slots,
        "max_tm_score_delta": max(abs(row["single_tm_score"] - row["batch_tm_score"]) for row in rows),
        "single_ranking": sorted(range(len(rows)), key=lambda i: rows[i]["single_tm_score"], reverse=True),
        "batch_ranking": sorted(range(len(rows)), key=lambda i: rows[i]["batch_tm_score"], reverse=True),
        "note": "Coordinates are PDB-rounded C-alpha coordinates, without alignment. Timing includes warm-up; use run summaries for performance comparisons.",
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
