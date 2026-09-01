# 1CLL Best K-of-N vs O3

This repository runs a three-way comparison on the 1CLL protein:

1. Best K-of-N with official stochastic Boltz-2
2. O3 Bayesian optimization in the learned `U` subspace
3. O3-random using the same deterministic PF-ODE decoder without BO

The canonical runner is `experiments/1cll/k10_n100/run.py`.

## Quick start

Run these commands from the repository root:

```bash
cd "C:\Smile_lab\budgetting O3\Budgeting-O3"
```

Check the installation and GPU:

```bash
.venv\Scripts\python.exe experiments\1cll\k10_n100\verify.py --gpu
```

Run the complete N20 comparison with five fresh random replicate seeds:

```bash
.venv\Scripts\python.exe experiments\1cll\k10_n100\run.py --budget n20_k2 --method all --replicates 5 --random-seeds --run-id n20_k2_run01
```

This single command runs all three methods with the same five seeds. The seeds
are printed in the log and saved in every method's metadata.

Use these budget names for other experiments:

```text
n20_k2    N=20,  K=2
n50_k5    N=50,  K=5
n100_k10  N=100, K=10
```

Replace `n20_k2` in the command with the desired budget and use a new
`--run-id` for each fresh experiment.

## Repeating an experiment

Copy the five seeds from the previous log or `run_metadata.json`, then use:

```bash
.venv\Scripts\python.exe experiments\1cll\k10_n100\run.py --budget n20_k2 --method all --replicates 5 --seed-list SEED1 SEED2 SEED3 SEED4 SEED5 --run-id n20_k2_repeat01
```

Do not use `--random-seeds` when reproducing a previous run; it generates a
new seed list.

For only Best K-of-N and O3, use `--method both`. For only the same-decoder
random diagnostic, use `--method random-pfode`.

## Results

Results are separated by method and budget:

```text
outputs/1cll/k2_n20/best_k_of_n/
outputs/1cll/k2_n20/o3/
outputs/1cll/k2_n20/random_pfode/
```

Each run contains an aggregate CSV, per-replicate summaries, logs, and seed
metadata. Use a new run ID instead of mixing results from older experiments.

## Important details

- The frozen input MSA is `experiments/1cll/k10_n100/inputs/1CLL_0.csv`.
- MSA-server retrieval is disabled for the canonical comparison.
- O3 uses the custom vendored Boltz-2 adapter and deterministic PF-ODE.
- O3-random uses that same custom decoder but samples all `N` latents randomly.
- Best K-of-N uses the isolated official `boltz==2.2.1` environment.
- For N20, O3 logs and uses `10 random Z -> best 5 seeds -> 2 random U -> 8 BO`.
- Previous outputs and model caches are not required to understand or configure a new run.

See [experiments/1cll/k10_n100/README.md](experiments/1cll/k10_n100/README.md)
for the canonical bundle contents.
