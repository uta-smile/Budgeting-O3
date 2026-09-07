# 1CLL Best K-of-N vs O3

This repository reproduces the 1CLL TM-score experiment from [How to Spend
Your Oracle Budget](https://arxiv.org/abs/2608.12192) for the three hardware-
feasible budgets through `(N=100, K=10)`:

1. Best K-of-N with the official stochastic `boltz==2.2.1` sampler.
2. O3 with the custom deterministic Boltz-2 probability-flow ODE decoder.
3. Random PF-ODE, an optional same-decoder diagnostic without Bayesian
   optimization.

## Run

From the repository root, one command installs the environments, downloads
1CLL if needed, and runs the comparison:

```bash
sh run_experiment.sh --method both --only n100_k10 --replicates 5 --random-seeds --run-id n100_run01
```

With no arguments, `sh run_experiment.sh` runs Best K-of-N and O3 for all
three supported budgets, each with five fresh random replicate seeds.

The shell wrapper keeps uv and Boltz caches on the project filesystem and
defaults the public baseline to FP32 for the lab GTX Titan. Supported methods
are `best-k-of-n`, `o3`, `both`, `random-pfode`, and `all`. Supported budgets
are `n20_k2`, `n50_k5`, and `n100_k10`.

Use a one-replicate smoke run before the full experiment:

```bash
sh run_experiment.sh --method both --only n100_k10 --replicates 1 --random-seeds --run-id smoke01
```

For the setup/GPU checks without an experiment:

```bash
sh run_experiment.sh --smoke
```

## Seeds and resume

Fresh unique replicate seeds are the default; `--random-seeds` states that
choice explicitly. The runner prints and records them in method metadata.
Reproduce them with `--seed-list`:

```bash
sh run_experiment.sh --method both --only n100_k10 --replicates 5 --seed-list SEED1 SEED2 SEED3 SEED4 SEED5 --run-id n100_repeat01
```

To pair a new O3 run with an existing completed baseline, load its recorded
seed list directly:

```bash
sh run_experiment.sh --method o3 --only n100_k10 --replicates 5 --seeds-from-baseline-run n100_run01
```

This reuses the replicate seeds, not the generated baseline structures. That
distinction is required by the paper: Best K-of-N uses stochastic Boltz-2,
while O3 phase 1 must decode explicit Gaussian latents with deterministic
PF-ODE. Feeding stochastic baseline samples into O3 would change the method.

Resume an interrupted run with the same run ID and seed source:

```bash
sh run_experiment.sh --method o3 --only n100_k10 --replicates 5 --seeds-from-baseline-run n100_run01 --resume
```

O3 validates saved phase-1 latents against the recorded seed before reusing
them and resumes from the contiguous saved PDB/latent prefix. Best K-of-N also
skips completed samples with `--resume`. When a run ID already has a manifest,
`--resume --run-id ID` automatically restores its seed list; an incompatible
explicit seed list is rejected instead of overwriting the manifest.

The seed list is written before generation to
`outputs/1cll/kK_nN/runs/<run-id>/run_manifest.json`, so even an early failure
does not lose the randomized seeds.

## Protocol invariants

- Both methods consume the same checked-in Boltz YAML,
  `data/1cll_boltz_input.yaml`: chain A, the 144-residue observed 1CLL
  construct, and `msa: empty` (no MSA server).
- The reference is `data/1CLL.pdb`, chain A. Generated structures are written
  as PDB and ranked by TM-align normalized by the 144-residue reference chain.
- Best K-of-N invokes the isolated official package without `--step_scale`, so
  Boltz-2 retains its stochastic `step_scale=1.5` and `gamma_0=0.8` defaults.
- O3 uses explicit `z ~ N(0,I)`, disables churn and stochastic SE(3)
  augmentation, and uses `step_scale=1.0` for the PF-ODE Euler update.
- The 1CLL generator tensor is padded to the paper's 1,184 coordinate slots,
  giving `D=3,552`; the 1,134 real generated atom slots are unpadded before
  PDB writing and scoring.
- O3 spends exactly `M + 2 + (N-M-2) = N` oracle calls. At N=100 it uses
  `M=50`, `d=5`, two random U points, and 48 BO rounds. Results report both
  mean-of-K and max-of-K over the top K TM-scores.

Results are grouped consistently by target, budget, and method:

```text
outputs/1cll/k10_n100/best_k_of_n/runs/<run-id>/
outputs/1cll/k10_n100/o3/runs/<run-id>/
outputs/1cll/k10_n100/random_pfode/runs/<run-id>/
```

See [the protocol audit](docs/protocol_audit.md) for paper-to-code evidence and
explicit ambiguities, and [the experiment bundle](experiments/1cll/k10_n100/README.md)
for artifact details and verification commands.
