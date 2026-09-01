# Canonical 1CLL experiment bundle

This folder contains the only runner needed for the comparison:

- `best-k-of-n`: official stochastic Boltz-2 baseline
- `o3`: deterministic PF-ODE with `U`-space Bayesian optimization
- `random-pfode`: deterministic PF-ODE with random `Z` samples and no BO

## Run it

From the repository root:

```bash
.venv\Scripts\python.exe experiments\1cll\k10_n100\verify.py --gpu
.venv\Scripts\python.exe experiments\1cll\k10_n100\run.py --budget n20_k2 --method all --replicates 5 --random-seeds --run-id n20_k2_run01
```

The first command checks the setup and GPU. The second command runs all three
methods with the same five freshly generated seeds.

Available budgets:

```text
n20_k2    N=20,  K=2,  O3: M=10, d=5,  8 BO rounds
n50_k5    N=50,  K=5,  O3: M=25, d=7, 23 BO rounds
n100_k10  N=100, K=10, O3: M=50, d=5, 48 BO rounds
```

Change only `--budget` and `--run-id` for another experiment. Use
`--method both` for Best K-of-N plus O3, or `--method random-pfode` for only
the random diagnostic.

## Seeds

`--random-seeds` generates one unique seed per replicate. The launcher prints
the list and records it in every method's metadata. To reproduce a run, use
the logged values explicitly:

```bash
.venv\Scripts\python.exe experiments\1cll\k10_n100\run.py --budget n20_k2 --method all --replicates 5 --seed-list SEED1 SEED2 SEED3 SEED4 SEED5 --run-id n20_k2_repeat01
```

The default fixed schedule is also available with `--seed-start` and
`--seed-step`.

## Results

```text
outputs/1cll/k2_n20/best_k_of_n/
outputs/1cll/k2_n20/o3/
outputs/1cll/k2_n20/random_pfode/
```

The frozen input MSA is `inputs/1CLL_0.csv`; MSA-server retrieval is disabled.
The public baseline uses its isolated environment in `public_boltz/`. O3 and
O3-random use the custom adapter in `adapters/boltz2_pfode.py` and the
vendored Boltz source.
