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

Set `--replicates 3` for a three-replicate comparison; supported values are 1,
3, and 5.

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

Both methods use single-sequence conditioning: Boltz's official `msa: empty`
marker is supplied, no MSA is retrieved, and no MSA server is contacted.
Best-K-of-N uses the stock public Boltz-2 package in its
isolated `public_boltz/` environment. Its only compatibility switch is the
same `--no_kernels` flag used by the reference notebook. O3 and O3-random use
the custom adapter in `adapters/boltz2_pfode.py` and the vendored Boltz source.

The primary comparison metric is `mean_of_K`, the mean TM-score of the K
returned structures. `max_of_K` is the secondary best-found metric. Aggregate
reports do not score methods by the mean of all N generated structures.
