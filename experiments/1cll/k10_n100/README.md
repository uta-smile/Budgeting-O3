# Canonical 1CLL experiment bundle

This folder contains the only runner needed for the comparison:

- `best-k-of-n`: official stochastic Boltz-2 baseline
- `o3`: deterministic PF-ODE with `U`-space Bayesian optimization
- `random-pfode`: deterministic PF-ODE with random `Z` samples and no BO

## Run it

From the repository root:

```bash
sh run_experiment.sh --smoke
sh run_experiment.sh --method all --only n100_k10 --replicates 10 --batch-size 4 --gpus auto --random-seeds --run-id batch4_all_run01
```

The shell wrapper applies the Linux lab defaults automatically. Override them
only when using a different validated cache or GPU:

```bash
cd "/mnt/SSD3/Chase/budgetting O3"
export BOLTZ_PUBLIC_PRECISION=32
export BOLTZ_CACHE="$PWD/.boltz"
export UV_CACHE_DIR="$PWD/.uv-cache"
```

The wrapper defaults to these same values, so no export is required for the
normal lab run.

The first command checks the setup and GPU. The second one-line command runs
all three methods with the same ten freshly generated seeds, using one Linux
process per visible GPU. The paper uses five seeds; ten is an optional
higher-confidence repeat count.

Running `sh run_experiment.sh` with no arguments executes Best K-of-N and O3
for n20, n50, and n100 with five fresh seeds per budget.

Supported replicate counts are 1, 3, 5, and 10. Use five to match the paper;
ten doubles the repeat count and compute cost.

`--gpus auto` discovers visible devices and assigns replicate seeds to them in
stable round-robin order. For ten seeds on four GPUs, the workers receive
3, 3, 2, and 2 seeds. Use `--gpus 0,1,2,3` to choose devices explicitly, or
omit the option for the original sequential path. The same worker handles its
seed shard for every selected method, and `--batch-size` applies separately on
each GPU. Multi-worker execution is Linux-only; Windows rejects more than one
selected GPU. This is replicate-level process parallelism, not a change to
Boltz's DataLoader worker count.

Worker logs, completion markers, and `schedule.json` are stored below
`outputs/1cll/kK_nN/runs/<run-id>/workers/`. Resume with the same arguments and
`--resume`; the saved manifest restores the seed list, completed replicates
are reused, and a different GPU subset may be selected for the resumed session.

Available budgets:

```text
n20_k2    N=20,  K=2,  O3: M=10, d=5,  8 BO rounds
n50_k5    N=50,  K=5,  O3: M=25, d=7, 23 BO rounds
n100_k10  N=100, K=10, O3: M=50, d=5, 48 BO rounds
```

Change only `--budget` and `--run-id` for another experiment. Use
`--method both` for Best K-of-N plus O3, or `--method random-pfode` for only
the random diagnostic.

Both backends read the same checked-in Boltz input at
`data/1cll_boltz_input.yaml`. The O3 adapter reads its remaining settings from
the shared root configuration at `configs/1cll.yaml`.
Its `inference_precision: auto` setting selects BF16 on capable GPUs and
FP32 on devices such as the GTX Titan. Leave this default in place on the lab
node, or set `O3_INFERENCE_PRECISION=32` explicitly.

The wrapper defaults `BOLTZ_CACHE` to `$PWD/.boltz`. The project-local
checkpoint has been validated as a PyTorch archive.

If several checkouts share one lab installation, set `BOLTZ_PUBLIC_PROJECT`
to the shared public-Boltz project and `BOLTZ_PUBLIC_CACHE` to its shared model
cache; otherwise the bundle-local defaults are used.

Best-K-of-N uses `BOLTZ_PUBLIC_PRECISION`; on the GTX Titan keep it set to
`32` so it runs in FP32 from the start. If it is unset, the official CLI is
attempted in FP16 first and may need an FP32 retry after a CUDA SVD failure.

## Seeds

Fresh random seeds are the default; `--random-seeds` makes that choice
explicit. The launcher prints the unique list and records it in every
method's metadata. To reproduce a run, use the logged values explicitly:

```bash
sh run_experiment.sh --only n20_k2 --method all --replicates 5 --seed-list SEED1 SEED2 SEED3 SEED4 SEED5 --run-id n20_k2_repeat01
```

For an interrupted O3 run, reuse the same run ID and seed list with
`--resume`. Completed seeds are skipped and partial seeds continue from their
saved phase-1/BO outputs, after which the run-level `aggregate.csv` is
regenerated:

```bash
sh run_experiment.sh --only n50_k5 --method o3 --replicates 5 --run-id n50_k5_run01 --resume --seed-list SEED1 SEED2 SEED3 SEED4 SEED5
```

Alternatively, `--seeds-from-baseline-run RUN_ID` reads the saved seed list
from a completed Best K-of-N run. Only the seeds are shared: stochastic
baseline structures cannot be reused as deterministic O3 phase-1 outputs.

The fixed arithmetic schedule is available with `--fixed-seed-schedule`,
`--seed-start`, and `--seed-step`.

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
The public command omits `--step_scale`, retaining stochastic Boltz-2's 1.5
default; deterministic O3 uses the configured unit PF-ODE step.

The primary comparison metric is `mean_of_K`, the mean TM-score of the K
returned structures. `max_of_K` is the secondary best-found metric. Aggregate
reports do not score methods by the mean of all N generated structures.
