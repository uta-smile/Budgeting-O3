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
sh run_experiment.sh --method all --only n100_k10 --replicates 10 --batch-size 4 --gpus auto --random-seeds --run-id batch4_all_run01
```

This one-line command runs stochastic Best K-of-N, O3, and Random PF-ODE with
the same ten fresh replicate seeds, distributed across every visible GPU on
Linux. The paper reports five seeds; ten is an optional higher-confidence
repeat count and doubles the experiment cost.

With no arguments, `sh run_experiment.sh` runs Best K-of-N and O3 for all
three supported budgets, each with five fresh random replicate seeds.

To run the complete experiment directly:

```bash
sh run_experiment.sh
```

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

## Faster inference with batching

The command above runs all three methods for `n100_k10` with ten replicates and
batch size 4. To run only the two primary paper-comparison methods with five
replicates:

```bash
sh run_experiment.sh --method both --only n100_k10 --replicates 5 --batch-size 4 --random-seeds --run-id batch4_run02
```

Start with `--batch-size 2` if GPU memory is limited. The public baseline
loads its model once per replicate and generates bounded batches, including
a smaller final batch when N is not divisible by the batch size. O3 batches
its M initial latent vectors and its two initial U points; every adaptive BO
round still uses one candidate and waits for its score. N, K, sampling steps,
sampler settings, precision selection and TM-score ranking are unchanged.
The optional random-PF-ODE diagnostic remains sequential.

Omitting `--batch-size` preserves the original execution path. Explicit
`--batch-size 1` enables the persistent baseline while keeping inference
sequential. The persistent baseline seeds each batch with
`common.sample_seed(run_seed, batch_start_index)`. It does not reproduce
the old per-sample random stream, even at size 1, because seeding happens
immediately before prediction. Batch seeds, counts, and within-batch
confidence ranks are recorded; all N structures are ranked together by
external TM-score. Public confidence sidecars and pLDDT B-factors are omitted
in this path to avoid the existing single-chain sidecar failure. The public
model and stochastic sampler remain unmodified.

Use the same batch-size flag when resuming. Completed baseline batches are
reused; an incomplete batch is regenerated with its original seed and size.
Changing execution mode or batch size within a run is rejected. O3 likewise
replays an interrupted initialization batch at its original size. Numerical
batching differences can alter O3 seed selection and later BO decisions.

Validate saved O3 latents on the lab GPU before running the full comparison:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run python experiments/1cll/k10_n100/validate_batching.py \
  --latent-dir /path/to/o3/replicate/latents \
  --output-dir outputs/batch_validation01 --batch-size 4 --limit 8
```

The output directory must be new. `comparison.json` reports per-latent
TM-scores, C-alpha coordinate differences, rankings, and generation times
for sizes 1 and 4. Increase `--limit` to M to check the full seed ranking.
This is a numerical check, not proof of identical experiment trajectories.

The comparison folder also contains `execution_timings_this_session.json`
with per-method wall time including model startup. Run summaries record
batch settings and timing for the current invocation;
O3 separates generation, scoring, and CPU acquisition time. Baseline batch
completion files record generation time, and `boltz_batches.log` contains
public inference output. Resume timings exclude work from earlier sessions.
Compare end-to-end times on the same GPU and precision; batching may improve
throughput without eliminating O3's CPU acquisition pauses. Final CUDA
correctness and speed still require the lab run.

## Multiple GPUs on Linux

Add `--gpus auto` to run one isolated replicate worker per visible GPU. Each
worker sets `CUDA_VISIBLE_DEVICES` to one physical GPU, where Boltz sees that
device as `cuda:0`. Replicate seeds are assigned in stable round-robin order;
for ten seeds and four GPUs, the workers receive 3, 3, 2, and 2 seeds. The same
seed stays on the same GPU for Best K-of-N, O3, and Random PF-ODE when using
`--method all`, preserving paired comparisons.

Select a subset explicitly with `--gpus 0,1,2,3`. If there are more selected
GPUs than replicate seeds, only the needed GPUs are started. `--batch-size` is
the per-GPU inference batch size, so each worker has its own model and GPU
memory allocation. The methods run in sequence inside each worker while all
GPU workers run concurrently.

Parallel replicate workers are enabled on Linux. A request for more than one
GPU is rejected on Windows; omit `--gpus` there to retain the sequential
single-worker path. This setting is separate from PyTorch DataLoader workers,
which remain conservative for model input loading.

The launcher prepares the shared public-Boltz cache before starting workers,
and O3 serializes first-run asset extraction and input preprocessing. Worker
logs and the exact seed assignment are saved under
`outputs/1cll/kK_nN/runs/<run-id>/workers/`; the latest mapping is also written
to `gpu_schedule_this_session.json` in the run folder.

Resume an interrupted multi-GPU run with the same run ID:

```bash
sh run_experiment.sh --method all --only n100_k10 --replicates 10 --batch-size 4 --gpus auto --run-id batch4_all_run01 --resume
```

The manifest restores the original seeds. You may change the selected GPU
subset when resuming; completed replicate artifacts are validated and reused.

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
