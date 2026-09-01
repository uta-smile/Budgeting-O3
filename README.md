# O3 + Boltz-2 experiment

This project contains the budgeted O3 Bayesian-optimization loop described in
the attached experiment notes for the 1CLL calmodulin benchmark. It runs all
six `(N, K)` settings and five random replicates by default.

The canonical 1CLL `N=100, K=10` comparison is under
`experiments/1cll/k10_n100/`; its legacy-output labels are in
`experiments/1cll/k10_n100/legacy_outputs.md`.

The repository includes a small research fork of the official Boltz-2 source.
Its built-in adapter derives the latent dimension from the preprocessed atom
tensor, uses that latent as the initial Gaussian coordinate state, then runs
an opt-in deterministic, noise-free probability-flow sampler. The current
1CLL input has 1,134 real atoms explicitly padded to the paper's 1,184 sampler
slots and therefore a 3,552-value latent. Padding is removed before structures are written and
scored. The ordinary public Boltz command remains available unchanged.

The paper describes 1CLL as 1,184 atom-coordinate slots. Because the sequence
itself produces 1,134 real Boltz atoms, the remaining positions are masked
padding in the model input and are excluded from the written structure and
TM-score. The runner records the generator-derived `latent_dim` in every run
summary.

The O3 chart uses the paper's measure-preserving Knothe--Rosenblatt transform:
inverse-Beta stick breaking followed by square roots, producing non-negative
unit-norm weights on the positive hypersphere. This preserves the unit-Gaussian
statistics of linear combinations of independently sampled seed latents. The
implementation matches the public `latent-spaces-by-example` convention: it
reflects `u` to `1-u`, clips inverse-Beta inputs to `[1e-15, 1-1e-15]`, and
supports the forward/inverse API `from_u_to_w`, `from_w_to_u`, `from_u_to_z`,
and `from_z_to_u`. O3 projects the selected seed latents into `u` with the
inverse chart and reuses their already-computed scores.

## Paper-figure reproducibility note

The paper defines Best K-of-N as selecting the K highest external-oracle
TM-scores and reports the mean of those returned structures. That is the
default implemented here (`selection_metric: oracle_tm_score`). A direct
Boltz-2 v2.2.1 control on the local 1CLL input produced a top-10 mean of
0.7760 with cached MSA and 0.7938 in single-sequence mode, consistent with the
current adapter. In the same cached-MSA control, selecting by Boltz's predicted
`ptm` instead produced 0.5809, while averaging all 100 oracle scores produced
0.5548. The public paper does not specify either alternative or release the
benchmark code, so these are diagnostic alternatives and are not substituted
for the stated oracle-ranking algorithm.

## On Windows with Git Bash

Open Git Bash in the project directory, make sure `uv` and an NVIDIA CUDA
driver are available, then execute:

```bash
sh run_experiment.sh
```

On the first run, the script installs the environment, downloads the official
Boltz-2 weights/chemical data and the 1CLL reference, preprocesses the
single-sequence input, and caches those files. The full sweep is large; first
check the setup with:

```bash
sh run_experiment.sh --only n100_k10 --replicates 1
```

The default 1CLL configuration uses Boltz's explicit single-sequence mode
(`msa: empty`). MSA retrieval is not specified in the paper and is not part
of either the O3 or Best K-of-N algorithm, so it is left out of the main
comparison. To test MSA as a separate model-input ablation, change the input
to `msa: 0` and set `boltz2.use_msa_server: true`, using a separate
`processed_dir`.

This can improve structural accuracy, but makes preprocessing dependent on
network access and server responses. If an old offline preprocessing cache
already exists, use a fresh `processed_dir` or regenerate that cache so the
new feature set is actually used. Do not mix MSA and single-sequence caches.
For a paper-faithful comparison, O3 uses deterministic Boltz-2 PF-ODE sampling, while Best K-of-N uses ordinary stochastic Boltz-2 sampling. Both paths receive an explicit $z\sim\mathcal{N}(0,I)$ latent; O3 sets EDM churn `gamma_0=0`, while the stochastic baseline uses the upstream `gamma_0=0.8` and random SE(3) augmentation. Each run summary records these choices as `generator_sampling`, `explicit_latent`, and `stochastic_gamma_0`. Using deterministic sampling for both methods is an ablation, not the paper baseline.

The adapter also matches the official Boltz-2 inference precision (`bf16-mixed`)
and `torch.set_float32_matmul_precision("highest")`; set
`boltz2.inference_precision: 32` only for an explicitly labelled precision
ablation.

The configured main-text benchmark uses `(N, K)` values `(20, 2)`, `(50, 5)`,
`(100, 10)`, `(200, 20)`, `(500, 50)`, and `(1000, 100)`. O3 spends exactly
`M + 2 + (N - M - 2) = N` oracle calls; the selected `d` seed scores are
reused rather than rescored. The appendix table in the paper contains
inconsistent K values for some rows, so those values are not silently mixed
into the main benchmark configuration.

The default reference structure is downloaded to data/1CLL.pdb on the first
run. New outputs are stored without overwriting prior sweeps:

```text
outputs/1cll/<method>/<budget>/runs/<run_id>/
  seed_0000/{evaluations.csv, summary.json, ...}
  sweep_summary.csv
  run_metadata.json
```

The default run_id is a timestamp; provide --run-id to choose a label.

Seeds are configurable. Set `seed` in `configs/1cll.yaml`, or override it at
runtime with `--seed-start`. By default, five replicates starting at 0 use
0, 1, 2, 3, and 4:

```bash
sh run_experiment.sh --seed-start 12345 --replicates 5
```

Use `--seed-step` (or `seed_step` in the config) when you want a different
spacing, for example `--seed-start 100 --seed-step 10` produces 100, 110, 120,
and so on. The selected seeds are recorded in `run_metadata.json`.

## Optional adapter override

If the lab has its own compatible generator, select the legacy bridge and set
its `module:function`:

```bash
export O3_ADAPTER_SPEC=adapters.boltz_pfode_adapter:create
export O3_BOLTZ_GENERATOR_SPEC=my_boltz_adapter:generate
sh run_experiment.sh
```

Alternatively, set `O3_ADAPTER_SPEC` to a complete adapter factory. The
factory must return an object with `generate()` and `score()` methods, as
implemented in `adapters/boltz_pfode_adapter.py`.

## Useful overrides

Run only one budget, or use one replicate for a smoke test:

```bash
sh run_experiment.sh --only n100_k10 --replicates 1
```

Run the fair no-O3 Best K-of-N baseline through the isolated public Boltz-2
experiment bundle. This keeps the custom vendored Boltz implementation out of
the baseline:

```bash
python experiments/1cll/k10_n100/run.py --method best-k-of-n --replicates 5
```

The same runner supports the smaller paper budgets with distinct output
folders:

```bash
python experiments/1cll/k10_n100/run.py --budget n20_k2 --method both --replicates 5
python experiments/1cll/k10_n100/run.py --budget n50_k5 --method both --replicates 5
```

The root `run_experiment.sh` entry point is reserved for O3. The diagnostic
model-pTM ranking remains available only in the legacy baseline module and is
not part of the canonical K=10, N=100 comparison. The canonical bundle writes
new results under `outputs/1cll/k10_n100/<method>/runs/<run_id>/`; older output
paths remain legacy diagnostics and are not reused.

Legacy pre-refactor output locations were `outputs/1cll/best_k_of_n/` and
`outputs/1cll/o3/`; they are retained only as diagnostics.

The shell script does not SSH, submit jobs, or configure the lab scheduler.
It places UV’s large package cache under `.uv-cache/` beside the project so
CUDA dependencies do not consume a small home-directory quota.

The project uses RDKit >=2025.03.1 on Windows and RDKit 2024.3.2 on Linux, plus Pillow 10.4.0 and pandas 2.2.3, so UV can
use compatible binary wheels on the lab node’s glibc 2.23 system. PyTorch is selected by platform: Windows uses the official CUDA 12.8 build
(PyTorch 2.7.0) for RTX 50-series/Blackwell support, while Linux retains the
lab node's CUDA 11.8 build (PyTorch 2.5.1). The optional cuEquivariance kernels
are disabled, but the model still uses the GPU through regular PyTorch CUDA.
The vendored package uses PyTorch Lightning 2.5.0.post0, matching the official
Boltz-2 checkpoint.
