# O3 + Boltz-2 experiment

This project contains the budgeted O3 Bayesian-optimization loop described in
the attached experiment notes for the 1CLL calmodulin benchmark. It runs all
six `(N, K)` settings and five random replicates by default.

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
statistics of linear combinations of independently sampled seed latents.

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

The default 1CLL configuration has MSA retrieval enabled. Because the input
uses `msa: empty`, the configured Boltz-2 adapter requests an MSA from
`https://api.colabfold.com` during preprocessing. Therefore, the command
above runs with MSA enabled by default:

```bash
sh run_experiment.sh --only n100_k10 --replicates 5
```

This can improve structural accuracy, but makes preprocessing dependent on
network access and server responses. If an old offline preprocessing cache
already exists, use a fresh `processed_dir` or regenerate that cache so the
MSA is fetched. Set `boltz2.use_msa_server: false` in `configs/1cll.yaml`
if the experiment protocol requires offline, reproducible features.

For a paper-faithful comparison, O3 uses deterministic Boltz-2 PF-ODE sampling, while Best K-of-N uses the ordinary stochastic Boltz-2 sampler. Each run summary records this as generator_sampling. Using deterministic sampling for both methods is an ablation, not the paper baseline.

The default reference structure is downloaded to data/1CLL.pdb on the first
run. New outputs are stored without overwriting prior sweeps:

```text
outputs/1cll/<method>/<budget>/runs/<run_id>/
  seed_0000/{evaluations.csv, summary.json, ...}
  sweep_summary.csv
  run_metadata.json
```

The default run_id is a timestamp; provide --run-id to choose a label.

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

Run the fair no-O3 Best K-of-N baseline using the same Boltz-2 generator and
TM-score oracle. This generates N random latent samples and returns the top K:

```bash
sh run_experiment.sh --method best-k-of-n --only n100_k10 --replicates 5
```

Baseline results use outputs/1cll/best_k_of_n/<budget>/runs/<run_id>/; O3 results use
outputs/1cll/o3/<budget>/runs/<run_id>/.

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
