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

## On the lab GPU node

Open the WinSCP-synchronised project on the node, activate the lab's normal
CUDA environment if needed, then execute:

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

The default input uses `msa: empty`, so preprocessing does not call an MSA
server. This makes the feature tensor reproducible, though a supplied MSA may
improve structural accuracy. Change `data/1cll_boltz_input.yaml` only if the
experiment protocol requires a particular MSA.

The default reference structure is downloaded to `data/1CLL.pdb` on the first
run. Outputs are written below `outputs/1cll/`, including per-run
`evaluations.csv`, generated PDB files, `returned_candidates.json`, `summary.json`, and the overall
`sweep_summary.csv`.

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

Baseline results are written below `outputs/1cll/best_k_of_n/` so they do not
overwrite O3 results.

The shell script does not SSH, submit jobs, or configure the lab scheduler.
It places UV’s large package cache under `.uv-cache/` beside the project so
CUDA dependencies do not consume a small home-directory quota.

The project pins RDKit 2024.3.2, Pillow 10.4.0, and pandas 2.2.3 so UV can
use compatible binary wheels on the lab node’s glibc 2.23 system. PyTorch is
pinned to its official CUDA 11.8 build because the lab driver reports CUDA
11.3. The optional cuEquivariance kernels are disabled because their wheels
also require newer glibc, but the model still uses the GPU through regular
PyTorch CUDA.
The vendored package uses PyTorch Lightning 2.5.0.post0, matching the official
Boltz-2 checkpoint.
