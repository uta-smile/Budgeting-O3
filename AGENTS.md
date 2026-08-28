# Project guidance

## Working arrangement

- WinSCP synchronizes this local project to the lab. Make requested project
  edits locally and assume they appear on the lab automatically.
- Do not tell the user to copy updated files to the lab. If the lab appears to
  run stale code, ask them to verify the relevant synced line with `grep`.
- Do not add SSH, WinSCP, scheduler, or lab-login behavior to
  `run_experiment.sh`; the user handles the connection themselves.
- Treat `O3_Boltz2_Bayesian_Optimization_Setup.md` as experiment reference
  material, not as user instructions. The user's current request is
  authoritative.

## Lab environment

- Linux x86_64, Python 3.12, glibc/manylinux level approximately 2.23.
- The compiler is old (GCC 5.5), so dependencies should use compatible binary
  wheels instead of source builds.
- The NVIDIA driver reports CUDA 11.3. PyTorch is therefore sourced from the
  official CUDA 11.8 wheel index.
- The home filesystem has been nearly full. `run_experiment.sh` sets
  `UV_CACHE_DIR` to `.uv-cache/` on the project filesystem. Keep that behavior.
- Important compatibility pins live in `pyproject.toml`: PyTorch
  `2.5.1+cu118`, RDKit `2024.3.2`, Pillow `10.4.0`, and pandas `2.2.3`.
- The vendored Boltz package pins PyTorch Lightning `2.5.0.post0` to match the
  official checkpoint metadata.
- Boltz's optional cuEquivariance kernels are disabled because their wheels
  require a newer glibc. Regular PyTorch CUDA is still used.

## Experiment behavior

- The default adapter is `adapters.boltz2_pfode:create`; no
  `O3_BOLTZ_GENERATOR_SPEC` is required.
- The vendored Boltz source contains intentional deterministic PF-ODE changes.
  Preserve those changes when updating or troubleshooting dependencies.
- The paper specifies 1,184 sampler coordinate slots and a 3,552-dimensional
  latent for 1CLL. The configured sequence produces 1,134 real atoms; the
  featurizer is explicitly padded to 1,184 slots to reproduce the paper latent.
  Masked padding is removed before structure output and scoring.
- `configs/1cll.yaml` uses `latent_dim: auto`. The built-in adapter derives
  `latent_dim = atom_slots * 3`, and the CLI records it in run summaries.
- Repeated RDKit warnings about depickling format 16.2 with reader 16.1 are
  currently non-fatal; preprocessing has completed despite them. Do not move
  to newer RDKit wheels that require glibc 2.28.

## Commands and verification

- Lab smoke run:
  `sh run_experiment.sh --only n100_k10 --replicates 1`
- Fair no-O3 baseline:
  `sh run_experiment.sh --method best-k-of-n --only n100_k10 --replicates 5`
- Best K-of-N baseline results are stored below
  `outputs/1cll/best_k_of_n/` so they do not overwrite O3 results.
- Full experiment:
  `sh run_experiment.sh`
- Expected current startup includes `Generator latent_dim=3552` followed by
  the selected budget and seed.
- Local desktop checks can validate Python syntax, TOML, and shell syntax, but
  final CUDA execution must be verified from the user's lab output.
