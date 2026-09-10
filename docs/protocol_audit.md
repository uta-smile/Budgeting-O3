# 1CLL protocol audit

Audit date: 2026-09-04. Paper: Kalisz et al., *How to Spend Your Oracle
Budget: Practical Guidance for Protein Structure Prediction Models*, arXiv
2608.12192v1.

## Requirements traced to code

| Requirement | Paper evidence | Repository evidence |
| --- | --- | --- |
| Target | Section 4: 1CLL calmodulin, 144 residues | `data/1cll_boltz_input.yaml`; adapter and verifier require one 144-residue chain A |
| Latent | Section 4: `Z = R^3552`, described as 1,184 atom coordinates | `atom_slots: 1184`; adapter derives `latent_dim = atom_slots * 3` and records both values |
| Oracle | Section 4: C-alpha TM-score against 1CLL, normalized by the 144-residue ground truth | `TMScoreOracle` puts reference chain A first in `tm_align` and returns `tm_norm_chain1` |
| Budgets | Section 4: `(20,2)`, `(50,5)`, `(100,10)`, then larger settings | Configuration and public runner expose only the first three, per the hardware limit |
| Repeats | Section 4: five random seeds | Runner supports 1/3/5/10 replicates, generates OS-random seeds by default, prints them, and saves the exact list; 5 reproduces the paper and 10 is an optional higher-confidence extension. The optional Linux GPU scheduler changes only replicate placement. |
| Best K-of-N | Section 3: draw N ordinary samples and return the K highest oracle scores | Isolated public `boltz==2.2.1` CLI; N one-sample seeded calls; top K selected by external TM-score |
| O3 phase 1 | Section 3.1: score M deterministic generations and retain the best d latent seeds | `run_o3`: M explicit standard-normal latents, deterministic decode, oracle sort, top d |
| O3 chart | Algorithm 1: KR transform to positive-sphere weights, then `z = w^T Z` | `SurrogateChart` and independent reference tests |
| O3 BO | Section 3.1: d seed projections plus two random U points; RBF/constant-mean single-task GP; LogEI; `N-M-2` rounds | `run_o3` and `_fit_and_acquire`; exact N-call assertion |
| O3 decoder | Footnote 1: probability-flow conversion and stochastic SE(3) augmentation disabled | Vendored opt-in sampler accepts explicit coordinates, disables churn/noise/augmentation in deterministic mode, and is covered by a no-randomness unit test |
| Returned metrics | Section 4: mean-of-K and max-of-K | Both runners rank all N oracle-scored structures and write both metrics |

## Method separation

The baseline and O3 share the same checked-in Boltz input YAML and reference,
but they intentionally do not share generated structures:

- As of 2026-09-09, Best K-of-N explicitly uses `--step_scale 1.0`, retaining
  stochastic `gamma_0=0.8`. This user-requested controlled comparison differs
  from historical runs, which used the stock step scale of 1.5.
- O3 passes an explicit `z`, sets deterministic mode, suppresses churn and
  SE(3) randomness, and uses `step_scale=1.0` for the Euler PF-ODE update.
- A baseline run's recorded replicate seeds may be imported with
  `--seeds-from-baseline-run`. Reusing its stochastic structures as O3 seeds
  would violate the paper's deterministic-generator requirement.

## Paper ambiguities and explicit choices

These points are not fully specified by the paper and should not be presented
as direct reproductions of an unstated setting:

1. The paper says the prediction starts from amino-acid sequence but does not
   identify an MSA source or retrieval protocol. This repository chooses
   explicit Boltz single-sequence mode (`msa: empty`) for both methods.
2. Appendix Table A.1 has K values of 6 and 7 for the N=20 and N=50 O3 rows,
   conflicting with Section 4's `(20,2)` and `(50,5)` benchmark definition.
   The runner uses the Section 4 K values for the returned batch; M and d come
   from Table A.1.
3. The paper names Boltz-2 but does not state a package release in the PDF.
   The baseline is pinned to the public `boltz==2.2.1` package and the custom
   source is audited against that installation.
4. The 144-residue sequence expands to 1,134 canonical Boltz heavy-atom
   records, while the paper states 1,184 coordinate slots and D=3,552. The
   adapter pads the sampler tensor to 1,184 and removes the 50 masked slots
   before PDB output. This exactly matches the paper's latent width but is an
   explicit reconstruction choice, not a mechanism described in the PDF.
5. The public CLI normally selects FP16. The lab GTX Titan requires FP32 to
   avoid CUDA SVD failures. Precision is treated as a hardware compatibility
   setting; stochastic step scale, churn, seeds, and model package remain
   unchanged and are recorded in new summaries.
6. The paper specifies an RBF GP but not optimizer bounds for its ARD length
   scales. This implementation constrains them to `[0.01, 10]` on the unit
   cube to prevent near-rank-one covariance failures observed in the lab, and
   accepts finite models returned with recoverable SciPy line-search warnings.
   Optimizer timeouts still fail the run. This is a recorded numerical
   stabilization, not a hyperparameter claimed by the paper.

## Verification status

- Local suite: deterministic sampler, budget accounting, chart mapping,
  canonical paths/input, seed schedules, resume equivalence, and TM-score
  chain normalization.
- Vendor audit: every Python source difference from public `boltz==2.2.1` is
  covered by `vendor_patch_allowlist.json`; the public CLI file itself is
  restored to upstream precision behavior.
- Existing lab evidence: a completed five-seed N=50 run records CUDA 11.8,
  PyTorch 2.5.1+cu118, GTX TITAN X, FP32 O3, D=3,552, 1,184 sampler slots,
  50 oracle evaluations per seed, identical seed lists across methods, and
  no MSA.
- The project-local O3 and public-baseline caches contain byte-identical
  `boltz2_conf.ckpt` files (SHA-256
  `090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1`).
  New method metadata records the checkpoint digest directly.
- Still required after code changes: run the one-replicate N=100 lab smoke
  command and inspect its startup/runtime metadata. A desktop without the
  NVIDIA driver cannot establish final CUDA execution.
