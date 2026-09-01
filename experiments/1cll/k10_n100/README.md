# 1CLL K-of-N experiment bundle

This is the canonical experiment bundle for the 1CLL Best K-of-N and O3 comparison.
It supports `n20_k2`, `n50_k5`, and `n100_k10` budgets.

Historical outputs are not inputs to this canonical experiment.

The bundle uses the 144-residue 1CLL construct and the frozen MSA in `inputs/1CLL_0.csv`. Both methods consume this same MSA. The public baseline uses an isolated UV project pinned to `boltz==2.2.1`; O3 uses the repository's vendored Boltz fork only because O3 requires explicit latent initialization and deterministic PF-ODE sampling.

Paired runs use one shared reproducible seed schedule for both methods. The
default five replicate seeds are `20250117, 20251126, 20252135, 20253144,
20254153`; the launcher prints them and records them in each method's metadata.
Use `--seed-start` and `--seed-step` to choose a different schedule.

The archived notebook is in `notebook/`. Its 100 scored structures are represented by the compact fixture `inputs/notebook_1CLL_TMscore_results.csv`; the top-10 mean is `0.6788870863920918` and the top score is `0.7958910827539285`.

Run the public baseline, O3, or both methods:

```bash
python experiments/1cll/k10_n100/run.py --method both --replicates 1 --run-id smoke
python experiments/1cll/k10_n100/run.py --method both --replicates 5 --run-id paper_n100_k10
python experiments/1cll/k10_n100/run.py --method best-k-of-n --replicates 1 --run-id notebook_reproduction --resume
```

The paper's smaller budgets are also available. The budget name is included
in the command's default run ID and in the output folder:

```bash
python experiments/1cll/k10_n100/run.py --budget n20_k2 --method both --replicates 5
python experiments/1cll/k10_n100/run.py --budget n50_k5 --method both --replicates 5
```

These write to `outputs/1cll/k2_n20/` and `outputs/1cll/k5_n50/`, respectively.
Without `--run-id`, a timestamped budget label is generated; use the same
explicit `--run-id` together with `--resume` to continue an interrupted run.
Their O3 settings are `M=10, d=5` and `M=25, d=7`; the resulting BO rounds
are `8` and `23` because the budget accounts for `M + 2 + nrounds = N`.

To isolate BO from the decoder, run the same-decoder random PF-ODE diagnostic.
It uses all `N` calls as independent random samples from `Z` and writes to a
separate `random_pfode` output directory:

```bash
python experiments/1cll/k10_n100/run.py --budget n20_k2 --method random-pfode --replicates 5 --run-id n20_k2_random_pfode01
```

Run verification before a full experiment:

```bash
python experiments/1cll/k10_n100/verify.py
python experiments/1cll/k10_n100/verify.py --audit-vendor --gpu
```

New results are written to `outputs/1cll/k10_n100/<method>/runs/<run_id>/`. Existing historical outputs are not reused.
