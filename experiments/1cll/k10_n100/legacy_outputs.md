# Legacy outputs

The pre-refactor directories below are retained for traceability and are not
inputs to the canonical `k10_n100` experiment:

- `outputs/1cll/best_k_of_n/n100_k10/runs/paper_parity_baseline_5seed/`
- `outputs/1cll/o3/n100_k10/runs/parity_o3_n100/`
- `outputs/official_boltz_1cll_tm/`

They mix older MSA-server, MSA-subsampling, adapter, and sampling settings.
The canonical `n20_k2`, `n50_k5`, and `n100_k10` experiments use the frozen
`inputs/1CLL_0.csv` MSA and record its checksum in every new run's provenance.
