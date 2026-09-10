"""Run the public Boltz CLI with an explicitly selected Lightning precision.

The stock ``boltz`` entry point hard-codes FP16 for Boltz-2. This small
wrapper is used only for an explicit user override or the public runner's
FP32 recovery after a CUDA SVD failure; it still imports the installed,
version-pinned public package.
"""

from __future__ import annotations

import os

import boltz.main as boltz_main


precision = os.environ.get("BOLTZ_PUBLIC_PRECISION", "32")
if precision not in {"16-mixed", "bf16-mixed", "32", "32-true"}:
    raise SystemExit(
        "BOLTZ_PUBLIC_PRECISION must be one of: 16-mixed, bf16-mixed, 32, 32-true"
    )

_Trainer = boltz_main.Trainer


class _PrecisionTrainer(_Trainer):
    def __init__(self, *args, **kwargs):
        kwargs["precision"] = precision
        super().__init__(*args, **kwargs)


boltz_main.Trainer = _PrecisionTrainer
boltz_main.cli()
