#!/usr/bin/env sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT_DIR"

# Keep large dependency archives on the project filesystem instead of a
# potentially quota-limited home-directory cache. Override this when needed.
UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"
export UV_CACHE_DIR
mkdir -p "$UV_CACHE_DIR"

# Keep model assets off the quota-limited home filesystem.  FP32 is the safe
# public-Boltz precision on the lab's GTX Titan; callers may override either.
BOLTZ_CACHE="${BOLTZ_CACHE:-$ROOT_DIR/.boltz}"
BOLTZ_PUBLIC_PRECISION="${BOLTZ_PUBLIC_PRECISION:-32}"
export BOLTZ_CACHE BOLTZ_PUBLIC_PRECISION
mkdir -p "$BOLTZ_CACHE"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  exit 2
fi
if [ ! -f "vendor/boltz/pyproject.toml" ]; then
  echo "The customized Boltz-2 source is missing from vendor/boltz." >&2
  echo "Populate vendor/boltz before running this O3 experiment." >&2
  exit 2
fi
if ! grep -q "PDB-output/TM-score fix enabled" "src/o3_boltz/cli.py" \
  || ! grep -q "def _load_structure" "src/o3_boltz/tmscore.py" \
  || ! grep -q "sample_.*pdb" "src/o3_boltz/o3.py"; then
  echo "This project copy is stale and is missing the PDB scoring fix." >&2
  echo "Refresh the project files, then start the script again." >&2
  exit 2
fi

uv sync

REFERENCE_PDB="data/1CLL.pdb"
if [ ! -s "$REFERENCE_PDB" ]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to download data/1CLL.pdb on the first run." >&2
    exit 2
  fi
  mkdir -p "$(dirname "$REFERENCE_PDB")"
  curl -fsSL "https://files.rcsb.org/download/1CLL.pdb" -o "$REFERENCE_PDB"
fi

# The comparison bundle routes Best K-of-N to an isolated, unmodified public
# Boltz 2.2.1 environment (step_scale=1.0) and O3 to the vendored deterministic adapter. It is
# the single supported experiment entry point.
if [ "$#" -eq 0 ]; then
  for BUDGET in n20_k2 n50_k5 n100_k10; do
    uv run python experiments/1cll/run.py \
      --only "$BUDGET" --comparison-report
  done
else
  uv run python experiments/1cll/run.py \
    "$@" --comparison-report
fi
