#!/usr/bin/env sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT_DIR"

# Keep large dependency archives on the project filesystem instead of a
# potentially quota-limited home-directory cache. Override this when needed.
UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"
export UV_CACHE_DIR
mkdir -p "$UV_CACHE_DIR"

# The SSH/login step is intentionally outside this file. Run this script from
# the WinSCP-synchronised project directory on the lab node.
# The repository now includes the deterministic adapter. O3_ADAPTER_SPEC can
# override it for a lab-specific implementation.
ADAPTER_SPEC="${O3_ADAPTER_SPEC:-adapters.boltz2_pfode:create}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  exit 2
fi
if [ ! -f "vendor/boltz/pyproject.toml" ]; then
  echo "The vendored Boltz source is missing. Verify that WinSCP finished syncing." >&2
  exit 2
fi
if ! grep -q "PDB-output/TM-score fix enabled" "src/o3_boltz/cli.py" \
  || ! grep -q "def _load_structure" "src/o3_boltz/tmscore.py" \
  || ! grep -q "sample_.*pdb" "src/o3_boltz/o3.py"; then
  echo "This project copy is stale and is missing the PDB scoring fix." >&2
  echo "Run WinSCP synchronization, then start the script again." >&2
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

uv run python -m o3_boltz.cli \
  --config configs/1cll.yaml \
  --adapter "$ADAPTER_SPEC" \
  "$@"
