from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from tmtools import tm_align
from tmtools.io import get_residue_data, get_structure


def _load_structure(path: Path) -> Any:
    """Load PDB or mmCIF regardless of the filename extension.

    Older tmtools releases default to PDB parsing and do not infer the format
    from ``.cif``. Detecting the content here keeps scoring compatible with
    both the current PDB writer and valid legacy mmCIF artifacts.
    """

    if not path.exists():
        raise FileNotFoundError(f"Structure file does not exist: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Structure file is empty: {path}")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first_line = next((line.strip() for line in handle if line.strip()), "")

    is_mmcif = first_line.lower().startswith(("data_", "#"))
    if is_mmcif:
        try:
            return get_structure(str(path), format="mmcif")
        except TypeError:
            # tmtools 0.1.x has no format keyword; use its underlying parser.
            from Bio.PDB.MMCIFParser import MMCIFParser

            return MMCIFParser(QUIET=True).get_structure(path.stem, str(path))
    # PDB is the default in all supported tmtools versions.
    return get_structure(str(path))


class TMScoreOracle:
    """TM-align oracle normalized by the reference-chain length."""

    def __init__(self, reference_path: Path, reference_chain: str | None = None):
        structure = _load_structure(reference_path)
        self.reference_coords, self.reference_sequence = get_residue_data(
            self._select_chain(structure, reference_chain)
        )
        if len(self.reference_sequence) == 0:
            raise ValueError(f"No protein residues found in {reference_path}")

    @staticmethod
    def _select_chain(structure: Any, chain_id: str | None) -> Any:
        try:
            model = next(structure.get_models())
        except StopIteration as exc:
            raise ValueError(
                "Structure contains no readable model or atom records"
            ) from exc
        if chain_id is not None and chain_id in model:
            return model[chain_id]
        try:
            return next(model.get_chains())
        except StopIteration as exc:
            raise ValueError("Structure contains no readable chains") from exc

    def score(self, structure_path: Path, chain_id: str | None = None) -> float:
        structure = _load_structure(structure_path)
        coords, sequence = get_residue_data(self._select_chain(structure, chain_id))
        if len(sequence) == 0:
            raise ValueError(f"No protein residues found in {structure_path}")

        # Put the ground truth first so tm_norm_chain1 is normalized by the
        # reference length, matching the oracle definition in the notes.
        result = tm_align(
            self.reference_coords,
            coords,
            self.reference_sequence,
            sequence,
        )
        return float(result.tm_norm_chain1)
