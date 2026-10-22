"""Vendored TribChem code compared with the TribChem sources installed in the
historic environment (see tests/vendoring.py).

Skipped when TribChem is not importable, that is outside the historic venv.
"""

from pathlib import Path

import pytest

tribchem = pytest.importorskip('tribchem')

import vendoring  # noqa: E402

SOURCE_ROOT = Path(tribchem.__file__).resolve().parents[1]


@pytest.mark.parametrize('rel', vendoring.vendored_files())
def test_matches_tribchem_sources(rel):
    assert vendoring.check(rel, vendoring.read_manifest(), SOURCE_ROOT) == []
