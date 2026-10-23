"""Golden values reproduced in the historic environment, where they are
produced (see golden_cases.py), and input data equal to the installed
TribChem sources.

Skipped when TribChem is not importable, that is outside the historic venv.
"""

from pathlib import Path

import pytest

tribchem = pytest.importorskip('tribchem')

import golden_cases  # noqa: E402

PACKAGE = Path(tribchem.__file__).resolve().parent


@pytest.mark.golden
@pytest.mark.parametrize('name', list(golden_cases.CASES))
def test_tribchem_reproduces_golden(name):
    diffs = golden_cases.compare(golden_cases.read_golden(name),
                                 golden_cases.run(name, golden_cases.HISTORIC))
    assert not diffs, f'{len(diffs)} differences, the first ones: {diffs[:10]}'


@pytest.mark.parametrize('rel', golden_cases.data_files())
def test_data_equals_installed_sources(rel):
    assert (golden_cases.DATA / rel).read_bytes() == (PACKAGE / rel).read_bytes()
