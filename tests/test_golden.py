"""Vendored TribChem modules against the golden values of the historic
environment (see golden_cases.py), and the copied input data against the
TribChem sources."""

import shutil

import pytest

import golden_cases
import vendoring

# Cases whose vendored results may differ from the golden values:
# {case: ({'rtol': ..., 'atol': ...}, reason)}. Empty: every case must match
# exactly.
TOLERANCES = {}


def differences(name, root):
    tolerance, _ = TOLERANCES.get(name, (None, None))
    return golden_cases.compare(golden_cases.read_golden(name),
                                golden_cases.run(name, root), tolerance)


@pytest.mark.golden
@pytest.mark.parametrize('name', list(golden_cases.CASES))
def test_vendored_matches_golden(name):
    diffs = differences(name, golden_cases.VENDORED)
    assert not diffs, f'{len(diffs)} differences, the first ones: {diffs[:10]}'


def test_every_case_has_a_golden_file():
    assert golden_cases.golden_names() == sorted(golden_cases.CASES)


def test_golden_environment_is_the_historic_lock(read_lock):
    pins = read_lock('historic') | {'tribchem': '1.0.0'}
    recorded = golden_cases.read_golden('environment')
    assert {name: recorded[name] for name in golden_cases.RECORDED} == \
        {name: pins[name] for name in golden_cases.RECORDED}


@pytest.mark.parametrize('rel', golden_cases.data_files())
def test_data_matches_tribchem(rel):
    assert golden_cases.check_data(rel, vendoring.read_manifest()) == []


def test_check_data_detects_changes(tmp_path):
    rel = 'physics/tribology/tests/grid.txt'
    (tmp_path / rel).parent.mkdir(parents=True)
    shutil.copy(golden_cases.DATA / rel, tmp_path / rel)
    with open(tmp_path / rel, 'a') as f:
        f.write('\n')
    assert golden_cases.check_data(rel, vendoring.read_manifest(), tmp_path)


def test_slab_dicts_of_the_matching_script():
    assert sorted(golden_cases.slab_dicts()) == \
        ['ag111', 'al111', 'alc111', 'cu111', 'cuau111', 'pt111']


@pytest.mark.parametrize('expected, actual, tolerance, n', [
    ({'a': [1, 2.5, 'x', None]}, {'a': [1, 2.5, 'x', None]}, None, 0),
    ({'a': 1.0}, {'a': 1.0000000001}, None, 1),
    ({'a': 1.0}, {'a': 1.0000000001}, {'rtol': 1e-9, 'atol': 0}, 0),
    ({'a': 1}, {'a': 1.0}, None, 1),
    ({'a': True}, {'a': 1}, None, 1),
    ({'a': 1, 'b': 2}, {'a': 1, 'c': 2}, None, 2),
    ([1, 2], [1, 2, 3], None, 1),
    ([float('nan')], [float('nan')], None, 0),
])
def test_compare(expected, actual, tolerance, n):
    assert len(golden_cases.compare(expected, actual, tolerance)) == n


def test_strip_uuid():
    tag = 'cu(1, 1, 1)_al(1, 1, 1)_ontop_1_9b2f0c1e-3d4a-4f5b-8c6d-7e8f9a0b1c2d'
    assert golden_cases.strip_uuid(tag) == 'cu(1, 1, 1)_al(1, 1, 1)_ontop_1_<uuid4>'
    with pytest.raises(ValueError):
        golden_cases.strip_uuid('cu(1, 1, 1)_al(1, 1, 1)_ontop_1')
