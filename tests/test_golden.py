"""Vendored TribChem modules against the golden values of the historic
environment (see golden_cases.py), and the copied input data against the
TribChem sources."""

import copy
import re
import shutil

import pytest

import golden_cases
import vendoring

# Cases whose vendored results may differ from the golden values:
# {case: ({'rtol': ..., 'atol': ...}, reason)}. The others must match exactly.
# The densities of the charge_step cases reach the original as CHGCAR files (12 significant
# digits) and the port as pp.x files (10 digits, 1pe17.9): the difference of the three, which
# removes most of each, shows the last digit of the largest density in the planar averages
# and in the curve of the displacement, up to 5e-10 e/A^3 in the three cases (measured). The
# tolerance covers only the numbers that come from the density, by their path in the result:
# the density column of every planar average, the integrals of the displacement and its
# curve. Everything else is compared exactly: structures, k-points, range of integration,
# abscissae of the averages and of the curve, widths, grid, rescale, entries and keys.
CHARGE_DENSITY_PATHS = (
    r'/calls/\d+/data/average/\w+/\w+/[xyz]/1/\d+',
    r'/calls/\d+/data/displacement/\w+/[xyz]/\w+/integral',
    r'/calls/\d+/data/displacement/\w+/[xyz]/grid/1/\d+',
)
CHARGE_TOLERANCE = ({'rtol': 1e-6, 'atol': 1e-9, 'paths': CHARGE_DENSITY_PATHS},
                    'density from a 10-digit pp.x file instead of a 12-digit CHGCAR')
CHARGE_CASES = [name for name in golden_cases.CASES if name.startswith('charge_step_')]
TOLERANCES = {name: CHARGE_TOLERANCE for name in CHARGE_CASES}


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


@pytest.mark.parametrize('expected, actual, n', [
    ({'a': [1.0, 2.0]}, {'a': [1.0, 2.0 + 1e-12]}, 0),
    ({'a': [1.0, 2.0]}, {'a': [1.0 + 1e-12, 2.0]}, 1),
    ({'a': [1.0, 2.0]}, {'a': [1.0 + 1e-12, 2.0 + 1e-12]}, 1),
    ({'a': [1.0, 2.0]}, {'a': [1.0, 2.0 + 1e-3]}, 1),
    ({'a': {'1': 1.0, '10': 1.0}}, {'a': {'1': 1.0 + 1e-12, '10': 1.0 + 1e-12}}, 1),
])
def test_compare_with_a_tolerance_only_at_some_paths(expected, actual, n):
    tolerance = {'rtol': 1e-9, 'atol': 0.0, 'paths': [r'/a/1']}
    assert len(golden_cases.compare(expected, actual, tolerance)) == n


def numbers_of(data, path=''):
    """``(path, number)`` of every number of a JSON tree, booleans excluded, in order."""
    if isinstance(data, dict):
        for key, value in data.items():
            yield from numbers_of(value, f'{path}/{key}')
    elif isinstance(data, list):
        for i, value in enumerate(data):
            yield from numbers_of(value, f'{path}/{i}')
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        yield path, data


def perturbed(data, change, selected, path=''):
    """A copy of a JSON tree where ``change`` is applied to the floats whose path is ``selected``."""
    if isinstance(data, dict):
        return {key: perturbed(value, change, selected, f'{path}/{key}')
                for key, value in data.items()}
    if isinstance(data, list):
        return [perturbed(value, change, selected, f'{path}/{i}') for i, value in enumerate(data)]
    if isinstance(data, float) and selected(path):
        return change(data)
    return data


def changed(data, pattern, change):
    """A copy of a JSON tree with ``change`` applied to the first number whose path matches."""
    path = next(path for path, _ in numbers_of(data) if re.fullmatch(pattern, path))
    other = copy.deepcopy(data)
    *parents, last = path.strip('/').split('/')
    node = other
    for key in parents:
        node = node[int(key)] if isinstance(node, list) else node[key]
    last = int(last) if isinstance(node, list) else last
    node[last] = change(node[last])
    return other


def covered_by(name):
    tolerance, _ = TOLERANCES[name]
    return lambda path: golden_cases.tolerated(tolerance, path) is not None


@pytest.mark.parametrize('name', CHARGE_CASES)
def test_the_tolerance_of_the_charge_cases_covers_the_density_and_nothing_else(name):
    """Numbers that come from the density may differ in their last digits; no other number may.

    A change of 5e-6 in every float outside of the paths of the tolerance (structures, k-points,
    range, abscissae, widths) is one difference for each of them; the last digits of those in the
    paths are none, and a larger change is.
    """
    tolerance, _ = TOLERANCES[name]
    golden = golden_cases.read_golden(name)
    covered = covered_by(name)
    floats = [(path, value) for path, value in numbers_of(golden) if isinstance(value, float)]
    inside = [path for path, _ in floats if covered(path)]
    outside = [path for path, _ in floats if not covered(path)]
    assert inside and len(outside) > len(inside)
    last_digits = perturbed(golden, lambda value: value * (1 + 1e-8) + 1e-11, covered)
    assert golden_cases.compare(golden, last_digits, tolerance) == []
    assert len(golden_cases.compare(golden, last_digits)) == len(inside)
    larger = perturbed(golden, lambda value: value * (1 + 1e-4) + 1e-6, covered)
    assert len(golden_cases.compare(golden, larger, tolerance)) == len(inside)
    others = perturbed(golden, lambda value: value + 5e-6, lambda path: not covered(path))
    assert len(golden_cases.compare(golden, others, tolerance)) == len(outside)


@pytest.mark.parametrize('name', CHARGE_CASES)
@pytest.mark.parametrize('pattern, change', [
    (r'/inputs/integrate_in/0', lambda value: value + 5e-6),
    (r'/inputs/integrate_in/1', lambda value: value - 5e-6),
    (r'/inputs/structures/0/sites/0/xyz/2', lambda value: value + 5e-6),
    (r'/inputs/structures/1/lattice/matrix/2/2', lambda value: value + 5e-6),
    (r'/inputs/kpoints/0/kpoints/0/0', lambda value: value + 1),
    (r'/calls/\d+/data/\d+/sites/0/abc/2', lambda value: value + 5e-6),
    (r'/calls/\d+/data/average/\w+/\w+/z/0/0', lambda value: value + 5e-6),
    (r'/calls/\d+/data/displacement/\w+/z/interface/width', lambda value: value + 5e-6),
    (r'/calls/\d+/data/displacement/\w+/z/grid/0/0', lambda value: value + 5e-6),
    (r'/calls/\d+/data/displacement/\w+/z/rescale', lambda value: value + 1),
], ids=['lower-limit', 'upper-limit', 'atom', 'cell', 'k-points', 'structure-written',
        'abscissa-of-average', 'width', 'abscissa-of-curve', 'rescale'])
def test_the_tolerance_of_the_charge_cases_does_not_cover_the_inputs(name, pattern, change):
    """A change of the inputs, of the geometry or of the parameters is a difference."""
    tolerance, _ = TOLERANCES[name]
    golden = golden_cases.read_golden(name)
    other = changed(golden, pattern, change)
    assert len(golden_cases.compare(golden, other, tolerance)) == 1


def test_strip_uuid():
    tag = 'cu(1, 1, 1)_al(1, 1, 1)_ontop_1_9b2f0c1e-3d4a-4f5b-8c6d-7e8f9a0b1c2d'
    assert golden_cases.strip_uuid(tag) == 'cu(1, 1, 1)_al(1, 1, 1)_ontop_1_<uuid4>'
    with pytest.raises(ValueError):
        golden_cases.strip_uuid('cu(1, 1, 1)_al(1, 1, 1)_ontop_1')
