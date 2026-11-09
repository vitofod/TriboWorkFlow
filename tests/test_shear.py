# Author: Vito Foderà
# Created: 2026-11-09

"""Tests of the step shear (``triboworkflow.workflows.interface.shear``).

The step reads the PES interpolated by ``plotpes`` and the cell of the interface,
and writes the shear strength along three strings. The tests use the example
run with two Cu(111) slabs, matched once, whose PES has 8 points in the cell
(800 centres of the interpolation), with the PES relaxed along z analysed on
synthetic energies and interpolated. The string method runs with parameters
that keep it short (``FAST``): the algorithm and the units are the same; the
default parameters, as the original ran them, are in the golden cases
``shear_step_*`` and in ``tests/shear_cost.py``.

Notes
-----
To remove: the golden cases ``shear_step_*`` (``tests/golden_cases.py``)
compare the step with TribChem's ``FT_StartShearStrength`` and
``FT_CalcShearStrength``.
"""

import dataclasses
import json
import logging
import shutil

import numpy as np
import pytest

from cli_helpers import fake_scheduler_bin, fake_slurm  # noqa: F401 (fixtures)
from test_pes import edit_config, finish, pes_run
from test_plotpes import analysis_group, run_files, the_node_runs, with_analysis_job
from triboworkflow.cli.analysis_job import run_analysis
from triboworkflow.cli.main import main
from triboworkflow.cli.run import build_orchestrator, load_run
from triboworkflow.core import store
from triboworkflow.core.settings.schema import (SHEAR_MAX_SEED, SHEAR_MIN_POINTS, ShearMepParams,
                                                ShearParams)
from triboworkflow.core.store import PrerequisiteError, StoreError, SystemWriter, read_system
from triboworkflow.execution import layout
from triboworkflow.execution.lock import RunLock
from triboworkflow.vendored.highput.utils.tasktools import read_default_params
from triboworkflow.vendored.physics.tribology.mep import MEP
from triboworkflow.vendored.physics.tribology.pes import PES
from triboworkflow.workflows.interface import pes, plotpes, shear
from triboworkflow.workflows.interface.match import MatchStep, match_provenance
from triboworkflow.workflows.interface.registry import get_step

CUCU = 'mp-30_mp-30'
STEPS = ('match', 'pes', 'plotpes', 'shear')
# Parameters of the string method that keep the tests short.
FAST = {'npts': 21, 'extent': [1.5, 1.5], 'h': 0.01, 'nstepmax': 25, 'tol': 1e-3, 'delta': 1e-4}
KEYS = {'nstep', 'tol', 'shear_strength_GPa', 'lxy', 'dvx', 'dvy', 'vz', 've', 'force'}
OPTIMIZATIONS = ('bs_line', 'x', 'y')
LEAVES = [('shearstr', f'along_{opt}') for opt in OPTIMIZATIONS]


def with_shear(run_dir, delta=None, seed=None, mep=None):
    """The tables [interface.shear] and [interface.shear.mep] of the example, replaced.

    ``delta`` is the step of the derivative of the shear strength, ``mep`` the keys of the
    string method (with its own ``delta``).
    """
    config = run_dir / 'config.toml'
    text = config.read_text(encoding='utf-8')
    lines = ['[interface.shear]']
    lines += [f'delta = {delta!r}'] if delta is not None else []
    lines += [f'seed = {seed!r}'] if seed is not None else []
    lines += ['[interface.shear.mep]']
    lines += [f'{key} = {value!r}' for key, value in (mep or {}).items()]
    start, end = text.index('[interface.shear]'), text.index('[namelists.CONTROL]')
    config.write_text(text[:start] + '\n'.join(lines) + '\n\n' + text[end:], encoding='utf-8')


def cucu_run(tmp_path, name='run', steps=STEPS, **mep):
    """The example run with Cu(111) as both slabs, and the fast string method."""
    run_dir = pes_run(tmp_path, name=name, steps=steps)
    edit_config(run_dir, 'file = "slabs/Al111.json"\nmid = "mp-134"',
                'file = "slabs/Cu111.json"\nmid = "mp-30"')
    with_shear(run_dir, mep=FAST | mep)
    return run_dir


def matched_run(tmp_path, name='run', **kwargs):
    run_dir = cucu_run(tmp_path, name, **kwargs)
    MatchStep().analyse(run_dir, CUCU, [])
    return run_dir


def energies_of(run_dir, name='pes'):
    inputs = pes.pes_inputs(run_dir, CUCU, load_run(run_dir), name)
    return {point: -1000.0 - 0.01 * ((i * 3) % 5) for i, (point, _, _) in enumerate(inputs.points)}


@pytest.fixture(scope='module')
def plotted(tmp_path_factory):
    """The Cu/Cu run matched, with the relaxed PES analysed and interpolated."""
    run_dir = matched_run(tmp_path_factory.mktemp('shear'))
    pes.RelaxedPesStep().analyse(run_dir, CUCU, finish(run_dir, 'pes', energies_of(run_dir),
                                                       system=CUCU))
    plotpes.RelaxedPlotPesStep().analyse(run_dir, CUCU, [])
    return run_dir


@pytest.fixture
def plotted_copy(plotted, tmp_path):
    copied = tmp_path / 'copy'
    shutil.copytree(plotted, copied)
    return copied


@pytest.fixture(scope='module')
def analysed(plotted, tmp_path_factory):
    """The run with the shear strength analysed with the fast parameters."""
    run_dir = tmp_path_factory.mktemp('analysed') / 'run'
    shutil.copytree(plotted, run_dir)
    shear.ShearStep().analyse(run_dir, CUCU, [])
    return run_dir


@pytest.fixture
def analysed_copy(analysed, tmp_path):
    copied = tmp_path / 'analysed'
    shutil.copytree(analysed, copied)
    return copied


def sdir(run_dir):
    return layout.system_dir(run_dir, CUCU)


def document(run_dir):
    return read_system(sdir(run_dir)).document


def system_bytes(run_dir):
    return (sdir(run_dir) / 'system.json').read_bytes()


def inputs_of(run_dir):
    return shear.shear_inputs(sdir(run_dir), read_system(sdir(run_dir)), load_run(run_dir))


def params_of(run_dir):
    return load_run(run_dir).interface.shear


def by_the_vendored_code(inputs, shear_delta=0.01, seed=0, mep=None):
    """What the vendored classes give on the inputs, called as the original does, through JSON.

    ``mep`` are the parameters of the string method (none: its own defaults) and
    ``shear_delta`` the step of the derivative of the shear strength.
    """
    mep = mep or {}
    cell = np.array(inputs.cell)
    solver = PES(np.array(inputs.pes_jm2), cell)
    solver.make_pes(replicate_of=(5, 5), density=20, tol=1e-4)
    data = {}
    for opt in OPTIMIZATIONS:
        np.random.seed(seed)
        string = MEP(rbf=solver._rbf, cell=cell)
        string.get_mep(method='zerotemp', optimization=opt, **mep)
        string.get_shear_strenght(delta=shear_delta, npts=mep.get('npts', 101), optimization=opt)
        data.update(string.mep_data)
    return json.loads(json.dumps(data))


class Calls:
    """The calls of the step to the vendored classes, which still run."""

    def __init__(self, monkeypatch):
        self.calls = []
        calls = self.calls

        class SpyPES(PES):
            def __init__(self, data, cell):
                calls.append(('PES', np.array(data), np.array(cell)))
                super().__init__(data, cell)

            def make_pes(self, **kwargs):
                calls.append(('make_pes', kwargs))
                return super().make_pes(**kwargs)

        class SpyMEP(MEP):
            def __init__(self, rbf, cell):
                calls.append(('MEP', rbf, np.array(cell)))
                super().__init__(rbf, cell)

            def get_mep(self, method='zerotemp', optimization='bs_line', **kwargs):
                calls.append(('get_mep', method, optimization, kwargs,
                              np.random.get_state()[1].copy()))
                return super().get_mep(method=method, optimization=optimization, **kwargs)

            def get_shear_strenght(self, delta=0.01, npts=101, optimization='bs_line'):
                calls.append(('get_shear_strenght', delta, npts, optimization))
                return super().get_shear_strenght(delta=delta, npts=npts,
                                                  optimization=optimization)

        monkeypatch.setattr(shear, 'PES', SpyPES)
        monkeypatch.setattr(shear, 'MEP', SpyMEP)

    def names(self):
        return [call[0] for call in self.calls]


# --- the step -----------------------------------------------------------------------------

def test_the_step_is_registered_and_never_run_by_the_orchestrator():
    step = get_step('shear')
    assert isinstance(step, shear.ShearStep) and step.name == 'shear'
    assert step.manual_analysis is True


def test_no_calculation(plotted):
    step = shear.ShearStep()
    assert step.calculations(plotted, CUCU) == []
    assert step.plan(plotted, CUCU) == {}
    assert step.prepare(plotted, CUCU) == []


def test_the_leaves_are_the_three_optimizations():
    assert shear.ENTRY == 'shearstr'
    assert shear.OPTIMIZATIONS == OPTIMIZATIONS
    assert shear.shear_leaves() == LEAVES


def test_expected_entries(plotted):
    expected = shear.ShearStep().expected(plotted, CUCU)
    assert set(expected) == {('pes', 'energy', 'pes_jm2'), ('structure', 'init', 'interface')}
    assert expected[('pes', 'energy', 'pes_jm2')] == \
        plotpes.RelaxedPlotPesStep().provenance(plotted, CUCU)
    assert expected[('structure', 'init', 'interface')] == match_provenance(load_run(plotted))


def test_expected_needs_the_steps_behind_the_interpolation(plotted_copy):
    # The relaxed PES was completed again with another energy and not analysed: its
    # interpolation is read as written, but it is not current.
    calc_id = pes.RelaxedPesStep().calculations(plotted_copy, CUCU)[0].calc_id
    record = store.read_record(sdir(plotted_copy), calc_id)
    results = {**record.results, 'output': {**record.results['output'], 'energy': -999.0}}
    store.write_record(sdir(plotted_copy), dataclasses.replace(record, results=results))
    with pytest.raises(PrerequisiteError, match='steps upstream of plotpes not done or not '
                                                'current: pes'):
        shear.ShearStep().expected(plotted_copy, CUCU)
    with pytest.raises(PrerequisiteError, match='steps upstream of plotpes'):
        shear.ShearStep().analyse(plotted_copy, CUCU, [])


# --- inputs -------------------------------------------------------------------------------

def test_inputs_read_from_the_document(plotted):
    doc = document(plotted)
    inputs = inputs_of(plotted)
    assert inputs.pes_jm2 == doc['pes']['energy']['pes_jm2']
    assert inputs.cell == doc['structure']['init']['interface']['lattice']['matrix']
    assert inputs.ids == {key: doc[key] for key in ('mid', 'formula', 'name', 'miller')}
    assert inputs.read() == {'pes_jm2': inputs.pes_jm2, 'cell': inputs.cell}


def test_only_the_pes_and_the_cell_are_read(plotted):
    # The grid of the PES and the computational parameters, which the original read and did not
    # use, are not inputs of the step.
    assert set(inputs_of(plotted).read()) == {'pes_jm2', 'cell'}
    assert {field.name for field in dataclasses.fields(inputs_of(plotted))} == \
        {'ids', 'pes_jm2', 'cell'}


def test_the_cell_is_that_of_the_final_structure_of_the_minimum(plotted):
    # plotpes interpolates on the cell of the final structure of the lowest point; the string
    # is built on the cell of the interface after match, as the original did. A relaxation of
    # the atoms at fixed cell leaves it the same, up to the rounding with which the lattice of
    # each point is rebuilt when the slabs are shifted (here 8e-11 angstrom in one element).
    doc = document(plotted)
    plotpes_inputs = plotpes.plotpes_inputs(sdir(plotted), read_system(sdir(plotted)),
                                            load_run(plotted), 'plotpes')
    np.testing.assert_allclose(np.array(inputs_of(plotted).cell), plotpes_inputs.cell,
                               rtol=1e-9, atol=1e-9)
    assert inputs_of(plotted).cell == doc['structure']['init']['interface']['lattice']['matrix']


def test_without_plotpes_the_step_stops(tmp_path):
    run_dir = matched_run(tmp_path)
    pes.RelaxedPesStep().analyse(run_dir, CUCU, finish(run_dir, 'pes', energies_of(run_dir),
                                                       system=CUCU))
    before = system_bytes(run_dir)
    with pytest.raises(PrerequisiteError, match=r'pes\.energy\.pes_jm2'):
        shear.ShearStep().analyse(run_dir, CUCU, [])
    with pytest.raises(PrerequisiteError, match=r'run plotpes'):
        inputs_of(run_dir)
    assert system_bytes(run_dir) == before
    assert not shear.ShearStep().done(run_dir, CUCU)


def test_without_match_the_step_stops(tmp_path):
    run_dir = cucu_run(tmp_path)
    with pytest.raises(PrerequisiteError, match='run match first'):
        inputs_of(run_dir)


def test_with_only_the_static_pes_the_step_stops(tmp_path):
    # The original read the relaxed PES only and failed with a KeyError without it; here the step
    # is blocked with the entry that it needs, even if the static PES is interpolated.
    run_dir = matched_run(tmp_path, steps=('match', 'pes_scf', 'plotpes_scf', 'shear'))
    pes.StaticPesStep().analyse(run_dir, CUCU, finish(run_dir, 'pes_scf',
                                                      energies_of(run_dir, 'pes_scf'),
                                                      system=CUCU))
    plotpes.StaticPlotPesStep().analyse(run_dir, CUCU, [])
    before = system_bytes(run_dir)
    with pytest.raises(PrerequisiteError, match=r'pes\.energy\.pes_jm2'):
        shear.ShearStep().analyse(run_dir, CUCU, [])
    assert system_bytes(run_dir) == before


# --- the cell -----------------------------------------------------------------------------

SQUARE = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 20.0]]
HEXAGONAL = [[2.5672, 0.0, 0.0], [1.2836, 2.2233, 0.0], [0.0, 0.0, 20.0]]
RECTANGLE = [[3.3142, 0.0, 0.0], [0.0, 4.4495, 0.0], [0.0, 0.0, 20.0]]
ROTATED = (3.0 * np.array([[np.cos(np.pi / 6), np.sin(np.pi / 6), 0.0],
                           [-np.sin(np.pi / 6), np.cos(np.pi / 6), 0.0],
                           [0.0, 0.0, 0.0]]) + np.diag([0.0, 0.0, 20.0])).tolist()


@pytest.mark.parametrize('cell', [SQUARE, [[3.0, 0.0, 0.0], [0.0, 3.0004, 0.0], [0.0, 0.0, 9.0]]])
def test_a_square_cell_gives_no_warning(cell):
    assert shear.cell_warning(cell) is None


@pytest.mark.parametrize('cell, text', [
    (HEXAGONAL, 'a = 2.5672, b = 2.5672 angstrom, gamma = 60.00 degrees'),
    (RECTANGLE, 'a = 3.3142, b = 4.4495 angstrom, gamma = 90.00 degrees'),
    (ROTATED, 'a = 3.0000, b = 3.0000 angstrom, gamma = 90.00 degrees'),
    ([[3.0, 0.0, 0.0], [0.0, 3.01, 0.0], [0.0, 0.0, 20.0]],
     'a = 3.0000, b = 3.0100 angstrom, gamma = 90.00 degrees'),
])
def test_a_cell_that_is_not_a_square_is_told(cell, text):
    warning = shear.cell_warning(cell)
    assert 'not a square' in warning and text in warning
    # The algorithm is not changed: it uses the two diagonal elements of the cell only.
    assert 'diagonal elements' in warning


def test_the_cell_of_the_example_is_told_in_the_log(plotted_copy, caplog):
    with caplog.at_level(logging.WARNING, logger=shear.__name__):
        shear.ShearStep().analyse(plotted_copy, CUCU, [])
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
                and r.name == shear.__name__]
    assert len(warnings) == 1 and warnings[0].startswith(f'{CUCU}: the cell of the interface is '
                                                         'not a square')


# --- analysis -----------------------------------------------------------------------------

def test_entries_written(analysed):
    shearstr = document(analysed)['shearstr']
    assert list(shearstr) == [f'along_{opt}' for opt in OPTIMIZATIONS]
    for entry in shearstr.values():
        assert set(entry) == KEYS
        assert entry['nstep'] <= FAST['nstepmax'] and entry['tol'] > 0
        assert isinstance(entry['shear_strength_GPa'], float) and entry['shear_strength_GPa'] > 0
        # Six numbers under each of the six names (A98): see the next test.
        for key in ('lxy', 'dvx', 'dvy', 'vz', 've', 'force'):
            assert len(entry[key]) == 6


def test_the_six_keys_hold_the_first_six_rows_of_the_table_of_the_path(plotted):
    # The vendored code writes data[0] ... data[5] of its table, whose columns are the profiles
    # along the path (length, derivatives, integral of the force, energy, force) and whose rows
    # are the points: the keys hold the values at each of the first six points, not the
    # profiles. The step writes what the code gives (A98).
    inputs = inputs_of(plotted)
    cell = np.array(inputs.cell)
    solver = PES(np.array(inputs.pes_jm2), cell)
    solver.make_pes(replicate_of=(5, 5), density=20, tol=1e-4)
    string = MEP(rbf=solver._rbf, cell=cell)
    string.get_mep(method='zerotemp', optimization='x', **FAST)
    _, table = string.get_shear_strenght(delta=0.01, npts=FAST['npts'], optimization='x')
    assert table.shape == (FAST['npts'], 6)
    entry = by_the_vendored_code(inputs, mep=FAST)['along_x']
    for row, key in enumerate(('lxy', 'dvx', 'dvy', 'vz', 've', 'force')):
        assert entry[key] == table[row].tolist()


def test_same_values_as_the_vendored_code(analysed):
    expected = by_the_vendored_code(inputs_of(analysed), mep=FAST)
    assert document(analysed)['shearstr'] == expected


def test_the_arguments_of_the_string_method_are_the_settings():
    params = ShearParams(mep=ShearMepParams(npts=15, extent=[1.0, 1.25], h=0.02, nstepmax=12,
                                            tol=1e-2, delta=1e-3))
    arguments = shear.mep_arguments(params)
    assert arguments == {'npts': 15, 'extent': [1.0, 1.25], 'h': 0.02, 'nstepmax': 12,
                         'tol': 1e-2, 'delta': 1e-3}
    # A copy: the algorithm may not change the settings.
    arguments['extent'][0] = 9.0
    assert params.mep.extent == [1.0, 1.25]


def test_the_default_arguments_are_the_defaults_of_the_string_method():
    # What the original left to the defaults of the MEP is given explicitly, with the same values.
    assert shear.mep_arguments(ShearParams()) == read_default_params(MEP.defaults, 'zerotemp', {})


def test_the_default_arguments_give_the_numbers_of_the_omitted_ones(plotted):
    # The algorithm stopped early to be short: explicit values equal the defaults that it
    # fills in itself.
    inputs = inputs_of(plotted)
    cell = np.array(inputs.cell)
    solver = PES(np.array(inputs.pes_jm2), cell)
    solver.make_pes(replicate_of=(5, 5), density=20, tol=1e-4)
    results = []
    for arguments in ({'nstepmax': 30}, {**shear.mep_arguments(ShearParams()), 'nstepmax': 30}):
        string = MEP(rbf=solver._rbf, cell=cell)
        string.get_mep(method='zerotemp', optimization='x', **arguments)
        string.get_shear_strenght(delta=0.01, npts=101, optimization='x')
        results.append((string.mep, json.dumps(string.mep_data)))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_the_calls_are_those_of_the_original(plotted_copy, monkeypatch):
    spy = Calls(monkeypatch)
    shear.ShearStep().analyse(plotted_copy, CUCU, [])
    inputs = inputs_of(plotted_copy)
    assert spy.names() == ['PES', 'make_pes'] + ['MEP', 'get_mep', 'get_shear_strenght'] * 3
    pes_call, make_call = spy.calls[:2]
    np.testing.assert_array_equal(pes_call[1], np.array(inputs.pes_jm2))
    np.testing.assert_array_equal(pes_call[2], np.array(inputs.cell))
    assert make_call[1] == {'replicate_of': (5, 5), 'density': 20, 'tol': 1e-4}
    rbf = spy.calls[2][1]
    for k, opt in enumerate(OPTIMIZATIONS):
        string, mep, ss = spy.calls[2 + 3 * k:5 + 3 * k]
        # The same interpolation of the PES for the three strings, and the same cell.
        assert string[1] is rbf
        np.testing.assert_array_equal(string[2], np.array(inputs.cell))
        assert mep[1:4] == ('zerotemp', opt, FAST)
        assert ss[1:] == (0.01, FAST['npts'], opt)


def test_the_parameters_of_the_settings_reach_the_algorithm(tmp_path, plotted, monkeypatch):
    run_dir = tmp_path / 'other'
    shutil.copytree(plotted, run_dir)
    other = {'npts': 15, 'extent': [1.0, 1.25], 'h': 0.02, 'nstepmax': 12, 'tol': 1e-2,
             'delta': 1e-3}
    with_shear(run_dir, delta=0.02, seed=5, mep=other)
    spy = Calls(monkeypatch)
    shear.ShearStep().analyse(run_dir, CUCU, [])
    mep_calls = [call for call in spy.calls if call[0] == 'get_mep']
    assert [call[3] for call in mep_calls] == [other] * 3
    assert [call[1:] for call in spy.calls if call[0] == 'get_shear_strenght'] == \
        [(0.02, 15, opt) for opt in OPTIMIZATIONS]
    # The same numbers as the vendored code called by hand with the same parameters.
    assert document(run_dir)['shearstr'] == by_the_vendored_code(inputs_of(run_dir),
                                                                 shear_delta=0.02, seed=5,
                                                                 mep=other)
    # The maximum number of steps of the settings stops the strings.
    assert all(entry['nstep'] <= other['nstepmax']
               for entry in document(run_dir)['shearstr'].values())


def test_the_least_points_of_the_string_are_enough(plotted, tmp_path):
    # The entries hold six numbers under each key (A98), taken from the first six rows of
    # the table of the path: a string of six points is the shortest that can be written.
    assert SHEAR_MIN_POINTS == 6
    run_dir = tmp_path / 'six'
    shutil.copytree(plotted, run_dir)
    with_shear(run_dir, mep=FAST | {'npts': SHEAR_MIN_POINTS, 'nstepmax': 10})
    shear.ShearStep().analyse(run_dir, CUCU, [])
    entries = document(run_dir)['shearstr']
    assert list(entries) == [f'along_{opt}' for opt in OPTIMIZATIONS]
    assert all(set(entry) == KEYS and entry['shear_strength_GPa'] > 0 for entry in entries.values())
    assert shear.ShearStep().done(run_dir, CUCU)


def test_the_largest_seed_of_numpy_runs(plotted, tmp_path):
    assert SHEAR_MAX_SEED == 2**32 - 1
    run_dir = tmp_path / 'seed'
    shutil.copytree(plotted, run_dir)
    with_shear(run_dir, seed=SHEAR_MAX_SEED, mep=FAST | {'nstepmax': 5})
    shear.ShearStep().analyse(run_dir, CUCU, [])
    assert shear.ShearStep().done(run_dir, CUCU)
    assert read_system(sdir(run_dir)).provenance(LEAVES[0]).parameters['seed'] == SHEAR_MAX_SEED


def test_the_seed_is_set_before_each_string(plotted_copy, monkeypatch):
    spy = Calls(monkeypatch)
    shear.ShearStep().analyse(plotted_copy, CUCU, [])
    np.random.seed(0)
    state = np.random.get_state()[1]
    assert all(np.array_equal(call[4], state) for call in spy.calls if call[0] == 'get_mep')


def test_the_seed_changes_the_string_of_bs_line_only(plotted, tmp_path):
    runs = {}
    for seed in (0, 1):
        runs[seed] = tmp_path / f'seed{seed}'
        shutil.copytree(plotted, runs[seed])
        config = (runs[seed] / 'config.toml').read_text(encoding='utf-8')
        (runs[seed] / 'config.toml').write_text(
            config.replace('[interface.shear]', f'[interface.shear]\nseed = {seed}'),
            encoding='utf-8')
        shear.ShearStep().analyse(runs[seed], CUCU, [])
    first, second = (document(runs[seed])['shearstr'] for seed in (0, 1))
    assert first['along_x'] == second['along_x'] and first['along_y'] == second['along_y']
    assert first['along_bs_line'] != second['along_bs_line']
    assert second == by_the_vendored_code(inputs_of(runs[1]), seed=1, mep=FAST)


def test_the_state_of_the_random_generator_is_kept(plotted_copy):
    np.random.seed(123)
    state = np.random.get_state()
    shear.ShearStep().analyse(plotted_copy, CUCU, [])
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])


def test_the_message_of_the_analysis(plotted_copy, caplog):
    with caplog.at_level(logging.INFO, logger=shear.__name__):
        shear.ShearStep().analyse(plotted_copy, CUCU, [])
    lines = [r.getMessage() for r in caplog.records if r.name == shear.__name__
             and r.levelno == logging.INFO]
    entries = document(plotted_copy)['shearstr']
    first = f'{CUCU}: shear strength along bs_line ' \
            f'{entries["along_bs_line"]["shear_strength_GPa"]:.4f} GPa'
    (line,) = [line for line in lines if line.startswith(f'{CUCU}: shear strength')]
    assert line.startswith(first) and 'along x ' in line and 'along y ' in line
    # The part of the interpolation and of each string, while they run and in the message.
    assert lines[0].startswith('interpolation of the PES: ') and '800 centres' in lines[0]
    assert [line.split(':')[0] for line in lines[1:4]] == ['string bs_line', 'string x',
                                                           'string y']
    assert any(line.startswith(f'{CUCU}: PES of 8 points, 800 centres') for line in lines)
    # With the parameters of the example run, which are not the recommended ones, they are told.
    other = [line for line in lines if 'other than the recommended' in line]
    assert len(other) == 1
    for text in ('mep.npts = 21 (recommended 101)', 'mep.nstepmax = 25 (recommended 99999)',
                 'mep.tol = 0.001 (recommended 1e-08)'):
        assert text in other[0]
    assert 'mep.h' not in other[0] and 'seed' not in other[0]


def test_the_recommended_parameters_are_not_told():
    assert shear.not_recommended(ShearParams()) == []


@pytest.mark.parametrize('changes, told', [
    ({'delta': 0.02}, ['delta = 0.02 (recommended 0.01)']),
    ({'mep': {'extent': [1.0, 1.2]}}, ['mep.extent = [1.0, 1.2] (recommended [1.5, 1.5])']),
    ({'mep': {'h': 0.001, 'tol': 1e-7, 'delta': 0.01}},
     ['mep.h = 0.001 (recommended 0.01)', 'mep.tol = 1e-07 (recommended 1e-08)',
      'mep.delta = 0.01 (recommended 0.0001)']),
    ({'seed': 4}, []),
])
def test_the_parameters_other_than_the_recommended_are_told(changes, told):
    mep = ShearMepParams(**changes.get('mep', {}))
    params = ShearParams(**{k: v for k, v in changes.items() if k != 'mep'}, mep=mep)
    assert shear.not_recommended(params) == told


def test_provenance_of_the_entries(analysed):
    document_ = read_system(sdir(analysed))
    recorded = [document_.provenance(leaf) for leaf in LEAVES]
    assert recorded[0] == shear.ShearStep().provenance(analysed, CUCU)
    assert all(provenance == recorded[0] for provenance in recorded)
    assert recorded[0].step == 'shear'
    assert recorded[0].parameters == {
        'method': 'zerotemp', 'delta': 0.01, 'seed': 0,
        'mep': {'npts': 21, 'extent': [1.5, 1.5], 'h': 0.01, 'nstepmax': 25, 'tol': 0.001,
                'delta': 0.0001},
        'replicate_of': [5, 5], 'density': 20, 'tol': 1e-4, 'optimizations': list(OPTIMIZATIONS)}


def test_the_entries_of_the_other_steps_stay(analysed, plotted):
    before, after = document(plotted), document(analysed)
    assert {key: value for key, value in after.items() if key != 'shearstr'} == before
    read = read_system(sdir(analysed))
    for leaf in plotpes.interpolation_leaves('pes'):
        assert read.provenance(leaf) == read_system(sdir(plotted)).provenance(leaf)
    assert plotpes.RelaxedPlotPesStep().done(analysed, CUCU)
    assert pes.RelaxedPesStep().done(analysed, CUCU)
    assert MatchStep().done(analysed, CUCU)


def test_analysis_again_gives_the_same_document(analysed_copy):
    before = document(analysed_copy)
    shear.ShearStep().analyse(analysed_copy, CUCU, [])
    assert document(analysed_copy) == before
    assert shear.ShearStep().done(analysed_copy, CUCU)


# --- done ---------------------------------------------------------------------------------

def test_done_only_after_the_analysis(plotted_copy):
    assert not shear.ShearStep().done(plotted_copy, CUCU)
    shear.ShearStep().analyse(plotted_copy, CUCU, [])
    assert shear.ShearStep().done(plotted_copy, CUCU)


def test_done_without_running_the_string_method(analysed_copy, monkeypatch):
    spy = Calls(monkeypatch)
    assert shear.ShearStep().done(analysed_copy, CUCU)
    assert spy.calls == []


@pytest.mark.parametrize('leaf', LEAVES)
def test_done_needs_the_three_entries(analysed_copy, leaf):
    with SystemWriter(sdir(analysed_copy)) as writer:
        writer.document.replace(leaf, {})
        writer.save()
    assert not shear.ShearStep().done(analysed_copy, CUCU)


@pytest.mark.parametrize('old, new', [
    ('[interface.shear]', '[interface.shear]\ndelta = 0.02'),
    ('[interface.shear]', '[interface.shear]\nseed = 1'),
    ('npts = 21', 'npts = 22'),
    ('extent = [1.5, 1.5]', 'extent = [1.5, 1.4]'),
    ('h = 0.01', 'h = 0.02'),
    ('nstepmax = 25', 'nstepmax = 26'),
    ('tol = 0.001', 'tol = 0.002'),
    ('delta = 0.0001', 'delta = 0.0002'),
])
def test_done_follows_the_parameters(analysed_copy, old, new):
    # Any parameter of the analysis changed, and the entries are no longer current.
    assert shear.ShearStep().done(analysed_copy, CUCU)
    edit_config(analysed_copy, old, new)
    assert not shear.ShearStep().done(analysed_copy, CUCU)


def test_done_follows_the_pes_read(analysed_copy):
    provenance = read_system(sdir(analysed_copy)).provenance(('pes', 'energy', 'pes_jm2'))
    with SystemWriter(sdir(analysed_copy)) as writer:
        points = writer.document.get(['pes', 'energy', 'pes_jm2'])
        writer.document.set(['pes', 'energy', 'pes_jm2'],
                            [[x, y, e + 1e-9] for x, y, e in points])
        writer.document.set_provenance(('pes', 'energy', 'pes_jm2'), provenance)
        writer.save()
    # pes_jm2 keeps its provenance of plotpes, but the points read changed.
    assert not shear.ShearStep().done(analysed_copy, CUCU)


def test_done_follows_the_cell(analysed_copy):
    provenance = read_system(sdir(analysed_copy)).provenance(('structure', 'init', 'interface'))
    with SystemWriter(sdir(analysed_copy)) as writer:
        matrix = writer.document.get(['structure', 'init', 'interface', 'lattice', 'matrix'])
        matrix[0][0] += 1e-6
        writer.document.set(['structure', 'init', 'interface', 'lattice', 'matrix'], matrix)
        writer.document.set_provenance(('structure', 'init', 'interface'), provenance)
        writer.save()
    assert not shear.ShearStep().done(analysed_copy, CUCU)


def test_done_follows_its_pes_and_match(analysed_copy):
    # The relaxed PES was completed again and not analysed: neither it nor what it feeds is done.
    calc_id = pes.RelaxedPesStep().calculations(analysed_copy, CUCU)[0].calc_id
    record = store.read_record(sdir(analysed_copy), calc_id)
    results = {**record.results, 'output': {**record.results['output'], 'energy': -999.0}}
    store.write_record(sdir(analysed_copy), dataclasses.replace(record, results=results))
    assert not pes.RelaxedPesStep().done(analysed_copy, CUCU)
    assert not shear.ShearStep().done(analysed_copy, CUCU)


def test_done_follows_match(analysed_copy):
    assert shear.ShearStep().done(analysed_copy, CUCU)
    edit_config(analysed_copy, 'max_area = 160.0', 'max_area = 150.0')
    assert not MatchStep().done(analysed_copy, CUCU)
    assert not shear.ShearStep().done(analysed_copy, CUCU)


def test_pes_analysed_again_makes_the_shear_stale(analysed_copy):
    calc_ids = [f'pes/{point}' for point in document(analysed_copy)['highsym']['inter_unique']]
    pes.RelaxedPesStep().analyse(analysed_copy, CUCU, calc_ids)
    assert not plotpes.RelaxedPlotPesStep().done(analysed_copy, CUCU)
    assert not shear.ShearStep().done(analysed_copy, CUCU)
    # And the step does not run on a PES that is not interpolated again.
    with pytest.raises(PrerequisiteError, match=r'pes\.energy\.pes_jm2'):
        shear.ShearStep().analyse(analysed_copy, CUCU, [])


# --- errors: all or nothing ---------------------------------------------------------------

def test_failure_of_a_string_writes_nothing(plotted_copy, monkeypatch):
    # The original saved after each string and was done after the first: here nothing is
    # written unless the three strings are.
    class Failing(MEP):
        def get_mep(self, method='zerotemp', optimization='bs_line', **kwargs):
            if optimization == 'y':
                raise MemoryError('no memory left')
            return super().get_mep(method=method, optimization=optimization, **kwargs)

    monkeypatch.setattr(shear, 'MEP', Failing)
    before = system_bytes(plotted_copy)
    with pytest.raises(MemoryError):
        shear.ShearStep().analyse(plotted_copy, CUCU, [])
    assert system_bytes(plotted_copy) == before
    assert 'shearstr' not in document(plotted_copy)
    assert not SystemWriter(sdir(plotted_copy)).lock_path.exists()
    assert not shear.ShearStep().done(plotted_copy, CUCU)


def test_entries_changed_during_the_analysis(plotted_copy, monkeypatch):
    compute = shear.compute_shear

    def change(inputs, params):
        result = compute(inputs, params)
        # Another analysis of the PES in the meantime, with other energies.
        points = document(plotted_copy)['highsym']['inter_unique']
        calc_ids = finish(plotted_copy, 'pes', {p: -1000.0 - 0.03 * i
                                                for i, p in enumerate(points)},
                          prepare=False, system=CUCU)
        pes.RelaxedPesStep().analyse(plotted_copy, CUCU, calc_ids)
        return result

    monkeypatch.setattr(shear, 'compute_shear', change)
    with pytest.raises(PrerequisiteError):
        shear.ShearStep().analyse(plotted_copy, CUCU, [])
    assert 'shearstr' not in document(plotted_copy)
    assert not shear.ShearStep().done(plotted_copy, CUCU)


def test_the_inputs_changed_during_the_analysis_are_refused(plotted_copy, monkeypatch):
    compute = shear.compute_shear

    def change(inputs, params):
        result = compute(inputs, params)
        # The interpolation is written again with another point, keeping its provenance.
        provenance = read_system(sdir(plotted_copy)).provenance(('pes', 'energy', 'pes_jm2'))
        with SystemWriter(sdir(plotted_copy)) as writer:
            points = writer.document.get(['pes', 'energy', 'pes_jm2'])
            writer.document.set(['pes', 'energy', 'pes_jm2'],
                                [[x, y, e + 1e-9] for x, y, e in points])
            writer.document.set_provenance(('pes', 'energy', 'pes_jm2'), provenance)
            writer.save()
        return result

    monkeypatch.setattr(shear, 'compute_shear', change)
    with pytest.raises(StoreError, match='the entries read changed during the analysis'):
        shear.ShearStep().analyse(plotted_copy, CUCU, [])
    assert 'shearstr' not in document(plotted_copy)


# --- the orchestrator and the command line ------------------------------------------------

def statuses(orch):
    return {name: orch.entry(CUCU, name)['status'] for name in orch.order}


def cli(run_dir, *argv):
    return main([*argv, '-r', str(run_dir)])


def no_string(monkeypatch):
    """Record the calls to the analysis, which the orchestrator never makes."""
    calls = []
    compute = shear.compute_shear
    monkeypatch.setattr(shear, 'compute_shear',
                        lambda *args: calls.append(args) or compute(*args))
    return calls


def test_the_passes_never_analyse_the_shear(plotted_copy, fake_slurm, monkeypatch):
    calls = no_string(monkeypatch)
    orch = build_orchestrator(plotted_copy)
    report = orch.run_pass()
    assert statuses(orch) == {'match': 'completed', 'pes': 'completed', 'plotpes': 'completed',
                              'shear': 'manual'}
    assert orch.entry(CUCU, 'shear')['reason'] == (
        'to analyse by hand: run "triboworkflow analyse shear" where there is enough memory, or '
        'give [execution.analysis] to analyse it in a job of its own')
    assert report.finished and not report.success and calls == []


def test_analysis_by_hand_completes_the_step(plotted_copy, fake_slurm, monkeypatch):
    calls = no_string(monkeypatch)
    build_orchestrator(plotted_copy).run_pass()
    assert build_orchestrator(plotted_copy).analyse_step(CUCU, 'shear').status == 'completed'
    assert len(calls) == 1
    orch = build_orchestrator(plotted_copy)
    for _ in range(2):
        assert orch.run_pass().success and statuses(orch) == dict.fromkeys(STEPS, 'completed')
    assert len(calls) == 1 and all(get_step(name).done(plotted_copy, CUCU) for name in STEPS)


def test_analyse_by_hand_from_the_command_line(plotted_copy, fake_slurm, fake_scheduler_bin,
                                               monkeypatch, capsys):
    calls = no_string(monkeypatch)
    assert cli(plotted_copy, 'prepare', 'shear') == 0  # left to the user, not an error
    assert f'{CUCU}/shear: manual (to analyse by hand: run "triboworkflow analyse shear"' in \
        capsys.readouterr().out
    assert cli(plotted_copy, 'submit', 'shear') == 1
    assert 'it is analysed by hand, with "analyse shear"' in capsys.readouterr().err
    assert calls == []
    assert cli(plotted_copy, 'analyse', 'shear') == 0
    assert f'{CUCU}/shear: completed' in capsys.readouterr().out and len(calls) == 1
    # Again: recognised from the entries, nothing computed.
    before = run_files(plotted_copy)
    assert cli(plotted_copy, 'analyse', 'shear') == 0
    capsys.readouterr()
    assert len(calls) == 1
    assert {p: d for p, d in run_files(plotted_copy).items() if p != 'cli.log'} == \
        {p: d for p, d in before.items() if p != 'cli.log'}


def test_shear_without_plotpes_in_the_run_is_blocked(tmp_path, fake_slurm):
    run_dir = matched_run(tmp_path, steps=('match', 'pes', 'shear'))
    pes.RelaxedPesStep().analyse(run_dir, CUCU, finish(run_dir, 'pes', energies_of(run_dir),
                                                       system=CUCU))
    orch = build_orchestrator(run_dir)
    orch.run_pass()
    assert statuses(orch) == {'match': 'completed', 'pes': 'completed', 'shear': 'blocked'}
    reason = orch.entry(CUCU, 'shear')['reason']
    assert 'entries missing or not current' in reason and 'pes.energy.pes_jm2' in reason


def test_shear_waits_for_plotpes_in_the_run(tmp_path, fake_slurm):
    run_dir = matched_run(tmp_path)
    pes.RelaxedPesStep().analyse(run_dir, CUCU, finish(run_dir, 'pes', energies_of(run_dir),
                                                       system=CUCU))
    orch = build_orchestrator(run_dir)
    orch.run_pass()
    # plotpes is left to the user and shear, which needs it, does not go on.
    assert statuses(orch) == {'match': 'completed', 'pes': 'completed', 'plotpes': 'manual',
                              'shear': 'blocked'}
    assert orch.entry(CUCU, 'shear')['reason'] == 'depends on plotpes manual'


def test_the_job_analyses_while_the_orchestrator_holds_the_run(plotted_copy, analysed):
    path = analysis_group(plotted_copy, 'shear')
    with RunLock(plotted_copy, 'triboworkflow start'):  # the orchestrator in the background
        outcome = run_analysis(path)
    assert (outcome['result'], outcome['error']) == ('analysed', None)
    assert shear.ShearStep().done(plotted_copy, CUCU)
    # The entries of the analysis by hand, with their provenance.
    found, expected = read_system(sdir(plotted_copy)), read_system(sdir(analysed))
    for leaf in LEAVES:
        assert found.get(leaf) == expected.get(leaf)
        assert found.provenance(leaf) == expected.provenance(leaf)
    assert not (plotted_copy / 'run.lock').exists()
    # A job again (a late one): recognised from the entries, nothing computed.
    before = system_bytes(plotted_copy)
    assert run_analysis(path)['result'] == 'done'
    assert system_bytes(plotted_copy) == before


def test_the_job_without_the_pes(tmp_path):
    run_dir = matched_run(tmp_path)
    before = system_bytes(run_dir)
    outcome = run_analysis(analysis_group(run_dir, 'shear'))
    assert outcome['result'] == 'blocked' and 'pes.energy.pes_jm2' in outcome['error']
    assert system_bytes(run_dir) == before


def test_a_run_analyses_the_shear_in_a_job_of_its_own(tmp_path, plotted, fake_slurm):
    run_dir = tmp_path / 'run'
    shutil.copytree(plotted, run_dir)
    # plotpes is done: the job of the shear is the only one.
    with_analysis_job(run_dir)
    orch = build_orchestrator(run_dir)
    report = orch.run_pass()
    group = f'{CUCU}.shear.1'
    assert statuses(orch) == {'match': 'completed', 'pes': 'completed', 'plotpes': 'completed',
                              'shear': 'running'}
    assert report.submitted == [group] and fake_slurm.submitted == 1
    assert not shear.ShearStep().done(run_dir, CUCU)  # the orchestrator never analyses
    assert the_node_runs(run_dir, fake_slurm, group)['result'] == 'analysed'
    for _ in range(2):
        report = orch.run_pass()
        assert report.success and statuses(orch) == dict.fromkeys(STEPS, 'completed')
    assert fake_slurm.submitted == 1 and shear.ShearStep().done(run_dir, CUCU)
