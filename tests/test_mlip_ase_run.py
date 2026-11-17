# Author: Vito Foderà
# Created: 2026-11-14

"""Tests of the ASE executor (``triboworkflow.engines.mlip.ase_run``).

The model is ASE's EMT (``mlip_ase_helpers``): no MLIP library is installed, no GPU is needed
(``check_cuda`` is replaced where the executor runs in this process), and nothing the tests run
is a calculation of the workflow.
"""

import importlib.util
import json
import pathlib
import subprocess
import sys

import ase
import numpy as np
import pytest
from ase.calculators.emt import EMT
from ase.io import read
from pymatgen.core import Structure

import mlip_ase_helpers as helpers
from mlip_ase_helpers import helper_call, run, slab
from triboworkflow.core.structures import read_structure
from triboworkflow.engines.mlip import ase_run

FMAX = 0.01


@pytest.fixture(autouse=True)
def fake_cuda(monkeypatch):
    monkeypatch.setattr(ase_run, 'check_cuda', lambda: dict(helpers.FAKE_DEVICE))


@pytest.fixture(autouse=True)
def in_the_calculation_directory(tmp_path, monkeypatch):
    """The workflow starts the executor in the directory of the calculation: a file written
    with a relative path (a restart file) must land there, not in the directory of the tests."""
    monkeypatch.chdir(tmp_path)


def emt_atoms(directory):
    """The atoms of ``structure_in.extxyz`` with EMT: the reference of the executor."""
    atoms = read(directory / ase_run.STRUCTURE_FILE, format='extxyz')
    atoms.calc = EMT()
    return atoms


def constrained_fmax(atoms):
    """Largest force on an atom with the constraints applied: the criterion of the optimizers."""
    return np.linalg.norm(atoms.get_forces(), axis=1).max()


# --- single point --------------------------------------------------------------------------

def test_single_point(tmp_path):
    result = run(tmp_path, slab(), calculation='scf')
    reference = emt_atoms(tmp_path)
    assert result['ok'] and result['calculation'] == 'scf'
    assert (result['geometry'], result['stop_reason']) == ('n/a', 'completed')
    assert result['steps'] is None and result['fallback'] is None
    assert result['energy'] == pytest.approx(reference.get_potential_energy(), abs=1e-12)
    assert np.allclose(result['forces'], reference.get_forces(), atol=1e-12)
    # The stress is ASE's, eV/A^3 with the sign of ASE: the conversion is the workflow's.
    assert np.allclose(result['stress'], reference.get_stress(voigt=False), atol=1e-12)
    assert np.allclose(result['positions'], reference.get_positions(), atol=1e-12)
    assert result['symbols'] == ['Cu'] * 12
    assert np.allclose(result['cell'], reference.cell[:], atol=1e-12)
    assert not (tmp_path / ase_run.LOG_FILE).exists()


def test_observed(tmp_path):
    result = run(tmp_path, slab(), calculation='scf', package='ase', model_path='/models/a.model')
    observed = result['observed']
    assert observed['versions']['ase'] == ase.__version__
    assert observed['versions']['torch'] == helpers.FAKE_DEVICE['torch']
    assert observed['versions']['python'] == sys.version.split()[0]
    assert observed['device'] == helpers.FAKE_DEVICE['device']
    assert observed['model_path'] == '/models/a.model'
    assert result['elapsed'] >= 0.0


def test_package_not_installed_is_recorded_as_null(tmp_path):
    result = run(tmp_path, slab(), calculation='scf', package='no-such-distribution')
    assert result['observed']['versions']['no-such-distribution'] is None


def test_forces_are_recorded_without_constraints(tmp_path):
    fixed = [[0, 0, 0]] * 4 + [[1, 1, 1]] * 8
    result = run(tmp_path, slab(if_pos=fixed), calculation='scf')
    atoms = emt_atoms(tmp_path)
    assert len(atoms.constraints) == 12
    assert np.allclose(atoms.get_forces(apply_constraint=True)[:4], 0.0)
    raw = np.array(result['forces'])
    assert np.abs(raw[:4]).max() > 0.1
    assert np.allclose(raw, atoms.get_forces(apply_constraint=False), atol=1e-12)


def test_last_structure_of_a_single_point_is_the_input(tmp_path):
    structure = slab()
    run(tmp_path, structure, calculation='scf')
    last = read_structure(tmp_path / ase_run.LAST_STRUCTURE_FILE, format='extxyz')
    assert np.allclose(last.cart_coords, structure.cart_coords, atol=1e-7)


# --- relaxation ----------------------------------------------------------------------------

def test_relax_converges_with_bfgs_line_search(tmp_path):
    result = run(tmp_path, slab(), fmax=FMAX)
    start = emt_atoms(tmp_path)
    assert (result['geometry'], result['stop_reason']) == ('converged', 'completed')
    steps = result['steps']
    assert steps['BFGSLineSearch'] > 0 and steps['FIRE2'] == 0
    assert steps['total'] == steps['BFGSLineSearch'] and result['fallback'] is None
    final = emt_atoms(tmp_path)
    final.positions = result['positions']
    assert constrained_fmax(final) < FMAX
    assert result['energy'] == pytest.approx(final.get_potential_energy(), abs=1e-10)
    assert result['energy'] < start.get_potential_energy()
    assert np.allclose(result['forces'], final.get_forces(), atol=1e-10)
    assert 'BFGSLineSearch:' in (tmp_path / ase_run.LOG_FILE).read_text()


def test_the_cell_is_not_relaxed(tmp_path):
    structure = slab()
    run(tmp_path, structure)
    last = read_structure(tmp_path / ase_run.LAST_STRUCTURE_FILE, format='extxyz')
    assert np.allclose(last.lattice.matrix, structure.lattice.matrix, atol=1e-7)


def test_no_restart_file_is_written_or_read(tmp_path):
    stale = tmp_path / 'restart.json'
    tmp_path.mkdir(exist_ok=True)
    stale.write_text('not an ASE restart')
    result = run(tmp_path, slab(), fmax=FMAX)
    assert result['ok'] and result['geometry'] == 'converged'
    assert stale.read_text() == 'not an ASE restart'
    assert sorted(p.name for p in tmp_path.glob('*.json')) == ['job.json', 'restart.json', 'result.json']


class FailingLineSearch(ase_run.BFGSLineSearch):
    """BFGSLineSearch whose line search fails at its ``fail_at``-th step."""

    fail_at = 2
    error = 'LineSearch failed!'

    def step(self, *args, **kwargs):
        if self.nsteps + 1 == self.fail_at:
            raise RuntimeError(self.error)
        return super().step(*args, **kwargs)


def test_failed_line_search_is_a_lack_of_convergence_and_fire2_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(ase_run, 'BFGSLineSearch', FailingLineSearch)
    result = run(tmp_path, slab(), fmax=0.05)
    assert result['ok'] and result['fallback'] == 'line_search_failed'
    steps = result['steps']
    assert steps['BFGSLineSearch'] == 1 and steps['FIRE2'] > 0
    assert steps['total'] == 1 + steps['FIRE2']
    assert (result['geometry'], result['stop_reason']) == ('converged', 'completed')
    final = emt_atoms(tmp_path)
    final.positions = result['positions']
    assert constrained_fmax(final) < 0.05
    assert not (tmp_path / 'restart.json').exists()


def test_another_runtime_error_is_not_caught(tmp_path, monkeypatch):
    class OutOfMemory(FailingLineSearch):
        error = 'CUDA out of memory'

    monkeypatch.setattr(ase_run, 'BFGSLineSearch', OutOfMemory)
    result = run(tmp_path, slab(), fmax=FMAX)
    assert result == {'ok': False, 'error': 'RuntimeError: CUDA out of memory'}


def test_steps_are_counted_per_optimizer_and_added(tmp_path):
    result = run(tmp_path, slab(), fmax=1e-8, maxsteps=3)
    assert result['steps'] == {'BFGSLineSearch': 3, 'FIRE2': 3, 'total': 6}
    assert result['fallback'] == 'max_steps'
    assert (result['geometry'], result['stop_reason']) == ('max_steps', 'max_steps')


def test_time_runs_out_between_two_steps(tmp_path):
    call = helper_call('slow_emt', delay=0.05)
    result = run(tmp_path, slab(), model=call, fmax=1e-8, maxsteps=500, max_seconds=0.5)
    assert (result['geometry'], result['stop_reason']) == ('not_converged', 'max_seconds')
    steps = result['steps']
    assert 0 < steps['BFGSLineSearch'] < 500 and steps['FIRE2'] == 0
    assert 0.5 <= result['elapsed'] < 3.0
    # The structure of the last step is on disk, and it is the one of the result.
    last = read(tmp_path / ase_run.LAST_STRUCTURE_FILE, format='extxyz')
    assert np.allclose(last.get_positions(), result['positions'], atol=1e-7)


def test_converged_at_the_start_takes_no_step(tmp_path):
    structure = slab()
    first = run(tmp_path / 'a', structure, fmax=1e-4)
    relaxed = Structure(structure.lattice, structure.species, first['positions'],
                        coords_are_cartesian=True, labels=structure.labels)
    again = run(tmp_path / 'b', relaxed, fmax=1e-2)
    assert again['steps'] == {'BFGSLineSearch': 0, 'FIRE2': 0, 'total': 0}
    assert (again['geometry'], again['stop_reason']) == ('converged', 'completed')


# --- constraints ---------------------------------------------------------------------------

def layers_if_pos():
    """Bottom layer fixed, middle layer free along z only, top layer free."""
    return [[0, 0, 0]] * 4 + [[0, 0, 1]] * 4 + [[1, 1, 1]] * 4


def test_constraints_of_the_input_hold_during_the_relaxation(tmp_path):
    structure = slab(if_pos=layers_if_pos())
    result = run(tmp_path, structure, fmax=FMAX)
    start = structure.cart_coords
    end = np.array(result['positions'])
    assert (result['geometry'], result['stop_reason']) == ('converged', 'completed')
    assert np.allclose(end[:4], start[:4], atol=1e-7)
    assert np.allclose(end[4:8, :2], start[4:8, :2], atol=1e-7)
    assert np.abs(end[4:8, 2] - start[4:8, 2]).max() > 1e-3
    assert np.abs(end[8:] - start[8:]).max() > 1e-3


def test_convergence_uses_the_forces_with_constraints_the_record_those_without(tmp_path):
    result = run(tmp_path, slab(if_pos=layers_if_pos()), fmax=FMAX)
    raw = np.linalg.norm(result['forces'], axis=1)
    final = emt_atoms(tmp_path)
    final.positions = result['positions']
    assert result['geometry'] == 'converged'
    assert constrained_fmax(final) < FMAX
    # The fixed atoms feel a force larger than fmax; it does not stop the convergence.
    assert raw[:4].max() > 10 * FMAX
    assert np.allclose(result['forces'], final.get_forces(apply_constraint=False), atol=1e-10)


def test_last_structure_keeps_constraints_and_site_properties(tmp_path):
    structure = slab(if_pos=layers_if_pos())
    run(tmp_path, structure, fmax=FMAX)
    last = read_structure(tmp_path / ase_run.LAST_STRUCTURE_FILE, format='extxyz')
    assert last.site_properties['if_pos'] == structure.site_properties['if_pos']
    assert last.site_properties['interface_region'] == structure.site_properties['interface_region']
    assert list(last.labels) == list(structure.labels)


def test_last_structure_is_replaced_whole_or_not_at_all(tmp_path, monkeypatch):
    helpers.write_job(tmp_path, slab())
    atoms = read(tmp_path / ase_run.STRUCTURE_FILE, format='extxyz')
    path = tmp_path / ase_run.LAST_STRUCTURE_FILE
    ase_run.save_last_structure(atoms, path)
    before = path.read_text()

    def interrupted(temporary, *args, **kwargs):
        temporary.write_text('half of a struc')
        raise KeyboardInterrupt

    monkeypatch.setattr(ase_run, 'write', interrupted)
    atoms.positions += 1.0
    with pytest.raises(KeyboardInterrupt):
        ase_run.save_last_structure(atoms, path)
    assert path.read_text() == before


# --- failures ------------------------------------------------------------------------------

def test_a_result_of_an_earlier_execution_does_not_stand_for_this_one(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / ase_run.RESULT_FILE).write_text(json.dumps({'ok': True, 'energy': 0.0}))
    result = run(tmp_path, slab(), model={'factory': 'no_such_module:f', 'args': [], 'kwargs': {}})
    assert result['ok'] is False and 'no_such_module' in result['error']


def test_a_calculation_that_is_not_scf_or_relax_is_an_error(tmp_path):
    result = run(tmp_path, slab(), calculation='vc-relax')
    assert result == {'ok': False, 'error': "ValueError: calculation 'vc-relax': only "
                                            '"scf" and "relax"'}


def test_the_energy_must_be_finite(tmp_path):
    result = run(tmp_path, slab(), calculation='scf', model=helper_call('nan_emt'))
    assert result == {'ok': False, 'error': 'ValueError: non-finite energy'}


def test_usage():
    assert ase_run.main([]) == 2


# --- GPU -----------------------------------------------------------------------------------

class FakeCuda:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        return self.available

    def current_device(self):
        return 0

    def get_device_name(self, index):
        return f'GPU {index}'


def fake_torch(monkeypatch, available):
    module = type(sys)('torch')
    module.__version__ = '2.5.0'
    module.cuda = FakeCuda(available)
    monkeypatch.setitem(sys.modules, 'torch', module)


def test_check_cuda_reports_torch_and_the_device(monkeypatch):
    monkeypatch.undo()          # the autouse fixture replaced check_cuda
    fake_torch(monkeypatch, True)
    assert ase_run.check_cuda() == {'torch': '2.5.0', 'device': 'cuda:0 GPU 0'}


def test_check_cuda_fails_without_cuda(monkeypatch):
    monkeypatch.undo()
    fake_torch(monkeypatch, False)
    with pytest.raises(RuntimeError, match='CUDA is not available'):
        ase_run.check_cuda()


def test_check_cuda_fails_without_torch(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, 'torch', None)
    with pytest.raises(RuntimeError, match='torch cannot be imported'):
        ase_run.check_cuda()


def test_cuda_is_checked_before_the_model_is_loaded(tmp_path, monkeypatch):
    def no_cuda():
        raise RuntimeError('CUDA is not available: test')

    monkeypatch.setattr(ase_run, 'check_cuda', no_cuda)
    result = run(tmp_path, slab(), model={'factory': 'no_such_module:f', 'args': [], 'kwargs': {}})
    assert result == {'ok': False, 'error': 'RuntimeError: CUDA is not available: test'}
    assert not (tmp_path / ase_run.LAST_STRUCTURE_FILE).exists()


@pytest.mark.skipif(importlib.util.find_spec('torch') is not None,
                    reason='torch is installed: the executor may find a GPU')
def test_executor_started_as_a_script_without_a_gpu_fails_with_a_clear_error(tmp_path):
    job = helpers.write_job(tmp_path, slab())
    done = subprocess.run([sys.executable, ase_run.__file__, str(job)], capture_output=True,
                          text=True, cwd=tmp_path)
    result = json.loads((tmp_path / ase_run.RESULT_FILE).read_text())
    assert done.returncode == 1 and result['ok'] is False
    assert 'torch cannot be imported' in result['error']
    assert 'Traceback' in done.stderr


# --- isolation -----------------------------------------------------------------------------

def test_the_executor_imports_neither_the_workflow_nor_pymatgen_nor_an_mlip_library():
    code = (
        'import importlib.util, sys\n'
        f'spec = importlib.util.spec_from_file_location("ase_run", {ase_run.__file__!r})\n'
        'module = importlib.util.module_from_spec(spec)\n'
        'spec.loader.exec_module(module)\n'
        'loaded = [name for name in sys.modules if name.split(".")[0] in '
        '("triboworkflow", "pymatgen", "torch", "mace", "deepmd")]\n'
        'print(loaded)\n')
    done = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == '[]'
    assert pathlib.Path(ase_run.__file__).name == 'ase_run.py'
