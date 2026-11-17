# Author: Vito Foderà
# Created: 2026-11-14

"""Tests of the job and the results of an MLIP calculation with ASE
(``triboworkflow.engines.mlip.ase_job``).

The executor runs in this process with ASE's EMT in place of the model (``mlip_ase_helpers``):
what is checked is the exchange of files, the states and the units, not a model.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from pymatgen.core import Lattice, Structure

import mlip_ase_helpers as helpers
from mlip_ase_helpers import helper_call, run, slab
from triboworkflow.core.settings.schema import MlipParams
from triboworkflow.core.structures import extxyz_text, read_extxyz_text
from triboworkflow.engines.mlip import ase_job, ase_run
from triboworkflow.engines.mlip.registry import ModelError
from triboworkflow.engines.qe.results import Parsed

AseStatus = ase_job.AseStatus


@pytest.fixture(autouse=True)
def fake_cuda(monkeypatch):
    monkeypatch.setattr(ase_run, 'check_cuda', lambda: dict(helpers.FAKE_DEVICE))


def reason_of(directory, calculation, structure):
    """Reason of a reading that failed."""
    parsed = ase_job.parse_result(directory, calculation, structure)
    assert parsed.parsing == 'failed' and parsed.results is None and parsed.structure is None
    return parsed.reason


# --- job -----------------------------------------------------------------------------------

def test_job_of_a_deepmd_model():
    mlip = MlipParams(type='DP', path='/models/dp.pb', fmax=0.05, maxsteps=7)
    assert ase_job.job_of('relax', mlip, max_seconds=3600) == {
        'calculation': 'relax',
        'model': {'factory': 'deepmd.calculator:DP', 'args': ['/models/dp.pb'], 'kwargs': {}},
        'package': 'deepmd-kit', 'model_path': '/models/dp.pb', 'fmax': 0.05, 'maxsteps': 7,
        'max_seconds': 3600}


def test_job_of_uses_the_defaults_of_the_settings_without_a_time_budget():
    mlip = MlipParams(type='DP', path='/models/dp.pb')
    job = ase_job.job_of('scf', mlip)
    assert (job['fmax'], job['maxsteps']) == (mlip.fmax, mlip.maxsteps) == (0.01, 500)
    assert job['max_seconds'] is None


def test_job_of_a_mace_model_passes_the_kwargs_as_written():
    kwargs = {'default_dtype': 'float64', 'head': 'mp_pbe'}
    job = ase_job.job_of('scf', MlipParams(type='MACE', path='/models/m.model', kwargs=kwargs))
    assert job['model'] == {
        'factory': 'mace.calculators:MACECalculator', 'args': [],
        'kwargs': {'model_path': '/models/m.model', 'device': 'cuda', 'default_dtype': 'float64',
                   'head': 'mp_pbe'}}
    assert job['package'] == 'mace-torch'
    assert kwargs == {'default_dtype': 'float64', 'head': 'mp_pbe'}


@pytest.mark.parametrize('model_type', ['MACE', 'MACE_MP'])
def test_a_mace_model_without_default_dtype_is_refused_at_launch(model_type):
    with pytest.raises(ModelError, match='default_dtype'):
        ase_job.job_of('relax', MlipParams(type=model_type, path='/models/m.model'))


def test_job_of_refuses_another_calculation():
    with pytest.raises(ValueError, match="'md'"):
        ase_job.job_of('md', MlipParams(type='DP', path='/models/dp.pb'))


def test_the_job_written_is_the_one_the_executor_reads(tmp_path):
    job = ase_job.job_of('relax', MlipParams(type='DP', path='/models/dp.pb', fmax=0.02,
                                             maxsteps=300), max_seconds=120)
    path = ase_job.write_job(tmp_path, job)
    assert path == tmp_path / ase_job.JOB_FILE
    assert json.loads(path.read_text()) == job
    # Same job and files, with EMT where the model would be: the keys and file names agree.
    (tmp_path / ase_run.STRUCTURE_FILE).write_text(extxyz_text(slab()))
    ase_job.write_job(tmp_path, {**job, 'model': helpers.EMT_CALL})
    assert ase_run.main([str(path)]) == 0
    result = ase_job.read_result(tmp_path)
    assert result['ok'] and result['geometry'] == 'converged'
    assert result['observed']['model_path'] == '/models/dp.pb'
    assert result['observed']['versions']['deepmd-kit'] is None


def test_the_executor_is_the_script_of_the_package():
    assert ase_job.EXECUTOR_PATH == Path(ase_run.__file__)


# --- stress --------------------------------------------------------------------------------

def cu_bulk(scale):
    atoms = bulk('Cu', cubic=True)
    atoms.set_cell(atoms.cell[:] * scale, scale_atoms=True)
    return Structure(Lattice(atoms.cell[:]), ['Cu'] * len(atoms), atoms.get_positions(),
                     coords_are_cartesian=True)


def test_the_factor_of_the_stress_is_the_one_of_the_units_of_ase():
    assert ase_job.EV_PER_A3_TO_KBAR == pytest.approx(10 / units.GPa, rel=1e-7)


def test_stress_to_kbar_changes_the_sign_and_the_units():
    stress = [[0.1, 0.0, 0.01], [0.0, -0.2, 0.0], [0.01, 0.0, 0.3]]
    assert np.allclose(ase_job.stress_to_kbar(stress), -np.array(stress) * 1602.176634)


@pytest.mark.parametrize('scale, sign', [(0.97, 1), (1.03, -1)])
def test_the_pressure_of_the_stress_is_minus_the_derivative_of_the_energy(tmp_path, scale, sign):
    structure = cu_bulk(scale)
    run(tmp_path / 'calc', structure, calculation='scf')
    stress = ase_job.parse_result(tmp_path / 'calc', 'scf', structure).results['output']['stress']
    pressure = np.trace(stress) / 3

    delta = 1e-4
    energies, volumes = [], []
    for name, factor in (('low', 1 - delta), ('high', 1 + delta)):
        strained = cu_bulk(scale * factor)
        energies.append(run(tmp_path / name, strained, calculation='scf')['energy'])
        volumes.append(strained.volume)
    derivative = (energies[1] - energies[0]) / (volumes[1] - volumes[0])          # eV/A^3
    assert pressure * sign > 100                  # a compressed cell has a positive pressure
    assert pressure == pytest.approx(-derivative * 10 / units.GPa, rel=1e-3)


# --- states --------------------------------------------------------------------------------

def test_states_of_a_single_point(tmp_path):
    run(tmp_path, slab(), calculation='scf')
    assert ase_job.ase_status(tmp_path) == AseStatus('ended', 'n/a', 'n/a', 'completed')


def test_states_of_a_relaxation_that_converged(tmp_path):
    result = run(tmp_path, slab(), fmax=0.01)
    status = ase_job.ase_status(tmp_path)
    assert (status.process, status.scf, status.geometry, status.stop_reason) == (
        'ended', 'n/a', 'converged', 'completed')
    assert status.steps == result['steps'] and status.fallback is None
    assert status.steps['total'] == status.steps['BFGSLineSearch'] + status.steps['FIRE2'] > 0


def test_a_relaxation_that_ran_out_of_steps_is_over_but_not_read(tmp_path):
    structure = slab()
    run(tmp_path, structure, fmax=1e-8, maxsteps=3)
    status = ase_job.ase_status(tmp_path)
    assert (status.process, status.geometry, status.stop_reason) == ('ended', 'max_steps',
                                                                       'max_steps')
    assert status.steps == {'BFGSLineSearch': 3, 'FIRE2': 3, 'total': 6}
    assert status.fallback == 'max_steps'
    assert ase_job.parse_result(tmp_path, 'relax', structure) == Parsed(None)


def test_a_relaxation_that_ran_out_of_time_is_over_but_not_read(tmp_path):
    structure = slab()
    run(tmp_path, structure, model=helper_call('slow_emt', delay=0.05), fmax=1e-8,
        maxsteps=500, max_seconds=0.5)
    status = ase_job.ase_status(tmp_path)
    assert (status.process, status.geometry, status.stop_reason) == (
        'ended', 'not_converged', 'max_seconds')
    assert ase_job.parse_result(tmp_path, 'relax', structure) == Parsed(None)


def test_a_failed_execution_is_a_crash_with_its_error(tmp_path):
    structure = slab()
    run(tmp_path, structure, calculation='scf', model=helper_call('nan_emt'))
    assert ase_job.ase_status(tmp_path) == AseStatus(
        'crashed', None, None, 'error', error='ValueError: non-finite energy')
    assert ase_job.parse_result(tmp_path, 'scf', structure) == Parsed(None)


def test_no_result_means_that_the_executor_did_not_end(tmp_path):
    assert ase_job.read_result(tmp_path) is None
    assert ase_job.ase_status(tmp_path) == AseStatus(None, None, None, 'truncated')
    assert ase_job.parse_result(tmp_path, 'relax', slab()) == Parsed(None)


@pytest.mark.parametrize('text, message', [
    ('{"ok": tru', 'not the result of the executor'),
    ('[]', 'not the result of the executor'),
    ('{"calculation": "scf"}', 'not the result of the executor'),
    ('{"ok": true, "calculation": "scf"}', 'lacks geometry'),
    ('{"ok": false}', 'lacks error')])
def test_a_damaged_result_is_an_error_of_the_states_and_a_failed_reading(tmp_path, text, message):
    (tmp_path / ase_run.RESULT_FILE).write_text(text)
    with pytest.raises(ase_job.AseResultError, match=message):
        ase_job.ase_status(tmp_path)
    assert message in reason_of(tmp_path, 'scf', slab())


def test_a_result_that_is_not_text_is_a_damaged_result(tmp_path):
    (tmp_path / ase_run.RESULT_FILE).write_bytes(b'\xff\xfe{"ok": true}')
    assert 'not the result of the executor' in reason_of(tmp_path, 'scf', slab())


# --- results -------------------------------------------------------------------------------

def test_results_of_a_relaxation_have_the_keys_and_the_units_of_the_record(tmp_path):
    structure = slab(if_pos=[[0, 0, 0]] * 4 + [[1, 1, 1]] * 8)
    structure.add_site_property('magmom', [0.0] * len(structure))
    result = run(tmp_path, structure, fmax=0.01)
    parsed = ase_job.parse_result(tmp_path, 'relax', structure)
    assert parsed.parsing == 'ok' and parsed.reason is None and parsed.values is None

    results, output = parsed.results, parsed.results['output']
    assert set(results) == {'nsites', 'input', 'output'} and results['nsites'] == 12
    assert set(output) == {'structure', 'density', 'energy', 'energy_per_atom', 'forces', 'stress'}
    assert output['energy'] == result['energy']
    assert output['energy_per_atom'] == pytest.approx(result['energy'] / 12, abs=1e-12)
    assert output['forces'] == result['forces']
    assert np.array(output['forces']).shape == (12, 3) and np.array(output['stress']).shape == (3, 3)
    assert np.allclose(output['stress'], ase_job.stress_to_kbar(result['stress']))

    final = Structure.from_dict(output['structure'])
    assert np.allclose(final.cart_coords, result['positions'], atol=1e-12)
    assert np.allclose(final.lattice.matrix, structure.lattice.matrix)
    assert final.labels == structure.labels
    assert final.site_properties['interface_region'] == structure.site_properties['interface_region']
    assert final.site_properties['if_pos'] == structure.site_properties['if_pos']
    assert 'magmom' not in final.site_properties and 'magmom' in structure.site_properties
    assert output['density'] == pytest.approx(final.density)
    assert output['density'] > 0
    assert np.allclose(Structure.from_dict(results['input']['structure']).cart_coords,
                       structure.cart_coords)
    assert np.allclose(parsed.structure.cart_coords, final.cart_coords)
    assert parsed.observed == result['observed']
    json.dumps(results, allow_nan=False)


def test_the_text_of_the_final_structure_reads_back(tmp_path):
    structure = slab(if_pos=[[0, 0, 0]] * 4 + [[1, 1, 1]] * 8)
    run(tmp_path, structure)
    parsed = ase_job.parse_result(tmp_path, 'relax', structure)
    written = read_extxyz_text(parsed.structure_text)
    assert np.allclose(written.cart_coords, parsed.structure.cart_coords, atol=1e-7)
    assert written.site_properties['interface_region'] == structure.site_properties['interface_region']
    assert written.site_properties['if_pos'] == structure.site_properties['if_pos']


def test_results_of_a_single_point_keep_the_positions_of_the_input(tmp_path):
    structure = slab()
    result = run(tmp_path, structure, calculation='scf')
    parsed = ase_job.parse_result(tmp_path, 'scf', structure)
    assert parsed.parsing == 'ok'
    # Not the positions of the result (rounded by the extxyz of the input): the input's own.
    assert np.allclose(parsed.structure.cart_coords, structure.cart_coords, atol=1e-12)
    assert parsed.results['output']['energy'] == result['energy']


def test_a_model_without_stress_gives_results_without_stress(tmp_path):
    structure = slab()
    run(tmp_path, structure, calculation='scf')
    path = tmp_path / ase_run.RESULT_FILE
    path.write_text(json.dumps({**json.loads(path.read_text()), 'stress': None}))
    parsed = ase_job.parse_result(tmp_path, 'scf', structure)
    assert parsed.parsing == 'ok' and 'stress' not in parsed.results['output']


# --- result and input ----------------------------------------------------------------------

def test_a_result_of_another_calculation_fails_the_reading(tmp_path):
    structure = slab()
    run(tmp_path, structure, calculation='scf')
    assert "'scf' calculation, not 'relax'" in reason_of(tmp_path, 'relax', structure)


def test_a_result_of_a_structure_with_another_number_of_atoms_fails_the_reading(tmp_path):
    structure = slab()
    run(tmp_path, structure, calculation='scf')
    smaller = structure.copy()
    smaller.remove_sites([11])
    assert 'has 12 atoms, the input structure 11' in reason_of(tmp_path, 'scf', smaller)


def test_a_result_of_another_element_fails_the_reading(tmp_path):
    structure = slab()
    run(tmp_path, structure, calculation='scf')
    other = structure.copy()
    other.replace(0, 'Ag')
    assert 'elements of the result' in reason_of(tmp_path, 'scf', other)


def test_a_result_of_another_cell_fails_the_reading(tmp_path):
    structure = slab()
    run(tmp_path, structure, calculation='scf')
    larger = Structure(Lattice(structure.lattice.matrix * 1.01), structure.species,
                       structure.frac_coords, labels=structure.labels)
    assert 'cell of the result' in reason_of(tmp_path, 'scf', larger)


def test_a_single_point_must_keep_the_positions_of_the_input_and_a_relaxation_need_not(tmp_path):
    structure = slab()
    moved = structure.copy()
    moved.translate_sites([0], [0.05, 0.0, 0.0], frac_coords=False, to_unit_cell=False)
    run(tmp_path / 'scf', structure, calculation='scf')
    assert 'positions of the result' in reason_of(tmp_path / 'scf', 'scf', moved)
    run(tmp_path / 'relax', structure)
    assert ase_job.parse_result(tmp_path / 'relax', 'relax', moved).parsing == 'ok'
