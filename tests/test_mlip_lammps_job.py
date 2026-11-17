# Author: Vito Foderà
# Created: 2026-11-14

"""Tests of the LAMMPS side of the MLIP engine: the input of a run, and the states and results of
its output.

LAMMPS is not run. The input is checked as text; the output is made of files written in the
format the workflow reads, with the numbers of ASE's EMT in the frame of LAMMPS (see
``mlip_lammps_helpers``). EMT lives in the tests only.
"""

import re
import subprocess
import sys

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.lammps.coordinatetransform import Prism
from pymatgen.core import Lattice, Structure

import mlip_lammps_helpers as helpers
from mlip_ase_helpers import slab
from triboworkflow.core.settings.schema import MlipParams
from triboworkflow.engines.mlip import lammps_job
from triboworkflow.engines.mlip.registry import ModelError, ModelType, register_model_type
from triboworkflow.engines.qe.results import Parsed

DEEPMD = MlipParams(type='DP', path='/models/dpa31-3m-omat24.pth')
MACE = MlipParams(type='MACE', path='/models/mace-omat-0-medium-f64.model-mliap_lammps.pt')
BOTTOM_AND_MIDDLE = [[0, 0, 0]] * 4 + [[0, 0, 1]] * 4 + [[1, 1, 1]] * 4

RELAX_OF_DEEPMD = """\
units metal
atom_style atomic
newton on
boundary p p p
read_data structure.data

pair_style deepmd /models/dpa31-3m-omat24.pth
pair_coeff * * Cu

thermo 10
thermo_style custom step pe fmax fnorm press
group blocked_xy id 5:8
fix lock_xy blocked_xy setforce 0.0 0.0 NULL
group blocked_xyz id 1:4
fix lock_xyz blocked_xyz setforce 0.0 0.0 0.0

dump trajectory all custom 10 trajectory.dump id type xu yu zu
dump_modify trajectory sort id format float %.15g flush yes
min_style cg
min_modify norm max
minimize 0.0 0.01 500 1000000

undump trajectory
unfix lock_xy
unfix lock_xyz

dump final all custom 1 final.dump id type xu yu zu fx fy fz
dump_modify final sort id format float %.15g
run 0
undump final
variable energy equal pe
variable pressure_xx equal pxx
variable pressure_yy equal pyy
variable pressure_zz equal pzz
variable pressure_xy equal pxy
variable pressure_xz equal pxz
variable pressure_yz equal pyz
print "TRIBO_RESULT ${energy} ${pressure_xx} ${pressure_yy} ${pressure_zz} ${pressure_xy} \
${pressure_xz} ${pressure_yz}" file result.txt screen no
"""


def lines_of(text):
    return text.splitlines()


# --- the script ---------------------------------------------------------------------------------

def test_relax_script_of_a_deepmd_model():
    script = lammps_job.lammps_input('relax', DEEPMD, slab(if_pos=BOTTOM_AND_MIDDLE))
    assert script == RELAX_OF_DEEPMD.replace('\\\n', '')


def test_deepmd_takes_the_path_of_the_model_not_a_search():
    # DEV-52: SCS replaced the path with the models found in ../../Training/NN*/
    script = lammps_job.lammps_input('relax', DEEPMD, slab())
    assert 'pair_style deepmd /models/dpa31-3m-omat24.pth' in lines_of(script)
    assert 'Training' not in script


def test_mace_script_uses_the_converted_model_and_the_atom_map():
    script = lines_of(lammps_job.lammps_input('relax', MACE, slab()))
    assert ('pair_style mliap unified /models/mace-omat-0-medium-f64.model-mliap_lammps.pt 0'
            in script)
    assert 'pair_coeff * * Cu' in script
    # the map is declared before the box is read
    assert script.index('atom_modify map yes') < script.index('read_data structure.data')


def test_the_atom_map_is_only_for_mace():
    assert 'atom_modify map yes' not in lammps_job.lammps_input('relax', DEEPMD, slab())


def test_a_mace_model_file_is_refused():
    # DEV-53: SCS deduced the name of the converted file from the .model without saying
    mlip = MlipParams(type='MACE', path='/models/mace-omat-0-medium-f64.model')
    with pytest.raises(ModelError, match='converted for LAMMPS') as error:
        lammps_job.lammps_input('relax', mlip, slab())
    assert error.value.key == 'path'


def test_mace_mp_is_not_available_with_lammps():
    mlip = MlipParams(type='MACE_MP', path='/models/mace-mp.model')
    with pytest.raises(ModelError, match="not available with the backend 'lammps'"):
        lammps_job.lammps_input('relax', mlip, slab())


def test_the_arguments_of_the_loaders_of_ase_are_refused():
    mlip = MlipParams(type='MACE', path=MACE.path, kwargs={'default_dtype': 'float64'})
    with pytest.raises(ModelError, match="remove 'default_dtype'") as error:
        lammps_job.lammps_input('relax', mlip, slab())
    assert error.value.key == 'kwargs'


def test_a_relative_path_is_refused():
    with pytest.raises(ModelError, match='absolute path'):
        lammps_job.lammps_input('relax', MlipParams(type='DP', path='model.pth'), slab())


def test_a_type_without_pair_style_is_refused(mlip_types_restored):
    register_model_type(ModelType('GRACE', 'grace-tensorpotential', 'tensorpotential:TPCalculator'))
    with pytest.raises(ModelError, match="no pair_style is known for 'GRACE'"):
        lammps_job.lammps_input('relax', MlipParams(type='GRACE', path='/models/g.pb'), slab())


def test_only_scf_and_relax():
    with pytest.raises(ValueError, match="calculation 'md': only scf and relax"):
        lammps_job.lammps_input('md', DEEPMD, slab())


def test_single_point_is_a_run_of_zero_steps():
    script = lines_of(lammps_job.lammps_input('scf', DEEPMD, slab(if_pos=BOTTOM_AND_MIDDLE),
                                              max_seconds=60))
    assert 'run 0' in script
    for absent in ('minimize', 'min_modify', 'timer', 'trajectory', 'fix', 'group', 'unfix'):
        assert not any(absent in line for line in script), absent
    assert script[-1].startswith('print "TRIBO_RESULT ')


def test_relax_uses_the_maximum_force_of_an_atom_as_criterion():
    # DEV-55: ftol = fmax on the force of one atom, as fmax of ASE; SCS left the norm of all
    mlip = MlipParams(type='DP', path=DEEPMD.path, fmax=0.002, maxsteps=7)
    script = lines_of(lammps_job.lammps_input('relax', mlip, slab()))
    assert script.index('min_modify norm max') < script.index('minimize 0.0 0.002 7 1000000')
    assert 'min_style cg' in script


@pytest.mark.parametrize('seconds, text', [(3600, '1:00:00'), (65, '0:01:05'), (0.4, '0:00:01'),
                                           (86399.9, '23:59:59')])
def test_the_time_budget_covers_the_minimization_only(seconds, text):
    script = lines_of(lammps_job.lammps_input('relax', DEEPMD, slab(), max_seconds=seconds))
    assert script.index(f'timer timeout {text}') < script.index('min_style cg')
    # the run of zero steps that follows must not be stopped at once by the budget
    assert script.index('timer timeout off') < script.index('run 0')
    assert script.index('minimize 0.0 0.01 500 1000000') < script.index('timer timeout off')


def test_without_a_time_budget_there_is_no_timer():
    assert 'timer' not in lammps_job.lammps_input('relax', DEEPMD, slab())


@pytest.mark.parametrize('if_pos, expected', [
    (None, []),
    ([[1, 1, 1]] * 12, []),
    ([[0, 0, 0]] * 12, ['group blocked_xyz id 1:12', 'fix lock_xyz blocked_xyz setforce 0.0 0.0 0.0']),
    ([[1, 1, 1]] * 2 + [[1, 1, 0]] * 3 + [[1, 1, 1]] + [[1, 1, 0]] + [[1, 1, 1]] * 5,
     ['group blocked_z id 3:5 7', 'fix lock_z blocked_z setforce NULL NULL 0.0']),
    ([[0, 1, 1]] * 3 + [[1, 1, 1]] * 9,
     ['group blocked_x id 1:3', 'fix lock_x blocked_x setforce 0.0 NULL NULL']),
    ([[0, 0, 1]] * 2 + [[1, 0, 1]] + [[1, 1, 1]] * 9,
     ['group blocked_xy id 1:2', 'fix lock_xy blocked_xy setforce 0.0 0.0 NULL',
      'group blocked_y id 3', 'fix lock_y blocked_y setforce NULL 0.0 NULL']),
])
def test_locked_components_become_groups_and_fixes(if_pos, expected):
    script = lines_of(lammps_job.lammps_input('relax', DEEPMD, slab(if_pos=if_pos)))
    found = [line for line in script if line.startswith(('group', 'fix'))]
    assert found == expected
    # every lock is removed before the run that gives the forces without constraints
    unfix = [line.split()[1] for line in script if line.startswith('unfix')]
    assert sorted(unfix) == sorted(line.split()[1] for line in expected if line.startswith('fix'))
    if expected:
        assert max(script.index(f'unfix {name}') for name in unfix) < script.index('run 0')


@pytest.mark.parametrize('ids, text', [([1], '1'), ([1, 2, 3], '1:3'), ([2, 4, 5, 6, 9], '2 4:6 9'),
                                       ([1, 2, 4, 5], '1:2 4:5'), ([7, 9], '7 9')])
def test_ranges_of_atom_ids(ids, text):
    assert lammps_job._id_ranges(ids) == text


def test_a_cell_tilted_beyond_half_a_length_asks_for_a_large_tilt():
    # the hexagonal cell of the slab is tilted by exactly half of the length: allowed by LAMMPS
    assert 'box tilt large' not in lammps_job.lammps_input('relax', DEEPMD, slab())
    base = slab()
    lattice = Lattice([[10.0, 0, 0], [7.0, 8.0, 0], [0, 0, 20.0]])
    sheared = Structure(lattice, base.species[:1], [[0.1, 0.1, 0.1]], labels=['Cu1'])
    script = lines_of(lammps_job.lammps_input('relax', DEEPMD, sheared))
    assert script.index('box tilt large') < script.index('read_data structure.data')


# --- the data file ------------------------------------------------------------------------------

def parse_data(text):
    """Cell lines, masses and atoms of a data file: (header, {type: mass}, rows of the atoms)."""
    masses = dict((int(a), float(b)) for a, b in re.findall(r'^\s*(\d+)\s+([\d.e+-]+) # \w+$',
                                                              text, re.M))
    atoms = text.split('Atoms # atomic')[1].strip().splitlines()
    return text.split('Masses')[0], masses, np.array([line.split() for line in atoms], float)


def test_data_file_has_the_types_the_masses_and_the_positions_of_the_structure():
    structure = slab()
    header, masses, atoms = parse_data(lammps_job.lammps_data(structure))
    assert '12 atoms' in header and '1 atom types' in header
    assert masses == {1: pytest.approx(63.546)}
    assert atoms[:, 0].tolist() == list(range(1, 13)) and set(atoms[:, 1]) == {1}
    prism = Prism(np.array(structure.lattice.matrix))
    assert atoms[:, 2:] == pytest.approx(prism.vector_to_lammps(structure.cart_coords), abs=1e-12)


def test_atom_types_follow_the_first_appearance_of_the_elements():
    structure = slab()
    structure.replace(2, 'Fe')
    structure.replace(0, 'Fe')
    assert lammps_job.species_of(structure) == ['Fe', 'Cu']
    _, masses, atoms = parse_data(lammps_job.lammps_data(structure))
    assert atoms[:3, 1].tolist() == [1, 2, 1]
    assert masses[1] == pytest.approx(55.845) and masses[2] == pytest.approx(63.546)
    script = lammps_job.lammps_input('scf', DEEPMD, structure)
    assert 'pair_coeff * * Fe Cu' in lines_of(script)


def test_cell_of_the_data_file_is_the_one_lammps_requires():
    # first vector along x, second in the xy plane: an arbitrary orientation is rotated into it
    structure = helpers.rotated(slab())
    text = lammps_job.lammps_data(structure)
    header = parse_data(text)[0]
    xhi, yhi, zhi = (float(re.search(rf'0\.0\s+(\S+)\s+{axis}lo {axis}hi', header).group(1))
                     for axis in 'xyz')
    xy, xz, yz = (float(value) for value in re.search(r'(\S+)\s+(\S+)\s+(\S+)\s+xy xz yz',
                                                       header).groups())
    cell = np.array([[xhi, 0, 0], [xy, yhi, 0], [xz, yz, zhi]])
    assert cell @ cell.T == pytest.approx(structure.lattice.matrix @ structure.lattice.matrix.T)
    # the positions are rotated with the cell: distances do not change
    atoms = parse_data(text)[2][:, 2:]
    assert np.linalg.norm(atoms[0] - atoms[5]) == pytest.approx(
        np.linalg.norm(structure.cart_coords[0] - structure.cart_coords[5]))


def test_write_input_and_remove_outputs(tmp_path):
    structure = slab(if_pos=BOTTOM_AND_MIDDLE)
    path = lammps_job.write_input(tmp_path, 'relax', DEEPMD, structure, max_seconds=100)
    assert path == tmp_path / lammps_job.INPUT_FILE
    assert path.read_text() == lammps_job.lammps_input('relax', DEEPMD, structure, 100)
    assert (tmp_path / lammps_job.DATA_FILE).read_text() == lammps_job.lammps_data(structure)
    for name in lammps_job.OUTPUT_FILES:
        (tmp_path / name).write_text('from an earlier attempt')
    lammps_job.remove_outputs(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [lammps_job.DATA_FILE, lammps_job.INPUT_FILE])
    lammps_job.remove_outputs(tmp_path)  # nothing to remove is not an error


# --- states -------------------------------------------------------------------------------------

def status_of(tmp_path, log, calculation='relax'):
    if log is not None:
        (tmp_path / lammps_job.LOG_FILE).write_text(log)
    return lammps_job.lammps_status(tmp_path, calculation)


def test_no_log_or_a_log_not_closed_is_truncated(tmp_path):
    expected = lammps_job.LammpsStatus(None, None, None, 'truncated')
    assert status_of(tmp_path, None) == expected
    assert status_of(tmp_path, helpers.log_text(closed=False)) == expected
    assert status_of(tmp_path, '') == expected


def test_an_error_is_a_crash(tmp_path):
    status = status_of(tmp_path, helpers.log_text(error='Unknown pair style deepmd'))
    assert status.process == 'crashed' and status.stop_reason == 'error'
    assert status.error == 'ERROR: Unknown pair style deepmd (src/pair.cpp:123)'
    assert status.scf is None and status.geometry is None


def test_a_single_point_that_ended(tmp_path):
    status = status_of(tmp_path, helpers.log_text(relax=False), 'scf')
    assert status == lammps_job.LammpsStatus('ended', 'n/a', 'n/a', 'completed')


@pytest.mark.parametrize('criterion, geometry, reason', [
    ('force tolerance', 'converged', 'completed'),
    ('forces are zero', 'converged', 'completed'),
    ('max iterations', 'max_steps', 'max_steps'),
    ('max force evaluations', 'max_steps', 'max_steps'),
    ('walltime limit reached', 'not_converged', 'max_seconds'),
    ('linesearch alpha is zero', 'not_converged', 'optimizer_failed'),
    ('search direction is not downhill', 'not_converged', 'optimizer_failed'),
    ('something LAMMPS adds one day', 'not_converged', 'optimizer_failed'),
])
def test_states_follow_the_stopping_criterion_of_minimize(tmp_path, criterion, geometry, reason):
    status = status_of(tmp_path, helpers.log_text(criterion, iterations=41, evaluations=60))
    assert status == lammps_job.LammpsStatus('ended', 'n/a', geometry, reason,
                                             steps={'cg': 41, 'total': 41}, criterion=criterion)
    assert status.fallback is None


def test_a_closed_log_of_a_relaxation_without_the_summary_is_an_error(tmp_path):
    with pytest.raises(lammps_job.LammpsOutputError, match='no summary of the minimization'):
        status_of(tmp_path, helpers.log_text(relax=False))


# --- results ------------------------------------------------------------------------------------

@pytest.mark.parametrize('turn', [False, True], ids=['frame of LAMMPS', 'any frame'])
@pytest.mark.parametrize('calculation', ['relax', 'scf'])
def test_results_are_in_the_frame_of_the_input(tmp_path, calculation, turn):
    structure = slab(if_pos=BOTTOM_AND_MIDDLE)
    structure = helpers.rotated(structure) if turn else structure
    energy, forces, pressure = helpers.write_outputs(tmp_path, structure, calculation)
    parsed = lammps_job.parse_result(tmp_path, calculation, structure)
    assert parsed.parsing == 'ok' and parsed.reason is None
    output = parsed.results['output']
    assert output['energy'] == pytest.approx(energy, abs=1e-12)
    assert output['energy_per_atom'] == pytest.approx(energy / 12, abs=1e-12)
    # forces on every atom, constraints off, in the frame of the input
    assert np.array(output['forces']) == pytest.approx(forces, abs=1e-10)
    assert np.abs(forces[:4, 2]).max() > 1e-3          # a locked component keeps its force
    # kbar, from the bar of LAMMPS, in the frame of the input
    assert np.array(output['stress']) == pytest.approx(pressure * 1E-3, abs=1e-9)
    assert parsed.results['nsites'] == 12


def test_results_have_the_keys_of_a_pw_output_and_the_site_properties_of_the_input(tmp_path):
    structure = slab(if_pos=BOTTOM_AND_MIDDLE)
    structure.add_site_property('magmom', [0.5] * 12)
    helpers.write_outputs(tmp_path, structure)
    parsed = lammps_job.parse_result(tmp_path, 'relax', structure)
    assert set(parsed.results) == {'nsites', 'input', 'output'}
    assert set(parsed.results['output']) == {'structure', 'density', 'energy', 'energy_per_atom',
                                             'forces', 'stress'}
    final = parsed.structure
    assert set(final.site_properties) == {'if_pos', 'interface_region'}   # magmom left out
    assert final.site_properties['interface_region'] == structure.site_properties['interface_region']
    assert final.lattice == structure.lattice and final.labels == structure.labels
    assert parsed.results['output']['density'] == pytest.approx(final.density)
    assert parsed.structure_text.startswith('12\n')


def test_the_sign_and_the_units_of_the_stress_are_those_of_pw_x(tmp_path):
    # pressure of a compressed cell is positive and the mean of the diagonal is -dE/dV
    def structure_at(scale):
        atoms = bulk('Cu', cubic=True)
        atoms.set_cell(atoms.cell * scale, scale_atoms=True)
        return helpers.rotated(Structure(Lattice(atoms.cell[:]), atoms.get_chemical_symbols(),
                                         atoms.get_positions(), coords_are_cartesian=True,
                                         labels=['Cu1'] * 4))

    def energy_at(scale):
        return helpers.emt_values(structure_at(scale))[0]

    step = 1E-5
    for scale, sign in ((0.97, 1), (1.03, -1)):
        structure = structure_at(scale)
        helpers.write_outputs(tmp_path, structure, 'scf')
        stress = np.array(lammps_job.parse_result(tmp_path, 'scf', structure).results['output']
                          ['stress'])
        derivative = (energy_at(scale + step) - energy_at(scale - step)) / (
            structure_at(scale + step).volume - structure_at(scale - step).volume)
        assert np.sign(np.trace(stress)) == sign
        assert np.trace(stress) / 3 == pytest.approx(-derivative * 1602.176634, rel=1e-4)


def test_positions_of_a_relaxation_are_the_images_nearest_to_the_input(tmp_path):
    structure = helpers.rotated(slab(if_pos=BOTTOM_AND_MIDDLE))
    relaxed = structure.cart_coords + np.random.default_rng(3).normal(0, 0.05, (12, 3))
    # LAMMPS counts the images of an atom that crossed the box: two atoms are one cell away
    helpers.write_outputs(tmp_path, structure, positions=relaxed,
                          shift={0: [1, 0, 0], 7: [0, -1, 1]})
    parsed = lammps_job.parse_result(tmp_path, 'relax', structure)
    assert parsed.parsing == 'ok'
    assert parsed.structure.cart_coords == pytest.approx(relaxed, abs=1e-10)


def test_a_single_point_keeps_the_positions_of_the_input(tmp_path):
    structure = slab()
    helpers.write_outputs(tmp_path, structure, 'scf', shift={3: [0, 1, 0]})
    parsed = lammps_job.parse_result(tmp_path, 'scf', structure)
    assert parsed.parsing == 'ok'
    assert parsed.structure.cart_coords == pytest.approx(structure.cart_coords, abs=1e-12)


def test_observed_is_the_version_of_lammps_and_the_pair_style(tmp_path):
    structure = slab()
    lammps_job.write_input(tmp_path, 'relax', DEEPMD, structure)
    helpers.write_outputs(tmp_path, structure)
    observed = lammps_job.parse_result(tmp_path, 'relax', structure).observed
    assert observed == {'lammps': helpers.VERSION,
                        'pair_style': 'deepmd /models/dpa31-3m-omat24.pth'}


def test_nothing_is_read_while_lammps_has_not_ended_or_the_geometry_has_not_converged(tmp_path):
    structure = slab()
    helpers.write_outputs(tmp_path, structure)
    log = tmp_path / lammps_job.LOG_FILE
    for text in (helpers.log_text(closed=False), helpers.log_text(error='Out of memory'),
                 helpers.log_text('max iterations'), helpers.log_text('walltime limit reached'),
                 helpers.log_text('linesearch alpha is zero')):
        log.write_text(text)
        assert lammps_job.parse_result(tmp_path, 'relax', structure) == Parsed(None)
    log.unlink()
    assert lammps_job.parse_result(tmp_path, 'relax', structure) == Parsed(None)


def failed(tmp_path, structure, calculation='relax'):
    parsed = lammps_job.parse_result(tmp_path, calculation, structure)
    assert parsed.parsing == 'failed' and parsed.results is None and parsed.structure is None
    return parsed.reason


def test_a_log_without_the_summary_is_a_failed_parsing(tmp_path):
    structure = slab()
    helpers.write_outputs(tmp_path, structure)
    (tmp_path / lammps_job.LOG_FILE).write_text(helpers.log_text(relax=False))
    assert 'no summary of the minimization' in failed(tmp_path, structure)


def test_missing_files_are_a_failed_parsing(tmp_path):
    structure = slab()
    helpers.write_outputs(tmp_path, structure)
    (tmp_path / lammps_job.FINAL_FILE).unlink()
    assert 'final.dump has no complete snapshot' in failed(tmp_path, structure)
    helpers.write_outputs(tmp_path, structure)
    (tmp_path / lammps_job.RESULT_FILE).unlink()
    assert 'result.txt is missing' in failed(tmp_path, structure)


@pytest.mark.parametrize('line', ['TRIBO_RESULT nan 1 2 3 4 5 6', 'TRIBO_RESULT 1 2 3',
                                  'TRIBO_RESULT 1 2 3 4 5 6 x', 'other line', ''])
def test_a_damaged_result_line_is_a_failed_parsing(tmp_path, line):
    structure = slab()
    helpers.write_outputs(tmp_path, structure)
    (tmp_path / lammps_job.RESULT_FILE).write_text(line + '\n')
    assert 'has no line "TRIBO_RESULT"' in failed(tmp_path, structure)


def test_a_snapshot_with_the_forces_missing_is_a_failed_parsing(tmp_path):
    structure = slab()
    helpers.write_outputs(tmp_path, structure)
    (tmp_path / lammps_job.FINAL_FILE).write_text(
        helpers.dump_text(structure, columns='id type xu yu zu'))
    assert 'final.dump has no complete snapshot with positions and forces' in failed(tmp_path,
                                                                                     structure)


def test_output_of_another_structure_is_a_failed_parsing(tmp_path):
    structure = slab()
    helpers.write_outputs(tmp_path, structure)
    path = tmp_path / lammps_job.FINAL_FILE
    other = Structure(structure.lattice, structure.species[:11], structure.frac_coords[:11],
                      labels=structure.labels[:11])
    path.write_text(helpers.dump_text(other, forces=np.zeros((11, 3))))
    assert 'the output has 11 atoms, the input structure 12' in failed(tmp_path, structure)
    path.write_text(helpers.dump_text(structure, types=[1] * 11 + [2]))
    assert 'atom types of the output differ' in failed(tmp_path, structure)
    path.write_text(helpers.dump_text(structure).replace('\n1 1 ', '\n13 1 ', 1))
    assert 'atom ids of the output are not 1..N' in failed(tmp_path, structure)


def test_a_single_point_whose_atoms_moved_is_a_failed_parsing(tmp_path):
    structure = slab()
    helpers.write_outputs(tmp_path, structure, 'scf',
                          positions=structure.cart_coords + [0.01, 0, 0])
    assert 'positions of the output differ' in failed(tmp_path, structure, 'scf')


# --- the last structure -------------------------------------------------------------------------

def frames(structure, count=3):
    positions = [structure.cart_coords + 0.01 * (i + 1) for i in range(count)]
    text = ''.join(helpers.dump_text(structure, steps=(10 * (i + 1),), positions=p,
                                     columns='id type xu yu zu')
                   for i, p in enumerate(positions))
    return text, positions


def test_last_structure_is_the_last_snapshot_of_the_trajectory(tmp_path):
    structure = helpers.rotated(slab(if_pos=BOTTOM_AND_MIDDLE))
    text, positions = frames(structure)
    (tmp_path / lammps_job.TRAJECTORY_FILE).write_text(text)
    last = lammps_job.last_structure(tmp_path, structure)
    assert last.cart_coords == pytest.approx(positions[-1], abs=1e-10)
    assert last.site_properties['if_pos'] == structure.site_properties['if_pos']


@pytest.mark.parametrize('cut', [1, 7, 90, 3000], ids=['one character', 'seven characters',
                                                       'ninety characters', 'inside the header'])
def test_a_snapshot_cut_while_it_was_written_is_not_used(tmp_path, cut):
    structure = slab()
    text, positions = frames(structure)
    last_frame = text.rindex('ITEM: TIMESTEP')
    assert last_frame + 1 < len(text) - cut or cut == 3000
    (tmp_path / lammps_job.TRAJECTORY_FILE).write_text(text[:max(len(text) - cut, last_frame + 40)])
    last = lammps_job.last_structure(tmp_path, structure)
    assert last.cart_coords == pytest.approx(positions[1], abs=1e-10)


def test_a_snapshot_with_atoms_missing_is_not_used(tmp_path):
    structure = slab()
    text, positions = frames(structure)
    lines = text.splitlines(keepends=True)
    (tmp_path / lammps_job.TRAJECTORY_FILE).write_text(''.join(lines[:-3]))
    assert lammps_job.last_structure(tmp_path, structure).cart_coords == pytest.approx(
        positions[1], abs=1e-10)


def test_the_final_snapshot_is_preferred_to_the_trajectory(tmp_path):
    structure = slab()
    text, _ = frames(structure)
    (tmp_path / lammps_job.TRAJECTORY_FILE).write_text(text)
    helpers.write_outputs(tmp_path, structure, positions=structure.cart_coords + 0.5)
    last = lammps_job.last_structure(tmp_path, structure)
    assert last.cart_coords == pytest.approx(structure.cart_coords + 0.5, abs=1e-10)


def test_there_may_be_no_structure_to_restart_from(tmp_path):
    structure = slab()
    assert lammps_job.last_structure(tmp_path, structure) is None
    (tmp_path / lammps_job.TRAJECTORY_FILE).write_text('ITEM: TIMESTEP\n0\nITEM: NUMBER')
    assert lammps_job.last_structure(tmp_path, structure) is None


def test_a_snapshot_of_another_structure_is_an_error(tmp_path):
    structure = slab()
    other = Structure(structure.lattice, structure.species[:5], structure.frac_coords[:5])
    (tmp_path / lammps_job.TRAJECTORY_FILE).write_text(
        helpers.dump_text(other, columns='id type xu yu zu'))
    with pytest.raises(lammps_job.LammpsOutputError, match='the output has 5 atoms'):
        lammps_job.last_structure(tmp_path, structure)


# --- no MLIP library ----------------------------------------------------------------------------

def test_import_needs_no_mlip_library():
    code = ('import sys\n'
            'from triboworkflow.core.settings.schema import MlipParams\n'
            'from triboworkflow.engines.mlip import lammps_job\n'
            'from pymatgen.core import Lattice, Structure\n'
            'structure = Structure(Lattice.cubic(4.0), ["Cu"], [[0, 0, 0]])\n'
            'mlip = MlipParams(type="DP", path="/m.pth")\n'
            'lammps_job.lammps_input("relax", mlip, structure)\n'
            'lammps_job.lammps_data(structure)\n'
            'loaded = [name for name in ("torch", "mace", "deepmd") if name in sys.modules]\n'
            'assert not loaded, loaded\n')
    done = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
