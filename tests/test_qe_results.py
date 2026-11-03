# Author: Vito Foderà
# Created: 2026-11-03

"""Tests of ``triboworkflow.engines.qe.results``.

The outputs are synthetic (``pwo_helpers.py``): the lines pw.x prints for a
structure, with the formats of the real outputs. The real outputs are read in
``test_qe_results_corpus.py``.
"""

import re

import numpy as np
import pytest
from pymatgen.core.structure import Structure

from triboworkflow.core import store
from triboworkflow.core.structures.readers import read_extxyz_text
from triboworkflow.engines.qe import results as qe
from triboworkflow.engines.qe.results import (PwOutputError, PwSpecies, parse_output, pw_values,
                                              read_pw_values)
from triboworkflow.engines.qe.status import pw_status

from pwo_helpers import (MD5_FE, MD5_H, PSEUDO_DIR, SPECIES, JOB_DONE, header, pw_output,
                         relaxed, scf_cycle, structure, structure_in)

FORCES = ((0.00123456, -0.00234567, 0.00345678), (-0.00123456, 0.00234567, -0.00345678))
STRESS = ((-0.00000242, 0.00000010, 0.0), (0.00000010, -0.00000242, 0.0), (0.0, 0.0, -0.00000300))


def replace(text, old, new):
    assert old in text, old
    return text.replace(old, new)


# --- Constants -------------------------------------------------------------------------------

def test_constants_are_those_of_qe():
    # CODATA 2018, as QE 7.1-7.3 computes them (Modules/constants.f90).
    assert qe.RY_TO_EV == pytest.approx(13.605693122994, abs=1e-12)
    assert qe.BOHR_TO_ANGSTROM == pytest.approx(0.529177210903, abs=1e-15)
    assert qe.RY_KBAR == pytest.approx(147105.078, abs=1e-3)


def test_constants_agree_with_pymatgen():
    from pymatgen.core.units import Ry_to_eV, bohr_to_ang
    assert qe.RY_TO_EV == pytest.approx(Ry_to_eV, rel=1e-15)
    assert qe.BOHR_TO_ANGSTROM == pytest.approx(bohr_to_ang, rel=1e-15)


# --- Header ----------------------------------------------------------------------------------

def test_header_of_a_scf():
    s = structure()
    values = pw_values(pw_output(s, forces=FORCES), 'scf')
    assert (values.calculation, values.version, values.runs, values.nat) == ('scf', '7.2', 1, 2)
    alat = np.linalg.norm(s.lattice.matrix[0]) / qe.BOHR_TO_ANGSTROM
    assert values.alat == pytest.approx(alat, abs=5e-7)
    assert np.allclose(np.array(values.axes) * values.alat * qe.BOHR_TO_ANGSTROM,
                       s.lattice.matrix, atol=1e-5)
    assert values.labels == ('Fe1', 'H')
    assert np.allclose(np.array(values.positions) * values.alat * qe.BOHR_TO_ANGSTROM,
                       s.cart_coords, atol=1e-6)
    assert values.species == (
        PwSpecies('Fe1', 'Fe', f'{PSEUDO_DIR}/{SPECIES[0][2]}', MD5_FE),
        PwSpecies('H', 'H', f'{PSEUDO_DIR}/{SPECIES[1][2]}', MD5_H))


@pytest.mark.parametrize('version', ['7.1', '7.2', '7.3', '7.3.1', '6.8', '8.0'])
def test_versions_read(version):
    # Versions outside 7.1.x-7.3.x are read too; the callers log a warning.
    assert pw_values(pw_output(structure(), version=version), 'scf').version == version


@pytest.mark.parametrize('line', [None, '     MD5 check sum: Not computed, couldn\'t open file',
                                  '     MD5 check sum: 327243EA9DB589A56BC50E6EC7945B51',
                                  '     Pseudo is Ultrasoft, Zval =  1.0'])
def test_checksum_not_printed_is_none(line):
    lines = [line if line is not None else '', f'     MD5 check sum: {MD5_H}']
    values = pw_values(pw_output(structure(), md5_lines=lines), 'scf')
    assert [sp.md5 for sp in values.species] == [None, MD5_H]


@pytest.mark.parametrize('old, new, match', [
    ('     number of atoms/cell      =            2\n', '', 'number of atoms/cell'),
    ('     number of atomic types    =            2\n', '', 'number of atomic types'),
    ('     celldm(1)=', '     celldmX1)=', 'celldm'),
    ('     crystal axes: (cart. coord. in units of alat)', '     crystal', 'crystal axes'),
    ('               a(2) =', '               a(3) =', 'a\\(2\\)'),
    ('     PseudoPot. # 2 for H  read from file:\n', '\n', 'pseudopotentials'),
    ('     atomic species   valence    mass     pseudopotential\n', '\n', 'species table'),
    ('positions (alat units)', 'positions (cryst. coord.)', 'positions'),
    ('tau(   2) = (', 'tau(   2) = (**********', 'atom 2'),
])
def test_header_not_complete_is_an_error(old, new, match):
    with pytest.raises(PwOutputError, match=match):
        pw_values(replace(pw_output(structure()), old, new), 'scf')


def test_header_with_a_number_that_does_not_fit_is_an_error():
    text = replace(pw_output(structure()), '               a(1) = (',
                   '               a(1) = (*******')
    with pytest.raises(PwOutputError, match='a\\(1\\)'):
        pw_values(text, 'scf')


def test_calculation_must_be_scf_or_relax():
    with pytest.raises(ValueError, match='vc-relax'):
        pw_values(pw_output(structure()), 'vc-relax')


# --- Energy, forces, stress ------------------------------------------------------------------

def test_energy_forces_and_stress_of_a_scf():
    values = pw_values(pw_output(structure(), energy=-253.46613429, forces=FORCES, stress=STRESS),
                       'scf')
    assert values.energy == -253.46613429
    assert values.forces == FORCES
    assert values.stress == STRESS
    assert (values.final_energy, values.final_positions) == (None, None)


def test_forces_and_stress_absent_without_tprnfor_and_tstress():
    values = pw_values(pw_output(structure()), 'scf')
    assert (values.forces, values.stress) == (None, None)


def test_total_forces_before_their_contributions_and_the_dispersion():
    values = pw_values(pw_output(structure(), forces=FORCES, verbosity_high=True,
                                 dispersion=True), 'scf')
    assert values.forces == FORCES


def test_energy_and_forces_of_the_last_scf_cycle_of_a_relax():
    values = pw_values(pw_output(structure(), 'relax', energy=-31.1, forces=FORCES, steps=3),
                       'relax')
    assert values.energy == -31.1
    assert values.forces == FORCES


def test_energy_of_the_last_cycle_only():
    # A last cycle that prints no "!" line leaves the energy unknown, even if
    # an earlier cycle printed one.
    s = structure()
    text = pw_output(s, job_done=False) + '\n'.join(
        ['     Self-consistent Calculation', '',
         '     total energy              =     -31.00000000 Ry', ''] + JOB_DONE) + '\n'
    assert pw_values(text, 'scf').energy is None


def test_exclamation_mark_elsewhere_in_a_line_is_not_the_energy():
    text = replace(pw_output(structure()), '!    total energy              =',
                   '     note: !    total energy              =')
    assert pw_values(text, 'scf').energy is None


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '*****************'])
def test_energy_that_is_not_finite_is_an_error(value):
    text = replace(pw_output(structure(), energy=-31.1),
                   '!    total energy              =      -31.10000000 Ry',
                   f'!    total energy              = {value:>17s} Ry')
    with pytest.raises(PwOutputError, match='total energy'):
        pw_values(text, 'scf')


def test_energy_not_in_ry_is_an_error():
    text = replace(pw_output(structure()), '-31.10000000 Ry\n     smearing',
                   '-31.10000000 eV\n     smearing')
    with pytest.raises(PwOutputError, match='Ry'):
        pw_values(text, 'scf')


@pytest.mark.parametrize('old, new, match', [
    ('    0.00123456   -0.00234567    0.00345678', '    0.00123456  ************    0.00345678',
     'atom 1'),
    ('    0.00123456   -0.00234567    0.00345678', '           NaN   -0.00234567    0.00345678',
     'atom 1'),
    ('     atom    2 type  2   force', '     atom    3 type  2   force', 'atom 2'),
])
def test_forces_that_cannot_be_read_are_an_error(old, new, match):
    text = replace(pw_output(structure(), forces=FORCES), old, new)
    with pytest.raises(PwOutputError, match=match):
        pw_values(text, 'scf')


def test_stress_that_is_not_finite_is_an_error():
    text = replace(pw_output(structure(), stress=STRESS), '  -0.00000300', '          NaN')
    with pytest.raises(PwOutputError, match='stress'):
        pw_values(text, 'scf')


# --- Relaxation ------------------------------------------------------------------------------

def test_final_energy_and_coordinates_of_a_relax():
    s = structure()
    values = pw_values(pw_output(s, 'relax', energy=-8125.08157944, forces=FORCES,
                                 final_energy=-8125.0815794407), 'relax')
    assert values.energy == -8125.08157944
    assert values.final_energy == -8125.0815794407
    assert values.final_labels == ('Fe1', 'H')
    assert np.allclose(values.final_positions, relaxed(s), atol=1e-10)
    assert values.final_if_pos == (None, None)


def test_flags_of_the_final_coordinates():
    s = structure()
    values = pw_values(pw_output(s, 'relax', final_if_pos=[(0, 0, 0), (1, 1, 1)]), 'relax')
    assert values.final_if_pos == ((0, 0, 0), None)


def test_final_energy_absent():
    assert pw_values(pw_output(structure(), 'relax', final_energy=None), 'relax').final_energy \
        is None


def test_final_coordinates_not_closed_are_not_read():
    values = pw_values(pw_output(structure(), 'relax', closed=False), 'relax')
    assert (values.final_labels, values.final_positions, values.final_if_pos) == (None,) * 3
    assert pw_status(pw_output(structure(), 'relax', closed=False),
                     'relax').final_coordinates == 'incomplete'


def test_final_coordinates_of_a_scf_are_not_read():
    values = pw_values(pw_output(structure(), 'scf'), 'scf')
    assert values.final_positions is None


@pytest.mark.parametrize('old, new, match', [
    ('ATOMIC_POSITIONS (angstrom)\nFe1', 'ATOMIC_POSITIONS (bohr)\nFe1', 'angstrom'),
    ('Begin final coordinates\n', 'Begin final coordinates\nCELL_PARAMETERS (angstrom)\n',
     'cell'),
    ('End final coordinates', 'H      0.0 0.0 0.0\nEnd final coordinates', '3 positions'),
    ('Begin final coordinates\n\nATOMIC_POSITIONS (angstrom)', 'Begin final coordinates\n',
     'ATOMIC_POSITIONS'),
])
def test_final_coordinates_that_cannot_be_read_are_an_error(old, new, match):
    text = replace(pw_output(structure(), 'relax'), old, new)
    with pytest.raises(PwOutputError, match=match):
        pw_values(text, 'relax')


@pytest.mark.parametrize('value', ['NaN', '********************'])
def test_final_position_that_is_not_finite_is_an_error(value):
    s = structure()
    x = f'{relaxed(s)[1][0]:20.10f}'
    text = pw_output(s, 'relax')
    index = text.rindex(x)
    text = text[:index] + f'{value:>20s}' + text[index + 20:]
    with pytest.raises(PwOutputError, match='final'):
        pw_values(text, 'relax')


# --- Runs ------------------------------------------------------------------------------------

def test_only_the_last_run_is_read():
    s = structure()
    first = pw_output(s, 'relax', energy=-30.0, final_energy=-30.0, version='7.1')
    last = pw_output(s, 'relax', energy=-31.1, final_positions=relaxed(s, (0.05, 0, 0)))
    values = pw_values(first + last, 'relax')
    assert (values.runs, values.version, values.energy, values.final_energy) == \
        (2, '7.2', -31.1, -31.1)
    assert np.allclose(values.final_positions, relaxed(s, (0.05, 0, 0)), atol=1e-10)


def test_last_run_without_its_values_does_not_take_those_of_a_run_before():
    s = structure()
    text = pw_output(s, forces=FORCES, stress=STRESS) + '\n'.join(header(s)) + '\n'
    values = pw_values(text, 'scf')
    assert (values.runs, values.energy, values.forces, values.stress) == (2, None, None, None)


def test_read_pw_values_names_the_file(tmp_path):
    path = tmp_path / 'output.pwo'
    path.write_text(replace(pw_output(structure()), 'celldm(1)=', 'celldmX1)='))
    with pytest.raises(PwOutputError, match='output.pwo'):
        read_pw_values(path, 'scf')
    path.write_text(pw_output(structure()))
    assert read_pw_values(path, 'scf').energy == -31.1


# --- Results ---------------------------------------------------------------------------------

EV_A = qe.RY_TO_EV / qe.BOHR_TO_ANGSTROM
FREE, FIXED = [1, 1, 1], [0, 0, 0]


def parsed(calculation='scf', s=None, **kwargs):
    s = structure() if s is None else s
    return parse_output(pw_output(s, calculation, **kwargs), calculation, structure_in(s))


def test_results_of_a_scf():
    s = structure(magmom=[2.0, 0.0])
    p = parse_output(pw_output(s, energy=-253.46613429, forces=FORCES, stress=STRESS), 'scf',
                     structure_in(s))
    assert (p.parsing, p.reason) == ('ok', None)
    r = p.results
    assert list(r) == ['nsites', 'input', 'output']
    assert list(r['output']) == ['structure', 'density', 'energy', 'energy_per_atom', 'forces',
                                 'stress']
    assert r['nsites'] == 2
    assert r['output']['energy'] == -253.46613429 * qe.RY_TO_EV
    assert r['output']['energy_per_atom'] == r['output']['energy'] / 2
    assert np.allclose(r['output']['forces'], np.array(FORCES) * EV_A, rtol=1e-15, atol=0)
    assert np.allclose(r['output']['stress'], np.array(STRESS) * qe.RY_KBAR, rtol=1e-15, atol=0)
    s_in = structure_in(s)
    given = Structure.from_dict(r['input']['structure'])
    assert np.array_equal(given.lattice.matrix, s_in.lattice.matrix)
    assert np.allclose(given.cart_coords, s_in.cart_coords, atol=1e-12)
    assert (given.labels, given.site_properties) == (s_in.labels, s_in.site_properties)
    final = Structure.from_dict(r['output']['structure'])
    assert np.array_equal(final.lattice.matrix, s_in.lattice.matrix)
    assert np.allclose(final.cart_coords, s_in.cart_coords, atol=1e-12)
    assert final.labels == ['Fe1', 'H']
    assert r['output']['density'] == pytest.approx(final.density, rel=1e-15)
    # magmom is an initial moment: in the input structure only.
    assert Structure.from_dict(r['input']['structure']).site_properties['magmom'] == [2.0, 0.0]
    assert 'magmom' not in final.site_properties


def test_results_are_json_of_a_record():
    p = parsed('relax', structure(if_pos=[FIXED, FREE]), forces=FORCES,
               final_if_pos=[FIXED, FREE])
    record = store.Record(system='s', step='pes', label='aa', calc_type='slab_pos_relax',
                          engine=store.Engine('qe', 'pw.x'), energy_model={'version': '7.2'},
                          fingerprint='sha256:' + 'a' * 64,
                          status=store.Status('ended', 'converged', 'converged', p.parsing),
                          results=p.results, observed=p.observed)
    again = store.Record.from_dict(record.to_dict())
    assert (again.results, again.observed) == (p.results, p.observed)
    assert again.is_complete


def test_results_of_a_relax():
    s = structure(if_pos=[FIXED, FREE])
    p = parse_output(pw_output(s, 'relax', energy=-8125.08157944, final_energy=-8125.0815794407,
                               forces=FORCES, final_if_pos=[FIXED, FREE]),
                     'relax', structure_in(s))
    assert p.parsing == 'ok'
    out = p.results['output']
    assert out['energy'] == -8125.0815794407 * qe.RY_TO_EV
    assert 'stress' not in out
    final = Structure.from_dict(out['structure'])
    assert np.allclose(final.cart_coords, relaxed(s), atol=1e-10)
    assert np.array_equal(final.lattice.matrix, structure_in(s).lattice.matrix)
    assert final.site_properties['if_pos'] == [FIXED, FREE]
    assert p.structure.labels == ['Fe1', 'H']


def test_relax_without_final_energy_takes_the_last_total_energy():
    p = parsed('relax', energy=-31.1, final_energy=None, forces=FORCES)
    assert (p.parsing, p.results['output']['energy']) == ('ok', -31.1 * qe.RY_TO_EV)


def test_final_energy_that_differs_from_the_last_total_energy_fails():
    p = parsed('relax', energy=-31.1, final_energy=-31.10000002, forces=FORCES)
    assert p.parsing == 'failed'
    assert 'Final energy' in p.reason
    assert p.results is None and p.structure is None and p.values is not None


def test_final_energy_within_the_digits_printed():
    p = parsed('relax', energy=-31.1, final_energy=-31.1000000049, forces=FORCES)
    assert p.parsing == 'ok'


def test_scf_without_forces_and_stress():
    p = parsed('scf')
    assert p.parsing == 'ok'
    assert 'forces' not in p.results['output'] and 'stress' not in p.results['output']


def test_relax_without_forces_fails():
    p = parsed('relax')
    assert (p.parsing, p.reason) == ('failed', 'no forces in the last SCF cycle of a relaxation')


def test_converged_relax_with_final_coordinates_not_closed_fails():
    p = parsed('relax', forces=FORCES, closed=False)
    assert p.parsing == 'failed'
    assert 'final coordinates' in p.reason


@pytest.mark.parametrize('calculation, old, new', [
    ('scf', '     convergence has been achieved in   7 iterations',
     '     convergence NOT achieved after 100 iterations: stopping'),
    ('scf', '   JOB DONE.', ''),
    ('relax', '     bfgs converged in', '     bfgs failed after'),
    ('relax', '     bfgs converged in   2 scf cycles and   1 bfgs steps',
     '     The maximum number of steps has been reached.'),
])
def test_output_not_over_and_converged_is_not_read(calculation, old, new):
    s = structure()
    text = replace(pw_output(s, calculation, forces=FORCES), old, new)
    assert parse_output(text, calculation, structure_in(s)) == qe.Parsed(None)


def test_text_that_is_not_a_pw_output_is_an_error():
    with pytest.raises(PwOutputError, match='PWSCF'):
        parse_output('nothing\n', 'scf', structure_in(structure()))


def test_value_that_cannot_be_read_fails_with_the_reason():
    s = structure()
    text = replace(pw_output(s), '-31.10000000 Ry\n     smearing',
                   '************* Ry\n     smearing')
    p = parse_output(text, 'scf', structure_in(s))
    assert (p.parsing, p.values, p.results) == ('failed', None, None)
    assert 'total energy' in p.reason


def test_extxyz_rounding_is_allowed():
    # The header is computed from the structure with all its decimals, the
    # input structure is the one of structure_in.extxyz (eight decimals).
    s = structure()
    assert np.abs(structure_in(s).cart_coords - s.cart_coords).max() > 1e-10
    assert parsed('scf', s).parsing == 'ok'


def moved(s, cell=None, positions=None):
    return Structure(s.lattice.matrix if cell is None else cell, s.species,
                     s.cart_coords if positions is None else positions,
                     coords_are_cartesian=True, labels=s.labels,
                     site_properties=s.site_properties)


@pytest.mark.parametrize('change, match', [
    (lambda s: moved(s, cell=s.lattice.matrix + [[0, 0, 0], [0, 0, 0], [0, 0, 2e-5]]), 'cell'),
    (lambda s: moved(s, positions=s.cart_coords + [[0, 0, 0], [0, 1e-5, 0]]), 'atoms \\[2\\]'),
    (lambda s: Structure(s.lattice, s.species, s.cart_coords, coords_are_cartesian=True,
                         labels=['Fe', 'H']), 'labels'),
    (lambda s: Structure(s.lattice, ['Fe', 'H', 'H'], [*s.cart_coords, [0, 0, 6]],
                         coords_are_cartesian=True, labels=['Fe1', 'H', 'H']), '2 atoms'),
])
def test_scf_output_of_another_input_fails(change, match):
    s = structure()
    p = parse_output(pw_output(s), 'scf', structure_in(change(s)))
    assert p.parsing == 'failed'
    assert re.search(match, p.reason), p.reason


def test_relax_header_positions_are_not_compared():
    # In a restart the header prints the positions read from the restart file.
    s = structure()
    p = parse_output(pw_output(s, 'relax', forces=FORCES), 'relax',
                     structure_in(moved(s, positions=s.cart_coords + 0.1)))
    assert p.parsing == 'ok'


@pytest.mark.parametrize('kwargs, match', [
    (dict(final_if_pos=[FIXED, FREE]), 'if_pos'),
    (dict(final_if_pos=[(1, 1, 0), FREE]), 'if_pos'),
])
def test_relax_with_flags_other_than_the_input_fails(kwargs, match):
    s = structure(if_pos=[FREE, FREE])
    p = parse_output(pw_output(s, 'relax', forces=FORCES, **kwargs), 'relax', structure_in(s))
    assert p.parsing == 'failed'
    assert match in p.reason


def test_relax_with_final_labels_other_than_the_input_fails():
    s = structure()
    text = pw_output(s, 'relax', forces=FORCES)
    head, _, tail = text.rpartition('ATOMIC_POSITIONS (angstrom)\nFe1 ')
    p = parse_output(head + 'ATOMIC_POSITIONS (angstrom)\nFe  ' + tail, 'relax', structure_in(s))
    assert p.parsing == 'failed'
    assert 'labels of the final coordinates' in p.reason


def test_structure_text_is_the_extxyz_of_the_final_structure():
    s = structure(if_pos=[FIXED, FREE])
    p = parse_output(pw_output(s, 'relax', forces=FORCES, final_if_pos=[FIXED, FREE]), 'relax',
                     structure_in(s))
    back = read_extxyz_text(p.structure_text)
    assert back.labels == ['Fe1', 'H']
    assert np.allclose(back.cart_coords, relaxed(s), atol=1e-8)
    assert back.site_properties['if_pos'] == [FIXED, FREE]
    assert qe.Parsed(None).structure_text is None


# --- Identity --------------------------------------------------------------------------------

MODEL = {'version': '7.2',
         'species': [{'label': label, 'pseudopotential': upf, 'md5': md5}
                     for label, _, upf, md5 in SPECIES],
         'k_points': {'mode': 'gamma'}, 'namelists': {'SYSTEM': [['ecutwfc', 30.0]]}}


@pytest.mark.parametrize('printed, expected, observed', [
    ('7.3.1', '7.3', '7.3'), ('7.3.1', '7.3.1', '7.3.1'), ('7.3', '7.3.1', '7.3'),
    ('7.4', '7.3', '7.4'), ('7.31', '7.3', '7.31'), ('8.0.1', '8', '8.0.1'),
    ('7.2', None, '7.2'), ('7.2MaX', '7.2', '7.2MaX'),
])
def test_observed_version_has_the_components_of_the_expected_one(printed, expected, observed):
    assert qe.observed_version(printed, expected) == observed


def test_observed_identity_has_the_form_of_the_energy_model():
    p = parsed('scf')
    assert p.observed == {
        'version': '7.2',
        'species': [{'label': label, 'pseudopotential': upf, 'md5': md5,
                     'path': f'{PSEUDO_DIR}/{upf}'} for label, _, upf, md5 in SPECIES]}


def record_of(p, energy_model=MODEL):
    return store.Record(system='s', step='pes', label='aa', calc_type='slab_from_scratch',
                        engine=store.Engine('qe', 'pw.x', p.values.version),
                        energy_model=energy_model, fingerprint='sha256:' + 'a' * 64,
                        status=store.Status('ended', 'converged', 'n/a', p.parsing),
                        results=p.results, observed=p.observed)


@pytest.mark.parametrize('version, expected, differences', [
    ('7.2', '7.2', []),
    ('7.3.1', '7.3', []),
    ('7.3.1', '7.3.1', []),
    ('7.3', '7.3.1', ["version: expected '7.3.1', observed '7.3'"]),
    ('8.0', '7.2', ["version: expected '7.2', observed '8.0'"]),
    ('8.0', '8.0', []),
])
def test_version_compared_on_the_components_given(version, expected, differences):
    s = structure()
    model = {**MODEL, 'version': expected}
    p = parse_output(pw_output(s, version=version), 'scf', structure_in(s), model)
    record = record_of(p, model)
    assert record.engine.version == version
    assert store.identity_differences(record) == differences


def test_output_with_other_pseudopotentials_keeps_its_results_and_is_not_reused():
    s = structure()
    other = ('Fe1', 'Fe', 'Fe.pbe-spn-kjpaw.UPF', 'c' * 32)
    p = parse_output(pw_output(s, species=(other, SPECIES[1])), 'scf', structure_in(s), MODEL)
    assert p.parsing == 'ok' and p.results is not None
    record = record_of(p)
    assert record.is_complete
    assert store.identity_differences(record) == [
        f"species[0].pseudopotential: expected '{SPECIES[0][2]}', observed 'Fe.pbe-spn-kjpaw.UPF'",
        f"species[0].md5: expected '{MD5_FE}', observed '{'c' * 32}'"]
    assert not store.can_reuse(record, record.fingerprint)


def test_checksum_not_printed_is_a_difference():
    s = structure()
    p = parse_output(pw_output(s, md5_lines=['', f'     MD5 check sum: {MD5_H}']), 'scf',
                     structure_in(s), MODEL)
    assert store.identity_differences(record_of(p)) == [
        f"species[0].md5: expected '{MD5_FE}', observed None"]


def test_observed_when_the_values_are_read_only():
    s = structure()
    failed = parsed('relax')  # no forces: the values are read, the results are not
    assert failed.parsing == 'failed' and failed.observed['version'] == '7.2'
    text = replace(pw_output(s), '-31.10000000 Ry\n     smearing',
                   '************* Ry\n     smearing')
    assert parse_output(text, 'scf', structure_in(s)).observed is None
    assert parsed('scf', job_done=False).observed is None


def test_scf_cycle_helper_prints_the_lines_of_the_states():
    # The synthetic outputs are complete for the states as well.
    status = pw_status(pw_output(structure(), 'relax'), 'relax')
    assert (status.process, status.scf, status.geometry, status.final_coordinates,
            status.stop_reason) == ('ended', 'converged', 'converged', 'complete', 'completed')
    assert scf_cycle(2, -1.0)[0].strip() == 'Self-consistent Calculation'
