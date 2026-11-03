# Author: Vito Foderà
# Created: 2026-11-02

"""Tests of ``triboworkflow.engines.qe.pwi``: the pw.x input writer.

The reference is ``tests/data/qe/example_relax.pwi``, the reference relax
example with its placeholders filled with illustrative values, and
``example_relax.toml``, the settings with the same parameters.
"""

import contextlib
import io
import math
from pathlib import Path
from types import SimpleNamespace

import ase.io
import numpy as np
import pytest
from ase.constraints import FixAtoms
from ase.io.espresso import read_fortran_namelist
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Kpoints

import golden_cases
from triboworkflow.core import structures as st
from triboworkflow.core.settings import load_settings
from triboworkflow.core.settings.schema import KPoints, Namelists, Species
from triboworkflow.engines.qe.pwi import (CALC_TYPES, KPointsCard, PwInputError, calculation_for,
                                          fortran_value, grid_card, kpoints_card,
                                          mesh_from_density, namelist_lines, pw_input, vasp_grid,
                                          workflow_input)
from triboworkflow.execution.worker import read_input, with_max_seconds

DATA = Path(__file__).resolve().parent / 'data' / 'qe'

SPECIES = [Species('Fe1', 55.845, 'Fe.pbe.UPF', 0.5),
           Species('Fe2', 55.845, 'Fe.pbe.UPF', -0.5),
           Species('O', 15.999, 'O.pbe.UPF'),
           Species('H', 1.008, 'H.pbe.UPF')]  # no atom of H in STRUCTURE
GRID = KPointsCard('automatic', (4, 4, 1), (0, 0, 0))
FREE = [(1, 1, 1)] * 3


def structure():
    """Fe1, Fe2 and O in an orthorhombic cell."""
    return Structure(Lattice.orthorhombic(2.87, 2.87, 20.0), ['Fe', 'Fe', 'O'],
                     [[0.0, 0.0, 5.0], [1.435, 1.435, 6.435], [0.0, 0.0, 7.2]],
                     coords_are_cartesian=True, labels=['Fe1', 'Fe2', 'O'])


def namelists(**tables):
    return Namelists(**{name: dict(values) for name, values in tables.items()})


def write(nl=None, **kwargs):
    args = dict(structure=structure(), species=SPECIES, namelists=nl or namelists(),
                k_points=GRID, if_pos=FREE)
    args.update(kwargs)
    return pw_input(**args)


def parse(text):
    """Namelists (lower-case names and keys) and card lines of a pw.x input."""
    data, cards = read_fortran_namelist(io.StringIO(text))
    return ({name.lower(): {k.lower(): v for k, v in values.items()} for name, values in data.items()},
            cards)


def keys_in_order(text, name):
    """Keys of a namelist as written, in their order."""
    lines = text.splitlines()
    start = lines.index(f'&{name}')
    end = lines.index('/', start)
    return [line.split('=')[0].strip() for line in lines[start + 1:end]]


def card(text, name):
    """Lines of a card after its title, up to the first blank line."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.split()[:1] == [name])
    block = []
    for line in lines[start + 1:]:
        if not line.strip():
            break
        block.append(line.split())
    return lines[start].split(), block


def example():
    settings = load_settings(DATA / 'example_relax.toml')
    settings.require_complete()
    loaded = st.load_structure(settings)
    k = settings.k_points
    result = pw_input(loaded.structure, settings.atomic_species, settings.namelists,
                      KPointsCard(k.mode, tuple(k.grid), tuple(k.offset)), loaded.constraints)
    return settings, result


# --- the reference example ----------------------------------------------------

def test_equivalent_to_the_reference_example(tmp_path):
    _, result = example()
    reference = (DATA / 'example_relax.pwi').read_text()
    assert result.calculation == 'relax'
    assert result.changes == ()
    assert parse(result.text)[0] == parse(reference)[0]
    for name in ('ATOMIC_SPECIES', 'K_POINTS'):
        assert card(result.text, name) == card(reference, name), name
    path = tmp_path / 'input.pwi'
    path.write_text(result.text)
    got, ref = st.read_structure(path, 'pwi'), st.read_structure(DATA / 'example_relax.pwi', 'pwi')
    assert np.array_equal(got.lattice.matrix, ref.lattice.matrix)
    assert np.array_equal(got.cart_coords, ref.cart_coords)
    assert [s.label for s in got] == [s.label for s in ref]
    assert got.site_properties['if_pos'] == ref.site_properties['if_pos'] == [None] * 5 + [[0, 0, 0]]
    assert got.properties == ref.properties


def test_reference_example_keys_in_the_order_of_the_settings():
    settings, result = example()
    assert keys_in_order(result.text, 'CONTROL') == list(settings.namelists.CONTROL)
    assert keys_in_order(result.text, 'SYSTEM') == (['ibrav', 'nat', 'ntyp']
                                                    + list(settings.namelists.SYSTEM)
                                                    + ['starting_magnetization(1)'])
    assert keys_in_order(result.text, 'ELECTRONS') == list(settings.namelists.ELECTRONS)
    assert keys_in_order(result.text, 'IONS') == ['ion_dynamics']


def test_reference_example_read_back_by_ase(tmp_path):
    _, result = example()
    path = tmp_path / 'input.pwi'
    path.write_text(result.text)
    atoms = ase.io.read(path, format='espresso-in')
    ref = st.read_structure(DATA / 'example_relax.pwi', 'pwi')
    assert np.allclose(atoms.cell[:], ref.lattice.matrix, atol=1e-12)
    assert np.allclose(atoms.positions, ref.cart_coords, atol=1e-12)
    # ASE reads an atom fixed in all three components as FixAtoms.
    assert [(type(c), c.get_indices().tolist()) for c in atoms.constraints] == [(FixAtoms, [5])]


def test_max_seconds_of_the_settings_set_by_the_worker(tmp_path):
    _, result = example()
    text, replaced = with_max_seconds(result.text, 3600)
    assert replaced
    path = tmp_path / 'run.pwi'
    path.write_text(text)
    assert read_input(path) == ('relax', 3600)
    assert parse(text)[0]['control'] == {**parse(result.text)[0]['control'], 'max_seconds': 3600}


def test_max_seconds_added_by_the_worker_when_not_given(tmp_path):
    text, replaced = with_max_seconds(write().text, 1200)
    assert not replaced
    path = tmp_path / 'run.pwi'
    path.write_text(text)
    assert read_input(path) == ('scf', 1200)


# --- namelists ------------------------------------------------------------------

def test_generated_system_keys_and_species():
    text = write(namelists(SYSTEM={'ecutwfc': 40.0, 'nspin': 2})).text
    system = parse(text)[0]['system']
    assert system == {'ibrav': 0, 'nat': 3, 'ntyp': 4, 'ecutwfc': 40.0, 'nspin': 2,
                      'starting_magnetization(1)': 0.5, 'starting_magnetization(2)': -0.5}
    # Every species, in the order of the settings, the one without atoms too.
    assert card(text, 'ATOMIC_SPECIES')[1] == [['Fe1', '55.845', 'Fe.pbe.UPF'],
                                               ['Fe2', '55.845', 'Fe.pbe.UPF'],
                                               ['O', '15.999', 'O.pbe.UPF'],
                                               ['H', '1.008', 'H.pbe.UPF']]


def test_keys_absent_from_the_settings_not_written():
    data = parse(write(namelists(CONTROL={'tprnfor': True})).text)[0]
    assert data['control'] == {'tprnfor': True}
    assert data['electrons'] == {}
    assert 'ions' not in data


def test_keys_written_with_their_spelling_and_order():
    nl = namelists(CONTROL={'Calculation': 'relax', 'forc_conv_thr': 1e-3, 'TPRNFOR': True},
                   SYSTEM={'london_c6(2)': 1.5, 'Ecutwfc': 30})
    text = write(nl).text
    assert keys_in_order(text, 'CONTROL') == ['Calculation', 'forc_conv_thr', 'TPRNFOR']
    assert keys_in_order(text, 'SYSTEM')[3:5] == ['london_c6(2)', 'Ecutwfc']
    assert '   Calculation      = \'relax\'\n' in text


def test_fortran_values():
    assert fortran_value(True) == '.true.' and fortran_value(False) == '.false.'
    assert fortran_value(84600) == '84600'
    assert fortran_value(1.0e-4) == '0.0001' and fortran_value(1e-6) == '1e-06'
    assert fortran_value(45.0) == '45.0'
    assert fortran_value('path/to/PSEUDO') == "'path/to/PSEUDO'"
    assert fortran_value(r'C:\pseudo') == "'C:\\pseudo'"
    assert float(fortran_value(0.1 + 0.2)) == 0.1 + 0.2


@pytest.mark.parametrize('value', [math.nan, math.inf, "a'b", 'two\nlines', None, [1, 2]])
def test_values_that_cannot_be_written(value):
    with pytest.raises(PwInputError, match='namelists.SYSTEM.x'):
        namelist_lines('SYSTEM', [('x', value)], 'namelists.SYSTEM')


@pytest.mark.parametrize('key', ['ecut wfc', 'nat=3', '1st', 'london_c6(1', 'a(b)'])
def test_keys_that_cannot_be_written(key):
    with pytest.raises(PwInputError, match='not a pw.x key'):
        write(namelists(SYSTEM={key: 1}))


def test_index_with_blanks_accepted():
    assert keys_in_order(write(namelists(SYSTEM={'london_c6( 1 )': 1.0})).text,
                         'SYSTEM')[3] == 'london_c6( 1 )'


@pytest.mark.parametrize('key', ['nat', 'NTYP', 'ibrav', 'starting_magnetization(1)'])
def test_keys_written_from_the_structure_not_accepted(key):
    with pytest.raises(PwInputError, match='written from the structure and the species'):
        write(namelists(SYSTEM={key: 1}))


# --- calculation, &IONS and nstep -----------------------------------------------

def test_scf_without_ions_and_with_one_step(tmp_path):
    nl = namelists(CONTROL={'calculation': 'scf', 'nstep': 150, 'tprnfor': True},
                   IONS={'ion_dynamics': 'bfgs'})
    result = write(nl)
    data = parse(result.text)[0]
    assert result.calculation == 'scf'
    assert '&IONS' not in result.text and 'ions' not in data
    assert data['control'] == {'calculation': 'scf', 'nstep': 1, 'tprnfor': True}
    assert keys_in_order(result.text, 'CONTROL') == ['calculation', 'nstep', 'tprnfor']
    assert result.changes == (
        'namelists.CONTROL.nstep = 150 written as 1: an scf calculation makes a single step',
        'namelists.IONS (ion_dynamics) not written: an scf calculation does not move the atoms')
    path = tmp_path / 'input.pwi'
    path.write_text(result.text)
    assert read_input(path) == ('scf', None)


def test_scf_by_default():
    result = write(namelists(IONS={'ion_dynamics': 'bfgs'}))
    assert result.calculation == 'scf' and '&IONS' not in result.text
    assert len(result.changes) == 1


def test_scf_with_one_step_or_without_nstep_unchanged():
    for control in ({'calculation': 'scf', 'nstep': 1}, {'calculation': 'scf'}):
        result = write(namelists(CONTROL=control))
        assert result.changes == ()
        assert parse(result.text)[0]['control'] == control


def test_relax_keeps_nstep_and_writes_ions_even_empty():
    result = write(namelists(CONTROL={'calculation': 'relax', 'nstep': 150}))
    assert result.changes == ()
    assert parse(result.text)[0]['control']['nstep'] == 150
    assert '&IONS\n/\n' in result.text


@pytest.mark.parametrize('calculation', ['vc-relax', 'nscf', 'md'])
def test_other_calculations_not_written(calculation):
    with pytest.raises(PwInputError, match='only scf, relax'):
        write(namelists(CONTROL={'calculation': calculation}))


# --- keys set by the caller --------------------------------------------------------

def test_control_keys_of_the_caller_first():
    control = {'calculation': 'relax', 'prefix': 'pes_scf_hs1', 'outdir': 'OUT'}
    result = write(namelists(CONTROL={'tprnfor': True}, IONS={'ion_dynamics': 'bfgs'}),
                   control=control)
    assert keys_in_order(result.text, 'CONTROL') == ['calculation', 'prefix', 'outdir', 'tprnfor']
    assert result.calculation == 'relax' and '&IONS' in result.text


def test_calculation_of_the_caller_decides():
    result = write(namelists(CONTROL={'nstep': 50}, IONS={'ion_dynamics': 'bfgs'}),
                   control={'calculation': 'scf'})
    assert result.calculation == 'scf' and len(result.changes) == 2


def test_control_key_given_twice_rejected():
    with pytest.raises(PwInputError, match='namelists.CONTROL.Prefix: set for this calculation'):
        write(namelists(CONTROL={'Prefix': 'mine'}), control={'prefix': 'pes_scf_hs1'})


def test_prefix_with_an_apostrophe_rejected():
    with pytest.raises(PwInputError, match='apostrophe'):
        write(control={'prefix': "pes_scf_o'h"})


# --- cards ----------------------------------------------------------------------------

def test_cell_and_positions_in_angstrom():
    text = write().text
    assert parse(text)[0]['system']['ibrav'] == 0
    title, rows = card(text, 'CELL_PARAMETERS')
    assert title == ['CELL_PARAMETERS', 'angstrom']
    assert rows[0] == ['2.87000000000000', '0.00000000000000', '0.00000000000000']
    title, rows = card(text, 'ATOMIC_POSITIONS')
    assert title == ['ATOMIC_POSITIONS', 'angstrom']
    assert rows[1] == ['Fe2', '1.4350000000', '1.4350000000', '6.4350000000']


def test_if_pos_only_for_atoms_with_a_fixed_component():
    flags = st.AppliedConstraints([(1, 1, 1), (0, 0, 1), (1, 0, 1)], ['default', 'file', 'override'])
    rows = card(write(if_pos=flags).text, 'ATOMIC_POSITIONS')[1]
    assert [row[4:] for row in rows] == [[], ['0', '0', '1'], ['1', '0', '1']]


@pytest.mark.parametrize('flags, message', [
    ([(1, 1, 1)] * 2, '2 entries for 3 atoms'),
    ([(1, 1, 1), (1, 1), (1, 1, 1)], 'atom 2: three integers'),
    ([(1, 1, 1), (1, 1, 1), (1.0, 1, 1)], 'atom 3: three integers'),
])
def test_if_pos_errors(flags, message):
    with pytest.raises(PwInputError, match=message):
        write(if_pos=flags)


def test_k_points_cards():
    assert card(write().text, 'K_POINTS') == (['K_POINTS', 'automatic'],
                                              [['4', '4', '1', '0', '0', '0']])
    text = write(k_points=KPointsCard('gamma')).text
    assert card(text, 'K_POINTS') == (['K_POINTS', 'gamma'], [])
    assert 'K_POINTS gamma\n\nCELL_PARAMETERS' in text


def test_k_points_grid_from_numpy_integers():
    k = KPointsCard('automatic', np.array([8, 8, 1]), [0, 0, 1])
    assert (k.grid, k.offset) == ((8, 8, 1), (0, 0, 1))
    assert all(type(v) is int for v in k.grid + k.offset)


@pytest.mark.parametrize('args, message', [
    (('crystal', None, None), 'only automatic, gamma'),
    (('gamma', (1, 1, 1), (0, 0, 0)), 'no grid'),
    (('automatic', (8, 8), (0, 0, 0)), 'grid must be three integers'),
    (('automatic', (8, 8, 1.0), (0, 0, 0)), 'grid must be three integers'),
    (('automatic', (8, 8, 1), None), 'offset must be three integers'),
])
def test_k_points_card_errors(args, message):
    with pytest.raises(PwInputError, match=message):
        KPointsCard(*args)


# --- k-points of the settings and of the workflows ------------------------------------

def test_k_points_modes_of_the_settings():
    assert kpoints_card(KPoints('automatic', [8, 8, 1], [0, 0, 1])) == \
        KPointsCard('automatic', (8, 8, 1), (0, 0, 1))
    assert kpoints_card(KPoints('gamma')) == KPointsCard('gamma')


@pytest.mark.parametrize('rule, expected', [
    (True, KPointsCard('gamma')),
    (False, KPointsCard('automatic', (1, 1, 1), (0, 0, 0))),
])
def test_unshifted_1x1x1(rule, expected):
    k = KPoints('automatic', [1, 1, 1], [0, 0, 0], gamma_for_unshifted_1x1x1=rule)
    assert kpoints_card(k) == expected


def test_shifted_1x1x1_stays_automatic():
    assert kpoints_card(KPoints('automatic', [1, 1, 1], [1, 1, 1])) == \
        KPointsCard('automatic', (1, 1, 1), (1, 1, 1))


def test_kdens_grid():
    bulk = Structure(Lattice.cubic(3.6), ['Cu'], [[0, 0, 0]])
    # |b| = 2 pi / 3.6 = 1.745 1/angstrom: round(1.745 * 6) = 10.
    assert kpoints_card(KPoints('kdens', kdens=6.0), bulk) == \
        KPointsCard('automatic', (10, 10, 10), (0, 0, 0))
    # 20 angstrom along z, atoms within 2.2: a slab for 'auto', one k-point along z.
    assert kpoints_card(KPoints('kdens', kdens=6.0), structure()) == \
        KPointsCard('automatic', (13, 13, 1), (0, 0, 0))


def test_kdens_grid_1x1x1_is_gamma():
    k = KPoints('kdens', kdens=6.0)
    assert kpoints_card(k, structure(), material_type='molecule') == KPointsCard('gamma')
    k.gamma_for_unshifted_1x1x1 = False
    assert kpoints_card(k, structure(), material_type='molecule') == \
        KPointsCard('automatic', (1, 1, 1), (0, 0, 0))


def test_kdens_needs_the_structure():
    with pytest.raises(PwInputError, match='needs the structure'):
        kpoints_card(KPoints('kdens', kdens=6.0))


@pytest.mark.parametrize('calc_type', ['slab_from_scratch', 'slab_pos_relax', 'interface_z_relax'])
def test_kdens_grid_as_the_workflow_code(calc_type):
    """The grid of the vendored VaspInputSet for the calc_types of the
    interfaces, and the same grid for a single calculation with 'auto'."""
    from triboworkflow.vendored.highput.utils.vasp_tools import VaspInputSet
    s = Structure(Lattice.hexagonal(2.556, 25), ['Cu', 'Cu', 'H'],
                  [[0, 0, 0.1], [1 / 3, 2 / 3, 0.18], [2 / 3, 1 / 3, 0.25]])
    kdens = golden_cases.COMP_PARAMS['kdens']
    with contextlib.redirect_stdout(io.StringIO()):
        vis = VaspInputSet(s.copy(), golden_cases.COMP_PARAMS, calc_type).get_vasp_settings()
    expected = ((17, 17, 1), (0, 0, 0))
    assert vasp_grid(vis.kpoints) == expected
    assert mesh_from_density(s, kdens, 'slab') == expected
    assert kpoints_card(KPoints('kdens', kdens=float(kdens)), s) == grid_card(*expected)


@pytest.mark.parametrize('kpoints', [
    Kpoints.monkhorst_automatic((4, 4, 1)),
    Kpoints.gamma_automatic((4, 4, 1), shift=(0.5, 0, 0)),
    Kpoints(kpts=[[0, 0, 0], [0.5, 0, 0]], num_kpts=2, style=Kpoints.supported_modes.Reciprocal,
            kpts_weights=[1, 1]),
])
def test_other_vasp_k_points_not_converted(kpoints):
    with pytest.raises(PwInputError, match='only an automatic Gamma-centred grid'):
        vasp_grid(kpoints)


# --- calc_types of the workflows ---------------------------------------------------------

def workflow_settings(**k_points):
    """The parts of the settings that workflow_input reads."""
    return SimpleNamespace(
        atomic_species=[Species('Cu', 63.546, 'Cu.pbe.UPF'), Species('H', 1.008, 'H.pbe.UPF')],
        namelists=namelists(CONTROL={'tprnfor': True}, SYSTEM={'ecutwfc': 40.0},
                            IONS={'ion_dynamics': 'bfgs'}),
        k_points=KPoints('kdens', kdens=6.0, **k_points))


@pytest.mark.parametrize('calc_type, calculation, flags', [
    ('slab_from_scratch', 'scf', [[], [], []]),
    ('slab_pos_relax', 'relax', [[], [], []]),
    ('interface_z_relax', 'relax', [['0', '0', '1'], ['0', '0', '1'], []]),   # H atoms free
])
def test_input_of_each_calc_type(calc_type, calculation, flags):
    """calculation, &IONS, constraints and k-points of the calc_types of the
    interfaces, from the structure and k-points of the vendored VaspInputSet."""
    from triboworkflow.vendored.highput.utils.vasp_tools import VaspInputSet
    s = Structure(Lattice.hexagonal(2.556, 25), ['Cu', 'Cu', 'H'],
                  [[0, 0, 0.1], [1 / 3, 2 / 3, 0.18], [2 / 3, 1 / 3, 0.25]])
    with contextlib.redirect_stdout(io.StringIO()):
        vis = VaspInputSet(s, golden_cases.COMP_PARAMS, calc_type).get_vasp_settings()
    result = workflow_input(s, vis.kpoints, calc_type, workflow_settings(), 'pes_hs1', 'OUT')
    data = parse(result.text)[0]
    assert result.calculation == calculation
    assert data['control'] == {'calculation': calculation, 'prefix': 'pes_hs1', 'outdir': 'OUT',
                               'tprnfor': True}
    assert ('ions' in data) == (calculation == 'relax')
    assert [row[4:] for row in card(result.text, 'ATOMIC_POSITIONS')[1]] == flags
    assert card(result.text, 'K_POINTS') == (['K_POINTS', 'automatic'], [['17', '17', '1', '0', '0', '0']])


def test_workflow_grid_1x1x1_follows_the_rule_of_the_settings():
    s = structure()
    one = Kpoints.gamma_automatic((1, 1, 1))
    species = [Species('Fe1', 55.845, 'Fe.UPF'), Species('Fe2', 55.845, 'Fe.UPF'), Species('O', 16.0, 'O.UPF')]
    for rule, mode in ((True, 'gamma'), (False, 'automatic')):
        settings = workflow_settings(gamma_for_unshifted_1x1x1=rule)
        settings.atomic_species = species
        result = workflow_input(s, one, 'slab_from_scratch', settings, 'p', 'OUT')
        assert card(result.text, 'K_POINTS')[0] == ['K_POINTS', mode]


@pytest.mark.parametrize('calc_type', ['bulk_full_relax', 'slab_shape_relax', 'interface_pos_relax',
                                       'slab_follow_up', 'slab_nscf', 'atom', 'unknown'])
def test_calc_types_not_mapped_yet(calc_type):
    with pytest.raises(PwInputError, match=f'calc_type {calc_type!r} has no pw.x calculation yet'):
        calculation_for(calc_type)


def test_mapped_calc_types():
    assert {t: calculation_for(t) for t in CALC_TYPES} == {
        'slab_from_scratch': 'scf', 'slab_pos_relax': 'relax', 'interface_z_relax': 'relax'}


def test_label_of_the_structure_not_in_the_species():
    with pytest.raises(PwInputError, match='not in atomic_species: Fe2'):
        write(species=[SPECIES[0], SPECIES[2]])


@pytest.mark.parametrize('species, message', [
    ([Species('Fe1', 55.8, 'Fe.UPF'), Species('Fe1', 55.8, 'Fe.UPF')], 'given twice'),
    ([Species('Fe 1', 55.8, 'Fe.UPF')], r'atomic_species\[0\].label'),
    ([Species('Fe1', 55.8, 'my Fe.UPF')], r'atomic_species\[0\].pseudopotential'),
    ([Species('Fe1', 55.8, '')], r'atomic_species\[0\].pseudopotential'),
])
def test_species_that_cannot_be_written(species, message):
    with pytest.raises(PwInputError, match=message):
        write(species=species)
