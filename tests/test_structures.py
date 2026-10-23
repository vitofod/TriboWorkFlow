"""Structures: readers, ASE <-> pymatgen conversion, writers, constraints and
the slabs of the workflows (triboworkflow.core.structures).

The historic path of the slabs is compared with TribChem in the golden cases
custom_slab_* (golden_cases.py); here are its checks and records.
"""

import contextlib
import copy
import io
import json
import tomllib
from pathlib import Path

import ase.io
import ase.units
import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms, FixCartesian, FixedLine, FixScaled
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.io.ase import AseAtomsAdaptor

import golden_cases
from triboworkflow.core import structures as st
from triboworkflow.core.settings import settings_from_dict
from triboworkflow.core.settings.schema import ConstraintsParams, Override

ROOT = Path(__file__).resolve().parents[1]

# Fe1 and Fe2: same element, opposite starting_magnetization; flags on three
# atoms out of four (atom 2 without flags), two keys on one line.
PWI = """&CONTROL
  calculation = 'relax'
/
&SYSTEM
  ibrav = 0, nat = 4, ntyp = 2
  ecutwfc = 30, ecutrho = 240
  nspin = 2
  starting_magnetization(1) = 0.5
  starting_magnetization(2) = -0.5
/
&ELECTRONS
/
&IONS
/
CELL_PARAMETERS angstrom
  2.87 0.0 0.0
  0.0 2.87 0.0
  0.0 0.0 20.0
ATOMIC_SPECIES
Fe1 55.845 Fe.pbe.UPF
Fe2 55.845 Fe.pbe.UPF
ATOMIC_POSITIONS angstrom
Fe1 0.0 0.0 2.0 0 0 0
Fe2 1.435 1.435 3.435
Fe1 0.0 0.0 4.87 0 0 1
Fe2 1.435 1.435 6.305 1 1 1
K_POINTS automatic
8 8 1 0 0 0
"""
PWI_IF_POS = [[0, 0, 0], None, [0, 0, 1], [1, 1, 1]]

CIF = """data_test
_cell_length_a 3.0
_cell_length_b 3.0
_cell_length_c 3.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Fe_a Fe 0.5 0.5 0.5 1.0
O1 O 0.0 0.0 0.0 1.0
Fe_b Fe 0.0 0.5 0.5 1.0
"""

XYZ = """3
three copper atoms
Cu 0.0 0.0 0.0
Cu 1.278 0.7378653 2.087
Cu 0.0 1.4757306 4.174
"""
XYZ_CELL = [[2.556, 0.0, 0.0], [-1.278, 2.2135959, 0.0], [0.0, 0.0, 20.0]]


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def poscar(tmp_path, name='hex', file='POSCAR', **changes):
    return write(tmp_path, file, golden_cases.poscar_text(**(golden_cases.CUSTOM_SLABS[name] | changes)))


def fe_structure():
    """Fe1/Fe2 with opposite moments and mixed constraints (if_pos)."""
    s = Structure(Lattice.from_parameters(2.87, 2.87, 20, 90, 90, 90), ['Fe'] * 4,
                  [[0, 0, 0.1], [0.5, 0.5, 0.17], [0, 0, 0.24], [0.5, 0.5, 0.31]],
                  labels=['Fe1', 'Fe2', 'Fe1', 'Fe2'],
                  site_properties={'magmom': [2.2, -2.2, 2.2, -2.2], 'if_pos': PWI_IF_POS})
    return s


def summary(structure):
    """Data that a lossless round trip keeps (positions compared apart)."""
    props = {k: [None if v is None else np.asarray(v).tolist() for v in values]
             for k, values in structure.site_properties.items()}
    return {'species': [str(s) for s in structure.species], 'labels': structure.labels,
            'site_properties': props}


def assert_same_structure(a, b):
    assert summary(a) == summary(b)
    np.testing.assert_allclose(a.lattice.matrix, b.lattice.matrix, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a.cart_coords, b.cart_coords, rtol=0, atol=1e-10)


# --- formats ------------------------------------------------------------------

@pytest.mark.parametrize('name, fmt', [
    ('slab.cif', 'cif'), ('a.xyz', 'xyz'), ('a.extxyz', 'extxyz'), ('in.pwi', 'pwi'),
    ('POSCAR', 'poscar'), ('CONTCAR_relaxed', 'poscar'), ('cu.vasp', 'poscar'), ('cu.json', 'json'),
])
def test_detect_format(name, fmt):
    assert st.detect_format(name) == fmt


def test_detect_format_unknown():
    with pytest.raises(st.StructureError, match='cannot tell the format'):
        st.detect_format('structure.in')


def test_format_json_in_the_settings():
    data = tomllib.loads((ROOT / 'examples' / 'single_input.toml').read_text())
    data['structure']['format'] = 'json'
    assert settings_from_dict(data).structure.format == 'json'


def test_one_structure_files_take_frame_zero_only(tmp_path):
    with pytest.raises(st.StructureError, match='frame must be 0'):
        st.read_structure(write(tmp_path, 'a.pwi', PWI), frame=1)


# --- PWI ----------------------------------------------------------------------

def test_pwi_keeps_labels_elements_and_constraints(tmp_path):
    s = st.read_structure(write(tmp_path, 'a.pwi', PWI))
    assert s.labels == ['Fe1', 'Fe2', 'Fe1', 'Fe2']
    assert [site.specie.symbol for site in s] == ['Fe'] * 4
    assert s.site_properties == {'if_pos': PWI_IF_POS}
    np.testing.assert_allclose(s.cart_coords[1], [1.435, 1.435, 3.435])
    np.testing.assert_allclose(s.lattice.matrix, np.diag([2.87, 2.87, 20.0]))


def test_pwi_starting_magnetization_is_provenance_not_magmom(tmp_path):
    s = st.read_structure(write(tmp_path, 'a.pwi', PWI))
    assert 'magmom' not in s.site_properties
    assert s.properties['qe_species'] == [
        {'label': 'Fe1', 'mass': 55.845, 'pseudopotential': 'Fe.pbe.UPF', 'starting_magnetization': 0.5},
        {'label': 'Fe2', 'mass': 55.845, 'pseudopotential': 'Fe.pbe.UPF', 'starting_magnetization': -0.5}]


def test_pwi_starting_magnetization_stays_with_its_label(tmp_path):
    """Xsorb's read_espresso_in_custom looks the values up by element:
    KeyError 'Fe' with Fe1/Fe2 only, and with an extra Fe species both Fe1 and
    Fe2 get the value of Fe."""
    text = PWI.replace('ntyp = 2', 'ntyp = 3') \
              .replace('Fe2 55.845 Fe.pbe.UPF\n', 'Fe2 55.845 Fe.pbe.UPF\nFe 55.845 Fe.pbe.UPF\n') \
              .replace('starting_magnetization(2) = -0.5',
                       'starting_magnetization(2) = -0.5\n  starting_magnetization(3) = 0.1')
    s = st.read_structure(write(tmp_path, 'a.pwi', text))
    assert {q['label']: q['starting_magnetization'] for q in s.properties['qe_species']} == \
        {'Fe1': 0.5, 'Fe2': -0.5, 'Fe': 0.1}


def test_pwi_units_alat_and_crystal(tmp_path):
    text = """&SYSTEM
  ibrav = 0, nat = 1, ntyp = 1, celldm(1) = 10.0
/
CELL_PARAMETERS alat
1 0 0
0 1 0
0 0 2
ATOMIC_SPECIES
O 15.999 O.UPF
ATOMIC_POSITIONS crystal
O 0.5 0.5 0.25
"""
    s = st.read_structure(write(tmp_path, 'a.pwi', text))
    a = 10.0 * ase.units.Bohr
    np.testing.assert_allclose(s.lattice.matrix, np.diag([a, a, 2 * a]))
    np.testing.assert_allclose(s.cart_coords[0], [0.5 * a, 0.5 * a, 0.5 * a])
    assert 'if_pos' not in s.site_properties


@pytest.mark.parametrize('edit, message', [
    (('ibrav = 0', 'ibrav = 2'), 'only ibrav = 0'),
    (('Fe2 1.435 1.435 3.435', 'Fe3 1.435 1.435 3.435'), 'not in ATOMIC_SPECIES: Fe3'),
    (('CELL_PARAMETERS angstrom\n  2.87 0.0 0.0\n  0.0 2.87 0.0\n  0.0 0.0 20.0\n', ''),
     'CELL_PARAMETERS missing'),
])
def test_pwi_errors(tmp_path, edit, message):
    with pytest.raises(st.StructureError, match=message):
        st.read_structure(write(tmp_path, 'a.pwi', PWI.replace(*edit)))


def test_tribchem_pwi_reader_is_not_used(tmp_path):
    """TribChem's physics/dft/qe.py ReadQEinput needs its own card order; this file
    (CELL_PARAMETERS first) gives ValueError there (verified in the historic
    environment), and is read here."""
    assert len(st.read_structure(write(tmp_path, 'a.pwi', PWI))) == 4


# --- CIF ----------------------------------------------------------------------

def test_cif_keeps_order_element_label_and_site_id(tmp_path):
    s = st.read_structure(write(tmp_path, 'a.cif', CIF))
    assert [site.specie.symbol for site in s] == ['Fe', 'O', 'Fe']
    assert s.labels == ['Fe', 'O', 'Fe']
    assert s.site_properties['cif_label'] == ['Fe_a', 'O1', 'Fe_b']
    np.testing.assert_allclose(s.frac_coords, [[0.5, 0.5, 0.5], [0, 0, 0], [0, 0.5, 0.5]])


def test_pymatgen_cif_parser_sorts_the_sites():
    """Why CIF files are read with ASE (preserve_atom_order)."""
    s = Structure.from_str(CIF, fmt='cif')
    assert [site.specie.symbol for site in s] == ['Fe', 'Fe', 'O']


def test_cif_partial_occupancy_rejected(tmp_path):
    with pytest.raises(st.StructureError, match='partial occupancies'):
        st.read_structure(write(tmp_path, 'a.cif', CIF.replace('O1 O 0.0 0.0 0.0 1.0',
                                                              'O1 O 0.0 0.0 0.0 0.5')))


# --- XYZ ----------------------------------------------------------------------

def test_xyz_needs_the_cell_of_the_settings(tmp_path):
    path = write(tmp_path, 'a.xyz', XYZ)
    with pytest.raises(st.StructureError, match='no cell'):
        st.read_structure(path)
    s = st.read_structure(path, cell=XYZ_CELL)
    np.testing.assert_allclose(s.lattice.matrix, XYZ_CELL)
    np.testing.assert_allclose(s.cart_coords[1], [1.278, 0.7378653, 2.087])
    assert s.site_properties == {}


def test_xyz_with_lattice_is_extxyz(tmp_path):
    text = XYZ.replace('three copper atoms', 'Lattice="2.556 0 0 -1.278 2.2135959 0 0 0 20" pbc="T T T"')
    with pytest.raises(st.StructureError, match='format = "extxyz"'):
        st.read_structure(write(tmp_path, 'a.xyz', text), cell=XYZ_CELL)


def test_xyz_has_elements_only(tmp_path):
    with pytest.raises(st.StructureError):
        st.read_structure(write(tmp_path, 'a.xyz', XYZ.replace('Cu 0.0 0.0 0.0', 'Cu1 0.0 0.0 0.0')),
                          cell=XYZ_CELL)


def test_write_xyz_keeps_elements_and_positions_only(tmp_path):
    s = fe_structure()
    st.write_structure(s, tmp_path / 'a.xyz', 'xyz')
    back = st.read_structure(tmp_path / 'a.xyz', cell=s.lattice.matrix)
    np.testing.assert_allclose(back.cart_coords, s.cart_coords, atol=1e-6)
    assert back.labels == ['Fe'] * 4 and back.site_properties == {}


# --- extxyz -------------------------------------------------------------------

def test_extxyz_round_trip_of_labels_moments_and_constraints(tmp_path):
    s = fe_structure()
    st.write_structure(s, tmp_path / 'a.extxyz')
    back = st.read_structure(tmp_path / 'a.extxyz')
    assert_same_structure(s, back)
    assert back.site_properties['magmom'] == [2.2, -2.2, 2.2, -2.2]
    assert back.site_properties['if_pos'] == PWI_IF_POS


@pytest.mark.parametrize('fmt, file, text', [('pwi', 'a.pwi', PWI), ('cif', 'a.cif', CIF)])
def test_extxyz_round_trip_from_the_formats(tmp_path, fmt, file, text):
    s = st.read_structure(write(tmp_path, file, text))
    st.write_structure(s, tmp_path / 'a.extxyz')
    assert_same_structure(s, st.read_structure(tmp_path / 'a.extxyz'))


def test_extxyz_move_mask_is_cartesian_also_for_ase(tmp_path):
    """move_mask holds cartesian flags; atoms without constraints stay free
    for a plain ASE reader too."""
    st.write_structure(fe_structure(), tmp_path / 'a.extxyz')
    atoms = ase.io.read(tmp_path / 'a.extxyz')
    masks = {int(c.get_indices()[0]): c.mask.tolist() for c in atoms.constraints}
    assert all(isinstance(c, FixCartesian) for c in atoms.constraints)
    assert masks == {0: [True, True, True], 1: [False, False, False],
                     2: [True, True, False], 3: [False, False, False]}


@pytest.mark.parametrize('constraints', [
    [FixCartesian([1], [True, True, False])],
    [FixAtoms([2]), FixCartesian([1], [True, True, False])],
    [FixCartesian([1], [True, True, False]), FixAtoms([2])],
], ids=['cartesian-only', 'fixatoms-first', 'fixcartesian-first'])
def test_ase_extxyz_writer_problems(tmp_path, constraints):
    """ASE alone writes unconstrained atoms as fixed, drops the
    FixCartesian after a FixAtoms, or fails; through write_structure the
    same constraints come back unchanged."""
    atoms = Atoms('Fe3', positions=[[0, 0, 0], [1.5, 1.5, 1.5], [0, 1.5, 1.5]],
                  cell=np.eye(3) * 3, pbc=True)
    atoms.set_constraint(constraints)
    expected = st.to_structure(atoms).site_properties['if_pos']
    try:
        ase.io.write(tmp_path / 'ase.extxyz', atoms)
        native = st.to_structure(ase.io.read(tmp_path / 'ase.extxyz')).site_properties['if_pos']
        assert native != expected
    except AttributeError:
        pass
    st.write_structure(st.to_structure(atoms), tmp_path / 'ours.extxyz')
    assert st.read_structure(tmp_path / 'ours.extxyz').site_properties['if_pos'] == expected


def test_extxyz_move_mask_of_fixatoms_gives_every_atom(tmp_path):
    atoms = Atoms('Cu3', positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2]], cell=np.eye(3) * 4, pbc=True)
    atoms.set_constraint(FixAtoms([0]))
    ase.io.write(tmp_path / 'a.extxyz', atoms)  # move_mask:L:1
    assert st.read_structure(tmp_path / 'a.extxyz').site_properties['if_pos'] == \
        [[0, 0, 0], [1, 1, 1], [1, 1, 1]]


def test_extxyz_frames(tmp_path):
    a = Atoms('Cu', positions=[[0, 0, 0]], cell=np.eye(3) * 3, pbc=True)
    b = Atoms('Ag', positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    ase.io.write(tmp_path / 'a.extxyz', [a, b])
    assert st.read_structure(tmp_path / 'a.extxyz', frame=1).species[0].symbol == 'Ag'
    with pytest.raises(st.StructureError, match='frame 2 does not exist'):
        st.read_structure(tmp_path / 'a.extxyz', frame=2)


def test_extxyz_without_cell(tmp_path):
    ase.io.write(tmp_path / 'a.extxyz', Atoms('Cu2', positions=[[0, 0, 0], [1, 1, 1]]))
    with pytest.raises(st.StructureError, match='no periodic cell'):
        st.read_structure(tmp_path / 'a.extxyz')


def test_extxyz_selective_dynamics_converted_only_when_cartesian(tmp_path):
    s = st.read_structure(poscar(tmp_path))  # hexagonal, c along z, flags F F F / T T T
    st.write_structure(s, tmp_path / 'a.extxyz')
    assert st.read_structure(tmp_path / 'a.extxyz').site_properties['if_pos'] == \
        [[0, 0, 0], [0, 0, 0], [1, 1, 1]]
    flags = [[False, True, False]] * 3  # free along b, which is not a cartesian axis
    s = st.read_structure(poscar(tmp_path, flags=flags))
    with pytest.raises(st.StructureError, match='atoms 1, 2, 3'):
        st.write_structure(s, tmp_path / 'b.extxyz')


def test_write_unknown_format(tmp_path):
    with pytest.raises(st.StructureError, match='only xyz and extxyz'):
        st.write_structure(fe_structure(), tmp_path / 'a.cif', 'cif')


# --- POSCAR -------------------------------------------------------------------

def test_poscar_as_tribchem_reads_it(tmp_path):
    path = poscar(tmp_path)
    s = st.read_structure(path)
    ref = Structure.from_file(path)  # custom_slab.py:120
    assert json.dumps(s.as_dict(), default=str) == json.dumps(ref.as_dict(), default=str)
    assert [list(map(bool, f)) for f in s.site_properties['selective_dynamics']] == \
        [[False] * 3, [False] * 3, [True] * 3]


# --- JSON ---------------------------------------------------------------------

def slab_fields(slab):
    d = slab.as_dict()
    return {'matrix': d['lattice']['matrix'],
            'sites': [{k: site[k] for k in ('abc', 'label', 'properties', 'species')}
                      for site in d['sites']],
            **{k: d[k] for k in ('miller_index', 'shift', 'scale_factor', 'reconstruction',
                                 'energy', 'oriented_unit_cell')}}


@pytest.mark.parametrize('name', sorted(golden_cases.slab_dicts()))
def test_json_slab_of_tribchem_without_losses(tmp_path, name):
    dct = golden_cases.slab_dicts()[name]
    path = write(tmp_path, f'{name}.json', json.dumps(dct))
    slab = st.read_structure(path)
    assert isinstance(slab, Slab)
    expected = {'matrix': dct['lattice']['matrix'],
                'sites': [{k: site[k] for k in ('abc', 'label', 'properties', 'species')}
                          for site in dct['sites']],
                **{k: dct[k] for k in ('miller_index', 'shift', 'scale_factor', 'reconstruction',
                                       'energy')},
                'oriented_unit_cell': Structure.from_dict(copy.deepcopy(dct['oriented_unit_cell'])).as_dict()}
    diffs = golden_cases.compare(golden_cases.to_data(json.loads(json.dumps(expected))),
                                 golden_cases.to_data(json.loads(json.dumps(slab_fields(slab)))))
    assert not diffs, diffs[:5]


def test_slab_from_dict_alone_loses_data(tmp_path):
    """Slab.from_dict reorients the lattice and drops reconstruction and
    labels; read_json keeps them."""
    dct = copy.deepcopy(golden_cases.slab_dicts()['cu111'])
    rot = golden_cases._ROT30
    dct['lattice']['matrix'] = (np.array(dct['lattice']['matrix']) @ rot).tolist()
    dct['reconstruction'] = 'test-reconstruction'
    dct['sites'][0]['label'] = 'Cu1'
    for site in dct['sites']:
        site['xyz'] = (np.array(site['abc']) @ np.array(dct['lattice']['matrix'])).tolist()
    plain = Slab.from_dict(copy.deepcopy(dct))
    ours = st.read_json(write(tmp_path, 'a.json', json.dumps(dct)))
    assert plain.reconstruction is None and plain.labels[0] == 'Cu'
    assert not np.allclose(plain.lattice.matrix, dct['lattice']['matrix'])
    assert ours.reconstruction == 'test-reconstruction' and ours.labels[0] == 'Cu1'
    np.testing.assert_array_equal(ours.lattice.matrix, dct['lattice']['matrix'])


def test_json_structure_without_losses(tmp_path):
    s = fe_structure()
    s.properties = {'note': 'kept'}
    back = st.read_structure(write(tmp_path, 'a.json', json.dumps(s.as_dict())))
    assert back.as_dict() == s.as_dict()


@pytest.mark.parametrize('edit, message', [
    (lambda d: d['lattice'].pop('matrix'), 'no matrix'),
    (lambda d: d.update(extra=1), 'not supported in a Slab dict: extra'),
    (lambda d: d.update(charge=1.0), 'charge = 1.0: a pymatgen Slab'),
    (lambda d: d.update({'@class': 'Molecule'}), 'Structure or Slab dict'),
])
def test_json_errors(tmp_path, edit, message):
    dct = copy.deepcopy(golden_cases.slab_dicts()['cu111'])
    edit(dct)
    with pytest.raises(st.StructureError, match=message):
        st.read_structure(write(tmp_path, 'a.json', json.dumps(dct)))


# --- ASE <-> pymatgen -----------------------------------------------------------

def test_pymatgen_ase_pymatgen_without_losses():
    s = fe_structure()
    s.add_site_property('cif_label', ['A', 'B', 'C', 'D'])
    atoms = st.to_atoms(s)
    assert list(atoms.arrays['qe_label']) == ['Fe1', 'Fe2', 'Fe1', 'Fe2']
    np.testing.assert_array_equal(atoms.get_initial_magnetic_moments(), [2.2, -2.2, 2.2, -2.2])
    assert_same_structure(s, st.to_structure(atoms))


def test_selective_dynamics_through_ase_is_fixscaled():
    s = Structure(Lattice.hexagonal(2.5, 20), ['Cu', 'Cu'], [[0, 0, 0.1], [1 / 3, 2 / 3, 0.2]],
                  site_properties={'selective_dynamics': [[False, True, False], [True] * 3]})
    atoms = st.to_atoms(s)
    assert [type(c) for c in atoms.constraints] == [FixScaled, FixScaled]
    assert st.to_structure(atoms).site_properties['selective_dynamics'] == \
        [[False, True, False], [True, True, True]]


def test_pymatgen_adaptor_loses_partial_constraints():
    """Why convert.py handles the constraints itself."""
    s = fe_structure()
    s.remove_site_property('if_pos')
    s.add_site_property('selective_dynamics', [[False, False, True]] * 4)
    with pytest.raises(ValueError, match='only some dimensions'):
        AseAtomsAdaptor.get_atoms(s)
    atoms = st.to_atoms(fe_structure())
    with pytest.warns(UserWarning, match='Only FixAtoms'):
        assert 'selective_dynamics' not in AseAtomsAdaptor.get_structure(atoms).site_properties


@pytest.mark.parametrize('order', ['fixatoms-first', 'fixcartesian-first'])
def test_mixed_ase_constraints_in_both_orders(order):
    atoms = Atoms('Fe3', positions=[[0, 0, 0], [1.5, 1.5, 1.5], [0, 1.5, 1.5]],
                  cell=np.eye(3) * 3, pbc=True)
    constraints = [FixAtoms([0]), FixCartesian([1], [True, True, False])]
    atoms.set_constraint(constraints if order == 'fixatoms-first' else constraints[::-1])
    assert st.to_structure(atoms).site_properties['if_pos'] == [[0, 0, 0], [0, 0, 1], None]


@pytest.mark.parametrize('constraints, message', [
    ([FixCartesian([0], [True, True, False]), FixScaled([1], [True, True, False])], 'two frames'),
    ([FixedLine(0, [0, 0, 1])], 'FixedLine is not supported'),
    ([FixAtoms([0]), FixCartesian([0], [True, True, False])], 'more than one ASE constraint'),
])
def test_ase_constraints_not_representable(constraints, message):
    atoms = Atoms('Fe2', positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=np.eye(3) * 3, pbc=True)
    atoms.set_constraint(constraints)
    with pytest.raises(st.StructureError, match=message):
        st.to_structure(atoms)


# --- constraints ----------------------------------------------------------------

HEX = golden_cases._HEX


@pytest.mark.parametrize('matrix, flags, expected', [
    (HEX, [False, False, True], (0, 0, 1)),
    (HEX, [True, False, False], (1, 0, 0)),
    (HEX, [True, True, False], (1, 1, 0)),
    (HEX, [True, True, True], (1, 1, 1)),
    (HEX, [False, False, False], (0, 0, 0)),
    (HEX, [False, True, False], None),                              # b not along an axis
    (HEX @ golden_cases._ROT30, [True, False, False], None),         # a rotated
    (golden_cases.CUSTOM_SLABS['tilted']['lattice'], [False, False, True], None),
])
def test_if_pos_from_lattice_flags(matrix, flags, expected):
    if expected is None:
        with pytest.raises(st.StructureError, match='not cartesian flags'):
            st.if_pos_from_lattice_flags(matrix, flags)
    else:
        assert st.if_pos_from_lattice_flags(matrix, flags) == expected


def params(**kwargs):
    kwargs['overrides'] = [Override(**o) for o in kwargs.get('overrides', [])]
    return ConstraintsParams(**kwargs)


def test_precedence_override_file_default(tmp_path):
    s = st.read_structure(write(tmp_path, 'a.pwi', PWI))
    c = st.resolve_if_pos(s, params(default_if_pos=[0, 0, 1],
                                    overrides=[{'atoms': [4], 'if_pos': [0, 0, 0]}]))
    assert c.if_pos == [(0, 0, 0), (0, 0, 1), (0, 0, 1), (0, 0, 0)]
    assert c.source == ['file', 'default', 'file', 'override']
    c = st.resolve_if_pos(s, params(read_from_file=False))
    assert c.if_pos == [(1, 1, 1)] * 4 and c.source == ['default'] * 4


def test_precedence_with_lattice_flags(tmp_path):
    flags = [[False] * 3, [False, True, False], [False, False, True]]
    s = st.read_structure(poscar(tmp_path, flags=flags))
    with pytest.raises(st.StructureError, match='atoms 2 .*overrides'):
        st.resolve_if_pos(s, params())
    c = st.resolve_if_pos(s, params(overrides=[{'atoms': [2], 'if_pos': [1, 1, 0]}]))
    assert c.if_pos == [(0, 0, 0), (1, 1, 0), (0, 0, 1)]
    assert c.source == ['file', 'override', 'file']


def test_override_beyond_the_atoms(tmp_path):
    s = st.read_structure(write(tmp_path, 'a.pwi', PWI))
    with pytest.raises(st.StructureError, match='atom 5 is beyond the 4 atoms'):
        st.resolve_if_pos(s, params(overrides=[{'atoms': [5], 'if_pos': [0, 0, 0]}]))


def test_both_frames_rejected():
    s = fe_structure()
    s.add_site_property('selective_dynamics', [[True] * 3] * 4)
    with pytest.raises(st.StructureError, match='one frame'):
        st.cartesian_if_pos(s)


@pytest.mark.parametrize('calc_type, expected', [
    ('interface_z_relax', [(0, 0, 1), (0, 0, 1), (1, 1, 1)]),   # H free (vasp_tools.py:550-551)
    ('slab_pos_relax', [(1, 1, 1)] * 3),
    ('slab_from_scratch', [(1, 1, 1)] * 3),
])
def test_constraints_of_the_calc_type(calc_type, expected):
    """In the workflows the constraints are those that TribChem's
    VaspInputSet sets for the calc_type, with the H exception."""
    from triboworkflow.vendored.highput.utils.vasp_tools import VaspInputSet
    s = Structure(Lattice.hexagonal(2.556, 25), ['Cu', 'Cu', 'H'],
                  [[0, 0, 0.1], [1 / 3, 2 / 3, 0.18], [2 / 3, 1 / 3, 0.25]])
    with contextlib.redirect_stdout(io.StringIO()):
        VaspInputSet(s, golden_cases.COMP_PARAMS, calc_type).get_vasp_settings()
    c = st.calc_type_if_pos(s)
    assert c.if_pos == expected and c.source == ['calc_type'] * 3


# --- slabs of the workflows -------------------------------------------------------

def test_workflow_slab_custom_path(tmp_path):
    original = st.read_structure(poscar(tmp_path))
    ws = st.workflow_slab(original, [1, 1, 1])
    assert ws.path == 'custom_slab' and not ws.rotated
    assert ws.dropped == ['selective_dynamics'] and ws.added == ['magmom']
    assert ws.slab.site_properties == {'magmom': [0, 0, 0]}
    assert 'selective_dynamics' in ws.original.site_properties  # provenance
    json.dumps(ws.provenance())


def test_workflow_slab_rotation_is_explicit(tmp_path):
    original = st.read_structure(poscar(tmp_path, 'rotated'))
    ws = st.workflow_slab(original, [1, 1, 1])
    assert ws.rotated
    np.testing.assert_allclose(ws.rotation, golden_cases._ROT30.T, atol=1e-9)
    np.testing.assert_allclose(ws.slab.frac_coords, original.frac_coords, atol=1e-12)
    np.testing.assert_allclose(ws.slab.cart_coords, original.cart_coords @ ws.rotation, atol=1e-9)
    np.testing.assert_allclose(ws.slab.lattice.matrix[0], [2.556, 0, 0], atol=1e-9)
    assert ws.provenance()['original_matrix'] == original.lattice.matrix.tolist()


def test_slab_constructor_already_reorients():
    """The rotation is already in the Slab constructor (reorient_lattice=True)."""
    s = Structure(HEX @ golden_cases._ROT30, ['Cu'], [[0, 0, 0]])
    slab = Slab(s.lattice, s.species, s.frac_coords, (1, 1, 1), s, 0, np.array((1, 1, 1)))
    np.testing.assert_allclose(slab.lattice.matrix, HEX, atol=1e-9)


@pytest.mark.parametrize('name, message', [
    ('tilted', 'c is not perpendicular to a'), ('lefthanded', 'left-handed cell')])
def test_workflow_slab_unverified_cells(tmp_path, name, message):
    """TribChem accepts these cells and transforms them
    (golden custom_slab_tilted and custom_slab_lefthanded); here an error."""
    with pytest.raises(st.StructureError, match=message):
        st.workflow_slab(st.read_structure(poscar(tmp_path, name)), [1, 1, 1])
    golden = golden_cases.read_golden(f'custom_slab_{name}')['slab']['lattice']['matrix']
    assert not np.allclose(golden, golden_cases.CUSTOM_SLABS[name]['lattice'], atol=1e-6)


def test_workflow_slab_rejects_custom_labels(tmp_path):
    """Fe1/Fe2 are refused, never turned into Fe."""
    s = fe_structure()
    with pytest.raises(st.StructureError, match='Fe1, Fe2'):
        st.workflow_slab(s, [1, 1, 0])
    st.write_structure(s, tmp_path / 'a.extxyz')
    with pytest.raises(st.StructureError, match='a.extxyz: QE labels'):
        st.read_workflow_slab(tmp_path / 'a.extxyz', [1, 1, 0])


def test_workflow_slab_keeps_the_moments_of_the_file(tmp_path):
    s = fe_structure()
    for site in s:
        site.label = None
    ws = st.workflow_slab(s, [1, 0, 0])
    assert ws.slab.site_properties['magmom'] == [2.2, -2.2, 2.2, -2.2]
    assert ws.added == [] and ws.dropped == ['if_pos']
    assert 'magmom' not in ws.slab.oriented_unit_cell.site_properties  # as the POSCAR of TribChem


def test_workflow_slab_database_path(tmp_path):
    dct = golden_cases.slab_dicts()['pt111']
    path = write(tmp_path, 'pt.json', json.dumps(dct))
    ws = st.read_workflow_slab(path, [1, 1, 1])
    assert ws.path == 'database' and not ws.rotated and ws.added == []
    assert json.dumps(ws.slab.as_dict(), default=str) == \
        json.dumps(Slab.from_dict(copy.deepcopy(dct)).as_dict(), default=str)  # run.py:1028
    with pytest.raises(st.StructureError, match='miller_index'):
        st.read_workflow_slab(path, [1, 0, 0])


# --- settings -------------------------------------------------------------------

def single_settings(tmp_path, structure):
    data = tomllib.loads((ROOT / 'examples' / 'single_input.toml').read_text())
    data['structure'].update(structure)
    settings = settings_from_dict(data)
    settings.source = tmp_path / 'settings.toml'
    return settings


def test_load_structure(tmp_path):
    (tmp_path / 'inputs').mkdir()
    write(tmp_path, 'inputs/a.pwi', PWI)
    settings = single_settings(tmp_path, {
        'file': 'inputs/a.pwi',
        'constraints': {'default_if_pos': [0, 0, 1], 'overrides': [{'atoms': [1], 'if_pos': [1, 1, 1]}]}})
    loaded = st.load_structure(settings)
    assert loaded.file == tmp_path / 'inputs' / 'a.pwi'
    assert loaded.constraints.source == ['override', 'default', 'file', 'file']
    assert loaded.constraints.as_dict()['if_pos'] == [[1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1]]


def test_load_structure_with_the_cell_of_the_settings(tmp_path):
    write(tmp_path, 'a.xyz', XYZ)
    settings = single_settings(tmp_path, {'file': 'a.xyz',
                                          'cell': {'source': 'config', 'vectors': XYZ_CELL}})
    np.testing.assert_allclose(st.load_structure(settings).structure.lattice.matrix, XYZ_CELL)


def test_load_structure_placeholder(tmp_path):
    with pytest.raises(st.StructureError, match='no structure file'):
        st.load_structure(single_settings(tmp_path, {'file': ''}))


def test_load_workflow_slabs(tmp_path):
    poscar(tmp_path, file='bottom.vasp')
    write(tmp_path, 'top.json', json.dumps(golden_cases.slab_dicts()['pt111']))
    data = tomllib.loads((ROOT / 'examples' / 'interface.toml').read_text())
    data['interface']['slabs'][0]['file'] = 'bottom.vasp'
    data['interface']['slabs'][1]['file'] = 'top.json'
    settings = settings_from_dict(data)
    settings.source = tmp_path / 'settings.toml'
    bottom, top = st.load_workflow_slabs(settings)
    assert (bottom.path, top.path) == ('custom_slab', 'database')
    assert isinstance(bottom.slab, Slab) and list(bottom.slab.miller_index) == [1, 1, 1]
