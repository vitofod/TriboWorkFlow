# Author: Vito Foderà
# Created: 2026-11-09

"""Tests of the structure read from a pw.x output (``triboworkflow.engines.qe.pwo``).

The outputs are synthetic (``pwo_helpers``), written for the lines the reader
looks at. The tests on real outputs read the directory named by
``TRIBOWORKFLOW_PWO_CORPUS`` and are skipped without it; real outputs are
never stored in the repository.
"""

import re
import tomllib
import warnings
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from triboworkflow.core.settings import settings_from_dict
from triboworkflow.core.structures import (FORMATS, StructureError, detect_format, read_structure,
                                           read_workflow_slab)
from triboworkflow.core.structures.model import IF_POS, QE_SPECIES
from triboworkflow.engines.qe import results as qe
from triboworkflow.engines.qe.pwo import pwo_structure, read_pwo_structure
from triboworkflow.engines.qe.results import pw_values

from pwo_helpers import SPECIES, pw_output, relaxed, structure
from test_qe_status_corpus import CORPUS, CORPUS_VARIABLE, read_corpus_file

ROOT = Path(__file__).resolve().parents[1]

# Cell, positions and cell of a slab whose c is along z.
CU_SPECIES = (('Cu', 'Cu', 'Cu.pbe-dn-kjpaw.UPF', 'a' * 32),)
FE_UP_DOWN = (('Fe1', 'Fe', 'Fe.pbe-sp-van_ak.UPF', 'b' * 32),
              ('Fe2', 'Fe', 'Fe.pbe-sp-van_ak.UPF', 'b' * 32))


def cu_slab(tilt=0.0):
    """Two Cu atoms in a cell with c along z, or tilted towards x by ``tilt``."""
    return Structure(Lattice([[2.55, 0.0, 0.0], [0.0, 4.42, 0.0], [tilt, 0.0, 20.0]]),
                     ['Cu', 'Cu'], [[0.0, 0.0, 1.0], [1.275, 2.21, 3.0]],
                     coords_are_cartesian=True)


def assert_geometry(got, expected_cell, expected_positions, atol=1e-5):
    np.testing.assert_allclose(got.lattice.matrix, expected_cell, rtol=0, atol=atol)
    np.testing.assert_allclose(got.cart_coords, expected_positions, rtol=0, atol=atol)


# --- scf and relax -------------------------------------------------------------------------

def test_scf_gives_the_atoms_of_the_header():
    s = structure()
    got = pwo_structure(pw_output(s, 'scf'))
    assert_geometry(got, s.lattice.matrix, s.cart_coords)
    assert got.labels == ['Fe1', 'H']
    assert [str(site.specie) for site in got] == ['Fe', 'H']
    assert IF_POS not in got.site_properties


def test_scf_species_are_kept_as_for_a_pwi_file():
    got = pwo_structure(pw_output(structure(), 'scf'))
    assert got.properties[QE_SPECIES] == [
        {'label': 'Fe1', 'mass': 1.008, 'pseudopotential': 'Fe.pbe-sp-van_ak.UPF',
         'starting_magnetization': None},
        {'label': 'H', 'mass': 1.008, 'pseudopotential': 'H.pbe-rrkjus_psl.1.0.0.UPF',
         'starting_magnetization': None}]


def test_relax_gives_the_final_coordinates_and_their_flags():
    s = structure()
    flags = [None, (1, 1, 0)]
    got = pwo_structure(pw_output(s, 'relax', final_if_pos=flags))
    assert_geometry(got, s.lattice.matrix, relaxed(s), atol=1e-5)
    # The final positions have ten decimals, the cell six decimals of alat.
    np.testing.assert_allclose(got.cart_coords, relaxed(s), rtol=0, atol=1e-9)
    assert got.labels == ['Fe1', 'H']
    assert got.site_properties[IF_POS] == [None, [1, 1, 0]]


def test_relax_without_flags_has_no_if_pos():
    got = pwo_structure(pw_output(structure(), 'relax'))
    assert IF_POS not in got.site_properties


def test_an_atom_given_all_free_reads_as_without_constraint():
    # pw.x prints the flags of the atoms with a fixed component only.
    got = pwo_structure(pw_output(structure(), 'relax', final_if_pos=[(1, 1, 1), (0, 0, 0)]))
    assert got.site_properties[IF_POS] == [None, [0, 0, 0]]


def test_only_the_last_run_is_read():
    first = structure()
    second = Structure(first.lattice, ['Fe', 'H'], relaxed(first, (0.5, 0.25, -0.125)),
                       coords_are_cartesian=True, labels=['Fe1', 'H'])
    text = pw_output(first, 'scf') + pw_output(second, 'scf')
    assert_geometry(pwo_structure(text), second.lattice.matrix, second.cart_coords)


def test_the_cell_has_the_digits_pw_x_prints():
    s = structure()
    got = pwo_structure(pw_output(s, 'scf'))
    values = pw_values(pw_output(s, 'scf'), 'scf')
    printed = np.asarray(values.axes) * values.alat * qe.BOHR_TO_ANGSTROM
    np.testing.assert_allclose(got.lattice.matrix, printed, rtol=0, atol=1e-12)
    # Not exact: six decimals of alat, a few 1e-6 angstrom for this cell.
    assert 0 < np.abs(got.lattice.matrix - s.lattice.matrix).max() < 1e-5


# --- magnetization -------------------------------------------------------------------------

def test_the_starting_magnetization_is_provenance_only():
    got = pwo_structure(pw_output(structure(), 'scf', magnetization=[0.4, 0.0]))
    assert [sp['starting_magnetization'] for sp in got.properties[QE_SPECIES]] == [0.4, 0.0]
    assert 'magmom' not in got.site_properties


def test_a_table_for_another_label_is_an_error():
    text = pw_output(structure(), 'scf', magnetization=[0.4, 0.0])
    text = re.sub(r'(\n\s+)Fe1(\s+0\.400)', r'\1Fe2\2', text)
    with pytest.raises(StructureError, match="starting magnetization of species 'Fe1'"):
        pwo_structure(text)


def test_a_noncollinear_table_is_not_read():
    text = pw_output(structure(), 'scf', magnetization=[0.4, 0.0])
    text = text.replace('atomic species   magnetization', 'atomic species   magnetization   angle1')
    with pytest.raises(StructureError, match='collinear'):
        pwo_structure(text)


def test_species_of_the_same_element_keep_their_labels():
    s = Structure(Lattice.cubic(5.0), ['Fe', 'Fe'], [[0, 0, 0], [2.5, 2.5, 2.5]],
                  coords_are_cartesian=True, labels=['Fe1', 'Fe2'])
    got = pwo_structure(pw_output(s, 'scf', species=FE_UP_DOWN, magnetization=[0.5, -0.5]))
    assert got.labels == ['Fe1', 'Fe2']
    assert [sp['starting_magnetization'] for sp in got.properties[QE_SPECIES]] == [0.5, -0.5]


# --- runs that are not read ----------------------------------------------------------------

def test_a_run_that_did_not_end_is_not_read():
    with pytest.raises(StructureError, match='not over and converged.*process None'):
        pwo_structure(pw_output(structure(), 'scf', job_done=False))


def test_a_scf_that_did_not_converge_is_not_read():
    text = pw_output(structure(), 'scf')
    text = text.replace('     convergence has been achieved in   7 iterations\n', '')
    with pytest.raises(StructureError, match='SCF None'):
        pwo_structure(text)


def test_a_relax_with_open_final_coordinates_is_not_read():
    with pytest.raises(StructureError, match='final coordinates incomplete'):
        pwo_structure(pw_output(structure(), 'relax', closed=False))


def test_a_relax_that_changes_the_cell_is_not_read():
    text = pw_output(structure(), 'relax')
    text = text.replace('Begin final coordinates\n', 'Begin final coordinates\n\nCELL_PARAMETERS '
                        '(angstrom)\n 1.0 0.0 0.0\n 0.0 1.0 0.0\n 0.0 0.0 1.0\n')
    with pytest.raises(StructureError, match='fixed cell'):
        pwo_structure(text)


def test_final_positions_in_another_unit_are_not_read():
    head, _, tail = pw_output(structure(), 'relax').rpartition('ATOMIC_POSITIONS (angstrom)')
    with pytest.raises(StructureError, match='only angstrom'):
        pwo_structure(head + 'ATOMIC_POSITIONS (bohr)' + tail)


def test_a_text_that_is_not_an_output_is_an_error():
    with pytest.raises(StructureError, match='not a pw.x output'):
        pwo_structure('not an output\n')


def test_a_label_that_contradicts_its_pseudopotential_is_an_error():
    species = (('Fe1', 'Cu', 'Cu.pbe.UPF', 'c' * 32), SPECIES[1])
    with pytest.raises(StructureError, match="'Fe1' starts with the element Fe.*is for Cu"):
        pwo_structure(pw_output(structure(), 'scf', species=species))


def test_a_label_that_names_no_element_is_an_error():
    s = Structure(Lattice.cubic(5.0), ['Fe'], [[0, 0, 0]], coords_are_cartesian=True,
                  labels=['Zz'])
    with pytest.raises(StructureError, match="label 'Zz'"):
        pwo_structure(pw_output(s, 'scf', species=(('Zz', 'Zz', 'Zz.UPF', 'd' * 32),)))


def test_an_unsupported_version_is_read_with_a_warning():
    with pytest.warns(UserWarning, match='pw.x 6.7'):
        got = pwo_structure(pw_output(structure(), 'scf', version='6.7'))
    assert len(got) == 2
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        pwo_structure(pw_output(structure(), 'scf', version='7.3'))


# --- read_structure and the settings -------------------------------------------------------

def test_read_structure_takes_the_pwo_format_from_the_name(tmp_path):
    s = structure()
    path = tmp_path / 'output.pwo'
    path.write_text(pw_output(s, 'relax'))
    assert 'pwo' in FORMATS and detect_format(path) == 'pwo'
    got = read_structure(path)
    np.testing.assert_allclose(got.cart_coords, relaxed(s), rtol=0, atol=1e-9)
    assert read_pwo_structure(path) == got


def test_read_structure_accepts_the_format_for_any_name(tmp_path):
    path = tmp_path / 'run.out'
    path.write_text(pw_output(structure(), 'scf'))
    with pytest.raises(StructureError, match='cannot tell the format'):
        read_structure(path)
    assert len(read_structure(path, format='pwo')) == 2


def test_a_pwo_file_holds_one_structure(tmp_path):
    path = tmp_path / 'output.pwo'
    path.write_text(pw_output(structure(), 'scf'))
    with pytest.raises(StructureError, match='frame must be 0'):
        read_structure(path, frame=1)


def test_read_structure_names_the_file_in_its_errors(tmp_path):
    path = tmp_path / 'output.pwo'
    path.write_text(pw_output(structure(), 'scf', job_done=False))
    with pytest.raises(StructureError, match=re.escape(str(path)) + ': the last run'):
        read_structure(path)


def test_a_given_cell_replaces_the_printed_one(tmp_path):
    s = structure()
    path = tmp_path / 'output.pwo'
    path.write_text(pw_output(s, 'scf'))
    got = read_structure(path, cell=s.lattice.matrix)
    np.testing.assert_allclose(got.lattice.matrix, s.lattice.matrix, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got.cart_coords, pwo_structure(path.read_text()).cart_coords,
                               rtol=0, atol=1e-12)


def test_the_settings_accept_the_pwo_format():
    data = tomllib.loads((ROOT / 'examples' / 'single_input.toml').read_text())
    data['structure']['format'] = 'pwo'
    assert settings_from_dict(data).structure.format == 'pwo'


# --- slabs of the workflows ----------------------------------------------------------------

def test_a_slab_with_c_along_z_is_read_from_an_output(tmp_path):
    path = tmp_path / 'slab.pwo'
    path.write_text(pw_output(cu_slab(), 'relax', species=CU_SPECIES))
    slab = read_workflow_slab(path, (1, 1, 1))
    assert slab.path == 'custom_slab' and not slab.rotated
    assert f'properties.{QE_SPECIES}' in slab.dropped


def test_a_slab_whose_c_is_tilted_is_refused_with_an_explicit_error(tmp_path):
    path = tmp_path / 'slab.pwo'
    path.write_text(pw_output(cu_slab(tilt=1.0), 'relax', species=CU_SPECIES))
    with pytest.raises(StructureError, match='c is not perpendicular to a'):
        read_workflow_slab(path, (1, 1, 1))


# --- real outputs --------------------------------------------------------------------------

corpus = pytest.mark.skipif(not CORPUS, reason=f'{CORPUS_VARIABLE} is not set')

RELAX = 'file_output_relax_convergenza_raggiunta_con_commenti.pwo'
SCF = 'file_output_scf_convergenza_raggiunta_con_stress_e_pressioneDiCella_e_con_commenti.pwo'
RELAX_FIXED = 'relax_andato_correttamente_a_convergenza.pwo'
NOT_OVER = [
    'relax_convergenza_non_raggiunta_dopo_200_iterazioni_con_commenti.pwo',
    'relax_crash_con_errore_con_commenti.pwo',
    'scf_interrotto_per_crash_durante_calcolo_con_commenti.pwo',
    'scf_interrotto_per_crash_durante_calcolo__con_messaggio_sistema_con_commenti.pwo']


def corpus_copy(tmp_path, name):
    """A copy of a corpus file without its comment lines, to be read by path."""
    path = tmp_path / name
    path.write_text(read_corpus_file(name))
    return path


@corpus
@pytest.mark.parametrize('name, calculation', [(RELAX, 'relax'), (SCF, 'scf'),
                                               (RELAX_FIXED, 'relax')])
def test_corpus_structure_is_what_the_output_prints(tmp_path, name, calculation):
    path = corpus_copy(tmp_path, name)
    values = pw_values(path.read_text(), calculation)
    scale = values.alat * qe.BOHR_TO_ANGSTROM
    got = read_structure(path)
    assert len(got) == values.nat
    np.testing.assert_allclose(got.lattice.matrix, np.asarray(values.axes) * scale, rtol=0,
                               atol=1e-12)
    expected = (values.final_positions if calculation == 'relax'
                else np.asarray(values.positions) * scale)
    np.testing.assert_allclose(got.cart_coords, expected, rtol=0, atol=1e-9)
    assert got.labels == list(values.final_labels if calculation == 'relax' else values.labels)
    assert [sp['label'] for sp in got.properties[QE_SPECIES]] == [s.label for s in values.species]


@corpus
def test_corpus_species_flags_and_magnetization(tmp_path):
    got = read_structure(corpus_copy(tmp_path, RELAX_FIXED))
    assert [(sp['label'], sp['mass'], sp['pseudopotential'], sp['starting_magnetization'])
            for sp in got.properties[QE_SPECIES]] == [
        ('Fe', 55.845, 'Fe.pbe-sp-van_ak.UPF', 0.4),
        ('C', 12.011, 'C.pbe-n-rrkjus_psl.1.0.0.UPF', 0.0),
        ('H', 1.008, 'H.pbe-rrkjus_psl.1.0.0.UPF', 0.0)]
    assert [str(site.specie) for site in got] == ['Fe'] * 32 + ['C', 'H', 'H']
    assert sum(f is not None for f in got.site_properties[IF_POS]) == 5


@corpus
@pytest.mark.parametrize('name', NOT_OVER)
def test_corpus_runs_that_are_not_over_are_not_read(tmp_path, name):
    with pytest.raises(StructureError, match='not over and converged'):
        read_structure(corpus_copy(tmp_path, name))


@corpus
def test_corpus_cell_with_a_tilted_a_is_refused_as_a_slab(tmp_path):
    # a(1) = (0.998727, 0, -0.050443) alat with c = (0, 0, 3.304038) alat.
    with pytest.raises(StructureError, match='c is not perpendicular to a'):
        read_workflow_slab(corpus_copy(tmp_path, RELAX_FIXED), (1, 1, 1))
