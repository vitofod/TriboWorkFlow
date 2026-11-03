# Author: Vito Foderà
# Created: 2026-11-03

"""Tests of ``triboworkflow.engines.qe.results`` on real pw.x outputs.

Real outputs are never stored in the repository: they are read from the
directory named by ``TRIBOWORKFLOW_PWO_CORPUS``, as in
``test_qe_status_corpus.py``, whose comment lines (``###``) are removed
before reading. Without the variable these tests are skipped.

The inputs of these outputs are not in the corpus: the input structure of
each one is rebuilt from the header of its last run (cell, labels,
positions) and from the flags of its final coordinates, so that the checks
against the input pass by construction and the values read are what is
tested.

The values are also compared with those of ASE's ``read_espresso_out``, the
reader Xsorb starts from. ASE converts with the constants of
``ase.io.espresso`` (CODATA 2006), this package with those of QE (CODATA
2018): the values of ASE are brought to the constants of QE before the
comparison, which then allows only the digits printed.
"""

import io

import ase.io
import numpy as np
import pytest
from ase.io.espresso import units as ase_units
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from triboworkflow.engines.qe import results as qe
from triboworkflow.engines.qe.results import parse_output, pw_values

from test_qe_status_corpus import CORPUS, CORPUS_VARIABLE, read_corpus_file

pytestmark = pytest.mark.skipif(not CORPUS, reason=f'{CORPUS_VARIABLE} is not set')

RELAX = 'file_output_relax_convergenza_raggiunta_con_commenti.pwo'
SCF = 'file_output_scf_convergenza_raggiunta_con_stress_e_pressioneDiCella_e_con_commenti.pwo'
RELAX_FIXED = 'relax_andato_correttamente_a_convergenza.pwo'

# (file, calculation, parsing, nsites, energy in Ry as printed, number of
# atoms with flags in the final coordinates)
EXPECTED = [
    (RELAX, 'relax', 'ok', 50, -12411.1500181581, 0),
    (SCF, 'scf', 'ok', 1, -253.46613429, None),
    (RELAX_FIXED, 'relax', 'ok', 35, -8125.0815794407, 5),
    ('relax_convergenza_non_raggiunta_dopo_200_iterazioni_con_commenti.pwo', 'relax', None,
     None, None, None),
    ('relax_crash_con_errore_con_commenti.pwo', 'relax', None, None, None, None),
    ('scf_interrotto_per_crash_durante_calcolo_con_commenti.pwo', 'scf', None, None, None, None),
    ('scf_interrotto_per_crash_durante_calcolo__con_messaggio_sistema_con_commenti.pwo', 'scf',
     None, None, None, None),
]


def input_structure(text, calculation):
    """Input structure of an output, rebuilt from its header and final coordinates."""
    values = pw_values(text, calculation)
    scale = values.alat * qe.BOHR_TO_ANGSTROM
    structure = Structure(Lattice(np.asarray(values.axes) * scale),
                          [label.rstrip('0123456789') for label in values.labels],
                          np.asarray(values.positions) * scale, coords_are_cartesian=True,
                          labels=list(values.labels))
    if values.final_if_pos is not None:
        structure.add_site_property('if_pos', [[1, 1, 1] if f is None else list(f)
                                               for f in values.final_if_pos])
    return structure


@pytest.mark.parametrize('name, calculation, parsing, nsites, energy, flagged', EXPECTED,
                         ids=[case[0] for case in EXPECTED])
def test_corpus_file(name, calculation, parsing, nsites, energy, flagged):
    text = read_corpus_file(name)
    structure = input_structure(text, calculation) if parsing else None
    p = parse_output(text, calculation, structure)
    assert (p.parsing, p.reason) == (parsing, None)
    if parsing is None:
        assert p == qe.Parsed(None)
        return
    out = p.results['output']
    assert p.results['nsites'] == nsites
    assert out['energy'] == energy * qe.RY_TO_EV
    assert len(out['forces']) == nsites
    if flagged is not None:
        assert sum(f is not None for f in p.values.final_if_pos) == flagged


def test_corpus_final_energy_equals_the_last_total_energy():
    for name, last in ((RELAX, -12411.15001816), (RELAX_FIXED, -8125.08157944)):
        values = pw_values(read_corpus_file(name), 'relax')
        assert values.energy == last
        assert abs(values.final_energy - values.energy) <= 0.5e-8


def test_corpus_stress_in_kbar_agrees_with_the_columns_printed():
    # total   stress  (Ry/bohr**3)  (kbar)  P= -0.36, diagonal -0.00000242 Ry/bohr^3.
    text = read_corpus_file(SCF)
    stress = np.array(parse_output(text, 'scf', input_structure(text, 'scf')).results['output']
                      ['stress'])
    printed = np.array([[-0.36, 0.00, -0.00], [0.00, -0.36, 0.00], [-0.00, 0.00, -0.36]])
    # Two decimals in kbar; eight in Ry/bohr^3, i.e. 0.5e-8 * RY_KBAR.
    assert np.all(np.abs(stress - printed) <= 0.005 + 0.5e-8 * qe.RY_KBAR)
    assert np.trace(stress) / 3 == pytest.approx(-0.36, abs=0.005 + 0.5e-8 * qe.RY_KBAR)


def test_corpus_species_and_checksums():
    values = pw_values(read_corpus_file(RELAX_FIXED), 'relax')
    assert [(s.label, s.pseudopotential.rsplit('/', 1)[1], s.md5) for s in values.species] == [
        ('Fe', 'Fe.pbe-sp-van_ak.UPF', '327243ea9db589a56bc50e6ec7945b51'),
        ('C', 'C.pbe-n-rrkjus_psl.1.0.0.UPF', '1fdd950c42eb37d8afe3dd2f8d12311e'),
        ('H', 'H.pbe-rrkjus_psl.1.0.0.UPF', '3f4114867bd07edc7f1424d066fc1888')]


# --- Comparison with ASE ---------------------------------------------------------------------

RY = qe.RY_TO_EV / ase_units['Ry']
BOHR = qe.BOHR_TO_ANGSTROM / ase_units['Bohr']


@pytest.mark.parametrize('name, calculation', [(RELAX, 'relax'), (SCF, 'scf'),
                                               (RELAX_FIXED, 'relax')])
def test_corpus_values_agree_with_ase(name, calculation):
    text = read_corpus_file(name)
    p = parse_output(text, calculation, input_structure(text, calculation))
    out = p.results['output']
    atoms = ase.io.read(io.StringIO(text), format='espresso-out', index=-1)
    # Energy: ASE reads the last "!" line (eight decimals); Final energy has ten.
    assert out['energy'] == pytest.approx(atoms.get_potential_energy() * RY,
                                          abs=0.5e-8 * qe.RY_TO_EV + 1e-9)
    # Forces: the same numbers, with other constants.
    assert np.allclose(out['forces'], atoms.get_forces() * RY / BOHR, rtol=1e-12, atol=1e-15)
    if calculation == 'relax':
        # The final coordinates are the positions of the last SCF cycle, read
        # by ASE from the last ATOMIC_POSITIONS, in angstrom: no constant.
        final = Structure.from_dict(out['structure'])
        assert np.allclose(final.cart_coords, atoms.get_positions(), rtol=0, atol=1e-10)
        assert final.labels == list(p.values.final_labels)
    else:
        # ASE: -stress in eV/angstrom^3, Voigt order xx yy zz yz xz xy.
        s = np.array(out['stress']) / qe.RY_KBAR * qe.RY_TO_EV / qe.BOHR_TO_ANGSTROM ** 3
        voigt = -np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
        assert np.allclose(voigt, atoms.get_stress() * RY / BOHR ** 3, rtol=1e-12, atol=1e-20)
