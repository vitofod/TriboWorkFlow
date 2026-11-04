# Author: Vito Foderà
# Created: 2026-11-04

"""Tests of the reading of the charge density of pp.x (``engines/qe/density.py``).

The files are synthetic (``charge_files.py``), in the formats of pp.x;
``TRIBOWORKFLOW_QE_PP_EXAMPLE`` may point to the folder of the pp.x example
of QE (``PP/examples/example01``, never stored in the repository), whose
``reference/sicharge`` is read too.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

import charge_files
from triboworkflow.core.store import Engine, Record
from triboworkflow.engines.qe import density
from triboworkflow.engines.qe.density import DensityError
from triboworkflow.vendored.physics.electronics.charge import chgcar_integrator as ci
from triboworkflow.vendored.physics.electronics.charge.analyzer import QEChargeAnalyzer

BOHR = density.BOHR_TO_ANGSTROM
GRID = (6, 5, 40)
CELLS = {'orthorhombic': np.diag([2.5, 2.8, 20.0]),
         'hexagonal': np.array([[2.556, 0.0, 0.0], [-1.278, 2.556 * math.sqrt(3) / 2, 0.0],
                                [0.0, 0.0, 20.0]])}
LAYERS = {'bot': [(4.0, 11.0), (6.1, 11.0), (8.2, 11.0)],
          'top': [(10.6, 3.0), (12.9, 3.0), (15.2, 3.0)]}
SPECIES = {'bot': ('Cu', 11.0), 'top': ('Al', 3.0)}
ELECTRONS = {'interface': 42.0, 'bot': 33.0, 'top': 9.0}
FP = 'sha256:' + 'a' * 64


def part(cell_name='orthorhombic', name='interface', labels=None):
    """Structure, filplot text and density (e/A^3) of a toy interface or slab."""
    cell = CELLS[cell_name]
    slabs = ['bot', 'top'] if name == 'interface' else [name]
    layers = [layer for s in slabs for layer in LAYERS[s]]
    rho = charge_files.slab_density(cell, GRID, layers,
                                    transfer=(9.4, 0.4, 0.5) if name == 'interface' else None)
    inplane = [(0.0, 0.0), (0.5, 0.5), (0.0, 0.5)]
    frac = [(*inplane[i % 3], z / cell[2, 2]) for s in slabs
            for i, (z, _) in enumerate(LAYERS[s])]
    names = [SPECIES[s][0] for s in slabs for _ in LAYERS[s]]
    structure = Structure(Lattice(cell), names, frac, labels=labels or names)
    return structure, rho


def filplot_text(cell_name='orthorhombic', name='interface', rho=None, structure=None, **kwargs):
    s, r = part(cell_name, name)
    structure = structure or s
    rho = r if rho is None else rho
    cell = np.asarray(structure.lattice.matrix)
    alat = np.linalg.norm(cell[0]) / BOHR
    slabs = ['bot', 'top'] if name == 'interface' else [name]
    species = [SPECIES[s] for s in slabs]
    types = [1 + [n for n, _ in species].index(str(e)) for e in structure.species]
    tau = np.asarray(structure.cart_coords) / (alat * BOHR)
    args = dict(values=rho * BOHR ** 3, celldm=[alat], species=species,
                atoms=list(zip(tau.tolist(), types)), ibrav=0, at=cell / (alat * BOHR),
                gcutm=160.0 / (2 * math.pi / alat) ** 2, dual=4.0, ecut=40.0)
    args.update(kwargs)
    return charge_files.filplot_text(**args)


@pytest.fixture
def interface():
    structure, rho = part()
    return structure, rho, density.parse_filplot(filplot_text())


# =============================================================================
# Reading
# =============================================================================

@pytest.mark.parametrize('cell_name', list(CELLS))
def test_reads_the_header_and_values(cell_name):
    structure, rho = part(cell_name)
    f = density.parse_filplot(filplot_text(cell_name))
    assert f.title == '' and f.grid == GRID and f.padded == GRID and f.ibrav == 0
    assert f.alat == pytest.approx(np.linalg.norm(CELLS[cell_name][0]) / BOHR, abs=1e-8)
    # celldm(1) has eight decimals in bohr: c (7.8 alat) to 2e-8 A.
    assert np.allclose(f.cell, CELLS[cell_name], rtol=0, atol=3e-8)
    assert (f.ecut, f.dual, f.plot_num) == (40.0, 4.0, 0)
    assert f.species == (('Cu', 11.0), ('Al', 3.0))
    assert f.labels == ('Cu',) * 3 + ('Al',) * 3 and f.types == (1, 1, 1, 2, 2, 2)
    assert np.allclose(f.cart_positions, structure.cart_coords, rtol=0, atol=3e-8)
    assert f.values.shape == (GRID[2], GRID[1], GRID[0])
    assert np.allclose(f.values, rho * BOHR ** 3, rtol=1e-9, atol=0)
    assert f.volume == pytest.approx(abs(np.linalg.det(CELLS[cell_name])), rel=1e-8)
    assert f.electrons == pytest.approx(42.0, rel=1e-8)


def test_padding_is_dropped():
    _, rho = part()
    plain = density.parse_filplot(filplot_text())
    padded = density.parse_filplot(filplot_text(nrx=(7, 6, 41), pad=999.0))
    assert padded.padded == (7, 6, 41) and padded.grid == GRID
    assert np.array_equal(padded.values, plain.values)
    assert padded.electrons == plain.electrons


def test_title_is_kept_and_does_not_matter():
    f = density.parse_filplot(filplot_text(title='  some title'))
    assert f.title == '  some title' and f.electrons == pytest.approx(42.0, rel=1e-8)


def test_three_digit_exponents_as_fortran_writes_them():
    _, rho = part()
    rho = rho.copy()
    rho[0, 0, 0] = 1.5E-110 / BOHR ** 3
    text = filplot_text(rho=rho)
    assert '1.500000000-110' in text
    f = density.parse_filplot(text)
    assert f.values[0, 0, 0] == pytest.approx(1.5E-110, rel=1e-9)


def replace_line(text, n, new):
    lines = text.splitlines()
    lines[n] = new
    return '\n'.join(lines) + '\n'


@pytest.mark.parametrize('change, message', [
    (lambda t: t.rsplit('\n', 2)[0] + '\n', 'values after the atoms instead of'),
    (lambda t: t + '  1.000000000E+00\n', 'values after the atoms instead of'),
    (lambda t: t.replace('E-', '*-', 1).replace('*-', '*******', 1),
     'did not fit its Fortran format'),
    (lambda t: replace_line(t, -1, '              NaN' + '  1.000000000E+00' * 4),
     'not a finite number'),
    (lambda t: replace_line(t, 1, '       6       5      40       6       5      40       6'),
     '7 fields instead of 8'),
    (lambda t: replace_line(t, 1, ''.join(f'{n:8d}' for n in (6, 5, 40, 7, 5, 40, 6, 2))),
     'exceed the leading dimensions'),
    # Lines: 0 title, 1 sizes, 2 ibrav, 3-5 cell, 6 cutoffs, 7-8 species, 9-14 atoms.
    (lambda t: replace_line(t, 7, '   2   Cu   11.00'), 'species 2 instead of 1'),
    (lambda t: replace_line(t, 9, '   2       0.000000000    0.000000000    0.640000000    1'),
     'atom 2 instead of 1'),
    (lambda t: replace_line(t, 9, '   1       0.000000000    0.000000000    0.640000000    3'),
     'species 3 of atom 1 is not in 1..2'),
    (lambda t: replace_line(t, 9, '   1       0.000000000    *************    0.640000000    1'),
     'did not fit its Fortran format'),
    (lambda t: replace_line(t, 3, '  1.0  0.0'), '2 fields instead of 3'),
    (lambda t: '', 'empty file'),
])
def test_reading_errors(change, message):
    with pytest.raises(DensityError, match=message.replace('(', r'\(').replace('.', r'\.')
                       .replace('*', r'\*')):
        density.parse_filplot(change(filplot_text()))


def test_read_filplot_names_the_file(tmp_path):
    path = tmp_path / 'density.filplot'
    path.write_text(filplot_text())
    assert density.read_filplot(path).grid == GRID
    path.write_text('\n')
    with pytest.raises(DensityError, match='density.filplot: line 2: missing'):
        density.read_filplot(path)


# =============================================================================
# Structure and CHGCAR form
# =============================================================================

def test_structure_agrees(interface):
    structure, _, f = interface
    assert density.structure_differences(f, structure) == []


def test_labels_are_compared_on_two_characters():
    structure, _ = part(labels=['Cu1', 'Cu1', 'Cu2', 'Al', 'Al', 'Al'])
    f = density.parse_filplot(filplot_text())
    assert density.structure_differences(f, structure) == []


def test_structure_differences(interface):
    structure, _, f = interface
    moved = structure.copy()
    moved.translate_sites([4], [0, 0, 2e-6], frac_coords=False, to_unit_cell=False)
    assert density.structure_differences(f, moved) == [
        'positions of atoms [5] differ from those of the structure beyond the digits printed']
    other = Structure(Lattice(np.diag([2.5, 2.8, 20.00001])), structure.species,
                      structure.cart_coords, coords_are_cartesian=True, labels=structure.labels)
    assert 'cell' in density.structure_differences(f, other)[0]
    swapped = Structure(structure.lattice, ['Al'] + [str(e) for e in structure.species[1:]],
                        structure.frac_coords)
    assert density.structure_differences(f, swapped)[0].startswith('species of atoms [1]')
    fewer = structure.copy()
    fewer.remove_sites([5])
    assert density.structure_differences(f, fewer) == ['6 atoms in the file, 5 in the structure']


@pytest.mark.parametrize('cell_name', list(CELLS))
def test_chgcar_has_the_conventions_of_a_vasp_chgcar(cell_name):
    structure, rho = part(cell_name)
    chg = density.to_chgcar(density.parse_filplot(filplot_text(cell_name)), structure)
    assert chg.data['total'].shape == GRID
    assert chg.structure == structure and chg.structure is not structure
    # data = rho * V (e), indexed along the cell vectors: its mean is the electron count.
    assert chg.data['total'].mean() == pytest.approx(42.0, rel=1e-8)
    assert np.allclose(chg.data['total'] / structure.volume, rho.transpose(2, 1, 0), rtol=1e-8)
    x, avg = ci.chgcar_average_along_axis(chg, 2)
    assert np.allclose(avg, rho.mean(axis=(1, 2)), rtol=1e-8)
    assert x[-1] == pytest.approx(structure.lattice.c)
    x, avg = ci.chgcar_average_along_axis(chg, 0)
    assert np.allclose(avg, rho.mean(axis=(0, 1)), rtol=1e-8)


def test_charge_displacement_of_interface_and_slabs():
    """The algebra of the workflow on the three objects equals the difference of the files."""
    chgcars, rhos = [], []
    for name in ('interface', 'bot', 'top'):
        structure, rho = part(name=name)
        chgcars.append(density.to_chgcar(density.parse_filplot(filplot_text(name=name)),
                                         structure))
        rhos.append(rho)
    with pytest.warns(UserWarning, match='Structures are different'):
        combined = ci.chgcar_algebra_from_list(chgcars, ['-', '-'], 2)
    difference = (rhos[0] - rhos[1] - rhos[2]).transpose(2, 1, 0)
    volume = chgcars[0].structure.volume
    assert np.allclose(combined.data['total'] / volume, difference, rtol=0, atol=1e-8)
    integral, width, _, _ = ci.charge_displacement(combined, 2, 8.2, 10.6, 10000)
    assert width == pytest.approx(2.4) and integral > 0


def test_chgcar_errors(interface):
    structure, _, f = interface
    with pytest.raises(DensityError, match=r'plot_num = 1: not the charge density'):
        density.to_chgcar(density.parse_filplot(filplot_text(plot_num=1)), structure)
    moved = structure.copy()
    moved.translate_sites([0], [0, 0, 0.01], frac_coords=False)
    with pytest.raises(DensityError, match='disagrees with the structure'):
        density.to_chgcar(f, moved)


def test_the_analyzer_reads_the_same_file(tmp_path, interface):
    """QEChargeAnalyzer reads the same file, with its own Bohr constant (2e-7 apart)."""
    _, rho, f = interface
    path = tmp_path / 'interface'
    path.write_text(filplot_text())
    analyzer = QEChargeAnalyzer(str(path), unpack=False)
    assert analyzer.charge_grid.shape == f.values.shape
    assert np.allclose(analyzer.charge_grid, rho, rtol=1e-6, atol=0)
    assert not np.allclose(analyzer.charge_grid, rho, rtol=1e-8, atol=0)
    analyzer.average_charge('xyz', to_file=False, to_plot=False)
    assert analyzer.charge['cell']['rho_tot'] == pytest.approx(42.0, rel=1e-6)


# =============================================================================
# Comparison of interface and slabs
# =============================================================================

def three(**changes):
    files = {}
    for name in ('interface', 'bot', 'top'):
        files[name] = density.parse_filplot(filplot_text(name=name, **changes.get(name, {})))
    return files


def record(label, model):
    return Record(system='Cu111_Al111', step='charge', label=label, calc_type='slab_from_scratch',
                  engine=Engine('qe', 'pw.x'), energy_model=model, fingerprint=FP)


def test_comparable_densities():
    files = three()
    assert density.comparison_differences(files) == []
    model = {'version': '7.2', 'species': [{'label': 'Cu', 'md5': 'x'}]}
    density.check_comparable(files, [record(n, model) for n in files])


def test_comparison_differences():
    _, rho = part(name='bot')
    coarse = rho[::2]
    files = three(bot={'rho': coarse}, top={'ecut': 45.0, 'plot_num': 1})
    found = density.comparison_differences(files)
    assert found[0] == 'top: plot_num = 1, not the charge density'
    assert 'bot: grid (6, 5, 20), (6, 5, 40) in interface' in found
    assert "top: ecut = 45.0, 40.0 in interface" in found
    assert len(found) == 3


def test_comparison_of_cells():
    structure, rho = part(name='top')
    stretched = Structure(Lattice(np.diag([2.5, 2.8, 20.001])), structure.species,
                          structure.cart_coords, coords_are_cartesian=True)
    found = density.comparison_differences(three(top={'structure': stretched}))
    assert len(found) == 1 and found[0].startswith('top: cell ')
    # gcutm follows alat = |a1|, not changed above.
    found = density.comparison_differences(three(bot={'gcutm': 9.0}))
    assert len(found) == 1 and found[0].startswith('bot: gcutm = 9.0, ')


def test_check_comparable_lists_everything():
    files = three(top={'ecut': 45.0})
    models = [{'version': '7.2'}, {'version': '7.2'}, {'version': '7.3'}]
    with pytest.raises(DensityError) as info:
        density.check_comparable(files, [record(n, m) for n, m in zip(files, models)])
    message = str(info.value)
    assert 'top: ecut = 45.0' in message and 'different energy models' in message
    with pytest.raises(DensityError, match='energy model not known'):
        density.check_comparable(three(), [record(n, {'version': None}) for n in files])


# =============================================================================
# Example of QE
# =============================================================================

EXAMPLE = os.environ.get('TRIBOWORKFLOW_QE_PP_EXAMPLE')


@pytest.mark.skipif(not EXAMPLE, reason='TRIBOWORKFLOW_QE_PP_EXAMPLE not set')
def test_pp_example_of_silicon():
    """reference/sicharge of PP/examples/example01: Si fcc (ibrav 2), 20^3 grid, 8 electrons."""
    folder = Path(EXAMPLE) / 'reference'
    f = density.read_filplot(folder / 'sicharge')
    assert f.title == '' and f.grid == (20, 20, 20) and f.ibrav == 2 and f.axes is None
    assert f.alat == 10.2 and f.species == (('Si', 4.0),) and f.types == (1, 1)
    assert np.allclose(f.positions, [[0, 0, 0], [0.25, 0.25, 0.25]])
    assert (f.ecut, f.dual, f.plot_num) == (18.0, 4.0, 0)
    scf = (folder / 'si.scf.out').read_text()
    assert 'FFT dimensions: (  20,  20,  20)' in scf
    assert 'unit-cell volume          =     265.3020' in scf
    # fcc cell of pw.x (ibrav = 2): volume alat^3 / 4.
    assert float(f.values.mean()) * f.alat ** 3 / 4 == pytest.approx(8.0, rel=1e-6)
    with pytest.raises(DensityError, match='ibrav = 2'):
        f.cell
