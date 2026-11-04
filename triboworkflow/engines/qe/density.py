# Author: Vito Foderà
# Created: 2026-11-04

"""Charge density written by pp.x, and its form for the charge analysis.

pp.x run with ``plot_num = 0`` writes the electron (pseudo-)charge density of
a pw.x calculation on the real-space grid of the density (the ``filplot`` of
``&INPUTPP``). ``read_filplot`` reads that file into a ``Filplot``, with the
values and units of the file; ``to_chgcar`` turns it into the object the
charge analysis works on, a pymatgen ``Chgcar`` with the conventions of a VASP
CHGCAR; ``comparison_differences`` and ``check_comparable`` tell whether the
densities of an interface and of its slabs can be combined.

The file
--------
pp.x writes it with these Fortran formats (QE 7.1-7.3, ``plot_io``):

==============================  ==================================================
Format                          Content
==============================  ==================================================
``(a)``                         title, 75 characters; blank for the density
``(8i8)``                       ``nr1x nr2x nr3x nr1 nr2 nr3 nat ntyp``
``(i6,2x,6f16.8)``              ``ibrav`` and ``celldm(1:6)`` (``alat`` in bohr)
list-directed, three lines      cell vectors ``a1``, ``a2``, ``a3`` in units of
                                ``alat``, only with ``ibrav = 0``
``(3f20.10,i6)``                ``gcutm`` (units of ``(2 pi/alat)^2``), ``dual``,
                                ``ecutwfc`` (Ry), ``plot_num``
``(i4,3x,a2,3x,f5.2)``          per species: index, name (first two characters of
                                the label), valence
``(i4,3x,3f15.9,3x,i2)``        per atom: index, position (units of ``alat``),
                                species
``5(1pe17.9)``                  ``nr1x * nr2x * nr3`` values, ``x`` fastest
==============================  ==================================================

The values fill the arrays of the FFT grid, whose leading dimensions
``nr1x``, ``nr2x`` may exceed ``nr1``, ``nr2`` (with ESSL, or on NEC SX, a
power of 2 or an even size gets one more element); only ``nr3`` planes are
written. The padding is dropped on reading. With ``plot_num = 0`` the values
are in e/bohr^3 and integrate to the number of electrons (``spin_component =
0``: the total density, also with ``nspin = 2`` or noncollinear spin). The
grid points are at ``(i/nr1, j/nr2, k/nr3)`` in crystal coordinates, the first
at the origin of the cell.

Reading is strict: every line in its place, the indices of species and atoms
in order, exactly the number of values announced, every value a finite
number. A field that did not fit its format (``****``) is an error. Values
with an exponent beyond two digits, which Fortran writes without the ``E``
(``1.234567890-100``), are read as such.

The CHGCAR form
---------------
``to_chgcar`` gives a pymatgen ``Chgcar`` whose ``data['total']`` is the
density times the volume of the cell (e), indexed ``[i, j, k]`` along the
three cell vectors, and whose structure is the one of the calculation, in
angstrom: the meaning and the units of a VASP CHGCAR read by pymatgen. The
average of ``data['total']`` divided by the volume is the density in e/A^3,
its average the number of electrons. The structure must agree with the file
(``structure_differences``): same atoms, labels (the two characters written),
cell and positions to the digits printed. Only ``ibrav = 0``, the cells of
this package, gives the cell; files with another ``ibrav`` are read but not
converted.

Comparable densities
--------------------
The densities of an interface and of its slabs are subtracted point by
point, so they must lie on the same grid in the same cell, from the same
cutoffs (``comparison_differences``): the same ``nr1, nr2, nr3``, the same
cell to the digits printed, the same ``ecutwfc``, ``dual`` and ``gcutm``, the
charge density (``plot_num = 0``) in each. The grid starts at the origin of
the cell in every file, so the same cell is the same grid in space. Their
calculations must share one energy model (``core.store.check_energy_model``):
``check_comparable`` checks both.

Notes
-----
The constant ``BOHR_TO_ANGSTROM`` is the one of QE 7.1-7.3 (CODATA 2018,
``results.BOHR_TO_ANGSTROM``).

To remove: TribChem read the CHGCAR files of the three VASP calculations with
``Chgcar.from_file`` (``chgcar_integrator.extract_chgcar_data``) and averaged
``data['total'] / volume``; the analyzer of pp.x files
(``analyzer.QEChargeAnalyzer``) reads the same filplot, with its own Bohr
constant, and is not called by the workflow.
"""

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pymatgen.io.vasp.outputs import Chgcar

from triboworkflow.core.store import StoreError, check_energy_model

from .results import BOHR_TO_ANGSTROM

TITLE_LENGTH = 75
DENSITY_PLOT_NUM = 0

# Half of the last digit printed: celldm(1) (bohr), positions (units of
# alat), and the cell and positions of the extxyz files (angstrom).
_HALF_ALAT = 0.5E-8
_HALF_TAU = 0.5E-9
_HALF_EXTXYZ = 0.5E-8
# Relative precision of the list-directed cell vectors (17 digits) and of
# gcutm, computed from alat.
_LIST_DIRECTED = 1E-15
_GCUTM = 1E-8
_MARGIN = 1.01

# A real written by Fortran with a three-digit exponent, without the E.
_FORTRAN_EXPONENT = re.compile(r'([+-]?\d*\.\d*)([+-]\d{3})')


class DensityError(ValueError):
    """A charge-density file that cannot be read, or densities that do not agree."""


@dataclass(frozen=True)
class Filplot:
    """Content of a pp.x filplot, in the units of the file.

    Parameters
    ----------
    title : str
        First line, without trailing blanks.
    grid : tuple of int
        ``(nr1, nr2, nr3)``, points along the three cell vectors.
    padded : tuple of int
        ``(nr1x, nr2x, nr3x)``, leading dimensions of the arrays of pp.x.
    ibrav : int
        Bravais-lattice index.
    celldm : tuple of float
        ``celldm(1:6)``; ``celldm[0]`` is ``alat``, in bohr.
    axes : ndarray, shape (3, 3), or None
        Cell vectors (rows) in units of ``alat``; given with ``ibrav = 0``.
    gcutm : float
        Cutoff of the density, in units of ``(2 pi / alat)^2``.
    dual : float
        Ratio of the density cutoff to the wavefunction cutoff.
    ecut : float
        Wavefunction cutoff ``ecutwfc``, in Ry.
    plot_num : int
        Quantity written (0: the charge density).
    species : tuple of (str, float)
        Name (the first two characters of the label) and valence of each
        species, in the order of pw.x.
    positions : ndarray, shape (nat, 3)
        Positions of the atoms, in units of ``alat``.
    types : tuple of int
        Species of each atom, counted from 1.
    values : ndarray, shape (nr3, nr2, nr1)
        Values on the grid without padding, index of the third vector first;
        e/bohr^3 for the charge density.
    """

    title: str
    grid: tuple
    padded: tuple
    ibrav: int
    celldm: tuple
    axes: np.ndarray | None
    gcutm: float
    dual: float
    ecut: float
    plot_num: int
    species: tuple
    positions: np.ndarray
    types: tuple
    values: np.ndarray

    @property
    def alat(self):
        """``celldm(1)``, in bohr."""
        return self.celldm[0]

    @property
    def labels(self):
        """Species name of every atom, as written (two characters at most)."""
        return tuple(self.species[t - 1][0] for t in self.types)

    @property
    def cell(self):
        """Cell vectors (rows), in angstrom; only with ``ibrav = 0``.

        Raises
        ------
        DensityError
            With another ``ibrav``: the cell is given by ``celldm`` in the
            conventions of pw.x, not read here.
        """
        if self.ibrav != 0 or self.axes is None:
            raise DensityError(f'ibrav = {self.ibrav}: only files with ibrav = 0 give the cell '
                               '(the cells of this package)')
        return self.axes * (self.alat * BOHR_TO_ANGSTROM)

    @property
    def cart_positions(self):
        """Positions of the atoms, in angstrom."""
        return self.positions * (self.alat * BOHR_TO_ANGSTROM)

    @property
    def volume(self):
        """Volume of the cell, in angstrom^3 (``ibrav = 0`` only)."""
        return abs(float(np.linalg.det(self.cell)))

    @property
    def electrons(self):
        """Integral of the values over the cell: the electrons, for the density.

        Mean value times the volume in bohr^3 (``ibrav = 0`` only).
        """
        return float(self.values.mean()) * self.volume / BOHR_TO_ANGSTROM ** 3


def _number(token, where):
    """A finite float from a token of the file."""
    try:
        value = float(token)
    except ValueError:
        match = _FORTRAN_EXPONENT.fullmatch(token)
        if match is None:
            hint = ' (a value that did not fit its Fortran format)' if '*' in token else ''
            raise DensityError(f'{where}: {token!r} is not a number{hint}') from None
        value = float(f'{match.group(1)}E{match.group(2)}')
    if not math.isfinite(value):
        raise DensityError(f'{where}: {token!r} is not a finite number')
    return value


def _integer(token, where):
    if not re.fullmatch(r'[+-]?\d+', token):
        hint = ' (a value that did not fit its Fortran format)' if '*' in token else ''
        raise DensityError(f'{where}: {token!r} is not an integer{hint}')
    return int(token)


def _fields(lines, n, count, what):
    """Fields of line ``n`` (counted from 0), which must be ``count``."""
    if n >= len(lines):
        raise DensityError(f'line {n + 1}: missing, {what} expected')
    fields = lines[n].split()
    if len(fields) != count:
        raise DensityError(f'line {n + 1}: {len(fields)} fields instead of {count} ({what})')
    return fields


def parse_filplot(text):
    """Read the text of a pp.x filplot.

    Parameters
    ----------
    text : str

    Returns
    -------
    Filplot

    Raises
    ------
    DensityError
        If a line is missing or out of format, an index is out of order, a
        number is not finite or did not fit its format, or the number of
        values is not ``nr1x * nr2x * nr3``.
    """
    lines = text.splitlines()
    if not lines:
        raise DensityError('empty file')
    title = lines[0].rstrip()
    sizes = [_integer(f, 'line 2') for f in _fields(lines, 1, 8, 'nr1x nr2x nr3x nr1 nr2 nr3 '
                                                     'nat ntyp')]
    nr1x, nr2x, nr3x, nr1, nr2, nr3, nat, ntyp = sizes
    if min(sizes) < 1 or nr1 > nr1x or nr2 > nr2x or nr3 > nr3x:
        raise DensityError(f'line 2: sizes {sizes} are not positive or exceed the leading '
                           'dimensions')
    fields = _fields(lines, 2, 7, 'ibrav celldm(1:6)')
    ibrav = _integer(fields[0], 'line 3')
    celldm = tuple(_number(f, 'line 3') for f in fields[1:])
    if celldm[0] <= 0:
        raise DensityError(f'line 3: celldm(1) = {celldm[0]!r} is not positive')
    n = 3
    axes = None
    if ibrav == 0:
        axes = np.array([[_number(f, f'line {n + i + 1}')
                          for f in _fields(lines, n + i, 3, f'cell vector a{i + 1}')]
                         for i in range(3)])
        n += 3
    fields = _fields(lines, n, 4, 'gcutm dual ecutwfc plot_num')
    gcutm, dual, ecut = (_number(f, f'line {n + 1}') for f in fields[:3])
    plot_num = _integer(fields[3], f'line {n + 1}')
    n += 1
    species = []
    for nt in range(1, ntyp + 1):
        index, name, valence = _fields(lines, n, 3, f'species {nt}: index, name, valence')
        if _integer(index, f'line {n + 1}') != nt:
            raise DensityError(f'line {n + 1}: species {index} instead of {nt}')
        species.append((name, _number(valence, f'line {n + 1}')))
        n += 1
    positions, types = [], []
    for na in range(1, nat + 1):
        fields = _fields(lines, n, 5, f'atom {na}: index, position, species')
        if _integer(fields[0], f'line {n + 1}') != na:
            raise DensityError(f'line {n + 1}: atom {fields[0]} instead of {na}')
        positions.append([_number(f, f'line {n + 1}') for f in fields[1:4]])
        ityp = _integer(fields[4], f'line {n + 1}')
        if not 1 <= ityp <= ntyp:
            raise DensityError(f'line {n + 1}: species {ityp} of atom {na} is not in 1..{ntyp}')
        types.append(ityp)
        n += 1
    tokens = ' '.join(lines[n:]).split()
    expected = nr1x * nr2x * nr3
    if len(tokens) != expected:
        raise DensityError(f'{len(tokens)} values after the atoms instead of nr1x * nr2x * nr3 = '
                           f'{nr1x} * {nr2x} * {nr3} = {expected}')
    try:
        flat = np.array(tokens, dtype=float)
    except ValueError:
        flat = np.array([_number(t, 'values') for t in tokens])
    bad = ~np.isfinite(flat)
    if bad.any():
        raise DensityError(f'values: {tokens[int(np.flatnonzero(bad)[0])]!r} is not a finite '
                           'number')
    values = flat.reshape(nr3, nr2x, nr1x)[:, :nr2, :nr1].copy()
    return Filplot(title=title, grid=(nr1, nr2, nr3), padded=(nr1x, nr2x, nr3x), ibrav=ibrav,
                   celldm=celldm, axes=axes, gcutm=gcutm, dual=dual, ecut=ecut,
                   plot_num=plot_num, species=tuple(species), positions=np.array(positions),
                   types=tuple(types), values=values)


def read_filplot(path):
    """Read a pp.x filplot (``parse_filplot``); errors name the file."""
    try:
        return parse_filplot(Path(path).read_text(encoding='utf-8'))
    except DensityError as error:
        raise DensityError(f'{path}: {error}') from None


def structure_differences(filplot, structure):
    """Where a filplot disagrees with the structure of its calculation.

    Parameters
    ----------
    filplot : Filplot
    structure : pymatgen.core.structure.Structure
        Structure of the calculation, in angstrom (``structure_out.extxyz``:
        the input of a ``scf``, the final one of a ``relax``).

    Returns
    -------
    list of str
        One sentence per difference: ``ibrav`` other than 0, number of atoms,
        labels (their first two characters), cell or positions beyond the
        digits printed; empty when they agree.
    """
    if filplot.ibrav != 0:
        return [f'ibrav = {filplot.ibrav}: only files with ibrav = 0 give the cell']
    if len(filplot.types) != len(structure):
        return [f'{len(filplot.types)} atoms in the file, {len(structure)} in the structure']
    found = []
    labels = tuple(label[:2] for label in structure.labels)
    if filplot.labels != labels:
        wrong = [i + 1 for i, (a, b) in enumerate(zip(filplot.labels, labels)) if a != b]
        found.append(f'species of atoms {wrong}: {[filplot.labels[i - 1] for i in wrong]} in the '
                     f'file, {[labels[i - 1] for i in wrong]} in the structure')
    alat = filplot.alat
    expected = np.asarray(structure.lattice.matrix)
    tol = _MARGIN * (BOHR_TO_ANGSTROM * _HALF_ALAT * np.abs(filplot.axes) + _HALF_EXTXYZ) \
        + _LIST_DIRECTED * np.abs(expected)
    if np.any(np.abs(filplot.cell - expected) > tol):
        found.append(f'cell {filplot.cell.tolist()} A in the file, {expected.tolist()} A in the '
                     'structure, beyond the digits printed')
    expected = np.asarray(structure.cart_coords)
    tol = _MARGIN * (BOHR_TO_ANGSTROM * (_HALF_TAU * alat + _HALF_ALAT * np.abs(filplot.positions))
                     + _HALF_EXTXYZ)
    far = np.any(np.abs(filplot.cart_positions - expected) > tol, axis=1)
    if far.any():
        found.append(f'positions of atoms {[int(i) + 1 for i in np.flatnonzero(far)]} differ from '
                     'those of the structure beyond the digits printed')
    return found


def to_chgcar(filplot, structure):
    """The charge density of a filplot as a pymatgen ``Chgcar``.

    Parameters
    ----------
    filplot : Filplot
        A charge density (``plot_num = 0``) with ``ibrav = 0``.
    structure : pymatgen.core.structure.Structure
        Structure of the calculation (``structure_differences``).

    Returns
    -------
    pymatgen.io.vasp.outputs.Chgcar
        ``data['total']`` = density (e/A^3) times the volume of the cell of
        ``structure`` (A^3), shape ``(nr1, nr2, nr3)``; structure a copy of
        ``structure``.

    Raises
    ------
    DensityError
        If the file is not a charge density or disagrees with the structure.
    """
    if filplot.plot_num != DENSITY_PLOT_NUM:
        raise DensityError(f'plot_num = {filplot.plot_num}: not the charge density '
                           f'(plot_num = {DENSITY_PLOT_NUM})')
    differences = structure_differences(filplot, structure)
    if differences:
        raise DensityError('the file disagrees with the structure of its calculation: '
                           + '; '.join(differences))
    rho = filplot.values / BOHR_TO_ANGSTROM ** 3
    data = np.ascontiguousarray(rho.transpose(2, 1, 0)) * structure.volume
    return Chgcar(structure.copy(), {'total': data})


def comparison_differences(filplots):
    """Where densities to be combined disagree.

    Parameters
    ----------
    filplots : dict
        ``{name: Filplot}``, e.g. the interface and its two slabs; the first
        is the reference.

    Returns
    -------
    list of str
        One sentence per difference from the first file: ``plot_num`` other
        than the charge density, grid, cell beyond the digits printed,
        ``ecutwfc``, ``dual``, ``gcutm``; empty when they agree.
    """
    items = list(filplots.items())
    found = [f'{name}: plot_num = {f.plot_num}, not the charge density' for name, f in items
             if f.plot_num != DENSITY_PLOT_NUM]
    if not items:
        return found
    ref_name, ref = items[0]
    for name, f in items[1:]:
        if f.grid != ref.grid:
            found.append(f'{name}: grid {f.grid}, {ref.grid} in {ref_name}')
        if (f.ibrav, ref.ibrav) != (0, 0):
            found.append(f'{name}: ibrav = {f.ibrav}, {ref_name}: ibrav = {ref.ibrav}; only '
                         'files with ibrav = 0 give the cell')
        else:
            tol = _MARGIN * 2 * BOHR_TO_ANGSTROM * _HALF_ALAT * np.maximum(np.abs(f.axes),
                                                                           np.abs(ref.axes)) \
                + _LIST_DIRECTED * np.abs(ref.cell)
            if np.any(np.abs(f.cell - ref.cell) > tol):
                found.append(f'{name}: cell {f.cell.tolist()} A, {ref.cell.tolist()} A in '
                             f'{ref_name}')
        for key in ('ecut', 'dual'):
            if getattr(f, key) != getattr(ref, key):
                found.append(f'{name}: {key} = {getattr(f, key)!r}, {getattr(ref, key)!r} in '
                             f'{ref_name}')
        if not math.isclose(f.gcutm, ref.gcutm, rel_tol=_GCUTM):
            found.append(f'{name}: gcutm = {f.gcutm!r}, {ref.gcutm!r} in {ref_name}')
    return found


def check_comparable(filplots, records=None):
    """Check that densities can be combined point by point.

    Parameters
    ----------
    filplots : dict
        ``{name: Filplot}`` (``comparison_differences``).
    records : sequence of triboworkflow.core.store.Record, optional
        Records of their calculations: they must share one energy model
        (``core.store.check_energy_model``).

    Raises
    ------
    DensityError
        Listing every difference of the files and the problem of the energy
        models, if any.
    """
    found = comparison_differences(filplots)
    if records is not None:
        try:
            check_energy_model(records)
        except StoreError as error:
            found.append(str(error))
    if found:
        raise DensityError('densities that cannot be combined: ' + '; '.join(found))
