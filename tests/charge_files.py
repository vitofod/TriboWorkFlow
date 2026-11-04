# Author: Vito Foderà
# Created: 2026-11-04

"""Synthetic charge-density files for the tests.

Two writers, with numpy only, so that the golden cases can run them in the
historic environment too:

- ``filplot_text``: the file that pp.x writes with ``plot_num = 0``
  (``filplot`` of ``&INPUTPP``), with the Fortran formats of QE 7.2: title
  line of 75 characters, ``8i8`` grid line, ``i6,2x,6f16.8`` lattice line,
  the three cell vectors in units of ``alat`` when ``ibrav = 0``,
  ``3f20.10,i6`` cutoff line, ``i4,3x,a2,3x,f5.2`` species (name cut to two
  characters), ``i4,3x,3f15.9,3x,i2`` atoms in units of ``alat``, then the
  ``nr1x * nr2x * nr3`` values in ``5(1pe17.9)``, x running fastest;
- ``chgcar_text``: a VASP CHGCAR as pymatgen reads it (POSCAR block, blank
  line, grid, values times the cell volume, x running fastest).

``slab_density`` builds the density of a toy interface and of its two slabs
on the same grid: Gaussian layers along z, modulated in the plane, with an
extra charge between the slabs in the interface only.
"""

import math

import numpy as np

TITLE_LENGTH = 75

# Bohr radius in angstrom, CODATA 2018, as in QE 7.2 (Modules/constants.f90).
BOHR_TO_ANGSTROM = 0.529177210903E-10 * 100.0 * 1.0E8


def fortran_e(value, width=17, digits=9):
    """``value`` in the Fortran edit descriptor ``1pe<width>.<digits>``.

    Exponents beyond two digits lose the ``E``, as Fortran writes them
    (``1.234567890-100``).
    """
    mantissa, exponent = f'{value:.{digits}E}'.split('E')
    exp = int(exponent)
    text = f'{mantissa}E{exp:+03d}' if abs(exp) < 100 else f'{mantissa}{exp:+04d}'
    return text.rjust(width)


def list_directed(values):
    """Reals as gfortran writes them with ``write(unit, *)``."""
    return ''.join(f'{v:25.17f}' for v in values)


def filplot_text(*, values, celldm, species, atoms, ibrav=0, at=None, nrx=None,
                 gcutm=100.0, dual=4.0, ecut=40.0, plot_num=0, title='', pad=0.0):
    """Text of a pp.x filplot.

    Parameters
    ----------
    values : ndarray, shape (nr3, nr2, nr1)
        Values on the grid, z index first; for ``plot_num = 0`` the density
        in e/bohr^3.
    celldm : sequence of float
        ``celldm(1:6)`` (fewer values are padded with zeros); ``celldm(1)``
        is ``alat`` in bohr.
    species : list of (str, float)
        Name and valence of each species, in order.
    atoms : list of (sequence of 3 float, int)
        Position in units of ``alat`` and species index (from 1) of each atom.
    ibrav : int
        Bravais-lattice index; with 0 ``at`` is required.
    at : array (3, 3), optional
        Cell vectors (rows) in units of ``alat``.
    nrx : tuple of 3 int, optional
        Leading dimensions ``nr1x, nr2x, nr3x`` (default: the grid itself).
        Only ``nr1x * nr2x * nr3`` values are written, padding along x and y
        set to ``pad``.
    """
    values = np.asarray(values, dtype=float)
    nr3, nr2, nr1 = values.shape
    nr1x, nr2x, nr3x = nrx or (nr1, nr2, nr3)
    celldm = list(celldm) + [0.0] * (6 - len(celldm))
    lines = [title.ljust(TITLE_LENGTH)[:TITLE_LENGTH]]
    lines.append(''.join(f'{n:8d}' for n in (nr1x, nr2x, nr3x, nr1, nr2, nr3,
                                              len(atoms), len(species))))
    lines.append(f'{ibrav:6d}  ' + ''.join(f'{c:16.8f}' for c in celldm))
    if ibrav == 0:
        lines += [list_directed(v) for v in np.asarray(at, dtype=float)]
    lines.append(f'{gcutm:20.10f}{dual:20.10f}{ecut:20.10f}{plot_num:6d}')
    for nt, (name, zv) in enumerate(species, 1):
        lines.append(f'{nt:4d}   {name[:2]:<2s}   {zv:5.2f}')
    for na, (tau, ityp) in enumerate(atoms, 1):
        lines.append(f'{na:4d}   ' + ''.join(f'{t:15.9f}' for t in tau) + f'   {ityp:2d}')
    grid = np.full((nr3, nr2x, nr1x), pad)
    grid[:, :nr2, :nr1] = values
    flat = grid.ravel()
    lines += [''.join(fortran_e(v) for v in flat[i:i + 5]) for i in range(0, flat.size, 5)]
    return '\n'.join(lines) + '\n'


def structure_filplot(structure, grid=(4, 4, 4), electrons=8.0, **kwargs):
    """Filplot of a uniform density of ``electrons`` in the cell of a pymatgen structure.

    ``ibrav = 0`` with ``alat`` the length of the first vector, as pp.x
    writes it for the inputs of this package; the species are the labels of
    the structure in the order they first appear, each with valence 1.
    Other keyword arguments go to ``filplot_text``.
    """
    cell = np.asarray(structure.lattice.matrix)
    alat = np.linalg.norm(cell[0]) / BOHR_TO_ANGSTROM
    labels = list(dict.fromkeys(structure.labels))
    tau = np.asarray(structure.cart_coords) / (alat * BOHR_TO_ANGSTROM)
    volume = abs(np.linalg.det(cell)) / BOHR_TO_ANGSTROM ** 3
    n1, n2, n3 = grid
    args = dict(values=np.full((n3, n2, n1), electrons / volume), celldm=[alat],
                species=[(label, 1.0) for label in labels],
                atoms=[(t, labels.index(label) + 1) for t, label in zip(tau.tolist(),
                                                                        structure.labels)],
                ibrav=0, at=cell / (alat * BOHR_TO_ANGSTROM))
    args.update(kwargs)
    return filplot_text(**args)


def chgcar_text(*, lattice, species, counts, frac, values):
    """Text of a VASP CHGCAR with the total density only.

    ``values`` (shape (nz, ny, nx), z index first) is the density in e/A^3;
    the file holds it times the cell volume, as VASP writes it.
    """
    lattice = np.asarray(lattice, dtype=float)
    values = np.asarray(values, dtype=float)
    volume = abs(np.linalg.det(lattice))
    nz, ny, nx = values.shape
    lines = ['synthetic charge density', '   1.00000000000000']
    lines += [''.join(f'{x:22.16f}' for x in v) for v in lattice]
    lines += ['   ' + '   '.join(species), ''.join(f'{n:6d}' for n in counts), 'Direct']
    lines += [''.join(f'{x:20.16f}' for x in f) for f in frac]
    lines += ['', f'{nx:5d}{ny:5d}{nz:5d}']
    flat = (values * volume).ravel()
    lines += [''.join(f' {v:.11E}' for v in flat[i:i + 5]) for i in range(0, flat.size, 5)]
    return '\n'.join(lines) + '\n'


def slab_density(cell, grid, layers, width=0.6, modulation=0.2, transfer=None):
    """Toy density in e/A^3 on the grid of an orthogonal-in-z cell.

    Parameters
    ----------
    cell : array (3, 3)
        Cell vectors (rows) in angstrom; the third along z.
    grid : tuple of 3 int
        Points ``(n1, n2, n3)`` along the three vectors.
    layers : list of (float, float)
        Height z (A) and electrons of each atomic layer.
    width : float
        Standard deviation (A) of the Gaussian of a layer.
    modulation : float
        Relative in-plane modulation, ``cos(2 pi i/n1) cos(2 pi j/n2)``.
    transfer : (float, float, float), optional
        Height, width and electrons of an extra Gaussian sheet (the charge
        moved to the interface); it is compensated by a uniform background,
        so the total stays the sum of the layers.

    Returns
    -------
    ndarray, shape (n3, n2, n1)
        Density, z index first, integrating to the electrons of the layers.
    """
    cell = np.asarray(cell, dtype=float)
    n1, n2, n3 = grid
    c = cell[2, 2]
    area = abs(np.linalg.det(cell)) / c
    z = np.arange(n3) * c / n3
    profile = np.zeros(n3)

    def sheet(z0, sigma, electrons):
        d = (z - z0 + c / 2) % c - c / 2
        g = np.exp(-0.5 * (d / sigma) ** 2)
        return electrons * g / (g.sum() * c / n3) / area

    for z0, electrons in layers:
        profile += sheet(z0, width, electrons)
    if transfer is not None:
        z0, sigma, electrons = transfer
        profile += sheet(z0, sigma, electrons) - electrons / (area * c)
    i = np.arange(n1)[None, None, :]
    j = np.arange(n2)[None, :, None]
    plane = 1.0 + modulation * np.cos(2 * math.pi * i / n1) * np.cos(2 * math.pi * j / n2)
    return profile[:, None, None] * plane
