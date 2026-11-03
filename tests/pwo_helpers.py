# Author: Vito Foderà
# Created: 2026-11-03

"""Synthetic pw.x outputs of a structure, for the tests of the reading of the results.

``pw_output`` writes the lines pw.x prints for a ``scf`` or a ``relax`` of a
structure, with the formats of the real outputs: the summary (``nat``,
``ntyp``, ``celldm(1)``, ``nstep`` for a relax), the cell in units of
``alat``, the pseudopotentials with their checksums, the species table, the
positions in units of ``alat``, the SCF cycles with energy, forces and
stress, the end of the optimization and the final coordinates. Only those
lines and a few around them are written.
"""

import numpy as np
from pymatgen.core import Lattice, Structure

from triboworkflow.core.structures import extxyz_text
from triboworkflow.core.structures.readers import read_extxyz_text
from triboworkflow.engines.qe.results import BOHR_TO_ANGSTROM

MD5_FE = '327243ea9db589a56bc50e6ec7945b51'
MD5_H = '3f4114867bd07edc7f1424d066fc1888'
# (label, element, UPF file, checksum) of each species, in the order of ATOMIC_SPECIES.
SPECIES = (('Fe1', 'Fe', 'Fe.pbe-sp-van_ak.UPF', MD5_FE),
           ('H', 'H', 'H.pbe-rrkjus_psl.1.0.0.UPF', MD5_H))
PSEUDO_DIR = '/hpc/pseudo'
JOB_DONE = ['', '=' + '-' * 78 + '=', '   JOB DONE.', '=' + '-' * 78 + '=']


def structure(if_pos=None, magmom=None):
    """Two atoms, Fe1 and H, in a cell that is not orthogonal, with many decimals.

    The decimals beyond those of an extxyz file (eight for the positions)
    make the input written by pw.x differ from ``structure_in.extxyz`` as it
    does in a real calculation.
    """
    s = Structure(Lattice([[4.0123456789012, 0.0, 0.0], [1.0123456789012, 5.0123456789012, 0.0],
                           [0.0, 0.2123456789012, 12.0123456789012]]), ['Fe', 'H'],
                  [[0.5123456789012, 0.6234567890123, 3.0345678901234],
                   [1.2512345678901, 2.1023456789012, 4.7034567890123]],
                  coords_are_cartesian=True, labels=['Fe1', 'H'])
    if if_pos is not None:
        s.add_site_property('if_pos', if_pos)
    if magmom is not None:
        s.add_site_property('magmom', magmom)
    return s


def structure_in(s):
    """``s`` as read back from its extxyz file, i.e. ``structure_in.extxyz``."""
    return read_extxyz_text(extxyz_text(s))


def relaxed(s, shift=(0.01, -0.02, 0.03)):
    """Cartesian positions of ``s`` moved by ``shift``, in angstrom."""
    return [tuple(p) for p in np.asarray(s.cart_coords) + np.asarray(shift)]


def header(s, calculation='scf', version='7.2', species=SPECIES, md5_lines=None):
    """Header of a run of ``s``, from the version line to the positions."""
    matrix = np.asarray(s.lattice.matrix) / BOHR_TO_ANGSTROM
    alat = float(np.linalg.norm(matrix[0]))
    lines = ['', f'     Program PWSCF v.{version} starts on  1Jan2026 at 10: 0: 0 ', '',
             '     Reading input from run.pwi', '',
             '     bravais-lattice index     =            0',
             f'     lattice parameter (alat)  = {alat:12.4f}  a.u.',
             f'     number of atoms/cell      = {len(s):12d}',
             f'     number of atomic types    = {len(species):12d}',
             '     mixing beta               =       0.3000',
             '     Exchange-correlation= PBE']
    if calculation == 'relax':
        lines += ['     energy convergence thresh.=      1.0E-04',
                  '     force convergence thresh. =      1.0E-03',
                  '     nstep                     =           50']
    lines += ['', f'     celldm(1)={alat:11.6f}  celldm(2)={0:11.6f}  celldm(3)={0:11.6f}',
              f'     celldm(4)={0:11.6f}  celldm(5)={0:11.6f}  celldm(6)={0:11.6f}', '',
              '     crystal axes: (cart. coord. in units of alat)']
    for i, (x, y, z) in enumerate(matrix / alat, 1):
        lines.append(f'               a({i}) = ({x:11.6f}{y:11.6f}{z:11.6f} )  ')
    lines.append('')
    for i, (label, element, upf, md5) in enumerate(species, 1):
        md5_line = f'     MD5 check sum: {md5}' if md5_lines is None else md5_lines[i - 1]
        lines += ['', f'     PseudoPot. #{i:2d} for {element:2s} read from file:',
                  f'     {PSEUDO_DIR}/{upf}', md5_line,
                  '     Pseudo is Ultrasoft, Zval =  1.0', '']
    lines += ['     atomic species   valence    mass     pseudopotential']
    for label, element, _, _ in species:
        lines.append(f'        {label:6s}         1.00     1.00800     {element:2s}( 1.00)')
    lines += ['', '   Cartesian axes', '',
              '     site n.     atom                  positions (alat units)']
    for n, (site, (x, y, z)) in enumerate(zip(s, np.asarray(s.cart_coords) /
                                             BOHR_TO_ANGSTROM / alat), 1):
        lines.append(f'{n:10d}           {site.label:3s}tau({n:4d}) = ({x:12.7f}{y:12.7f}'
                     f'{z:12.7f}  )')
    return lines + ['']


def scf_cycle(nat, energy, forces=None, stress=None, verbosity_high=False, dispersion=False):
    """One converged SCF cycle with its energy, and forces and stress when given (Ry units)."""
    lines = ['     Self-consistent Calculation', '',
             '     iteration #  1     ecut=    30.00 Ry     beta= 0.30',
             f'     total energy              = {energy + 0.01:17.8f} Ry', '',
             '     End of self-consistent calculation', '',
             f'!    total energy              = {energy:17.8f} Ry',
             '     smearing contrib. (-TS)   =      -0.00270760 Ry',
             f'     internal energy E=F+TS    = {energy + 0.0027076:17.8f} Ry', '',
             '     The total energy is F=E-TS. E is the sum of the following terms:', '',
             '     convergence has been achieved in   7 iterations', '']
    if forces is not None:
        lines += ['     Forces acting on atoms (cartesian axes, Ry/au):', '']
        lines += [f'     atom {n:4d} type {1 if n == 1 else 2:2d}   force = '
                  f'{fx:14.8f}{fy:14.8f}{fz:14.8f}' for n, (fx, fy, fz) in enumerate(forces, 1)]
        if verbosity_high:
            lines.append('     The non-local contrib.  to forces')
            lines += [f'     atom {n:4d} type  1   force = {9.0:14.8f}{9.0:14.8f}{9.0:14.8f}'
                      for n in range(1, nat + 1)]
        if dispersion:
            lines += ['', '     Dispersion contribution to forces:']
            lines += [f'     atom {n:4d} type  1   force = {8.0:14.8f}{8.0:14.8f}{8.0:14.8f}'
                      for n in range(1, nat + 1)]
        lines += ['', '     Total force =     0.000100     Total SCF correction =     0.000000', '']
    if stress is not None:
        lines += ['     Computing stress (Cartesian axis) and pressure', '',
                  '          total   stress  (Ry/bohr**3)                   (kbar)     P='
                  '       -0.36']
        lines += [f'{a:13.8f}{b:13.8f}{c:13.8f}{a * 147105.08:13.2f}{b * 147105.08:13.2f}'
                  f'{c * 147105.08:13.2f}' for a, b, c in stress]
        lines.append('')
    return lines


def final_coordinates(labels, positions, if_pos=None, closed=True):
    lines = ['Begin final coordinates', '', 'ATOMIC_POSITIONS (angstrom)']
    for i, (label, (x, y, z)) in enumerate(zip(labels, positions)):
        flags = if_pos[i] if if_pos is not None else None
        mask = '' if flags is None or 0 not in flags else ''.join(f'{f:4d}' for f in flags)
        lines.append(f'{label:4s}{x:20.10f}{y:20.10f}{z:20.10f}{mask}')
    return lines + (['End final coordinates', ''] if closed else [])


def pw_output(s, calculation='scf', version='7.2', energy=-31.1, forces=None, stress=None,
              species=SPECIES, md5_lines=None, steps=2, final_energy='same', final_positions=None,
              final_if_pos=None, closed=True, verbosity_high=False, dispersion=False,
              job_done=True):
    """Text of a converged pw.x run of ``s``.

    ``forces`` (Ry/bohr, one triple per atom) and ``stress`` (Ry/bohr^3, three
    rows) are printed in the last SCF cycle when given. A ``relax`` makes
    ``steps`` SCF cycles, the last one with ``energy``, then converges:
    ``Final energy`` is ``energy`` with ten decimals (``final_energy='same'``),
    another value, or absent (None); the final coordinates are
    ``final_positions`` (``relaxed(s)`` by default) with the flags
    ``final_if_pos``, left open when ``closed`` is False.
    """
    nat = len(s)
    lines = header(s, calculation, version, species, md5_lines)
    if calculation == 'scf':
        lines += scf_cycle(nat, energy, forces, stress, verbosity_high, dispersion)
    else:
        for step in range(steps - 1):
            lines += scf_cycle(nat, energy + 0.1 * (steps - 1 - step), forces, stress)
            lines += [f'     number of bfgs steps    = {step:3d}', '',
                      'ATOMIC_POSITIONS (angstrom)']
            lines += [f'{site.label:4s}{x:20.10f}{y:20.10f}{z:20.10f}'
                      for site, (x, y, z) in zip(s, s.cart_coords)]
            lines.append('')
        lines += scf_cycle(nat, energy, forces, stress, verbosity_high, dispersion)
        lines += [f'     bfgs converged in {steps:3d} scf cycles and {steps - 1:3d} bfgs steps',
                  '     (criteria: energy <  1.0E-04 Ry, force <  1.0E-03 Ry/Bohr)', '',
                  '     End of BFGS Geometry Optimization', '']
        if final_energy == 'same':
            final_energy = energy
        if final_energy is not None:
            lines += [f'     Final energy             = {final_energy:18.10f} Ry', '']
        lines += final_coordinates([site.label for site in s],
                                   relaxed(s) if final_positions is None else final_positions,
                                   final_if_pos, closed)
    return '\n'.join(lines + (JOB_DONE if job_done else [])) + '\n'
