"""Writing of xyz and extxyz files.

- xyz: plain XYZ, elements and cartesian positions in angstrom only. The
  format has no place for cell, QE labels, constraints or moments: they are
  not written.
- extxyz: ASE's writer, with cell, ``pbc="T T T"``, ``initial_magmoms``
  (``magmom``), ``qe_label`` when a label differs from its element, the other
  per-atom data as extra columns, and the constraints as ``move_mask``, which
  is cartesian. Flags along the lattice vectors are converted when possible
  (``constraints.cartesian_if_pos``), otherwise StructureError. Every atom
  gets its own ``FixCartesian``: ASE writes the atoms left out of a list of
  ``FixCartesian`` as fixed, and keeps only the first kind of a mixed list
  (``ase/io/extxyz.py:894-910``). When some atoms had no constraints in
  the source, the column ``constraint_given`` marks the others, so that
  ``read_structure`` gives them back as ``None``. The structure properties
  (e.g. ``qe_species`` of a PWI) are not written.
"""

from pathlib import Path

import ase.io
import numpy as np
from ase.constraints import FixCartesian

from .constraints import cartesian_if_pos
from .convert import to_atoms
from .model import CONSTRAINT_GIVEN, StructureError


def write_structure(structure, path, format='extxyz'):
    """Write ``structure`` to ``path`` as ``xyz`` or ``extxyz``."""
    path = Path(path)
    if format == 'xyz':
        atoms = to_atoms(structure)
        atoms.set_constraint()
        ase.io.write(path, atoms, format='xyz')
    elif format == 'extxyz':
        ase.io.write(path, extxyz_atoms(structure), format='extxyz')
    else:
        raise StructureError(f'cannot write format {format!r}: only xyz and extxyz')


def extxyz_atoms(structure):
    """The ASE Atoms that ``write_structure`` writes as extxyz."""
    if_pos = cartesian_if_pos(structure)
    atoms = to_atoms(structure)
    atoms.set_constraint()
    atoms.info = {}
    for name, values in atoms.arrays.items():
        if values.dtype == object:
            raise StructureError(f'site property {name!r} has values that extxyz cannot hold '
                                 '(e.g. None or lists of different length)')
    if if_pos is not None:
        constraints = []
        for i, flags in enumerate(if_pos):
            flags = (1, 1, 1) if flags is None else flags
            constraints.append(FixCartesian([i], mask=[f == 0 for f in flags]))
        atoms.set_constraint(constraints)
        if any(flags is None for flags in if_pos):
            atoms.set_array(CONSTRAINT_GIVEN, np.array([flags is not None for flags in if_pos]))
    return atoms
