# Author: Vito Foderà
# Created: 2026-10-23

"""Writing of xyz and extxyz files.

- xyz: plain XYZ, elements and cartesian positions in angstrom only. The
  format has no place for cell, QE labels, constraints or moments: they are
  not written.
- extxyz: cell, ``pbc="T T T"``, ``initial_magmoms`` (``magmom``),
  ``qe_label`` when a label differs from its element, the other per-atom
  data as extra columns, and the constraints as ``move_mask``, which is
  cartesian. Flags along the lattice vectors are converted when possible
  (``constraints.cartesian_if_pos``), otherwise ``StructureError``. When some
  atoms had no constraints in the source, the column ``constraint_given``
  marks the others, so that ``read_structure`` gives them back as ``None``.
  The structure properties (e.g. ``qe_species`` of a PWI) are not written.

``extxyz_text`` gives the same text without writing a file;
``structure_differences`` compares two extxyz texts on what a pw.x input
takes from a structure (``INPUT_FIELDS``) and on the other per-atom data.

Notes
-----
The files are written by ASE. Every atom gets its own ``FixCartesian``:
ASE 3.26 writes the atoms left out of a list of ``FixCartesian`` as fixed,
and keeps only the first kind of a mixed list (``ase/io/extxyz.py:894-910``).
"""

import io
from pathlib import Path

import ase.io
import numpy as np
from ase.constraints import FixCartesian

from .constraints import cartesian_if_pos
from .convert import to_atoms
from .model import CONSTRAINT_GIVEN, IF_POS, StructureError
from .readers import read_extxyz_text

# What a structure gives a pw.x input: the differences that make an extxyz
# no longer the picture of its input (``structure_differences``).
INPUT_FIELDS = ('cell', 'positions', 'labels', 'constraints')


def write_structure(structure, path, format='extxyz'):
    """Write a structure to a file.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
    path : str or pathlib.Path
    format : {'extxyz', 'xyz'}, optional

    Raises
    ------
    StructureError
        For another format, or data that the format cannot hold.
    """
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
    """The ASE Atoms that ``write_structure`` writes as extxyz.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure

    Returns
    -------
    ase.Atoms
        With one ``FixCartesian`` per atom when the structure has constraints,
        and the array ``constraint_given`` when some atoms have none.

    Raises
    ------
    StructureError
        For per-atom data with None or ragged values, or constraints that are
        not cartesian in this cell.
    """
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


def extxyz_text(structure):
    """Text of the extxyz file that ``write_structure`` writes for a structure.

    Raises
    ------
    StructureError
        As ``extxyz_atoms``.
    """
    buffer = io.StringIO()
    ase.io.write(buffer, extxyz_atoms(structure), format='extxyz')
    return buffer.getvalue()


def structure_differences(kept, new):
    """What differs between two extxyz texts, as read back by ``read_structure``.

    Parameters
    ----------
    kept, new : str
        Texts of two extxyz files, e.g. the one of a calculation prepared
        earlier and the one prepared now.

    Returns
    -------
    fields : list of str
        The ``INPUT_FIELDS`` that differ: ``cell``, ``positions`` (cartesian,
        or the number of atoms), ``labels`` (QE label or element of an atom),
        ``constraints`` (``if_pos``).
    other : list of str
        The other per-atom data that differ (e.g. ``magmom``), by name; empty
        when only the text differs, e.g. its comment line.

    Raises
    ------
    StructureError
        If a text cannot be read as extxyz.
    """
    a, b = read_extxyz_text(kept), read_extxyz_text(new)
    fields = []
    if not np.array_equal(a.lattice.matrix, b.lattice.matrix):
        fields.append('cell')
    if len(a) != len(b):
        return fields + ['positions', 'labels', 'constraints'], []
    if not np.array_equal(a.cart_coords, b.cart_coords):
        fields.append('positions')
    if [(s.label, s.specie.symbol) for s in a] != [(s.label, s.specie.symbol) for s in b]:
        fields.append('labels')
    props_a, props_b = a.site_properties, b.site_properties
    if not _same_values(props_a.get(IF_POS), props_b.get(IF_POS)):
        fields.append('constraints')
    other = sorted(name for name in set(props_a) | set(props_b)
                   if name != IF_POS and not _same_values(props_a.get(name), props_b.get(name)))
    return fields, other


def _same_values(x, y):
    if x is None or y is None:
        return x is None and y is None
    return len(x) == len(y) and all(np.array_equal(u, v) for u, v in zip(x, y))
