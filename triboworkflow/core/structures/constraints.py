# Author: Vito Foderà
# Created: 2026-10-23

"""Constraints of the atoms: frames, conversion and the ones applied.

``if_pos`` (cartesian) and ``selective_dynamics`` (lattice vectors) are
described in ``model.py``. pw.x applies ``if_pos`` to the cartesian force
components, so flags along the lattice vectors become ``if_pos`` only when the
directions they leave free span the same space as some cartesian axes, e.g.
``[F, F, T]`` with c along z gives ``0 0 1``. Any other case raises
``StructureError``: no conversion is made for generic cells.

The constraints applied to a calculation are an ``AppliedConstraints``, with
the origin of every atom's ``if_pos``:

- outside the workflows the precedence of the settings holds (override >
  file > ``default_if_pos``, ``resolve_if_pos``);
- inside them the constraints of the ``calc_type`` hold
  (``calc_type_if_pos``), and the ones read from the files are only
  provenance.
"""

from dataclasses import dataclass

import numpy as np

from .model import IF_POS, SELECTIVE_DYNAMICS, StructureError

# A component of a lattice vector at most this large (angstrom) counts as
# zero when deciding whether the vector lies along cartesian axes.
AXIS_TOL = 1e-6

SOURCES = ('override', 'file', 'default', 'calc_type')


@dataclass
class AppliedConstraints:
    """The ``if_pos`` of every atom and where it comes from.

    Attributes
    ----------
    if_pos : list of tuple of int
        pw.x flags of every atom, in the order of the structure.
    source : list of str
        Origin of each entry, one of ``SOURCES``.
    """

    if_pos: list[tuple[int, int, int]]
    source: list[str]

    def as_dict(self):
        """JSON data of the constraints."""
        return {'if_pos': [list(f) for f in self.if_pos], 'source': list(self.source)}


def file_constraints(structure):
    """The constraints carried by a structure.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure

    Returns
    -------
    key : {'if_pos', 'selective_dynamics'} or None
        The site property holding the constraints; None without constraints.
    values : list or None
        Its values, one per atom.

    Raises
    ------
    StructureError
        If the structure has constraints in both frames.
    """
    props = structure.site_properties
    present = [k for k in (IF_POS, SELECTIVE_DYNAMICS) if k in props]
    if len(present) == 2:
        raise StructureError('the structure has both if_pos (cartesian) and selective_dynamics '
                             '(lattice vectors): give the constraints in one frame')
    if not present:
        return None, None
    return present[0], props[present[0]]


def if_pos_from_lattice_flags(matrix, flags, tol=AXIS_TOL):
    """Cartesian ``if_pos`` equal to flags along the lattice vectors.

    Parameters
    ----------
    matrix : array_like, shape (3, 3)
        Lattice vectors as rows, in angstrom.
    flags : sequence of bool
        True for each lattice vector along which the atom is free.
    tol : float, optional
        Components up to this value (angstrom) count as zero.

    Returns
    -------
    tuple of int
        The three pw.x flags.

    Raises
    ------
    StructureError
        If the free lattice vectors do not span the space of a set of
        cartesian axes.
    """
    free = [np.asarray(v, float) for v, f in zip(matrix, flags) if f]
    axes = {j for v in free for j in range(3) if abs(v[j]) > tol}
    # The lattice vectors are independent: the free ones span exactly the
    # space of the axes they touch when they touch as many axes as they are.
    if len(axes) != len(free):
        raise StructureError(f'flags {list(flags)} along the lattice vectors are not cartesian '
                             f'flags in this cell: the free directions do not lie along x, y, z')
    return tuple(int(j in axes) for j in range(3))


def cartesian_if_pos(structure):
    """The constraints of a structure as cartesian ``if_pos``.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure

    Returns
    -------
    list of (tuple of int or None) or None
        The flags of every atom (None where the file gives none), or None if
        the structure has no constraints. Flags along the lattice vectors are
        converted with ``if_pos_from_lattice_flags``.

    Raises
    ------
    StructureError
        If some flags along the lattice vectors cannot be converted.
    """
    key, values = file_constraints(structure)
    if key is None:
        return None
    if key == IF_POS:
        return [None if v is None else tuple(int(x) for x in v) for v in values]
    matrix, result, bad = structure.lattice.matrix, [], []
    for i, flags in enumerate(values):
        try:
            result.append(None if flags is None else if_pos_from_lattice_flags(matrix, flags))
        except StructureError:
            bad.append(i + 1)
    if bad:
        raise StructureError(f'selective_dynamics of atoms {_atoms(bad)} (counted from 1) cannot be '
                             'written as cartesian if_pos in this cell')
    return result


def resolve_if_pos(structure, params):
    """Constraints of a single calculation: override > file > default_if_pos.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
    params : triboworkflow.core.settings.schema.ConstraintsParams
        The ``[structure.constraints]`` table; atoms are counted from 1 in its
        overrides.

    Returns
    -------
    AppliedConstraints
        Values taken as given, with sources ``override``, ``file`` or
        ``default``.

    Raises
    ------
    StructureError
        If an override names an atom beyond the structure, or flags of the
        file along the lattice vectors are not cartesian and not overridden.
    """
    n = len(structure)
    overrides = {}
    for k, override in enumerate(params.overrides):
        for atom in override.atoms:
            if atom > n:
                raise StructureError(f'structure.constraints.overrides[{k}].atoms: atom {atom} '
                                     f'is beyond the {n} atoms of the structure')
            overrides[atom - 1] = tuple(override.if_pos)
    key, values = file_constraints(structure) if params.read_from_file else (None, None)
    if_pos, source, bad = [], [], []
    for i in range(n):
        if i in overrides:
            if_pos.append(overrides[i])
            source.append('override')
        elif key is not None and values[i] is not None:
            if key == IF_POS:
                if_pos.append(tuple(int(x) for x in values[i]))
            else:
                try:
                    if_pos.append(if_pos_from_lattice_flags(structure.lattice.matrix, values[i]))
                except StructureError:
                    bad.append(i + 1)
                    continue
            source.append('file')
        else:
            if_pos.append(tuple(params.default_if_pos))
            source.append('default')
    if bad:
        raise StructureError(f'selective_dynamics of atoms {_atoms(bad)} (counted from 1) cannot be '
                             'written as cartesian if_pos in this cell; give these atoms in '
                             'structure.constraints.overrides')
    return AppliedConstraints(if_pos, source)


def calc_type_if_pos(structure):
    """Constraints set on a structure for its ``calc_type``.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        A structure prepared by the vendored ``VaspInputSet``, which adds
        ``selective_dynamics`` for ``*pos_relax`` (every atom free) and
        ``*z_relax`` (free along c only, H atoms free).

    Returns
    -------
    AppliedConstraints
        With source ``calc_type``; every atom free when the structure has no
        ``selective_dynamics``.

    Raises
    ------
    StructureError
        If the flags are not cartesian in this cell.

    Notes
    -----
    To remove: ``VaspInputSet.get_vasp_settings`` sets the flags in
    ``highput/utils/vasp_tools.py:536-557`` of TribChem.
    """
    n = len(structure)
    flags = structure.site_properties.get(SELECTIVE_DYNAMICS)
    if flags is None:
        return AppliedConstraints([(1, 1, 1)] * n, ['calc_type'] * n)
    matrix = structure.lattice.matrix
    return AppliedConstraints([if_pos_from_lattice_flags(matrix, f) for f in flags],
                              ['calc_type'] * n)


def with_if_pos(structure, if_pos):
    """Copy of a structure whose constraints are the given ``if_pos`` of every atom.

    The constraints of the structure, in either frame, are replaced: e.g. the
    extxyz of a calculation then shows the flags written in its input.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
    if_pos : AppliedConstraints or sequence of tuple of int
        The three pw.x flags of every atom (0 fixed, 1 free).

    Returns
    -------
    pymatgen.core.structure.Structure

    Raises
    ------
    StructureError
        If ``if_pos`` does not give one entry per atom.
    """
    flags = if_pos.if_pos if isinstance(if_pos, AppliedConstraints) else list(if_pos)
    if len(flags) != len(structure):
        raise StructureError(f'if_pos: {len(flags)} entries for {len(structure)} atoms')
    copy = structure.copy()
    for key in (IF_POS, SELECTIVE_DYNAMICS):
        if key in copy.site_properties:
            copy.remove_site_property(key)
    copy.add_site_property(IF_POS, [[int(x) for x in f] for f in flags])
    return copy


def _atoms(indices, shown=10):
    text = ', '.join(map(str, indices[:shown]))
    return text + (f' and {len(indices) - shown} more' if len(indices) > shown else '')
