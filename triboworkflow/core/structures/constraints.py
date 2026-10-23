"""Constraints of the atoms: frames, conversion and the ones applied.

``if_pos`` (cartesian) and ``selective_dynamics`` (lattice vectors) are
described in ``model.py``. pw.x applies ``if_pos`` to the cartesian force
components, so flags along the lattice vectors become ``if_pos`` only when the
directions they leave free span the same space as some cartesian axes, e.g.
``[F, F, T]`` with c along z gives ``0 0 1``. Any other case raises
``StructureError``: no conversion is made for generic cells.

The constraints applied to a calculation are an ``AppliedConstraints``, with
the origin of every atom's ``if_pos``. Outside the workflows the precedence of
the settings holds (override > file > ``default_if_pos``,
``resolve_if_pos``); inside them the constraints of the ``calc_type`` hold, as
in TribChem, and the ones read from the files are only provenance.
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
    """The ``if_pos`` of every atom and where it comes from (``SOURCES``)."""

    if_pos: list[tuple[int, int, int]]
    source: list[str]

    def as_dict(self):
        return {'if_pos': [list(f) for f in self.if_pos], 'source': list(self.source)}


def file_constraints(structure):
    """``(key, values)`` of the constraints carried by ``structure``.

    ``key`` is ``'if_pos'``, ``'selective_dynamics'`` or None (no constraints).
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
    """``if_pos`` equal to flags along the lattice vectors (True free).

    The lattice vectors left free must span the space of a set of cartesian
    axes; otherwise StructureError.
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
    """The constraints of ``structure`` as cartesian ``if_pos`` (None where not given).

    None if the structure has no constraints. Flags along the lattice vectors
    are converted with ``if_pos_from_lattice_flags``.
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

    ``params`` is the ``[structure.constraints]`` table of the settings.
    The values are taken as given; atoms are counted from 1 in the overrides.
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
    """Constraints set on ``structure`` by TribChem for its ``calc_type``.

    ``VaspInputSet.get_vasp_settings`` adds ``selective_dynamics`` for
    ``*pos_relax`` and ``*z_relax`` (``highput/utils/vasp_tools.py:536-557``,
    H atoms free in ``z_relax``); without it every atom is free.
    """
    n = len(structure)
    flags = structure.site_properties.get(SELECTIVE_DYNAMICS)
    if flags is None:
        return AppliedConstraints([(1, 1, 1)] * n, ['calc_type'] * n)
    matrix = structure.lattice.matrix
    return AppliedConstraints([if_pos_from_lattice_flags(matrix, f) for f in flags],
                              ['calc_type'] * n)


def _atoms(indices, shown=10):
    text = ', '.join(map(str, indices[:shown]))
    return text + (f' and {len(indices) - shown} more' if len(indices) > shown else '')
