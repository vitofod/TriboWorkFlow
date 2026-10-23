"""Conversion between pymatgen structures and ASE Atoms, without losses.

pymatgen's ``AseAtomsAdaptor`` does the conversion of cell, positions,
magnetic moments, charges and extra arrays. Two things it does not keep, and
that are done here (pymatgen 2024.6.10, ``pymatgen/io/ase.py``):

- constraints: ``get_atoms`` raises on ``selective_dynamics`` free along only
  some directions (166-174), and ``get_structure`` keeps ``FixAtoms`` only,
  dropping ``FixCartesian`` and ``FixScaled`` with a warning (258-271);
- QE labels, which pymatgen keeps in ``site.label`` and ASE does not have:
  here they travel as the ASE array ``qe_label``.

Constraints map as follows (``model.py``): ``if_pos`` <-> ``FixCartesian``
(an explicit ``1 1 1`` is a ``FixCartesian`` with nothing fixed, so that it
stays apart from an atom without constraints), ``selective_dynamics`` <->
``FixScaled``, ``FixAtoms`` -> fully fixed in the frame of the others. Any
other ASE constraint, both frames in one Atoms, or an atom in two
constraints raises ``StructureError``.

A ``Slab`` becomes plain Atoms: ASE has no place for the slab metadata.
"""

from collections import defaultdict

import numpy as np
from ase.constraints import FixAtoms, FixCartesian, FixScaled
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from .constraints import file_constraints
from .model import CONSTRAINT_KEYS, IF_POS, QE_LABEL, SELECTIVE_DYNAMICS, StructureError


def to_atoms(structure):
    """ASE Atoms of a pymatgen structure, with its constraints and QE labels."""
    if not structure.is_ordered:
        raise StructureError('only ordered structures (no partial occupancies) are supported')
    key, values = file_constraints(structure)
    props = {k: v for k, v in structure.site_properties.items() if k not in CONSTRAINT_KEYS}
    plain = Structure(structure.lattice, structure.species, structure.frac_coords,
                      site_properties=props, properties=dict(structure.properties))
    atoms = AseAtomsAdaptor.get_atoms(plain, msonable=False)
    labels = structure.labels
    if any(label != site.specie.symbol for label, site in zip(labels, structure)):
        atoms.set_array(QE_LABEL, np.array(labels, dtype=str))
    if key is not None:
        atoms.set_constraint(_ase_constraints(key, values))
    return atoms


def to_structure(atoms):
    """pymatgen Structure of ASE Atoms, with constraints and QE labels."""
    work = atoms.copy()
    work.calc = atoms.calc
    key, values = _site_constraints(atoms)
    work.set_constraint()
    labels = work.arrays.pop(QE_LABEL, None)
    structure = AseAtomsAdaptor.get_structure(work)
    if labels is not None:
        for site, label in zip(structure, labels):
            site.label = str(label)
    if key is not None:
        structure.add_site_property(key, values)
    return structure


def _ase_constraints(key, values):
    """ASE constraints of ``if_pos`` or ``selective_dynamics`` values."""
    groups = defaultdict(list)
    for i, value in enumerate(values):
        if value is None:
            continue
        if key == IF_POS:
            if any(int(v) not in (0, 1) for v in value):
                raise StructureError(f'if_pos {list(value)} of atom {i + 1}: ASE constraints '
                                     'represent only 0 (fixed) and 1 (free)')
            fixed = tuple(int(v) == 0 for v in value)
        else:
            fixed = tuple(not bool(v) for v in value)
        groups[fixed].append(i)
    cls = FixCartesian if key == IF_POS else FixScaled
    return [cls(indices, mask=fixed) for fixed, indices in groups.items()]


def _site_constraints(atoms):
    """``(key, values)`` of the constraints of ASE Atoms, as in ``file_constraints``."""
    if not atoms.constraints:
        return None, None
    values, kinds = [None] * len(atoms), set()
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            mask, kind = (True, True, True), None
        elif isinstance(constraint, (FixCartesian, FixScaled)):
            mask = tuple(bool(m) for m in constraint.mask)
            kind = IF_POS if isinstance(constraint, FixCartesian) else SELECTIVE_DYNAMICS
            kinds.add(kind)
        else:
            raise StructureError(f'ASE constraint {type(constraint).__name__} is not supported: '
                                 'only FixAtoms, FixCartesian and FixScaled')
        for i in constraint.get_indices():
            if values[i] is not None:
                raise StructureError(f'atom {i + 1} is in more than one ASE constraint')
            values[i] = mask
    if len(kinds) == 2:
        raise StructureError('FixCartesian and FixScaled in the same Atoms: the constraints are '
                             'in two frames (cartesian and lattice vectors)')
    key = kinds.pop() if kinds else IF_POS
    if key == IF_POS:
        return key, [None if m is None else [int(not f) for f in m] for m in values]
    return key, [None if m is None else [not f for f in m] for m in values]
