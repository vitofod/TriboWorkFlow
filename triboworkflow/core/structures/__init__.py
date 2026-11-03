# Author: Vito Foderà
# Created: 2026-10-22

"""Reading, conversion and writing of atomic structures.

The data carried by a structure (elements, QE labels, constraints, moments)
are described in ``model.py``; the formats and their limits in
``docs/structures.md``.
"""

from .constraints import (AppliedConstraints, calc_type_if_pos, cartesian_if_pos,
                          if_pos_from_lattice_flags, resolve_if_pos, with_if_pos)
from .convert import to_atoms, to_structure
from .inputs import StructureInput, load_structure, load_workflow_slabs
from .model import StructureError
from .readers import FORMATS, detect_format, read_extxyz_text, read_json, read_structure
from .slab import WorkflowSlab, check_slab_cell, custom_slab, read_workflow_slab, workflow_slab
from .writers import INPUT_FIELDS, extxyz_text, structure_differences, write_structure

__all__ = ['FORMATS', 'INPUT_FIELDS', 'AppliedConstraints', 'StructureError', 'StructureInput',
           'WorkflowSlab', 'calc_type_if_pos', 'cartesian_if_pos', 'check_slab_cell',
           'custom_slab', 'detect_format', 'extxyz_text', 'if_pos_from_lattice_flags',
           'load_structure', 'load_workflow_slabs', 'read_extxyz_text', 'read_json',
           'read_structure', 'read_workflow_slab', 'resolve_if_pos', 'structure_differences',
           'to_atoms', 'to_structure', 'with_if_pos', 'workflow_slab', 'write_structure']
