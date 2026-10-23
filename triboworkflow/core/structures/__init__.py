"""Reading and writing of atomic structures (data conventions: ``model.py``)."""

from .constraints import (AppliedConstraints, calc_type_if_pos, cartesian_if_pos,
                          if_pos_from_lattice_flags, resolve_if_pos)
from .convert import to_atoms, to_structure
from .inputs import StructureInput, load_structure, load_workflow_slabs
from .model import StructureError
from .readers import FORMATS, detect_format, read_json, read_structure
from .slab import WorkflowSlab, check_slab_cell, custom_slab, read_workflow_slab, workflow_slab
from .writers import write_structure

__all__ = ['FORMATS', 'AppliedConstraints', 'StructureError', 'StructureInput', 'WorkflowSlab',
           'calc_type_if_pos', 'cartesian_if_pos', 'check_slab_cell', 'custom_slab',
           'detect_format', 'if_pos_from_lattice_flags', 'load_structure', 'load_workflow_slabs',
           'read_json', 'read_structure', 'read_workflow_slab', 'resolve_if_pos', 'to_atoms',
           'to_structure', 'workflow_slab', 'write_structure']
