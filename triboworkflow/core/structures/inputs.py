"""Structures named in the settings.

Relative paths of structure files are taken from the folder of the settings
file.

- ``load_structure``: the structure of a single calculation (``[structure]``),
  with the cell of ``[structure.cell]`` and the constraints resolved by the
  precedence of ``[structure.constraints]``;
- ``load_workflow_slabs``: the two slabs of ``[interface]``, as the workflow
  uses them (``slab.workflow_slab``); their constraints are only provenance,
  since the workflow applies those of the ``calc_type``, as TribChem.
"""

from dataclasses import dataclass
from pathlib import Path

from pymatgen.core.structure import Structure

from .constraints import AppliedConstraints, resolve_if_pos
from .model import StructureError
from .readers import read_structure
from .slab import read_workflow_slab


@dataclass
class StructureInput:
    """The structure of a single calculation and the constraints applied to it."""

    structure: Structure
    constraints: AppliedConstraints
    file: Path
    format: str
    frame: int


def structure_path(settings, file, where):
    """Path of a structure file of the settings; relative to the settings file."""
    if not file:
        raise StructureError(f'{where}: no structure file given')
    path = Path(file)
    if not path.is_absolute() and settings.source is not None:
        path = Path(settings.source).parent / path
    return path


def load_structure(settings):
    """``StructureInput`` of the ``[structure]`` table of ``settings``."""
    params = settings.structure
    if params is None:
        raise StructureError('the settings have no [structure] table')
    path = structure_path(settings, params.file, 'structure.file')
    cell = params.cell.vectors if params.cell.source == 'config' else None
    structure = read_structure(path, params.format, params.frame, cell=cell)
    return StructureInput(structure, resolve_if_pos(structure, params.constraints),
                          path, params.format, params.frame)


def load_workflow_slabs(settings):
    """``WorkflowSlab`` of the two ``[[interface.slabs]]``: bottom, then top."""
    if settings.interface is None:
        raise StructureError('the settings have no [interface] table')
    slabs = []
    for i, params in enumerate(settings.interface.slabs):
        path = structure_path(settings, params.file, f'interface.slabs[{i}].file')
        slabs.append(read_workflow_slab(path, params.miller))
    return slabs
