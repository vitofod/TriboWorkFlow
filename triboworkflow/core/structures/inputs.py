# Author: Vito Foderà
# Created: 2026-10-23

"""Structures named in the settings.

Relative paths of structure files are taken from the folder of the settings
file.

- ``load_structure``: the structure of a single calculation (``[structure]``),
  with the cell of ``[structure.cell]`` and the constraints resolved by the
  precedence of ``[structure.constraints]``;
- ``load_workflow_slabs``: the two slabs of ``[interface]``, as the workflow
  uses them (``slab.workflow_slab``). Their constraints are only provenance:
  the workflow applies those of each ``calc_type``
  (``constraints.calc_type_if_pos``).
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
    """The structure of a single calculation and the constraints applied to it.

    Attributes
    ----------
    structure : pymatgen.core.structure.Structure
        As read, with the constraints of the file as provenance.
    constraints : AppliedConstraints
        The ``if_pos`` of every atom and its source.
    file : pathlib.Path
    format : str
    frame : int
    """

    structure: Structure
    constraints: AppliedConstraints
    file: Path
    format: str
    frame: int


def structure_path(settings, file, where):
    """Path of a structure file named in the settings.

    Parameters
    ----------
    settings : triboworkflow.core.settings.Settings
    file : str
        The path as written in the settings.
    where : str
        TOML path of the field, for the error message.

    Returns
    -------
    pathlib.Path
        Relative paths are taken from the folder of ``settings.source``.

    Raises
    ------
    StructureError
        If ``file`` is empty (a placeholder).
    """
    if not file:
        raise StructureError(f'{where}: no structure file given')
    path = Path(file)
    if not path.is_absolute() and settings.source is not None:
        path = Path(settings.source).parent / path
    return path


def load_structure(settings):
    """The structure of a single calculation.

    Parameters
    ----------
    settings : triboworkflow.core.settings.Settings
        With a ``[structure]`` table.

    Returns
    -------
    StructureInput

    Raises
    ------
    StructureError
        If the settings have no ``[structure]``, or the file cannot be read or
        its constraints resolved.
    """
    params = settings.structure
    if params is None:
        raise StructureError('the settings have no [structure] table')
    path = structure_path(settings, params.file, 'structure.file')
    cell = params.cell.vectors if params.cell.source == 'config' else None
    structure = read_structure(path, params.format, params.frame, cell=cell)
    return StructureInput(structure, resolve_if_pos(structure, params.constraints),
                          path, params.format, params.frame)


def load_workflow_slabs(settings):
    """The two slabs of the interface workflow.

    Parameters
    ----------
    settings : triboworkflow.core.settings.Settings
        With an ``[interface]`` table.

    Returns
    -------
    list of WorkflowSlab
        Bottom slab, then top slab.

    Raises
    ------
    StructureError
        If the settings have no ``[interface]``, or a slab cannot be read or
        used by the workflow.
    """
    if settings.interface is None:
        raise StructureError('the settings have no [interface] table')
    slabs = []
    for i, params in enumerate(settings.interface.slabs):
        path = structure_path(settings, params.file, f'interface.slabs[{i}].file')
        slabs.append(read_workflow_slab(path, params.miller))
    return slabs
