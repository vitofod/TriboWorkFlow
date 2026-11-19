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
  (``constraints.calc_type_if_pos``);
- ``resolve_species``: the mass and the pseudopotential of the species written
  ``"auto"`` in the settings, read from the structure files.
"""

from dataclasses import dataclass
from pathlib import Path

from pymatgen.core.structure import Structure

from .constraints import AppliedConstraints, resolve_if_pos
from .model import QE_SPECIES, StructureError
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


# Species of the structure files read by this process, by file (path, time of the last change,
# size), format and frame: the settings are loaded again by every step at every pass of the
# orchestrator, and an output of pw.x may be large.
_SPECIES_READ = {}


def _file_species(path, fmt=None, frame=None):
    """``properties[QE_SPECIES]`` of a structure file, read once per process while it is unchanged."""
    stat = Path(path).stat()
    key = (str(Path(path).resolve()), stat.st_mtime_ns, stat.st_size, fmt, frame)
    if key not in _SPECIES_READ:
        args = () if fmt is None and frame is None else (fmt, frame)
        _SPECIES_READ[key] = read_structure(path, *args).properties.get(QE_SPECIES)
    return _SPECIES_READ[key]


def resolve_species(settings):
    """Replace the ``"auto"`` mass and pseudopotential of the species by those of the files.

    For each species of ``[[atomic_species]]`` with ``mass`` or
    ``pseudopotential`` equal to ``"auto"``, the value is read from the
    ``ATOMIC_SPECIES`` of the structure files of the settings (the slabs of
    ``[interface]``, or ``[structure]``), which must be inputs or outputs of
    pw.x, for the species of the same label. The order and the labels of the
    species, ``starting_magnetization`` and the other values stay as written.
    Nothing is read when no species asks for it. A file is read once per
    process while it does not change (its path, time of the last change and
    size).

    Parameters
    ----------
    settings : triboworkflow.core.settings.Settings
        Changed in place.

    Returns
    -------
    list of str
        One sentence for each value read, with the file it comes from;
        empty when nothing was read.

    Raises
    ------
    StructureError
        If a file is not an input or an output of pw.x, a label is in no
        file, or the files give different values for the same label.
    """
    from triboworkflow.core.settings.schema import AUTO

    wanted = [(i, s) for i, s in enumerate(settings.atomic_species)
              if AUTO in (s.mass, s.pseudopotential)]
    if not wanted:
        return []
    if settings.interface is not None:
        files = [structure_path(settings, slab.file, f'interface.slabs[{i}].file')
                 for i, slab in enumerate(settings.interface.slabs)]
        species = [_file_species(path) for path in files]
    else:
        params = settings.structure
        files = [structure_path(settings, params.file, 'structure.file')]
        species = [_file_species(files[0], params.format, params.frame)]
    found = {}
    for path, entries in zip(files, species):
        if entries is None:
            raise StructureError(f'{path}: "{AUTO}" in atomic_species needs structure files that '
                                 'are inputs or outputs of pw.x, which give the mass and the '
                                 'pseudopotential of each species; this one is not')
        for entry in entries:
            found.setdefault(entry['label'], []).append((path, entry))
    sentences = []
    for i, species in wanted:
        entries = found.get(species.label)
        if not entries:
            raise StructureError(f'atomic_species[{i}]: no species {species.label!r} in '
                                 f'{", ".join(str(p) for p in files)}')
        for name in ('mass', 'pseudopotential'):
            if getattr(species, name) != AUTO:
                continue
            values = {entry[name] for _, entry in entries}
            if len(values) > 1:
                raise StructureError(
                    f'atomic_species[{i}].{name}: the files give different values for the species '
                    f'{species.label!r}: ' + ', '.join(f'{entry[name]} in {path}'
                                                       for path, entry in entries))
            (value,) = values
            setattr(species, name, value)
            sentences.append(f'atomic_species[{i}].{name} = {value} for {species.label}, read '
                             f'from {entries[0][0]}')
    return sentences


def load_workflow_slabs(settings):
    """The two slabs of the interface workflow.

    Parameters
    ----------
    settings : triboworkflow.core.settings.Settings
        With an ``[interface]`` table.

    Returns
    -------
    list of WorkflowSlab
        In the order of ``[[interface.slabs]]``; the matching decides which
        one is on top.

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
