# Author: Vito Foderà
# Created: 2026-11-09

"""Structure of a finished pw.x calculation, read from its output.

``read_pwo_structure`` gives the atoms of the last run of an output as a
pymatgen ``Structure``, with the data a structure file of this package
carries (``core.structures``): the cell, the cartesian positions in angstrom,
the order of the atoms, the QE labels, the constraints and the species. It is
the reader of the format ``pwo`` of ``core.structures.read_structure``.

What is read
------------
Only the last run of the file is read (a restart appends its output to the
file), and only if that run is over with the SCF converged and, for a
relaxation, the geometry converged: the states are those of ``status.py``.
Otherwise the structure is not read, and the error names the states. The kind
of calculation is told by the summary of the run, which prints ``nstep`` for
a relaxation only.

=====================  =============================================================
Calculation            Positions
=====================  =============================================================
``scf``                those of the header of the run, i.e. the atoms of the input
``relax``              the closed block ``Begin final coordinates`` ...
                       ``End final coordinates``, in angstrom
=====================  =============================================================

The cell is the ``crystal axes`` of the header times ``celldm(1)``, in
angstrom, whatever ``ibrav`` the input gave: pw.x prints the axes for every
``ibrav``. A relaxation that changes the cell (``vc-relax``) is not read.

The data of the structure are:

- ``site.label``: the label of the species, as in ``ATOMIC_POSITIONS``, kept
  whole (``Fe1`` stays ``Fe1``); the element is the symbol the label starts
  with, as for a PWI file, and must agree with the element of the
  pseudopotential that pw.x printed for the species;
- ``if_pos`` (site property): the flags that follow the final positions of a
  relaxation. pw.x prints them for the atoms that have a fixed component only,
  so an atom given ``1 1 1`` in the input reads as an atom without a constraint
  (``None``). The header of a ``scf`` tells nothing of the constraints: the
  property is not set;
- ``structure.properties['qe_species']``: for each species, in the order of
  the output, the ``label``, the ``mass``, the name of the UPF file
  (``pseudopotential``) and ``starting_magnetization``, as for a PWI file.
  ``starting_magnetization`` is the value of the table ``Starting magnetic
  structure``, which pw.x prints for spin-polarized calculations; None
  without the table. It is provenance only and never becomes ``magmom``.

Precision
---------
The structure has the digits that pw.x prints, not those of its input: the
cell is known to half a unit of the sixth decimal of ``alat`` (the axes and
``celldm(1)`` have six decimals), a few 1e-6 angstrom for the cells of a
slab; the positions of a ``scf`` to half a unit of the seventh decimal in
units of ``alat``; those of a relaxation to the digits of the final
coordinates. A PWI file, which is the input itself, is exact. A cell that
must be exact can be given to ``read_structure`` (``cell``), which replaces
it and keeps the cartesian positions.
"""

import re
import warnings
from pathlib import Path

import numpy as np
from ase.io.espresso import label_to_symbol
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from triboworkflow.core.structures.model import IF_POS, QE_SPECIES, StructureError

from .results import _POSITIONS, _SPECIES_TABLE, BOHR_TO_ANGSTROM, pw_values
from .status import _NSTEP, PwOutputError, _last_run, _starts, _summary_value, pw_status, \
    version_warning

_MAGNETIC_HEADER = 'Starting magnetic structure'
_MAGNETIC_TABLE = re.compile(r'atomic species\s+magnetization\s*$')


def _calculation(run):
    """``'relax'`` if the summary of the run prints ``nstep``, else ``'scf'``."""
    return 'relax' if _summary_value(run, _NSTEP, int) is not None else 'scf'


def _check_over(status):
    """Error unless the run is over with the SCF, and the geometry of a relax, converged."""
    ok = status.process == 'ended' and status.scf == 'converged'
    if status.calculation == 'relax':
        ok = ok and status.geometry == 'converged' and status.final_coordinates == 'complete'
    if not ok:
        raise StructureError(
            f'the last run of the output is not over and converged, so its structure is not read: '
            f'process {status.process}, SCF {status.scf}, geometry {status.geometry}, final '
            f'coordinates {status.final_coordinates}, stop reason {status.stop_reason}')


def _species_table(run, species):
    """Masses of the species table of the header, in the order of the species."""
    table = next((i for i, line in enumerate(run) if _SPECIES_TABLE.search(line)), None)
    if table is None:
        raise StructureError('no species table ("atomic species   valence    mass ...") in the '
                             'last run')
    rows = [line.split() for line in run[table + 1:table + 1 + len(species)]]
    masses = []
    for s, words in zip(species, rows):
        try:
            if words[0] != s.label:
                raise ValueError
            masses.append(float(words[2]))
        except (IndexError, ValueError):
            raise StructureError(f'the species table of the last run does not give the mass of '
                                 f'species {s.label!r}: {" ".join(words)!r}') from None
    if len(masses) != len(species):
        raise StructureError(f'the species table has fewer than {len(species)} rows')
    return masses


def _magnetization(run, species):
    """``starting_magnetization`` of each species from the table of the header, or Nones."""
    end = next((i for i, line in enumerate(run) if _POSITIONS in line), len(run))
    start = next((i for i in range(end) if _starts(run[i], _MAGNETIC_HEADER)), None)
    if start is None:
        return [None] * len(species)
    head = next((i for i in range(start + 1, min(start + 3, end))
                 if _starts(run[i], 'atomic species')), None)
    if head is None:
        raise StructureError(f'no "atomic species   magnetization" line after "{_MAGNETIC_HEADER}"')
    if not _MAGNETIC_TABLE.search(run[head]):
        raise StructureError(f'"{run[head].strip()}": only the table of a collinear starting '
                             'magnetization ("atomic species   magnetization") is read')
    values = []
    for s, line in zip(species, run[head + 1:head + 1 + len(species)]):
        words = line.split()
        try:
            if len(words) != 2 or words[0] != s.label:
                raise ValueError
            values.append(float(words[1]))
        except ValueError:
            raise StructureError(f'the starting magnetization of species {s.label!r}: '
                                 f'{line.strip()!r} is not "<label> <value>"') from None
    if len(values) != len(species):
        raise StructureError(f'the table of the starting magnetization has fewer than '
                             f'{len(species)} rows')
    return values


def _elements(species):
    """Element of each label, from the label as for a PWI, checked with the pseudopotential."""
    elements = {}
    for s in species:
        try:
            symbol = label_to_symbol(s.label)
        except (KeyError, ValueError):
            raise StructureError(f'label {s.label!r}: not an element symbol followed by '
                                 'anything else') from None
        if symbol.lower() != s.element.strip().lower():
            raise StructureError(f'label {s.label!r} starts with the element {symbol}, but the '
                                 f'pseudopotential printed for it is for {s.element}')
        elements[s.label] = symbol
    return elements


def pwo_structure(text):
    """Structure of the last run of a pw.x output.

    Parameters
    ----------
    text : str
        Content of the output (see the module notes for what is read).

    Returns
    -------
    pymatgen.core.structure.Structure
        With the labels, ``if_pos`` and ``qe_species`` described in the
        module notes.

    Raises
    ------
    StructureError
        If the text is not a pw.x output, the last run is not over and
        converged, a line or a block of its header is missing or malformed, a
        relaxation changes the cell or prints positions in another unit than
        angstrom, a label does not name an element or contradicts its
        pseudopotential, or the table of the starting magnetization is not
        the collinear one.
    """
    try:
        run, _, _ = _last_run(text)
        calculation = _calculation(run)
        status = pw_status(text, calculation)
        _check_over(status)
        values = pw_values(text, calculation)
    except PwOutputError as error:
        raise StructureError(str(error)) from None
    note = version_warning(values.version)
    if note is not None:
        warnings.warn(note, stacklevel=3)
    scale = values.alat * BOHR_TO_ANGSTROM
    cell = np.asarray(values.axes) * scale
    if calculation == 'scf':
        labels, positions, flags = values.labels, np.asarray(values.positions) * scale, None
    else:
        if values.final_labels != values.labels:
            raise StructureError(f'labels of the final coordinates {list(values.final_labels)} '
                                 f'differ from those of the header {list(values.labels)}')
        labels, positions, flags = values.final_labels, np.asarray(values.final_positions), \
            values.final_if_pos
    elements = _elements(values.species)
    unknown = sorted({label for label in labels if label not in elements})
    if unknown:
        raise StructureError(f'labels of the positions not in the species table: '
                             f'{", ".join(unknown)}')
    masses = _species_table(run, values.species)
    magnetization = _magnetization(run, values.species)
    qe_species = [{'label': s.label, 'mass': mass,
                   'pseudopotential': s.pseudopotential.rsplit('/', 1)[-1],
                   'starting_magnetization': start}
                  for s, mass, start in zip(values.species, masses, magnetization)]
    structure = Structure(Lattice(cell), [elements[label] for label in labels], positions,
                          coords_are_cartesian=True, labels=list(labels),
                          properties={QE_SPECIES: qe_species})
    if flags is not None and any(f is not None for f in flags):
        structure.add_site_property(IF_POS, [None if f is None else list(f) for f in flags])
    return structure


def read_pwo_structure(path):
    """Structure of the last run of a pw.x output file.

    The file is read as UTF-8; bytes that are not UTF-8 are replaced.

    Parameters
    ----------
    path : path-like
        Output file of pw.x.

    Returns
    -------
    pymatgen.core.structure.Structure
        See ``pwo_structure``.

    Raises
    ------
    StructureError
        See ``pwo_structure``.
    """
    return pwo_structure(Path(path).read_text(encoding='utf-8', errors='replace'))
