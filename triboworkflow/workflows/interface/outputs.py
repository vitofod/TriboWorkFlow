# Author: Vito Foderà
# Created: 2026-11-17

"""Files that the steps of the interface workflow write next to their entries.

The steps ``adhesion``, ``ppes``, ``charge`` and ``shear`` write, in the
folder of the step in the folder of the system, a text file with the
parameters and the numbers of their analysis (``<step>....output``), their
figures (``figures``) and, for ``charge``, the charge displacement in three
dimensions as a Gaussian cube file, which OVITO and VESTA read
(``write_cube``).

Each of these files, and the figure of ``plotpes`` and ``plotpes_scf``,
carries the stamp of the analysis that wrote it
(``analysis_stamp``): a digest of the provenance saved with the entries, in
a line of a text file, in the metadata of a PNG or a PDF, in the comment of
a cube file. A step is done only when each of its files carries the stamp of
the provenance saved (``written_by``): a file of another analysis is never
taken for the current one. The files are written under a temporary name
(``staged``) and renamed once ``system.json`` is saved (``place``): an
interruption before the save leaves the files as they were, one after it
leaves files whose stamp is not the one saved, and the step runs again.
"""

import contextlib
import copy
import datetime
import io
import os
import re
import warnings
from pathlib import Path

import numpy as np

from triboworkflow.core.store import content_digest
from triboworkflow.vendored.highput.utils.manipulate_struct import ManipulateStruct
from triboworkflow.vendored.highput.utils.tasktools import select_struct_func

# Angstrom in a bohr: the unit of length of a cube file.
BOHR = 0.529177210903
# The slab each atom of an interface belongs to, a site property given by match.
SLAB_PROPERTY = 'interface_region'
# The stamp of an analysis in its files, and the bytes read at the start and at the end of a
# file to find it: a text file has it in its first lines, a PNG before its image, a PDF in
# its information at the end.
STAMP = re.compile(rb'triboworkflow analysis sha256:[0-9a-f]{64}')
STAMP_BYTES = 1 << 16


def analysis_stamp(provenance):
    """Stamp of the analysis that saved ``provenance``, written in each of its files.

    Parameters
    ----------
    provenance : triboworkflow.core.store.Provenance or None
        The provenance saved with the entries of the step, with the results
        consumed.

    Returns
    -------
    str or None
        ``triboworkflow analysis sha256:<digest>``: other parameters,
        entries read or results give another stamp. None without a
        provenance.
    """
    if provenance is None:
        return None
    digest = content_digest({'fingerprint': provenance.fingerprint,
                             'results': provenance.results})
    return f'triboworkflow analysis {digest}'


def file_stamp(path):
    """The stamp of an analysis written in a file (``analysis_stamp``), or None."""
    with open(path, 'rb') as file:
        data = file.read(2 * STAMP_BYTES)
        if len(data) == 2 * STAMP_BYTES:
            file.seek(-STAMP_BYTES, os.SEEK_END)
            data = data[:STAMP_BYTES] + b'\n' + file.read()
    found = STAMP.search(data)
    return found.group(0).decode('ascii') if found else None


def written_by(paths, stamp):
    """Whether each file exists and carries ``stamp``, the stamp of the analysis saved."""
    return stamp is not None and all(Path(path).is_file() and file_stamp(path) == stamp
                                     for path in paths)


def partial_path(path):
    """Temporary name of a file, renamed to ``path`` once the entries are saved.

    It ends as ``path`` (``.png``, ``.pdf``), from which matplotlib takes the format.
    """
    path = Path(path)
    return path.with_name(f'.{path.stem}.partial{path.suffix}')


@contextlib.contextmanager
def staged(paths):
    """The temporary names of ``paths``, removed at the end if still there.

    Parameters
    ----------
    paths : iterable of path-like
        Files of a step. Their folders are created.

    Yields
    ------
    dict
        Path -> temporary path, in the same folder. ``place`` renames them,
        right after the entries are saved.
    """
    partials = {Path(path): partial_path(path) for path in paths}
    for path in partials:
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        yield partials
    finally:
        for partial in partials.values():
            partial.unlink(missing_ok=True)


def place(partials):
    """Rename every temporary file of ``staged`` to its path."""
    for path, partial in partials.items():
        os.replace(partial, path)


def header(title, system, step, now=None, stamp=None):
    """First lines of a text file of a step: what it holds, the system, when and by whom.

    Parameters
    ----------
    title : str
        E.g. ``Adhesion of the interface``.
    system : str
    step : str
    now : datetime.datetime or None, optional
        Time written; the current local time by default.
    stamp : str or None, optional
        ``analysis_stamp`` of the analysis, on a line of its own.

    Returns
    -------
    list of str
    """
    now = now or datetime.datetime.now().astimezone()
    lines = [f'# {title} {system}',
             f'# Written {now.isoformat(timespec="seconds")} by the step {step} of triboworkflow']
    if stamp is not None:
        lines.append(f'# {stamp} (the provenance saved in system.json)')
    return lines


def slab_distance(structure):
    """Distance along z between the two slabs of an interface, in angstrom, or None.

    Parameters
    ----------
    structure : dict
        An interface as the document stores it, each atom with the slab it
        belongs to (site property ``interface_region``: ``bot_slab`` or
        ``top_slab``).

    Returns
    -------
    float or None
        The lowest atom of the top slab minus the highest atom of the bottom
        slab, once the interface is brought around z = 0 as the step
        ``charge`` does (``ManipulateStruct.fix_interface``); None when an
        atom carries no slab or the slabs overlap along z, so that no
        distance is told.
    """
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed), warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Interface_init not provided', UserWarning)
        interface = select_struct_func(structure['@class']).from_dict(copy.deepcopy(structure))
        fixed = ManipulateStruct(interface).fix_interface(interface)
    slabs = [site.properties.get(SLAB_PROPERTY) for site in fixed]
    z = np.array([site.coords[2] for site in fixed])
    bottom = z[[slab == 'bot_slab' for slab in slabs]]
    top = z[[slab == 'top_slab' for slab in slabs]]
    if not len(bottom) or not len(top) or len(bottom) + len(top) != len(z):
        return None
    distance = float(top.min() - bottom.max())
    return distance if distance > 0 else None


def write_cube(path, chgcar, comment):
    """Write the density of a ``Chgcar`` as a Gaussian cube file, in e/bohr^3.

    Parameters
    ----------
    path : path-like
    chgcar : pymatgen.io.vasp.outputs.Chgcar
        ``data['total']`` is the density times the volume of the cell, as in
        a CHGCAR; the structure gives the cell and the atoms.
    comment : str
        Second line of the file (with the ``analysis_stamp``, if any).

    Notes
    -----
    The format: two lines of comment; the number of atoms and the origin;
    for each of the three vectors of the cell the points of the grid along it
    and the step between two points; one line per atom (atomic number,
    charge 0, position); then the values, x slowest and z fastest, six per
    line, each column of z on new lines. Lengths in bohr.
    """
    structure = chgcar.structure
    data = np.asarray(chgcar.data['total']) / structure.volume * BOHR ** 3
    lines = [f'Cube file of {structure.composition.reduced_formula} written by triboworkflow',
             comment, f'{len(structure):5d}{0:12.6f}{0:12.6f}{0:12.6f}']
    for points, vector in zip(data.shape, structure.lattice.matrix):
        step = vector / points / BOHR
        lines.append(f'{points:5d}{step[0]:12.6f}{step[1]:12.6f}{step[2]:12.6f}')
    for site in structure:
        x, y, z = site.coords / BOHR
        lines.append(f'{site.specie.Z:5d}{0:12.6f}{x:12.6f}{y:12.6f}{z:12.6f}')
    with open(path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines) + '\n')
        nz = data.shape[2]
        for column in data.reshape(-1, nz):
            for start in range(0, nz, 6):
                file.write(''.join(f'{value:13.5E}' for value in column[start:start + 6]) + '\n')
