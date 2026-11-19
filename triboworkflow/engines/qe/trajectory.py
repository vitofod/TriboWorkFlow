# Author: Vito Foderà
# Created: 2026-11-19

"""Frames of a pw.x relaxation, read from its output.

``trajectory_text`` gives the extxyz text of the configurations of a
``relax`` that have results, one frame each, so that the whole relaxation can
be looked at (e.g. ``ase gui trajectory.extxyz``). The parsing writes it in
``trajectory.extxyz`` at the end of each attempt, converged or not
(``execution/parsing.py``).

Frames
------
One frame per configuration of the output followed by a result, in their
order: the first at the positions of the header, the next ones at the
positions printed after each ionic step (``ATOMIC_POSITIONS``). The final
coordinates that pw.x prints again at the end of a relaxation are the
positions of its last SCF cycle, not a frame of their own. An output cut
short (a job stopped at its walltime, a crash) gives the frames printed
until then; a configuration whose SCF cycle was cut gives a frame without
energy and forces when pw.x already printed the moments of the sites, none
otherwise.

Each frame has the cell, the species, the labels and the other per-atom data
of the input structure (``structure_in.extxyz``: ``move_mask``,
``interface_region``, ...), as the final structure of the record, and the
positions of the output. Its results are:

- ``energy``: the total energy of the SCF cycle (the ``!`` line), in eV;
- ``forces``: the total force on every atom, in eV/angstrom;
- ``magmoms``: the magnetic moment of every site, in Bohr magnetons, as pw.x
  prints it, in a spin-polarized calculation.

Energy and forces are converted with the constants of QE
(``results.RY_TO_EV``, ``results.BOHR_TO_ANGSTROM``), as the results of the
record: the last frame of a relaxation that converged has the energy and the
forces of the record. The stress is not written.

Notes
-----
As Xsorb reads the frames of its calculations (``xsorb/calculations/results.py``,
``get_atoms_traj_from_calc``, Enrico Pedretti, CC BY 4.0): ASE's
``read_espresso_out`` with ``index=':'``. ASE converts with the constants of
CODATA 2006 (``ase.io.espresso.units``), which differ from those of QE by about
1e-7 in the energy (0.015 eV on 165 000 eV): energies and forces are taken
back to Ry and Ry/bohr and converted with those of QE. The positions of the
header are read in units of ``celldm(1)``, with the seven decimals pw.x
prints; those of the ionic steps in angstrom, with ten.
"""

import io

import ase.io
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.espresso import units as ase_units

from triboworkflow.core.structures.writers import extxyz_atoms

from .results import BOHR_TO_ANGSTROM, RY_TO_EV, _final_structure

# From the units of ASE's reader to those of QE.
_ENERGY = RY_TO_EV / ase_units['Ry']
_FORCE = (RY_TO_EV / BOHR_TO_ANGSTROM) / (ase_units['Ry'] / ase_units['Bohr'])


def trajectory_frames(text, structure):
    """The frames of a relaxation, as ASE Atoms with their results.

    Parameters
    ----------
    text : str
        Content of the output of pw.x.
    structure : pymatgen.core.structure.Structure
        Structure of the input (``structure_in.extxyz``).

    Returns
    -------
    list of ase.Atoms
        One per configuration with results (module notes), with a
        ``SinglePointCalculator`` holding ``energy``, ``forces`` and
        ``magmoms`` where the output gives them; empty when none.

    Raises
    ------
    ValueError
        If a configuration of the output has another number of atoms than
        the input structure.
    Exception
        Whatever ASE's reader raises on an output it cannot read.
    """
    images = ase.io.read(io.StringIO(text), index=':', format='espresso-out')
    frames = []
    for n, image in enumerate(images, 1):
        if len(image) != len(structure):
            raise ValueError(f'configuration {n} of the output has {len(image)} atoms, the input '
                             f'structure {len(structure)}')
        frame = extxyz_atoms(_final_structure(structure, image.positions))
        found = image.calc.results if image.calc is not None else {}
        results = {}
        if found.get('energy') is not None:
            results['energy'] = float(found['energy']) * _ENERGY
        if found.get('forces') is not None:
            results['forces'] = np.asarray(found['forces']) * _FORCE
        if found.get('magmoms') is not None:
            results['magmoms'] = np.asarray(found['magmoms'], dtype=float)
        frame.calc = SinglePointCalculator(frame, **results)
        frames.append(frame)
    return frames


def trajectory_text(text, structure):
    """Text of ``trajectory.extxyz`` for the output of a relaxation, or None.

    Parameters
    ----------
    text : str
        Content of the output of pw.x.
    structure : pymatgen.core.structure.Structure
        Structure of the input (``structure_in.extxyz``).

    Returns
    -------
    str or None
        The frames of ``trajectory_frames`` in the extxyz format, the results
        with them; None when the output has no configuration with results.

    Raises
    ------
    As ``trajectory_frames``, and ``StructureError`` for per-atom data that
    extxyz cannot hold.
    """
    frames = trajectory_frames(text, structure)
    if not frames:
        return None
    buffer = io.StringIO()
    ase.io.write(buffer, frames, format='extxyz')
    return buffer.getvalue()
