# Author: Vito Foderà
# Created: 2026-11-14

"""Helpers for the tests of the LAMMPS side of the MLIP engine.

LAMMPS is not installed and never run: the tests write the files that LAMMPS writes, in the
format the workflow reads, with the numbers of ASE's EMT (a classical potential for a few
metals, in the tests only) put in the frame of LAMMPS with the same rotation ASE uses to write the
data file. What the files look like in a real LAMMPS is checked on the cluster, not here.
"""

import numpy as np
from ase.calculators.emt import EMT
from ase.calculators.lammps.coordinatetransform import Prism
from pymatgen.core import Lattice, Structure

from triboworkflow.core.structures import to_atoms
from triboworkflow.engines.mlip import lammps_job

# 1 eV/angstrom^3 in bar.
EV_PER_A3_TO_BAR = 1.602176634E6
VERSION = '22 Jul 2025 - Update 1'

SUMMARY = """Minimization stats:
  Stopping criterion = {criterion}
  Energy initial, next-to-last, final =
     -1.2000000000000    -1.2300000000000    -1.2300000000000
  Force two-norm initial, final = 1.2 0.0123
  Force max component initial, final = 0.5 0.0092
  Final line search alpha, max atom move = 0.1 0.0001
  Iterations, force evaluations = {iterations} {evaluations}
"""


def rotated(structure, angle=0.7, axis=(1.0, 2.0, 3.0)):
    """The same structure turned by an angle about an axis: its cell is no longer LAMMPS'."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross
    props = {key: structure.site_properties[key] for key in structure.site_properties}
    return Structure(Lattice(structure.lattice.matrix @ rotation.T), structure.species,
                     structure.cart_coords @ rotation.T, coords_are_cartesian=True,
                     labels=structure.labels, site_properties=props)


def emt_values(structure):
    """Energy (eV), forces (eV/A) and pressure tensor (bar, sign of LAMMPS) of the EMT."""
    atoms = to_atoms(structure)
    atoms.set_constraint()
    atoms.calc = EMT()
    return (atoms.get_potential_energy(), atoms.get_forces(),
            -atoms.get_stress(voigt=False) * EV_PER_A3_TO_BAR)


def log_text(criterion='force tolerance', iterations=37, evaluations=52, relax=True, closed=True,
             error=None):
    """Log of LAMMPS: the version, the summary of ``minimize``, the last line."""
    text = f'LAMMPS ({VERSION})\nReading data file ...\n'
    if relax:
        text += SUMMARY.format(criterion=criterion, iterations=iterations,
                               evaluations=evaluations)
    if error:
        return text + f'ERROR: {error} (src/pair.cpp:123)\nLast command: pair_coeff * * Cu\n'
    return text + ('Total wall time: 0:00:04\n' if closed else 'Loop time of 0.01 on 1 procs\n')


def dump_text(structure, steps=(0,), forces=None, positions=None, types=None, shift=None,
              columns='id type xu yu zu fx fy fz'):
    """Text of a dump of LAMMPS with a snapshot per step, in the frame of LAMMPS.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        The structure of the input: its atoms, and its cell for the rotation into LAMMPS'.
    forces, positions : array_like, optional
        In the frame of the structure; the EMT forces, and the positions of the structure.
    types : list of int, optional
        Types of the atoms; by default those of the data file.
    shift : dict, optional
        ``{atom index: lattice vectors to add to its position}``: an atom in another image.
    """
    prism = Prism(np.array(structure.lattice.matrix))
    species = lammps_job.species_of(structure)
    positions = structure.cart_coords if positions is None else np.asarray(positions, float)
    positions = positions.copy()
    for index, vectors in (shift or {}).items():
        positions[index] += np.asarray(vectors, float) @ structure.lattice.matrix
    forces = np.zeros_like(positions) if forces is None else np.asarray(forces, float)
    types = types or [species.index(site.specie.symbol) + 1 for site in structure]
    lammps_positions = prism.vector_to_lammps(positions)
    lammps_forces = prism.vector_to_lammps(forces)
    xhi, yhi, zhi, xy, xz, yz = prism.get_lammps_prism()
    text = ''
    for step in steps:
        text += (f'ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{len(structure)}\n'
                 f'ITEM: BOX BOUNDS xy xz yz pp pp pp\n0 {xhi + max(xy, 0)} {xy}\n'
                 f'0 {yhi} {xz}\n0 {zhi} {yz}\nITEM: ATOMS {columns}\n')
        for i in range(len(structure)):
            row = [i + 1, types[i], *lammps_positions[i], *lammps_forces[i]]
            text += ' '.join(f'{value:.15g}' for value in row[:len(columns.split())]) + '\n'
    return text


def result_text(energy, pressure, prism):
    """Line of ``result.txt``: the pressure tensor of the frame of the input, in LAMMPS'."""
    tensor = prism.rot_mat.T @ pressure @ prism.rot_mat
    values = [energy, tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2],
              tensor[1, 2]]
    return f'{lammps_job.RESULT_TAG} ' + ' '.join(f'{value:.15g}' for value in values) + '\n'


def write_outputs(directory, structure, calculation='relax', criterion='force tolerance',
                  iterations=37, **dump):
    """Write the files LAMMPS writes for a calculation that ended, with the values of the EMT.

    Returns
    -------
    tuple
        Energy, forces (frame of the structure) and pressure tensor (bar, frame of the
        structure), the values the workflow must give back.
    """
    directory.mkdir(parents=True, exist_ok=True)
    energy, forces, pressure = emt_values(structure)
    (directory / lammps_job.LOG_FILE).write_text(
        log_text(criterion, iterations, relax=calculation == 'relax'))
    (directory / lammps_job.FINAL_FILE).write_text(
        dump_text(structure, steps=(iterations,), forces=forces, **dump))
    (directory / lammps_job.RESULT_FILE).write_text(
        result_text(energy, pressure, Prism(np.array(structure.lattice.matrix))))
    return energy, forces, pressure
