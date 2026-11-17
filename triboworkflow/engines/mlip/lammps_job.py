# Author: Vito Foderà
# Created: 2026-11-14

"""MLIP calculations with LAMMPS: the input of a run, and the states and results of its output.

LAMMPS is a program that the workflow starts, as it starts ``pw.x``, so there is no executor
script: this module writes what LAMMPS reads and reads what it wrote. It imports no MLIP library
(``torch``, ``mace``, ``deepmd``) and runs wherever the workflow does. The command that starts
LAMMPS (``lmp -in input.lammps -log log.lammps``, with the plugins and the options of the build
of the model) belongs to the configuration of the machine.

- ``write_input`` writes ``input.lammps`` and ``structure.data`` from the ``[mlip]`` settings
  and the structure of the calculation; ``lammps_input`` and ``lammps_data`` give their text.
- ``lammps_status`` tells from the log how LAMMPS ended; ``parse_result`` turns the output into
  the results of the record, with the keys and the units of ``engines/qe/results.py`` (eV,
  angstrom, eV/angstrom, kbar), and the final structure; ``last_structure`` gives the last
  structure that LAMMPS wrote, for a new attempt.

Files, all in the directory of the calculation:

=================== ====== ==============================================================
File                       Content
=================== ====== ==============================================================
``input.lammps``    input  the script
``structure.data``  input  cell and positions of the atoms, in the frame of LAMMPS
``log.lammps``      output the log of LAMMPS
``trajectory.dump`` output the positions every 10 iterations (relaxation only)
``final.dump``      output positions and forces of the final structure, constraints off
``result.txt``      output energy and pressure tensor of the final structure
=================== ====== ==============================================================

The script reads the structure, loads the model (``pair_style`` and ``pair_coeff``), and for a
relaxation locks the constrained components of the atoms (``fix setforce``) and minimizes. It
then removes the locks and runs zero steps (``run 0``): the energy, the forces and the stress of
that run are those of the model on the final structure, without constraints. A single point is
the ``run 0`` alone. The cell is fixed, as in every relaxation of the workflow.

The relaxation stops when the largest force on an atom, with the constraints applied (the forces
the minimizer sees), is below ``fmax``: ``min_modify norm max`` makes the tolerance of
``minimize`` a limit on the force of one atom, as ``fmax`` of ASE is, and not on the norm of the
forces of all the atoms. The forces recorded are those of the model, without constraints.

States of the record: a LAMMPS that ended has ``process = 'ended'`` and, like any model without
an SCF cycle, ``scf = 'n/a'``. ``geometry`` and ``stop_reason`` follow the criterion with which
``minimize`` stopped:

=============================================== ================= ==================
Stopping criterion of ``minimize``              ``geometry``      ``stop_reason``
=============================================== ================= ==================
force tolerance, forces are zero                ``converged``     ``completed``
max iterations, max force evaluations           ``max_steps``     ``max_steps``
walltime limit reached                          ``not_converged`` ``max_seconds``
any other (line search failed, ...)             ``not_converged`` ``optimizer_failed``
=============================================== ================= ==================

An ``ERROR`` in the log gives ``process = 'crashed'`` and ``stop_reason = 'error'``; a log that
is missing or ends before LAMMPS closed it, ``process = None`` and ``stop_reason = 'truncated'``:
LAMMPS was killed or is still running, and the scheduler tells which.

Notes
-----
The lines of the script follow SCS (``exploration/workflow/md/lammps.py``, CC BY 4.0):
``units metal`` and the header (``:421-428``), ``pair_style`` and ``pair_coeff`` (``:187-189``,
``:204-211``, ``:232-240``), ``minimize`` (``:329-333``) and the data file written by ASE
(``:109-122``). The frame of the output is the one of the input: ASE writes the cell as
LAMMPS requires, rotated so that the first vector lies along x, and ``Prism`` gives the rotation
back; positions, forces and the pressure tensor are rotated back with it.

To remove: differences from SCS. ``pair_style deepmd`` takes the path of the model, where SCS
replaced it with the models found in ``../../Training/NN*/`` (A104, DEV-52). A MACE model is the
file already converted for LAMMPS, and a ``.model`` is an error, where SCS replaced the name of
the file without saying (A106, DEV-53). ``min_modify norm max`` and ``ftol = fmax``, where SCS
left the norm of all the forces (DEV-55). The model deviation of SCS (``relative``, ``out_freq``)
is not written (A105). The results are read from the output instead of from the trajectory, and
the structure comes back in the frame of the input, where SCS reads the dump as LAMMPS wrote it.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.calculators.lammps.coordinatetransform import Prism
from ase.io.lammpsdata import write_lammps_data
from monty.json import jsanitize
from pymatgen.core.structure import Structure

from triboworkflow.core.store.atomic import write_text
from triboworkflow.core.structures import cartesian_if_pos, to_atoms
from triboworkflow.core.structures.model import MAGMOM
from triboworkflow.engines.qe.results import Parsed

from .registry import ModelError, check_model

INPUT_FILE = 'input.lammps'
DATA_FILE = 'structure.data'
LOG_FILE = 'log.lammps'
TRAJECTORY_FILE = 'trajectory.dump'
FINAL_FILE = 'final.dump'
RESULT_FILE = 'result.txt'
# Files LAMMPS writes: removed before a new attempt, so that each belongs to the last one.
OUTPUT_FILES = (LOG_FILE, TRAJECTORY_FILE, FINAL_FILE, RESULT_FILE)
CALCULATIONS = ('scf', 'relax')

# Tag of the line of ``result.txt``: the energy, then the six components of the pressure tensor.
RESULT_TAG = 'TRIBO_RESULT'
PRESSURE_KEYS = ('pxx', 'pyy', 'pzz', 'pxy', 'pxz', 'pyz')

# Iterations between two snapshots of the trajectory, and between two lines of the thermo output.
TRAJECTORY_EVERY = 10
# Largest number of force evaluations of ``minimize`` (SCS ``lammps.py:333``).
MAX_EVALUATIONS = 1000000
# Pressure of LAMMPS in bar (units metal); the record has kbar.
BAR_TO_KBAR = 1.0E-3
# Suffix of the model that MACE converts for LAMMPS (ML-IAP).
MACE_CONVERTED_SUFFIX = '.model-mliap_lammps.pt'

# Largest difference, in angstrom, between the atoms of the output and those of the input.
_TOLERANCE = 1E-6

_STOP = re.compile(r'^\s*Stopping criterion = (.+?)\s*$', re.M)
_ITERATIONS = re.compile(r'^\s*Iterations, force evaluations = (\d+)\s+(\d+)', re.M)
_ERROR = re.compile(r'^ERROR.*$', re.M)
_ENDED = re.compile(r'^Total wall time:', re.M)
_VERSION = re.compile(r'^LAMMPS \((.+)\)', re.M)


class LammpsOutputError(ValueError):
    """The output of LAMMPS cannot be read, or does not agree with the input structure."""


@dataclass(frozen=True)
class LammpsStatus:
    """States of the execution of LAMMPS.

    Parameters
    ----------
    process : str or None
        ``'ended'`` (the log is closed), ``'crashed'`` (the log has an ``ERROR``) or None (no
        log, or one that is not closed).
    scf : str or None
        ``'n/a'`` when LAMMPS ended: the model has no SCF cycle; None otherwise.
    geometry : str or None
        ``'n/a'`` for a single point; for a relaxation ``'converged'``, ``'max_steps'`` or
        ``'not_converged'``. None unless LAMMPS ended.
    stop_reason : str or None
        ``'completed'``, ``'max_steps'``, ``'max_seconds'``, ``'optimizer_failed'``, ``'error'``
        or ``'truncated'``.
    error : str or None
        The ``ERROR`` line of the log of a crashed execution.
    steps : dict or None
        Iterations of the minimizer and their total, ``{'cg': n, 'total': n}``; None for a
        single point.
    criterion : str or None
        The stopping criterion of ``minimize``, as the log has it.
    fallback : None
        There is a single minimizer, so nothing falls back on another.
    """

    process: str | None
    scf: str | None
    geometry: str | None
    stop_reason: str | None
    error: str | None = None
    steps: dict | None = None
    criterion: str | None = None
    fallback: str | None = None


def _deepmd_pair_style(path):
    return f'pair_style deepmd {path}'


def _mace_pair_style(path):
    if path.endswith('.model'):
        raise ModelError('path', f'{path!r} is a MACE model, not the file converted for LAMMPS: '
                                 'convert it (create_lammps_model.py of MACE) and give the path '
                                 f'of the result, which ends with {MACE_CONVERTED_SUFFIX!r}')
    return f'pair_style mliap unified {path} 0'


# type of model -> (line of pair_style from the path of the model, needs ``atom_modify map yes``)
PAIR_STYLES = {'DP': (_deepmd_pair_style, False), 'MACE': (_mace_pair_style, True)}


def species_of(structure):
    """Elements of a structure in the order of the atom types of LAMMPS.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure

    Returns
    -------
    list of str
        The elements in the order in which they first appear: type ``i + 1`` of the data file
        and the ``i``-th name of ``pair_coeff``.
    """
    return list(dict.fromkeys(site.specie.symbol for site in structure))


def _prism(structure):
    """The rotation between the frame of the structure and the one of LAMMPS."""
    return Prism(np.array(structure.lattice.matrix))


def lammps_data(structure):
    """Text of the data file of LAMMPS (``structure.data``) of a structure.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        Periodic and ordered.

    Returns
    -------
    str
        Atom types in the order of ``species_of``, masses of the elements, the cell as LAMMPS
        wants it (first vector along x, second in the xy plane) and the positions rotated with
        it. Positions are not wrapped into the cell.
    """
    buffer = io.StringIO()
    write_lammps_data(buffer, to_atoms(structure), specorder=species_of(structure),
                      prismobj=_prism(structure), masses=True)
    return buffer.getvalue()


def _id_ranges(ids):
    """Atom ids (ascending) as the ranges ``group ... id`` takes: ``[1, 2, 3, 7]`` -> '1:3 7'."""
    ranges, start = [], 0
    for position in range(1, len(ids) + 1):
        if position == len(ids) or ids[position] != ids[position - 1] + 1:
            first, last = ids[start], ids[position - 1]
            ranges.append(str(first) if first == last else f'{first}:{last}')
            start = position
    return ' '.join(ranges)


def _locks(structure):
    """Groups of atoms that share the same locked components: (axes, atom ids) sorted by axes.

    ``axes`` is a string such as ``'xy'``; atoms with no constraint, or with every component
    free, are in no group.
    """
    flags = cartesian_if_pos(structure)
    groups = {}
    for atom_id, atom_flags in enumerate(flags or [], start=1):
        if atom_flags is None:
            continue
        axes = ''.join(axis for axis, flag in zip('xyz', atom_flags) if flag == 0)
        if axes:
            groups.setdefault(axes, []).append(atom_id)
    return sorted(groups.items())


def _hms(seconds):
    seconds = max(1, int(seconds))
    return f'{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}'


def lammps_input(calculation, mlip, structure, max_seconds=None):
    """Text of the script of LAMMPS (``input.lammps``) for a calculation.

    Parameters
    ----------
    calculation : {'scf', 'relax'}
        ``'scf'`` is a single point: the energy, forces and stress of the structure.
    mlip : triboworkflow.core.settings.schema.MlipParams
        The ``[mlip]`` settings. ``path`` is written as it is; ``kwargs`` must be empty, because
        they are the arguments of the loaders of ASE and LAMMPS has none.
    structure : pymatgen.core.structure.Structure
        The structure of the calculation, with the constraints (``if_pos``) it relaxes with. A
        component with flag 0 is locked: ``fix setforce`` zeroes its force during the relaxation.
    max_seconds : float, optional
        Time budget of the minimization, in seconds, checked by LAMMPS (``timer timeout``); None
        for none. It counts from the start of ``minimize``, not of the process.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If ``calculation`` is neither ``'scf'`` nor ``'relax'``.
    triboworkflow.engines.mlip.registry.ModelError
        If the model cannot be loaded with LAMMPS (``registry.check_model``: the type is not
        available with it, the path is not absolute); the type has no ``pair_style`` here;
        ``kwargs`` is not empty; or a MACE model is the ``.model`` file instead of the one
        converted for LAMMPS.
    """
    if calculation not in CALCULATIONS:
        raise ValueError(f'calculation {calculation!r}: only {" and ".join(CALCULATIONS)}')
    check_model(mlip.type, mlip.path, mlip.kwargs, backend='lammps')
    if mlip.type not in PAIR_STYLES:
        raise ModelError('type', f'no pair_style is known for {mlip.type!r}; known: '
                                 f'{", ".join(map(repr, PAIR_STYLES))}')
    if mlip.kwargs:
        raise ModelError('kwargs', 'the arguments of a loader of ASE do not reach LAMMPS: the '
                                   'precision and the head of the model are those of the file '
                                   f'(path); remove {", ".join(map(repr, mlip.kwargs))}')
    pair_style, atom_map = PAIR_STYLES[mlip.type]
    xhi, yhi, _, xy, xz, yz = _prism(structure).get_lammps_prism()
    lines = ['units metal', 'atom_style atomic']
    lines += ['atom_modify map yes'] if atom_map else []
    lines += ['newton on', 'boundary p p p']
    # LAMMPS accepts a tilt of up to half the length of the box, a hexagonal cell included.
    tilted = max(abs(xy) / xhi, abs(xz) / xhi, abs(yz) / yhi) > 0.5 + 1E-8
    lines += ['box tilt large'] if tilted else []
    lines += [f'read_data {DATA_FILE}', '', pair_style(mlip.path),
              f'pair_coeff * * {" ".join(species_of(structure))}', '',
              f'thermo {TRAJECTORY_EVERY}', 'thermo_style custom step pe fmax fnorm press']
    if calculation == 'relax':
        locks = _locks(structure)
        for axes, ids in locks:
            forces = ' '.join('0.0' if axis in axes else 'NULL' for axis in 'xyz')
            lines += [f'group blocked_{axes} id {_id_ranges(ids)}',
                      f'fix lock_{axes} blocked_{axes} setforce {forces}']
        lines += ['', f'dump trajectory all custom {TRAJECTORY_EVERY} {TRAJECTORY_FILE} '
                      'id type xu yu zu',
                  'dump_modify trajectory sort id format float %.15g flush yes']
        lines += [f'timer timeout {_hms(max_seconds)}'] if max_seconds is not None else []
        lines += ['min_style cg', 'min_modify norm max',
                  f'minimize 0.0 {float(mlip.fmax)!r} {int(mlip.maxsteps)} {MAX_EVALUATIONS}',
                  '', 'undump trajectory']
        lines += [f'unfix lock_{axes}' for axes, _ in locks]
        lines += ['timer timeout off'] if max_seconds is not None else []
    lines += ['', f'dump final all custom 1 {FINAL_FILE} id type xu yu zu fx fy fz',
              'dump_modify final sort id format float %.15g', 'run 0', 'undump final',
              'variable energy equal pe']
    lines += [f'variable pressure_{key[1:]} equal {key}' for key in PRESSURE_KEYS]
    values = ' '.join(['${energy}'] + ['${pressure_' + key[1:] + '}' for key in PRESSURE_KEYS])
    lines += [f'print "{RESULT_TAG} {values}" file {RESULT_FILE} screen no']
    return '\n'.join(lines) + '\n'


def write_input(directory, calculation, mlip, structure, max_seconds=None):
    """Write ``input.lammps`` and ``structure.data`` in the directory of a calculation.

    Parameters
    ----------
    directory : path-like
    calculation, mlip, structure, max_seconds
        As ``lammps_input``.

    Returns
    -------
    pathlib.Path
        The script; LAMMPS is started in the directory as ``<lmp> -in input.lammps -log
        log.lammps``.
    """
    script = lammps_input(calculation, mlip, structure, max_seconds)
    write_text(Path(directory) / DATA_FILE, lammps_data(structure))
    path = Path(directory) / INPUT_FILE
    write_text(path, script)
    return path


def remove_outputs(directory):
    """Remove the files that LAMMPS wrote in an earlier attempt, if any.

    Parameters
    ----------
    directory : path-like
        Directory of the calculation. ``OUTPUT_FILES`` are removed: the files of the next
        execution would otherwise be mixed with the ones of the last.
    """
    for name in OUTPUT_FILES:
        (Path(directory) / name).unlink(missing_ok=True)


def _read_text(directory, name):
    try:
        return (Path(directory) / name).read_text(encoding='utf-8', errors='replace')
    except FileNotFoundError:
        return None


def lammps_status(directory, calculation):
    """States of the execution of LAMMPS in a calculation directory.

    Parameters
    ----------
    directory : path-like
        With ``log.lammps``.
    calculation : {'scf', 'relax'}

    Returns
    -------
    LammpsStatus

    Raises
    ------
    LammpsOutputError
        If the log of a relaxation is closed but has no summary of ``minimize``.
    """
    log = _read_text(directory, LOG_FILE)
    if log is None:
        return LammpsStatus(None, None, None, 'truncated')
    error = _ERROR.search(log)
    if error:
        return LammpsStatus('crashed', None, None, 'error', error=error.group(0).strip())
    if not _ENDED.search(log):
        return LammpsStatus(None, None, None, 'truncated')
    if calculation == 'scf':
        return LammpsStatus('ended', 'n/a', 'n/a', 'completed')
    stops, iterations = _STOP.findall(log), _ITERATIONS.findall(log)
    if not stops or not iterations:
        raise LammpsOutputError(f'{LOG_FILE} has no summary of the minimization '
                                '("Stopping criterion", "Iterations, force evaluations")')
    criterion, steps = stops[-1], int(iterations[-1][0])
    lowered = criterion.lower()
    if 'force tolerance' in lowered or 'forces are zero' in lowered:
        geometry, reason = 'converged', 'completed'
    elif 'max iterations' in lowered or 'max force evaluations' in lowered:
        geometry, reason = 'max_steps', 'max_steps'
    elif 'walltime' in lowered:
        geometry, reason = 'not_converged', 'max_seconds'
    else:
        geometry, reason = 'not_converged', 'optimizer_failed'
    return LammpsStatus('ended', 'n/a', geometry, reason, steps={'cg': steps, 'total': steps},
                        criterion=criterion)


def _last_frame(directory, name):
    """Last complete snapshot of a dump file: its columns as arrays sorted by atom id, or None.

    A snapshot is complete when its header is whole, it has all its atoms with all their
    columns and, if it is the last of the file, the file ends with a newline: a file cut while
    LAMMPS wrote it ends inside a number that would still read as one.
    """
    text = _read_text(directory, name)
    if text is None:
        return None
    chunks = text.split('ITEM: TIMESTEP')[1:]
    for index in range(len(chunks) - 1, -1, -1):
        if index == len(chunks) - 1 and not text.endswith('\n'):
            continue
        lines = chunks[index].strip().split('\n')
        try:
            natoms = int(lines[2])
            if (lines[1].strip() != 'ITEM: NUMBER OF ATOMS' or not lines[3].startswith('ITEM: BOX')
                    or not lines[7].startswith('ITEM: ATOMS')):
                continue
            columns = lines[7].split()[2:]
            rows = [line.split() for line in lines[8:8 + natoms]]
            if len(rows) != natoms or any(len(row) != len(columns) for row in rows):
                continue
            values = np.array(rows, dtype=float).reshape(natoms, len(columns))
        except (ValueError, IndexError):
            continue
        values = values[np.argsort(values[:, columns.index('id')])]
        return {column: values[:, i] for i, column in enumerate(columns)}
    return None


def _nearest_image(structure, positions):
    """Positions equal to ``positions`` up to lattice vectors, the nearest to the input ones."""
    cell, reference = structure.lattice.matrix, structure.cart_coords
    delta = positions - reference
    delta -= np.round(delta @ np.linalg.inv(cell)) @ cell
    return reference + delta


def _positions(frame, structure, prism):
    """Positions of a snapshot in the frame of the input, checked against the structure."""
    species = species_of(structure)
    expected = np.array([species.index(site.specie.symbol) + 1 for site in structure])
    if len(frame['id']) != len(structure):
        raise LammpsOutputError(f'the output has {len(frame["id"])} atoms, the input structure '
                                f'{len(structure)}')
    if not np.array_equal(frame['id'], np.arange(1, len(structure) + 1)):
        raise LammpsOutputError('the atom ids of the output are not 1..N')
    if not np.array_equal(frame['type'].astype(int), expected):
        raise LammpsOutputError('the atom types of the output differ from those of the input '
                                f'structure ({", ".join(species)} in this order)')
    lammps = np.column_stack([frame['xu'], frame['yu'], frame['zu']])
    return _nearest_image(structure, prism.vector_to_ase(lammps))


def _final_structure(structure, positions):
    props = {k: v for k, v in structure.site_properties.items() if k != MAGMOM}
    return Structure(structure.lattice, structure.species, positions, coords_are_cartesian=True,
                     labels=structure.labels, site_properties=props)


def _result_line(directory):
    """Energy (eV) and pressure tensor (bar, LAMMPS frame) of ``result.txt``."""
    text = _read_text(directory, RESULT_FILE)
    if text is None:
        raise LammpsOutputError(f'{RESULT_FILE} is missing: LAMMPS did not reach the end of the '
                                'script')
    for line in text.splitlines():
        words = line.split()
        if words[:1] == [RESULT_TAG]:
            try:
                values = [float(word) for word in words[1:]]
            except ValueError:
                break
            if len(values) == 1 + len(PRESSURE_KEYS) and np.isfinite(values).all():
                pxx, pyy, pzz, pxy, pxz, pyz = values[1:]
                return values[0], np.array([[pxx, pxy, pxz], [pxy, pyy, pyz], [pxz, pyz, pzz]])
    raise LammpsOutputError(f'{RESULT_FILE} has no line "{RESULT_TAG}" with the energy and the six '
                            'components of the pressure, finite')


def _observed(directory):
    observed = {}
    version = _VERSION.search(_read_text(directory, LOG_FILE) or '')
    if version:
        observed['lammps'] = version.group(1)
    for line in (_read_text(directory, INPUT_FILE) or '').splitlines():
        if line.startswith('pair_style '):
            observed['pair_style'] = line[len('pair_style '):]
    return observed


def parse_result(directory, calculation, structure):
    """Results and state ``parsing`` of a calculation from the output of LAMMPS.

    The output is read only when LAMMPS ended and the geometry converged (a single point has
    none): otherwise ``parsing`` and the results stay None, as for pw.x.

    Parameters
    ----------
    directory : path-like
        With ``log.lammps``, ``final.dump`` and ``result.txt``.
    calculation : {'scf', 'relax'}
    structure : pymatgen.core.structure.Structure
        Structure of the input (``structure_in.extxyz``).

    Returns
    -------
    triboworkflow.engines.qe.results.Parsed
        ``parsing`` is ``'ok'`` with the results, the final structure and ``observed`` (the
        version of LAMMPS and the ``pair_style`` of the script); ``'failed'`` with the
        ``reason``, and no exception, when the output cannot be read or does not agree with the
        input (number of atoms, their types, and for a single point the positions); None when
        there is nothing to read. The results have the keys ``nsites``, ``input.structure``,
        ``output.structure`` (site properties of the input but ``magmom``), ``output.density``
        (g/cm^3), ``output.energy`` and ``output.energy_per_atom`` (eV), ``output.forces``
        (eV/angstrom, on every atom, as the model gives them, constraints off) and
        ``output.stress`` (kbar, sign of pw.x). Positions, forces and stress are in the frame of
        the input structure; the positions are the images nearest to the input ones.
    """
    try:
        status = lammps_status(directory, calculation)
        if status.process != 'ended' or status.geometry not in ('converged', 'n/a'):
            return Parsed(None)
        frame = _last_frame(directory, FINAL_FILE)
        if frame is None or any(key not in frame for key in ('xu', 'yu', 'zu', 'fx', 'fy', 'fz')):
            raise LammpsOutputError(f'{FINAL_FILE} has no complete snapshot with positions and '
                                    'forces')
        prism = _prism(structure)
        positions = _positions(frame, structure, prism)
        energy, pressure = _result_line(directory)
        if calculation == 'scf' and np.abs(positions - structure.cart_coords).max() > _TOLERANCE:
            raise LammpsOutputError('the positions of the output differ from those of the input '
                                    'structure')
    except LammpsOutputError as error:
        return Parsed('failed', reason=str(error))
    forces = prism.vector_to_ase(np.column_stack([frame['fx'], frame['fy'], frame['fz']]))
    final = _final_structure(structure, structure.cart_coords if calculation == 'scf'
                             else positions)
    nsites = len(structure)
    output = {'structure': final.as_dict(), 'density': float(final.density), 'energy': energy,
              'energy_per_atom': energy / nsites, 'forces': forces.tolist(),
              'stress': (prism.tensor2_to_ase(pressure) * BAR_TO_KBAR).tolist()}
    results = {'nsites': nsites, 'input': {'structure': structure.as_dict()}, 'output': output}
    return Parsed('ok', results=jsanitize(results, strict=False), structure=final,
                  observed=_observed(directory))


def last_structure(directory, structure):
    """The last structure LAMMPS wrote, where a new attempt restarts from.

    Parameters
    ----------
    directory : path-like
        With ``final.dump`` (LAMMPS ran to the end of the script) or ``trajectory.dump`` (it
        was stopped or killed during the minimization).
    structure : pymatgen.core.structure.Structure
        Structure of the input.

    Returns
    -------
    pymatgen.core.structure.Structure or None
        The structure of the input with the positions of the last complete snapshot, in the
        frame of the input; None if there is none.

    Raises
    ------
    LammpsOutputError
        If the snapshot does not agree with the input (number of atoms, their types).
    """
    for name in (FINAL_FILE, TRAJECTORY_FILE):
        frame = _last_frame(directory, name)
        if frame is not None and all(key in frame for key in ('xu', 'yu', 'zu')):
            return _final_structure(structure, _positions(frame, structure, _prism(structure)))
    return None
