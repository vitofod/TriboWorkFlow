# Author: Vito Foderà
# Created: 2026-11-14

"""Job and results of an MLIP calculation with ASE, on the side of the workflow.

The calculation itself is run by the executor ``ase_run.py``, a script of the Python environment
of the MLIP that never imports the workflow. This module is the workflow's half of the exchange
and imports no MLIP library, so it runs wherever the workflow does:

- ``job_of`` and ``write_job`` make the ``job.json`` that the executor reads, from the ``[mlip]``
  settings. The call that creates the calculator comes from the registry
  (``registry.ase_calculator_call``), which refuses a MACE model without ``default_dtype``;
- ``ase_status`` tells from ``result.json`` how the executor ended;
- ``parse_result`` turns ``result.json`` into the results of the record, with the keys and the
  units of ``engines/qe/results.py`` (eV, angstrom, eV/angstrom, kbar), and the final structure.

``result.json`` holds what the executor measured in the units of ASE; the conversion is made
here, in ``stress_to_kbar``, so that it is tested with a known case without a model.

The states are those of the record. An executor that ended has ``process = 'ended'`` and, like
a calculation without an SCF cycle, ``scf = 'n/a'``; ``geometry`` and ``stop_reason`` are
``converged``/``completed`` for a relaxation or a single point that finished,
``max_steps``/``max_steps`` when the steps of every optimizer were used up and
``not_converged``/``max_seconds`` when the time was. A ``result.json`` that says the executor
failed gives ``process = 'crashed'`` and the error; a missing one, ``process = None`` and
``stop_reason = 'truncated'``: the executor was killed, or is still running, and the scheduler
tells which.

Notes
-----
The stress is recorded as ``pw.x`` prints it, the sign of VASP: positive for a compressed
cell. ASE has the opposite sign, so ``output.stress`` is minus ASE's, in kbar. The mean of its
diagonal is the pressure, ``-dE/dV``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from monty.json import jsanitize
from pymatgen.core.structure import Structure

from triboworkflow.core.store.atomic import write_text
from triboworkflow.core.structures.model import MAGMOM
from triboworkflow.engines.qe.results import Parsed

from .ase_run import RESULT_FILE
from .registry import ase_calculator_call, get_model_type

EXECUTOR_PATH = Path(__file__).with_name('ase_run.py')
JOB_FILE = 'job.json'
CALCULATIONS = ('scf', 'relax')

# 1 eV/angstrom^3 in kbar: 1 eV = 1.602176634E-19 J (exact), 1 angstrom^3 = 1E-30 m^3,
# 1 kbar = 1E8 Pa.
EV_PER_A3_TO_KBAR = 1602.176634

# Largest difference, in angstrom, between the cell and the positions of the input structure and
# those of the result: the extxyz of the input holds eight decimals.
_TOLERANCE = 1E-6

# Keys of a result.json of an executor that ended well (see ``ase_run``).
_REQUIRED = ('calculation', 'geometry', 'stop_reason', 'steps', 'fallback', 'energy', 'forces',
             'stress', 'positions', 'symbols', 'cell', 'observed')


class AseResultError(ValueError):
    """``result.json`` cannot be read, or does not agree with the input structure."""


@dataclass(frozen=True)
class AseStatus:
    """States of the execution of the executor.

    Parameters
    ----------
    process : str or None
        ``'ended'`` (``result.json`` says the executor finished), ``'crashed'`` (it says the
        execution failed) or None (no ``result.json``).
    scf : str or None
        ``'n/a'`` when the executor ended: the model has no SCF cycle; None otherwise.
    geometry : str or None
        ``'n/a'`` for a single point; for a relaxation ``'converged'``, ``'max_steps'`` or
        ``'not_converged'`` (stopped for time). None unless the executor ended.
    stop_reason : str or None
        ``'completed'``, ``'max_steps'``, ``'max_seconds'``, ``'error'`` (the executor failed)
        or ``'truncated'`` (no ``result.json``).
    error : str or None
        The error of a failed execution, as ``result.json`` has it.
    steps : dict or None
        Steps of ``BFGSLineSearch`` and ``FIRE2`` and their sum (``total``); None for a single
        point.
    fallback : str or None
        Why ``FIRE2`` ran: ``'max_steps'`` or ``'line_search_failed'``; None if it did not.
    """

    process: str | None
    scf: str | None
    geometry: str | None
    stop_reason: str | None
    error: str | None = None
    steps: dict | None = None
    fallback: str | None = None


def job_of(calculation, mlip, max_seconds=None):
    """Content of ``job.json`` for a calculation.

    Parameters
    ----------
    calculation : {'scf', 'relax'}
        ``'scf'`` is a single point.
    mlip : triboworkflow.core.settings.schema.MlipParams
        The ``[mlip]`` settings. ``kwargs`` go to the loader as written, without defaults.
    max_seconds : float, optional
        Time budget of the execution, model loading included; None for none.

    Returns
    -------
    dict
        The keys that ``ase_run`` reads: ``calculation``, ``model`` (the call that creates the
        calculator), ``package``, ``model_path``, ``fmax``, ``maxsteps`` and ``max_seconds``.

    Raises
    ------
    ValueError
        If ``calculation`` is neither ``'scf'`` nor ``'relax'``.
    triboworkflow.engines.mlip.registry.ModelError
        If the model cannot be loaded with ASE (``ase_calculator_call``): e.g. a MACE model
        without ``default_dtype``.
    """
    if calculation not in CALCULATIONS:
        raise ValueError(f'calculation {calculation!r}: only {" and ".join(CALCULATIONS)}')
    return {'calculation': calculation,
            'model': ase_calculator_call(mlip.type, mlip.path, mlip.kwargs),
            'package': get_model_type(mlip.type).package, 'model_path': mlip.path,
            'fmax': mlip.fmax, 'maxsteps': mlip.maxsteps, 'max_seconds': max_seconds}


def write_job(directory, job):
    """Write ``job.json`` in the directory of a calculation.

    Parameters
    ----------
    directory : path-like
    job : dict
        As ``job_of`` returns.

    Returns
    -------
    pathlib.Path
        The file; the executor is started as ``<python of the MLIP> EXECUTOR_PATH job.json`` in
        the directory.
    """
    path = Path(directory) / JOB_FILE
    write_text(path, json.dumps(job, indent=2, allow_nan=False) + '\n')
    return path


def stress_to_kbar(stress):
    """Stress of ASE (eV/angstrom^3) in kbar, with the sign of pw.x.

    Parameters
    ----------
    stress : array_like
        3x3 stress tensor with the sign of ASE.

    Returns
    -------
    list of list of float
        Minus the stress, times 1602.176634 kbar per eV/angstrom^3.
    """
    return (-np.asarray(stress, dtype=float) * EV_PER_A3_TO_KBAR).tolist()


def read_result(directory):
    """Content of ``result.json`` of a calculation.

    Parameters
    ----------
    directory : path-like

    Returns
    -------
    dict or None
        None if there is no ``result.json``.

    Raises
    ------
    AseResultError
        If the file is not JSON, or lacks a key that an executor that ended well writes.
    """
    path = Path(directory) / RESULT_FILE
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except FileNotFoundError:
        return None
    try:
        result = json.loads(text)
        required = _REQUIRED if result['ok'] else ('error',)
        missing = [key for key in required if key not in result]
    except (ValueError, KeyError, TypeError) as error:
        raise AseResultError(f'{RESULT_FILE} is not the result of the executor: '
                             f'{type(error).__name__}: {error}') from None
    if missing:
        raise AseResultError(f'{RESULT_FILE} lacks {", ".join(missing)}')
    return result


def _status_of(result):
    if result is None:
        return AseStatus(None, None, None, 'truncated')
    if not result['ok']:
        return AseStatus('crashed', None, None, 'error', error=result['error'])
    return AseStatus('ended', 'n/a', result['geometry'], result['stop_reason'],
                     steps=result['steps'], fallback=result['fallback'])


def ase_status(directory):
    """States of the execution of the executor in a calculation directory.

    Parameters
    ----------
    directory : path-like
        With ``result.json`` once the executor ended.

    Returns
    -------
    AseStatus

    Raises
    ------
    AseResultError
        As ``read_result``.
    """
    return _status_of(read_result(directory))


def _check_agreement(result, calculation, structure):
    """Check that the result is of the calculation and of the structure of the input."""
    if result['calculation'] != calculation:
        raise AseResultError(f'the result is of a {result["calculation"]!r} calculation, not '
                             f'{calculation!r}')
    symbols = [site.specie.symbol for site in structure]
    if len(result['positions']) != len(structure) or len(result['forces']) != len(structure):
        raise AseResultError(f'the result has {len(result["positions"])} atoms, the input '
                             f'structure {len(structure)}')
    if result['symbols'] != symbols:
        raise AseResultError(f'elements of the result {result["symbols"]} differ from those of '
                             f'the input structure {symbols}')
    if np.abs(np.asarray(result['cell']) - structure.lattice.matrix).max() > _TOLERANCE:
        raise AseResultError('the cell of the result differs from that of the input structure')
    if calculation == 'scf' and np.abs(np.asarray(result['positions'])
                                       - structure.cart_coords).max() > _TOLERANCE:
        raise AseResultError('the positions of the result differ from those of the input '
                             'structure')


def _final_structure(structure, positions):
    props = {k: v for k, v in structure.site_properties.items() if k != MAGMOM}
    return Structure(structure.lattice, structure.species, positions, coords_are_cartesian=True,
                     labels=structure.labels, site_properties=props)


def parse_result(directory, calculation, structure):
    """Results and state ``parsing`` of a calculation from its ``result.json``.

    The result is read only when the executor ended and the geometry converged (a single point
    has none): otherwise ``parsing`` and the results stay None, as for pw.x.

    Parameters
    ----------
    directory : path-like
        With ``result.json``.
    calculation : {'scf', 'relax'}
    structure : pymatgen.core.structure.Structure
        Structure of the input (``structure_in.extxyz``).

    Returns
    -------
    triboworkflow.engines.qe.results.Parsed
        ``parsing`` is ``'ok'`` with the results, the final structure and ``observed`` (what
        the executor found: versions, device, path of the model); ``'failed'`` with the
        ``reason``, and no exception, when ``result.json`` cannot be read or does not agree with
        the input (calculation, number of atoms, elements, cell, and for a single point the
        positions); None when there is nothing to read. The results have the keys ``nsites``,
        ``input.structure``, ``output.structure`` (site properties of the input but
        ``magmom``), ``output.density`` (g/cm^3), ``output.energy`` and
        ``output.energy_per_atom`` (eV), ``output.forces`` (eV/angstrom, on every atom, as the
        model gives them, constraints not applied) and ``output.stress`` (kbar, sign of pw.x;
        absent if the model has no stress).
    """
    try:
        result = read_result(directory)
        status = _status_of(result)
        if status.process != 'ended' or status.geometry not in ('converged', 'n/a'):
            return Parsed(None)
        _check_agreement(result, calculation, structure)
    except AseResultError as error:
        return Parsed('failed', reason=str(error))
    positions = structure.cart_coords if calculation == 'scf' else result['positions']
    final = _final_structure(structure, positions)
    nsites = len(structure)
    output = {'structure': final.as_dict(), 'density': float(final.density),
              'energy': result['energy'], 'energy_per_atom': result['energy'] / nsites,
              'forces': result['forces']}
    if result['stress'] is not None:
        output['stress'] = stress_to_kbar(result['stress'])
    results = {'nsites': nsites, 'input': {'structure': structure.as_dict()}, 'output': output}
    return Parsed('ok', results=jsanitize(results, strict=False), structure=final,
                  observed=result['observed'])
