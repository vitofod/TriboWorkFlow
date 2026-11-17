# Author: Vito Foderà
# Created: 2026-11-14

"""Executor of MLIP calculations with ASE: one structure, a single point or a relaxation.

A script, not a part of the workflow code. It runs in the Python environment of the MLIP
(``torch`` and the library of the model), which is not the one of the workflow, so it imports
only the standard library, ``ase`` (with ``numpy``) and the calculator named in the job. It
never imports ``triboworkflow``. The workflow starts it in the directory of a calculation::

    <python of the MLIP environment> ase_run.py job.json

and reads ``result.json`` when it ends. Calculations run on GPU only: before the model is
loaded the executor checks that ``torch`` sees CUDA, and fails with a clear error if it does
not (DeePMD would otherwise fall back to the CPU without a warning).

Files, all in the directory of ``job.json``:

- ``job.json`` (input): see below;
- ``structure_in.extxyz`` (input): cell, positions and constraints (``move_mask``) of the atoms;
- ``structure_last.extxyz`` (output): the structure after every step, replaced atomically; the
  last one is where a new attempt restarts from, however the execution ended;
- ``optimizer.log`` (output): log of the optimizers of ASE (relaxation only);
- ``result.json`` (output): written last; its presence says that the executor ended.

``job.json``:

``calculation`` : ``'scf'`` (single point) or ``'relax'``. ``model`` : the call that creates the
calculator, ``{'factory': 'module:function', 'args': [...], 'kwargs': {...}}`` (JSON types).
``package`` : the distribution of the model library, whose version is recorded. ``model_path``
: path of the model, recorded as resolved. ``fmax`` (eV/Å) and ``maxsteps`` : convergence
threshold and steps of each optimizer (relaxation only). ``max_seconds`` : time budget of the
whole execution, model loading included, or null; it is checked between two steps.

``result.json``: ``{'ok': false, 'error': ...}`` if the execution failed, else ``ok`` true and

- ``geometry``: ``'converged'``, ``'max_steps'`` (the steps of every optimizer are used up),
  ``'not_converged'`` (stopped for time) or ``'n/a'`` (single point);
- ``stop_reason``: ``'completed'`` (converged, or a single point), ``'max_steps'`` or
  ``'max_seconds'``;
- ``steps``: steps of each optimizer and their sum, ``{'BFGSLineSearch': n, 'FIRE2': m,
  'total': n + m}``, null for a single point; ``fallback``: why ``FIRE2`` ran, ``'max_steps'``
  or ``'line_search_failed'``, null if it did not;
- ``energy`` (eV), ``forces`` (eV/Å, on every atom, constraints not applied), ``stress`` (3x3,
  eV/Å³ with the sign of ASE, null if the calculator has none), ``positions`` (Å), all of the
  last structure; ``symbols`` and ``cell`` (Å) of the atoms, for the workflow to check them
  against its input;
- ``observed``: versions of the libraries, device and path of the model, as found;
- ``elapsed``: seconds.

The relaxation keeps the cell fixed. It converges when the largest force on an atom, with the
constraints applied (the forces the optimizer sees), is below ``fmax``; the forces recorded are
those of the model, without constraints. Exit code 0 when ``result.json`` was written with
``ok`` true, 1 otherwise; a relaxation that did not converge is not an error.

Notes
-----
The relaxation follows Xsorb (``xsorb/optimization/unified_optimize.py:56-61`` and ``:93-101``,
CC BY 4.0): ``BFGSLineSearch(c1=0.1)`` and, if it does not converge, ``FIRE2``, each with
``maxsteps`` steps and the same ``fmax``. The calculators are created as SCS does
(``exploration/workflow/md/ase.py:322-335``, CC BY 4.0), from the call that
``registry.ase_calculator_call`` describes.

To remove: differences from Xsorb. No ``restart.json`` for either optimizer: ``FIRE2`` would
read the file of ``BFGSLineSearch`` and fail, and a file left by an earlier run would be read as
a restart (A101, DEV-50). A failed line search (``RuntimeError('LineSearch failed!')``) is a
lack of convergence, so ``FIRE2`` runs; any other error is not caught. The steps of each
optimizer are recorded, where the log of Xsorb reports those of ``FIRE2`` only (A107, DEV-54).
The time is checked between steps and CUDA before the model is loaded (A109, DEV-56).
"""

import importlib
import importlib.metadata
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import ase
import numpy as np
from ase.io import read, write
from ase.optimize import FIRE2, BFGSLineSearch

STRUCTURE_FILE = 'structure_in.extxyz'
LAST_STRUCTURE_FILE = 'structure_last.extxyz'
LOG_FILE = 'optimizer.log'
RESULT_FILE = 'result.json'

# Message of the error that BFGSLineSearch raises when its line search finds no step.
LINE_SEARCH_FAILED = 'LineSearch failed!'


def check_cuda():
    """Check that ``torch`` sees a CUDA device.

    Returns
    -------
    dict
        ``torch`` (version) and ``device`` (index and name of the current CUDA device).

    Raises
    ------
    RuntimeError
        If ``torch`` cannot be imported or CUDA is not available.
    """
    try:
        import torch
    except ImportError as error:
        raise RuntimeError('torch cannot be imported: MLIP calculations run on GPU only, in the '
                           'environment of the MLIP') from error
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available: MLIP calculations run on GPU only (ask for a '
                           'GPU in the jobscript)')
    index = torch.cuda.current_device()
    return {'torch': torch.__version__, 'device': f'cuda:{index} {torch.cuda.get_device_name(index)}'}


def load_calculator(call):
    """Create the ASE calculator described by a call.

    Parameters
    ----------
    call : dict
        ``{'factory': 'module:function', 'args': [...], 'kwargs': {...}}``.

    Returns
    -------
    ase.calculators.calculator.Calculator
    """
    module_name, function_name = call['factory'].split(':')
    factory = getattr(importlib.import_module(module_name), function_name)
    return factory(*call['args'], **call['kwargs'])


def save_last_structure(atoms, path):
    """Write the atoms as extxyz, replacing the file atomically.

    A process killed while writing leaves the previous structure, never half of one.
    """
    temporary = path.with_name(path.name + '.tmp')
    write(temporary, atoms, format='extxyz', write_results=False)
    os.replace(temporary, path)


def optimize(optimizer, fmax, maxsteps, deadline):
    """Run an optimizer step by step until it converges, runs out of steps or of time.

    Returns
    -------
    converged, timed_out : bool
        Both false when the steps are used up.
    """
    for converged in optimizer.irun(fmax=fmax, steps=maxsteps):
        if converged:
            return True, False
        if time.monotonic() >= deadline:
            return False, True
    return False, False


def relax(atoms, fmax, maxsteps, deadline, log, on_step):
    """Relax the atoms with ``BFGSLineSearch``, then ``FIRE2`` if it does not converge.

    Parameters
    ----------
    atoms : ase.Atoms
        With the calculator; relaxed in place.
    fmax : float
        Largest force on an atom, constraints applied, in eV/Å.
    maxsteps : int
        Steps of each optimizer.
    deadline : float
        ``time.monotonic()`` after which no step is started; ``math.inf`` for none.
    log : file object
        Log of the optimizers.
    on_step : callable
        Called with no argument on the structure of every step.

    Returns
    -------
    dict
        ``geometry``, ``stop_reason``, ``steps`` and ``fallback`` of ``result.json``.
    """
    fallback = None
    optimizer = BFGSLineSearch(atoms, restart=None, logfile=log, c1=0.1)
    optimizer.attach(on_step)
    try:
        converged, timed_out = optimize(optimizer, fmax, maxsteps, deadline)
    except RuntimeError as error:
        if str(error) != LINE_SEARCH_FAILED:
            raise
        converged, timed_out, fallback = False, False, 'line_search_failed'
    steps = {'BFGSLineSearch': optimizer.nsteps, 'FIRE2': 0}
    if not (converged or timed_out):
        fallback = fallback or 'max_steps'
        optimizer = FIRE2(atoms, restart=None, logfile=log)
        optimizer.attach(on_step)
        converged, timed_out = optimize(optimizer, fmax, maxsteps, deadline)
        steps['FIRE2'] = optimizer.nsteps
    steps['total'] = steps['BFGSLineSearch'] + steps['FIRE2']
    if converged:
        geometry, stop_reason = 'converged', 'completed'
    elif timed_out:
        geometry, stop_reason = 'not_converged', 'max_seconds'
    else:
        geometry, stop_reason = 'max_steps', 'max_steps'
    return {'geometry': geometry, 'stop_reason': stop_reason, 'steps': steps, 'fallback': fallback}


def final_values(atoms):
    """Energy, forces, stress and positions of the atoms, in the units of ASE.

    Raises
    ------
    ValueError
        If a value is not finite.
    """
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces(apply_constraint=False)
    try:
        stress = atoms.get_stress(voigt=False)
    except NotImplementedError:
        stress = None
    values = {'energy': energy, 'forces': forces, 'stress': stress, 'positions': atoms.get_positions()}
    for name, value in values.items():
        if value is not None and not np.isfinite(value).all():
            raise ValueError(f'non-finite {name}')
    return {name: None if value is None else np.asarray(value).tolist()
            for name, value in values.items()}


def version_of(package):
    """Installed version of a distribution, or None."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_job(job_path):
    """Run the calculation of a job.

    Parameters
    ----------
    job_path : str or pathlib.Path
        The ``job.json``; the other files are in its directory.

    Returns
    -------
    dict
        The content of ``result.json`` of a successful execution.

    Raises
    ------
    Exception
        Whatever the reading of the job, the check of CUDA, the loading of the model or the
        calculation raises.
    """
    started = time.monotonic()
    job_path = Path(job_path)
    directory = job_path.resolve().parent
    job = json.loads(job_path.read_text(encoding='utf-8'))
    calculation = job['calculation']
    if calculation not in ('scf', 'relax'):
        raise ValueError(f'calculation {calculation!r}: only "scf" and "relax"')
    deadline = math.inf if job['max_seconds'] is None else started + job['max_seconds']
    device = check_cuda()
    atoms = read(directory / STRUCTURE_FILE, format='extxyz')
    atoms.calc = load_calculator(job['model'])
    last_structure = directory / LAST_STRUCTURE_FILE

    def on_step():
        save_last_structure(atoms, last_structure)

    result = {'ok': True, 'calculation': calculation}
    if calculation == 'scf':
        on_step()
        result.update(geometry='n/a', stop_reason='completed', steps=None, fallback=None)
    else:
        with open(directory / LOG_FILE, 'w', encoding='utf-8') as log:
            result.update(relax(atoms, job['fmax'], job['maxsteps'], deadline, log, on_step))
    result.update(final_values(atoms))
    result.update(symbols=atoms.get_chemical_symbols(), cell=atoms.cell[:].tolist())
    result['elapsed'] = time.monotonic() - started
    versions = {'python': sys.version.split()[0], 'ase': ase.__version__,
                job['package']: version_of(job['package']), 'torch': device['torch']}
    result['observed'] = {'versions': versions, 'device': device['device'],
                          'model_path': os.path.realpath(job['model_path'])}
    return result


def write_result(path, result):
    """Write ``result.json`` atomically (the file says that the executor ended)."""
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def main(argv=None):
    """Command line: ``ase_run.py job.json``. Returns the exit code."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print('usage: ase_run.py job.json', file=sys.stderr)
        return 2
    result_path = Path(argv[0]).resolve().parent / RESULT_FILE
    # A result left by an earlier execution must not stand for this one.
    result_path.unlink(missing_ok=True)
    try:
        result = run_job(argv[0])
    except Exception as error:  # the error is the result: the workflow reads result.json
        traceback.print_exc()
        result = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
    write_result(result_path, result)
    print(f"{'done' if result['ok'] else 'failed'}: {result_path}")
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
