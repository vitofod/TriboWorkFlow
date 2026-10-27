# Author: Vito Foderà
# Created: 2026-10-27

"""Groups of jobs: manifest, jobscript and submission.

The calculations of a group are spread over several jobs, all submitted from
the same jobscript. Each job runs one worker, which goes through the list of
the group and takes the calculations one at a time (``worker.py``). A group
lives in ``<run>/jobs/<name>/`` (see ``layout.py``):

- ``manifest.json``: the calculations, in the order the workers take them,
  and how to run them (scheduler, command, walltime, time margin);
- ``jobscript.sh``: the jobscript of the user, copied unchanged, with one
  line added at the end that starts the worker on the manifest;
- ``submitted.json``: the ids of the jobs submitted so far.

The jobscript of the user is never changed. The jobs of a group differ only
by their ids.

Notes
-----
The copy of the jobscript with the launch line appended follows Xsorb
(``xsorb/io/jobs.py``, CC BY 4.0); one jobscript submitted once per group of
calculations, with a worker that loops over them, follows SCS
(``labelling/QuantumEspresso/qe.py``, ``labelling/labelling.py``).
"""

import logging
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

from triboworkflow.core.guard import hpc_only
from triboworkflow.core.store.atomic import read_json, write_json
from triboworkflow.core.store.record import check_keys, check_name, split_calc_id

from .layout import INPUT_FILE, calc_dir, job_dir

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1
MANIFEST_FILE = 'manifest.json'
JOBSCRIPT_FILE = 'jobscript.sh'
SUBMITTED_FILE = 'submitted.json'

# Module started by the line added to the jobscript.
WORKER_MODULE = 'triboworkflow.execution.worker'

_MANIFEST_KEYS = ('schema_version', 'name', 'scheduler', 'run_command', 'walltime',
                  'max_seconds_margin', 'calculations')


class JobError(ValueError):
    """A group of jobs that cannot be written, read or submitted."""


def _seconds(value, where):
    if type(value) is not int or value <= 0:
        raise JobError(f'{where}: {value!r} is not an integer > 0')
    return value


@dataclass(frozen=True)
class Manifest:
    """Calculations of a group of jobs and how to run them.

    Parameters
    ----------
    name : str
        Name of the group, i.e. of its directory in ``<run>/jobs/``.
    scheduler : str
        Scheduler the jobs run under; it tells the worker where to read its
        job id.
    run_command : str
        Command up to the input file, e.g. ``mpirun -np 32 pw.x -nk 4``.
    walltime : int
        Walltime of each job, in seconds.
    max_seconds_margin : int
        Seconds kept free at the end of the walltime; < ``walltime``.
    calculations : tuple of (str, str)
        ``(system, calc_id)`` of each calculation, in the order the workers
        take them; no repetitions.

    Raises
    ------
    JobError
        If a field is not valid.
    """

    name: str
    scheduler: str
    run_command: str
    walltime: int
    max_seconds_margin: int
    calculations: tuple

    def __post_init__(self):
        try:
            check_name(self.name, 'name')
        except ValueError as error:
            raise JobError(str(error)) from None
        for field_name in ('scheduler', 'run_command'):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise JobError(f'{field_name}: a non-empty string is required, not {value!r}')
        _seconds(self.walltime, 'walltime')
        _seconds(self.max_seconds_margin, 'max_seconds_margin')
        if self.max_seconds_margin >= self.walltime:
            raise JobError(f'max_seconds_margin: must be < walltime ({self.walltime}), '
                           f'got {self.max_seconds_margin}')
        calculations = tuple(self.calculations)
        if any(not isinstance(c, (tuple, list)) or len(c) != 2 for c in calculations):
            raise JobError(f'calculations: {calculations!r} is not a sequence of '
                           '(system, calc_id)')
        calculations = tuple(tuple(c) for c in calculations)
        if not calculations:
            raise JobError('calculations: the group has no calculations')
        seen = set()
        for calc in calculations:
            system, calc_id = calc
            try:
                check_name(system, 'system')
                split_calc_id(calc_id)
            except ValueError as error:
                raise JobError(f'calculations: {error}') from None
            if calc in seen:
                raise JobError(f'calculations: {system}/{calc_id} is listed twice')
            seen.add(calc)
        object.__setattr__(self, 'calculations', calculations)

    @classmethod
    def from_execution(cls, name, calculations, execution):
        """Manifest of a group from the ``[execution]`` settings.

        Parameters
        ----------
        name : str
            Name of the group.
        calculations : iterable of (str, str)
            ``(system, calc_id)`` of each calculation.
        execution : triboworkflow.core.settings.schema.ExecutionParams
            Settings of ``[execution]``.
        """
        return cls(name=name, scheduler=execution.scheduler, run_command=execution.run_command,
                   walltime=execution.walltime, max_seconds_margin=execution.max_seconds_margin,
                   calculations=tuple(calculations))

    def to_dict(self):
        """JSON object of the manifest."""
        return {'schema_version': MANIFEST_VERSION, 'name': self.name,
                'scheduler': self.scheduler, 'run_command': self.run_command,
                'walltime': self.walltime, 'max_seconds_margin': self.max_seconds_margin,
                'calculations': [{'system': s, 'calc_id': c} for s, c in self.calculations]}

    @classmethod
    def from_dict(cls, data, source=None):
        """Manifest from its JSON object; ``source`` prefixes the error messages."""
        try:
            check_keys(data, _MANIFEST_KEYS, 'manifest')
            version = data['schema_version']
            if type(version) is not int or version != MANIFEST_VERSION:
                raise JobError(f'schema_version: {version!r}; this version reads '
                               f'{MANIFEST_VERSION}')
            if not isinstance(data['calculations'], list):
                raise JobError('calculations: a JSON array is required')
            calculations = []
            for calc in data['calculations']:
                check_keys(calc, ('system', 'calc_id'), 'calculations')
                calculations.append((calc['system'], calc['calc_id']))
            return cls(name=data['name'], scheduler=data['scheduler'],
                       run_command=data['run_command'], walltime=data['walltime'],
                       max_seconds_margin=data['max_seconds_margin'],
                       calculations=tuple(calculations))
        except ValueError as error:  # JobError, and StoreError from check_keys
            raise JobError(f'{source}: {error}' if source else str(error)) from None


def read_manifest(path):
    """Read a manifest; the run is the directory three levels above it.

    Returns
    -------
    manifest : Manifest
    run_dir : pathlib.Path
        Directory of the run (``<run>/jobs/<name>/manifest.json``).
    """
    path = Path(path).resolve()
    manifest = Manifest.from_dict(read_json(path), source=path)
    if path.parent.name != manifest.name:
        raise JobError(f'{path}: manifest of the group {manifest.name!r} found in '
                       f'{path.parent}')
    return manifest, path.parent.parent.parent


def jobscript_path(settings):
    """Jobscript of ``[execution]``, relative to the folder of the settings file.

    Without a settings file (settings built from a dict) a relative path is
    relative to the current directory.
    """
    path = Path(settings.execution.jobscript)
    if not path.is_absolute() and settings.source is not None:
        path = Path(settings.source).parent / path
    return path


def worker_line(manifest_path, python=None):
    """Line added to the jobscript: the worker started on the manifest.

    Parameters
    ----------
    manifest_path : path-like
        Manifest of the group, written as an absolute path.
    python : str or None, optional
        Python interpreter of the worker; the one running this code by
        default.
    """
    python = python or sys.executable
    manifest = os.path.abspath(manifest_path)
    return f'{shlex.quote(python)} -m {WORKER_MODULE} {shlex.quote(manifest)}'


def write_job(run_dir, manifest, jobscript, python=None):
    """Write the directory of a group of jobs: manifest and jobscript.

    Parameters
    ----------
    run_dir : path-like
        Directory of the run.
    manifest : Manifest
        Calculations of the group and how to run them.
    jobscript : path-like
        Jobscript of the user; it is copied, never changed.
    python : str or None, optional
        Python interpreter of the worker (see ``worker_line``).

    Returns
    -------
    pathlib.Path
        Directory of the group, ``<run>/jobs/<name>``.

    Raises
    ------
    JobError
        If a calculation has no ``input.pwi``, the jobscript is missing or
        the group already exists.
    """
    missing = [f'{s}/{c}' for s, c in manifest.calculations
               if not (calc_dir(run_dir, s, c) / INPUT_FILE).is_file()]
    if missing:
        raise JobError(f'calculations without {INPUT_FILE} in {run_dir}: {", ".join(missing)}')
    jobscript = Path(jobscript)
    if not jobscript.is_file():
        raise JobError(f'jobscript {jobscript} not found')
    template = jobscript.read_bytes()

    path = job_dir(run_dir, manifest.name)
    try:
        path.mkdir(parents=True)
    except FileExistsError:
        raise JobError(f'group of jobs {path} already exists') from None
    write_json(path / MANIFEST_FILE, manifest.to_dict())
    line = worker_line(path / MANIFEST_FILE, python)
    (path / JOBSCRIPT_FILE).write_bytes(template + os.fsencode(f'\n{line}\n'))
    return path


def submit_jobs(path, scheduler, count):
    """Submit the jobscript of a group once per worker.

    ``count`` workers are asked for, but never more than the calculations of
    the group: a worker without a calculation to take would only exit. The
    reduction is logged. Each job id is added to ``submitted.json`` as soon
    as the job is submitted. Submissions run from the directory of the
    group, where the scheduler writes the logs of the jobs. In a dry run
    nothing is submitted or written, and the commands are left in
    ``scheduler.commands``.

    Parameters
    ----------
    path : path-like
        Directory of the group (``write_job``).
    scheduler : triboworkflow.execution.scheduler.JobScheduler
        Scheduler of the jobs.
    count : int
        Number of workers chosen by the user (``groups`` of ``[execution]``).

    Returns
    -------
    list
        Job ids (str), or None for each job in a dry run; one per job
        submitted, i.e. ``min(count, number of calculations)``.

    Raises
    ------
    NotOnHPCError
        If the submission command of the scheduler is not found.
    JobSchedulerError
        If a submission fails; the jobs submitted before stay in
        ``submitted.json``.
    """
    path = Path(path)
    if type(count) is not int or count <= 0:
        raise JobError(f'number of jobs: {count!r} is not an integer > 0')
    jobscript = path / JOBSCRIPT_FILE
    if not jobscript.is_file():
        raise JobError(f'{jobscript} not found: write the group first')
    manifest, _ = read_manifest(path / MANIFEST_FILE)
    jobs = min(count, len(manifest.calculations))
    if jobs < count:
        logger.warning('Group %s: %d jobs submitted instead of %d, one per calculation',
                       manifest.name, jobs, count)
    submitted = path / SUBMITTED_FILE
    job_ids = []
    for _ in range(jobs):
        with hpc_only(scheduler.scheduler_name):
            job_id = scheduler.submit_job(str(jobscript), cwd=path)
        job_ids.append(job_id)
        if not scheduler.dry_run:
            data = read_json(submitted) if submitted.exists() else {'job_ids': []}
            data['job_ids'].append(job_id)
            write_json(submitted, data)
    return job_ids
