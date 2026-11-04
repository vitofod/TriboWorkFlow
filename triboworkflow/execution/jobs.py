# Author: Vito Foderà
# Created: 2026-10-27

"""Groups of jobs: manifest, jobscript and submission.

The calculations of a group are spread over several jobs, all submitted from
the same jobscript. Each job runs one worker, which goes through the list of
the group and takes the calculations one at a time (``worker.py``). A group
lives in ``<run>/jobs/<name>/`` (see ``layout.py``):

- ``manifest.json``: the calculations, in the order the workers take them,
  each with the attempt to run (``Launch``), and how to run them
  (scheduler, command, digest of the jobscript of the user, walltime, time
  margin);
- ``jobscript.sh``: the jobscript of the user, copied unchanged, with one
  line added at the end that starts the worker on the manifest;
- ``submitted.json``: the ids of the jobs submitted so far and, while a
  submission runs, the time it started (``pending``). A ``pending`` left by
  an interruption tells that a job may exist without its id;
- ``withdrawn.json``: written when the group is withdrawn before any of its
  jobs was submitted (``JobManager.withdraw``); a withdrawn group is never
  submitted and its launches no longer count for any calculation.

The jobscript of the user is never changed. The jobs of a group differ only
by their ids.

Each calculation of a manifest names the attempt the group runs: its number
``n``, why it runs (``LAUNCH_KINDS``), the level of the remedies and the
parameters changed. A worker runs a calculation only while its record has
``n - 1`` attempts, so every attempt belongs to one group: a group whose
attempt was already made, e.g. a late job of a group that was replaced, finds
nothing to run. A new attempt of a calculation always goes in a new group.

Notes
-----
The copy of the jobscript with the launch line appended follows Xsorb
(``xsorb/io/jobs.py``, CC BY 4.0); one jobscript submitted once per group of
calculations, with a worker that loops over them, follows SCS
(``labelling/QuantumEspresso/qe.py``, ``labelling/labelling.py``).
"""

import contextlib
import copy
import datetime
import logging
import os
import shlex
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

from triboworkflow.core.guard import NotOnHPCError, hpc_only
from triboworkflow.core.store.atomic import check_json_value, read_json, write_json
from triboworkflow.core.store.fingerprint import bytes_digest, check_fingerprint
from triboworkflow.core.store.record import check_keys, check_name, split_calc_id

from .layout import INPUT_FILE, calc_dir, job_dir
from .scheduler import JobSchedulerError

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 3
MANIFEST_FILE = 'manifest.json'
JOBSCRIPT_FILE = 'jobscript.sh'
SUBMITTED_FILE = 'submitted.json'
WITHDRAWN_FILE = 'withdrawn.json'

# Key of submitted.json with the start of a submission whose outcome is not
# recorded yet.
PENDING_KEY = 'pending'

# Module started by the line added to the jobscript.
WORKER_MODULE = 'triboworkflow.execution.worker'

# Why an attempt runs: the first one; again with the input of the attempt it
# repeats, after a crash or a kill; at the next level of the remedies, to help
# convergence; asked for by the user.
LAUNCH_KINDS = ('initial', 'unchanged', 'remedy', 'manual')

_MANIFEST_KEYS = ('schema_version', 'name', 'scheduler', 'run_command', 'jobscript_digest',
                  'walltime', 'max_seconds_margin', 'calculations')
_LAUNCH_KEYS = ('system', 'calc_id', 'n', 'kind', 'level', 'values', 'initial_values', 'restart')


class JobError(ValueError):
    """A group of jobs that cannot be written, read or submitted."""


def _seconds(value, where):
    if type(value) is not int or value <= 0:
        raise JobError(f'{where}: {value!r} is not an integer > 0')
    return value


@dataclass(frozen=True)
class Launch:
    """One calculation of a group: the attempt to run and how.

    Parameters
    ----------
    system : str
        Name of the system.
    calc_id : str
        ``<step>/<label>`` of the calculation.
    n : int
        Number of the attempt, counted from 1. A worker runs it only while
        the record of the calculation has ``n - 1`` attempts.
    kind : str
        Why the attempt runs, one of ``LAUNCH_KINDS``: ``'initial'`` for
        attempt 1 only; the others for a later attempt.
    level : int
        Level of the remedies of the attempt, from 1 (``input.pwi`` as
        prepared); ``'initial'`` runs at level 1, ``'remedy'`` at level 2 or
        more. An unchanged or manual launch keeps the level, and so the
        input, of the attempt it repeats.
    values : dict, optional
        Values of the parameters of the remedies at ``level``, by name, that
        replace those of ``input.pwi`` (some may be equal to them); empty at
        level 1, not empty above.
    initial_values : dict, optional
        Effective values of the same parameters at level 1, read from the
        summary of pw.x, from which the levels are computed; empty at level 1,
        not empty above.
    restart : bool, optional
        True if the attempt continues the previous one from the files that
        pw.x left in the output directory (``restart_mode = 'restart'``);
        False, the default, if it starts from scratch. Always False for the
        first attempt.

    Raises
    ------
    JobError
        If a field is not valid.
    """

    system: str
    calc_id: str
    n: int
    kind: str
    level: int
    values: dict = field(default_factory=dict)
    initial_values: dict = field(default_factory=dict)
    restart: bool = False

    def __post_init__(self):
        where = f'calculations: {self.system}/{self.calc_id}'
        try:
            check_name(self.system, 'system')
            split_calc_id(self.calc_id)
        except ValueError as error:
            raise JobError(f'calculations: {error}') from None
        for name in ('n', 'level'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise JobError(f'{where}: {name} = {value!r} is not an integer >= 1')
        if self.kind not in LAUNCH_KINDS:
            raise JobError(f'{where}: kind = {self.kind!r} is not one of '
                           f'{", ".join(LAUNCH_KINDS)}')
        if self.kind == 'initial' and (self.n, self.level) != (1, 1):
            raise JobError(f'{where}: an initial launch is attempt 1 at level 1, not attempt '
                           f'{self.n} at level {self.level}')
        if self.kind != 'initial' and self.n == 1:
            raise JobError(f'{where}: attempt 1 is an initial launch, not {self.kind!r}')
        if self.kind == 'remedy' and self.level < 2:
            raise JobError(f'{where}: a remedy runs at level 2 or more, not {self.level}')
        for name in ('values', 'initial_values'):
            value = getattr(self, name)
            if not isinstance(value, dict):
                raise JobError(f'{where}: {name} must be a JSON object, not {value!r}')
            try:
                check_json_value(value, f'{where}: {name}')
            except ValueError as error:
                raise JobError(str(error)) from None
            if self.level == 1 and value:
                raise JobError(f'{where}: at level 1 the input runs as prepared, '
                               f'{name} = {value!r}')
            if self.level > 1 and not value:
                raise JobError(f'{where}: {name} is empty at level {self.level}')
            object.__setattr__(self, name, copy.deepcopy(value))
        if type(self.restart) is not bool:
            raise JobError(f'{where}: restart = {self.restart!r} is not true or false')
        if self.restart and self.n == 1:
            raise JobError(f'{where}: attempt 1 has no previous attempt to continue')

    @classmethod
    def initial(cls, system, calc_id):
        """Launch of the first attempt of a calculation."""
        return cls(system, calc_id, 1, 'initial', 1)

    @property
    def calc(self):
        """``(system, calc_id)``."""
        return self.system, self.calc_id

    def remedies(self):
        """What the worker writes in ``remedies`` of the attempt: kind, level, values, restart."""
        return {'kind': self.kind, 'level': self.level, 'values': copy.deepcopy(self.values),
                'initial_values': copy.deepcopy(self.initial_values), 'restart': self.restart}

    def to_dict(self):
        """JSON object of the launch."""
        return {'system': self.system, 'calc_id': self.calc_id, 'n': self.n, 'kind': self.kind,
                'level': self.level, 'values': copy.deepcopy(self.values),
                'initial_values': copy.deepcopy(self.initial_values), 'restart': self.restart}

    @classmethod
    def from_dict(cls, data):
        """Launch from its JSON object, with exactly the keys of ``to_dict``."""
        check_keys(data, _LAUNCH_KEYS, 'calculations')
        return cls(**data)


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
    calculations : tuple of Launch
        The attempt to run for each calculation, in the order the workers
        take them; each calculation once.
    jobscript_digest : str or None, optional
        ``file_digest`` of the jobscript of the user, which sets the
        resources of the jobs; ``write_job`` sets it from the jobscript it
        copies.

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
    jobscript_digest: str | None = None

    def __post_init__(self):
        try:
            check_name(self.name, 'name')
        except ValueError as error:
            raise JobError(str(error)) from None
        for field_name in ('scheduler', 'run_command'):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise JobError(f'{field_name}: a non-empty string is required, not {value!r}')
        if self.jobscript_digest is not None:
            try:
                check_fingerprint(self.jobscript_digest, 'jobscript_digest')
            except ValueError as error:
                raise JobError(str(error)) from None
        _seconds(self.walltime, 'walltime')
        _seconds(self.max_seconds_margin, 'max_seconds_margin')
        if self.max_seconds_margin >= self.walltime:
            raise JobError(f'max_seconds_margin: must be < walltime ({self.walltime}), '
                           f'got {self.max_seconds_margin}')
        calculations = tuple(self.calculations)
        if any(not isinstance(c, Launch) for c in calculations):
            raise JobError(f'calculations: {calculations!r} is not a sequence of Launch')
        if not calculations:
            raise JobError('calculations: the group has no calculations')
        seen = set()
        for launch in calculations:
            if launch.calc in seen:
                raise JobError(f'calculations: {launch.system}/{launch.calc_id} is listed twice')
            seen.add(launch.calc)
        object.__setattr__(self, 'calculations', calculations)

    @classmethod
    def from_execution(cls, name, calculations, execution):
        """Manifest of a group from the ``[execution]`` settings.

        Parameters
        ----------
        name : str
            Name of the group.
        calculations : iterable of Launch
            The attempt to run for each calculation.
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
                'jobscript_digest': self.jobscript_digest,
                'walltime': self.walltime, 'max_seconds_margin': self.max_seconds_margin,
                'calculations': [launch.to_dict() for launch in self.calculations]}

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
            calculations = [Launch.from_dict(calc) for calc in data['calculations']]
            return cls(name=data['name'], scheduler=data['scheduler'],
                       run_command=data['run_command'], walltime=data['walltime'],
                       max_seconds_margin=data['max_seconds_margin'],
                       calculations=tuple(calculations),
                       jobscript_digest=data['jobscript_digest'])
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
    missing = [f'{c.system}/{c.calc_id}' for c in manifest.calculations
               if not (calc_dir(run_dir, c.system, c.calc_id) / INPUT_FILE).is_file()]
    if missing:
        raise JobError(f'calculations without {INPUT_FILE} in {run_dir}: {", ".join(missing)}')
    jobscript = Path(jobscript)
    if not jobscript.is_file():
        raise JobError(f'jobscript {jobscript} not found')
    template = jobscript.read_bytes()
    manifest = replace(manifest, jobscript_digest=bytes_digest(template))

    path = job_dir(run_dir, manifest.name)
    try:
        path.mkdir(parents=True)
    except FileExistsError:
        raise JobError(f'group of jobs {path} already exists') from None
    write_json(path / MANIFEST_FILE, manifest.to_dict())
    line = worker_line(path / MANIFEST_FILE, python)
    (path / JOBSCRIPT_FILE).write_bytes(template + os.fsencode(f'\n{line}\n'))
    return path


def is_withdrawn(path):
    """Whether the group in directory ``path`` was withdrawn (``withdrawn.json``)."""
    return (Path(path) / WITHDRAWN_FILE).is_file()


def surely_not_created(error):
    """Whether a failed submission surely created no job.

    True when the command is missing (``NotOnHPCError``) or the scheduler
    refused the job (``JobSubmissionError.accepted`` False, or an error that
    does not tell); False when the job may exist (``accepted`` None).
    """
    return isinstance(error, NotOnHPCError) or getattr(error, 'accepted', False) is not None


def _local_stamp():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def submit_jobs(path, scheduler, count, now=None):
    """Submit the jobscript of a group once per worker.

    ``count`` workers are asked for, but never more than the calculations of
    the group: a worker without a calculation to take would only exit. The
    reduction is logged. Before each submission ``submitted.json`` gets the
    time it starts (``pending``); the job id replaces it as soon as the job
    is submitted, so that an interruption in between leaves the trace of a
    job that may exist. A failure that surely created no job removes it; a
    failure after which the job may exist leaves it to the caller
    (``JobManager``), which records the submission as one of outcome not
    known. Submissions run from the directory of the group, where the
    scheduler writes the logs of the jobs. In a dry run nothing is submitted
    or written, and the commands are left in ``scheduler.commands``.

    Parameters
    ----------
    path : path-like
        Directory of the group (``write_job``).
    scheduler : triboworkflow.execution.scheduler.JobScheduler
        Scheduler of the jobs.
    count : int
        Number of workers chosen by the user (``groups`` of ``[execution]``).
    now : callable or None, optional
        Time of a submission that starts, as an ISO 8601 string; the local
        time by default.

    Returns
    -------
    list
        Job ids (str), or None for each job in a dry run; one per job
        submitted, i.e. ``min(count, number of calculations)``.

    Raises
    ------
    JobError
        If ``count`` is not valid, the group is not written, or
        ``submitted.json`` holds a submission whose outcome is not recorded.
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
    stamp = now or _local_stamp
    job_ids = []
    for _ in range(jobs):
        data = None
        if not scheduler.dry_run:
            existed = submitted.exists()
            data = read_json(submitted) if existed else {'job_ids': []}
            if data.get(PENDING_KEY) is not None:
                raise JobError(f'{submitted}: the submission started at {data[PENDING_KEY]} has '
                               'no outcome recorded; the job manager records it first (recover)')
            write_json(submitted, {**data, PENDING_KEY: stamp()})
        try:
            with hpc_only(scheduler.scheduler_name):
                job_id = scheduler.submit_job(str(jobscript), cwd=path)
        except (NotOnHPCError, JobSchedulerError) as error:
            if data is not None and surely_not_created(error):
                if existed or data['job_ids']:
                    write_json(submitted, data)
                else:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(submitted)
            raise
        job_ids.append(job_id)
        if data is not None:
            write_json(submitted, {**data, 'job_ids': data['job_ids'] + [job_id]})
    return job_ids
