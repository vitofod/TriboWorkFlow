# Author: Vito Foderà
# Created: 2026-10-27

"""Worker of a group of jobs: runs the calculations of a manifest one at a time.

Each job of a group runs one worker on the manifest of the group
(``jobs.py``)::

    python -m triboworkflow.execution.worker <run>/jobs/<name>/manifest.json

The worker goes once through the calculations, in the order of the manifest.
Each one names the attempt ``n`` to run (``jobs.Launch``): the worker runs it
only while the record has ``n - 1`` attempts, and skips it when attempt ``n``
was already made, by another worker of the group or by an earlier group, so
that a late job of a replaced group runs nothing. Taking a calculation and
running it are seven steps, and an eighth for a calculation that also gives
the charge density, each completed before the next begins:

1. the claim file ``claim`` is created with ``O_CREAT | O_EXCL``: exactly one
   worker succeeds, the others skip the calculation; the record is then read
   again, and the claim removed if attempt ``n`` was made in the meantime.
   The input of the attempt is prepared and checked in memory, and nothing
   is changed on disk before the claim holds the number of the attempt, the
   job id, how many times the scheduler restarted the job
   (``restart_count``, see below), the host, the pid and the time; a check
   that fails removes the claim and leaves the calculation free;
2. the record gets attempt ``n`` with the job, the host, the start time, the
   time limit, the run command, the digest of the jobscript (both from the
   manifest, compared by a later restart) and the launch (kind, level,
   parameters changed and restart, in ``remedies``), and the process
   ``running``; the results,
   ``observed`` and the engine version of the previous run are cleared. An
   attempt from scratch after the first gets the removal of the output
   directories as ``'started'`` (``Attempt.outdir_removal``);
3. for such an attempt, the directories that pw.x writes, ``outdir`` and
   ``wfcdir`` when given, are removed (``clear_scratch``), since pw.x from
   scratch leaves some files of the previous runs that a later restart could
   read. Both must be inside the calculation directory, checked again right
   before; the record then gets ``'removed'`` or ``'absent'``. A removal that
   fails gives ``'failed'``: the attempt ends without running, with the
   process ``crashed``, and the user decides. A worker stopped during the
   removal leaves ``'started'``, and the next attempt from scratch removes
   the directories again;
4. ``run.pwi`` is written: ``input.pwi`` with ``max_seconds`` set to the
   time limit of the attempt and, above level 1 of the remedies, the values
   of the level (``nstep`` in ``&CONTROL``, ``mixing_beta`` in
   ``&ELECTRONS``), each one replacing its line or added at the start of its
   namelist (``with_value``), and ``restart_mode = 'restart'`` for a launch
   that continues the previous attempt from the files of the output
   directory (``policy.can_restart``); ``input.pwi`` is never changed;
5. ``output.pwo`` is created with ``O_EXCL``: an output already there belongs
   to an attempt that was not archived, and nothing runs;
6. ``<run_command> -in run.pwi >> output.pwo`` runs in the calculation
   directory, with bash;
7. the record gets the end time, the exit code, the digests of the two
   files, the states read from the output, the reason of the stop, whether
   pw.x closed the run and the version it printed. An output without a
   closing line gives the process ``crashed``, since the worker saw the
   command end; so does an empty output. An output that cannot be read as
   pw.x output leaves the states unknown. When the calculation is over with
   the SCF and the geometry converged, its results are read
   (``parsing.py``): the record gets ``parsing``, the results and
   ``observed``, and ``structure_out.extxyz`` is written; an output that
   cannot be parsed leaves the reason in the attempt. A version of pw.x
   outside 7.1.x-7.3.x is read with a warning in the log;
8. for a calculation that gives the charge density (``Record.density``),
   and only when its output was parsed (pw.x over, SCF and geometry
   converged, results read): the attempt gets the post-processing
   (``DensityRun``: pp.x, ``pp_command`` of the manifest, ``input.ppi``),
   ``output.ppo`` is created with ``O_EXCL``, ``<pp_command> -in input.ppi
   >> output.ppo`` runs in the calculation directory, with bash, and the
   attempt and the record get the exit code, the digests of ``output.ppo``
   and ``density.filplot`` and the state ``density``: ``'ok'`` when pp.x
   closed its run without errors and the file is complete and agrees with
   the final structure (``parsing.density_problem``), ``'failed'`` with the
   reason otherwise. A failure is never run again automatically: the
   calculation goes to the user. pp.x runs in the time kept free at the end
   of the walltime (``max_seconds_margin``), with no limit of its own; then
   the worker takes no more calculations if no time is left.

The claim, ``run.pwi`` and ``output.pwo`` (and ``output.ppo`` and
``density.filplot``) stay in place: they are moved to ``attempts/<n>/`` and
the claim removed only by the archiving of the attempt (``archive.py``),
before a new attempt. A worker killed with its job (e.g. at the walltime)
leaves the attempt with the process ``running``, or, while pp.x runs, with
the post-processing begun and its state unknown: what happened is then known
from the scheduler and the files, not from the worker
(``reconcile.py``).

Time limit: ``max_seconds`` of each attempt is the walltime minus the time
since the worker started minus ``max_seconds_margin``, capped by the
``max_seconds`` of ``input.pwi`` when given. When nothing is left, the worker
takes no more calculations.

Restarts: a scheduler may requeue a job and run its jobscript again from the
start with the same id. The claim records ``restart_count``, the number of
restarts the scheduler gives to the run that wrote it, so that this run can
be told apart from a later one of the same job. It is the integer value of
the variable of ``RESTART_COUNT_VARIABLES``, the value as written if it is
not an integer, None if the variable is not set or the scheduler has none.
Slurm sets ``SLURM_RESTART_COUNT`` only after a restart.

Notes
-----
The loop over the calculations and the launch with bash in the calculation
directory follow SCS (``labelling/QuantumEspresso/run_qe.py``, CC BY 4.0);
the launch line ``-in <input> >> <output>`` follows Xsorb
(``xsorb/dft_codes/definitions.py``, CC BY 4.0).
"""

import argparse
import dataclasses
import datetime
import json
import logging
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import PurePosixPath

from ase.io.espresso import read_fortran_namelist

from triboworkflow.core.settings.schema import RELAUNCH_KEYS
from triboworkflow.core.store import (Attempt, AttemptFile, DensityRun, Status, file_digest,
                                      read_record, write_record)
from triboworkflow.engines.qe.ppi import PROGRAM as PP_PROGRAM
from triboworkflow.engines.qe.results import Parsed
from triboworkflow.engines.qe.status import (CALCULATIONS, PwOutputError, read_pw_status,
                                             record_status, version_warning)

from .jobs import read_manifest
from .layout import (CLAIM_FILE, DENSITY_FILE, INPUT_FILE, OUTPUT_FILE, PP_INPUT_FILE,
                     PP_OUTPUT_FILE, RUN_INPUT_FILE, calc_dir, system_dir)
from .parsing import density_problem, parse_calculation, with_parsed
from .restart import previous_parallelization, restart_outcome, restart_warnings

logger = logging.getLogger(__name__)

# Environment variable with the id of the running job, for each scheduler.
JOB_ID_VARIABLES = {'slurm': 'SLURM_JOB_ID', 'pbs': 'PBS_JOBID', 'torque': 'PBS_JOBID',
                    'lsf': 'LSB_JOBID', 'sge': 'JOB_ID'}
# Environment variable with the number of times the job was restarted or
# requeued, for the schedulers that set one.
RESTART_COUNT_VARIABLES = {'slurm': 'SLURM_RESTART_COUNT'}
_COUNT = re.compile(r'[0-9]+')

# Keys the worker writes in run.pwi, with their namelist: the time limit of
# the attempt, the values of a level of the remedies (RELAUNCH_KEYS) and the
# restart from the files of the output directory.
RUN_KEYS = {'max_seconds': 'control', 'nstep': 'control', 'mixing_beta': 'electrons',
            'restart_mode': 'control'}
# Keys of &CONTROL with the directories that pw.x writes: outdir, and wfcdir
# for the files of each process (outdir when not given).
SCRATCH_KEYS = ('outdir', 'wfcdir')

_NAMELIST_END = re.compile(r'^\s*/\s*$')


class WorkerError(RuntimeError):
    """A calculation that the worker cannot run as prepared."""


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def job_id_from_environment(scheduler, environ=None):
    """Id of the job the worker runs in, or None outside a job.

    Parameters
    ----------
    scheduler : str
        Name of the scheduler; one without a variable in
        ``JOB_ID_VARIABLES`` gives None.
    environ : mapping or None, optional
        Environment; ``os.environ`` by default.
    """
    environ = os.environ if environ is None else environ
    variable = JOB_ID_VARIABLES.get(scheduler)
    return (environ.get(variable) or None) if variable else None


def restart_count_from_environment(scheduler, environ=None):
    """Number of times the scheduler restarted the job the worker runs in.

    Parameters
    ----------
    scheduler : str
        Name of the scheduler; one without a variable in
        ``RESTART_COUNT_VARIABLES`` gives None.
    environ : mapping or None, optional
        Environment; ``os.environ`` by default.

    Returns
    -------
    int or str or None
        The value of the variable as an integer, or as written if it is not
        an integer; None if the variable is not set or empty. Slurm sets it
        only after a restart.
    """
    environ = os.environ if environ is None else environ
    variable = RESTART_COUNT_VARIABLES.get(scheduler)
    value = (environ.get(variable) or None) if variable else None
    return int(value) if value is not None and _COUNT.fullmatch(value) else value


def read_input(path):
    """Calculation and ``max_seconds`` of a prepared pw.x input.

    Returns
    -------
    calculation : str
        ``calculation`` of ``&CONTROL``; ``'scf'``, the default of pw.x,
        when not given.
    max_seconds : int or float or None
        ``max_seconds`` of ``&CONTROL``, None when not given.

    Raises
    ------
    WorkerError
        If the calculation is not ``scf`` or ``relax``.
    """
    with open(path) as f:
        data, _ = read_fortran_namelist(f)
    control = data.get('control', {})
    calculation = control.get('calculation', 'scf')
    if calculation not in CALCULATIONS:
        raise WorkerError(f'{path}: calculation = {calculation!r}; only '
                          f'{", ".join(CALCULATIONS)} run')
    cap = control.get('max_seconds')
    if cap is not None and (type(cap) not in (int, float) or not math.isfinite(cap)):
        raise WorkerError(f'{path}: max_seconds = {cap!r} is not a number')
    return calculation, cap


def with_value(text, key, value):
    """Text of a pw.x input with one key of ``RUN_KEYS`` set in its namelist.

    The line ``<key> = <value>`` is replaced, or added right after the start
    of the namelist when there is none. Every other line is kept as it is;
    comment lines are not looked at. Numbers are written as Python writes
    them, strings between apostrophes.

    Parameters
    ----------
    text : str
        Input with the namelist of the key, holding at most one line that
        assigns the key, alone on its line (a comment may follow).
    key : str
        A key of ``RUN_KEYS``, e.g. ``'max_seconds'`` (in ``&CONTROL``) or
        ``'mixing_beta'`` (in ``&ELECTRONS``).
    value : int, float or str
        Value to write.

    Returns
    -------
    text : str
        The input with the key set.
    replaced : bool
        Whether a line of the key was replaced (False: one was added).

    Raises
    ------
    WorkerError
        If the namelist is missing or not closed, or the key appears in it
        in another form (on a line with other assignments, or more than once).

    Examples
    --------
    >>> with_value("&ELECTRONS\\n  mixing_beta = 0.4\\n/\\n", 'mixing_beta', 0.2)
    ('&ELECTRONS\\n  mixing_beta = 0.2\\n/\\n', True)
    """
    namelist = RUN_KEYS[key]
    start_line = re.compile(rf'^\s*&{namelist}\b', re.IGNORECASE)
    key_word = re.compile(rf'\b{key}\b', re.IGNORECASE)
    key_line = re.compile(rf'^(\s*){key}\s*=\s*[^\s,!]+\s*(!.*)?$', re.IGNORECASE)
    written = f"'{value}'" if isinstance(value, str) else f'{value}'
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if start_line.match(line)), None)
    if start is None:
        raise WorkerError(f'no &{namelist.upper()} namelist in the input')
    end = next((i for i in range(start + 1, len(lines)) if _NAMELIST_END.match(lines[i])), None)
    if end is None:
        raise WorkerError(f'&{namelist.upper()} is not closed by "/"')
    found = [i for i in range(start + 1, end) if key_word.search(lines[i].split('!')[0])]
    if len(found) > 1 or (found and not key_line.match(lines[found[0]].rstrip('\r\n'))):
        shown = '; '.join(lines[i].strip() for i in found)
        raise WorkerError(f'{key} must be alone on one line of &{namelist.upper()}, found: '
                          f'{shown}')
    if found:
        i = found[0]
        indent = key_line.match(lines[i].rstrip('\r\n')).group(1)
        newline = lines[i][len(lines[i].rstrip('\r\n')):] or '\n'
        lines[i] = f'{indent}{key} = {written}{newline}'
    else:
        lines.insert(start + 1, f'  {key} = {written}\n')
    return ''.join(lines), bool(found)


def inside_calculation(path, key, value):
    """Directory named by a key of ``&CONTROL``, checked to lie inside the calculation directory.

    Parameters
    ----------
    path : pathlib.Path
        Directory of the calculation, where pw.x runs.
    key : str
        A key of ``SCRATCH_KEYS``, for the messages.
    value : object
        Value of the key in ``input.pwi``.

    Returns
    -------
    pathlib.Path
        The directory, which may not exist yet.

    Raises
    ------
    WorkerError
        If ``value`` is not a relative path that stays inside the calculation
        directory: not a string, absolute, ``.``, through ``..``, a symbolic
        link or a path through one, a path that resolves elsewhere, or a file
        that is not a directory.
    """
    where = f'{path / INPUT_FILE}: {key} = {value!r}'
    if not isinstance(value, str):
        raise WorkerError(f'{where} is not a directory inside the calculation')
    parts = PurePosixPath(value).parts
    if PurePosixPath(value).is_absolute() or not parts or '..' in parts or '\\' in value:
        raise WorkerError(f'{where} is not a directory inside the calculation')
    target = path
    for part in parts:
        target = target / part
        if target.is_symlink():
            raise WorkerError(f'{where}: {target.relative_to(path).as_posix()} is a symbolic '
                              'link')
    if target.exists() and (not target.is_dir()
                            or not target.resolve().is_relative_to(path.resolve())):
        raise WorkerError(f'{where} is not a directory inside the calculation')
    return target


def scratch_dirs(path, control):
    """Directories that pw.x writes for a calculation, each checked with ``inside_calculation``.

    Parameters
    ----------
    path : pathlib.Path
        Directory of the calculation.
    control : dict
        ``&CONTROL`` of ``input.pwi``.

    Returns
    -------
    list of pathlib.Path
        ``outdir``, then ``wfcdir`` when it is given and is another directory.

    Raises
    ------
    WorkerError
        If ``outdir`` is not given, or either directory is not inside the
        calculation directory.
    """
    if 'outdir' not in control:
        raise WorkerError(f'{path / INPUT_FILE}: no outdir: the files of the previous attempts '
                          'cannot be removed before an attempt from scratch')
    targets = []
    for key in SCRATCH_KEYS:
        if key in control:
            target = inside_calculation(path, key, control[key])
            if target not in targets:
                targets.append(target)
    return targets


def clear_scratch(path, control):
    """Remove the directories that pw.x writes for a calculation before an attempt from scratch.

    pw.x starting from scratch removes only some files of its directories and
    leaves the others, which a later restart could read although they come
    from an older attempt: ``outdir``, and ``wfcdir`` when given, are removed
    as a whole. They are checked again right before (``scratch_dirs``).

    Parameters
    ----------
    path : pathlib.Path
        Directory of the calculation.
    control : dict
        ``&CONTROL`` of ``input.pwi``.

    Returns
    -------
    bool
        True if a directory was removed, False if there was none.

    Raises
    ------
    WorkerError
        As ``scratch_dirs``; nothing is removed.
    OSError
        If a removal fails; part of the files may be gone.
    """
    removed = False
    for target in scratch_dirs(path, control):
        if target.exists():
            shutil.rmtree(target)
            removed = True
    return removed


class Worker:
    """Worker on the calculations of one manifest.

    Parameters
    ----------
    manifest_path : path-like
        Manifest of the group (``<run>/jobs/<name>/manifest.json``).
    clock : callable, optional
        Monotonic clock in seconds; ``time.monotonic`` by default.
    environ : mapping or None, optional
        Environment, read for the job id and the restart count;
        ``os.environ`` by default.

    Attributes
    ----------
    manifest : triboworkflow.execution.jobs.Manifest
    run_dir : pathlib.Path
    job_id, host : str or None
    restart_count : int or str or None
        See ``restart_count_from_environment``.
    """

    def __init__(self, manifest_path, clock=time.monotonic, environ=None):
        self.clock = clock
        self.started = clock()
        self.manifest, self.run_dir = read_manifest(manifest_path)
        self.job_id = job_id_from_environment(self.manifest.scheduler, environ)
        self.restart_count = restart_count_from_environment(self.manifest.scheduler, environ)
        self.host = socket.gethostname()

    def time_left(self):
        """Seconds the next attempt may run: walltime minus elapsed time minus margin."""
        elapsed = self.clock() - self.started
        return self.manifest.walltime - elapsed - self.manifest.max_seconds_margin

    def run(self):
        """Go once through the calculations of the manifest.

        Returns
        -------
        list of (str, str, str)
            ``(system, calc_id, outcome)`` of each calculation looked at, in
            order: ``'ran'``, ``'taken'`` (claimed by another worker),
            ``'skipped'`` (attempt ``n`` already made) or ``'error'`` (not
            run, see the log). The list stops where the time ran out.
        """
        outcomes = []
        for launch in self.manifest.calculations:
            left = math.floor(self.time_left())
            if left <= 0:
                logger.info('No time left for more calculations: stopping before %s/%s',
                            launch.system, launch.calc_id)
                break
            outcomes.append((launch.system, launch.calc_id, self.run_calculation(launch, left)))
        return outcomes

    def run_calculation(self, launch, left):
        """Run attempt ``n`` of one calculation, if it is the next one and no other worker has it.

        Parameters
        ----------
        launch : triboworkflow.execution.jobs.Launch
            Calculation of the manifest and attempt to run.
        left : int
            Seconds left for the attempt (``time_left``).

        Returns
        -------
        str
            ``'ran'``, ``'taken'``, ``'skipped'`` or ``'error'``.
        """
        system, calc_id, n = launch.system, launch.calc_id, launch.n
        path = calc_dir(self.run_dir, system, calc_id)
        claim = path / CLAIM_FILE
        # Attempt n must be the next one: a worker of a replaced group finds it made.
        try:
            if self._next_attempt(launch) is None:
                logger.info('%s/%s: attempt %d already made: nothing to run', system, calc_id, n)
                return 'skipped'
        except Exception as error:
            logger.error('%s/%s: not run: %s', system, calc_id, error)
            return 'error'
        # 1. Claim.
        try:
            fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            logger.info('%s/%s: taken by another worker', system, calc_id)
            return 'taken'
        try:
            record = self._next_attempt(launch)
            if record is not None:
                run_text, calculation, max_seconds, cap, scratch = self._prepare(launch, path,
                                                                                 left)
                if record.density:
                    self._check_density_input(path)
        except Exception as error:  # nothing was done: the calculation is left free
            os.close(fd)
            os.unlink(claim)
            logger.error('%s/%s: not run: %s', system, calc_id, error)
            return 'error'
        if record is None:  # made and archived since the first look: free for a new group
            os.close(fd)
            os.unlink(claim)
            logger.info('%s/%s: attempt %d already made: nothing to run', system, calc_id, n)
            return 'skipped'
        owner = {'n': n, 'job_id': self.job_id, 'restart_count': self.restart_count,
                 'host': self.host, 'pid': os.getpid(), 'since': _now()}
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            fh.write(json.dumps(owner) + '\n')
            fh.flush()
            os.fsync(fh.fileno())

        # 2. Attempt n in the record, process running, with the removal of the
        # output directories started when there is one.
        sdir = system_dir(self.run_dir, system)
        attempt = Attempt(n=n, job_id=self.job_id, host=self.host, started=_now(),
                          max_seconds=max_seconds, max_seconds_cap=cap,
                          run_command=self.manifest.run_command,
                          jobscript_digest=self.manifest.jobscript_digest,
                          input=AttemptFile(RUN_INPUT_FILE), output=AttemptFile(OUTPUT_FILE),
                          status=Status(process='running'), remedies=launch.remedies(),
                          outdir_removal=None if scratch is None else 'started')
        record = dataclasses.replace(record, status=Status(process='running'), results=None,
                                     observed=None,
                                     engine=dataclasses.replace(record.engine, version=None),
                                     attempts=record.attempts + (attempt,))
        write_record(sdir, record)
        # 3. Output directories of the previous attempts, before an attempt from scratch.
        if scratch is not None:
            try:
                removal = 'removed' if clear_scratch(path, scratch) else 'absent'
            except (OSError, WorkerError) as error:
                failed = Status(process='crashed')
                done = dataclasses.replace(attempt, ended=_now(), input=None, output=None,
                                           status=failed, outdir_removal='failed')
                write_record(sdir, dataclasses.replace(record, status=failed,
                                                       attempts=record.attempts[:-1] + (done,)))
                logger.error('%s/%s: attempt %d not run: the output directories of the previous '
                             'attempts were not removed: %s', system, calc_id, n, error)
                return 'error'
            attempt = dataclasses.replace(attempt, outdir_removal=removal)
            write_record(sdir, dataclasses.replace(record,
                                                   attempts=record.attempts[:-1] + (attempt,)))
            if removal == 'removed':
                logger.info('%s/%s: %s removed before attempt %d from scratch', system, calc_id,
                            ', '.join(scratch[key] for key in SCRATCH_KEYS if key in scratch), n)
        # 4. Input of the attempt.
        (path / RUN_INPUT_FILE).write_text(run_text)
        # 5. New output.
        try:
            os.close(os.open(path / OUTPUT_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
        except FileExistsError:
            raise WorkerError(f'{path / OUTPUT_FILE} exists: a previous attempt was not '
                              f'archived; attempt {n} did not run') from None
        # 6. Launch.
        command = f'{self.manifest.run_command} -in {RUN_INPUT_FILE} >> {OUTPUT_FILE}'
        logger.info('%s/%s: attempt %d at level %d, %s, max_seconds = %s%s: %s', system,
                    calc_id, n, launch.level, 'restart' if launch.restart else 'from scratch',
                    max_seconds,
                    ''.join(f', {key} = {value}' for key, value in sorted(launch.values.items())),
                    command)
        exit_code = subprocess.run(command, shell=True, cwd=path, executable='/bin/bash').returncode
        # 7. End of the attempt.
        self._finish(system, calc_id, path, n, calculation, exit_code)
        logger.info('%s/%s: attempt %d ended with exit code %d', system, calc_id, n, exit_code)
        # 8. Charge density, after pw.x.
        self._density(system, calc_id, path, n)
        return 'ran'

    def _check_density_input(self, path):
        """Check that pp.x can run for a calculation that gives the charge density."""
        if self.manifest.pp_command is None:
            raise WorkerError('the calculation gives the charge density and the manifest has no '
                              'pp_command to run pp.x')
        if not (path / PP_INPUT_FILE).is_file():
            raise WorkerError(f'{path / PP_INPUT_FILE} missing: the calculation gives the charge '
                              'density')

    def _density(self, system, calc_id, path, n):
        """Run pp.x after attempt ``n`` of a calculation that gives the charge density.

        Only when the record of the calculation asks for the density and the
        output of attempt ``n`` was parsed. The post-processing is recorded
        before pp.x starts, its outcome after it ends.
        """
        sdir = system_dir(self.run_dir, system)
        record = read_record(sdir, calc_id)
        last = record.attempts[-1]
        if not record.density or last.n != n or record.status.parsing != 'ok':
            return
        command = self.manifest.pp_command
        begun = DensityRun(PP_PROGRAM, command,
                           AttemptFile(PP_INPUT_FILE, file_digest(path / PP_INPUT_FILE)),
                           AttemptFile(PP_OUTPUT_FILE))
        attempt = dataclasses.replace(last, density=begun)
        write_record(sdir, dataclasses.replace(record, attempts=record.attempts[:-1] + (attempt,)))
        try:
            os.close(os.open(path / PP_OUTPUT_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
        except FileExistsError:
            # Not of this attempt: neither file is recorded.
            done = dataclasses.replace(begun, output=None, error=(
                f'{PP_OUTPUT_FILE} exists: an output of pp.x not archived; pp.x did not run'))
        else:
            line = f'{command} -in {PP_INPUT_FILE} >> {PP_OUTPUT_FILE}'
            logger.info('%s/%s: charge density of attempt %d: %s', system, calc_id, n, line)
            exit_code = subprocess.run(line, shell=True, cwd=path,
                                       executable='/bin/bash').returncode
            done = dataclasses.replace(begun, exit_code=exit_code, error=density_problem(path),
                                       **density_files(path))
        error, exit_code = done.error, done.exit_code
        status = dataclasses.replace(attempt.status, density='ok' if error is None else 'failed')
        attempt = dataclasses.replace(attempt, density=done, status=status)
        write_record(sdir, dataclasses.replace(record, status=status,
                                               attempts=record.attempts[:-1] + (attempt,)))
        if error is None:
            logger.info('%s/%s: charge density of attempt %d written (pp.x exit code %s)', system,
                        calc_id, n, exit_code)
        else:
            logger.error('%s/%s: charge density of attempt %d not usable (pp.x exit code %s): %s; '
                         'the calculation needs the user', system, calc_id, n, exit_code, error)

    def _next_attempt(self, launch):
        """Record of the calculation if attempt ``n`` is the next one, None if it was made.

        Raises
        ------
        WorkerError
            If the record has fewer than ``n - 1`` attempts.
        """
        record = read_record(system_dir(self.run_dir, launch.system), launch.calc_id)
        made = len(record.attempts)
        if made >= launch.n:
            return None
        if made < launch.n - 1:
            raise WorkerError(f'the manifest asks for attempt {launch.n}, the record has {made} '
                              'attempts')
        return record

    def _prepare(self, launch, path, left):
        """Text of ``run.pwi``, calculation, time limit, cap and directories to remove.

        ``run.pwi`` is ``input.pwi`` with ``max_seconds`` and the values of
        the level of the launch set (``with_value``), and ``restart_mode =
        'restart'`` for a launch that continues the previous attempt. A
        ``wfcdir`` must be inside the calculation directory
        (``inside_calculation``), since the files of each process belong to
        the calculation. A launch from scratch refuses an ``input.pwi`` that
        asks for a restart itself and, after the first attempt, checks the
        directories to remove (``scratch_dirs``): its ``&CONTROL`` is returned
        for ``clear_scratch``, None for any other launch. Nothing is changed
        on disk.
        """
        other = sorted(set(launch.values) - set(RELAUNCH_KEYS))
        if other:
            raise WorkerError(f'attempt {launch.n} runs at level {launch.level} with '
                              f'{", ".join(other)}: only {", ".join(RELAUNCH_KEYS)} are written '
                              'in the input')
        for name in (RUN_INPUT_FILE, OUTPUT_FILE, PP_OUTPUT_FILE, DENSITY_FILE):
            if (path / name).exists():
                raise WorkerError(f'{path / name} exists without a claim: a previous attempt '
                                  'was not archived')
        calculation, cap = read_input(path / INPUT_FILE)
        with open(path / INPUT_FILE) as f:
            control = read_fortran_namelist(f)[0].get('control', {})
        if 'wfcdir' in control:
            inside_calculation(path, 'wfcdir', control['wfcdir'])
        if not launch.restart:
            mode = control.get('restart_mode', 'from_scratch')
            if mode != 'from_scratch':
                raise WorkerError(f"{path / INPUT_FILE}: restart_mode = {mode!r}, attempt "
                                  f"{launch.n} starts from scratch: whether an attempt "
                                  'continues the previous one is decided by the relaunches')
        # The cap is written as given; the time left is a whole number of seconds.
        max_seconds = cap if cap is not None and cap <= left else left
        if max_seconds <= 0:
            raise WorkerError(f'max_seconds = {max_seconds} in {INPUT_FILE} leaves no time to '
                              'run')
        text = (path / INPUT_FILE).read_text()
        try:
            run_text, replaced = with_value(text, 'max_seconds', max_seconds)
            for key, value in sorted(launch.values.items()):
                run_text, _ = with_value(run_text, key, value)
            if launch.restart:
                run_text, _ = with_value(run_text, 'restart_mode', 'restart')
        except WorkerError as error:
            raise WorkerError(f'{path / INPUT_FILE}: {error}') from None
        if replaced != (cap is not None):
            raise WorkerError(f'{path / INPUT_FILE}: max_seconds is read as {cap!r}, but '
                              f'{"one" if replaced else "no"} max_seconds line was found in '
                              '&CONTROL')
        scratch = None
        if not launch.restart and launch.n > 1:
            scratch_dirs(path, control)
            scratch = control
        return run_text, calculation, max_seconds, cap, scratch

    def _finish(self, system, calc_id, path, n, calculation, exit_code):
        """Write the end of attempt ``n`` in the record, with the results of its output."""
        ended = _now()
        sdir = system_dir(self.run_dir, system)
        record = read_record(sdir, calc_id)
        last = record.attempts[-1] if record.attempts else None
        if last is None or last.n != n:
            raise WorkerError(f'{system}/{calc_id}: the last attempt in the record is not '
                              f'{n}; the end of attempt {n} is not written')
        version = restart = None
        try:
            pw = read_pw_status(path / OUTPUT_FILE, calculation)
        except PwOutputError as error:
            # An empty output: the command ended before pw.x wrote anything. Any
            # other output that cannot be read leaves the states unknown.
            logger.warning('%s/%s: %s', system, calc_id, error)
            empty = (path / OUTPUT_FILE).stat().st_size == 0
            status = Status(process='crashed') if empty else Status()
            stop_reason, clean_stop = None, None
        else:
            status, stop_reason, clean_stop = record_status(pw, 'finished'), pw.stop_reason, \
                pw.clean_stop
            version = pw.version
            note = version_warning(version)
            if note is not None:
                logger.warning('%s/%s: %s', system, calc_id, note)
            if last.remedies.get('restart') is True:
                restart = restart_outcome(pw, previous_parallelization(path, record))
                for note in restart_warnings(restart):
                    logger.warning('%s/%s: attempt %d: %s', system, calc_id, n, note)
        parsed = Parsed(None)
        if status.summary == 'completed':
            try:
                parsed = parse_calculation(path, calculation, record.energy_model)
            except (OSError, PwOutputError) as error:
                parsed = Parsed('failed', reason=f'output not read: {error}')
            if parsed.parsing == 'failed':
                logger.warning('%s/%s: output not parsed: %s', system, calc_id, parsed.reason)
        done = dataclasses.replace(
            last, ended=ended, exit_code=exit_code,
            input=AttemptFile(RUN_INPUT_FILE, file_digest(path / RUN_INPUT_FILE)),
            output=AttemptFile(OUTPUT_FILE, file_digest(path / OUTPUT_FILE)),
            stop_reason=stop_reason, clean_stop=clean_stop, status=status, restart=restart)
        write_record(sdir, with_parsed(record, done, parsed, version))


def density_files(path):
    """``output`` and ``density`` of a ``DensityRun``: the files written, with their digests."""
    return {role: AttemptFile(name, file_digest(path / name)) if (path / name).is_file() else None
            for role, name in (('output', PP_OUTPUT_FILE), ('density', DENSITY_FILE))}


def main(argv=None):
    """Entry point: ``python -m triboworkflow.execution.worker <manifest>``."""
    parser = argparse.ArgumentParser(description='Run the calculations of a group of jobs.')
    parser.add_argument('manifest', help='manifest.json of the group')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    worker = Worker(args.manifest)
    outcomes = worker.run()
    counts = {k: sum(1 for *_, outcome in outcomes if outcome == k)
              for k in ('ran', 'taken', 'skipped', 'error')}
    logger.info('Worker done: %d run, %d taken by other workers, %d already made, %d not run '
                'for an error, %d not reached, of %d calculations', counts['ran'],
                counts['taken'], counts['skipped'], counts['error'],
                len(worker.manifest.calculations) - len(outcomes),
                len(worker.manifest.calculations))
    return 0


if __name__ == '__main__':
    sys.exit(main())
