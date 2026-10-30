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
running it are six steps, each completed before the next begins:

1. the claim file ``claim`` is created with ``O_CREAT | O_EXCL``: exactly one
   worker succeeds, the others skip the calculation; the claim then holds
   the number of the attempt, the job id, how many times the scheduler
   restarted the job (``restart_count``, see below), the host, the pid and
   the time; the record is then read again, and the claim removed if
   attempt ``n`` was made in the meantime;
2. the record gets attempt ``n`` with the job, the host, the start time, the
   time limit and the launch of the manifest (kind, level and parameters
   changed, in ``remedies``), and the process ``running``;
3. ``run.pwi`` is written: ``input.pwi`` with ``max_seconds`` set to the
   time limit of the attempt; ``input.pwi`` is never changed;
4. ``output.pwo`` is created with ``O_EXCL``: an output already there belongs
   to an attempt that was not archived, and nothing runs;
5. ``<run_command> -in run.pwi >> output.pwo`` runs in the calculation
   directory, with bash;
6. the record gets the end time, the exit code, the digests of the two
   files, the states read from the output, the reason of the stop and
   whether pw.x closed the run. An output without a closing line gives the
   process ``crashed``, since the worker saw the command end; so does an
   empty output. An output that cannot be read as pw.x output leaves the
   states unknown.

The claim, ``run.pwi`` and ``output.pwo`` stay in place: they are moved to
``attempts/<n>/`` and the claim removed only by the archiving of the attempt
(``archive.py``), before a new attempt. A worker killed with its job (e.g. at
the walltime) leaves the attempt with the process ``running``: what happened
is then known from the scheduler, not from the worker.

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
Slurm sets ``SLURM_RESTART_COUNT`` only after a restart. A launch above level
1 of the remedies, whose values replace some of ``input.pwi``, is not run:
this version does not write them in the input.

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
import socket
import subprocess
import sys
import time

from ase.io.espresso import read_fortran_namelist

from triboworkflow.core.store import (Attempt, AttemptFile, Status, file_digest, read_record,
                                      write_record)
from triboworkflow.engines.qe.status import (CALCULATIONS, PwOutputError, read_pw_status,
                                             record_status)

from .jobs import read_manifest
from .layout import (CLAIM_FILE, INPUT_FILE, OUTPUT_FILE, RUN_INPUT_FILE, calc_dir,
                     system_dir)

logger = logging.getLogger(__name__)

# Environment variable with the id of the running job, for each scheduler.
JOB_ID_VARIABLES = {'slurm': 'SLURM_JOB_ID', 'pbs': 'PBS_JOBID', 'torque': 'PBS_JOBID',
                    'lsf': 'LSB_JOBID', 'sge': 'JOB_ID'}
# Environment variable with the number of times the job was restarted or
# requeued, for the schedulers that set one.
RESTART_COUNT_VARIABLES = {'slurm': 'SLURM_RESTART_COUNT'}
_COUNT = re.compile(r'[0-9]+')

_CONTROL_START = re.compile(r'^\s*&control\b', re.IGNORECASE)
_NAMELIST_END = re.compile(r'^\s*/\s*$')
_MAX_SECONDS_KEY = re.compile(r'\bmax_seconds\b', re.IGNORECASE)
_MAX_SECONDS_LINE = re.compile(r'^(\s*)max_seconds\s*=\s*[^\s,!]+\s*(!.*)?$', re.IGNORECASE)


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


def with_max_seconds(text, max_seconds):
    """Text of a pw.x input with ``max_seconds`` of ``&CONTROL`` set.

    The line ``max_seconds = <value>`` is replaced, or added right after
    ``&CONTROL`` when there is none. Every other line is kept as it is;
    comment lines are not looked at.

    Parameters
    ----------
    text : str
        Input with one ``&CONTROL`` namelist, holding at most one line that
        assigns ``max_seconds``, alone on its line (a comment may follow).
    max_seconds : int or float
        Value to write, in seconds.

    Returns
    -------
    text : str
        The input with ``max_seconds`` set.
    replaced : bool
        Whether a ``max_seconds`` line was replaced (False: one was added).

    Raises
    ------
    WorkerError
        If there is no ``&CONTROL``, or ``max_seconds`` appears in it in
        another form (on a line with other assignments, or more than once).
    """
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if _CONTROL_START.match(line)), None)
    if start is None:
        raise WorkerError('no &CONTROL namelist in the input')
    end = next((i for i in range(start + 1, len(lines)) if _NAMELIST_END.match(lines[i])), None)
    if end is None:
        raise WorkerError('&CONTROL is not closed by "/"')
    found = [i for i in range(start + 1, end)
             if _MAX_SECONDS_KEY.search(lines[i].split('!')[0])]
    if len(found) > 1 or (found and not _MAX_SECONDS_LINE.match(lines[found[0]].rstrip('\r\n'))):
        shown = '; '.join(lines[i].strip() for i in found)
        raise WorkerError(f'max_seconds must be alone on one line of &CONTROL, found: {shown}')
    if found:
        i = found[0]
        indent = _MAX_SECONDS_LINE.match(lines[i].rstrip('\r\n')).group(1)
        newline = lines[i][len(lines[i].rstrip('\r\n')):] or '\n'
        lines[i] = f'{indent}max_seconds = {max_seconds}{newline}'
    else:
        lines.insert(start + 1, f'  max_seconds = {max_seconds}\n')
    return ''.join(lines), bool(found)


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
                run_text, calculation, max_seconds, cap = self._prepare(launch, path, left)
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

        # 2. Attempt n in the record, process running.
        attempt = Attempt(n=n, job_id=self.job_id, host=self.host, started=_now(),
                          max_seconds=max_seconds, max_seconds_cap=cap,
                          input=AttemptFile(RUN_INPUT_FILE), output=AttemptFile(OUTPUT_FILE),
                          status=Status(process='running'), remedies=launch.remedies())
        write_record(system_dir(self.run_dir, system),
                     dataclasses.replace(record, status=Status(process='running'), results=None,
                                         attempts=record.attempts + (attempt,)))
        # 3. Input of the attempt.
        (path / RUN_INPUT_FILE).write_text(run_text)
        # 4. New output.
        try:
            os.close(os.open(path / OUTPUT_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
        except FileExistsError:
            raise WorkerError(f'{path / OUTPUT_FILE} exists: a previous attempt was not '
                              f'archived; attempt {n} did not run') from None
        # 5. Launch.
        command = f'{self.manifest.run_command} -in {RUN_INPUT_FILE} >> {OUTPUT_FILE}'
        logger.info('%s/%s: attempt %d, max_seconds = %s: %s', system, calc_id, n, max_seconds,
                    command)
        exit_code = subprocess.run(command, shell=True, cwd=path, executable='/bin/bash').returncode
        # 6. End of the attempt.
        self._finish(system, calc_id, path, n, calculation, exit_code)
        logger.info('%s/%s: attempt %d ended with exit code %d', system, calc_id, n, exit_code)
        return 'ran'

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
        """Text of ``run.pwi``, calculation, time limit and cap of an attempt."""
        if launch.changes_input:
            raise WorkerError(f'attempt {launch.n} runs at level {launch.level} with '
                              f'{", ".join(sorted(launch.values))}, which this version does not '
                              'write in the input')
        for name in (RUN_INPUT_FILE, OUTPUT_FILE):
            if (path / name).exists():
                raise WorkerError(f'{path / name} exists without a claim: a previous attempt '
                                  'was not archived')
        calculation, cap = read_input(path / INPUT_FILE)
        # The cap is written as given; the time left is a whole number of seconds.
        max_seconds = cap if cap is not None and cap <= left else left
        if max_seconds <= 0:
            raise WorkerError(f'max_seconds = {max_seconds} in {INPUT_FILE} leaves no time to '
                              'run')
        text = (path / INPUT_FILE).read_text()
        try:
            run_text, replaced = with_max_seconds(text, max_seconds)
        except WorkerError as error:
            raise WorkerError(f'{path / INPUT_FILE}: {error}') from None
        if replaced != (cap is not None):
            raise WorkerError(f'{path / INPUT_FILE}: max_seconds is read as {cap!r}, but '
                              f'{"one" if replaced else "no"} max_seconds line was found in '
                              '&CONTROL')
        return run_text, calculation, max_seconds, cap

    def _finish(self, system, calc_id, path, n, calculation, exit_code):
        """Write the end of attempt ``n`` in the record."""
        ended = _now()
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
        sdir = system_dir(self.run_dir, system)
        record = read_record(sdir, calc_id)
        last = record.attempts[-1] if record.attempts else None
        if last is None or last.n != n:
            raise WorkerError(f'{system}/{calc_id}: the last attempt in the record is not '
                              f'{n}; the end of attempt {n} is not written')
        done = dataclasses.replace(
            last, ended=ended, exit_code=exit_code,
            input=AttemptFile(RUN_INPUT_FILE, file_digest(path / RUN_INPUT_FILE)),
            output=AttemptFile(OUTPUT_FILE, file_digest(path / OUTPUT_FILE)),
            stop_reason=stop_reason, clean_stop=clean_stop, status=status)
        write_record(sdir, dataclasses.replace(record, status=status,
                                               attempts=record.attempts[:-1] + (done,)))


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
