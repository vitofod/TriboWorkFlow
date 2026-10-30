# Author: Vito Foderà
# Created: 2026-10-29

"""Reconciliation of the calculations of a run with their claims, jobs and outputs.

Before anything new is submitted, each calculation is brought up to date
with what its record, its claim, the scheduler and its output tell, and is
classified (``CalcState``):

===============  =========================================================
``state``        Meaning
===============  =========================================================
``new``          no attempt, and no group of jobs for the first one
``pending``      a group of jobs for the next attempt is not over yet
``running``      an attempt runs, or the end of its job is awaited
``held``         the job of the claim is held in the queue: it is kept as it
                 is, never replaced nor released, and no attempt is used
``completed``    the last attempt ended with the SCF and the geometry
                 converged
``unchanged``    the last attempt crashed, was killed or stopped at
                 ``max_seconds``: it may run again with the input unchanged
``remedy``       the last attempt ended without convergence (SCF, or the
                 steps of a relaxation)
``manual``       no automatic relaunch: the user decides
``uncertain``    what happened cannot be established now; the claim is kept
``not_taken``    every group of jobs for the next attempt is over, none of
                 their workers took it, and the last of them made progress
===============  =========================================================

Whether ``unchanged``, ``remedy`` and ``not_taken`` run again, and how, is
decided by the policy of the relaunches; nothing is archived or submitted
here. The states of the jobs are those of the job manager, read after its
cycle; a job the manager does not follow (e.g. one of a submission whose
outcome was not known) is asked to the scheduler.

A group of jobs makes progress when a worker takes at least one of the
attempts it lists. When the last group for the next attempt of a
calculation ended without progress, the calculation is ``manual``, with the
jobs of the group and their states: what stopped them is not deduced.

An attempt recorded as ``running`` is fixed only when the run of the job
that wrote its claim is over:

- the job is in a final state confirmed by the last query;
- the job is in the queue again, to run (``JobScheduler.is_queued``, not
  held): the scheduler requeued it, since a claim is written only while the
  job runs;
- a Slurm job runs with more restarts (``sacct`` field ``Restarts``) than the
  ``restart_count`` of the claim, read as 0 when null, since Slurm sets
  ``SLURM_RESTART_COUNT`` only after a restart: the job was requeued and runs
  again;
- a Slurm job is ``PREEMPTED`` at a query at least ``polling_interval``
  seconds after the first one that reported it so, with the restarts of
  the claim or more: Slurm did not requeue the run that wrote the claim.

Without the restart count (a claim without it, another scheduler, restarts
that cannot be read) the run of a job that is still active is not told
apart from a later one: the attempt stays ``running`` until the job ends,
and a preempted job is ``uncertain``. A claim without a job id (a worker
started by hand, or a scheduler whose job id is not known) is
``uncertain``. A job that the scheduler no longer reports is ``uncertain``,
since its absence does not tell how it ended.

The fixed attempt gets the state of the job as printed (for a requeued job,
the state of its new run), the states read from the output with
``record_status``, with the job ``'killed'`` when the scheduler stopped it
(``KILLED_STATES``, or requeued it) and ``'finished'`` otherwise, the reason
of the stop and the clean stop read from the output, and the digests of
``run.pwi`` and ``output.pwo``. The exit code and the end time stay null:
the worker did not see the command end. An output that pw.x closed decides
the states whatever the job did. A claim of an attempt that the record does
not have (the worker stopped between the claim and the record) gets the
attempt, with the launch of its group, once the run of its job is over.

Outcome of an ended attempt (``attempt_outcome``):

- ``completed``: ``Status.summary`` is ``'completed'``;
- ``remedy``: process ``ended``, stopped with ``scf_not_converged`` or
  ``max_steps``;
- ``unchanged``: process ``ended``, stopped at ``max_seconds``; process
  ``crashed`` or ``killed`` without an error of pw.x, when the worker saw
  the command end (an output without a closing line, or none); when only the
  job tells it, a job requeued or ended in a state of ``RELAUNCH_STATES``;
- ``manual``: the end of the process not known; an error of pw.x
  (``Error in routine``), reported with its routine, code and message read
  from the output, which is kept: the user may run the attempt again; any
  other reason of the stop (``user_request``, ``optimizer_failed``, not
  recognized); a job ended in any other state (e.g. Slurm ``CANCELLED``,
  ``DEADLINE``, ``OUT_OF_MEMORY``, HTCondor removed).
"""

import dataclasses
import datetime
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from triboworkflow.core.guard import hpc_only
from triboworkflow.core.store import (Attempt, AttemptFile, Status, file_digest, read_record,
                                      write_record)
from triboworkflow.engines.qe.status import PwOutputError, read_pw_status, record_status

from .archive import RESTART_COUNT_KEY, read_claim
from .jobs import MANIFEST_FILE, read_manifest
from .layout import (CLAIM_FILE, INPUT_FILE, JOBS_DIR, OUTPUT_FILE, RUN_INPUT_FILE, calc_dir,
                     job_dir, system_dir)
from .manager import HOLD_MESSAGES
from .scheduler import REVERSIBLE_STATES, JobSchedulerError
from .worker import WorkerError, read_input

STATES = ('new', 'pending', 'running', 'held', 'completed', 'unchanged', 'remedy', 'manual',
          'uncertain', 'not_taken')

# Reason of an attempt stopped by an error of pw.x, followed by its details.
PW_ERROR = 'pw.x error'

# Reasons of the stop of a run that pw.x closed without convergence.
REMEDY_STOPS = ('scf_not_converged', 'max_steps')

# Ended states of a job after which an attempt whose output does not tell the
# end may run again unchanged. Slurm: walltime, failure of a node or of its
# boot, preemption without requeue, end of the jobscript with or without an
# error. PBS, Torque and HTCondor: the end of the job, which says nothing more.
RELAUNCH_STATES = {
    'slurm': frozenset({'TIMEOUT', 'NODE_FAIL', 'BOOT_FAIL', 'PREEMPTED', 'COMPLETED',
                        'FAILED'}),
    'pbs': frozenset({'F'}),
    'torque': frozenset({'C'}),
    'htcondor': frozenset({'4'}),
    'condor': frozenset({'4'}),
}

# Ended states in which the scheduler stopped the job: an attempt with a
# truncated output was killed. After the other ended states it crashed.
KILLED_STATES = {
    'slurm': frozenset({'TIMEOUT', 'NODE_FAIL', 'BOOT_FAIL', 'PREEMPTED', 'DEADLINE',
                        'CANCELLED', 'OUT_OF_MEMORY'}),
    'htcondor': frozenset({'3'}),
    'condor': frozenset({'3'}),
}

_PROCESS_FROM_JOB = {'killed': 'killed', 'finished': 'crashed'}


@dataclass(frozen=True)
class CalcState:
    """Where a calculation stands after the reconciliation.

    Attributes
    ----------
    system, calc_id : str
        The calculation.
    state : str
        One of ``STATES`` (see the module docstring).
    attempts : int
        Attempts in the record.
    reason : str or None
        What decided ``unchanged``, ``remedy`` or ``manual`` (the reason of
        the stop with the details of an error of pw.x, the state of the job,
        or the evidence of a group without progress); what is not
        established for ``uncertain``; what is awaited for ``running`` and
        ``held``.
    released : bool
        True if the calculation has no claim: its last attempt, if any, was
        archived and the next one can be taken. False if the last attempt
        is still to be archived before a relaunch.
    groups : tuple of str
        Groups of jobs that list the next attempt.
    fixed : bool
        True if the reconciliation wrote the end of an attempt in the record.
    launch : triboworkflow.execution.jobs.Launch or None
        For a next attempt that every group listing it left untaken
        (``not_taken``, or ``manual`` without progress), the launch of the
        last of those groups; None otherwise.
    """

    system: str
    calc_id: str
    state: str
    attempts: int
    reason: str | None = None
    released: bool = False
    groups: tuple = ()
    fixed: bool = False
    launch: object = None


def attempt_outcome(attempt, scheduler):
    """Outcome of an ended attempt and its reason (see the module docstring).

    Parameters
    ----------
    attempt : triboworkflow.core.store.Attempt
        Attempt whose process is not ``running``.
    scheduler : triboworkflow.execution.scheduler.JobScheduler
        Scheduler of the job, which reads ``attempt.job_state``.

    Returns
    -------
    outcome : str
        ``'completed'``, ``'unchanged'``, ``'remedy'`` or ``'manual'``.
    reason : str or None
        The reason of the stop or the state of the job that decided it;
        ``PW_ERROR`` for an error of pw.x, whose details are in the output
        (``Reconciler`` adds them).

    Raises
    ------
    ValueError
        If the attempt is recorded as running.
    """
    status, stop = attempt.status, attempt.stop_reason
    if status.process == 'running':
        raise ValueError(f'attempt {attempt.n} is recorded as running')
    if status.summary == 'completed':
        return 'completed', None
    if status.process not in ('ended', 'crashed', 'killed'):
        return 'manual', 'end of the process not known'
    if status.process == 'ended':
        if stop in REMEDY_STOPS:
            return 'remedy', stop
        if stop == 'max_seconds':
            return 'unchanged', stop
        return 'manual', f'stop reason {stop}' if stop else 'stop reason not recognized'
    if stop == 'error':
        return 'manual', PW_ERROR
    if attempt.job_state is None or stop not in (None, 'truncated'):
        return 'unchanged', 'crashed'
    if scheduler.is_active(attempt.job_state):
        return 'unchanged', f'job {attempt.job_state}: requeued'
    if scheduler.state_name(attempt.job_state) in RELAUNCH_STATES.get(scheduler.scheduler_name, ()):
        return 'unchanged', f'job {attempt.job_state}'
    return 'manual', f'job {attempt.job_state}'


def read_groups(run_dir):
    """Manifests of the groups of jobs of a run, registered or not.

    Returns
    -------
    dict
        Name → ``Manifest``, for every group with a manifest in
        ``<run>/jobs/``, in the order of the names.
    """
    groups = {}
    root = Path(run_dir) / JOBS_DIR
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if (path / MANIFEST_FILE).is_file():
                manifest, _ = read_manifest(path / MANIFEST_FILE)
                groups[manifest.name] = manifest
    return groups


def launch_index(groups):
    """Launches of groups of jobs by calculation and attempt.

    Parameters
    ----------
    groups : dict
        Name → ``Manifest`` (``read_groups``).

    Returns
    -------
    dict
        ``{(system, calc_id): {n: [(group name, Launch), ...]}}``.
    """
    index = defaultdict(lambda: defaultdict(list))
    for name, manifest in groups.items():
        for launch in manifest.calculations:
            index[launch.calc][launch.n].append((name, launch))
    return index


class LaunchError(ValueError):
    """The launch of an attempt that the manifests do not give for certain."""


def _only_launch(entries):
    """The launch listed by every ``(group, Launch)`` of ``entries`` if they agree, else None."""
    distinct = {json.dumps(launch.remedies(), sort_keys=True) for _, launch in entries}
    return entries[0][1] if len(distinct) == 1 else None


def launch_for(run_dir, system, calc_id, n, group=None):
    """Launch of attempt ``n`` of a calculation, from the manifests of the groups.

    Parameters
    ----------
    run_dir : path-like
        Directory of the run.
    system, calc_id : str
        The calculation.
    n : int
        Number of the attempt.
    group : str or None, optional
        Group whose launch is taken; by default the launch that every group
        listing the attempt gives (the same kind, level and values).

    Returns
    -------
    triboworkflow.execution.jobs.Launch

    Raises
    ------
    LaunchError
        If no group lists the attempt, ``group`` does not list it, or the
        groups that list it give different launches: the message lists them,
        with kind and level, to choose one.
    """
    entries = launch_index(read_groups(run_dir)).get((system, calc_id), {}).get(n, [])
    where = f'attempt {n} of {system}/{calc_id}'
    if not entries:
        raise LaunchError(f'no group lists {where}: its launch is not known')
    if group is not None:
        for name, launch in entries:
            if name == group:
                return launch
        raise LaunchError(f'group {group} does not list {where}; groups that list it: '
                          f'{", ".join(name for name, _ in entries)}')
    launch = _only_launch(entries)
    if launch is None:
        listed = ', '.join(f'{name} ({launch.kind}, level {launch.level})'
                           for name, launch in entries)
        raise LaunchError(f'the groups that list {where} give different launches: {listed}; '
                          'choose one with --group')
    return launch


class Reconciler:
    """Reconciliation of the calculations of a run.

    Parameters
    ----------
    run_dir : path-like
        Directory of the run.
    manager : triboworkflow.execution.manager.JobManager
        Job manager of the run, after its cycle.
    polling_interval : int
        Seconds between two queries that confirm a preemption
        (``polling_interval`` of ``[execution]``).
    """

    def __init__(self, run_dir, manager, polling_interval):
        self.run_dir = Path(run_dir)
        self.manager = manager
        self.scheduler = manager.scheduler
        self.polling_interval = polling_interval
        self.groups = read_groups(self.run_dir)
        self.launches = launch_index(self.groups)

    def reconcile(self, calculations):
        """``CalcState`` of each ``(system, calc_id)``, in order."""
        return [self.calculation(system, calc_id) for system, calc_id in calculations]

    def calculation(self, system, calc_id):
        """Bring one calculation up to date and classify it.

        Returns
        -------
        CalcState
            A record that cannot be read gives ``uncertain`` with the error.
        """
        try:
            record = read_record(system_dir(self.run_dir, system), calc_id)
        except (OSError, ValueError) as error:
            return CalcState(system, calc_id, 'uncertain', 0, f'record not read: {error}')
        path = calc_dir(self.run_dir, system, calc_id)
        if (path / CLAIM_FILE).exists():
            return self._claimed(system, calc_id, path, record)

        k = len(record.attempts)
        state = _maker(system, calc_id, k, released=True)
        left = [name for name in (RUN_INPUT_FILE, OUTPUT_FILE) if (path / name).exists()]
        if left:
            return state('uncertain', f'{", ".join(left)} without a claim')
        if k and record.attempts[-1].status.process == 'running':
            return state('uncertain', f'attempt {k} recorded as running without a claim')
        groups = self._groups(system, calc_id, k + 1)
        if groups:
            if not all(self._group_over(name) for name in groups):
                return state('pending', groups=groups)
            last = max(groups, key=list(self.manager.groups).index)
            launch = next(launch for name, launch in self.launches[(system, calc_id)][k + 1]
                          if name == last)
            if not self._progress(last):
                return state('manual', self._no_progress(last), groups=groups, launch=launch)
            return state('not_taken', groups=groups, launch=launch)
        if not k:
            return state('new')
        return state(*self._outcome(path, record.attempts[-1]))

    # --- calculations with a claim ---------------------------------------------

    def _claimed(self, system, calc_id, path, record):
        k = len(record.attempts)
        state = _maker(system, calc_id, k, released=False)
        last = record.attempts[-1] if k else None
        claim = read_claim(path / CLAIM_FILE)
        if claim is None:
            return state('uncertain', 'empty claim: the worker stopped right after taking the '
                                      'calculation, or is taking it now')
        n = claim['n']
        if n == k + 1:
            return self._claim_without_attempt(system, calc_id, path, record, claim, state)
        if n != k:
            return state('uncertain', f'claim of attempt {n}, the record has {k} attempts')
        if last.status.process != 'running':
            return state(*self._outcome(path, last))
        if last.job_id is None:
            return state('uncertain', f'attempt {n} runs without a job id: a worker started by '
                                      'hand, or a scheduler that does not give the id')
        verdict, detail = self.run_over(last.job_id, claim)
        if verdict != 'over':
            return state(verdict, detail)
        try:
            status, stop, clean = self._end_from_output(path, self._job_end(detail))
        except (OSError, ValueError, WorkerError) as error:
            return state('uncertain', f'end of attempt {n} not read: {error}')
        files = {name: (AttemptFile(name, file_digest(path / name)) if (path / name).is_file()
                        else getattr(last, role))
                 for name, role in ((RUN_INPUT_FILE, 'input'), (OUTPUT_FILE, 'output'))}
        done = dataclasses.replace(last, job_state=detail, status=status, stop_reason=stop,
                                   clean_stop=clean, input=files[RUN_INPUT_FILE],
                                   output=files[OUTPUT_FILE])
        self._write(system, record, record.attempts[:-1] + (done,), status)
        return dataclasses.replace(state(*self._outcome(path, done)), fixed=True)

    def _claim_without_attempt(self, system, calc_id, path, record, claim, state):
        """A claim of attempt ``k + 1``: the worker stopped (or is) before writing it."""
        n = claim['n']
        if claim['job_id'] is None:
            return state('uncertain', f'claim of attempt {n} without a job id, not in the record')
        left = [name for name in (RUN_INPUT_FILE, OUTPUT_FILE) if (path / name).exists()]
        if left:
            return state('uncertain', f'{", ".join(left)} of attempt {n}, not in the record')
        verdict, detail = self.run_over(claim['job_id'], claim)
        if verdict != 'over':
            return state(verdict, detail)
        launch = self._launch(system, calc_id, n, claim['job_id'])
        status = Status(process=_PROCESS_FROM_JOB[self._job_end(detail)])
        attempt = Attempt(n=n, job_id=claim['job_id'], host=claim['host'],
                          started=claim['since'], job_state=detail, status=status,
                          remedies=launch.remedies() if launch is not None else {})
        self._write(system, record, record.attempts + (attempt,), status)
        return dataclasses.replace(state(*attempt_outcome(attempt, self.scheduler)),
                                   attempts=n, fixed=True)

    def _write(self, system, record, attempts, status):
        write_record(system_dir(self.run_dir, system),
                     dataclasses.replace(record, status=status, attempts=attempts))

    def _outcome(self, path, attempt):
        """``attempt_outcome``, with the details of an error of pw.x read from its output."""
        outcome, reason = attempt_outcome(attempt, self.scheduler)
        if reason != PW_ERROR:
            return outcome, reason
        output = path / (attempt.output.path if attempt.output is not None else OUTPUT_FILE)
        try:
            calculation, _ = read_input(path / INPUT_FILE)
            error = read_pw_status(output, calculation).error
        except (OSError, ValueError, WorkerError) as failure:
            return outcome, f'{PW_ERROR}; details not read from {output}: {failure}'
        if error is None:
            return outcome, f'{PW_ERROR}; no error found in {output}'
        return outcome, f'{PW_ERROR}: {error} (output: {output})'

    # --- jobs ------------------------------------------------------------------

    def run_over(self, job_id, claim):
        """Whether the run of a job that wrote a claim is over.

        Returns
        -------
        verdict : str
            ``'over'``, ``'running'``, ``'held'`` or ``'uncertain'``.
        detail : str
            For ``'over'`` the state of the job as printed, otherwise why.
        """
        job = self.manager.jobs.get(job_id)
        if job is None:
            try:
                with hpc_only(self.scheduler.scheduler_name):
                    found = self.scheduler.get_job_state(job_id)
            except JobSchedulerError as error:
                return 'uncertain', f'job {job_id}, not followed by the job manager: {error}'
            if found.state is None:
                return 'uncertain', f'job {job_id}, not followed by the job manager, not reported'
            return self._judge(job_id, found.state, found.active, claim, None, None,
                               found.hold_reason)
        state, active = job['state'], job['active']
        hold = job['hold']['reason'] if job['hold'] is not None else None
        if job['error'] is not None:
            if active:
                verdict = 'held' if job['hold'] is not None else 'running'
                return verdict, f'job {job_id} {state} at its last known state; the last ' \
                                f'query failed: {job["error"]}'
            return 'uncertain', f'job {job_id}: the last query failed (last known state ' \
                                f'{state}): {job["error"]}'
        if job['reported'] is not True:
            return 'uncertain', f'job {job_id} not reported by the scheduler (last known state ' \
                                f'{state}): how it ended is not known'
        return self._judge(job_id, state, active, claim, job['state_since'], job['checked'], hold)

    def _judge(self, job_id, state, active, claim, first, checked, hold_reason=None):
        """Verdict of ``run_over`` for a reported state."""
        scheduler = self.scheduler
        if active is None:
            return 'uncertain', f'job {job_id}: state {state!r} not known'
        if active and scheduler.is_held(state):
            return 'held', f'job {job_id} {state}: {HOLD_MESSAGES[hold_reason]}; kept as it is, ' \
                           'not released nor replaced'
        if active:
            if scheduler.is_queued(state):
                return 'over', state
            restarts, why = self._restarts(job_id, claim)
            if restarts == 'more':
                return 'over', state
            if restarts == 'fewer':
                return 'uncertain', why
            return 'running', f'job {job_id} {state}' + (f' ({why})' if why else '')
        name = scheduler.state_name(state)
        if name in REVERSIBLE_STATES.get(scheduler.scheduler_name, ()):
            if first is None or checked is None or _seconds(first, checked) < self.polling_interval:
                return 'running', f'job {job_id} {state}: to be confirmed by a later query'
            restarts, why = self._restarts(job_id, claim)
            if restarts in ('same', 'more'):
                return 'over', state
            return 'uncertain', f'job {job_id} {state}: {why}'
        if scheduler.is_final(state):
            return 'over', state
        return 'uncertain', f'job {job_id}: state {state!r} not final'

    def _restarts(self, job_id, claim):
        """``'more'``, ``'same'``, ``'fewer'`` restarts of the job than the claim, or None.

        None when the number is not known, with the reason as second value.
        """
        if self.scheduler.scheduler_name != 'slurm':
            return None, 'the scheduler does not tell the restarts'
        if RESTART_COUNT_KEY not in claim:
            return None, 'the claim has no restart count'
        recorded = claim[RESTART_COUNT_KEY]
        if recorded is None:
            recorded = 0
        elif type(recorded) is not int:
            return None, f'restart count of the claim {recorded!r} is not a number'
        try:
            with hpc_only(self.scheduler.scheduler_name):
                restarts = self.scheduler.get_restart_count(job_id)
        except JobSchedulerError as error:
            return None, f'restarts of job {job_id} not read: {error}'
        if restarts is None:
            return None, f'restarts of job {job_id} not reported'
        if restarts > recorded:
            return 'more', None
        if restarts == recorded:
            return 'same', None
        return 'fewer', f'job {job_id} has {restarts} restarts, the claim {recorded}'

    def _job_end(self, state):
        """``'killed'`` if the scheduler stopped or requeued the job, ``'finished'`` otherwise."""
        scheduler = self.scheduler
        if scheduler.is_active(state):
            return 'killed'
        killed = KILLED_STATES.get(scheduler.scheduler_name, ())
        return 'killed' if scheduler.state_name(state) in killed else 'finished'

    def _end_from_output(self, path, job):
        """Status, reason of the stop and clean stop of an attempt whose job ended."""
        calculation, _ = read_input(path / INPUT_FILE)
        output = path / OUTPUT_FILE
        if not output.is_file() or output.stat().st_size == 0:
            return Status(process=_PROCESS_FROM_JOB[job]), None, None
        try:
            pw = read_pw_status(output, calculation)
        except PwOutputError:
            return Status(), None, None
        return record_status(pw, job), pw.stop_reason, pw.clean_stop

    # --- groups ----------------------------------------------------------------

    def _groups(self, system, calc_id, n):
        return tuple(name for name, _ in self.launches.get((system, calc_id), {}).get(n, []))

    def _group_over(self, name):
        return name in self.manager.groups and self.manager.group_complete(name)

    def _progress(self, name):
        """Whether a worker of a group took at least one of the attempts it lists."""
        for launch in self.groups[name].calculations:
            path = calc_dir(self.run_dir, launch.system, launch.calc_id)
            try:
                made = len(read_record(system_dir(self.run_dir, launch.system),
                                       launch.calc_id).attempts)
            except (OSError, ValueError):
                continue
            if made >= launch.n:
                return True
            if (path / CLAIM_FILE).exists():  # taken, perhaps not yet in the record
                claim = read_claim(path / CLAIM_FILE)
                if claim is None or claim['n'] >= launch.n:
                    return True
        return False

    def _no_progress(self, name):
        """Evidence of a group of jobs that ended without progress."""
        jobs = [f'{job_id} {self.manager.jobs[job_id]["state"]}'
                for job_id in self.manager.group_jobs(name)]
        return (f'no progress: group {name} ended without any of its '
                f'{len(self.groups[name].calculations)} attempts taken; jobs: '
                f'{", ".join(jobs) or "none"}; logs of the jobs in '
                f'{job_dir(self.run_dir, name)}')

    def _launch(self, system, calc_id, n, job_id):
        """Launch of attempt ``n``: the one of the group of the job, or the only one listed."""
        entries = self.launches.get((system, calc_id), {}).get(n, [])
        job = self.manager.jobs.get(job_id)
        for name, launch in entries:
            if job is not None and name == job['group']:
                return launch
        return _only_launch(entries) if entries else None


def _maker(system, calc_id, attempts, released):
    def make(state, reason=None, groups=(), launch=None):
        return CalcState(system, calc_id, state, attempts, reason, released, groups,
                         launch=launch)
    return make


def _seconds(first, last):
    return (datetime.datetime.fromisoformat(last)
            - datetime.datetime.fromisoformat(first)).total_seconds()
