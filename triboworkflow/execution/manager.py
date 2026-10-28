# Author: Vito Foderà
# Created: 2026-10-28

"""Job manager: submission of the groups of jobs and polling of their states.

The manager keeps, for each group of jobs of a run (``jobs.py``), how many
jobs it asks for, the submissions that failed and the state of each job, in
``<run>/jobs.json``, written atomically after every change. A cycle
(``JobManager.cycle``):

1. recovers the ids in ``jobs/<name>/submitted.json`` that the registry does
   not have yet, e.g. after an interruption between a submission and the
   update of the registry, so that a submitted job is never submitted again;
2. asks the scheduler the state of each job whose state may still change
   (``JobScheduler.get_job_state``), also of the jobs that need attention;
3. submits the jobs still missing from each group: ``min(count,
   calculations of the group)`` jobs per group, where ``count`` is the number
   of workers asked for (``submit_jobs``). Every registered job counts,
   recovered or queued or possibly still running, and so does every
   submission whose outcome is not known.

A failed submission is never repeated automatically. If the job was surely
not created (``JobSubmissionError.accepted`` False), the group is marked
refused and ``clear_refusal`` lets it be submitted again. If the job may have
been created (e.g. ``sbatch`` timed out), the submission is kept as one of
outcome not known: it takes the place of a job in the group, stops the
submissions of the group, survives a restart, and is resolved by
``resolve_submission``, with the id of the job if it is found.

Each job is ``waiting``, ``held``, ``ended`` or needs ``attention``:

- waiting: queued or running at its last known state, also when the last
  query failed; or not yet reported by the scheduler, during the first
  ``report_wait`` seconds after it was registered;
- held: in the queue but held (``JobScheduler.is_held``), with the state as
  printed and the reason of the hold when the scheduler gives it
  (``HOLD_MESSAGES``). The hold is never released by the manager, and the
  job is never replaced;
- ended: in an ended state at the last query;
- attention: not reported after ``report_wait`` seconds, no longer reported
  after having been, or in a state that is not known.

A job is also ``uncertain`` while its last query failed or did not report
it. A job keeps its id, its last known state and the claims of its
calculations, which the manager never touches, and is asked again at every
cycle until its state is final (``JobScheduler.is_final``). A group is
complete only when all its jobs were submitted and are in a final state
confirmed by the last query, with no submission refused or of outcome not
known.

``JobManager.run`` repeats the cycle every ``interval`` seconds while a job is
waiting or a group has jobs to submit. Held jobs, jobs that need attention
and submissions of outcome not known do not keep it waiting: they stay in the
summary, keep their groups incomplete and are handled again by any later
cycle, so the other jobs are followed meanwhile. With a scheduler in a dry run the commands are only recorded, the
registry is not written and ``run`` stops after one cycle.

The manager does not decide what an ended job means for its calculations:
that is read from the records and the claims of the calculations.

Notes
-----
The structure follows SCS (``scheduler/JobManager.py``, ``BaseJobManager``
with its state file and its loop, ``HPCJobManager`` with one state query per
job), with the state file in JSON instead of YAML and the state of the
scheduler kept as printed instead of reduced to terminated.
"""

import datetime
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from triboworkflow.core.guard import hpc_only
from triboworkflow.core.store.atomic import read_json, write_json
from triboworkflow.core.store.record import check_keys

from .jobs import MANIFEST_FILE, SUBMITTED_FILE, JobError, read_manifest, submit_jobs
from .layout import JOBS_DIR, JOBS_REGISTRY_FILE, job_dir
from .scheduler import JobSchedulerError

logger = logging.getLogger(__name__)

REGISTRY_VERSION = 1

_REGISTRY_KEYS = ('schema_version', 'groups', 'jobs')
_GROUP_KEYS = ('planned', 'refused', 'unknown')
_UNKNOWN_KEYS = ('error', 'at')
_JOB_KEYS = ('group', 'since', 'state', 'active', 'hold', 'reported', 'checked', 'error')
_HOLD_KEYS = ('reason', 'detail')

# Report of a held job for each reason of the hold (``JobState.hold_reason``).
HOLD_MESSAGES = {
    'intervention': 'held, needs an intervention to be released',
    'dependency': 'held, waiting for other jobs',
    'start_time': 'held, waiting for its start time',
    None: 'held, reason to be checked',
}


class JobManagerError(ValueError):
    """A registry of jobs, a group or a request that the manager cannot use."""


def _local_now():
    return datetime.datetime.now().astimezone()


@dataclass(frozen=True)
class JobsSummary:
    """Jobs and groups of a run after a cycle of the manager.

    Attributes
    ----------
    waiting, ended, attention : list of str
        Ids of the jobs in each status (see the module docstring).
    held : dict
        For each held job, ``{'state', 'reason', 'detail', 'message'}``: the
        state as printed, the reason and detail of the hold
        (``JobState.hold_reason``, ``JobState.hold_detail``) and its report
        (``HOLD_MESSAGES``).
    uncertain : list of str
        Jobs whose last query failed or did not report them.
    to_submit : int
        Jobs still to submit, in the groups whose submissions are not stopped.
    refused : dict
        Error of each group whose last submission was refused.
    unknown : dict
        Number of submissions of outcome not known of each group that has some.
    complete, incomplete : list of str
        Names of the complete groups and of the others.
    """

    waiting: list
    ended: list
    attention: list
    held: dict
    uncertain: list
    to_submit: int
    refused: dict
    unknown: dict
    complete: list
    incomplete: list


class JobManager:
    """Groups of jobs of a run, their submission and the states of their jobs.

    Parameters
    ----------
    run_dir : path-like
        Directory of the run.
    scheduler : triboworkflow.execution.scheduler.JobScheduler
        Scheduler of the jobs.
    report_wait : int
        Seconds after its registration during which a job not reported by the
        scheduler is still waiting (``job_report_wait`` of ``[execution]``).
    now : callable or None, optional
        Current time, as an aware ``datetime.datetime``; the local time by
        default.

    Attributes
    ----------
    path : pathlib.Path
        Registry of the jobs, ``<run>/jobs.json``.
    registry : dict
        Content of the registry: ``groups`` (``planned`` jobs, error of the
        submission ``refused`` and submissions of outcome ``unknown`` of each
        group) and ``jobs`` (``group``, time ``since`` it was registered,
        last known ``state`` as printed by the scheduler, whether it was
        ``active``, the ``hold`` (``reason`` and ``detail``, or None if not
        held), whether the last successful query ``reported`` it, time
        ``checked`` of that query and ``error`` of the last failed one, for
        each job id).

    Raises
    ------
    JobManagerError
        If ``report_wait`` is not an integer > 0, or the registry exists and
        is not valid.
    """

    def __init__(self, run_dir, scheduler, report_wait, now=None):
        if type(report_wait) is not int or report_wait <= 0:
            raise JobManagerError(f'report_wait: {report_wait!r} is not an integer > 0')
        self.run_dir = Path(run_dir)
        self.scheduler = scheduler
        self.report_wait = report_wait
        self.path = self.run_dir / JOBS_REGISTRY_FILE
        self._now = now or _local_now
        self.registry = self._load()

    def _stamp(self):
        return self._now().isoformat(timespec='seconds')

    def _load(self):
        if not self.path.exists():
            return {'schema_version': REGISTRY_VERSION, 'groups': {}, 'jobs': {}}
        try:
            data = check_keys(read_json(self.path), _REGISTRY_KEYS, 'registry')
            if data['schema_version'] != REGISTRY_VERSION:
                raise JobManagerError(f'schema_version: {data["schema_version"]!r}; this '
                                      f'version reads {REGISTRY_VERSION}')
            for field in ('groups', 'jobs'):
                if not isinstance(data[field], dict):
                    raise JobManagerError(f'{field}: a JSON object is required')
            for name, group in data['groups'].items():
                check_keys(group, _GROUP_KEYS, f'group {name}')
                if not isinstance(group['unknown'], list):
                    raise JobManagerError(f'group {name}: unknown: a JSON array is required')
                for submission in group['unknown']:
                    check_keys(submission, _UNKNOWN_KEYS, f'group {name}: unknown')
            for job_id, job in data['jobs'].items():
                check_keys(job, _JOB_KEYS, f'job {job_id}')
                if job['hold'] is not None:
                    check_keys(job['hold'], _HOLD_KEYS, f'job {job_id}: hold')
                if job['group'] not in data['groups']:
                    raise JobManagerError(f'job {job_id}: group {job["group"]!r} not registered')
        except ValueError as error:  # JobManagerError, and StoreError from the store
            raise JobManagerError(f'{self.path}: {error}') from None
        return data

    def _save(self):
        if not self.scheduler.dry_run:
            write_json(self.path, self.registry)

    @property
    def groups(self):
        """Registered groups: name → ``{'planned', 'refused', 'unknown'}``."""
        return self.registry['groups']

    @property
    def jobs(self):
        """Known jobs: id → ``{'group', 'since', 'state', 'active', 'hold', 'reported',
        'checked', 'error'}``."""
        return self.registry['jobs']

    def group_jobs(self, name):
        """Ids of the jobs of a group, in the order they were registered."""
        return [job_id for job_id, job in self.jobs.items() if job['group'] == name]

    def _add_job(self, job_id, name):
        self.jobs[job_id] = {'group': name, 'since': self._stamp(), 'state': None,
                             'active': None, 'hold': None, 'reported': None, 'checked': None,
                             'error': None}

    def _group(self, name):
        if name not in self.groups:
            raise JobManagerError(f'group {name!r} is not registered in {self.path}')
        return self.groups[name]

    # --- groups ----------------------------------------------------------------

    def add_group(self, name, count):
        """Register a group written by ``write_job``, to be submitted by ``submit``.

        Parameters
        ----------
        name : str
            Name of the group.
        count : int
            Number of workers asked for (``groups`` of ``[execution]``); at
            most one job per calculation of the group is planned.

        Returns
        -------
        int
            Jobs planned for the group, ``min(count, calculations)``.

        Raises
        ------
        JobManagerError
            If the group is already registered.
        JobError
            If ``count`` is not an integer > 0 or the group has no manifest.
        """
        if name in self.groups:
            raise JobManagerError(f'group {name!r} is already registered in {self.path}')
        if type(count) is not int or count <= 0:
            raise JobError(f'number of jobs: {count!r} is not an integer > 0')
        path = job_dir(self.run_dir, name) / MANIFEST_FILE
        if not path.is_file():
            raise JobError(f'{path} not found: write the group first')
        manifest, _ = read_manifest(path)
        planned = min(count, len(manifest.calculations))
        self.groups[name] = {'planned': planned, 'refused': None, 'unknown': []}
        self._save()
        return planned

    def missing_jobs(self, name):
        """Jobs of a group still to submit: planned minus jobs registered and
        submissions of outcome not known."""
        group = self._group(name)
        return max(group['planned'] - len(self.group_jobs(name)) - len(group['unknown']), 0)

    def clear_refusal(self, name):
        """Let a group whose submission was refused be submitted again by ``submit``."""
        self._group(name)['refused'] = None
        self._save()

    def resolve_submission(self, name, job_id=None):
        """Resolve the oldest submission of outcome not known of a group.

        Parameters
        ----------
        name : str
            Name of the group.
        job_id : str or None, optional
            Id of the job that the submission created, if it was found: the
            job is registered and added to ``submitted.json``. None if the
            submission created no job: its place in the group is freed.

        Raises
        ------
        JobManagerError
            If the group has no such submission, or the job is already known.
        """
        group = self._group(name)
        if not group['unknown']:
            raise JobManagerError(f'group {name!r} has no submission of outcome not known')
        if job_id is not None:
            job_id = str(job_id)
            if job_id in self.jobs:
                raise JobManagerError(f'job {job_id} is already registered')
            submitted = job_dir(self.run_dir, name) / SUBMITTED_FILE
            data = read_json(submitted) if submitted.exists() else {'job_ids': []}
            data['job_ids'].append(job_id)
            write_json(submitted, data)
            self._add_job(job_id, name)
        group['unknown'].pop(0)
        self._save()

    # --- cycle -----------------------------------------------------------------

    def recover(self):
        """Add to the registry the jobs in ``submitted.json`` that it does not have.

        A group with ``submitted.json`` that is not registered (e.g. submitted
        by hand) is registered with the jobs already submitted, so that none
        is added to it.

        Returns
        -------
        list of str
            Ids added to the registry.
        """
        recovered = []
        jobs_root = self.run_dir / JOBS_DIR
        names = sorted(p.name for p in jobs_root.iterdir() if (p / SUBMITTED_FILE).is_file()) \
            if jobs_root.is_dir() else []
        for name in names:
            job_ids = read_json(job_dir(self.run_dir, name) / SUBMITTED_FILE)['job_ids']
            if name not in self.groups:
                self.groups[name] = {'planned': len(job_ids), 'refused': None, 'unknown': []}
                logger.warning('Group %s was not registered: %d submitted jobs recovered',
                               name, len(job_ids))
            for job_id in job_ids:
                if job_id not in self.jobs:
                    self._add_job(job_id, name)
                    recovered.append(job_id)
        if recovered:
            logger.warning('Jobs submitted and missing from the registry, recovered: %s',
                           ', '.join(recovered))
        self._save()
        return recovered

    def poll(self):
        """Ask the scheduler the state of each job whose state may still change.

        A job in a final state confirmed by the last query is not asked
        again. A job not reported keeps its last known state; a failed query
        keeps the state and records the error.

        Raises
        ------
        NotOnHPCError
            If the command of the scheduler is not found.
        """
        for job_id, job in self.jobs.items():
            if job['reported'] is True and self.scheduler.is_final(job['state']):
                continue
            try:
                with hpc_only(self.scheduler.scheduler_name):
                    state = self.scheduler.get_job_state(job_id)
            except JobSchedulerError as error:
                job['error'] = str(error)
                logger.warning('State of job %s not known, last known state %s kept: %s',
                               job_id, job['state'], error)
                continue
            if state.state is None:
                job.update(reported=False, checked=self._stamp(), error=None)
                logger.warning('Job %s not reported by the scheduler, last known state %s kept',
                               job_id, job['state'])
            else:
                hold = ({'reason': state.hold_reason, 'detail': state.hold_detail}
                        if state.held else None)
                job.update(state=state.state, active=state.active, hold=hold, reported=True,
                           checked=self._stamp(), error=None)
        self._save()

    def submit(self):
        """Submit the jobs still missing from each group whose submissions are not stopped.

        Returns
        -------
        list of str
            Ids of the jobs submitted.

        Raises
        ------
        NotOnHPCError
            If the submission command of the scheduler is not found.
        """
        submitted = []
        for name, group in self.groups.items():
            missing = self.missing_jobs(name)
            if group['refused'] is not None or group['unknown'] or missing == 0:
                continue
            try:
                job_ids = submit_jobs(job_dir(self.run_dir, name), self.scheduler, missing)
            except JobSchedulerError as error:
                if getattr(error, 'accepted', False) is None:
                    group['unknown'].append({'error': str(error), 'at': self._stamp()})
                    logger.error('Submission in group %s of outcome not known: the job may '
                                 'exist; it is not submitted again: %s', name, error)
                else:
                    group['refused'] = str(error)
                    logger.error('Submission in group %s refused, not retried: %s', name, error)
                self._save()
                self.recover()  # jobs submitted before the failure
                continue
            for job_id in job_ids:
                if job_id is not None:  # None in a dry run
                    self._add_job(job_id, name)
                    submitted.append(job_id)
            self._save()
        return submitted

    # --- status ----------------------------------------------------------------

    def job_status(self, job_id):
        """``'waiting'``, ``'held'``, ``'ended'`` or ``'attention'`` (see the module docstring)."""
        job = self.jobs[job_id]
        if job['state'] is None:  # never reported
            since = datetime.datetime.fromisoformat(job['since'])
            waited = (self._now() - since).total_seconds()
            return 'waiting' if waited < self.report_wait else 'attention'
        if job['reported'] is False or job['active'] is None:
            return 'attention'
        if job['active'] and job['hold'] is not None:
            return 'held'
        return 'waiting' if job['active'] else 'ended'

    def group_complete(self, name):
        """Whether all the jobs of a group were submitted and are in a confirmed final state."""
        group = self._group(name)
        job_ids = self.group_jobs(name)
        return (group['refused'] is None and not group['unknown']
                and len(job_ids) >= group['planned']
                and all(self.jobs[j]['reported'] is True and self.jobs[j]['error'] is None
                        and self.scheduler.is_final(self.jobs[j]['state']) for j in job_ids))

    def summary(self):
        """Jobs by status, jobs to submit, failed submissions and groups complete or not."""
        status = {job_id: self.job_status(job_id) for job_id in self.jobs}
        complete = [name for name in self.groups if self.group_complete(name)]
        return JobsSummary(
            waiting=[j for j, s in status.items() if s == 'waiting'],
            ended=[j for j, s in status.items() if s == 'ended'],
            attention=[j for j, s in status.items() if s == 'attention'],
            held={j: {'state': self.jobs[j]['state'], **self.jobs[j]['hold'],
                      'message': HOLD_MESSAGES[self.jobs[j]['hold']['reason']]}
                  for j, s in status.items() if s == 'held'},
            uncertain=[j for j, job in self.jobs.items()
                       if job['error'] is not None or job['reported'] is False],
            to_submit=sum(self.missing_jobs(name) for name, group in self.groups.items()
                          if group['refused'] is None and not group['unknown']),
            refused={name: group['refused'] for name, group in self.groups.items()
                     if group['refused'] is not None},
            unknown={name: len(group['unknown']) for name, group in self.groups.items()
                     if group['unknown']},
            complete=complete,
            incomplete=[name for name in self.groups if name not in complete])

    def cycle(self):
        """Recover, poll and submit; return the ``summary``."""
        self.recover()
        self.poll()
        self.submit()
        return self.summary()

    def run(self, interval, sleep=time.sleep, max_cycles=None):
        """Repeat ``cycle`` every ``interval`` seconds while a job is waiting or jobs are to submit.

        Parameters
        ----------
        interval : int
            Seconds between two cycles (``polling_interval`` of
            ``[execution]``).
        sleep : callable, optional
            Function waiting a number of seconds.
        max_cycles : int or None, optional
            Stop after this many cycles; no limit by default.

        Returns
        -------
        JobsSummary
            Summary of the last cycle: no job waiting and nothing to submit,
            unless ``max_cycles`` stopped the loop or the scheduler is in a
            dry run, which stops after one cycle. Jobs that need attention
            and submissions of outcome not known may remain.
        """
        cycles = 0
        while True:
            summary = self.cycle()
            cycles += 1
            if not summary.waiting and not summary.to_submit:
                return summary
            if self.scheduler.dry_run or (max_cycles is not None and cycles >= max_cycles):
                return summary
            sleep(interval)
