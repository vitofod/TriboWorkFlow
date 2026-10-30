# Author: Vito Foderà
# Created: 2026-10-28

"""Tests of the job manager (``triboworkflow.execution.manager``).

No scheduler is called: Slurm and HTCondor are simulated at the level of
their commands. ``sbatch`` numbers the jobs, ``sacct`` prints the state set by
the test, or nothing for a job it does not report; a submission can be
refused, or accepted with its answer lost (a timeout after the job was
created). Time is given by a clock that the tests move forward.
"""

import datetime
import json
import subprocess

import pytest

from triboworkflow.core.guard import NotOnHPCError
from triboworkflow.execution import layout
from triboworkflow.execution.jobs import (SUBMITTED_FILE, JobError, Launch, Manifest, submit_jobs,
                                          write_job)
from triboworkflow.execution.manager import JobManager, JobManagerError
from triboworkflow.execution.scheduler import SGE_HOLD_TYPES, JobScheduler, JobSchedulerError

CALCS = (('Cu111_Al111', 'pes_scf/aa'), ('Cu111_Al111', 'pes_scf/ab'), ('Cu111_Al111', 'pes_scf/ac'))
WAIT = 3600
START = datetime.datetime(2026, 10, 28, 10, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=1)))


class Clock:
    """Current time of the manager, moved forward by the tests."""

    def __init__(self):
        self.time = START

    def __call__(self):
        return self.time

    def advance(self, seconds):
        self.time += datetime.timedelta(seconds=seconds)


class SimulatedSlurm(JobScheduler):
    """Slurm whose commands are simulated.

    ``sbatch`` numbers the jobs and sets them PENDING; ``sacct`` prints
    ``states`` (nothing for a job missing from it). ``refuse`` makes the next
    submissions fail with an error and create no job; ``lose`` makes them
    create the job and then time out. ``fail`` maps a command name to an
    error it raises without doing anything.
    """

    def __init__(self, first_id=1000):
        super().__init__('slurm')
        self.next_id = first_id
        self.states, self.fail = {}, {}
        self.refuse = self.lose = 0
        self.calls = []

    def _run_command(self, cmd, input_data=None, cwd=None):
        name = cmd[0]
        self.calls.append(cmd)
        if name in self.fail:
            raise self.fail[name]
        if name == 'sbatch':
            if self.refuse:
                self.refuse -= 1
                raise JobSchedulerError('Command failed with return code 1: invalid account') \
                    from subprocess.CalledProcessError(1, cmd, output='', stderr='invalid account')
            job_id = str(self.next_id)
            self.next_id += 1
            self.states[job_id] = 'PENDING'
            if self.lose:
                self.lose -= 1
                raise JobSchedulerError('Command timed out after 30 seconds') \
                    from subprocess.TimeoutExpired(cmd, 30)
            return subprocess.CompletedProcess(cmd, 0, stdout=f'Submitted batch job {job_id}\n',
                                               stderr='')
        assert name == 'sacct', cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=self.states.get(cmd[-1], '') + '\n',
                                           stderr='')

    def count(self, name):
        return sum(1 for cmd in self.calls if cmd[0] == name)


class SimulatedCondor(JobScheduler):
    """HTCondor whose commands are simulated: one process per cluster, ``JobStatus`` in ``states``."""

    def __init__(self):
        super().__init__('htcondor')
        self.next_id, self.states = 591, {}

    def _run_command(self, cmd, input_data=None, cwd=None):
        if cmd[0] == 'condor_submit':
            job_id = str(self.next_id)
            self.next_id += 1
            self.states[job_id] = '1'
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f'1 job(s) submitted to cluster {job_id}.\n', stderr='')
        assert cmd[0] in ('condor_q', 'condor_history'), cmd
        state = self.states.get(cmd[1])
        in_queue = state is not None and state not in ('3', '4')
        listed = state is not None and (in_queue == (cmd[0] == 'condor_q'))
        return subprocess.CompletedProcess(cmd, 0, stdout=f'{cmd[1]}.0 {state}\n' if listed else '',
                                           stderr='')


def missing(name):
    error = JobSchedulerError(f"Command '{name}' not found")
    error.__cause__ = FileNotFoundError(2, 'No such file or directory', name)
    return error


@pytest.fixture
def run_dir(tmp_path):
    run = tmp_path / 'run'
    for system, calc_id in CALCS:
        path = layout.calc_dir(run, system, calc_id)
        path.mkdir(parents=True)
        (path / layout.INPUT_FILE).write_text('&CONTROL\n/\n')
    jobscript = tmp_path / 'job.sh'
    jobscript.write_text('#!/bin/bash\n')
    for name in ('batch1', 'batch2'):
        write_job(run, Manifest(name=name, scheduler='slurm', run_command='pw.x', walltime=3600,
                                max_seconds_margin=600,
                                calculations=tuple(Launch.initial(*c) for c in CALCS)), jobscript)
    return run


@pytest.fixture
def slurm():
    return SimulatedSlurm()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def manager(run_dir, slurm, clock):
    """Factory of managers of the run, as after a start or a restart."""
    return lambda scheduler=slurm: JobManager(run_dir, scheduler, WAIT, now=clock)


def registry(run_dir):
    return json.loads((run_dir / layout.JOBS_REGISTRY_FILE).read_text())


def submitted(run_dir, name):
    return json.loads((layout.job_dir(run_dir, name) / SUBMITTED_FILE).read_text())['job_ids']


# --- submission --------------------------------------------------------------------

@pytest.mark.parametrize('count, planned', [(1, 1), (2, 2), (3, 3), (5, 3)])
def test_at_most_one_job_per_calculation(manager, slurm, count, planned):
    jobs = manager()
    assert jobs.add_group('batch1', count) == planned
    summary = jobs.cycle()
    assert slurm.count('sbatch') == planned
    assert summary.waiting == jobs.group_jobs('batch1') == [str(1000 + i) for i in range(planned)]
    assert summary.to_submit == 0 and summary.attention == summary.uncertain == []
    assert summary.incomplete == ['batch1'] and summary.complete == []


def test_submitted_jobs_are_not_submitted_again(run_dir, manager, slurm):
    jobs = manager()
    jobs.add_group('batch1', 2)
    jobs.cycle()
    jobs.cycle()
    manager().cycle()  # restart: the registry is read again
    assert slurm.count('sbatch') == 2
    assert list(registry(run_dir)['jobs']) == ['1000', '1001']


def test_registry_content(run_dir, manager, slurm, clock):
    jobs = manager()
    jobs.add_group('batch1', 1)
    jobs.cycle()
    clock.advance(1800)
    slurm.states['1000'] = 'CANCELLED by 1000'
    jobs.cycle()
    assert registry(run_dir) == {
        'schema_version': 2,
        'groups': {'batch1': {'planned': 1, 'refused': None, 'unknown': []}},
        'jobs': {'1000': {'group': 'batch1', 'since': '2026-10-28T10:00:00+01:00',
                          'state': 'CANCELLED by 1000',
                          'state_since': '2026-10-28T10:30:00+01:00', 'active': False,
                          'hold': None, 'reported': True,
                          'checked': '2026-10-28T10:30:00+01:00', 'error': None}}}


def test_time_of_the_first_report_of_a_state(manager, slurm, clock):
    jobs = manager()
    jobs.add_group('batch1', 1)
    jobs.cycle()
    assert jobs.jobs['1000']['state_since'] is None  # submitted, not yet asked
    jobs.cycle()
    assert jobs.jobs['1000']['state_since'] == '2026-10-28T10:00:00+01:00'
    clock.advance(600)
    slurm.states['1000'] = 'PREEMPTED'
    jobs.cycle()
    for _ in range(2):  # same state: the first report is kept
        clock.advance(600)
        jobs.cycle()
    assert jobs.jobs['1000']['state_since'] == '2026-10-28T10:10:00+01:00'
    assert jobs.jobs['1000']['checked'] == '2026-10-28T10:30:00+01:00'
    clock.advance(600)
    del slurm.states['1000']  # not reported: state and its time kept
    jobs.cycle()
    assert (jobs.jobs['1000']['state'], jobs.jobs['1000']['state_since']) == \
        ('PREEMPTED', '2026-10-28T10:10:00+01:00')
    clock.advance(600)
    slurm.states['1000'] = 'PENDING'
    jobs.cycle()
    assert jobs.jobs['1000']['state_since'] == '2026-10-28T10:50:00+01:00'
    assert jobs.now() == clock.time


def test_group_errors(run_dir, slurm, manager):
    jobs = manager()
    jobs.add_group('batch1', 1)
    with pytest.raises(JobManagerError, match="group 'batch1' is already registered"):
        jobs.add_group('batch1', 1)
    for count in (0, -1, 1.5, True):
        with pytest.raises(JobError, match='is not an integer > 0'):
            jobs.add_group('batch2', count)
    with pytest.raises(JobError, match='not found: write the group first'):
        jobs.add_group('missing', 1)
    for wait in (0, 60.0, None):
        with pytest.raises(JobManagerError, match='report_wait'):
            JobManager(run_dir, slurm, wait)


def test_refused_submission_is_not_retried_until_cleared(manager, slurm):
    slurm.refuse = 1
    jobs = manager()
    jobs.add_group('batch1', 2)
    summary = jobs.cycle()
    assert 'invalid account' in summary.refused['batch1']
    assert jobs.group_jobs('batch1') == [] and summary.to_submit == 0
    manager().cycle()  # neither the next cycle nor a restart submits again
    assert slurm.count('sbatch') == 1
    jobs = manager()
    jobs.clear_refusal('batch1')  # no job was created: submitting again is safe
    summary = jobs.cycle()
    assert jobs.group_jobs('batch1') == ['1000', '1001'] and summary.refused == {}


def test_submission_away_from_hpc(manager, slurm):
    slurm.fail['sbatch'] = missing('sbatch')
    jobs = manager()
    jobs.add_group('batch1', 1)
    with pytest.raises(NotOnHPCError):
        jobs.cycle()


# --- submission accepted, answer lost -----------------------------------------------

def test_lost_answer_is_never_submitted_again(run_dir, manager, slurm, clock):
    slurm.lose = 1
    jobs = manager()
    jobs.add_group('batch1', 1)
    summary = jobs.cycle()
    assert slurm.states == {'1000': 'PENDING'}  # the job exists, its id is not known
    assert summary.unknown == {'batch1': 1} and summary.to_submit == 0
    assert summary.incomplete == ['batch1']
    assert 'timed out' in jobs.groups['batch1']['unknown'][0]['error']
    for _ in range(3):  # later cycles and restarts
        clock.advance(WAIT)
        summary = manager().cycle()
    assert slurm.count('sbatch') == 1
    assert summary.unknown == {'batch1': 1} and summary.incomplete == ['batch1']


def test_lost_answer_resolved_with_the_job_found(run_dir, manager, slurm):
    slurm.lose = 1
    jobs = manager()
    jobs.add_group('batch1', 1)
    jobs.cycle()
    jobs.resolve_submission('batch1', '1000')
    assert submitted(run_dir, 'batch1') == ['1000']
    slurm.states['1000'] = 'RUNNING'
    summary = manager().cycle()
    assert summary.waiting == ['1000'] and summary.unknown == {}
    slurm.states['1000'] = 'COMPLETED'
    summary = manager().cycle()
    assert summary.complete == ['batch1'] and slurm.count('sbatch') == 1


def test_lost_answer_resolved_as_no_job(manager, slurm):
    slurm.lose = 1
    jobs = manager()
    jobs.add_group('batch1', 1)
    jobs.cycle()
    jobs.resolve_submission('batch1')  # the job was not created: its place is free
    summary = jobs.cycle()
    assert jobs.group_jobs('batch1') == ['1001'] and summary.unknown == {}
    with pytest.raises(JobManagerError, match='no submission of outcome not known'):
        jobs.resolve_submission('batch1')
    with pytest.raises(JobManagerError, match='not registered'):
        jobs.resolve_submission('batch3')


def test_lost_answer_stops_the_group(manager, slurm):
    # First job submitted, second created with its answer lost: the third waits.
    jobs = manager()
    jobs.add_group('batch1', 3)
    original = slurm._run_command

    def second_lost(cmd, input_data=None, cwd=None):
        if cmd[0] == 'sbatch' and slurm.count('sbatch') == 1:
            slurm.lose = 1
        return original(cmd, input_data, cwd)

    slurm._run_command = second_lost
    summary = jobs.cycle()
    assert jobs.group_jobs('batch1') == ['1000'] and summary.unknown == {'batch1': 1}
    assert summary.to_submit == 0 and slurm.count('sbatch') == 2
    jobs.cycle()
    assert slurm.count('sbatch') == 2
    with pytest.raises(JobManagerError, match='already registered'):
        jobs.resolve_submission('batch1', '1000')


# --- recovery ----------------------------------------------------------------------

def test_jobs_submitted_before_an_interruption_are_recovered(run_dir, manager, slurm):
    # Interruption between the submissions and the update of the registry.
    jobs = manager()
    jobs.add_group('batch1', 3)
    submit_jobs(layout.job_dir(run_dir, 'batch1'), slurm, 2)
    restarted = manager()
    assert restarted.recover() == ['1000', '1001']
    restarted.cycle()
    assert slurm.count('sbatch') == 3  # only the missing one
    assert restarted.group_jobs('batch1') == ['1000', '1001', '1002']


def test_group_submitted_by_hand_is_registered_without_new_jobs(run_dir, manager, slurm):
    submit_jobs(layout.job_dir(run_dir, 'batch2'), slurm, 2)
    jobs = manager()
    summary = jobs.cycle()
    assert jobs.groups == {'batch2': {'planned': 2, 'refused': None, 'unknown': []}}
    assert slurm.count('sbatch') == 2 and summary.to_submit == 0
    assert summary.waiting == ['1000', '1001']


def test_invalid_registry(run_dir, manager):
    path = run_dir / layout.JOBS_REGISTRY_FILE
    group = {'planned': 1, 'refused': None, 'unknown': []}
    job = {'group': 'b', 'since': None, 'state': None, 'state_since': None, 'active': None,
           'hold': None, 'reported': None, 'checked': None, 'error': None}
    old_job = {k: v for k, v in job.items() if k != 'state_since'}
    for content, message in [
        ({'schema_version': 1, 'groups': {}, 'jobs': {}}, 'schema_version: 1; this version '
                                                          'reads 2'),
        ({'schema_version': 2, 'groups': []}, 'missing jobs'),
        ({'schema_version': 2, 'groups': [], 'jobs': {}}, 'groups: a JSON object'),
        ({'schema_version': 2, 'groups': {'b': {'planned': 1}}, 'jobs': {}}, 'missing refused'),
        ({'schema_version': 2, 'groups': {'b': dict(group, unknown={})}, 'jobs': {}},
         'a JSON array'),
        ({'schema_version': 2, 'groups': {'b': dict(group, unknown=[{'error': 'x'}])}, 'jobs': {}},
         'missing at'),
        ({'schema_version': 2, 'groups': {}, 'jobs': {'1': job}}, "group 'b' not registered"),
        ({'schema_version': 2, 'groups': {'b': group}, 'jobs': {'1': dict(job, hold={})}},
         'hold: missing reason'),
        ({'schema_version': 2, 'groups': {'b': group}, 'jobs': {'1': old_job}},
         'missing state_since'),
    ]:
        path.write_text(json.dumps(content))
        with pytest.raises(JobManagerError, match=message):
            manager()


# --- polling -----------------------------------------------------------------------

def test_states_from_the_scheduler(manager, slurm):
    jobs = manager()
    jobs.add_group('batch1', 3)
    jobs.cycle()
    slurm.states.update({'1000': 'RUNNING', '1001': 'TIMEOUT', '1002': 'OUT_OF_ME+'})
    summary = jobs.cycle()
    assert summary.waiting == ['1000'] and summary.ended == ['1001']
    assert summary.attention == ['1002']  # a state cut short is not known
    assert [jobs.jobs[j]['state'] for j in ('1000', '1001', '1002')] == \
        ['RUNNING', 'TIMEOUT', 'OUT_OF_ME+']
    assert summary.incomplete == ['batch1']


def test_first_query_empty_then_the_job_appears(manager, slurm, clock):
    jobs = manager()
    jobs.add_group('batch1', 1)
    jobs.cycle()
    del slurm.states['1000']  # not yet reported by sacct
    clock.advance(1800)
    summary = jobs.cycle()
    assert summary.waiting == ['1000'] and summary.uncertain == ['1000']
    assert summary.ended == [] and summary.attention == []
    slurm.states['1000'] = 'RUNNING'
    clock.advance(1800)
    summary = jobs.cycle()
    assert summary.waiting == ['1000'] and summary.uncertain == []
    assert slurm.count('sbatch') == 1


def test_job_still_not_reported_needs_attention(run_dir, manager, slurm, clock):
    jobs = manager()
    jobs.add_group('batch1', 1)
    jobs.cycle()
    del slurm.states['1000']
    clock.advance(WAIT)
    summary = jobs.cycle()
    assert summary.attention == summary.uncertain == ['1000'] and summary.waiting == []
    assert summary.incomplete == ['batch1'] and summary.to_submit == 0
    asked = slurm.count('sacct')
    clock.advance(WAIT)
    summary = manager().cycle()  # restart with the job uncertain: asked again, not replaced
    assert slurm.count('sacct') == asked + 1 and slurm.count('sbatch') == 1
    assert summary.attention == ['1000']
    slurm.states['1000'] = 'RUNNING'
    summary = manager().cycle()
    assert summary.waiting == ['1000'] and summary.attention == summary.uncertain == []


def test_job_no_longer_reported_keeps_its_last_state(manager, slurm, clock):
    jobs = manager()
    jobs.add_group('batch1', 1)
    jobs.cycle()
    slurm.states['1000'] = 'RUNNING'
    jobs.cycle()
    del slurm.states['1000']  # e.g. a job missing from the history
    summary = jobs.cycle()
    job = jobs.jobs['1000']
    assert job['state'] == 'RUNNING' and job['active'] is True and job['reported'] is False
    assert summary.attention == summary.uncertain == ['1000'] and summary.waiting == []
    assert summary.incomplete == ['batch1'] and slurm.count('sbatch') == 1


def test_final_jobs_are_not_asked_again(manager, slurm):
    jobs = manager()
    jobs.add_group('batch1', 2)
    jobs.cycle()
    slurm.states.update({'1000': 'COMPLETED', '1001': 'PREEMPTED'})
    summary = jobs.cycle()
    assert summary.ended == ['1000', '1001'] and summary.incomplete == ['batch1']
    asked = slurm.count('sacct')
    slurm.states['1001'] = 'PENDING'  # requeued after the preemption
    summary = jobs.cycle()
    assert slurm.count('sacct') == asked + 1  # only the preempted one
    assert summary.waiting == ['1001'] and summary.ended == ['1000']


def test_group_complete_when_all_jobs_are_final(manager, slurm):
    jobs = manager()
    jobs.add_group('batch1', 2)
    jobs.add_group('batch2', 1)
    jobs.cycle()
    slurm.states.update({'1000': 'COMPLETED', '1001': 'FAILED', '1002': 'RUNNING'})
    summary = jobs.cycle()
    assert summary.complete == ['batch1'] and summary.incomplete == ['batch2']


def test_scheduler_error_keeps_the_previous_state(manager, slurm):
    jobs = manager()
    jobs.add_group('batch1', 1)
    jobs.cycle()
    slurm.states['1000'] = 'RUNNING'
    jobs.cycle()
    slurm.fail['sacct'] = JobSchedulerError('Command failed with return code 1: sacct: error')
    summary = jobs.cycle()
    job = jobs.jobs['1000']
    assert job['state'] == 'RUNNING' and job['active'] is True and 'return code 1' in job['error']
    assert summary.waiting == ['1000'] and summary.uncertain == ['1000']
    del slurm.fail['sacct']
    slurm.states['1000'] = 'COMPLETED'
    summary = jobs.cycle()
    assert jobs.jobs['1000']['error'] is None and summary.ended == ['1000']
    assert summary.uncertain == [] and summary.complete == ['batch1']


def test_polling_away_from_hpc(manager, slurm):
    jobs = manager()
    jobs.add_group('batch1', 1)
    jobs.cycle()
    slurm.fail['sacct'] = missing('sacct')
    with pytest.raises(NotOnHPCError):
        jobs.cycle()


def test_held_htcondor_job(run_dir, manager):
    condor = SimulatedCondor()
    submit_jobs(layout.job_dir(run_dir, 'batch1'), condor, 1)
    jobs = manager(condor)
    summary = jobs.cycle()
    assert summary.waiting == ['591']
    condor.states['591'] = '5'
    summary = jobs.cycle()
    assert summary.held == {'591': {'state': '5', 'reason': None, 'detail': None,
                                    'message': 'held, reason to be checked'}}
    assert summary.waiting == summary.attention == [] and summary.incomplete == ['batch1']
    assert jobs.jobs['591']['active'] is True  # still in the queue
    condor.states['591'] = '4'
    summary = jobs.cycle()
    assert summary.ended == ['591'] and summary.held == {} and summary.complete == ['batch1']


class SimulatedSGE(JobScheduler):
    """SGE whose commands are simulated: ``qsub`` numbers the jobs, the listing
    prints ``states``, the listing of each hold type the ids in ``holds``."""

    def __init__(self):
        super().__init__('sge')
        self.next_id, self.states, self.holds = 300, {}, {}
        self.calls = []

    def _run_command(self, cmd, input_data=None, cwd=None):
        self.calls.append(cmd)
        if cmd[0] == 'qsub':
            job_id = str(self.next_id)
            self.next_id += 1
            self.states[job_id] = 'qw'
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f'Your job {job_id} ("job.sh") has been submitted\n', stderr='')
        assert cmd[:3] == ['qstat', '-u', 'vito'], cmd
        if len(cmd) == 3:  # listing of the active jobs
            listed = list(self.states)
        else:
            assert cmd[3] == '-s' and cmd[4] in SGE_HOLD_TYPES, cmd
            listed = [j for j in self.states if j in self.holds.get(cmd[4], ())]
        lines = ''.join(f'{j} 0.5 job vito {self.states[j]} 10/28/2026 10:00:00 all.q 1\n'
                        for j in listed)
        return subprocess.CompletedProcess(cmd, 0, stdout=lines, stderr='')


def test_held_sge_jobs_with_their_reasons(run_dir, manager, monkeypatch):
    monkeypatch.setenv('USER', 'vito')
    sge = SimulatedSGE()
    jobs = manager(sge)
    jobs.add_group('batch1', 3)
    jobs.cycle()
    sge.states.update({'300': 'hqw', '301': 'hqw', '302': 'r'})
    sge.holds.update({'hj': {'300'}, 'hu': {'301'}})
    summary = jobs.cycle()
    assert summary.held == {
        '300': {'state': 'hqw', 'reason': 'dependency', 'detail': 'hj',
                'message': 'held, waiting for other jobs'},
        '301': {'state': 'hqw', 'reason': 'intervention', 'detail': 'hu',
                'message': 'held, needs an intervention to be released'}}
    assert summary.waiting == ['302'] and summary.attention == []
    # Held jobs are asked again, never replaced, never released.
    for _ in range(2):
        summary = manager(sge).cycle()
    assert sum(1 for cmd in sge.calls if cmd[0] == 'qsub') == 3
    assert not any(cmd[0] in ('qrls', 'qalter') for cmd in sge.calls)
    assert list(summary.held) == ['300', '301']
    sge.states['300'] = 'qw'  # the dependency ended: SGE released the hold
    summary = manager(sge).cycle()
    assert summary.waiting == ['300', '302'] and list(summary.held) == ['301']


# --- loop --------------------------------------------------------------------------

def test_run_until_no_job_is_waiting(manager, slurm, clock):
    # Each wait moves the jobs on: queued, then running, then ended.
    steps = iter([{'1000': 'PENDING', '1001': 'RUNNING'},
                  {'1000': 'RUNNING', '1001': 'COMPLETED'},
                  {'1000': 'FAILED'}])
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        clock.advance(seconds)
        slurm.states.update(next(steps))

    jobs = manager()
    jobs.add_group('batch1', 2)
    summary = jobs.run(1800, sleep=sleep)
    assert waits == [1800] * 3
    assert summary.ended == ['1000', '1001'] and summary.waiting == [] and summary.to_submit == 0
    assert summary.complete == ['batch1']


def test_run_does_not_wait_for_jobs_that_need_attention(manager, slurm, clock):
    jobs = manager()
    jobs.add_group('batch1', 2)
    jobs.cycle()
    slurm.states['1000'] = 'COMPLETED'
    del slurm.states['1001']
    clock.advance(WAIT)
    summary = jobs.run(1800, sleep=pytest.fail)
    assert summary.ended == ['1000'] and summary.attention == ['1001']
    assert summary.incomplete == ['batch1'] and slurm.count('sbatch') == 2


def test_run_stops_after_max_cycles(manager):
    jobs = manager()
    jobs.add_group('batch1', 1)
    waits = []
    summary = jobs.run(60, sleep=waits.append, max_cycles=3)
    assert waits == [60, 60] and summary.waiting == ['1000']


def test_dry_run(run_dir, manager):
    scheduler = JobScheduler('slurm', dry_run=True)
    jobs = manager(scheduler)
    jobs.add_group('batch1', 2)
    summary = jobs.run(60, sleep=pytest.fail)
    assert [cmd[0][0] for cmd in scheduler.commands] == ['sbatch', 'sbatch']
    assert summary.to_submit == 2 and summary.waiting == []
    assert not (run_dir / layout.JOBS_REGISTRY_FILE).exists()
    assert not (layout.job_dir(run_dir, 'batch1') / SUBMITTED_FILE).exists()
