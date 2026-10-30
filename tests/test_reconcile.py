# Author: Vito Foderà
# Created: 2026-10-29

"""Tests of the reconciliation of the calculations (``triboworkflow.execution.reconcile``).

No scheduler is called and pw.x never runs: the run is the one of
``run_helpers``, with Slurm simulated at the level of its commands, the real
job manager and calculation directories written in the state a worker
leaves. The other schedulers are read from a registry written by the test.
"""

import pytest

from run_helpers import (DROP, HEADER, POLL, SCF_DONE, SCF_MAX_SECONDS, SCF_TRUNCATED, SINCE,
                         SYSTEM, Run, text)
from triboworkflow.core import store
from triboworkflow.execution import layout
from triboworkflow.execution.archive import archive_attempt
from triboworkflow.execution.jobs import Launch
from triboworkflow.execution.reconcile import CalcState, Reconciler, attempt_outcome
from triboworkflow.execution.scheduler import JobScheduler, JobSchedulerError


@pytest.fixture
def run(tmp_path):
    return Run(tmp_path)


def started(run, state='RUNNING', **start):
    """A calculation with attempt 1 running in job 1000, in ``state``."""
    calc_id = run.calc()
    (job_id,) = run.group('g1', Launch.initial(SYSTEM, calc_id))
    run.poll(**{job_id: 'RUNNING'})
    run.start(calc_id, job_id, **start)
    if state != 'RUNNING':
        run.poll(60, **{job_id: state})
    return calc_id, job_id


# --- without attempts ------------------------------------------------------------------

def test_new_calculation(run):
    calc_id = run.calc()
    assert run.reconcile(calc_id) == CalcState(SYSTEM, calc_id, 'new', 0, released=True)


def test_first_attempt_queued(run):
    calc_id = run.calc()
    run.group('g1', Launch.initial(SYSTEM, calc_id))
    state = run.reconcile(calc_id)
    assert (state.state, state.groups, state.released) == ('pending', ('g1',), True)


def test_not_taken_by_a_group_that_made_progress(run):
    taken, left = run.calc('aa'), run.calc('ab')
    (job_id,) = run.group('g1', Launch.initial(SYSTEM, taken), Launch.initial(SYSTEM, left))[:1]
    run.poll(**{job_id: 'RUNNING'})
    run.start(taken, job_id)
    run.finish(taken, status=store.Status('ended', 'converged', 'n/a'), stop_reason='completed',
               clean_stop=True)
    run.poll(60, **{j: 'COMPLETED' for j in run.manager.group_jobs('g1')})
    assert run.reconcile(taken).state == 'completed'
    state = run.reconcile(left)
    assert (state.state, state.groups, state.attempts, state.released) == \
        ('not_taken', ('g1',), 0, True)


def test_group_without_progress_needs_the_user(run):
    calc_id = run.calc()
    (job_id,) = run.group('g1', Launch.initial(SYSTEM, calc_id))
    run.poll(60, **{job_id: 'FAILED'})  # the job ended and no worker took anything
    state = run.reconcile(calc_id)
    assert (state.state, state.groups) == ('manual', ('g1',))
    assert state.reason == (f'no progress: group g1 ended without any of its 1 attempts taken; '
                            f'jobs: {job_id} FAILED; logs of the jobs in '
                            f'{layout.job_dir(run.dir, "g1")}')


def test_progress_is_judged_on_the_last_group(run):
    # g1 made progress but did not reach ab; g2, for ab alone, made none.
    taken, left = run.calc('aa'), run.calc('ab')
    first = run.group('g1', Launch.initial(SYSTEM, taken), Launch.initial(SYSTEM, left))[0]
    run.poll(**{first: 'RUNNING'})
    run.start(taken, first)
    run.poll(60, **{j: 'TIMEOUT' for j in run.manager.group_jobs('g1')})
    assert run.reconcile(left).state == 'not_taken'
    (second,) = run.group('g2', Launch.initial(SYSTEM, left))
    assert run.reconcile(left).state == 'pending'
    run.poll(60, **{second: 'COMPLETED'})
    state = run.reconcile(left)
    assert (state.state, state.groups) == ('manual', ('g1', 'g2'))
    assert state.reason.startswith('no progress: group g2 ended')


def test_claim_counts_as_progress(run):
    taken, left = run.calc('aa'), run.calc('ab')
    first = run.group('g1', Launch.initial(SYSTEM, taken), Launch.initial(SYSTEM, left))[0]
    run.claim(taken, 1, first)  # taken, the worker stopped before the record
    run.poll(60, **{j: 'COMPLETED' for j in run.manager.group_jobs('g1')})
    assert run.reconcile(left).state == 'not_taken'


# --- table of the restart (27/10) --------------------------------------------------------

def test_job_queued_or_running_is_followed_without_copies(run):
    calc_id, job_id = started(run)
    before = run.files(calc_id), run.record(calc_id)
    state = run.reconcile(calc_id)
    assert (state.state, state.fixed) == ('running', False)
    assert (run.files(calc_id), run.record(calc_id)) == before


def test_finished_and_reusable_is_recorded_as_completed(run):
    # The worker was killed after pw.x closed the run, before writing its end.
    calc_id, job_id = started(run, 'TIMEOUT', output=SCF_DONE)
    state = run.reconcile(calc_id)
    assert (state.state, state.fixed, state.released) == ('completed', True, False)
    (attempt,) = run.record(calc_id).attempts
    assert attempt.status == run.record(calc_id).status == store.Status('ended', 'converged', 'n/a')
    assert (attempt.job_state, attempt.stop_reason, attempt.clean_stop) == ('TIMEOUT', 'completed',
                                                                          True)
    assert (attempt.exit_code, attempt.ended) == (None, None)
    path = run.path(calc_id)
    assert attempt.output == store.AttemptFile('output.pwo', store.file_digest(path / 'output.pwo'))
    assert attempt.input == store.AttemptFile('run.pwi', store.file_digest(path / 'run.pwi'))


@pytest.mark.parametrize('job_state, process, outcome', [
    ('TIMEOUT', 'killed', 'unchanged'),
    ('NODE_FAIL', 'killed', 'unchanged'),
    ('BOOT_FAIL', 'killed', 'unchanged'),
    ('COMPLETED', 'crashed', 'unchanged'),
    ('FAILED', 'crashed', 'unchanged'),
    ('CANCELLED by 1000', 'killed', 'manual'),
    ('CANCELLED+', 'killed', 'manual'),
    ('OUT_OF_MEMORY', 'killed', 'manual'),
    ('DEADLINE', 'killed', 'manual'),
])
def test_truncated_output_and_job_state(run, job_state, process, outcome):
    calc_id, job_id = started(run, job_state)
    files = run.files(calc_id)
    state = run.reconcile(calc_id)
    assert (state.state, state.reason, state.fixed) == (outcome, f'job {job_state}', True)
    attempt = run.record(calc_id).attempts[0]
    assert attempt.status == store.Status(process, None, 'n/a')
    assert (attempt.job_state, attempt.stop_reason, attempt.clean_stop) == (job_state, 'truncated',
                                                                          False)
    assert run.files(calc_id) == files  # nothing archived: the claim stays
    assert run.reconcile(calc_id) == CalcState(SYSTEM, calc_id, outcome, 1, f'job {job_state}')


def test_finished_and_incomplete_is_left_to_the_policy(run):
    calc_id, job_id = started(run)
    run.finish(calc_id, status=store.Status('crashed', None, 'n/a'), stop_reason='truncated',
               exit_code=137, clean_stop=False)
    record = run.record(calc_id)
    assert run.reconcile(calc_id) == CalcState(SYSTEM, calc_id, 'unchanged', 1, 'crashed')
    assert run.record(calc_id) == record


def test_job_no_longer_reported_is_uncertain(run):
    calc_id, job_id = started(run)
    del run.slurm.states[job_id]
    run.poll(60)
    files, record = run.files(calc_id), run.record(calc_id)
    state = run.reconcile(calc_id)
    assert state.state == 'uncertain' and 'not reported' in state.reason
    assert (run.files(calc_id), run.record(calc_id)) == (files, record)


def test_failed_query_keeps_the_last_known_state(run):
    calc_id, job_id = started(run)
    run.slurm.fail['State%30'] = JobSchedulerError('Command timed out after 30 seconds')
    run.poll(60)
    assert run.reconcile(calc_id).state == 'running'


def test_output_that_pw_did_not_write(run):
    calc_id, job_id = started(run, 'TIMEOUT', output='mpirun: error while loading\n')
    state = run.reconcile(calc_id)
    assert (state.state, state.reason) == ('manual', 'end of the process not known')
    assert run.record(calc_id).attempts[0].status == store.Status()


@pytest.mark.parametrize('output', [None, ''])
def test_no_output_from_the_job(run, output):
    calc_id, job_id = started(run, 'TIMEOUT', output=output)
    state = run.reconcile(calc_id)
    assert (state.state, state.reason) == ('unchanged', 'job TIMEOUT')
    assert run.record(calc_id).attempts[0].status == store.Status('killed')


ERROR = [' ' + '%' * 78, '     Error in routine  cdiaghg (309):',
         '     problems computing cholesky', ' ' + '%' * 78, '', '     stopping ...']
DETAIL = 'error in routine cdiaghg (309): problems computing cholesky'


def test_error_of_pw_needs_the_user_with_its_details(run):
    calc_id, job_id = started(run, 'FAILED', output=text(HEADER, ERROR))
    output = run.path(calc_id) / 'output.pwo'
    state = run.reconcile(calc_id)
    assert (state.state, state.reason) == ('manual', f'pw.x error: {DETAIL} (output: {output})')
    attempt = run.record(calc_id).attempts[0]
    assert (attempt.status.process, attempt.stop_reason) == ('crashed', 'error')
    assert output.read_text() == text(HEADER, ERROR)  # kept for the user


def test_error_of_pw_seen_by_the_worker_and_archived(run):
    calc_id, job_id = started(run, output=text(HEADER, ERROR))
    run.finish(calc_id, status=store.Status('crashed', None, 'n/a'), stop_reason='error',
               exit_code=1, clean_stop=False,
               output=store.AttemptFile('output.pwo',
                                        store.file_digest(run.path(calc_id) / 'output.pwo')))
    assert run.reconcile(calc_id).reason.startswith(f'pw.x error: {DETAIL}')
    archive_attempt(run.dir, SYSTEM, calc_id)  # e.g. by the user, to run it again
    state = run.reconcile(calc_id)
    archived = run.path(calc_id) / 'attempts' / '1' / 'output.pwo'
    assert (state.state, state.reason, state.released) == \
        ('manual', f'pw.x error: {DETAIL} (output: {archived})', True)


def test_output_closed_by_pw_decides_over_the_job(run):
    calc_id, job_id = started(run, 'CANCELLED by 1000', output=SCF_MAX_SECONDS)
    state = run.reconcile(calc_id)
    assert (state.state, state.reason) == ('unchanged', 'max_seconds')
    assert run.record(calc_id).attempts[0].job_state == 'CANCELLED by 1000'


# --- requeued jobs ---------------------------------------------------------------------

@pytest.mark.parametrize('job_state', ['PENDING', 'REQUEUED'])
def test_job_in_the_queue_again(run, job_state):
    calc_id, job_id = started(run, job_state)
    state = run.reconcile(calc_id)
    assert (state.state, state.reason) == ('unchanged', f'job {job_state}: requeued')
    attempt = run.record(calc_id).attempts[0]
    assert (attempt.status.process, attempt.job_state) == ('killed', job_state)


@pytest.mark.parametrize('recorded, restarts, expected', [
    (None, 0, 'running'), (None, 1, 'unchanged'), (0, 1, 'unchanged'), (1, 1, 'running'),
    (1, 3, 'unchanged'), (2, 1, 'uncertain'),
])
def test_restarts_of_the_job_against_the_claim(run, recorded, restarts, expected):
    calc_id, job_id = started(run, restart_count=recorded)
    run.slurm.restarts[job_id] = restarts
    state = run.reconcile(calc_id)
    assert state.state == expected
    process = run.record(calc_id).attempts[0].status.process
    assert process == ('killed' if expected == 'unchanged' else 'running')
    if expected == 'unchanged':
        assert state.reason == 'job RUNNING: requeued'


@pytest.mark.parametrize('claim', [{'restart_count': DROP}, {'restart_count': 'x'}])
def test_run_not_known_waits_for_the_end_of_the_job(run, claim):
    # A claim without a usable restart count: a later run of the job is not told apart.
    calc_id, job_id = started(run, **claim)
    run.slurm.restarts[job_id] = 1
    assert run.reconcile(calc_id).state == 'running'
    run.poll(60, **{job_id: 'TIMEOUT'})
    assert run.reconcile(calc_id).state == 'unchanged'


def test_restarts_not_read_waits(run):
    calc_id, job_id = started(run)
    run.slurm.fail['Restarts'] = JobSchedulerError('Command failed with return code 1: down')
    state = run.reconcile(calc_id)
    assert state.state == 'running' and 'restarts of job 1000 not read' in state.reason


def test_claim_without_job_id_is_uncertain(run):
    calc_id, job_id = started(run)
    record = run.record(calc_id)
    store.write_record(run.sdir, store.Record(**{**record.__dict__, 'attempts': (
        store.Attempt(**{**record.attempts[0].__dict__, 'job_id': None}),)}))
    run.claim(calc_id, 1, None)
    state = run.reconcile(calc_id)
    assert state.state == 'uncertain' and 'without a job id' in state.reason
    assert run.record(calc_id).attempts[0].status.process == 'running'


# --- preempted jobs --------------------------------------------------------------------

def test_preemption_confirmed_by_a_later_query(run):
    calc_id, job_id = started(run, 'PREEMPTED')
    state = run.reconcile(calc_id)
    assert state.state == 'running' and 'to be confirmed' in state.reason
    run.poll(POLL - 1)
    assert run.reconcile(calc_id).state == 'running'
    run.poll(1)
    state = run.reconcile(calc_id)
    assert (state.state, state.reason, state.fixed) == ('unchanged', 'job PREEMPTED', True)
    assert run.record(calc_id).attempts[0].status.process == 'killed'


def test_preempted_then_requeued(run):
    calc_id, job_id = started(run, 'PREEMPTED')
    assert run.reconcile(calc_id).state == 'running'
    run.poll(POLL, **{job_id: 'PENDING'})
    state = run.reconcile(calc_id)
    assert (state.state, state.reason) == ('unchanged', 'job PENDING: requeued')


def test_preempted_again_after_a_requeue(run):
    calc_id, job_id = started(run, 'PREEMPTED')
    run.slurm.restarts[job_id] = 1
    run.poll(POLL)
    assert run.reconcile(calc_id).state == 'unchanged'


def test_preemption_without_restart_count_is_uncertain(run):
    calc_id, job_id = started(run, 'PREEMPTED', restart_count=DROP)
    run.poll(POLL)
    state = run.reconcile(calc_id)
    assert state.state == 'uncertain' and 'no restart count' in state.reason
    assert run.record(calc_id).attempts[0].status.process == 'running'


# --- claims without attempt, files without claim ------------------------------------------

def test_claim_of_an_attempt_not_in_the_record(run):
    # The worker stopped between the claim and the record.
    calc_id = run.calc()
    (job_id,) = run.group('g1', Launch.initial(SYSTEM, calc_id))
    run.poll(**{job_id: 'RUNNING'})
    run.claim(calc_id, 1, job_id)
    assert run.reconcile(calc_id).state == 'running'
    run.poll(60, **{job_id: 'TIMEOUT'})
    state = run.reconcile(calc_id)
    assert (state.state, state.reason, state.attempts, state.fixed) == \
        ('unchanged', 'job TIMEOUT', 1, True)
    (attempt,) = run.record(calc_id).attempts
    assert attempt == store.Attempt(1, job_id=job_id, host='node01', started=SINCE,
                                    job_state='TIMEOUT', status=store.Status('killed'),
                                    remedies={'kind': 'initial', 'level': 1, 'values': {},
                                              'initial_values': {}})
    assert archive_attempt(run.dir, SYSTEM, calc_id) == 1  # archived as attempt 1, not 2


@pytest.mark.parametrize('case', ['empty claim', 'files', 'running', 'other n'])
def test_inconsistent_calculations_are_uncertain(run, case):
    calc_id = run.calc()
    path = run.path(calc_id)
    if case == 'empty claim':
        (path / layout.CLAIM_FILE).write_text('')
    elif case == 'files':
        (path / layout.OUTPUT_FILE).write_text(SCF_TRUNCATED)
    elif case == 'running':
        run.start(calc_id, '1000')
        (path / layout.CLAIM_FILE).unlink()
    else:
        run.claim(calc_id, 3, '1000')
    before = run.files(calc_id), run.record(calc_id)
    assert run.reconcile(calc_id).state == 'uncertain'
    assert (run.files(calc_id), run.record(calc_id)) == before


def test_job_not_followed_by_the_manager(run):
    calc_id = run.calc()
    run.start(calc_id, '4242')
    run.slurm.states['4242'] = 'RUNNING'
    assert run.reconcile(calc_id).state == 'running'
    run.slurm.states['4242'] = 'TIMEOUT'
    assert run.reconcile(calc_id).state == 'unchanged'
    assert '4242' not in run.manager.jobs


# --- released calculations ----------------------------------------------------------------

def test_released_calculation_then_new_group(run):
    calc_id, job_id = started(run, 'TIMEOUT')
    assert run.reconcile(calc_id).state == 'unchanged'
    archive_attempt(run.dir, SYSTEM, calc_id)
    # Released, no group for attempt 2 yet: the same decision is taken again.
    assert run.reconcile(calc_id) == CalcState(SYSTEM, calc_id, 'unchanged', 1, 'job TIMEOUT',
                                               released=True)
    (second,) = run.group('g2', Launch(SYSTEM, calc_id, 2, 'unchanged', 1))
    state = run.reconcile(calc_id)
    assert (state.state, state.groups) == ('pending', ('g2',))
    run.poll(60, **{second: 'RUNNING'})
    run.start(calc_id, second, kind='unchanged')
    assert run.reconcile(calc_id).state == 'running'


# --- outcome of an ended attempt -----------------------------------------------------------

def attempt(process, scf=None, stop=None, job_state=None, geometry='n/a'):
    return store.Attempt(1, status=store.Status(process, scf, geometry), stop_reason=stop,
                         job_state=job_state)


@pytest.mark.parametrize('scheduler, made, expected', [
    ('slurm', attempt('ended', 'converged', 'completed'), ('completed', None)),
    ('slurm', attempt('ended', 'converged', 'completed', geometry='converged'),
     ('completed', None)),
    ('slurm', attempt('ended', 'not_converged', 'scf_not_converged'),
     ('remedy', 'scf_not_converged')),
    ('slurm', attempt('ended', 'converged', 'max_steps', geometry='max_steps'),
     ('remedy', 'max_steps')),
    ('slurm', attempt('ended', None, 'max_seconds'), ('unchanged', 'max_seconds')),
    ('slurm', attempt('ended', None, 'user_request'), ('manual', 'stop reason user_request')),
    ('slurm', attempt('ended', 'converged', 'optimizer_failed', geometry='not_converged'),
     ('manual', 'stop reason optimizer_failed')),
    ('slurm', attempt('ended', 'converged', None, geometry='not_converged'),
     ('manual', 'stop reason not recognized')),
    ('slurm', attempt(None), ('manual', 'end of the process not known')),
    ('slurm', attempt('crashed', stop='error'), ('manual', 'pw.x error')),
    ('slurm', attempt('crashed', stop='truncated'), ('unchanged', 'crashed')),
    ('slurm', attempt('crashed'), ('unchanged', 'crashed')),
    ('slurm', attempt('crashed', stop='error', job_state='TIMEOUT'), ('manual', 'pw.x error')),
    ('slurm', attempt('killed', stop='truncated', job_state='TIMEOUT'),
     ('unchanged', 'job TIMEOUT')),
    ('slurm', attempt('killed', stop='truncated', job_state='RUNNING'),
     ('unchanged', 'job RUNNING: requeued')),
    ('slurm', attempt('killed', job_state='CANCELLED by 1000'),
     ('manual', 'job CANCELLED by 1000')),
    ('slurm', attempt('killed', job_state='OUT_OF_ME+'), ('manual', 'job OUT_OF_ME+')),
    ('pbs', attempt('crashed', stop='truncated', job_state='F'), ('unchanged', 'job F')),
    ('torque', attempt('crashed', stop='truncated', job_state='C'), ('unchanged', 'job C')),
    ('htcondor', attempt('crashed', stop='truncated', job_state='4'), ('unchanged', 'job 4')),
    ('htcondor', attempt('killed', stop='truncated', job_state='3'), ('manual', 'job 3')),
])
def test_attempt_outcome(scheduler, made, expected):
    assert attempt_outcome(made, JobScheduler(scheduler)) == expected


def test_running_attempt_has_no_outcome():
    with pytest.raises(ValueError, match='recorded as running'):
        attempt_outcome(attempt('running'), JobScheduler('slurm'))


# --- other schedulers ----------------------------------------------------------------------

class Registry:
    """Job manager reduced to a registry written by the test."""

    def __init__(self, scheduler, jobs):
        self.scheduler, self.jobs, self.groups = scheduler, jobs, {}

    def group_complete(self, name):
        return False


def registry_job(state, active, hold=None, error=None):
    return {'group': 'g', 'since': SINCE, 'state': state, 'state_since': SINCE, 'active': active,
            'hold': hold, 'reported': True, 'checked': SINCE, 'error': error}


@pytest.mark.parametrize('name, job_state, active, process, outcome', [
    ('pbs', 'F', False, 'crashed', 'unchanged'),
    ('torque', 'C', False, 'crashed', 'unchanged'),
    ('htcondor', '4', False, 'crashed', 'unchanged'),
    ('htcondor', '3', False, 'killed', 'manual'),
    ('pbs', 'Q', True, 'killed', 'unchanged'),
    ('htcondor', '1', True, 'killed', 'unchanged'),
    ('pbs', 'R', True, 'running', 'running'),
])
def test_states_of_other_schedulers(run, name, job_state, active, process, outcome):
    calc_id = run.calc()
    run.start(calc_id, '77')
    jobs = {'77': registry_job(job_state, active)}
    state = Reconciler(run.dir, Registry(JobScheduler(name), jobs), POLL).calculation(SYSTEM,
                                                                                      calc_id)
    assert state.state == outcome
    assert run.record(calc_id).attempts[0].status.process == process


@pytest.mark.parametrize('name, job_state, reason, message', [
    ('pbs', 'H', None, 'held, reason to be checked'),
    ('torque', 'H', None, 'held, reason to be checked'),
    ('sge', 'hqw', 'dependency', 'held, waiting for other jobs'),
    ('sge', 'hRwq', 'intervention', 'held, needs an intervention to be released'),
    ('htcondor', '5', None, 'held, reason to be checked'),
])
@pytest.mark.parametrize('error', [None, 'Command timed out after 30 seconds'])
def test_held_job_is_kept_without_relaunch(run, name, job_state, reason, message, error):
    calc_id = run.calc()
    run.start(calc_id, '77')
    files, record = run.files(calc_id), run.record(calc_id)
    jobs = {'77': registry_job(job_state, True, {'reason': reason, 'detail': None}, error)}
    state = Reconciler(run.dir, Registry(JobScheduler(name), jobs), POLL).calculation(SYSTEM,
                                                                                      calc_id)
    assert (state.state, state.fixed, state.released) == ('held', False, False)
    assert (message if error is None else 'the last query failed') in state.reason
    assert (run.files(calc_id), run.record(calc_id)) == (files, record)


def test_held_job_of_a_claim_without_attempt(run):
    calc_id = run.calc()
    run.claim(calc_id, 1, '77')
    jobs = {'77': registry_job('5', True, {'reason': None, 'detail': None})}
    state = Reconciler(run.dir, Registry(JobScheduler('htcondor'), jobs), POLL).calculation(
        SYSTEM, calc_id)
    assert state.state == 'held' and run.record(calc_id).attempts == ()
