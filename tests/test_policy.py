# Author: Vito Foderà
# Created: 2026-10-29

"""Tests of the policy of the relaunches (``triboworkflow.execution.policy``).

No scheduler is called and pw.x never runs: the run is the one of
``run_helpers``. Histories of attempts are written in the records by hand,
with the outputs where the policy reads the summary of pw.x; the
calculations are then reconciled and decided as the workflow does.
"""

import pytest

from run_helpers import (CONVERGED, JOB_DONE, JOBSCRIPT_DIGEST, MAX_SECONDS, MAX_STEPS,
                         NOT_CONVERGED, PARALLEL, SCF_MAX_SECONDS, SYSTEM, Run, header, pwi, text)
from triboworkflow.core import store
from triboworkflow.execution import layout
from triboworkflow.execution.archive import archive_attempt
from triboworkflow.execution.jobs import Launch
from triboworkflow.execution.manager import JobManager
from triboworkflow.execution.policy import (Decision, LaunchSetup, PolicyError, decide,
                                            level_values, manual_launch, unchanged_relaunches)
from triboworkflow.execution.reconcile import CalcState, Reconciler

LEVEL_2 = {'mixing_beta': 0.2, 'nstep': 200}
INITIAL = {'mixing_beta': 0.4, 'nstep': 150}
# Run command and jobscript of the next attempt, as in [execution].
COMMAND = 'pw.x'
SETUP = LaunchSetup(COMMAND, JOBSCRIPT_DIGEST)
OTHER_JOBSCRIPT = 'sha256:' + 'b' * 64


@pytest.fixture
def run(tmp_path):
    return Run(tmp_path)


def made(n, kind, level=1, process='crashed', stop='truncated', scf=None, geometry='n/a',
         job_state=None, values=None, initial=None, run_command=COMMAND,
         jobscript=JOBSCRIPT_DIGEST):
    """Attempt ``n`` of a history, with its output archived in ``attempts/<n>/``."""
    if level > 1:
        values, initial = values or LEVEL_2, initial or INITIAL
    return store.Attempt(n, job_state=job_state, status=store.Status(process, scf, geometry),
                         stop_reason=stop, output=store.AttemptFile(f'attempts/{n}/output.pwo'),
                         run_command=run_command, jobscript_digest=jobscript,
                         remedies={'kind': kind, 'level': level, 'values': values or {},
                                   'initial_values': initial or {}, 'restart': False})


def history(run, calc_id, *attempts, output=None):
    """Write the attempts in the record; ``output`` is the archived output of the last one."""
    record = run.record(calc_id)
    store.write_record(run.sdir, store.Record(**{**record.__dict__, 'attempts': attempts,
                                                 'status': attempts[-1].status}))
    if output is not None:
        path = run.path(calc_id) / 'attempts' / str(attempts[-1].n) / 'output.pwo'
        path.parent.mkdir(parents=True)
        path.write_text(output)


def decision(run, calc_id):
    return decide(run.dir, run.reconcile(calc_id), SETUP)


# --- levels of the remedies ----------------------------------------------------------------

@pytest.mark.parametrize('n0, second, third', [
    (100, 200, 250), (150, 200, 250), (180, 200, 250), (200, 200, 250), (220, 250, 250),
    (250, 250, 250), (300, 300, 300),
])
def test_nstep_of_the_levels(n0, second, third):
    initial = {'mixing_beta': 0.4, 'nstep': n0}
    assert level_values(2, 'relax', initial)['nstep'] == second
    assert level_values(3, 'relax', initial)['nstep'] == third


@pytest.mark.parametrize('beta0, second, third', [
    (0.7, 0.2, 0.1), (0.4, 0.2, 0.1), (0.3, 0.2, 0.1), (0.2, 0.2, 0.1), (0.15, 0.15, 0.1),
    (0.1, 0.1, 0.1), (0.05, 0.05, 0.05),
])
def test_mixing_beta_of_the_levels(beta0, second, third):
    assert level_values(2, 'scf', {'mixing_beta': beta0}) == {'mixing_beta': second}
    assert level_values(3, 'scf', {'mixing_beta': beta0}) == {'mixing_beta': third}


def test_level_one_and_beyond():
    assert level_values(1, 'relax', INITIAL) == {}
    for level in (0, 4, 2.0):
        with pytest.raises(ValueError, match='is not an integer from 1 to 3'):
            level_values(level, 'relax', INITIAL)


# --- simple states -------------------------------------------------------------------------

@pytest.mark.parametrize('state, action', [
    ('pending', 'wait'), ('running', 'wait'), ('held', 'wait'), ('completed', 'done'),
    ('manual', 'manual'), ('uncertain', 'uncertain'),
])
def test_states_without_a_launch(run, state, action):
    calc_id = run.calc()
    made_state = CalcState(SYSTEM, calc_id, state, 0, 'why', released=True)
    assert decide(run.dir, made_state, SETUP) == Decision(SYSTEM, calc_id, action, None, 'why',
                                                            True)


def test_new_calculation_gets_its_first_attempt(run):
    calc_id = run.calc()
    assert decision(run, calc_id) == Decision(SYSTEM, calc_id, 'launch',
                                              Launch.initial(SYSTEM, calc_id), None, True)


def test_attempt_not_taken_runs_again_as_it_was(run):
    taken, left = run.calc('aa'), run.calc('ab')
    first = run.group('g1', Launch.initial(SYSTEM, taken), Launch.initial(SYSTEM, left))[0]
    run.claim(taken, 1, first)
    run.poll(60, **{j: 'COMPLETED' for j in run.manager.group_jobs('g1')})
    result = decision(run, left)
    assert (result.action, result.launch) == ('launch', Launch.initial(SYSTEM, left))
    assert result.reason == 'attempt 1 not taken by the groups g1'


# --- unchanged relaunches ------------------------------------------------------------------

def test_two_unchanged_relaunches_per_level(run):
    calc_id = run.calc()
    history(run, calc_id, made(1, 'initial', job_state='TIMEOUT', process='killed'))
    result = decision(run, calc_id)
    assert (result.action, result.launch, result.reason, result.released) == \
        ('launch', Launch(SYSTEM, calc_id, 2, 'unchanged', 1), 'job TIMEOUT', True)
    history(run, calc_id, made(1, 'initial'), made(2, 'unchanged'))
    assert decision(run, calc_id).launch == Launch(SYSTEM, calc_id, 3, 'unchanged', 1)
    history(run, calc_id, made(1, 'initial'), made(2, 'unchanged'), made(3, 'unchanged'))
    result = decision(run, calc_id)
    assert (result.action, result.launch) == ('manual', None)
    assert result.reason == 'crashed; 2 unchanged relaunches at level 1 already made'


def test_count_comes_from_the_record_after_a_restart(run):
    calc_id = run.calc()
    history(run, calc_id, made(1, 'initial'), made(2, 'unchanged'), made(3, 'unchanged'))
    restarted = JobManager(run.dir, run.slurm, 3600, now=run.clock)
    state = Reconciler(run.dir, restarted, 300).calculation(SYSTEM, calc_id)
    assert decide(run.dir, state, SETUP).action == 'manual'


I, U1, U2, M1, R2 = ('initial', 1), ('unchanged', 1), ('unchanged', 2), ('manual', 1), \
    ('remedy', 2)


@pytest.mark.parametrize('launches, count', [
    ([I], 0), ([I, U1], 1), ([I, U1, U1], 2), ([I, U1, U1, M1], 0), ([I, U1, U1, M1, U1], 1),
    ([I, U1, R2], 0), ([I, R2, U2], 1), ([I, U1, U1, R2, U2, U2], 2),
])
def test_count_restarts_at_a_new_level_or_a_manual_relaunch(launches, count):
    attempts = [made(i, kind, level) for i, (kind, level) in enumerate(launches, 1)]
    assert unchanged_relaunches(attempts) == count


def test_max_seconds_runs_again_in_a_new_job(run):
    calc_id = run.calc()
    history(run, calc_id, made(1, 'initial', process='ended', stop='max_seconds'),
            output=text(header(parallel=PARALLEL), MAX_SECONDS, JOB_DONE))
    result = decision(run, calc_id)
    assert (result.action, result.launch, result.reason) == \
        ('launch', Launch(SYSTEM, calc_id, 2, 'unchanged', 1),
         'max_seconds; from scratch: prefix and outdir are not both given in input.pwi')


# --- restart from the output directory -----------------------------------------------------

def with_outdir(input_text):
    """The input with the prefix and the output directory that the workflow writes."""
    return input_text.replace('&CONTROL\n', "&CONTROL\n  prefix = 'p'\n  outdir = 'OUT'\n")


def outdir_files(run, calc_id, names=('data-file-schema.xml', 'charge-density.dat')):
    save = run.path(calc_id) / 'OUT' / 'p.save'
    save.mkdir(parents=True)
    for name in names:
        (save / name).write_text('')


STEP_LIMIT = text(header(mixing_beta=0.4, nstep=150), CONVERGED, MAX_STEPS, JOB_DONE)
RELAX = with_outdir(pwi('relax', nstep=150, mixing_beta=0.4))
# Stops that leave files written per process, with the parallelization printed.
TIME_LIMIT = text(header(parallel=PARALLEL), MAX_SECONDS, JOB_DONE)
RELAX_NOT_CONVERGED = text(header(mixing_beta=0.4, nstep=150, parallel=PARALLEL), NOT_CONVERGED,
                           JOB_DONE)
PER_PROCESS = 'from scratch: files written per process, read correctly only with the same ' \
              'parallelization: '


def at_time_limit(**changes):
    return made(1, 'initial', process='ended', stop='max_seconds', **changes)


@pytest.mark.parametrize('case, input_text, attempt, output, restart, why', [
    ('max_seconds', with_outdir(pwi()), at_time_limit(), TIME_LIMIT, True,
     'restart from OUT/p.save'),
    ('max_seconds with hdf5', with_outdir(pwi()), at_time_limit(), TIME_LIMIT, True,
     'restart from OUT/p.save'),
    ('max_seconds, other command', with_outdir(pwi()), at_time_limit(run_command='mpirun pw.x'),
     TIME_LIMIT, False, f"{PER_PROCESS}attempt 1 ran with 'mpirun pw.x', the next one with "
                        "'pw.x'"),
    ('max_seconds, no command', with_outdir(pwi()), at_time_limit(run_command=None),
     TIME_LIMIT, False, f'{PER_PROCESS}the run command of attempt 1 is not recorded'),
    # The same command with other resources of the job.
    ('max_seconds, other jobscript', with_outdir(pwi()), at_time_limit(jobscript=OTHER_JOBSCRIPT),
     TIME_LIMIT, False, f'{PER_PROCESS}attempt 1 ran with another jobscript'),
    ('max_seconds, no jobscript', with_outdir(pwi()), at_time_limit(jobscript=None),
     TIME_LIMIT, False, f'{PER_PROCESS}the jobscript of attempt 1 is not recorded'),
    ('max_seconds, no parallelization', with_outdir(pwi()), at_time_limit(), SCF_MAX_SECONDS,
     False, f'{PER_PROCESS}the output of attempt 1 does not print its parallelization'),
    ('max_steps, other command', RELAX,
     made(1, 'initial', process='ended', stop='max_steps', scf='converged', geometry='max_steps',
          run_command='mpirun pw.x', jobscript=OTHER_JOBSCRIPT), STEP_LIMIT, True,
     'restart from OUT/p.save'),
    ('scf_not_converged in a relax', RELAX,
     made(1, 'initial', process='ended', stop='scf_not_converged', scf='not_converged',
          geometry='not_converged'), RELAX_NOT_CONVERGED, True, 'restart from OUT/p.save'),
    ('scf_not_converged in a relax, other command', RELAX,
     made(1, 'initial', process='ended', stop='scf_not_converged', scf='not_converged',
          geometry='not_converged', run_command='srun pw.x'), RELAX_NOT_CONVERGED, False,
     f"{PER_PROCESS}attempt 1 ran with 'srun pw.x', the next one with 'pw.x'"),
    ('scf_not_converged in an scf', with_outdir(pwi(mixing_beta=0.4)),
     made(1, 'initial', process='ended', stop='scf_not_converged', scf='not_converged'),
     text(header(mixing_beta=0.4), NOT_CONVERGED, JOB_DONE), False,
     'from scratch: an scf calculation without convergence is not continued'),
    ('files missing', with_outdir(pwi()), at_time_limit(), TIME_LIMIT, False,
     'from scratch: charge-density.dat or charge-density.hdf5 not found in OUT/p.save'),
    ('killed', with_outdir(pwi()), made(1, 'initial', job_state='TIMEOUT', process='killed'),
     None, False, None),
])
def test_restart_only_after_a_stop_that_pw_x_closed(run, case, input_text, attempt, output,
                                                    restart, why):
    calc_id = run.calc(input_text=input_text)
    names = {'files missing': ('data-file-schema.xml',),
             'max_seconds with hdf5': ('data-file-schema.xml', 'charge-density.hdf5')}
    outdir_files(run, calc_id, names.get(case, ('data-file-schema.xml', 'charge-density.dat')))
    history(run, calc_id, attempt, output=output)
    result = decision(run, calc_id)
    assert result.action == 'launch' and result.launch.restart is restart
    stop = attempt.stop_reason if attempt.status.process == 'ended' else f'job {attempt.job_state}'
    assert result.reason == (stop if why is None else f'{stop}; {why}')
    assert result.launch.remedies()['restart'] is restart


def test_unchanged_relaunch_keeps_the_values_of_its_level(run):
    calc_id = run.calc(input_text=pwi('relax'))
    history(run, calc_id, made(1, 'initial', process='ended', stop='scf_not_converged',
                               scf='not_converged', geometry='not_converged'),
            made(2, 'remedy', level=2, job_state='NODE_FAIL', process='killed'))
    launch = decision(run, calc_id).launch
    assert launch == Launch(SYSTEM, calc_id, 3, 'unchanged', 2, LEVEL_2, INITIAL)


# --- remedies ------------------------------------------------------------------------------

def not_converged(beta, nstep=None):
    return text(header(mixing_beta=beta, nstep=nstep), NOT_CONVERGED, JOB_DONE)


def test_scf_not_converged_goes_to_level_two(run):
    calc_id = run.calc()
    history(run, calc_id, made(1, 'initial', process='ended', stop='scf_not_converged',
                               scf='not_converged'), output=not_converged(0.7))
    result = decision(run, calc_id)
    assert (result.action, result.reason) == \
        ('launch', 'scf_not_converged; from scratch: an scf calculation without convergence is '
                   'not continued')
    assert result.launch == Launch(SYSTEM, calc_id, 2, 'remedy', 2, {'mixing_beta': 0.2},
                                   {'mixing_beta': 0.7})


def test_relax_at_the_step_limit_changes_both_values(run):
    calc_id = run.calc(input_text=pwi('relax', nstep=150, mixing_beta=0.4))
    output = text(header(mixing_beta=0.4, nstep=150), CONVERGED, MAX_STEPS, JOB_DONE)
    history(run, calc_id, made(1, 'initial', process='ended', stop='max_steps', scf='converged',
                               geometry='max_steps'), output=output)
    result = decision(run, calc_id)
    assert result.launch == Launch(SYSTEM, calc_id, 2, 'remedy', 2, LEVEL_2, INITIAL)


def test_level_three_from_the_recorded_initial_values(run):
    calc_id = run.calc(input_text=pwi('relax'))
    history(run, calc_id, made(1, 'initial', process='ended', stop='max_steps'),
            made(2, 'remedy', level=2, process='ended', stop='scf_not_converged',
                 scf='not_converged', geometry='not_converged',
                 initial={'mixing_beta': 0.15, 'nstep': 220}))
    result = decision(run, calc_id)
    assert result.launch == Launch(SYSTEM, calc_id, 3, 'remedy', 3,
                                   {'mixing_beta': 0.1, 'nstep': 250},
                                   {'mixing_beta': 0.15, 'nstep': 220})


def test_after_level_three_the_user_decides(run):
    calc_id = run.calc()
    history(run, calc_id, made(1, 'initial'), made(2, 'remedy', level=2),
            made(3, 'remedy', level=3, process='ended', stop='scf_not_converged',
                 scf='not_converged', values={'mixing_beta': 0.1}))
    result = decision(run, calc_id)
    assert (result.action, result.reason) == \
        ('manual', 'scf_not_converged at level 3, the last of the remedies')


@pytest.mark.parametrize('input_text, output, message', [
    (pwi(mixing_beta=0.35), not_converged(0.4),
     'mixing_beta = 0.35 in input.pwi, 0.4 in the summary of'),
    (pwi('relax', nstep=150), text(header(0.4, 200), NOT_CONVERGED, JOB_DONE),
     'nstep = 150 in input.pwi, 200 in the summary of'),
    (pwi(), text(header(), NOT_CONVERGED, JOB_DONE), 'the summary of pw.x does not print '
                                                     'mixing_beta'),
])
def test_effective_values_that_cannot_be_used(run, input_text, output, message):
    calc_id = run.calc(input_text=input_text)
    calculation = 'relax' if 'relax' in input_text else 'scf'
    history(run, calc_id, made(1, 'initial', process='ended', stop='scf_not_converged',
                               scf='not_converged',
                               geometry='not_converged' if calculation == 'relax' else 'n/a'),
            output=output)
    result = decision(run, calc_id)
    assert result.action == 'manual' and result.launch is None
    assert result.reason.startswith('scf_not_converged; no relaunch decided: ')
    assert message in result.reason


def test_mixing_beta_of_the_input_within_the_printed_decimals(run):
    calc_id = run.calc(input_text=pwi(mixing_beta=0.33333))
    history(run, calc_id, made(1, 'initial', process='ended', stop='scf_not_converged',
                               scf='not_converged'), output=not_converged(0.3333))
    assert decision(run, calc_id).launch.values == {'mixing_beta': 0.2}


# --- left to the user ----------------------------------------------------------------------

@pytest.mark.parametrize('attempt, reason', [
    (made(1, 'initial', process=None, stop=None), 'end of the process not known'),
    (made(1, 'initial', process='ended', stop='user_request'), 'stop reason user_request'),
    (made(1, 'initial', process='ended', stop='optimizer_failed', scf='converged',
          geometry='not_converged'), 'stop reason optimizer_failed'),
    (made(1, 'initial', process='ended', stop=None, scf='converged', geometry='not_converged'),
     'stop reason not recognized'),
    *[(made(1, 'initial', process='killed', job_state=state), f'job {state}')
      for state in ('CANCELLED by 1000', 'OUT_OF_MEMORY', 'DEADLINE')],
])
def test_left_to_the_user_also_after_a_restart(run, attempt, reason):
    calc_id = run.calc()
    history(run, calc_id, attempt)
    for manager in (run.manager, JobManager(run.dir, run.slurm, 3600, now=run.clock)):
        state = Reconciler(run.dir, manager, 300).calculation(SYSTEM, calc_id)
        assert decide(run.dir, state, SETUP) == Decision(SYSTEM, calc_id, 'manual', None,
                                                           reason, True)


def test_launch_not_recorded(run):
    calc_id = run.calc()
    history(run, calc_id, store.Attempt(1, status=store.Status('crashed'), stop_reason='truncated'))
    result = decision(run, calc_id)
    assert result.action == 'manual' and 'the launch of attempt 1 is not recorded' in result.reason


# --- manual relaunch -----------------------------------------------------------------------

def started(run, state='RUNNING'):
    calc_id = run.calc()
    (job_id,) = run.group('g1', Launch.initial(SYSTEM, calc_id))
    run.poll(**{job_id: 'RUNNING'})
    run.start(calc_id, job_id)
    if state != 'RUNNING':
        run.poll(60, **{job_id: state})
    return calc_id, job_id


def test_manual_relaunch_once_the_job_is_over(run):
    calc_id, job_id = started(run, 'CANCELLED by 1000')
    assert decision(run, calc_id).action == 'manual'
    launch = manual_launch(run.reconciler(), SYSTEM, calc_id, SETUP)
    assert launch == Launch(SYSTEM, calc_id, 2, 'manual', 1)
    # The manual attempt crashes: the count of unchanged relaunches starts again.
    history(run, calc_id, run.record(calc_id).attempts[0],
            made(2, 'manual', job_state='TIMEOUT', process='killed'))
    (run.path(calc_id) / layout.CLAIM_FILE).unlink()
    for path in (layout.RUN_INPUT_FILE, layout.OUTPUT_FILE):
        (run.path(calc_id) / path).unlink()
    assert decision(run, calc_id).launch == Launch(SYSTEM, calc_id, 3, 'unchanged', 1)


@pytest.mark.parametrize('job_state, found', [('RUNNING', 'is running'), ('PENDING', None)])
def test_manual_relaunch_of_a_running_attempt(run, job_state, found):
    calc_id, job_id = started(run, job_state)
    if found is None:  # requeued: its run is over, the attempt is fixed first
        assert manual_launch(run.reconciler(), SYSTEM, calc_id, SETUP).kind == 'manual'
    else:
        with pytest.raises(PolicyError, match=f'pes_scf/aa {found} .*its worker may still run'):
            manual_launch(run.reconciler(), SYSTEM, calc_id, SETUP)


def test_manual_relaunch_waits_for_the_job_of_the_claim(run):
    # The worker ended the attempt and may go on with other calculations.
    calc_id, job_id = started(run)
    run.finish(calc_id, status=store.Status('crashed', None, 'n/a'), stop_reason='truncated',
               exit_code=137)
    with pytest.raises(PolicyError, match=f'job {job_id} of the claim is running'):
        manual_launch(run.reconciler(), SYSTEM, calc_id, SETUP)
    run.poll(60, **{job_id: 'COMPLETED'})
    assert manual_launch(run.reconciler(), SYSTEM, calc_id, SETUP) == Launch(SYSTEM, calc_id, 2,
                                                                     'manual', 1)


def test_manual_relaunch_of_a_released_calculation(run):
    calc_id, job_id = started(run, 'CANCELLED')
    run.reconcile(calc_id)
    archive_attempt(run.dir, SYSTEM, calc_id)
    run.slurm.states[job_id] = 'RUNNING'  # whatever the job does now: the claim is gone
    assert manual_launch(run.reconciler(), SYSTEM, calc_id, SETUP).n == 2


def test_manual_relaunch_after_a_group_without_progress(run):
    calc_id = run.calc()
    (job_id,) = run.group('g1', Launch.initial(SYSTEM, calc_id))
    run.poll(60, **{job_id: 'FAILED'})
    assert run.reconcile(calc_id).state == 'manual'
    assert manual_launch(run.reconciler(), SYSTEM, calc_id, SETUP) == \
        Launch.initial(SYSTEM, calc_id)


def test_manual_relaunch_with_a_group_not_needed(run):
    calc_id, job_id = started(run, 'CANCELLED by 1000')
    with pytest.raises(PolicyError, match='group g1 not used: the launch is known'):
        manual_launch(run.reconciler(), SYSTEM, calc_id, SETUP, group='g1')


def test_manual_relaunch_takes_the_only_launch_of_the_groups(run):
    calc_id, job_id = started(run, 'CANCELLED by 1000')
    record = run.record(calc_id)  # attempt 1 recorded without its launch
    store.write_record(run.sdir, store.Record(**{
        **record.__dict__, 'attempts': (store.Attempt(**{**record.attempts[0].__dict__,
                                                         'remedies': {}}),)}))
    assert manual_launch(run.reconciler(), SYSTEM, calc_id, SETUP) == Launch(SYSTEM, calc_id, 2,
                                                                     'manual', 1)


def stopped(run, input_text, attempt, output):
    """A calculation whose last attempt ended and was archived, with its output."""
    calc_id = run.calc(input_text=input_text)
    history(run, calc_id, attempt, output=output)
    return calc_id


USER_STOP = made(1, 'initial', process='ended', stop='user_request')
SUMMARY = text(header(mixing_beta=0.4, nstep=150), JOB_DONE)


@pytest.mark.parametrize('level, values, initial', [
    (None, {}, {}), (1, {}, {}), (2, LEVEL_2, INITIAL),
    (3, {'mixing_beta': 0.1, 'nstep': 250}, INITIAL),
])
def test_manual_relaunch_at_a_level(run, level, values, initial):
    calc_id = stopped(run, pwi('relax', nstep=150, mixing_beta=0.4), USER_STOP, SUMMARY)
    launch = manual_launch(run.reconciler(), SYSTEM, calc_id, SETUP, level=level)
    assert launch == Launch(SYSTEM, calc_id, 2, 'manual', level or 1, values, initial)


def test_manual_relaunch_back_to_level_one(run):
    calc_id = run.calc(input_text=pwi('relax'))
    history(run, calc_id, made(1, 'initial'),
            made(2, 'remedy', level=2, process='ended', stop='user_request'))
    assert manual_launch(run.reconciler(), SYSTEM, calc_id, SETUP, level=1) == \
        Launch(SYSTEM, calc_id, 3, 'manual', 1)
    assert manual_launch(run.reconciler(), SYSTEM, calc_id, SETUP, level=3) == \
        Launch(SYSTEM, calc_id, 3, 'manual', 3, {'mixing_beta': 0.1, 'nstep': 250}, INITIAL)


@pytest.mark.parametrize('from_scratch, restart', [(False, True), (True, False)])
def test_manual_relaunch_continues_after_a_stop_by_the_user(run, from_scratch, restart, caplog):
    output = text(header(mixing_beta=0.4, nstep=150, parallel=PARALLEL), JOB_DONE)
    calc_id = stopped(run, with_outdir(pwi('relax', nstep=150, mixing_beta=0.4)), USER_STOP,
                      output)
    outdir_files(run, calc_id)
    assert decision(run, calc_id).action == 'manual'  # never relaunched by the workflow
    caplog.set_level('INFO')
    launch = manual_launch(run.reconciler(), SYSTEM, calc_id, SETUP, from_scratch=from_scratch)
    assert (launch.kind, launch.level, launch.restart) == ('manual', 1, restart)
    assert ('from scratch, as asked' if from_scratch else 'restart from OUT/p.save') in caplog.text


def test_manual_relaunch_after_a_stop_by_the_user_with_another_jobscript(run, caplog):
    output = text(header(mixing_beta=0.4, nstep=150, parallel=PARALLEL), JOB_DONE)
    attempt = made(1, 'initial', process='ended', stop='user_request', jobscript=OTHER_JOBSCRIPT)
    calc_id = stopped(run, with_outdir(pwi('relax', nstep=150, mixing_beta=0.4)), attempt, output)
    outdir_files(run, calc_id)
    caplog.set_level('INFO')
    assert manual_launch(run.reconciler(), SYSTEM, calc_id, SETUP).restart is False
    assert f'{PER_PROCESS}attempt 1 ran with another jobscript' in caplog.text


@pytest.mark.parametrize('from_scratch, restart', [(False, True), (True, False)])
def test_manual_relaunch_continues_unless_from_scratch(run, from_scratch, restart, caplog):
    attempt = made(1, 'initial', process='ended', stop='max_steps', scf='converged',
                   geometry='max_steps')
    calc_id = stopped(run, with_outdir(pwi('relax', nstep=150, mixing_beta=0.4)), attempt,
                      STEP_LIMIT)
    outdir_files(run, calc_id)
    caplog.set_level('INFO')
    launch = manual_launch(run.reconciler(), SYSTEM, calc_id, SETUP, level=2,
                           from_scratch=from_scratch)
    assert (launch.kind, launch.level, launch.restart) == ('manual', 2, restart)
    assert ('from scratch, as asked' if from_scratch else 'restart from OUT/p.save') in caplog.text


@pytest.mark.parametrize('level, message', [
    (4, 'level 4 is not an integer from 1 to 3'),
    (0, 'level 0 is not an integer from 1 to 3'),
    (2, 'no relaunch at level 2: .* the summary of pw.x does not print mixing_beta'),
])
def test_manual_relaunch_at_a_level_refused(run, level, message):
    calc_id = stopped(run, pwi(), USER_STOP, text(header(), JOB_DONE))
    with pytest.raises(PolicyError, match=message):
        manual_launch(run.reconciler(), SYSTEM, calc_id, SETUP, level=level)


def test_attempt_not_taken_keeps_its_launch(run):
    taken, left = run.calc('aa'), run.calc('ab')
    first = run.group('g1', Launch.initial(SYSTEM, taken), Launch.initial(SYSTEM, left))[0]
    run.claim(taken, 1, first)
    run.poll(60, **{j: 'FAILED' for j in run.manager.group_jobs('g1')})
    with pytest.raises(PolicyError, match='attempt 1 runs as listed, at level 1: no other level'):
        manual_launch(run.reconciler(), SYSTEM, left, SETUP, level=2)
    assert manual_launch(run.reconciler(), SYSTEM, left, SETUP, level=1,
                         from_scratch=True) == Launch.initial(SYSTEM, left)


@pytest.mark.parametrize('case, message', [
    ('completed', 'is completed: nothing to run again'),
    ('new', 'was never launched'),
    ('no job id', 'the claim gives no job'),
])
def test_manual_relaunch_refused(run, case, message):
    calc_id = run.calc()
    if case == 'completed':
        history(run, calc_id, made(1, 'initial', process='ended', stop='completed',
                                   scf='converged'))
    elif case == 'no job id':
        history(run, calc_id, made(1, 'initial'))
        run.claim(calc_id, 1, None)
    with pytest.raises(PolicyError, match=message):
        manual_launch(run.reconciler(), SYSTEM, calc_id, SETUP)
