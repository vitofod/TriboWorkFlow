# Author: Vito Foderà
# Created: 2026-10-27

"""Tests of the batch scheduler commands (``triboworkflow.execution.scheduler``).

No scheduler is called: ``subprocess.run`` is replaced by a fake that records
the commands and returns synthetic outputs. The column listings are shaped on
the columns that the parser reads; the listings of named attributes and the
states of single jobs on the formats of the manuals of Slurm, OpenPBS, Torque
and HTCondor.
"""

import json
import subprocess

import pytest

from triboworkflow.execution import scheduler as scheduler_module
from triboworkflow.execution.scheduler import (JobScheduler, JobSchedulerError, JobState,
                                               JobSubmissionError)


class FakeRun:
    """Replacement of ``subprocess.run``: records the calls, returns ``stdout`` or raises ``error``.

    ``stdout`` may be a list: one output per call, in order.
    """

    def __init__(self, stdout='', error=None):
        self.outputs = list(stdout) if isinstance(stdout, list) else None
        self.stdout, self.error, self.calls = stdout, error, []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        if self.error is not None:
            raise self.error
        stdout = self.outputs.pop(0) if self.outputs is not None else self.stdout
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr='')


@pytest.fixture
def fake_run(monkeypatch):
    def install(stdout='', error=None):
        fake = FakeRun(stdout, error)
        monkeypatch.setattr(scheduler_module.subprocess, 'run', fake)
        return fake
    return install


@pytest.fixture
def jobscript(tmp_path):
    path = tmp_path / 'job.sh'
    path.write_text('#!/bin/bash\necho start\n')
    return path


@pytest.fixture(autouse=True)
def user(monkeypatch):
    monkeypatch.setenv('USER', 'vito')


# --- construction ------------------------------------------------------------------

def test_names_ignore_case():
    assert JobScheduler('SLURM').scheduler_name == 'slurm'


def test_unknown_scheduler():
    with pytest.raises(JobSchedulerError, match="Scheduler 'moab' is not supported"):
        JobScheduler('moab')


def test_custom_table_from_a_dict(monkeypatch, fake_run, jobscript):
    monkeypatch.setattr(scheduler_module, 'SCHEDULER_CONFIG', dict(scheduler_module.SCHEDULER_CONFIG))
    table = {'submit': ['mysub'], 'cancel': ['mydel'], 'id_regex': r'job (\d+)',
             'id_format': r'^\d+$', 'list_active_jobs': ['myq']}
    scheduler = JobScheduler(table)
    assert scheduler.scheduler_name == 'custom'
    assert scheduler_module.SCHEDULER_CONFIG['custom'] is table
    run = fake_run('job 42 queued')
    assert scheduler.submit_job(str(jobscript)) == '42'
    assert run.calls[0][0] == ['mysub', str(jobscript)]


# --- submission --------------------------------------------------------------------

@pytest.mark.parametrize('name, output, job_id', [
    ('slurm', 'Submitted batch job 123456\n', '123456'),
    ('pbs', '4567.pbs-server\n', '4567.pbs-server'),
    ('torque', '4568.torque01.cluster\n', '4568.torque01.cluster'),
    ('lsf', 'Job <789> is submitted to queue <normal>.\n', '789'),
    ('sge', 'Your job 321 ("job.sh") has been submitted\n', '321'),
    ('htcondor', '1 job(s) submitted to cluster 55.\n', '55'),
    ('condor', '1 job(s) submitted to cluster 56.\n', '56'),
])
def test_submit_returns_the_printed_id(fake_run, jobscript, name, output, job_id):
    fake_run(output)
    assert JobScheduler(name).submit_job(str(jobscript)) == job_id


def test_submit_command_and_directory(fake_run, jobscript, tmp_path):
    run = fake_run('Submitted batch job 7')
    JobScheduler('slurm').submit_job(str(jobscript), script_args=['a', 'b'], cwd=tmp_path)
    cmd, kwargs = run.calls[0]
    assert cmd == ['sbatch', str(jobscript), 'a', 'b']
    assert kwargs['cwd'] == tmp_path
    assert kwargs['check'] is True and kwargs['timeout'] == 30


def test_submit_makes_the_path_absolute(fake_run, jobscript, monkeypatch):
    monkeypatch.chdir(jobscript.parent)
    run = fake_run('Submitted batch job 7')
    JobScheduler('slurm').submit_job('job.sh')
    assert run.calls[0][0] == ['sbatch', str(jobscript)]


def test_lsf_submits_on_stdin_with_the_arguments_exported(fake_run, jobscript):
    run = fake_run('Job <1> is submitted to queue <normal>.')
    JobScheduler('lsf').submit_job(str(jobscript), script_args=['in.pwi', 'out.pwo'])
    cmd, kwargs = run.calls[0]
    assert cmd == ['bsub']
    assert kwargs['input'] == 'export ARG1=in.pwi\nexport ARG2=out.pwo\n#!/bin/bash\necho start\n'


@pytest.mark.parametrize('path, args, message', [
    ('', None, 'Script path must be a non-empty string'),
    ('missing.sh', None, "missing.sh' not found"),
    ('JOB', 'a b', 'script_args must be a list of strings'),
    ('JOB', ['a', 1], 'All script_args must be strings'),
])
def test_submit_rejects_bad_arguments(fake_run, jobscript, path, args, message):
    run = fake_run('Submitted batch job 7')
    path = str(jobscript) if path == 'JOB' else path
    with pytest.raises(JobSchedulerError, match=message):
        JobScheduler('slurm').submit_job(path, script_args=args)
    assert run.calls == []


def test_submit_without_an_id_in_the_output(fake_run, jobscript):
    fake_run('sbatch: queued somewhere')
    with pytest.raises(JobSchedulerError, match='Could not extract job ID from output: sbatch: queued'):
        JobScheduler('slurm').submit_job(str(jobscript))


# --- errors of the commands ------------------------------------------------------------

def test_missing_command_keeps_the_cause(fake_run, jobscript):
    fake_run(error=FileNotFoundError(2, 'No such file or directory', 'sbatch'))
    with pytest.raises(JobSchedulerError, match="Command 'sbatch' not found") as error:
        JobScheduler('slurm').submit_job(str(jobscript))
    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_failed_command(fake_run, jobscript):
    fake_run(error=subprocess.CalledProcessError(1, ['sbatch'], stderr='invalid partition\n'))
    with pytest.raises(JobSchedulerError, match='Command failed with return code 1: invalid partition'):
        JobScheduler('slurm').submit_job(str(jobscript))


def test_timed_out_command(fake_run, jobscript):
    fake_run(error=subprocess.TimeoutExpired(['sbatch'], 5))
    with pytest.raises(JobSchedulerError, match='Command timed out after 5 seconds'):
        JobScheduler('slurm', timeout=5).submit_job(str(jobscript))


# Failures of a submission command: False if no job was created, None if one may have been.
@pytest.mark.parametrize('error, stdout, accepted', [
    (subprocess.CalledProcessError(1, ['sbatch'], output='', stderr='invalid partition\n'),
     '', False),
    (subprocess.CalledProcessError(1, ['sbatch'], output='Submitted batch job 77\n', stderr='x'),
     '', None),
    (subprocess.TimeoutExpired(['sbatch'], 5), '', None),
    (OSError('connection reset'), '', None),
    (None, 'sbatch: queued somewhere', None),
])
def test_submission_outcome(fake_run, jobscript, error, stdout, accepted):
    fake_run(stdout, error=error)
    with pytest.raises(JobSubmissionError) as raised:
        JobScheduler('slurm').submit_job(str(jobscript))
    assert raised.value.accepted is accepted


def test_nothing_run_is_not_a_failed_submission(fake_run, jobscript, tmp_path):
    # A missing command or jobscript: nothing was submitted, and no JobSubmissionError.
    fake_run(error=FileNotFoundError(2, 'No such file or directory', 'sbatch'))
    for path in (str(jobscript), str(tmp_path / 'missing.sh')):
        with pytest.raises(JobSchedulerError) as raised:
            JobScheduler('slurm').submit_job(path)
        assert not isinstance(raised.value, JobSubmissionError)


# --- cancellation --------------------------------------------------------------------

def test_cancel(fake_run):
    run = fake_run()
    assert JobScheduler('pbs').cancel_job('4567.pbs-server') is True
    assert run.calls[0][0] == ['qdel', '4567.pbs-server']


def test_cancel_failure_returns_false(fake_run):
    fake_run(error=subprocess.CalledProcessError(1, ['scancel'], stderr='invalid job id'))
    assert JobScheduler('slurm').cancel_job('99') is False


@pytest.mark.parametrize('job_id, message', [('', 'Job ID cannot be empty'),
                                             ('12a', 'Invalid job ID format: 12a')])
def test_cancel_rejects_bad_ids(fake_run, job_id, message):
    run = fake_run()
    with pytest.raises(JobSchedulerError, match=message):
        JobScheduler('slurm').cancel_job(job_id)
    assert run.calls == []


# --- job listings ---------------------------------------------------------------------

# Output, running, queued, active of the column listings. Lines whose first
# column is not a job id (headers, the array task 126_1) are skipped.
LISTINGS = {
    'slurm': ('123 normal job1 vito R 1:00 1 node01\n'
              '124 normal job2 vito PD 0:00 1 (Priority)\n'
              '125 normal job3 vito CG 2:00 1 node02\n'
              '126_1 normal arr vito R 1:00 1 node03\n',
              ['123', '125'], ['124'], ['123', '124', '125']),
    'lsf': ('JOBID USER STAT QUEUE FROM_HOST EXEC_HOST JOB_NAME SUBMIT_TIME\n'
            '789 vito RUN normal login1 node1 job1 Oct 27 10:00\n'
            '790 vito PEND normal login1 job2 Oct 27 10:01\n'
            '791 vito DONE normal login1 node2 job3 Oct 27 09:00\n',
            ['789'], ['790'], ['789', '790']),
    'sge': ('job-ID prior name user state submit/start at queue slots\n'
            '321 0.5 job1 vito r 10/27/2026 10:00:00 all.q@node1 1\n'
            '322 0.0 job2 vito qw 10/27/2026 10:01:00 1\n',
            ['321'], ['322'], ['321', '322']),
}


def pbs_json(jobs):
    """Output of ``qstat -f -F json`` (OpenPBS) for (id, owner, job_state) of each job."""
    return json.dumps({
        'timestamp': 1793180400, 'pbs_version': '23.06.06', 'pbs_server': 'server',
        'Jobs': {job_id: {'Job_Name': 'job', 'Job_Owner': owner, 'job_state': state,
                          'queue': 'workq'} for job_id, owner, state in jobs}}, indent=4)


def qstat_full(jobs):
    """Output of ``qstat -f -1`` (Torque) for (id, owner, job_state) of each job."""
    return ''.join(f'Job Id: {job_id}\n    Job_Name = job\n    Job_Owner = {owner}\n'
                   f'    job_state = {state}\n    queue = batch\n\n'
                   for job_id, owner, state in jobs)


# Listings of named attributes, shaped on the formats of the manuals: a
# running, a queued, a held and a finished job of the user and a running job
# of another user (PBS, Torque); for HTCondor ``condor_q vito -af:j JobStatus``
# with a running cluster of two processes.
OWN = [('4567.server', 'vito@login1', 'R'), ('4568.server', 'vito@login1', 'Q'),
       ('4569.server', 'vito@login1', 'H'), ('4571.server', 'elisa@login1', 'R')]
CONDOR_AF = '591.0 2\n592.0 1\n593.0 5\n594.0 2\n594.1 2\n'
NAMED = {
    'pbs': (pbs_json(OWN + [('4570.server', 'vito@login1', 'F')]),
            ['4567'], ['4568'], ['4567', '4568', '4569']),
    'torque': (qstat_full(OWN + [('4570.server', 'vito@login1', 'C')]),
               ['4567'], ['4568'], ['4567', '4568', '4569']),
    'htcondor': (CONDOR_AF, ['591', '594'], ['592', '593'], ['591', '592', '593', '594']),
    'condor': (CONDOR_AF, ['591', '594'], ['592', '593'], ['591', '592', '593', '594']),
}
ALL_LISTINGS = {**LISTINGS, **NAMED}


@pytest.mark.parametrize('name', ALL_LISTINGS)
def test_running_and_queued(fake_run, name):
    output, running, queued, _ = ALL_LISTINGS[name]
    fake_run(output)
    assert JobScheduler(name).get_lists_of_running_and_queued_job_ids() == (running, queued)


@pytest.mark.parametrize('name', ALL_LISTINGS)
def test_active_jobs(fake_run, name):
    output, _, _, active = ALL_LISTINGS[name]
    fake_run(output)
    assert JobScheduler(name).get_active_job_ids() == active


@pytest.mark.parametrize('name, cmd', [
    ('pbs', ['qstat', '-f', '-F', 'json']),
    ('torque', ['qstat', '-f', '-1']),
    ('htcondor', ['condor_q', 'vito', '-af:j', 'JobStatus']),
])
def test_listing_commands_of_named_attributes(fake_run, name, cmd):
    run = fake_run(NAMED[name][0])
    JobScheduler(name).get_active_job_ids()
    assert run.calls[0][0] == cmd


# The formats read by Xsorb: the alternative format of ``qstat -u`` (PBS,
# Torque) and the default grouped output of ``condor_q``, each with a running
# and a queued job.
QSTAT_U = ('server.example:\n'
           '                                                            Req\'d  Req\'d   Elap\n'
           'Job ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time\n'
           '--------------- -------- -------- ---------- ------ --- --- ------ ----- - -----\n'
           '1234.server     vito     workq    job1        12345   1   4    --  01:00 R 00:10\n'
           '1235.server     vito     workq    job2          --    1   4    --  01:00 Q   --\n')
CONDOR_Q_BATCH = ('-- Schedd: submit.example : <127.0.0.1:9618> @ 10/27/26 10:00:00\n'
                  'OWNER BATCH_NAME    SUBMITTED   DONE   RUN    IDLE  TOTAL JOB_IDS\n'
                  'vito  ID: 591      10/27 10:00      _      1      1      2 591.0-1\n')


@pytest.mark.parametrize('name, output', [
    ('pbs', QSTAT_U), ('torque', QSTAT_U), ('htcondor', CONDOR_Q_BATCH),
    ('pbs', ''), ('pbs', '[]'),
    ('pbs', pbs_json([('4567.server', 'vito@login1', 'R')]).replace('Job_Owner', 'owner')),
    ('torque', qstat_full([('4567.server', 'vito@login1', 'R')]).replace('job_state', 'state')),
])
def test_unexpected_listings_raise(fake_run, name, output):
    # A listing that cannot be read is never an empty queue.
    fake_run(output)
    scheduler = JobScheduler(name)
    for listing in (scheduler.get_active_job_ids,
                    scheduler.get_lists_of_running_and_queued_job_ids):
        with pytest.raises(JobSchedulerError):
            listing()


@pytest.mark.parametrize('name, output', [
    ('pbs', json.dumps({'timestamp': 1793180400, 'pbs_version': '23.06.06',
                        'pbs_server': 'server'})),
    ('torque', ''), ('htcondor', ''),
])
def test_listings_without_jobs(fake_run, name, output):
    fake_run(output)
    assert JobScheduler(name).get_lists_of_running_and_queued_job_ids() == ([], [])


def test_dry_run_listings_are_empty(fake_run):
    run = fake_run(error=AssertionError('no command may run in a dry run'))
    for name in ('pbs', 'torque', 'htcondor'):
        assert JobScheduler(name, dry_run=True).get_active_job_ids() == []
    assert run.calls == []


def test_custom_running_and_queued_raise(fake_run, monkeypatch):
    monkeypatch.setattr(scheduler_module, 'SCHEDULER_CONFIG', dict(scheduler_module.SCHEDULER_CONFIG))
    scheduler = JobScheduler({'submit': ['mysub'], 'cancel': ['mydel'], 'id_regex': r'(\d+)',
                              'id_format': r'^\d+$', 'list_active_jobs': ['myq']})
    run = fake_run('42 vito R\n43 vito Q\n')
    with pytest.raises(JobSchedulerError, match='cannot be told apart for a custom scheduler'):
        scheduler.get_lists_of_running_and_queued_job_ids()
    assert run.calls == []
    assert scheduler.get_active_job_ids() == ['42', '43']


@pytest.mark.parametrize('name, cmd', [
    ('slurm', ['squeue', '-h', '-u', 'vito', '-t', 'RUNNING,PENDING,CONFIGURING']),
    ('lsf', ['bjobs', '-u', 'vito']),
    ('sge', ['qstat', '-u', 'vito']),
])
def test_listing_command_with_the_user_name(fake_run, name, cmd):
    run = fake_run()
    JobScheduler(name).get_active_job_ids()
    assert run.calls[0][0] == cmd


def test_listing_without_user_name(fake_run, monkeypatch):
    monkeypatch.delenv('USER', raising=False)
    monkeypatch.delenv('USERNAME', raising=False)
    fake_run()
    with pytest.raises(JobSchedulerError, match='Cannot determine current username'):
        JobScheduler('slurm').get_active_job_ids()


def test_listing_failure_is_raised(fake_run):
    fake_run(error=FileNotFoundError(2, 'No such file or directory', 'squeue'))
    with pytest.raises(JobSchedulerError, match="Command 'squeue' not found"):
        JobScheduler('slurm').get_lists_of_running_and_queued_job_ids()


@pytest.mark.parametrize('name, job_id, key', [
    ('pbs', '4567.pbs-server', '4567'),
    ('torque', '4567', '4567'),
    ('slurm', '123', '123'),
    ('lsf', '789', '789'),
    ('htcondor', '591', '591'),
    ('htcondor', '591.0', '591'),
])
def test_job_key_matches_the_listings(name, job_id, key):
    assert JobScheduler(name).job_key(job_id) == key


@pytest.mark.parametrize('name, submitted', [
    ('pbs', '4567.server\n'), ('torque', '4567.server\n'),
    ('htcondor', '1 job(s) submitted to cluster 591.\n'),
])
def test_submitted_job_matches_the_parsed_listing(fake_run, jobscript, name, submitted):
    fake_run([submitted, NAMED[name][0]])
    scheduler = JobScheduler(name)
    job_id = scheduler.submit_job(str(jobscript))
    assert scheduler.job_key(job_id) in scheduler.get_active_job_ids()


# --- state of one job --------------------------------------------------------------

SACCT = ['sacct', '-X', '-n', '-o', 'State%30', '-j', '123']


@pytest.mark.parametrize('state, active', [
    ('PENDING', True), ('RUNNING', True), ('REQUEUED', True), ('RESIZING', True),
    ('SUSPENDED', True), ('CONFIGURING', True), ('COMPLETING', True), ('EXPEDITING', True),
    ('UPDATE_DB', True),
    ('COMPLETED', False), ('FAILED', False), ('TIMEOUT', False), ('NODE_FAIL', False),
    ('BOOT_FAIL', False), ('PREEMPTED', False), ('DEADLINE', False), ('CANCELLED', False),
    ('OUT_OF_MEMORY', False), ('CANCELLED by 1000', False),
    # Cut by the column width: the part before '+' is classified only if it is
    # a whole name, never by its prefix.
    ('CANCELLED+', False), ('OUT_OF_ME+', None), ('PEND+', None),
    # Flags not kept active by SCS, and a federation state: not known.
    ('REQUEUE_HOLD', None), ('SPECIAL_EXIT', None), ('STOPPED', None), ('REVOKED', None),
])
def test_slurm_state(fake_run, state, active):
    run = fake_run(f'{state:>10} \n')
    assert JobScheduler('slurm').get_job_state('123') == JobState('123', state, active)
    assert [call[0] for call in run.calls] == [SACCT]


def test_slurm_job_without_record(fake_run):
    fake_run('')
    assert JobScheduler('slurm').get_job_state('123') == JobState('123', None, None)


def test_slurm_more_records_raise(fake_run):
    fake_run('  RUNNING \n COMPLETED \n')
    with pytest.raises(JobSchedulerError, match='More than one record for job 123'):
        JobScheduler('slurm').get_job_state('123')


@pytest.mark.parametrize('name, state, active', [
    *[('pbs', s, True) for s in 'BEHQRSTUW'], ('pbs', 'F', False),
    ('pbs', 'M', None), ('pbs', 'X', None),
    *[('torque', s, True) for s in 'EHQRSTW'], ('torque', 'C', False), ('torque', 'F', None),
])
def test_pbs_and_torque_state(fake_run, name, state, active):
    output = (pbs_json if name == 'pbs' else qstat_full)([('4570.server', 'vito@login1', state)])
    run = fake_run(output)
    assert JobScheduler(name).get_job_state('4570.server') == \
        JobState('4570.server', state, active, held=state == 'H')
    expected = (['qstat', '-f', '-F', 'json', '-x', '4570.server'] if name == 'pbs'
                else ['qstat', '-f', '-1', '4570.server'])
    assert [call[0] for call in run.calls] == [expected]


@pytest.mark.parametrize('name, output', [
    ('pbs', pbs_json([('4571.server', 'vito@login1', 'R')])),
    ('torque', ''),
])
def test_pbs_and_torque_job_not_in_the_output(fake_run, name, output):
    fake_run(output)
    assert JobScheduler(name).get_job_state('4570.server') == JobState('4570.server', None, None)


def test_job_unknown_to_the_server_raises(fake_run):
    # qstat exits with an error for a job the server no longer knows.
    fake_run(error=subprocess.CalledProcessError(153, ['qstat'], stderr='qstat: Unknown Job Id'))
    with pytest.raises(JobSchedulerError, match='return code 153'):
        JobScheduler('torque').get_job_state('4570.server')


CONDOR_Q = ['condor_q', '591', '-af:j', 'JobStatus']
CONDOR_HISTORY = ['condor_history', '591', '-af:j', 'JobStatus']


@pytest.mark.parametrize('outputs, state, active, held, commands', [
    (['591.0 2\n'], '2', True, False, [CONDOR_Q]),
    # On hold: still in the queue, waiting for an intervention.
    (['591.0 5\n'], '5', True, True, [CONDOR_Q]),
    (['591.0 4\n'], '4', False, False, [CONDOR_Q]),
    (['', '591.0 4\n'], '4', False, False, [CONDOR_Q, CONDOR_HISTORY]),
    (['', '591.0 3\n'], '3', False, False, [CONDOR_Q, CONDOR_HISTORY]),
    (['', ''], None, None, False, [CONDOR_Q, CONDOR_HISTORY]),
    (['591.0 4\n591.1 2\n'], '4,2', True, False, [CONDOR_Q]),
    (['591.0 2\n591.1 5\n'], '2,5', True, True, [CONDOR_Q]),
    (['', '591.1 3\n591.0 4\n'], '3,4', False, False, [CONDOR_Q, CONDOR_HISTORY]),
    (['591.0 4\n591.1 9\n'], '4,9', None, False, [CONDOR_Q]),
])
def test_htcondor_state(fake_run, outputs, state, active, held, commands):
    run = fake_run(outputs)
    assert JobScheduler('htcondor').get_job_state('591') == JobState('591', state, active, held)
    assert [call[0] for call in run.calls] == commands


@pytest.mark.parametrize('name, state, held', [
    ('htcondor', '5', True), ('condor', '5', True), ('htcondor', '1', False),
    ('htcondor', None, False), ('slurm', 'PENDING', False), ('slurm', 'REQUEUE_HOLD', False),
    ('pbs', 'H', True), ('pbs', 'Q', False), ('torque', 'H', True), ('torque', 'W', False),
    ('sge', 'hqw', True), ('sge', 'hRwq', True), ('sge', 'qw', False), ('sge', 'r', False),
])
def test_held_states(name, state, held):
    assert JobScheduler(name).is_held(state) is held


@pytest.mark.parametrize('name, output', [
    ('pbs', json.dumps({'Jobs': {'4570.server': {'Job_Owner': 'vito@login1', 'job_state': 'H',
                                                 'Hold_Types': 'u'}}})),
    ('torque', 'Job Id: 4570.server\n    Job_Owner = vito@login1\n    job_state = H\n'
               '    Hold_Types = u\n'),
])
def test_pbs_and_torque_hold_types_kept_as_printed(fake_run, name, output):
    # Hold_Types is reported as printed, not interpreted: the reason is to be checked.
    fake_run(output)
    assert JobScheduler(name).get_job_state('4570.server') == \
        JobState('4570.server', 'H', True, True, None, 'u')


SGE_HELD = ('job-ID prior name user state submit/start at queue slots\n'
            '322 0.0 job2 vito hqw 10/27/2026 10:01:00 1\n')
SGE_HOLD_LINE = '322 0.0 job2 vito hqw 10/27/2026 10:01:00 1\n'


@pytest.mark.parametrize('listed_by, reason, detail', [
    (['hj'], 'dependency', 'hj'),
    (['hd'], 'dependency', 'hd'),
    (['ha'], 'start_time', 'ha'),
    (['hu'], 'intervention', 'hu'),
    (['hs', 'hj'], 'intervention', 'hs,hj'),
    ([], None, None),
])
def test_sge_hold_reason(fake_run, listed_by, reason, detail):
    types = ['hu', 'ho', 'hs', 'hd', 'hj', 'ha']
    run = fake_run([SGE_HELD] + [SGE_HOLD_LINE if t in listed_by else '' for t in types])
    assert JobScheduler('sge').get_job_state('322') == \
        JobState('322', 'hqw', True, True, reason, detail)
    assert [call[0] for call in run.calls[1:]] == \
        [['qstat', '-u', 'vito', '-s', t] for t in types]


def test_sge_job_not_held_needs_no_hold_listing(fake_run):
    run = fake_run(LISTINGS['sge'][0])
    assert JobScheduler('sge').get_job_state('322') == JobState('322', 'qw', True)
    assert len(run.calls) == 1


@pytest.mark.parametrize('name, state, final', [
    ('slurm', 'COMPLETED', True), ('slurm', 'CANCELLED+', True), ('slurm', 'PREEMPTED', False),
    ('slurm', 'RUNNING', False), ('slurm', 'OUT_OF_ME+', False), ('slurm', None, False),
    ('pbs', 'F', True), ('torque', 'C', True), ('htcondor', '4,3', True), ('htcondor', '4,2', False),
])
def test_final_states(name, state, final):
    assert JobScheduler(name).is_final(state) is final


def test_htcondor_unexpected_line_raises(fake_run):
    fake_run(['591.0 2 extra\n'])
    with pytest.raises(JobSchedulerError, match='Unexpected line in the output of HTCondor'):
        JobScheduler('htcondor').get_job_state('591')


@pytest.mark.parametrize('name, job_id, state, active', [
    ('lsf', '789', 'RUN', True), ('lsf', '790', 'PEND', True),
    ('lsf', '791', 'DONE', None), ('lsf', '999', None, None),
    ('sge', '321', 'r', True), ('sge', '322', 'qw', True), ('sge', '999', None, None),
])
def test_lsf_and_sge_state_from_the_listing(fake_run, name, job_id, state, active):
    # No history: a job missing from the listing of active jobs is not known.
    fake_run(LISTINGS[name][0])
    assert JobScheduler(name).get_job_state(job_id) == JobState(job_id, state, active)


def test_custom_state_raises(fake_run, monkeypatch):
    monkeypatch.setattr(scheduler_module, 'SCHEDULER_CONFIG', dict(scheduler_module.SCHEDULER_CONFIG))
    scheduler = JobScheduler({'submit': ['mysub'], 'cancel': ['mydel'], 'id_regex': r'(\d+)',
                              'id_format': r'^\d+$', 'list_active_jobs': ['myq']})
    run = fake_run('42 vito R\n')
    with pytest.raises(JobSchedulerError, match='not known for a custom scheduler'):
        scheduler.get_job_state('42')
    assert run.calls == []


def test_state_command_failure_raises(fake_run):
    fake_run(error=FileNotFoundError(2, 'No such file or directory', 'sacct'))
    with pytest.raises(JobSchedulerError, match="Command 'sacct' not found"):
        JobScheduler('slurm').get_job_state('123')


@pytest.mark.parametrize('name, commands', [
    ('slurm', [SACCT]),
    ('pbs', [['qstat', '-f', '-F', 'json', '-x', '123']]),
    ('htcondor', [['condor_q', '123', '-af:j', 'JobStatus'],
                  ['condor_history', '123', '-af:j', 'JobStatus']]),
])
def test_dry_run_state_is_not_known(fake_run, name, commands):
    run = fake_run(error=AssertionError('no command may run in a dry run'))
    scheduler = JobScheduler(name, dry_run=True)
    assert scheduler.get_job_state('123') == JobState('123', None, None)
    assert [command[0] for command in scheduler.commands] == commands
    assert run.calls == []


@pytest.mark.parametrize('name, state, queued', [
    ('slurm', 'PENDING', True), ('slurm', 'REQUEUED', True), ('slurm', 'RUNNING', False),
    ('slurm', 'SUSPENDED', False), ('slurm', 'PEND+', False), ('slurm', 'COMPLETED', False),
    ('pbs', 'Q', True), ('pbs', 'W', True), ('pbs', 'R', False), ('torque', 'Q', True),
    ('torque', 'C', False), ('lsf', 'PEND', True), ('lsf', 'RUN', False), ('sge', 'qw', True),
    ('sge', 'r', False), ('htcondor', '1', True), ('htcondor', '1,1', True),
    ('htcondor', '1,2', False), ('htcondor', '2', False), ('slurm', None, False),
    # Held jobs wait in the queue, but are not queued to run.
    ('pbs', 'H', False), ('torque', 'H', False), ('sge', 'hqw', False), ('sge', 'hRwq', False),
    ('htcondor', '5', False), ('htcondor', '1,5', False), ('lsf', 'PSUSP', False),
])
def test_queued_states(name, state, queued):
    assert JobScheduler(name).is_queued(state) is queued


@pytest.mark.parametrize('name, state, base', [
    ('slurm', 'CANCELLED by 1000', 'CANCELLED'), ('slurm', 'CANCELLED+', 'CANCELLED'),
    ('slurm', 'OUT_OF_ME+', 'OUT_OF_ME'), ('pbs', 'F', 'F'), ('htcondor', '4,3', '4,3'),
    ('slurm', None, None),
])
def test_state_names(name, state, base):
    assert JobScheduler(name).state_name(state) == base


def test_active_states():
    scheduler = JobScheduler('slurm')
    assert [scheduler.is_active(s) for s in ('RUNNING', 'TIMEOUT', 'OUT_OF_ME+', None)] == \
        [True, False, None, None]


# --- restarts of one job -------------------------------------------------------------

SACCT_RESTARTS = ['sacct', '-X', '-n', '-o', 'Restarts', '-j', '123']


@pytest.mark.parametrize('output, count', [('         0 \n', 0), ('  2\n', 2), ('', None),
                                           ('\n  \n', None)])
def test_slurm_restart_count(fake_run, output, count):
    run = fake_run(output)
    assert JobScheduler('slurm').get_restart_count('123') == count
    assert [call[0] for call in run.calls] == [SACCT_RESTARTS]


@pytest.mark.parametrize('output', ['  1 \n 2 \n', 'Unknown\n', '-1\n', '2+\n'])
def test_slurm_unexpected_restart_count_raises(fake_run, output):
    fake_run(output)
    with pytest.raises(JobSchedulerError, match='Not one number of restarts for job 123'):
        JobScheduler('slurm').get_restart_count('123')


def test_restart_count_command_failure_raises(fake_run):
    fake_run(error=subprocess.CalledProcessError(1, SACCT_RESTARTS, stderr='slurmdbd down'))
    with pytest.raises(JobSchedulerError, match='slurmdbd down'):
        JobScheduler('slurm').get_restart_count('123')


@pytest.mark.parametrize('name', ['pbs', 'torque', 'lsf', 'sge', 'htcondor', 'condor'])
def test_other_schedulers_do_not_tell_the_restarts(fake_run, name):
    run = fake_run(error=AssertionError('no command may run'))
    assert JobScheduler(name).get_restart_count('123') is None
    assert run.calls == []


def test_dry_run_restart_count_is_not_known(fake_run):
    run = fake_run(error=AssertionError('no command may run in a dry run'))
    scheduler = JobScheduler('slurm', dry_run=True)
    assert scheduler.get_restart_count('123') is None
    assert [command[0] for command in scheduler.commands] == [SACCT_RESTARTS]
    assert run.calls == []


# --- dry run -----------------------------------------------------------------------

def test_dry_run_records_and_runs_nothing(fake_run, jobscript, tmp_path):
    run = fake_run(error=AssertionError('no command may run in a dry run'))
    scheduler = JobScheduler('slurm', dry_run=True)
    assert scheduler.submit_job(str(jobscript), cwd=tmp_path) is None
    assert scheduler.cancel_job('12') is True
    assert scheduler.get_active_job_ids() == []
    assert scheduler.get_lists_of_running_and_queued_job_ids() == ([], [])
    assert run.calls == []
    assert scheduler.commands[0] == (['sbatch', str(jobscript)], None, tmp_path)
    assert [cmd[0] for cmd, _, _ in scheduler.commands] == ['sbatch', 'scancel', 'squeue', 'squeue']


def test_dry_run_still_checks_the_jobscript(fake_run):
    fake_run()
    with pytest.raises(JobSchedulerError, match='not found'):
        JobScheduler('slurm', dry_run=True).submit_job('missing.sh')


def test_dry_run_lsf_records_the_stdin(fake_run, jobscript):
    fake_run()
    scheduler = JobScheduler('lsf', dry_run=True)
    scheduler.submit_job(str(jobscript))
    assert scheduler.commands == [(['bsub'], '#!/bin/bash\necho start\n', None)]
