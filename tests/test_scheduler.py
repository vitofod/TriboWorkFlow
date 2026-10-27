# Author: Vito Foderà
# Created: 2026-10-27

"""Tests of the batch scheduler commands (``triboworkflow.execution.scheduler``).

No scheduler is called: ``subprocess.run`` is replaced by a fake that records
the commands and returns synthetic outputs. The job listings are shaped on the
columns that the parser reads.
"""

import subprocess

import pytest

from triboworkflow.execution import scheduler as scheduler_module
from triboworkflow.execution.scheduler import JobScheduler, JobSchedulerError


class FakeRun:
    """Replacement of ``subprocess.run``: records the calls, returns ``stdout`` or raises ``error``."""

    def __init__(self, stdout='', error=None):
        self.stdout, self.error, self.calls = stdout, error, []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        if self.error is not None:
            raise self.error
        return subprocess.CompletedProcess(cmd, 0, stdout=self.stdout, stderr='')


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

# Output, running, queued, active. Lines whose first column is not a job id
# (headers, the array task 126_1, the HTCondor id 57.0) are skipped. The
# lines of pbs and htcondor are shaped on the columns the parser reads, not on
# the outputs of their commands (see DOCUMENTED below).
LISTINGS = {
    'slurm': ('123 normal job1 vito R 1:00 1 node01\n'
              '124 normal job2 vito PD 0:00 1 (Priority)\n'
              '125 normal job3 vito CG 2:00 1 node02\n'
              '126_1 normal arr vito R 1:00 1 node03\n',
              ['123', '125'], ['124'], ['123', '124', '125']),
    'pbs': ('Job id Name User Time Use S Queue\n'
            '---------------- ---- ---- -------- - -----\n'
            '4567.server vito batch job1 1234 R 01:00\n'
            '4568.server vito batch job2 -- Q --\n'
            '4569.server vito batch job3 1235 F 02:00\n',
            ['4567'], ['4568'], ['4567', '4568']),
    'lsf': ('JOBID USER STAT QUEUE FROM_HOST EXEC_HOST JOB_NAME SUBMIT_TIME\n'
            '789 vito RUN normal login1 node1 job1 Oct 27 10:00\n'
            '790 vito PEND normal login1 job2 Oct 27 10:01\n'
            '791 vito DONE normal login1 node2 job3 Oct 27 09:00\n',
            ['789'], ['790'], ['789', '790']),
    'sge': ('job-ID prior name user state submit/start at queue slots\n'
            '321 0.5 job1 vito r 10/27/2026 10:00:00 all.q@node1 1\n'
            '322 0.0 job2 vito qw 10/27/2026 10:01:00 1\n',
            ['321'], ['322'], ['321', '322']),
    'htcondor': ('ID OWNER SUBMITTED RUN_TIME ST PRI SIZE CMD\n'
                 '55 vito 10/27 10:00 0+00:01:00 R 0 0.0 job.sh\n'
                 '56 vito 10/27 10:01 0+00:00:00 I 0 0.0 job.sh\n'
                 '57.0 vito 10/27 10:02 0+00:00:00 I 0 0.0 job.sh\n',
                 ['55'], ['56'], ['55', '56']),
}
READABLE = ('slurm', 'lsf', 'sge')

# Listings shaped on the documented output formats, each with one running and
# one queued job: the alternative format of ``qstat -u`` (PBS, Torque), the
# default grouped output of ``condor_q`` and the one of ``condor_q -nobatch``.
QSTAT_U = ('server.example:\n'
           '                                                            Req\'d  Req\'d   Elap\n'
           'Job ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time\n'
           '--------------- -------- -------- ---------- ------ --- --- ------ ----- - -----\n'
           '1234.server     vito     workq    job1        12345   1   4    --  01:00 R 00:10\n'
           '1235.server     vito     workq    job2          --    1   4    --  01:00 Q   --\n')
DOCUMENTED = {
    'pbs': QSTAT_U,
    'torque': QSTAT_U,
    'htcondor': ('-- Schedd: submit.example : <127.0.0.1:9618> @ 10/27/26 10:00:00\n'
                 'OWNER BATCH_NAME    SUBMITTED   DONE   RUN    IDLE  TOTAL JOB_IDS\n'
                 'vito  ID: 591      10/27 10:00      _      1      1      2 591.0-1\n'
                 '\n'
                 'Total for query: 2 jobs; 0 completed, 0 removed, 1 idle, 1 running, 0 held, '
                 '0 suspended\n'),
    'condor': (' ID      OWNER            SUBMITTED     RUN_TIME ST PRI SIZE CMD\n'
               ' 591.0   vito           10/27 10:00   0+00:01:00 R  0    0.0 job.sh\n'
               ' 591.1   vito           10/27 10:00   0+00:00:00 I  0    0.0 job.sh\n'),
}


@pytest.mark.parametrize('name', READABLE)
def test_running_and_queued(fake_run, name):
    output, running, queued, _ = LISTINGS[name]
    fake_run(output)
    assert JobScheduler(name).get_lists_of_running_and_queued_job_ids() == (running, queued)


@pytest.mark.parametrize('name', READABLE)
def test_active_jobs(fake_run, name):
    output, _, _, active = LISTINGS[name]
    fake_run(output)
    assert JobScheduler(name).get_active_job_ids() == active


@pytest.mark.parametrize('name', LISTINGS)
def test_parser_reads_its_columns(name):
    output, running, queued, active = LISTINGS[name]
    scheduler = JobScheduler(name)
    assert scheduler._split_running_and_queued(output) == (running, queued)
    assert scheduler._extract_active_job_ids(output) == active


@pytest.mark.parametrize('name', DOCUMENTED)
def test_documented_outputs_are_not_read(name):
    # A running and a queued job, and the lists come out empty.
    scheduler = JobScheduler(name)
    assert scheduler._split_running_and_queued(DOCUMENTED[name]) == ([], [])
    assert scheduler._extract_active_job_ids(DOCUMENTED[name]) == []


@pytest.mark.parametrize('name', ['pbs', 'torque', 'htcondor', 'condor'])
def test_unreadable_listings_raise(fake_run, name):
    run = fake_run(DOCUMENTED[name])
    scheduler = JobScheduler(name)
    for listing in (scheduler.get_active_job_ids,
                    scheduler.get_lists_of_running_and_queued_job_ids):
        with pytest.raises(JobSchedulerError, match=f'The job listing of {name} cannot be read'):
            listing()
    assert run.calls == []
    with pytest.raises(JobSchedulerError, match='cannot be read'):
        JobScheduler(name, dry_run=True).get_active_job_ids()


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
    ('sge', ['qstat', '-u', 'vito', '-s', 'r,qw']),
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
])
def test_job_key_matches_the_listings(name, job_id, key):
    assert JobScheduler(name).job_key(job_id) == key


def test_submitted_pbs_job_matches_the_parsed_listing(fake_run, jobscript):
    fake_run('4567.server\n')
    scheduler = JobScheduler('pbs')
    job_id = scheduler.submit_job(str(jobscript))
    assert scheduler.job_key(job_id) in scheduler._extract_active_job_ids(LISTINGS['pbs'][0])


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
