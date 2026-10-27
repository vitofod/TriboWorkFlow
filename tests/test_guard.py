# Author: Vito Foderà
# Created: 2026-10-27

"""Tests of the guard against submissions away from HPC (``triboworkflow.core.guard``).

``subprocess.run`` is replaced: no scheduler is called.
"""

import subprocess

import pytest

from triboworkflow.core.guard import NotOnHPCError, hpc_only
from triboworkflow.execution import scheduler as scheduler_module
from triboworkflow.execution.scheduler import JobScheduler, JobSchedulerError


@pytest.fixture
def jobscript(tmp_path):
    path = tmp_path / 'job.sh'
    path.write_text('#!/bin/bash\n')
    return str(path)


def raising(error):
    def run(cmd, **kwargs):
        raise error
    return run


def test_missing_submission_command(monkeypatch, jobscript):
    monkeypatch.setattr(scheduler_module.subprocess, 'run',
                        raising(FileNotFoundError(2, 'No such file or directory', 'sbatch')))
    with pytest.raises(NotOnHPCError) as error:
        with hpc_only('slurm'):
            JobScheduler('slurm').submit_job(jobscript)
    message = str(error.value)
    assert message.startswith("Command 'sbatch' not found: calculations run only on an HPC machine")
    assert "scheduler 'slurm' of [execution]" in message
    assert isinstance(error.value.__cause__, JobSchedulerError)


@pytest.mark.parametrize('error', [subprocess.CalledProcessError(1, ['sbatch'], stderr='bad partition'),
                                   subprocess.TimeoutExpired(['sbatch'], 30)])
def test_other_scheduler_errors_pass(monkeypatch, jobscript, error):
    monkeypatch.setattr(scheduler_module.subprocess, 'run', raising(error))
    with pytest.raises(JobSchedulerError):
        with hpc_only('slurm'):
            JobScheduler('slurm').submit_job(jobscript)


def test_missing_jobscript_is_not_a_missing_command():
    with pytest.raises(JobSchedulerError, match='not found'):
        with hpc_only('slurm'):
            JobScheduler('slurm').submit_job('missing.sh')


def test_other_errors_pass():
    with pytest.raises(KeyError):
        with hpc_only('slurm'):
            raise KeyError('x')


def test_dry_run_needs_no_scheduler(monkeypatch, jobscript):
    monkeypatch.setattr(scheduler_module.subprocess, 'run',
                        raising(AssertionError('no command may run in a dry run')))
    with hpc_only('slurm'):
        assert JobScheduler('slurm', dry_run=True).submit_job(jobscript) is None
