# Author: Vito Foderà
# Created: 2026-11-08

"""Tests of the analysis of a step in a job of its own (``triboworkflow.cli.analysis_job``).

The job is run in this process, as its line in the jobscript would run it on
a compute node; the step is fictitious and writes one entry of
``system.json``. The interpolation of the PES in a job is tested in
``test_plotpes.py``.
"""

import json

import pytest

from triboworkflow.cli import analysis_job
from triboworkflow.core.store import Provenance, SystemWriter, read_system
from triboworkflow.execution import layout
from triboworkflow.execution.jobs import (ANALYSIS_FILE, OUTCOME_FILE, WITHDRAWN_FILE, Analysis,
                                          write_analysis)
from triboworkflow.execution.lock import RunLock
from triboworkflow.execution.orchestrator import Step

SYSTEM = 'Cu111_Al111'
GROUP = f'{SYSTEM}.plotpes.1'
START, END = '2026-11-08T10:00:00+01:00', '2026-11-08T10:02:00+01:00'


class JobStep(Step):
    """A step analysed in a job: it reads the entry ``pes`` and writes ``plotpes``."""

    name = 'plotpes'
    manual_analysis = True

    def __init__(self, done=False, error=None):
        self.is_done, self.error = done, error
        self.calls = []

    def done(self, run_dir, system):
        self.calls.append('done')
        return self.is_done

    def expected(self, run_dir, system):
        return {('pes',): Provenance.build('pes', {})}

    def analyse(self, run_dir, system, calculations):
        self.calls.append(('analyse', calculations))
        if self.error is not None:
            raise self.error
        write_entry(run_dir, 'plotpes')


def write_entry(run_dir, name):
    with SystemWriter(layout.system_dir(run_dir, SYSTEM)) as writer:
        if not writer.document.exists:
            writer.document.create({})
        writer.document.set([name], {'done': True})
        writer.document.set_provenance([name], Provenance.build(name, {}))
        writer.save()


@pytest.fixture
def run_dir(tmp_path):
    """A run whose pes is done, with the group of the analysis of plotpes."""
    run = tmp_path / 'run'
    write_entry(run, 'pes')
    jobscript = tmp_path / 'analysis.sh'
    jobscript.write_text('#!/bin/bash\n#SBATCH --mem=200G\n')
    write_analysis(run, Analysis(name=GROUP, system=SYSTEM, step='plotpes', scheduler='slurm'),
                   jobscript)
    return run


def analysis_path(run_dir, group=GROUP):
    return layout.job_dir(run_dir, group) / ANALYSIS_FILE


def run_job(run_dir, step, group=GROUP):
    times = iter([START, END])
    return analysis_job.run_analysis(analysis_path(run_dir, group), get_step=lambda name: step,
                                     environ={'SLURM_JOB_ID': '4242'}, now=lambda: next(times))


def outcome_file(run_dir, group=GROUP):
    return json.loads((layout.job_dir(run_dir, group) / OUTCOME_FILE).read_text())


def test_the_step_is_analysed_in_the_job(run_dir):
    step = JobStep()
    # The orchestrator in the background holds the lock of the run for its whole life.
    with RunLock(run_dir, 'triboworkflow start'):
        lock = (run_dir / 'run.lock').read_bytes()
        outcome = run_job(run_dir, step)
        assert (run_dir / 'run.lock').read_bytes() == lock  # never taken nor changed
    assert step.calls == ['done', ('analyse', [])]
    assert read_system(layout.system_dir(run_dir, SYSTEM)).get(['plotpes']) == {'done': True}
    assert outcome == outcome_file(run_dir)
    assert {key: outcome[key] for key in ('schema_version', 'group', 'system', 'step', 'job_id',
                                          'started', 'ended', 'result', 'error')} == {
        'schema_version': 1, 'group': GROUP, 'system': SYSTEM, 'step': 'plotpes',
        'job_id': '4242', 'started': START, 'ended': END, 'result': 'analysed', 'error': None}
    assert isinstance(outcome['host'], str) and isinstance(outcome['pid'], int)
    # Neither the state of the orchestrator nor the registry of the jobs.
    assert not (run_dir / 'orchestrator.json').exists() and not (run_dir / 'jobs.json').exists()


def test_a_step_already_done_is_not_analysed(run_dir):
    step = JobStep(done=True)
    assert run_job(run_dir, step)['result'] == 'done'
    assert step.calls == ['done']


def test_a_group_replaced_by_a_later_one(run_dir, tmp_path):
    later = f'{SYSTEM}.plotpes.2'
    write_analysis(run_dir, Analysis(name=later, system=SYSTEM, step='plotpes',
                                     scheduler='slurm'), tmp_path / 'analysis.sh')
    step = JobStep()
    outcome = run_job(run_dir, step)
    assert (outcome['result'], outcome['error']) == \
        ('replaced', f'replaced by the group {later}: nothing analysed')
    assert step.calls == []
    # The later group runs, unaware of the earlier one.
    assert run_job(run_dir, step, later)['result'] == 'analysed'


def test_a_group_withdrawn(run_dir):
    (layout.job_dir(run_dir, GROUP) / WITHDRAWN_FILE).write_text('{}')
    step = JobStep()
    assert run_job(run_dir, step)['result'] == 'withdrawn' and step.calls == []


def test_prerequisites_missing(run_dir):
    (layout.system_dir(run_dir, SYSTEM) / 'system.json').unlink()
    step = JobStep()
    outcome = run_job(run_dir, step)
    assert outcome['result'] == 'blocked'
    assert outcome['error'].startswith('entries missing or not current: ')
    assert step.calls == ['done']


def test_an_error_of_the_analysis(run_dir, caplog):
    step = JobStep(error=MemoryError('Unable to allocate 88.3 GiB'))
    outcome = run_job(run_dir, step)
    assert (outcome['result'], outcome['error']) == \
        ('failed', 'MemoryError: Unable to allocate 88.3 GiB')
    assert outcome_file(run_dir) == outcome
    assert 'Traceback' in caplog.text  # in the log of the job
    assert not read_system(layout.system_dir(run_dir, SYSTEM)).get(['plotpes'])


def test_only_a_step_analysed_in_a_job(run_dir):
    step = JobStep()
    step.manual_analysis = False
    outcome = run_job(run_dir, step)
    assert (outcome['result'], outcome['error']) == \
        ('failed', 'plotpes is not a step analysed in a job of its own')
    assert step.calls == []


@pytest.mark.parametrize('step, code', [(JobStep(), 0), (JobStep(done=True), 0),
                                        (JobStep(error=RuntimeError('x')), 1)],
                         ids=['analysed', 'done', 'failed'])
def test_exit_code(run_dir, monkeypatch, step, code):
    monkeypatch.setattr(analysis_job, '_default_step', lambda name: step)
    assert analysis_job.main([str(analysis_path(run_dir))]) == code


def test_exit_code_of_a_group_not_read(run_dir, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(analysis_job, '_default_step', lambda name: pytest.fail('not built'))
    assert analysis_job.main([str(tmp_path / 'missing.json')]) == 2
    path = analysis_path(run_dir)
    path.write_text('{')
    assert analysis_job.main([str(path)]) == 2
    assert 'Group not read: ' in caplog.text
    assert not (path.parent / OUTCOME_FILE).exists()
