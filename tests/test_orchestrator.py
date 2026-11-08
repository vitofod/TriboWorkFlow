# Author: Vito Foderà
# Created: 2026-10-29

"""Tests of the orchestrator of the steps (``triboworkflow.execution.orchestrator``).

No scheduler is called and pw.x never runs: the run is the one of
``run_helpers``. The steps are fictitious: each one prepares a few
calculations and writes one entry of ``system.json`` with its provenance, so
that ``check_prerequisites`` and ``is_done`` work as for the real steps. The
workers are simulated by writing the claims and the attempts they would
leave; the jobs are set to their states by the test.
"""

import dataclasses
import datetime
import json
import re

import pytest

from run_helpers import (FP, JOB_DONE, MAX_SECONDS, NOT_CONVERGED, PARALLEL, PWI, SCF_DONE,
                         SCF_TRUNCATED, SYSTEM, Run, header, text)
from triboworkflow.core import store
from triboworkflow.core.settings.schema import ExecutionParams
from triboworkflow.core.store import Provenance, SystemWriter, is_done, read_system
from triboworkflow.core.store.record import RESTART_PARTS
from triboworkflow.execution import archive as archive_module
from triboworkflow.execution import layout
from triboworkflow.execution import orchestrator as orchestrator_module
from triboworkflow.execution.archive import released_claim
from triboworkflow.execution.jobs import (ANALYSIS_FILE, JOBSCRIPT_FILE, MANIFEST_FILE,
                                          OUTCOME_FILE, SUBMITTED_FILE, WITHDRAWN_FILE, Launch,
                                          Manifest, is_analysis, read_analysis, read_manifest,
                                          write_job)
from triboworkflow.execution.manager import JobManager
from triboworkflow.execution.orchestrator import (INTERFACE_GRAPH, Orchestrator,
                                                  OrchestratorError, Step, order_steps)
from triboworkflow.execution.policy import PolicyError

EXECUTION = ExecutionParams(scheduler='slurm', jobscript='job.sh', run_command='pw.x', groups=2,
                            walltime=3600, polling_interval=300, max_seconds_margin=60)
ALL = ('match', 'pes', 'plotpes', 'shear', 'adhesion', 'ppes', 'charge')
# Calculations of each fictitious step; match, plotpes and shear only analyse.
CALCS = {'pes': 2, 'adhesion': 2, 'ppes': 1, 'charge': 1}
# Fingerprint of the calculations after a change of the settings.
OTHER = 'sha256:' + 'b' * 64


class FakeStep(Step):
    """A step that prepares ``calcs`` calculations and writes the entry ``[name]``.

    Its provenance has a version: changing it makes the entry stale.
    ``versions`` holds the version each step expects of the others.
    ``fingerprint`` is the one its settings give to each calculation: a new
    value stands for settings changed. Its ``prepare`` keeps the calculations
    already there, whatever their fingerprint.
    """

    def __init__(self, run, name, versions):
        self.run, self.name, self.versions = run, name, versions
        self.fingerprint = FP
        self.input_text = PWI
        self.calls = []

    def plan(self, run_dir, system):
        return {f'{self.name}/c{i}': self.fingerprint for i in range(CALCS.get(self.name, 0))}

    def provenance(self, name):
        return Provenance.build(name, {'version': self.versions.get(name, 1)})

    def done(self, run_dir, system):
        sdir = layout.system_dir(run_dir, system)
        return is_done(read_system(sdir), [self.name], self.provenance(self.name), sdir)

    def expected(self, run_dir, system):
        return {(dep,): self.provenance(dep) for dep in INTERFACE_GRAPH[self.name]}

    def prepare(self, run_dir, system):
        self.calls.append('prepare')
        calc_ids = []
        for i in range(CALCS.get(self.name, 0)):
            calc_id = f'{self.name}/c{i}'
            if not self.run.path(calc_id).exists():
                self.run.calc(f'c{i}', input_text=self.input_text, step=self.name,
                              fingerprint=self.fingerprint)
            calc_ids.append(calc_id)
        return calc_ids

    def analyse(self, run_dir, system, calculations):
        self.calls.append('analyse')
        write_entry(run_dir, system, self.name, self.versions.get(self.name, 1))


def write_entry(run_dir, system, name, version=1):
    with SystemWriter(layout.system_dir(run_dir, system)) as writer:
        if not writer.document.exists:
            writer.document.create({})
        writer.document.set([name], {'done': True})
        writer.document.set_provenance([name], Provenance.build(name, {'version': version}))
        writer.save()


@pytest.fixture
def run(tmp_path):
    return Run(tmp_path)


def orchestrator(run, requested=ALL, versions=None, manager=None, steps=None):
    versions = {} if versions is None else versions
    steps = steps or {name: FakeStep(run, name, versions) for name in requested}
    return Orchestrator(run.dir, [SYSTEM], steps, manager or run.manager, EXECUTION,
                        run.jobscript)


OUTCOMES = {
    'completed': dict(status=store.Status('ended', 'converged', 'n/a', 'ok'),
                      stop_reason='completed', clean_stop=True),
    'crashed': dict(status=store.Status('crashed', None, 'n/a'), stop_reason='truncated',
                    exit_code=137, clean_stop=False),
    'user_request': dict(status=store.Status('ended', None, 'n/a'), stop_reason='user_request',
                         clean_stop=True),
    'not_converged': dict(status=store.Status('ended', 'not_converged', 'n/a'),
                          stop_reason='scf_not_converged', clean_stop=True),
    'max_seconds': dict(status=store.Status('ended', None, 'n/a'), stop_reason='max_seconds',
                        clean_stop=True),
    'not_parsed': dict(status=store.Status('ended', 'converged', 'n/a'), stop_reason='completed',
                       clean_stop=True),
    'parse_failed': dict(status=store.Status('ended', 'converged', 'n/a', 'failed'),
                         parse_error='no "!    total energy" in the last SCF cycle',
                         stop_reason='completed', clean_stop=True),
}


OUTPUTS = {'completed': SCF_DONE, 'crashed': SCF_TRUNCATED, 'user_request': SCF_DONE,
           'not_converged': text(header(mixing_beta=0.4), NOT_CONVERGED, JOB_DONE),
           'max_seconds': text(header(parallel=PARALLEL), MAX_SECONDS, JOB_DONE),
           'not_parsed': SCF_DONE, 'parse_failed': SCF_DONE}


def work(run, outcome='completed', only=None):
    """Workers take the next attempt of each calculation of the groups and end it.

    Every job is then set COMPLETED. ``only`` limits the calculations taken.
    The jobs of the analyses (``analysis.json``) are left to the tests.
    """
    for name in list(run.manager.groups):
        if is_analysis(layout.job_dir(run.dir, name)):
            continue
        manifest, _ = read_manifest(layout.job_dir(run.dir, name) / MANIFEST_FILE)
        jobs = run.manager.group_jobs(name)
        for launch in manifest.calculations:
            if only is not None and launch.calc_id not in only:
                continue
            path = run.path(launch.calc_id)
            if (len(run.record(launch.calc_id).attempts) != launch.n - 1
                    or (path / layout.CLAIM_FILE).exists()):
                continue
            run.start(launch.calc_id, jobs[0], launch=launch, output=OUTPUTS[outcome])
            run.finish(launch.calc_id, **OUTCOMES[outcome])
    run.slurm.states.update({job_id: 'COMPLETED' for job_id, job in run.manager.jobs.items()
                             if not is_analysis(layout.job_dir(run.dir, job['group']))})


def statuses(report):
    return {step.step: step.status for step in report.steps}


def prepared(orch):
    return {name: step.calls.count('prepare') for name, step in orch.steps.items()}


# --- graph ---------------------------------------------------------------------------------

def test_order_of_the_steps():
    assert order_steps(list(ALL), INTERFACE_GRAPH) == list(ALL)
    # The given order decides only between steps that are both free.
    assert order_steps(['charge', 'ppes', 'adhesion', 'pes'], INTERFACE_GRAPH) == \
        ['pes', 'adhesion', 'charge', 'ppes']
    assert order_steps(['shear', 'charge'], INTERFACE_GRAPH) == ['shear', 'charge']


def test_relaxed_and_static_pes_are_independent():
    # Both need only match: the given order decides between them.
    assert order_steps(['pes_scf', 'pes', 'match'], INTERFACE_GRAPH) == ['match', 'pes_scf', 'pes']
    assert order_steps(['pes', 'pes_scf'], INTERFACE_GRAPH) == ['pes', 'pes_scf']
    assert INTERFACE_GRAPH['pes'] == INTERFACE_GRAPH['pes_scf'] == ('match',)


def test_each_pes_has_its_interpolation():
    # plotpes interpolates pes, plotpes_scf interpolates pes_scf; both read
    # the high-symmetry points of match. shear stays on the relaxed PES.
    assert INTERFACE_GRAPH['plotpes'] == ('pes', 'match')
    assert INTERFACE_GRAPH['plotpes_scf'] == ('pes_scf', 'match')
    assert INTERFACE_GRAPH['shear'] == ('plotpes', 'pes', 'match')
    # Each interpolation right after its own PES, the given order deciding.
    assert order_steps(['plotpes_scf', 'plotpes', 'pes', 'pes_scf', 'match'], INTERFACE_GRAPH) == \
        ['match', 'pes', 'plotpes', 'pes_scf', 'plotpes_scf']


@pytest.mark.parametrize('requested, graph, message', [
    (['pes', 'band'], INTERFACE_GRAPH, 'steps not in the graph: band'),
    (['pes', 'pes'], INTERFACE_GRAPH, 'requested more than once'),
    (['a', 'b'], {'a': ('b',), 'b': ('a',)}, 'the graph has a cycle among a, b'),
])
def test_order_errors(requested, graph, message):
    with pytest.raises(OrchestratorError, match=message):
        order_steps(requested, graph)


# --- a whole run ---------------------------------------------------------------------------

def test_whole_workflow(run):
    orch = orchestrator(run)
    report = orch.run_pass()
    # match analyses only and completes; pes runs; the others wait.
    assert statuses(report) == {'match': 'completed', 'pes': 'running', 'plotpes': 'waiting',
                                'shear': 'waiting', 'adhesion': 'waiting', 'ppes': 'waiting',
                                'charge': 'waiting'}
    assert report.submitted == [f'{SYSTEM}.pes.1'] and run.slurm.submitted == 2
    for _ in range(6):
        work(run)
        report = orch.run_pass()
    assert report.finished and report.success
    assert statuses(report) == dict.fromkeys(ALL, 'completed')
    assert prepared(orch) == dict.fromkeys(ALL, 1)
    document = read_system(run.sdir)
    assert all(document.get([name]) == {'done': True} for name in ALL)


def test_run_until_no_step_waits(run):
    orch = orchestrator(run, ['match', 'pes'])
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        work(run)

    report = orch.run(sleep=sleep)
    assert report.success and sleeps == [EXECUTION.polling_interval]


@pytest.mark.parametrize('stop, max_passes, passes', [(lambda: True, None, 1), (None, 3, 3)])
def test_run_stops_when_asked(run, stop, max_passes, passes):
    orch = orchestrator(run, ['match', 'pes'])
    sleeps = []
    report = orch.run(sleep=sleeps.append, max_passes=max_passes, stop=stop)
    assert not report.finished and len(sleeps) == passes - 1
    state = json.loads((run.dir / 'orchestrator.json').read_text())
    assert state['systems'][SYSTEM]['pes']['status'] == 'running'


def test_steps_of_a_system_run_one_at_a_time_in_the_given_order(run):
    write_entry(run.dir, SYSTEM, 'pes')
    orch = orchestrator(run, ['charge', 'ppes', 'adhesion'])
    assert statuses(orch.run_pass()) == {'adhesion': 'running', 'charge': 'waiting',
                                         'ppes': 'waiting'}
    work(run)
    report = orch.run_pass()
    assert statuses(report) == {'adhesion': 'completed', 'charge': 'running', 'ppes': 'waiting'}
    assert orch.entry(SYSTEM, 'ppes')['reason'] == 'ready; charge runs first'
    work(run)
    assert statuses(orch.run_pass())['ppes'] == 'running'


def test_report_of_the_steps(run):
    orch = orchestrator(run, ['match', 'pes'])
    report = orch.run_pass()
    (pes,) = [step for step in report.steps if step.step == 'pes']
    assert (pes.status, pes.calcs, pes.attention) == ('running', {'launched': 2}, [])
    assert report.by_status('completed') == [(SYSTEM, 'match')]
    assert not report.finished and not report.success
    assert orch.summary(report) == ('steps 1 running, 1 completed; jobs 2 waiting, 0 held, '
                                    '0 to check; groups written 1')
    state = json.loads((run.dir / 'orchestrator.json').read_text())
    assert state['systems'][SYSTEM]['pes']['calculations'] == ['pes/c0', 'pes/c1']
    assert state['systems'][SYSTEM]['pes']['status'] == 'running'


# --- failures and blocks -------------------------------------------------------------------

def run_to(orch, run, step):
    """Passes until ``step`` runs."""
    for _ in range(8):
        report = orch.run_pass()
        if orch.entry(SYSTEM, step)['status'] == 'running':
            return report
        work(run)
    raise AssertionError(f'{step} never runs')


@pytest.mark.parametrize('failing, other', [('ppes', 'charge'), ('charge', 'ppes')])
def test_a_failed_branch_does_not_block_the_other(run, failing, other):
    orch = orchestrator(run, ['match', 'pes', 'adhesion', 'ppes', 'charge'])
    run_to(orch, run, 'ppes')
    work(run, 'user_request' if failing == 'ppes' else 'completed')
    orch.run_pass()
    work(run, 'user_request' if failing == 'charge' else 'completed')
    for _ in range(3):
        report = orch.run_pass()
        work(run)
    assert statuses(report)[failing] == 'failed' and statuses(report)[other] == 'completed'
    assert report.finished and not report.success
    (step,) = [s for s in report.steps if s.step == failing]
    assert step.reason == '1 calculations need the user'
    assert step.attention == [{'calc_id': f'{failing}/c0', 'state': 'manual',
                               'reason': 'stop reason user_request'}]


def test_descendants_of_a_failed_step_are_blocked(run):
    orch = orchestrator(run)
    orch.run_pass()
    work(run, 'user_request')
    report = orch.run_pass()
    assert statuses(report) == {'match': 'completed', 'pes': 'failed', 'plotpes': 'blocked',
                                'shear': 'blocked', 'adhesion': 'blocked', 'ppes': 'blocked',
                                'charge': 'blocked'}
    assert orch.entry(SYSTEM, 'adhesion')['reason'] == 'depends on pes failed'
    assert orch.entry(SYSTEM, 'ppes')['reason'] == 'depends on adhesion blocked'
    assert orch.entry(SYSTEM, 'shear')['reason'] == 'depends on plotpes blocked, pes failed'
    assert report.finished and prepared(orch)['adhesion'] == 0


def test_valid_prerequisites_are_reused(run):
    # adhesion is not requested: its entry, written before, is current.
    write_entry(run.dir, SYSTEM, 'adhesion')
    orch = orchestrator(run, ['ppes'])
    assert statuses(orch.run_pass()) == {'ppes': 'running'}


@pytest.mark.parametrize('written, missing', [(False, 'missing'), (True, 'stale')])
def test_missing_prerequisite_blocks_without_computing_it_again(run, written, missing):
    if written:
        write_entry(run.dir, SYSTEM, 'adhesion', version=1)
    orch = orchestrator(run, ['ppes'], versions={'adhesion': 2})
    report = orch.run_pass()
    assert statuses(report) == {'ppes': 'blocked'}
    assert orch.entry(SYSTEM, 'ppes')['reason'] == ('entries missing or not current: adhesion; '
                                                    'run the steps that compute them first')
    assert prepared(orch) == {'ppes': 0} and run.slurm.submitted == 0
    assert report.finished and not report.success


def test_completed_entries_are_not_computed_again(run):
    for name in ('match', 'pes'):
        write_entry(run.dir, SYSTEM, name)
    orch = orchestrator(run, ['match', 'pes', 'plotpes'])
    report = orch.run_pass()
    assert statuses(report) == dict.fromkeys(['match', 'pes', 'plotpes'], 'completed')
    assert prepared(orch) == {'match': 0, 'pes': 0, 'plotpes': 1}
    assert orch.entry(SYSTEM, 'pes')['reason'] == 'entries already done and current'


@pytest.mark.parametrize('method', ['prepare', 'analyse'])
def test_error_of_a_step_until_retry(run, method, monkeypatch):
    orch = orchestrator(run, ['match', 'pes'])
    step = orch.steps['pes']
    original = getattr(step, method)

    def broken(*args):
        raise RuntimeError('disk full')

    monkeypatch.setattr(step, method, broken)
    orch.run_pass()
    work(run)
    report = orch.run_pass()
    assert statuses(report)['pes'] == 'failed'
    assert orch.entry(SYSTEM, 'pes')['error'] == 'RuntimeError: disk full'
    monkeypatch.setattr(step, method, original)
    assert statuses(orch.run_pass())['pes'] == 'failed'  # not retried by itself
    orch.retry(SYSTEM, 'pes')
    for _ in range(3):
        report = orch.run_pass()
        work(run)
    assert statuses(report)['pes'] == 'completed'


# --- steps analysed by hand ------------------------------------------------------------------

class ManualStep(FakeStep):
    """A step without calculations whose analysis runs only by hand."""

    manual_analysis = True


BY_HAND = ('match', 'pes', 'plotpes', 'shear')
MANUAL = orchestrator_module.MANUAL_REASON.format(step='plotpes')


def manual_run(run, requested=BY_HAND, versions=None, manager=None):
    """Orchestrator whose plotpes is analysed only by hand."""
    versions = {} if versions is None else versions
    steps = {name: (ManualStep if name == 'plotpes' else FakeStep)(run, name, versions)
             for name in requested}
    return orchestrator(run, steps=steps, manager=manager)


def manual_restarted(run, requested=BY_HAND):
    """A new orchestrator and job manager, as each command of the CLI builds them."""
    run.manager = JobManager(run.dir, run.slurm, 3600, now=run.clock)
    return manual_run(run, requested, manager=run.manager)


def until_pes(orch, run):
    """Passes until pes is completed; the report of the last one."""
    for _ in range(2):
        report = orch.run_pass()
        work(run)
    return report


def test_a_step_analysed_by_hand_is_left_to_the_user(run, caplog):
    orch = manual_run(run)
    for _ in range(3):
        report = orch.run_pass()
        work(run)
    # Never prepared nor analysed in a pass; its descendants wait for it.
    assert statuses(report) == {'match': 'completed', 'pes': 'completed', 'plotpes': 'manual',
                                'shear': 'blocked'}
    assert orch.steps['plotpes'].calls == [] and prepared(orch)['shear'] == 0
    assert orch.entry(SYSTEM, 'plotpes')['reason'] == MANUAL
    assert MANUAL.startswith('to analyse by hand: run "triboworkflow analyse plotpes"')
    assert orch.entry(SYSTEM, 'shear')['reason'] == 'depends on plotpes manual'
    assert report.finished and not report.success
    assert orch.summary(report).startswith('steps 2 completed, 1 blocked, 1 manual;')
    ready = [r for r in caplog.records if r.getMessage() == f'{SYSTEM}/plotpes ready, {MANUAL}']
    assert len(ready) == 1  # logged once, not at every pass


def test_the_orchestrator_ends_with_a_step_left_to_the_user(run):
    orch = manual_run(run)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        work(run)

    report = orch.run(sleep=sleep)
    assert report.finished and statuses(report)['plotpes'] == 'manual'
    assert sleeps == [EXECUTION.polling_interval]


def test_after_the_analysis_by_hand_the_passes_go_on(run):
    until_pes(manual_run(run), run)
    by_hand = manual_restarted(run)
    assert by_hand.analyse_step(SYSTEM, 'plotpes').status == 'completed'
    assert by_hand.steps['plotpes'].calls == ['prepare', 'analyse']
    orch = manual_restarted(run)
    report = orch.run_pass()
    assert statuses(report) == dict.fromkeys(BY_HAND, 'completed') and report.success
    # Recognised from its entry: neither it nor pes is analysed again.
    assert orch.steps['plotpes'].calls == [] and orch.steps['pes'].calls == []


class Stopped(BaseException):
    """The process stopped during the analysis (a signal, the walltime)."""


def stopped_analysis(run, monkeypatch):
    """match and pes completed, then an analysis of plotpes by hand stopped before its end."""
    until_pes(manual_run(run), run)
    by_hand = manual_restarted(run)

    def stop(*args):
        raise Stopped

    monkeypatch.setattr(by_hand.steps['plotpes'], 'analyse', stop)
    with pytest.raises(Stopped):
        by_hand.analyse_step(SYSTEM, 'plotpes')
    state = json.loads((run.dir / 'orchestrator.json').read_text())['systems'][SYSTEM]
    assert (state['plotpes']['status'], state['plotpes']['calculations']) == ('running', [])
    assert not read_system(run.sdir).get(['plotpes'])


def test_an_analysis_by_hand_stopped_is_left_to_the_user_again(run, monkeypatch, caplog):
    stopped_analysis(run, monkeypatch)
    # adhesion, ready, is not held back by the analysis stopped.
    orch = manual_restarted(run, ('match', 'pes', 'adhesion', 'plotpes'))
    report = orch.run_pass()
    assert statuses(report) == {'match': 'completed', 'pes': 'completed', 'adhesion': 'running',
                                'plotpes': 'manual'}
    assert orch.steps['plotpes'].calls == [] and report.finished is False
    assert (f'{SYSTEM}/plotpes: its analysis by hand stopped before its end; it starts again '
            'from the beginning') in caplog.text
    by_hand = manual_restarted(run, ('match', 'pes', 'adhesion', 'plotpes'))
    assert by_hand.analyse_step(SYSTEM, 'plotpes').status == 'completed'
    assert by_hand.steps['plotpes'].calls == ['prepare', 'analyse']


def test_an_analysis_by_hand_stopped_is_completed_by_the_next_one(run, monkeypatch):
    stopped_analysis(run, monkeypatch)
    by_hand = manual_restarted(run)
    assert by_hand.analyse_step(SYSTEM, 'plotpes').status == 'completed'
    assert by_hand.steps['plotpes'].calls == ['analyse']
    assert read_system(run.sdir).get(['plotpes']) == {'done': True}


def test_prepare_and_submit_of_a_step_analysed_by_hand(run):
    for name in ('match', 'pes'):
        write_entry(run.dir, SYSTEM, name)
    orch = manual_run(run, ['plotpes'])
    assert orch.prepare_step(SYSTEM, 'plotpes').status == 'manual'
    assert orch.steps['plotpes'].calls == []
    with pytest.raises(OrchestratorError, match='no calculations to submit: it is analysed by '
                                                'hand, with "analyse plotpes"'):
        orch.submit_step(SYSTEM, 'plotpes')
    assert orch.entry(SYSTEM, 'plotpes')['status'] == 'manual'


def test_a_step_analysed_by_hand_without_its_prerequisites_is_blocked(run):
    orch = manual_run(run, ['plotpes'])
    report = orch.run_pass()
    assert statuses(report) == {'plotpes': 'blocked'} and orch.steps['plotpes'].calls == []
    assert orch.entry(SYSTEM, 'plotpes')['reason'].startswith('entries missing or not current: ')


def test_a_step_analysed_by_hand_done_or_no_longer_current(run):
    for name in ('match', 'pes', 'plotpes'):
        write_entry(run.dir, SYSTEM, name)
    orch = manual_run(run, ['plotpes'])
    report = orch.run_pass()
    assert statuses(report) == {'plotpes': 'completed'} and report.success
    assert orch.entry(SYSTEM, 'plotpes')['reason'] == 'entries already done and current'
    write_entry(run.dir, SYSTEM, 'plotpes', version=2)  # written with other settings
    report = orch.run_pass()
    assert statuses(report) == {'plotpes': 'manual'} and orch.steps['plotpes'].calls == []


# --- steps analysed in a job of their own --------------------------------------------------

GROUP = f'{SYSTEM}.plotpes.1'
NO_OUTCOME = orchestrator_module.NO_OUTCOME


def job_run(run, requested=BY_HAND, versions=None, manager=None):
    """Orchestrator whose plotpes is analysed in a job of its own ([execution.analysis])."""
    versions = {} if versions is None else versions
    steps = {name: (ManualStep if name == 'plotpes' else FakeStep)(run, name, versions)
             for name in requested}
    jobscript = run.dir.parent / 'analysis.sh'
    if not jobscript.exists():
        jobscript.write_text('#!/bin/bash\n#SBATCH --mem=200G\n')
    return Orchestrator(run.dir, [SYSTEM], steps, manager or run.manager, EXECUTION,
                        run.jobscript, analysis_jobscript=jobscript)


def job_restarted(run, requested=BY_HAND):
    """A new orchestrator and job manager, as after a restart or for a command of the CLI."""
    run.manager = JobManager(run.dir, run.slurm, 3600, now=run.clock)
    return job_run(run, requested, manager=run.manager)


def job_of(run, group=GROUP):
    (job_id,) = run.manager.group_jobs(group)
    return job_id


def the_job_ends(run, state='COMPLETED', analysed=True, group=GROUP, outcome=None):
    """What the job of the analysis leaves: the entry of plotpes if ``analysed``, its outcome.

    The job runs on a compute node, in a process of its own; the scheduler then
    reports it in ``state``.
    """
    if analysed:
        write_entry(run.dir, SYSTEM, 'plotpes')
    if outcome is not None:
        (layout.job_dir(run.dir, group) / OUTCOME_FILE).write_text(json.dumps(outcome))
    run.slurm.states[job_of(run, group)] = state


def test_a_step_analysed_in_a_job_of_its_own(run):
    orch = job_run(run)
    report = until_pes(orch, run)
    entry = orch.entry(SYSTEM, 'plotpes')
    assert (entry['status'], entry['job'], entry['calculations']) == ('running', GROUP, [])
    assert statuses(report) == {'match': 'completed', 'pes': 'completed', 'plotpes': 'running',
                                'shear': 'waiting'}
    assert report.submitted == [GROUP] and orch.step_report(SYSTEM, 'plotpes').job == GROUP
    path = layout.job_dir(run.dir, GROUP)
    analysis, _ = read_analysis(path / ANALYSIS_FILE)
    assert (analysis.system, analysis.step, analysis.scheduler) == (SYSTEM, 'plotpes', 'slurm')
    assert not (path / MANIFEST_FILE).exists()
    jobscript = (path / JOBSCRIPT_FILE).read_text()
    assert jobscript.startswith('#!/bin/bash\n#SBATCH --mem=200G\n')
    assert ' -m triboworkflow.cli.analysis_job ' in jobscript
    job_id = job_of(run)
    report = orch.run_pass()  # the job is in the queue
    assert orch.entry(SYSTEM, 'plotpes')['reason'] == f'job {job_id} PENDING'
    assert not report.finished
    submitted = run.slurm.submitted
    the_job_ends(run)
    report = orch.run_pass()
    assert statuses(report) == dict.fromkeys(BY_HAND, 'completed') and report.success
    assert orch.entry(SYSTEM, 'plotpes')['reason'] == f'analysed in the job of group {GROUP}'
    # Never prepared nor analysed by the orchestrator; one job, never submitted again.
    assert orch.steps['plotpes'].calls == [] and run.slurm.submitted == submitted
    state = json.loads((run.dir / 'orchestrator.json').read_text())
    assert state['schema_version'] == 2 and state['systems'][SYSTEM]['plotpes']['job'] == GROUP


def test_the_orchestrator_waits_for_the_job_of_the_analysis(run):
    orch = job_run(run)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        work(run)
        if len(sleeps) == 3:
            the_job_ends(run)

    report = orch.run(sleep=sleep)
    assert report.success and len(sleeps) == 3 and orch.steps['plotpes'].calls == []


@pytest.mark.parametrize('state', ['COMPLETED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL',
                                   'CANCELLED'])
def test_a_job_over_without_the_step_done_fails_it(run, state):
    orch = job_run(run)
    until_pes(orch, run)
    job_id = job_of(run)
    the_job_ends(run, state, analysed=False)
    report = orch.run_pass()
    entry = orch.entry(SYSTEM, 'plotpes')
    assert statuses(report)['plotpes'] == 'failed' and statuses(report)['shear'] == 'blocked'
    assert entry['reason'].startswith(f'job {job_id} {state} without completing the step: '
                                      f'{NO_OUTCOME}; output of the job in jobs/{GROUP}; ')
    assert entry['reason'].endswith('then "triboworkflow retry plotpes"')
    assert entry['error'] == f'AnalysisJobFailed: job {job_id} {state}: {NO_OUTCOME}'
    assert report.finished and not report.success
    submitted = run.slurm.submitted
    for _ in range(2):
        orch.run_pass()
    # Never submitted again automatically: the same job would fail in the same way.
    assert run.slurm.submitted == submitted and statuses(orch.report())['plotpes'] == 'failed'
    assert orch.steps['plotpes'].calls == []


def test_after_a_failed_job_retry_submits_a_new_one(run):
    orch = job_run(run)
    until_pes(orch, run)
    first = job_of(run)
    the_job_ends(run, 'FAILED', analysed=False,
                 outcome={'result': 'failed', 'error': 'MemoryError: Unable to allocate 88.3 GiB'})
    orch.run_pass()
    assert ('without completing the step: failed: MemoryError: Unable to allocate 88.3 GiB; '
            in orch.entry(SYSTEM, 'plotpes')['reason'])
    orch.retry(SYSTEM, 'plotpes')
    assert orch.entry(SYSTEM, 'plotpes')['job'] is None
    report = orch.run_pass()
    second = f'{SYSTEM}.plotpes.2'
    assert report.submitted == [second] and orch.entry(SYSTEM, 'plotpes')['job'] == second
    assert job_of(run, second) != first
    the_job_ends(run, group=second)
    assert orch.run_pass().success


def test_a_held_job_keeps_the_step_running(run, monkeypatch):
    orch = job_run(run)
    until_pes(orch, run)
    job_id = job_of(run)
    monkeypatch.setattr(run.slurm, 'is_held', lambda state: state == 'PENDING')
    report = orch.run_pass()
    assert statuses(report)['plotpes'] == 'running' and not report.finished
    assert orch.entry(SYSTEM, 'plotpes')['reason'].startswith(f'job {job_id} PENDING: held')


def test_a_job_no_longer_reported_keeps_the_step_running(run):
    orch = job_run(run)
    until_pes(orch, run)
    job_id = job_of(run)
    orch.run_pass()
    del run.slurm.states[job_id]
    report = orch.run_pass()
    assert statuses(report)['plotpes'] == 'running' and not report.finished
    assert orch.entry(SYSTEM, 'plotpes')['reason'].startswith(
        f'job {job_id}: not reported by the scheduler, or in a state not known (last known state '
        'PENDING)')


def test_a_preempted_job_is_over_once_not_requeued(run):
    orch = job_run(run)
    until_pes(orch, run)
    job_id = job_of(run)
    run.slurm.states[job_id] = 'PREEMPTED'
    orch.run_pass()
    assert orch.entry(SYSTEM, 'plotpes')['reason'] == \
        f'job {job_id} PREEMPTED: the scheduler may requeue it; to be confirmed by a later query'
    run.clock.time += datetime.timedelta(seconds=EXECUTION.polling_interval)
    report = orch.run_pass()
    assert statuses(report)['plotpes'] == 'failed'
    assert orch.entry(SYSTEM, 'plotpes')['reason'].startswith(
        f'job {job_id} PREEMPTED, not requeued without completing the step: ')


def stop(*args, **kwargs):
    raise Stopped


@pytest.mark.parametrize('where', ['add_group', '_save'])
def test_a_group_written_before_an_interruption_is_taken_again(run, monkeypatch, where):
    orch = job_run(run)
    orch.run_pass()
    work(run)
    # Interrupted after writing the group, before registering it or saving the state.
    monkeypatch.setattr(orch.manager if where == 'add_group' else orch, where, stop)
    with pytest.raises(Stopped):
        orch.run_pass()
    assert is_analysis(layout.job_dir(run.dir, GROUP)) and run.slurm.submitted == 2
    restarted = job_restarted(run)
    report = restarted.run_pass()
    assert report.submitted == [GROUP] and restarted.entry(SYSTEM, 'plotpes')['job'] == GROUP
    assert len(restarted.manager.group_jobs(GROUP)) == 1 and run.slurm.submitted == 3
    assert not layout.job_dir(run.dir, f'{SYSTEM}.plotpes.2').exists()


def test_a_restart_follows_the_same_job(run):
    until_pes(job_run(run), run)
    job_id, submitted = job_of(run), run.slurm.submitted
    restarted = job_restarted(run)
    restarted.run_pass()
    assert restarted.entry(SYSTEM, 'plotpes')['reason'] == f'job {job_id} PENDING'
    assert run.slurm.submitted == submitted
    the_job_ends(run)
    assert job_restarted(run).run_pass().success and run.slurm.submitted == submitted


def test_a_job_is_followed_also_without_the_table(run):
    until_pes(job_run(run), run)
    orch = manual_restarted(run)  # [execution.analysis] removed while the job runs
    orch.run_pass()
    assert orch.entry(SYSTEM, 'plotpes')['status'] == 'running'
    assert orch.entry(SYSTEM, 'plotpes')['job'] == GROUP
    the_job_ends(run)
    assert orch.run_pass().success


def test_the_steps_of_a_system_wait_for_the_job(run):
    orch = job_run(run, ('match', 'pes', 'plotpes', 'adhesion'))
    report = until_pes(orch, run)
    assert statuses(report)['plotpes'] == 'running' and statuses(report)['adhesion'] == 'waiting'
    assert orch.entry(SYSTEM, 'adhesion')['reason'] == 'ready; plotpes runs first'
    the_job_ends(run)
    report = orch.run_pass()
    assert statuses(report)['plotpes'] == 'completed' and statuses(report)['adhesion'] == 'running'


def test_the_job_waits_for_the_steps_that_run(run):
    orch = job_run(run, ('match', 'pes', 'adhesion', 'plotpes'))
    report = until_pes(orch, run)
    assert statuses(report)['adhesion'] == 'running' and statuses(report)['plotpes'] == 'waiting'
    assert orch.entry(SYSTEM, 'plotpes')['reason'] == 'ready; adhesion runs first'
    assert not layout.job_dir(run.dir, GROUP).exists()
    work(run)
    report = orch.run_pass()
    assert statuses(report)['adhesion'] == 'completed'
    assert orch.entry(SYSTEM, 'plotpes')['job'] == GROUP and report.submitted == [GROUP]


def test_analysis_by_hand_refused_while_the_job_may_run(run):
    until_pes(job_run(run), run)
    job_id = job_of(run)
    before = (run.dir / 'orchestrator.json').read_bytes()
    by_hand = job_restarted(run)
    message = (f'{SYSTEM}/plotpes not analysed by hand: its analysis runs in a job of its own '
               f'(group {GROUP}, job {job_id} submitted, not reported yet)')
    with pytest.raises(OrchestratorError, match=re.escape(message)):
        by_hand.analyse_step(SYSTEM, 'plotpes')
    assert by_hand.steps['plotpes'].calls == []
    assert (run.dir / 'orchestrator.json').read_bytes() == before
    # Cancelled, and its end read from the scheduler (e.g. by "submit"): analysed by hand.
    run.slurm.states[job_id] = 'CANCELLED'
    by_hand.manager.cycle()
    assert by_hand.analyse_step(SYSTEM, 'plotpes').status == 'completed'
    assert by_hand.steps['plotpes'].calls == ['prepare', 'analyse']
    assert by_hand.entry(SYSTEM, 'plotpes')['job'] is None


def test_analysis_by_hand_of_a_job_never_submitted(run):
    orch = job_run(run)
    orch.run_pass()
    work(run)
    run.slurm.refuse = 1
    orch.run_pass()
    orch.run_pass()
    reason = orch.entry(SYSTEM, 'plotpes')['reason']
    assert reason.startswith('submission of the job refused: ')
    assert reason.endswith(f'then "triboworkflow clear-refusal {GROUP}"')
    submitted = run.slurm.submitted
    by_hand = job_restarted(run)
    assert by_hand.analyse_step(SYSTEM, 'plotpes').status == 'completed'
    assert by_hand.steps['plotpes'].calls == ['prepare', 'analyse']
    # Its group is withdrawn: never submitted later.
    assert (layout.job_dir(run.dir, GROUP) / WITHDRAWN_FILE).is_file()
    assert job_restarted(run).run_pass().success and run.slurm.submitted == submitted


def test_prepare_and_submit_of_a_step_analysed_in_a_job(run):
    for name in ('match', 'pes'):
        write_entry(run.dir, SYSTEM, name)
    orch = job_run(run, ['plotpes'])
    report = orch.prepare_step(SYSTEM, 'plotpes')
    assert report.status == 'waiting' and report.reason.startswith(
        'ready: nothing to prepare; its analysis runs in a job of its own, submitted by '
        '"submit plotpes" or by the orchestrator')
    assert not (run.dir / 'jobs').exists()
    report, written = orch.submit_step(SYSTEM, 'plotpes')
    assert written == [GROUP] and report.status == 'running' and report.job == GROUP
    assert run.slurm.submitted == 1
    report, written = orch.submit_step(SYSTEM, 'plotpes')  # followed, nothing submitted
    assert written == [] and report.reason == f'job {job_of(run)} PENDING'
    assert orch.prepare_step(SYSTEM, 'plotpes').status == 'running'  # its job left as it is
    the_job_ends(run)
    report, written = orch.submit_step(SYSTEM, 'plotpes')
    assert report.status == 'completed' and written == [] and run.slurm.submitted == 1
    assert orch.steps['plotpes'].calls == []


# --- relaunches ----------------------------------------------------------------------------

def test_crash_runs_again_in_a_new_group(run):
    orch = orchestrator(run, ['match', 'pes'])
    orch.run_pass()
    work(run, 'crashed')
    report = orch.run_pass()
    assert report.submitted == [f'{SYSTEM}.pes.2']
    manifest, _ = read_manifest(layout.job_dir(run.dir, f'{SYSTEM}.pes.2') / MANIFEST_FILE)
    assert [(l.calc_id, l.n, l.kind) for l in manifest.calculations] == \
        [('pes/c0', 2, 'unchanged'), ('pes/c1', 2, 'unchanged')]
    assert (run.path('pes/c0') / 'attempts' / '1' / 'output.pwo').exists()
    assert not (run.path('pes/c0') / layout.CLAIM_FILE).exists()
    work(run, 'crashed')
    orch.run_pass()
    work(run, 'crashed')
    report = orch.run_pass()
    # Two unchanged relaunches at level 1, then the user.
    assert statuses(report)['pes'] == 'failed'
    assert [len(run.record(f'pes/c{i}').attempts) for i in (0, 1)] == [3, 3]
    assert run.slurm.submitted == 6


def group_launches(run, group):
    manifest, _ = read_manifest(layout.job_dir(run.dir, group) / MANIFEST_FILE)
    return {launch.calc_id: launch for launch in manifest.calculations}


def test_remedy_runs_at_the_next_level(run):
    orch = orchestrator(run, ['match', 'pes'])
    orch.run_pass()
    work(run, 'not_converged')
    report = orch.run_pass()
    (pes,) = [s for s in report.steps if s.step == 'pes']
    assert (pes.status, pes.calcs, pes.attention) == ('running', {'launched': 2}, [])
    assert report.submitted == [f'{SYSTEM}.pes.2']
    launches = group_launches(run, f'{SYSTEM}.pes.2')
    assert launches['pes/c0'] == Launch(SYSTEM, 'pes/c0', 2, 'remedy', 2, {'mixing_beta': 0.2},
                                        {'mixing_beta': 0.4}, False)
    assert not (run.path('pes/c0') / layout.CLAIM_FILE).exists()
    work(run)
    assert statuses(orch.run_pass())['pes'] == 'completed'
    assert run.record('pes/c0').attempts[-1].remedies['level'] == 2


def test_relaunch_after_max_seconds_continues_from_the_output_directory(run):
    orch = orchestrator(run, ['match', 'pes'])
    orch.steps['pes'].input_text = PWI.replace('&CONTROL\n',
                                               "&CONTROL\n  prefix = 'p'\n  outdir = 'OUT'\n")
    orch.run_pass()
    work(run, 'max_seconds')
    save = run.path('pes/c0') / 'OUT' / 'p.save'  # only pes/c0 has the files of a restart
    save.mkdir(parents=True)
    for name in ('data-file-schema.xml', 'charge-density.dat'):
        (save / name).write_text('')
    report = orch.run_pass()
    launches = group_launches(run, report.submitted[0])
    assert [(launch.kind, launch.level, launch.restart) for launch in launches.values()] == \
        [('unchanged', 1, True), ('unchanged', 1, False)]


def test_restart_with_another_parallelization_needs_the_user(run):
    orch = orchestrator(run, ['match', 'pes'])
    orch.steps['pes'].input_text = PWI.replace('&CONTROL\n',
                                               "&CONTROL\n  prefix = 'p'\n  outdir = 'OUT'\n")
    orch.run_pass()
    work(run, 'max_seconds')
    save = run.path('pes/c0') / 'OUT' / 'p.save'
    save.mkdir(parents=True)
    for name in ('data-file-schema.xml', 'charge-density.dat'):
        (save / name).write_text('')
    orch.run_pass()
    work(run)
    record = run.record('pes/c0')
    read = dict.fromkeys(RESTART_PARTS, True) | {'same_parallelization': False}
    store.write_record(run.sdir, dataclasses.replace(
        record, attempts=record.attempts[:-1] + (dataclasses.replace(record.attempts[-1],
                                                                     restart=read),)))
    report = orch.run_pass()
    (pes,) = [s for s in report.steps if s.step == 'pes']
    assert (pes.status, pes.calcs) == ('failed', {'manual': 1, 'completed': 1})
    assert pes.attention == [{'calc_id': 'pes/c0', 'state': 'completed', 'reason': (
        'attempt 2 completed, its results cannot be used: attempt 2 continued attempt 1 '
        '(max_seconds) from files written per process with another parallelization: they may '
        'have been read incorrectly')}]
    assert report.submitted == []


@pytest.mark.parametrize('case, reason', [
    ('not_parsed', 'attempt 1 completed, its output not parsed'),
    ('parse_failed', 'attempt 1 completed, its output not parsed: no "!    total energy" in the '
                     'last SCF cycle'),
    ('other model', 'attempt 1 completed, its results cannot be used: output of another energy '
                    'model (ecutwfc: expected 40.0, observed 45.0)'),
    # A charge density that failed after pw.x: to the user, never relaunched.
    ('density failed', 'attempt 1 completed, its results cannot be used: charge density failed: '
                       'pp.x stopped on an error in routine postproc (1): reading inputpp '
                       'namelist'),
])
def test_completed_with_results_not_usable_needs_the_user(run, case, reason):
    orch = orchestrator(run, ['match', 'pes'])
    orch.run_pass()
    work(run, 'completed' if case in ('other model', 'density failed') else case, only=['pes/c0'])
    work(run, only=['pes/c1'])
    if case == 'other model':
        store.write_record(run.sdir, dataclasses.replace(run.record('pes/c0'),
                                                         observed={'ecutwfc': 45.0}))
    if case == 'density failed':
        record = run.record('pes/c0')
        failed = dataclasses.replace(record.status, density='failed')
        attempt = dataclasses.replace(record.attempts[-1], status=failed, density=store.DensityRun(
            'pp.x', error='pp.x stopped on an error in routine postproc (1): reading inputpp '
                          'namelist'))
        store.write_record(run.sdir, dataclasses.replace(record, density=True, status=failed,
                                                         attempts=(attempt,)))
    report = orch.run_pass()
    (pes,) = [s for s in report.steps if s.step == 'pes']
    assert (pes.status, pes.calcs) == ('failed', {'manual': 1, 'completed': 1})
    assert pes.attention == [{'calc_id': 'pes/c0', 'state': 'completed', 'reason': reason}]
    assert report.submitted == [] and 'analyse' not in orch.steps['pes'].calls
    assert len(run.record('pes/c0').attempts) == 1


def test_held_job_is_waited_for(run, monkeypatch):
    orch = orchestrator(run, ['match', 'pes'])
    orch.run_pass()
    (job_id, _) = run.manager.group_jobs(f'{SYSTEM}.pes.1')
    run.start('pes/c0', job_id)
    monkeypatch.setattr(run.slurm, 'is_held', lambda state: state == 'RUNNING')
    run.slurm.states[job_id] = 'RUNNING'
    report = orch.run_pass()
    (pes,) = [s for s in report.steps if s.step == 'pes']
    assert pes.status == 'running' and pes.calcs['held'] == 1
    assert report.submitted == []


def test_relaunch_by_hand(run):
    orch = orchestrator(run, ['match', 'pes'])
    orch.run_pass()
    work(run, 'user_request')
    assert statuses(orch.run_pass())['pes'] == 'failed'
    group = orch.relaunch(SYSTEM, 'pes/c0')
    assert group == f'{SYSTEM}.pes.2' and orch.entry(SYSTEM, 'pes')['status'] == 'running'
    manifest, _ = read_manifest(layout.job_dir(run.dir, group) / MANIFEST_FILE)
    assert [(l.calc_id, l.n, l.kind) for l in manifest.calculations] == [('pes/c0', 2, 'manual')]
    with pytest.raises(PolicyError, match='pending'):
        orch.relaunch(SYSTEM, 'pes/c0')
    with pytest.raises(OrchestratorError, match='not a calculation of a step'):
        orch.relaunch(SYSTEM, 'ppes/c0')


def test_empty_claim_released_then_relaunched_and_submitted(run):
    orch = orchestrator(run, ['match', 'pes'])
    orch.run_pass()
    work(run, only=['pes/c1'])
    claim = run.path('pes/c0') / layout.CLAIM_FILE
    claim.write_text('')  # the worker of pes/c0 stopped right after creating its claim
    (pes,) = [s for s in orch.run_pass().steps if s.step == 'pes']
    assert pes.status == 'running' and pes.attention[0]['state'] == 'uncertain'
    queries = run.slurm.queries
    assert orch.release_claim(SYSTEM, 'pes/c0') == 1
    assert run.slurm.queries == queries and not claim.exists()
    (attempt,) = run.record('pes/c0').attempts
    assert attempt.remedies == {'kind': 'initial', 'level': 1, 'values': {}, 'initial_values': {},
                                'restart': False}
    (pes,) = [s for s in orch.run_pass().steps if s.step == 'pes']
    assert pes.status == 'failed'
    assert pes.attention == [{'calc_id': 'pes/c0', 'state': 'manual',
                              'reason': 'end of the process not known'}]
    group = orch.relaunch(SYSTEM, 'pes/c0')
    manifest, _ = read_manifest(layout.job_dir(run.dir, group) / MANIFEST_FILE)
    assert [(l.calc_id, l.n, l.kind, l.level) for l in manifest.calculations] == \
        [('pes/c0', 2, 'manual', 1)]
    assert run.slurm.submitted == 3  # two jobs of the first group, one for the relaunch
    orch.run_pass()
    assert run.slurm.submitted == 3


@pytest.mark.parametrize('point, reason', [
    ('claim', 'empty claim released in part: attempt 1 is not in the record yet; release the '
              'claim again'),
    ('record', 'empty claim released in part: attempt 1 is recorded, not archived yet; release '
               'the claim again'),
])
@pytest.mark.parametrize('group', [None, f'{SYSTEM}.pes.1'])
def test_release_claim_interrupted_is_completed_by_running_it_again(run, monkeypatch, point,
                                                                    reason, group):
    orch = orchestrator(run, ['match', 'pes'])
    orch.run_pass()
    work(run, only=['pes/c1'])
    claim = run.path('pes/c0') / layout.CLAIM_FILE
    claim.write_text('')
    orch.run_pass()

    def stop(*args, **kwargs):
        raise Interrupt
    with monkeypatch.context() as patch:
        if point == 'claim':  # after the claim got the number of its attempt, before the record
            patch.setattr(archive_module, 'write_record', stop)
        else:  # after the record got the attempt, before the claim was removed
            interrupt(patch, archive_module, 'write_record')
        with pytest.raises(Interrupt):
            orch.release_claim(SYSTEM, 'pes/c0', group)
    assert json.loads(claim.read_text()) == released_claim(1)
    assert len(run.record('pes/c0').attempts) == (0 if point == 'claim' else 1)
    (pes,) = [s for s in orch.run_pass().steps if s.step == 'pes']
    assert pes.attention == [{'calc_id': 'pes/c0', 'state': 'uncertain', 'reason': reason}]
    assert orch.release_claim(SYSTEM, 'pes/c0', group) == 1
    assert not claim.exists()
    (attempt,) = run.record('pes/c0').attempts  # the same attempt, none added
    assert attempt.remedies == {'kind': 'initial', 'level': 1, 'values': {}, 'initial_values': {},
                                'restart': False}
    (pes,) = [s for s in orch.run_pass().steps if s.step == 'pes']
    assert pes.status == 'failed'
    assert pes.attention == [{'calc_id': 'pes/c0', 'state': 'manual',
                              'reason': 'end of the process not known'}]
    written = orch.relaunch(SYSTEM, 'pes/c0')
    manifest, _ = read_manifest(layout.job_dir(run.dir, written) / MANIFEST_FILE)
    assert [(l.calc_id, l.n, l.kind, l.level) for l in manifest.calculations] == \
        [('pes/c0', 2, 'manual', 1)]
    assert run.slurm.submitted == 3  # two jobs of the first group, one for the relaunch


def test_release_claim_resumed_with_a_group_of_another_launch_is_refused(run, monkeypatch):
    orch = orchestrator(run, ['match', 'pes'])
    orch.run_pass()
    work(run, 'crashed')
    orch.run_pass()  # group pes.2 lists attempt 2 of both, unchanged
    claim = run.path('pes/c0') / layout.CLAIM_FILE
    claim.write_text('')  # the worker of attempt 2 stopped right after creating its claim
    with monkeypatch.context() as patch:
        interrupt(patch, archive_module, 'write_record')
        with pytest.raises(Interrupt):
            orch.release_claim(SYSTEM, 'pes/c0')
    # Another group lists attempt 2 of pes/c0 with another launch.
    write_job(run.dir, Manifest(name=f'{SYSTEM}.pes.9', scheduler='slurm', run_command='pw.x',
                                walltime=3600, max_seconds_margin=60,
                                calculations=(Launch(SYSTEM, 'pes/c0', 2, 'manual', 1),)),
              run.jobscript)
    before = run.files('pes/c0'), run.record('pes/c0')
    with pytest.raises(OrchestratorError, match=rf'attempt 2 is recorded with the launch '
                                                rf'unchanged, level 1; group {SYSTEM}\.pes\.9 '
                                                rf'gives manual, level 1'):
        orch.release_claim(SYSTEM, 'pes/c0', f'{SYSTEM}.pes.9')
    assert (run.files('pes/c0'), run.record('pes/c0')) == before


def test_release_claim_refuses_a_released_claim_on_an_attempt_of_a_worker(run):
    orch = orchestrator(run, ['match', 'pes'])
    orch.run_pass()
    work(run, only=['pes/c1'])  # attempt 1 of pes/c1 run by the worker of a job
    claim = run.path('pes/c1') / layout.CLAIM_FILE
    claim.write_text(json.dumps(released_claim(1)))
    before = run.files('pes/c1'), run.record('pes/c1')
    with pytest.raises(OrchestratorError, match='the claim is not empty'):
        orch.release_claim(SYSTEM, 'pes/c1')
    assert (run.files('pes/c1'), run.record('pes/c1')) == before


@pytest.mark.parametrize('content, message', [
    (None, 'no claim'),
    (json.dumps({'n': 1, 'job_id': '1000', 'host': 'node01', 'pid': 77, 'since': 'x'}),
     'the claim is not empty'),
    # As left by an interrupted release, but for another attempt than the next one.
    (json.dumps({'n': 2, 'job_id': None, 'host': None, 'pid': None, 'since': None}),
     'the claim is not empty'),
])
def test_release_claim_refuses_a_claim_with_content(run, content, message):
    orch = orchestrator(run, ['match', 'pes'])
    orch.run_pass()
    claim = run.path('pes/c0') / layout.CLAIM_FILE
    if content is not None:
        claim.write_text(content)
    before = run.files('pes/c0'), run.record('pes/c0')
    with pytest.raises(OrchestratorError, match=message):
        orch.release_claim(SYSTEM, 'pes/c0')
    assert (run.files('pes/c0'), run.record('pes/c0')) == before


def test_release_claim_of_a_calculation_not_of_a_step(run):
    orch = orchestrator(run, ['match', 'pes'])
    with pytest.raises(OrchestratorError, match='not a calculation of a step'):
        orch.release_claim(SYSTEM, 'pes/c0')


def test_relaunch_with_group_when_the_launch_is_not_recorded(run):
    orch = orchestrator(run, ['match', 'pes'])
    orch.run_pass()
    work(run, 'crashed')
    orch.run_pass()  # group pes.2 lists attempt 2 of both, unchanged
    # Another group lists attempt 2 of pes/c0 with another launch, and attempt 2
    # was recorded without its launch (its job not followed by the manager).
    write_job(run.dir, Manifest(name=f'{SYSTEM}.pes.3', scheduler='slurm', run_command='pw.x',
                                walltime=3600, max_seconds_margin=60,
                                calculations=(Launch(SYSTEM, 'pes/c0', 2, 'manual', 1),)),
              run.jobscript)
    record = run.record('pes/c0')
    second = store.Attempt(2, job_id='4242', status=store.Status('crashed'), stop_reason='truncated',
                           job_state='FAILED')
    store.write_record(run.sdir, store.Record(**{**record.__dict__,
                                                 'attempts': record.attempts + (second,),
                                                 'status': second.status}))
    with pytest.raises(PolicyError, match=rf'launch of attempt 2 is not recorded: .*different '
                                          rf'launches: {SYSTEM}\.pes\.2 \(unchanged, level 1\), '
                                          rf'{SYSTEM}\.pes\.3 \(manual, level 1\)'):
        orch.relaunch(SYSTEM, 'pes/c0')
    group = orch.relaunch(SYSTEM, 'pes/c0', group=f'{SYSTEM}.pes.2')
    manifest, _ = read_manifest(layout.job_dir(run.dir, group) / MANIFEST_FILE)
    assert [(l.calc_id, l.n, l.kind, l.level) for l in manifest.calculations] == \
        [('pes/c0', 3, 'manual', 1)]


def test_relaunch_refused_when_settings_changed(run):
    orch = orchestrator(run, ['match', 'pes'])
    orch.run_pass()
    work(run, 'user_request')
    orch.run_pass()
    orch.steps['pes'].fingerprint = OTHER
    before = (run.dir / 'orchestrator.json').read_bytes(), run.slurm.submitted
    with pytest.raises(OrchestratorError, match='settings changed since the preparation: '
                                                'pes/c0: fingerprint differs'):
        orch.relaunch(SYSTEM, 'pes/c0')
    assert ((run.dir / 'orchestrator.json').read_bytes(), run.slurm.submitted) == before
    assert sorted(p.name for p in (run.dir / 'jobs').iterdir()) == [f'{SYSTEM}.pes.1']


# --- restarts ------------------------------------------------------------------------------

def restarted(run, requested=('match', 'pes')):
    """A new orchestrator and job manager on the run; the workers follow the new manager."""
    run.manager = JobManager(run.dir, run.slurm, 3600, now=run.clock)
    return orchestrator(run, requested, manager=run.manager), run.manager


def test_restart_follows_the_jobs_without_new_submissions(run):
    orch = orchestrator(run, ['match', 'pes'])
    orch.run_pass()
    orch, manager = restarted(run)
    report = orch.run_pass()
    assert run.slurm.submitted == 2 and report.submitted == []
    assert statuses(report) == {'match': 'completed', 'pes': 'running'}
    assert prepared(orch) == {'match': 0, 'pes': 0}


class Interrupt(BaseException):
    """Stop of the orchestrator at a given point, as a kill: no handler of errors catches it."""


def interrupt(monkeypatch, target, name):
    original = getattr(target, name)

    def stop(*args, **kwargs):
        original(*args, **kwargs)
        raise Interrupt
    monkeypatch.setattr(target, name, stop)


@pytest.mark.parametrize('point', ['prepared', 'archived', 'written', 'registered', 'pass'])
def test_interrupted_pass_is_completed_without_duplicates(run, monkeypatch, point):
    orch = orchestrator(run, ['match', 'pes'])
    if point == 'prepared':
        with monkeypatch.context() as patch:
            interrupt(patch, orch.steps['pes'], 'prepare')
            with pytest.raises(Interrupt):
                orch.run_pass()
    else:
        orch.run_pass()
        work(run, 'crashed')
        target, name = {'archived': (orchestrator_module, 'archive_attempt'),
                        'written': (orchestrator_module, 'write_job'),
                        'registered': (run.manager, 'add_group'),
                        'pass': (orch, '_save')}[point]
        with monkeypatch.context() as patch:
            interrupt(patch, target, name)
            with pytest.raises(Interrupt):
                orch.run_pass()
    orch, manager = restarted(run)
    for _ in range(2):
        orch.run_pass()
    groups = sorted(p.name for p in (run.dir / 'jobs').iterdir())
    expected = [f'{SYSTEM}.pes.1'] + ([] if point == 'prepared' else [f'{SYSTEM}.pes.2'])
    assert groups == expected
    assert sorted(manager.groups) == expected
    assert run.slurm.submitted == 2 * len(expected)
    for name in expected:
        submitted = json.loads((layout.job_dir(run.dir, name) / SUBMITTED_FILE).read_text())
        assert len(submitted['job_ids']) == 2
    if point != 'prepared':
        # Each calculation has its second attempt in one group only.
        manifest, _ = read_manifest(layout.job_dir(run.dir, expected[1]) / MANIFEST_FILE)
        assert sorted(l.calc_id for l in manifest.calculations) == ['pes/c0', 'pes/c1']


def test_invalid_state_file(run):
    (run.dir).mkdir(parents=True, exist_ok=True)
    (run.dir / 'orchestrator.json').write_text(json.dumps({'schema_version': 3, 'systems': {}}))
    with pytest.raises(OrchestratorError, match='schema_version: 3; this version reads 2'):
        orchestrator(run, ['match'])


def test_state_file_of_schema_1_is_not_read(run):
    # Schema 1 had no group of the job of an analysis in the state of a step.
    entry = {'status': 'completed', 'reason': None, 'calculations': [], 'error': None,
             'calcs': {}, 'attention': [], 'updated': None}
    (run.dir / 'orchestrator.json').write_text(json.dumps(
        {'schema_version': 1, 'systems': {SYSTEM: {'match': entry}}}))
    with pytest.raises(OrchestratorError, match='schema_version: 1; this version reads 2'):
        orchestrator(run, ['match'])
    (run.dir / 'orchestrator.json').write_text(json.dumps(
        {'schema_version': 2, 'systems': {SYSTEM: {'match': entry}}}))
    with pytest.raises(OrchestratorError, match=f'{SYSTEM}.match: missing job'):
        orchestrator(run, ['match'])


# --- steps still current -------------------------------------------------------------------

def completed_pes(run):
    """match and pes completed by a first orchestrator."""
    orch = orchestrator(run, ['match', 'pes'])
    orch.run_pass()
    work(run)
    assert statuses(orch.run_pass()) == {'match': 'completed', 'pes': 'completed'}


def test_completed_step_no_longer_current_runs_again(run):
    completed_pes(run)
    # A new orchestrator on the same run, after a change that makes the entry of pes stale.
    orch, _ = restarted(run)
    orch.steps['pes'].versions['pes'] = 2
    assert orch.steps['pes'].done(run.dir, SYSTEM) is False
    report = orch.run_pass()
    assert statuses(report) == {'match': 'completed', 'pes': 'completed'}
    assert orch.steps['pes'].calls == ['prepare', 'analyse']
    assert orch.steps['pes'].done(run.dir, SYSTEM) is True
    assert run.slurm.submitted == 2  # its calculations are current: reused, not run again


def test_completed_step_still_current_is_not_run(run):
    completed_pes(run)
    orch, _ = restarted(run)
    report = orch.run_pass()
    assert statuses(report) == {'match': 'completed', 'pes': 'completed'}
    assert prepared(orch) == {'match': 0, 'pes': 0} and orch.steps['pes'].calls == []


def test_stale_prerequisite_not_requested_blocks(run):
    completed_pes(run)
    orch = orchestrator(run, ['adhesion'], versions={'pes': 2})
    report = orch.run_pass()
    assert statuses(report) == {'adhesion': 'blocked'}
    assert orch.entry(SYSTEM, 'adhesion')['reason'] == ('entries missing or not current: pes; '
                                                        'run the steps that compute them first')
    assert prepared(orch) == {'adhesion': 0} and run.slurm.submitted == 2
    state = json.loads((run.dir / 'orchestrator.json').read_text())
    assert state['systems'][SYSTEM]['pes']['status'] == 'completed'


# --- settings changed after the preparation --------------------------------------------------

def complete_by_hand(run, *calc_ids, outcome='completed'):
    """Each calculation run once by a worker, its job over."""
    for calc_id in calc_ids:
        run.start(calc_id, '900', output=OUTPUTS[outcome])
        run.finish(calc_id, **OUTCOMES[outcome])


@pytest.mark.parametrize('phase', ['submit_step', 'analyse_step', 'run_pass'])
def test_settings_changed_after_prepare_stop_the_step(run, phase):
    write_entry(run.dir, SYSTEM, 'match')
    orch = orchestrator(run, ['pes'])
    assert orch.prepare_step(SYSTEM, 'pes').status == 'running'
    orch.steps['pes'].fingerprint = OTHER  # e.g. ecutwfc changed in the settings
    if phase == 'submit_step':
        pes, written = orch.submit_step(SYSTEM, 'pes')
        assert written == []
    elif phase == 'analyse_step':
        complete_by_hand(run, 'pes/c0', 'pes/c1')
        pes = orch.analyse_step(SYSTEM, 'pes')
    else:
        (pes,) = orch.run_pass().steps
    assert pes.status == 'failed'
    assert pes.reason.startswith('settings changed since the preparation: '
                                 'pes/c0: fingerprint differs from the settings')
    assert orch.entry(SYSTEM, 'pes')['error'].startswith('StalePreparation: ')
    assert not (run.dir / 'jobs').exists() and run.slurm.submitted == 0
    assert 'analyse' not in orch.steps['pes'].calls
    assert statuses(orch.run_pass()) == {'pes': 'failed'}  # until retry


def pending_group_of_pes(run, monkeypatch):
    """pes prepared, its group written and registered, its jobs not submitted; the orchestrator.

    As after a submission resolved as without job, a refusal cleared or a crash.
    """
    write_entry(run.dir, SYSTEM, 'match')
    orch = orchestrator(run, ['pes'])
    orch.prepare_step(SYSTEM, 'pes')
    with monkeypatch.context() as patch:
        patch.setattr(run.manager, 'submit', lambda *args, **kwargs: [])
        orch.submit_step(SYSTEM, 'pes')
    assert run.manager.missing_jobs(f'{SYSTEM}.pes.1') == 2 and run.slurm.submitted == 0
    return orch


@pytest.mark.parametrize('phase', ['submit_step', 'run_pass'])
def test_group_written_before_settings_changed_is_not_submitted(run, monkeypatch, phase):
    orch = pending_group_of_pes(run, monkeypatch)
    orch.steps['pes'].fingerprint = OTHER
    if phase == 'submit_step':
        pes, _ = orch.submit_step(SYSTEM, 'pes')
    else:
        (pes,) = orch.run_pass().steps
    assert pes.status == 'failed' and pes.reason.startswith('settings changed')
    assert run.slurm.submitted == 0
    orch.run_pass()  # the step stays failed, its group is not submitted
    assert run.slurm.submitted == 0


def test_group_of_a_preparation_before_retry_is_not_submitted(run, monkeypatch):
    orch = pending_group_of_pes(run, monkeypatch)
    orch.steps['pes'].fingerprint = OTHER
    assert statuses(orch.run_pass()) == {'pes': 'failed'}
    orch.retry(SYSTEM, 'pes')  # nothing prepared now: the group is of the preparation before
    assert statuses(orch.run_pass()) == {'pes': 'failed'}  # its calculations are still there
    orch.run_pass()
    assert run.slurm.submitted == 0


@pytest.mark.parametrize('phase', ['submit_step', 'run_pass'])
def test_group_of_a_step_whose_plan_fails_is_not_submitted(run, monkeypatch, phase):
    orch = pending_group_of_pes(run, monkeypatch)

    def plan(run_dir, system):
        raise RuntimeError('settings not readable')
    monkeypatch.setattr(orch.steps['pes'], 'plan', plan)
    if phase == 'submit_step':
        pes, _ = orch.submit_step(SYSTEM, 'pes')
    else:
        (pes,) = orch.run_pass().steps
    assert pes.status == 'failed' and 'settings not readable' in pes.reason
    assert run.slurm.submitted == 0
    orch.run_pass()
    assert run.slurm.submitted == 0


def test_prepare_that_keeps_stale_calculations_fails_the_step(run):
    write_entry(run.dir, SYSTEM, 'match')
    run.calc('c0', step='pes')  # prepared with the settings of before
    run.calc('c1', step='pes')
    orch = orchestrator(run, ['pes'])
    orch.steps['pes'].fingerprint = OTHER
    report = orch.run_pass()
    assert statuses(report) == {'pes': 'failed'}
    assert orch.entry(SYSTEM, 'pes')['reason'].startswith(
        'settings changed since the preparation: ')
    assert not (run.dir / 'jobs').exists() and run.slurm.submitted == 0


def test_retry_after_settings_changed(run):
    write_entry(run.dir, SYSTEM, 'match')
    orch = orchestrator(run, ['pes'])
    orch.prepare_step(SYSTEM, 'pes')
    orch.steps['pes'].fingerprint = OTHER
    assert statuses(orch.run_pass()) == {'pes': 'failed'}
    orch.retry(SYSTEM, 'pes')
    # The calculations prepared before are still there: failed again.
    assert statuses(orch.run_pass()) == {'pes': 'failed'}
    orch.steps['pes'].fingerprint = FP  # settings as they were
    orch.retry(SYSTEM, 'pes')
    report = orch.run_pass()
    assert statuses(report) == {'pes': 'running'} and report.submitted == [f'{SYSTEM}.pes.1']


# --- phases of one step --------------------------------------------------------------------

def test_phases_by_hand(run):
    # Each phase with a new orchestrator and job manager, as each command of the CLI.
    orch, _ = restarted(run)
    assert orch.analyse_step(SYSTEM, 'match').status == 'completed'
    orch, _ = restarted(run)
    pes = orch.prepare_step(SYSTEM, 'pes')
    assert (pes.system, pes.step, pes.status) == (SYSTEM, 'pes', 'running')
    assert orch.entry(SYSTEM, 'pes')['calculations'] == ['pes/c0', 'pes/c1']
    assert not (run.dir / 'jobs').exists() and run.slurm.submitted == 0
    orch, _ = restarted(run)
    pes, written = orch.submit_step(SYSTEM, 'pes')
    assert written == [f'{SYSTEM}.pes.1'] and run.slurm.submitted == 2
    assert pes.status == 'running'
    work(run)
    orch, _ = restarted(run)
    pes, written = orch.submit_step(SYSTEM, 'pes')
    assert written == [] and (pes.status, pes.reason) == ('running',
                                                          'calculations completed: run analyse')
    orch, _ = restarted(run)
    assert orch.analyse_step(SYSTEM, 'pes').status == 'completed'
    assert orch.steps['pes'].calls == ['analyse']
    assert read_system(run.sdir).get(['pes']) == {'done': True}
    state = json.loads((run.dir / 'orchestrator.json').read_text())
    assert state['systems'][SYSTEM]['pes']['status'] == 'completed'


def prepared_pes(run):
    """match analysed and pes prepared by hand; the orchestrator."""
    orch = orchestrator(run, ['match', 'pes'])
    orch.analyse_step(SYSTEM, 'match')
    orch.prepare_step(SYSTEM, 'pes')
    return orch


def snapshot(run):
    return {name: (run.dir / name).read_bytes() if (run.dir / name).exists() else None
            for name in ('orchestrator.json', 'jobs.json')} | \
           {'system.json': (run.sdir / 'system.json').read_bytes()}


@pytest.mark.parametrize('case, reason', [
    ('running', 'attempt 1 recorded as running'),
    ('claim', 'claim of attempt 1, the record has 0 attempts'),
    ('empty claim', 'empty claim'),
    ('crashed', 'attempt 1 crashed'),
    ('not taken', f'attempt 1 listed in group {SYSTEM}.pes.1, not taken yet'),
    ('not_parsed', 'attempt 1 completed, its output not parsed'),
    ('parse_failed', 'attempt 1 completed, its output not parsed: no "!    total energy" in the '
                     'last SCF cycle'),
    ('other model', 'attempt 1 completed, its results cannot be used: output of another energy '
                    'model (ecutwfc: expected 40.0, observed 45.0)'),
])
def test_analyse_step_uses_only_the_files(run, case, reason):
    orch = prepared_pes(run)
    if case == 'not taken':
        orch.submit_step(SYSTEM, 'pes')
        work(run, only=['pes/c1'])
    else:
        complete_by_hand(run, 'pes/c1')
        if case == 'running':
            run.start('pes/c0', '900')
        elif case == 'claim':
            run.claim('pes/c0', 1, '900')
        elif case == 'empty claim':
            (run.path('pes/c0') / layout.CLAIM_FILE).write_text('')
        elif case in ('not_parsed', 'parse_failed'):
            complete_by_hand(run, 'pes/c0', outcome=case)
        elif case == 'other model':
            complete_by_hand(run, 'pes/c0')
            store.write_record(run.sdir, dataclasses.replace(run.record('pes/c0'),
                                                             observed={'ecutwfc': 45.0}))
        else:
            complete_by_hand(run, 'pes/c0', outcome='crashed')
    before, queries = snapshot(run), run.slurm.queries
    orch, _ = restarted(run)
    with pytest.raises(OrchestratorError) as error:
        orch.analyse_step(SYSTEM, 'pes')
    message = str(error.value)
    assert f'pes/c0: {reason}' in message and 'pes/c1' not in message
    assert run.slurm.queries == queries  # no query to the scheduler
    assert snapshot(run) == before and 'analyse' not in orch.steps['pes'].calls
    assert orch.entry(SYSTEM, 'pes')['status'] == 'running'


def test_analyse_step_of_a_step_not_prepared_is_refused(run):
    write_entry(run.dir, SYSTEM, 'match')
    orch = orchestrator(run, ['pes'])
    with pytest.raises(OrchestratorError, match='pes is not prepared'):
        orch.analyse_step(SYSTEM, 'pes')
    assert prepared(orch) == {'pes': 0} and not (run.dir / 'orchestrator.json').exists()


def test_submit_step_needs_a_prepared_step(run):
    orch = orchestrator(run, ['match', 'pes'])
    with pytest.raises(OrchestratorError, match='pes is waiting.*prepare pes'):
        orch.submit_step(SYSTEM, 'pes')
    assert not (run.dir / 'orchestrator.json').exists() and run.slurm.submitted == 0


def test_submit_step_twice_submits_nothing_twice(run):
    orch = prepared_pes(run)
    orch.submit_step(SYSTEM, 'pes')
    orch, manager = restarted(run)
    pes, written = orch.submit_step(SYSTEM, 'pes')
    assert written == [] and pes.status == 'running'
    assert run.slurm.submitted == 2 and sorted(manager.groups) == [f'{SYSTEM}.pes.1']


def test_phases_by_hand_then_the_orchestrator_goes_on(run):
    orch = prepared_pes(run)
    orch.submit_step(SYSTEM, 'pes')
    orch, manager = restarted(run)
    report = orch.run_pass()
    assert report.submitted == [] and statuses(report) == {'match': 'completed', 'pes': 'running'}
    work(run)
    report = orch.run_pass()
    assert statuses(report) == {'match': 'completed', 'pes': 'completed'}
    assert sorted(manager.groups) == [f'{SYSTEM}.pes.1'] and run.slurm.submitted == 2


def test_step_without_calculations_is_analysed(run):
    orch = orchestrator(run, ['match'])
    report = orch.analyse_step(SYSTEM, 'match')
    assert report.status == 'completed' and orch.steps['match'].calls == ['prepare', 'analyse']
    assert run.slurm.queries == 0 and run.slurm.submitted == 0
    assert read_system(run.sdir).get(['match']) == {'done': True}
