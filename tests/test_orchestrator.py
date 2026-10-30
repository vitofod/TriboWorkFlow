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

import json

import pytest

from run_helpers import (JOB_DONE, NOT_CONVERGED, SCF_DONE, SCF_TRUNCATED, SYSTEM, Run, header,
                         text)
from triboworkflow.core import store
from triboworkflow.core.settings.schema import ExecutionParams
from triboworkflow.core.store import Provenance, SystemWriter, is_done, read_system
from triboworkflow.execution import layout
from triboworkflow.execution import orchestrator as orchestrator_module
from triboworkflow.execution.jobs import MANIFEST_FILE, SUBMITTED_FILE, read_manifest
from triboworkflow.execution.manager import JobManager
from triboworkflow.execution.orchestrator import (INTERFACE_GRAPH, Orchestrator,
                                                  OrchestratorError, Step, order_steps)
from triboworkflow.execution.policy import PolicyError

EXECUTION = ExecutionParams(scheduler='slurm', jobscript='job.sh', run_command='pw.x', groups=2,
                            walltime=3600, polling_interval=300, max_seconds_margin=60)
ALL = ('match', 'pes', 'plotpes', 'shear', 'adhesion', 'ppes', 'charge')
# Calculations of each fictitious step; match, plotpes and shear only analyse.
CALCS = {'pes': 2, 'adhesion': 2, 'ppes': 1, 'charge': 1}


class FakeStep(Step):
    """A step that prepares ``calcs`` calculations and writes the entry ``[name]``.

    Its provenance has a version: changing it makes the entry stale.
    ``versions`` holds the version each step expects of the others.
    """

    def __init__(self, run, name, versions):
        self.run, self.name, self.versions = run, name, versions
        self.calls = []

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
                self.run.calc(f'c{i}', step=self.name)
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
    'completed': dict(status=store.Status('ended', 'converged', 'n/a'), stop_reason='completed',
                      clean_stop=True),
    'crashed': dict(status=store.Status('crashed', None, 'n/a'), stop_reason='truncated',
                    exit_code=137, clean_stop=False),
    'user_request': dict(status=store.Status('ended', None, 'n/a'), stop_reason='user_request',
                         clean_stop=True),
    'not_converged': dict(status=store.Status('ended', 'not_converged', 'n/a'),
                          stop_reason='scf_not_converged', clean_stop=True),
}


OUTPUTS = {'completed': SCF_DONE, 'crashed': SCF_TRUNCATED, 'user_request': SCF_DONE,
           'not_converged': text(header(mixing_beta=0.4), NOT_CONVERGED, JOB_DONE)}


def work(run, outcome='completed', only=None):
    """Workers take the next attempt of each calculation of the groups and end it.

    Every job is then set COMPLETED. ``only`` limits the calculations taken.
    """
    for name in list(run.manager.groups):
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
    run.slurm.states.update({job_id: 'COMPLETED' for job_id in run.manager.jobs})


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


def test_remedy_decided_and_not_applied(run):
    orch = orchestrator(run, ['match', 'pes'])
    orch.run_pass()
    work(run, 'not_converged')
    report = orch.run_pass()
    (pes,) = [s for s in report.steps if s.step == 'pes']
    assert (pes.status, pes.calcs) == ('failed', {'not applied': 2})
    assert pes.attention[0]['reason'] == ('scf_not_converged; attempt 2 at level 2 decided, its '
                                          'values are not written in the input by this version')
    assert report.submitted == [] and (run.path('pes/c0') / layout.CLAIM_FILE).exists()


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


# --- restarts ------------------------------------------------------------------------------

def restarted(run, requested=('match', 'pes')):
    manager = JobManager(run.dir, run.slurm, 3600, now=run.clock)
    return orchestrator(run, requested, manager=manager), manager


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
    (run.dir / 'orchestrator.json').write_text(json.dumps({'schema_version': 2, 'systems': {}}))
    with pytest.raises(OrchestratorError, match='schema_version: 2; this version reads 1'):
        orchestrator(run, ['match'])
