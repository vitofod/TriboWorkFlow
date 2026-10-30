# Author: Vito Foderà
# Created: 2026-10-29

"""Orchestrator of the steps of a workflow on HPC.

The steps of each system follow the graph of the entries they read and write
(``INTERFACE_GRAPH``): ``match`` → ``pes`` → {``plotpes`` → ``shear``;
``adhesion`` → {``ppes``; ``charge``}}, with ``shear`` reading also the
entries of ``pes`` and ``match``. A pass (``Orchestrator.run_pass``):

1. runs a cycle of the job manager (``JobManager.cycle``): jobs recovered,
   states asked, missing jobs submitted;
2. registers the groups of jobs written and not registered (an interruption
   between writing and registering a group);
3. for each system, goes through the requested steps in the order of the
   graph, the order given deciding only between steps that are both ready:

   - a step whose requested dependency failed or is blocked is ``blocked``;
     one whose dependency is not completed yet is ``waiting``;
   - a ready step starts only when no other step of the system is running,
     so the steps of a system run one at a time; systems are independent;
   - starting: a step whose entries are already done and current
     (``Step.done``) is ``completed`` without running; otherwise
     ``check_prerequisites`` checks the entries it reads, and a missing or
     stale one makes it ``blocked`` with what is missing, nothing being
     computed again; then ``Step.prepare`` writes its calculations;
   - a running step is followed: its calculations are reconciled
     (``reconcile.py``) and decided (``policy.py``); the launches go, after
     the archiving of the last attempt, in a new group of jobs, registered
     and submitted; when every calculation is completed, ``Step.analyse``
     writes the entries and the step is ``completed``; when what is left
     needs the user, the step has ``failed``, with the calculations and why;

4. saves the state in ``<run>/orchestrator.json`` and returns the report.

A failure blocks only the descendants of the step: the other branches go on,
and a branch that succeeds does not make the workflow succeed while other
requested steps failed or are blocked. A step failed for its calculations
goes back to running when the user relaunches them; one failed for an error
of its own (``prepare`` or ``analyse``) stays failed until ``retry``.

The state on disk is in two levels: the steps of each system in
``orchestrator.json`` (status, reason, calculations prepared, error), the jobs
in ``jobs.json`` (``manager.py``). Everything else is read again at every
pass from the records, the claims, the groups and the scheduler, so after a
restart the orchestrator recovers the jobs, reconciles the attempts and only
then decides new submissions, without duplicates. A pass interrupted at any
point is completed by the next one: an attempt archived without its group is
decided again, a group written and not registered is registered, a group
registered and not submitted is submitted.

A launch above level 1 of the remedies is decided and not submitted, since
this version does not write its values in the input: its calculation is
reported as needing the user.

Notes
-----
The phases saved after each change and read again at a restart follow SCS
(``main.py``, ``read_scs_restart``, ``write_restart``, ``forward_phase``),
with the state in JSON and a graph of the steps in place of the sequence of
phases; a failed step blocks its descendants instead of stopping the whole
run.
"""

import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from triboworkflow.core.store import PrerequisiteError, check_prerequisites, read_system
from triboworkflow.core.store.atomic import read_json, write_json
from triboworkflow.core.store.record import check_keys

from .archive import ArchiveError, archive_attempt
from .jobs import MANIFEST_FILE, SUBMITTED_FILE, Manifest, write_job
from .layout import JOBS_DIR, system_dir
from .policy import decide, manual_launch
from .reconcile import Reconciler

logger = logging.getLogger(__name__)

# Steps whose entries each step reads (verified on the entries read and
# written by the steps of the interface workflow).
INTERFACE_GRAPH = {
    'match': (),
    'pes': ('match',),
    'plotpes': ('pes',),
    'shear': ('plotpes', 'pes', 'match'),
    'adhesion': ('pes',),
    'ppes': ('adhesion',),
    'charge': ('adhesion',),
}

STATE_FILE = 'orchestrator.json'
STATE_VERSION = 1
STEP_STATES = ('waiting', 'running', 'completed', 'failed', 'blocked')

# Outcomes of the calculations of a running step after a pass: completed;
# waiting for their group, their run or their held job; launched in this
# pass; uncertain; launch decided and not written in the input; to the user.
CALC_OUTCOMES = ('completed', 'pending', 'running', 'held', 'launched', 'uncertain',
                 'not applied', 'manual')
_IN_FLIGHT = ('pending', 'running', 'held', 'launched', 'uncertain')

_STATE_KEYS = ('schema_version', 'systems')
_STEP_KEYS = ('status', 'reason', 'calculations', 'error', 'calcs', 'attention', 'updated')


class OrchestratorError(ValueError):
    """Steps, a graph or a state file that the orchestrator cannot use."""


class Step:
    """One step of a workflow, for any system.

    Subclasses give the work of a step; the orchestrator calls these methods
    and nothing else. Each one may be called again after a restart.

    Attributes
    ----------
    name : str
        Name of the step, a key of the graph.
    """

    name = None

    def done(self, run_dir, system):
        """Whether the entries of the step are written and current (``is_done``)."""
        return False

    def expected(self, run_dir, system):
        """``{entry: Provenance}`` expected now for the entries the step reads.

        ``check_prerequisites`` checks them before the step starts.
        """
        return {}

    def prepare(self, run_dir, system):
        """Write the calculations of the step and return their ``calc_id``.

        Inputs and records already prepared and current are kept as they
        are. A step of analysis only returns an empty list.
        """
        return []

    def analyse(self, run_dir, system, calculations):
        """Read the results of the completed calculations and write the entries."""


@dataclass(frozen=True)
class StepReport:
    """Where one step of one system stands after a pass.

    Attributes
    ----------
    system, step : str
    status : str
        One of ``STEP_STATES``.
    reason : str or None
        Why a step is waiting, blocked or failed.
    calcs : dict
        Number of calculations of the step by outcome (``CALC_OUTCOMES``).
    attention : list of dict
        ``{'calc_id', 'state', 'reason'}`` of each calculation that needs the
        user or is uncertain.
    """

    system: str
    step: str
    status: str
    reason: str | None
    calcs: dict
    attention: list


@dataclass(frozen=True)
class Report:
    """Steps and jobs of a run after a pass.

    Attributes
    ----------
    steps : list of StepReport
        Every requested step of every system, in the order of the pass.
    jobs : triboworkflow.execution.manager.JobsSummary
        Summary of the job manager.
    submitted : list of str
        Groups of jobs written in this pass.
    """

    steps: list
    jobs: object
    submitted: list

    def by_status(self, status):
        """``(system, step)`` of the steps in ``status``."""
        return [(s.system, s.step) for s in self.steps if s.status == status]

    @property
    def finished(self):
        """True when no step is waiting or running."""
        return all(s.status in ('completed', 'failed', 'blocked') for s in self.steps)

    @property
    def success(self):
        """True when every requested step is completed."""
        return all(s.status == 'completed' for s in self.steps)


def order_steps(requested, graph):
    """Requested steps in an order of the graph; the given order decides between free steps.

    Raises
    ------
    OrchestratorError
        If a step is not in the graph, is requested twice, or the graph has
        a cycle among the requested steps.
    """
    unknown = [name for name in requested if name not in graph]
    if unknown:
        raise OrchestratorError(f'steps not in the graph: {", ".join(unknown)}; steps: '
                                f'{", ".join(graph)}')
    if len(set(requested)) != len(requested):
        raise OrchestratorError(f'steps requested more than once: {requested}')
    ordered, left = [], list(requested)
    while left:
        free = next((name for name in left
                     if all(dep in ordered or dep not in requested for dep in graph[name])), None)
        if free is None:
            raise OrchestratorError(f'the graph has a cycle among {", ".join(left)}')
        ordered.append(free)
        left.remove(free)
    return ordered


class Orchestrator:
    """Steps of a workflow for the systems of a run.

    Parameters
    ----------
    run_dir : path-like
        Directory of the run.
    systems : sequence of str
        Systems of the run.
    steps : mapping
        Name → ``Step`` of each requested step, in the order given by the
        user (e.g. that of the settings).
    manager : triboworkflow.execution.manager.JobManager
        Job manager of the run.
    execution : triboworkflow.core.settings.schema.ExecutionParams
        ``[execution]``: run command, walltime, margin, workers per group,
        interval of the queries.
    jobscript : path-like
        Jobscript of the user (``jobs.jobscript_path``).
    graph : mapping, optional
        Steps whose entries each step reads; ``INTERFACE_GRAPH`` by default.

    Raises
    ------
    OrchestratorError
        If the steps do not fit the graph or the state file is not valid.
    """

    def __init__(self, run_dir, systems, steps, manager, execution, jobscript,
                 graph=INTERFACE_GRAPH):
        self.run_dir = Path(run_dir)
        self.systems = list(systems)
        self.steps = dict(steps)
        for name, step in self.steps.items():
            if step.name != name:
                raise OrchestratorError(f'step {name!r} is given as {step.name!r}')
        self.graph = {name: tuple(deps) for name, deps in graph.items()}
        self.order = order_steps(list(self.steps), self.graph)
        self.manager = manager
        self.execution = execution
        self.jobscript = Path(jobscript)
        self.path = self.run_dir / STATE_FILE
        self.state = self._load()

    # --- state file --------------------------------------------------------------

    def _load(self):
        state = {'schema_version': STATE_VERSION, 'systems': {}}
        if self.path.exists():
            try:
                state = check_keys(read_json(self.path), _STATE_KEYS, 'orchestrator state')
                if state['schema_version'] != STATE_VERSION:
                    raise OrchestratorError(f'schema_version: {state["schema_version"]!r}; '
                                            f'this version reads {STATE_VERSION}')
                for system, steps in state['systems'].items():
                    for name, entry in steps.items():
                        check_keys(entry, _STEP_KEYS, f'{system}.{name}')
                        if entry['status'] not in STEP_STATES:
                            raise OrchestratorError(f'{system}.{name}: status '
                                                    f'{entry["status"]!r} not known')
            except ValueError as error:  # OrchestratorError, and StoreError from the store
                raise OrchestratorError(f'{self.path}: {error}') from None
        for system in self.systems:
            steps = state['systems'].setdefault(system, {})
            for name in self.order:
                steps.setdefault(name, {'status': 'waiting', 'reason': None, 'calculations': None,
                                        'error': None, 'calcs': {}, 'attention': [],
                                        'updated': None})
        return state

    def _save(self):
        if not self.manager.scheduler.dry_run:
            write_json(self.path, self.state)

    def entry(self, system, step):
        """State of a step of a system: ``{'status', 'reason', 'calculations', ...}``."""
        return self.state['systems'][system][step]

    def _set(self, entry, status, reason=None, **fields):
        entry.update(status=status, reason=reason, **fields)
        entry['updated'] = self.manager.now().isoformat(timespec='seconds')

    # --- pass --------------------------------------------------------------------

    def run_pass(self):
        """One pass over the jobs and the steps; the state is saved at its end.

        Returns
        -------
        Report

        Raises
        ------
        NotOnHPCError
            If the commands of the scheduler are not found.
        """
        self.manager.cycle()
        self._register_written_groups()
        reconciler = Reconciler(self.run_dir, self.manager, self.execution.polling_interval)
        written = []
        for system in self.systems:
            for name in self.order:
                written += self._step(reconciler, system, name)
        if written:
            self.manager.submit()
        self._save()
        return self.report(written)

    def run(self, sleep=time.sleep, max_passes=None, stop=None):
        """Repeat ``run_pass`` every ``polling_interval`` seconds until no step waits or runs.

        Parameters
        ----------
        sleep : callable, optional
            Function waiting a number of seconds.
        max_passes : int or None, optional
            Stop after this many passes; no limit by default.
        stop : callable or None, optional
            Called after each pass; a true value stops the loop (e.g. at a
            signal). The state of the last pass is saved.

        Returns
        -------
        Report
            Of the last pass.
        """
        passes = 0
        while True:
            report = self.run_pass()
            passes += 1
            logger.info('Pass %d: %s', passes, self.summary(report))
            if report.finished or self.manager.scheduler.dry_run:
                return report
            if (max_passes is not None and passes >= max_passes) or (stop and stop()):
                return report
            sleep(self.execution.polling_interval)
            if stop and stop():
                return report

    def _register_written_groups(self):
        """Register the groups written and neither registered nor submitted."""
        root = self.run_dir / JOBS_DIR
        if not root.is_dir():
            return
        for path in sorted(root.iterdir()):
            if ((path / MANIFEST_FILE).is_file() and not (path / SUBMITTED_FILE).exists()
                    and path.name not in self.manager.groups):
                logger.warning('Group %s written and not registered: registered now', path.name)
                self.manager.add_group(path.name, self.execution.groups)

    def _step(self, reconciler, system, name):
        """Advance one step of one system; the names of the groups written."""
        entry = self.entry(system, name)
        if entry['status'] == 'completed':
            return []
        if entry['error'] is not None:
            self._set(entry, 'failed', entry['reason'])
            return []
        requested = [dep for dep in self.graph[name] if dep in self.steps]
        stopped = [dep for dep in requested
                   if self.entry(system, dep)['status'] in ('failed', 'blocked')]
        if stopped:
            states = ', '.join(f'{dep} {self.entry(system, dep)["status"]}' for dep in stopped)
            self._set(entry, 'blocked', f'depends on {states}')
            return []
        pending = [dep for dep in requested if self.entry(system, dep)['status'] != 'completed']
        if pending:
            self._set(entry, 'waiting', f'waits for {", ".join(pending)}')
            return []
        if entry['calculations'] is None:
            busy = [other for other in self.order if other != name
                    and self.entry(system, other)['status'] == 'running']
            if busy:
                self._set(entry, 'waiting', f'ready; {", ".join(busy)} runs first')
                return []
            if not self._start(system, name, entry):
                return []
        return self._follow(reconciler, system, name, entry)

    def _start(self, system, name, entry):
        """Start a step; False if it is done, blocked or failed without following it."""
        step = self.steps[name]
        sdir = system_dir(self.run_dir, system)
        try:
            if step.done(self.run_dir, system):
                self._set(entry, 'completed', 'entries already done and current')
                return False
            check_prerequisites(read_system(sdir), step.expected(self.run_dir, system), sdir)
        except PrerequisiteError as error:
            self._set(entry, 'blocked', str(error))
            return False
        except Exception as error:  # an error of the step: the other steps go on
            self._fail(entry, 'checking', error)
            return False
        try:
            calculations = list(step.prepare(self.run_dir, system))
        except Exception as error:
            self._fail(entry, 'preparing', error)
            return False
        self._set(entry, 'running', None, calculations=calculations)
        self._save()
        return True

    def _fail(self, entry, doing, error):
        logger.error('Step failed while %s: %s', doing, error)
        self._set(entry, 'failed', f'error while {doing}: {error}',
                  error=f'{type(error).__name__}: {error}')

    def _follow(self, reconciler, system, name, entry):
        """Reconcile, decide and relaunch the calculations of a running step."""
        states = reconciler.reconcile([(system, calc_id) for calc_id in entry['calculations']])
        calcs, attention, launches = Counter(), [], []

        def note(outcome, state, reason):
            calcs[outcome] += 1
            if outcome in ('uncertain', 'not applied', 'manual'):
                attention.append({'calc_id': state.calc_id, 'state': state.state,
                                  'reason': reason})

        for state in states:
            decision = decide(self.run_dir, state)
            if decision.action == 'done':
                note('completed', state, None)
            elif decision.action == 'wait':
                note(state.state, state, decision.reason)
            elif decision.action != 'launch':
                note(decision.action, state, decision.reason)
            elif decision.launch.changes_input:
                note('not applied', state, f'{decision.reason}; attempt {decision.launch.n} at '
                                           f'level {decision.launch.level} decided, its values '
                                           'are not written in the input by this version')
            else:
                try:
                    if not decision.released:
                        archive_attempt(self.run_dir, system, state.calc_id)
                except ArchiveError as error:
                    note('uncertain', state, f'not archived: {error}')
                    continue
                launches.append(decision.launch)
                note('launched', state, decision.reason)
        written = [self._write_group(system, name, launches)] if launches else []
        entry.update(calcs=dict(calcs), attention=attention)
        if calcs['completed'] == len(states):
            try:
                self.steps[name].analyse(self.run_dir, system, list(entry['calculations']))
            except Exception as error:
                self._fail(entry, 'analysing', error)
                return written
            self._set(entry, 'completed')
        elif not any(calcs[outcome] for outcome in _IN_FLIGHT):
            self._set(entry, 'failed', f'{len(attention)} calculations need the user')
        else:
            self._set(entry, 'running')
        return written

    def _write_group(self, system, name, launches):
        """Write and register a new group of jobs for launches of one step."""
        group = self._group_name(system, name)
        manifest = Manifest.from_execution(group, launches, self.execution)
        write_job(self.run_dir, manifest, self.jobscript)
        self.manager.add_group(group, self.execution.groups)
        logger.info('Group %s: %d calculations of %s/%s', group, len(launches), system, name)
        return group

    def _group_name(self, system, name):
        """``<system>.<step>.<k>``, with ``k`` one more than the groups of the step on disk."""
        prefix = f'{system}.{name}.'
        root = self.run_dir / JOBS_DIR
        taken = [int(p.name[len(prefix):]) for p in root.iterdir()
                 if p.name.startswith(prefix) and p.name[len(prefix):].isdigit()] \
            if root.is_dir() else []
        return f'{prefix}{max(taken, default=0) + 1}'

    # --- actions of the user -----------------------------------------------------

    def relaunch(self, system, calc_id):
        """Relaunch a calculation by hand (``policy.manual_launch``) in a new group.

        The step of the calculation is followed again at the next pass.

        Returns
        -------
        str
            Name of the group written.

        Raises
        ------
        PolicyError
            If its worker may still run, or there is nothing to run again.
        OrchestratorError
            If no step of the system has the calculation.
        """
        name = next((step for step in self.order
                     if calc_id in (self.entry(system, step)['calculations'] or ())), None)
        if name is None:
            raise OrchestratorError(f'{system}/{calc_id} is not a calculation of a step of the '
                                    'run')
        self.manager.cycle()
        reconciler = Reconciler(self.run_dir, self.manager, self.execution.polling_interval)
        launch = manual_launch(reconciler, system, calc_id)
        archive_attempt(self.run_dir, system, calc_id)  # nothing to do if already archived
        group = self._write_group(system, name, [launch])
        self.manager.submit()
        entry = self.entry(system, name)
        if entry['status'] == 'failed' and entry['error'] is None:
            self._set(entry, 'running', 'relaunched by hand')
        self._save()
        return group

    def retry(self, system, step):
        """Let a step failed for an error of its own start again at the next pass."""
        entry = self.entry(system, step)
        self._set(entry, 'waiting', 'retry asked', error=None, calculations=None)
        self._save()

    # --- report ------------------------------------------------------------------

    def report(self, submitted=()):
        """``Report`` of the current state."""
        steps = [StepReport(system, name, entry['status'], entry['reason'], dict(entry['calcs']),
                            list(entry['attention']))
                 for system in self.systems for name in self.order
                 for entry in (self.entry(system, name),)]
        return Report(steps, self.manager.summary(), list(submitted))

    @staticmethod
    def summary(report):
        """One line: steps by status and jobs waiting."""
        counts = Counter(step.status for step in report.steps)
        steps = ', '.join(f'{counts[s]} {s}' for s in STEP_STATES if counts[s])
        return (f'steps {steps}; jobs {len(report.jobs.waiting)} waiting, '
                f'{len(report.jobs.held)} held, {len(report.jobs.attention)} to check; '
                f'groups written {len(report.submitted)}')
