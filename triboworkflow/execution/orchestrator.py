# Author: Vito Foderà
# Created: 2026-10-29

"""Orchestrator of the steps of a workflow on HPC.

The steps of each system follow the graph of the entries they read and write
(``INTERFACE_GRAPH``): ``match`` → ``pes`` → {``plotpes`` → ``shear``;
``adhesion`` → {``ppes``; ``charge``}}, with ``shear`` reading also the
entries of ``pes`` and ``match``, and ``match`` → ``pes_scf`` →
``plotpes_scf``. ``pes`` and ``pes_scf`` (the potential energy surface relaxed
along z and the static one) are independent; ``plotpes`` and ``plotpes_scf``
interpolate the one and the other and read also the entries of ``match``;
``shear`` depends on ``pes`` for now. ``adhesion`` reads the surface chosen in
``[interface.adhesion]`` and the entries of ``match``
(``interface_graph``). A pass
(``Orchestrator.run_pass``):

1. runs a cycle of the job manager (``JobManager.cycle``): jobs recovered,
   states asked, missing jobs submitted, only for the steps whose
   calculations prepared are checked current first (``_held_back``);
2. registers the groups of jobs written and not registered (an interruption
   between writing and registering a group);
3. for each system, goes through the requested steps in the order of the
   graph, the order given deciding only between steps that are both ready:

   - a completed step is asked again whether its entries are done and
     current (``Step.done``): one no longer current goes back to waiting and
     is evaluated again in the same pass;
   - a step whose requested dependency failed or is blocked is ``blocked``;
     one whose dependency is not completed yet is ``waiting``;
   - a ready step whose analysis needs a compute node
     (``Step.manual_analysis``) is never analysed by the orchestrator: a
     step whose entries are already done and current is ``completed``, one
     whose prerequisites are missing or stale is ``blocked``; any other,
     with ``[execution.analysis]``, gets a group of jobs with
     ``analysis.json`` (``jobs.py``), submitted for one job that runs the
     analysis on a compute node (``cli/analysis_job.py``), and is
     ``running`` while the job is followed; once the job is over the step
     is ``completed`` from its entries, or ``failed`` with the state and
     the outcome of the job, never submitted again automatically. Without
     ``[execution.analysis]`` it is ``manual``, with the command that
     analyses it (``analyse_step``), and its descendants are blocked until
     it is completed;
   - a ready step starts only when no other step of the system is running,
     so the steps of a system run one at a time, the analysis in a job of
     its own included; systems are independent;
   - starting: a step whose entries are already done and current
     (``Step.done``) is ``completed`` without running; otherwise
     ``check_prerequisites`` checks the entries it reads, and a missing or
     stale one makes it ``blocked`` with what is missing, nothing being
     computed again; then ``Step.prepare`` writes its calculations;
   - right after the preparation, and before a running step is followed,
     the calculations prepared are compared with those the settings give now
     (``Step.plan``: ``calc_id`` and fingerprint): a calculation missing, one
     no longer given, or a fingerprint that differs makes the step
     ``failed`` until ``retry``, and none of its calculations is used;
   - a running step is followed: its calculations are reconciled
     (``reconcile.py``) and decided (``policy.py``); the launches go, after
     the archiving of the last attempt, in a new group of jobs, registered
     and submitted; when every calculation is completed, ``Step.analyse``
     writes the entries and the step is ``completed``; a calculation
     completed whose output was not parsed, or whose results cannot be used
     (``_unusable``), needs the user and is never relaunched; when what is
     left needs the user, the step has ``failed``, with the calculations and
     why;

4. saves the state in ``<run>/orchestrator.json`` and returns the report.

A failure blocks only the descendants of the step: the other branches go on,
and a branch that succeeds does not make the workflow succeed while other
requested steps failed or are blocked. A step failed for its calculations
goes back to running when the user relaunches them; one failed for an error
of its own (``prepare`` or ``analyse``), or whose job of the analysis ended
without completing it, stays failed until ``retry``. A step
left to the user (``manual``) ends the passes as a failure does, without
making the workflow succeed: once the user has analysed it, it is completed,
and the next passes go on with its descendants. An analysis by hand stopped
before its end leaves the step running with no calculations and no job: a
pass leaves it to the user again, or gives it to a job with
``[execution.analysis]``, and the next analysis starts it from the beginning.

The state on disk is in two levels: the steps of each system in
``orchestrator.json`` (status, reason, calculations prepared, error, group
of the job of the analysis), the jobs in ``jobs.json`` (``manager.py``).
Everything else is read again at every pass from the records, the claims, the groups and the scheduler, so after a
restart the orchestrator recovers the jobs, reconciles the attempts and only
then decides new submissions, without duplicates. A pass interrupted at any
point is completed by the next one: an attempt archived without its group is
decided again, a group written and not registered is registered, a group
registered and not submitted is submitted, and the group of an analysis
whose job is not over is taken again instead of writing another.

The three phases of a step can also be run one at a time, on one step, with
the same state: ``prepare_step`` (as the start in a pass, without the
scheduler), ``submit_step`` (a cycle of the jobs, then the step followed and
its launches submitted, without the analysis; for a step analysed in a job
of its own, that job) and ``analyse_step`` (the analysis, decided from the
files alone: records, claims, manifests and the registry of the jobs, never
the scheduler). Phases run by hand and passes take turns on a run.

Calculations that differ from the settings are never overwritten: the user
restores the settings, or replaces them (``replace_step``), which moves
their directories to ``<run>/replaced/`` and prepares the step again
(``preparation.py``). While a replacement is not completed, every phase,
pass and action of the user stops and names the command that completes it.

Notes
-----
The phases saved after each change and read again at a restart follow SCS
(``main.py``, ``read_scs_restart``, ``write_restart``, ``forward_phase``),
with the state in JSON and a graph of the steps in place of the sequence of
phases; a failed step blocks its descendants instead of stopping the whole
run.
"""

import datetime
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from triboworkflow.core.store import (PrerequisiteError, check_prerequisites, file_digest,
                                      read_record, read_system, reuse_problem)
from triboworkflow.core.store.atomic import read_json, write_json
from triboworkflow.core.store.record import check_keys

from .archive import ArchiveError, archive_attempt, read_claim, released_in_part
from .jobs import (MANIFEST_FILE, OUTCOME_FILE, SUBMITTED_FILE, Analysis, JobError, Manifest,
                   is_analysis, is_withdrawn, write_analysis, write_job)
from .layout import CLAIM_FILE, JOBS_DIR, calc_dir, job_dir, system_dir
from .manager import HOLD_MESSAGES
from .policy import LaunchSetup, decide, manual_launch
from .preparation import (check_calculations, complete_replacement, prepare_calculations,
                          start_replacement)
from .reconcile import LaunchError, Reconciler, launch_for, run_launches
from .replacement import mark_done, unfinished
from .scheduler import REVERSIBLE_STATES

logger = logging.getLogger(__name__)

# Steps whose entries each step reads (verified on the entries read and
# written by the steps of the interface workflow).
# pes and pes_scf need only match. plotpes interpolates pes and plotpes_scf
# interpolates pes_scf, each reading also the high-symmetry points of match.
# Which of the two PES shear reads is decided with that step, and until then
# it depends on pes. adhesion reads the PES that [interface.adhesion] names
# and the entries of match (the slabs, the sites): here the default, pes;
# interface_graph gives the graph of a run.
# To remove: Q38 (shear); the adhesion is decided in D56.
INTERFACE_GRAPH = {
    'match': (),
    'pes': ('match',),
    'pes_scf': ('match',),
    'plotpes': ('pes', 'match'),
    'plotpes_scf': ('pes_scf', 'match'),
    'shear': ('plotpes', 'pes', 'match'),
    'adhesion': ('pes', 'match'),
    'ppes': ('adhesion',),
    'charge': ('adhesion',),
}


def interface_graph(adhesion_pes='pes'):
    """Graph of the steps of a run: ``INTERFACE_GRAPH`` with the PES of ``adhesion``.

    Parameters
    ----------
    adhesion_pes : str, optional
        Step whose surface ``adhesion`` reads, ``[interface.adhesion].pes``:
        ``pes`` or ``pes_scf``.

    Returns
    -------
    dict
        Name → steps whose entries the step reads. With ``pes_scf``, ``adhesion``
        no longer depends on ``pes``: it neither waits for it nor fails with it.

    Raises
    ------
    OrchestratorError
        If ``adhesion_pes`` is not a surface of the graph.
    """
    if adhesion_pes not in ('pes', 'pes_scf'):
        raise OrchestratorError(f'adhesion reads pes or pes_scf, not {adhesion_pes!r}')
    return INTERFACE_GRAPH | {'adhesion': (adhesion_pes, 'match')}

STATE_FILE = 'orchestrator.json'
STATE_VERSION = 2
# A step is manual when it is ready and its analysis is left to the user.
STEP_STATES = ('waiting', 'running', 'completed', 'failed', 'blocked', 'manual')
# Reason of a manual step, left to the user without [execution.analysis].
MANUAL_REASON = ('to analyse by hand: run "triboworkflow analyse {step}" where there is enough '
                 'memory, or give [execution.analysis] to analyse it in a job of its own')
# What a job of an analysis that wrote no outcome tells.
NO_OUTCOME = 'no outcome written: the job stopped before its end (walltime, memory, node)'

# Outcomes of the calculations of a running step after a pass: completed;
# waiting for their group, their run or their held job; launched in this
# pass; uncertain; to the user.
CALC_OUTCOMES = ('completed', 'pending', 'running', 'held', 'launched', 'uncertain', 'manual')
_IN_FLIGHT = ('pending', 'running', 'held', 'launched', 'uncertain')

_STATE_KEYS = ('schema_version', 'systems')
_STEP_KEYS = ('status', 'reason', 'calculations', 'error', 'calcs', 'attention', 'job',
              'updated')


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
    manual_analysis : bool
        True for a step without calculations whose analysis needs more
        resources than the login node gives, so that the orchestrator never
        runs it: with ``[execution.analysis]`` a pass submits it in a job of
        its own on a compute node and follows the job; without it a pass
        leaves it ``manual``, to be analysed by hand
        (``Orchestrator.analyse_step``, command ``analyse``) on resources the
        user chooses. False by default.
    """

    name = None
    manual_analysis = False

    def done(self, run_dir, system):
        """Whether the entries of the step are written and current (``is_done``).

        Asked also of a completed step, at every pass and by every command:
        one no longer current runs again.
        """
        return False

    def expected(self, run_dir, system):
        """``{entry: Provenance}`` expected now for the entries the step reads.

        ``check_prerequisites`` checks them before the step starts.
        """
        return {}

    def calculations(self, run_dir, system):
        """The calculations as the settings make them now, nothing written.

        Each one as ``preparation.prepare_calculations`` takes it (e.g. a
        ``QECalculation`` of ``preparation.workflow_calculation``). The
        default ``plan`` and ``prepare`` are made of it, and a replacement
        (``Orchestrator.replace_step``) compares it with the directories. A
        step of analysis returns an empty list.
        """
        return []

    def plan(self, run_dir, system):
        """Calculations that the settings give now: ``{calc_id: fingerprint}``.

        Nothing is written. Before a step is followed, submitted or analysed,
        and right after ``prepare``, the calculations prepared are compared
        with this plan; a step of analysis returns an empty dict.
        """
        return {calc.calc_id: calc.record.fingerprint
                for calc in self.calculations(run_dir, system)}

    def prepare(self, run_dir, system):
        """Write the calculations of the step and return their ``calc_id``.

        Inputs and records already prepared and current are kept as they
        are (``prepare_calculations``). A calculation that differs from the
        settings is never overwritten: ``prepare`` raises an error listing
        those calculations. A step of analysis only returns an empty list.
        """
        return prepare_calculations(run_dir, system, self.calculations(run_dir, system),
                                    step=self.name)

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
        Why a step is waiting, blocked, failed or manual.
    calcs : dict
        Number of calculations of the step by outcome (``CALC_OUTCOMES``).
    attention : list of dict
        ``{'calc_id', 'state', 'reason'}`` of each calculation that needs the
        user or is uncertain.
    job : str or None
        Group of jobs of the analysis of the step in a job of its own, if
        any.
    """

    system: str
    step: str
    status: str
    reason: str | None
    calcs: dict
    attention: list
    job: str | None = None


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
        """True when no step is waiting or running; a manual step waits for the user."""
        return all(s.status in ('completed', 'failed', 'blocked', 'manual') for s in self.steps)

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


def _unusable(record):
    """Why the results of a calculation whose last attempt completed cannot be used.

    Its output was not parsed, or its parsing failed, or the record does not
    stand for its own fingerprint (``reuse_problem``: no results, an energy
    model not fully known, an output of another energy model). The
    fingerprint itself is compared with the settings by ``_prepared_current``.

    Returns
    -------
    str or None
        The reason, None if the results can be used.
    """
    k = len(record.attempts)
    if record.status.parsing is None:
        return f'attempt {k} completed, its output not parsed'
    if record.status.parsing == 'failed':
        return f'attempt {k} completed, its output not parsed: {record.attempts[-1].parse_error}'
    problem = reuse_problem(record, record.fingerprint)
    if problem is not None:
        return f'attempt {k} completed, its results cannot be used: {problem}'
    return None


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
    analysis_jobscript : path-like or None, optional
        Jobscript of ``[execution.analysis]`` (``jobs.analysis_jobscript_path``):
        the steps with ``Step.manual_analysis`` are analysed in a job of their
        own. None, the default: they are left to the user.

    Raises
    ------
    OrchestratorError
        If the steps do not fit the graph or the state file is not valid.
    """

    def __init__(self, run_dir, systems, steps, manager, execution, jobscript,
                 graph=INTERFACE_GRAPH, analysis_jobscript=None):
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
        self.analysis_jobscript = None if analysis_jobscript is None else Path(analysis_jobscript)
        self.path = self.run_dir / STATE_FILE
        self.state = self._load()

    def _launch_setup(self):
        """Command and jobscript of the attempts of the next groups (``policy.LaunchSetup``)."""
        try:
            digest = file_digest(self.jobscript)
        except OSError:
            digest = None
        return LaunchSetup(self.execution.run_command, digest)

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
                                        'error': None, 'calcs': {}, 'attention': [], 'job': None,
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
        OrchestratorError
            If a replacement of calculations is not completed.
        """
        self._check_replacements()
        held = self._held_back()
        self.manager.cycle(held)
        self._register_written_groups()
        reconciler = Reconciler(self.run_dir, self.manager, self.execution.polling_interval)
        written = []
        for system in self.systems:
            for name in self.order:
                written += self._step(reconciler, system, name)
        if written:
            self.manager.submit(held - set(written))
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

    def _check_replacements(self, allow=None):
        """Stop while a replacement of calculations is not completed, but the one of ``allow``.

        Raises
        ------
        OrchestratorError
            Naming each replacement not completed and the command that
            completes it.
        """
        found = [r for r in unfinished(self.run_dir) if (r.system, r.step) != allow]
        if found:
            raise OrchestratorError('; '.join(
                f'replacement of the calculations of {r.system}/{r.step} not completed '
                f'({r.path}): run "triboworkflow prepare {r.step} --replace" to complete it'
                for r in found))

    def _held_back(self):
        """Groups not to submit: all but those of the steps whose calculations are current.

        The jobs still missing from a group are submitted only while its step
        has calculations prepared, no error of its own, is not completed, and
        those calculations still match the settings: each such step is checked
        first (``_prepared_current``), and fails if they no longer match or
        its plan cannot be read. The groups of every other step are held back,
        among them those of a step failed for an error of its own and those of
        a preparation before ``retry``; so is a group of no step of the run.
        A group written later in the pass is written after that check.
        """
        current = set()
        for system in self.systems:
            for name in self.order:
                entry = self.entry(system, name)
                if (entry['error'] is None and entry['calculations'] is not None
                        and entry['status'] != 'completed'
                        and self._prepared_current(system, name, entry)):
                    current.add(f'{system}.{name}.')
        return {group for group in self.manager.groups
                if not any(group.startswith(prefix) for prefix in current)}

    def _register_written_groups(self):
        """Register the groups written and neither registered, submitted nor withdrawn.

        A group of calculations asks for the workers of ``[execution]``, the
        group of an analysis for one job.
        """
        root = self.run_dir / JOBS_DIR
        if not root.is_dir():
            return
        for path in sorted(root.iterdir()):
            analysis = is_analysis(path)
            if (((path / MANIFEST_FILE).is_file() or analysis)
                    and not (path / SUBMITTED_FILE).exists()
                    and path.name not in self.manager.groups and not is_withdrawn(path)):
                logger.warning('Group %s written and not registered: registered now', path.name)
                self.manager.add_group(path.name, 1 if analysis else self.execution.groups)

    def _busy(self, system, name):
        """The other steps of a system that run: the steps of a system run one at a time.

        A step analysed in a job of its own runs while its job is followed;
        one analysed by hand never runs in a pass, and found running it is an
        analysis by hand stopped before its end, which holds back nothing.
        """
        return [other for other in self.order if other != name
                and self.entry(system, other)['status'] == 'running'
                and (not self.steps[other].manual_analysis
                     or self.entry(system, other)['job'] is not None)]

    def _step(self, reconciler, system, name):
        """Advance one step of one system; the names of the groups written."""
        entry = self.entry(system, name)
        if self._still_completed(system, name, entry):
            return []
        if entry['error'] is not None:
            self._set(entry, 'failed', entry['reason'])
            return []
        requested = [dep for dep in self.graph[name] if dep in self.steps]
        stopped = [dep for dep in requested
                   if self.entry(system, dep)['status'] in ('failed', 'blocked', 'manual')]
        if stopped:
            states = ', '.join(f'{dep} {self.entry(system, dep)["status"]}' for dep in stopped)
            self._set(entry, 'blocked', f'depends on {states}')
            return []
        pending = [dep for dep in requested if self.entry(system, dep)['status'] != 'completed']
        if pending:
            self._set(entry, 'waiting', f'waits for {", ".join(pending)}')
            return []
        if self.steps[name].manual_analysis:
            if self.analysis_jobscript is None and entry['job'] is None:
                self._leave_to_user(system, name, entry)
                return []
            return self._analyse_in_job(system, name, entry)
        if entry['calculations'] is None:
            busy = self._busy(system, name)
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
        current = self._prepared_current(system, name, entry)
        self._save()
        return current

    def _leave_to_user(self, system, name, entry):
        """Leave a ready step analysed only by hand to the user; nothing is prepared nor analysed.

        As at a start, a step whose entries are already done and current is
        ``completed``, one whose prerequisites are missing or stale is
        ``blocked``, and an error of the step fails it; any other is
        ``manual``, with the command that analyses it (``MANUAL_REASON``). A
        step found running is an analysis by hand stopped before its end:
        nothing of it is done and current, and it is left to the user again.
        """
        if not self._ready(system, name, entry):
            return
        if entry['status'] == 'running':
            logger.warning('%s/%s: its analysis by hand stopped before its end; it starts again '
                           'from the beginning', system, name)
        if entry['status'] != 'manual':
            logger.warning('%s/%s ready, %s', system, name, MANUAL_REASON.format(step=name))
        self._set(entry, 'manual', MANUAL_REASON.format(step=name), calculations=None, calcs={},
                  attention=[])

    def _ready(self, system, name, entry):
        """Whether a step without calculations is ready to be analysed, checked as at a start.

        A step whose entries are already done and current is ``completed``,
        one whose prerequisites are missing or stale is ``blocked``, and an
        error of the step fails it: False for these, True otherwise.
        """
        step = self.steps[name]
        sdir = system_dir(self.run_dir, system)
        try:
            if step.done(self.run_dir, system):
                self._set(entry, 'completed', 'entries already done and current')
                return False
            check_prerequisites(read_system(sdir), step.expected(self.run_dir, system), sdir)
        except PrerequisiteError as error:
            self._set(entry, 'blocked', str(error), calculations=None)
            return False
        except Exception as error:  # an error of the step: the other steps go on
            self._fail(entry, 'checking', error)
            return False
        return True

    # --- analysis in a job of its own ------------------------------------------------

    def _analyse_in_job(self, system, name, entry, wait=True):
        """Start or follow the analysis of a step in a job of its own; the groups written.

        A step whose job is known is followed (``_follow_analysis``).
        Otherwise it is checked as at a start (``_ready``), waits while
        another step of the system runs (unless ``wait`` is False, for a
        phase run by hand), and its group is written with ``analysis.json``
        and registered for one job; the step is then ``running`` with that
        group in ``job`` and no calculations, and the state is saved. The
        last group of the step, written and never submitted before an
        interruption, or whose job may still run, is taken again instead of
        writing another: two jobs never run the same analysis because of a
        restart or a ``retry``.
        """
        if entry['job'] is not None:
            return self._follow_analysis(system, name, entry)
        if not self._ready(system, name, entry):
            return []
        if entry['status'] == 'running':
            logger.warning('%s/%s: its analysis by hand stopped before its end; it runs again '
                           'from the beginning, in a job of its own', system, name)
        busy = self._busy(system, name) if wait else []
        if busy:
            self._set(entry, 'waiting', f'ready; {", ".join(busy)} runs first', calculations=None)
            return []
        group = self._group_to_take_again(system, name)
        try:
            if group is None:
                group = self._group_name(system, name)
                write_analysis(self.run_dir, Analysis(group, system, name,
                                                      self.execution.scheduler),
                               self.analysis_jobscript)
                logger.info('Group %s: analysis of %s/%s in a job of its own', group, system,
                            name)
            else:
                logger.warning('Group %s of the analysis of %s/%s, whose job is not over, '
                               'taken again', group, system, name)
            if group not in self.manager.groups:
                self.manager.add_group(group, 1)
        except (JobError, OSError) as error:
            self._fail(entry, 'writing the job of the analysis', error)
            return []
        self._set(entry, 'running', f'analysis in a job of its own, group {group}',
                  calculations=[], job=group, calcs={}, attention=[])
        self._save()
        return [group]

    def _group_to_take_again(self, system, name):
        """The last group of the analysis of a step, if never submitted or its job not over."""
        prefix = f'{system}.{name}.'
        root = self.run_dir / JOBS_DIR
        found = sorted((int(p.name[len(prefix):]), p) for p in root.iterdir()
                       if p.name.startswith(prefix) and p.name[len(prefix):].isdigit()) \
            if root.is_dir() else []
        if not found:
            return None
        path = found[-1][1]
        if not is_analysis(path) or is_withdrawn(path):
            return None
        over, _ = self._analysis_job(path.name)
        return None if over else path.name

    def _analysis_job(self, group):
        """Where the job of the group of an analysis stands, from the job manager.

        Returns
        -------
        over : bool
            Whether every job of the group is over: in a final state
            confirmed by the last query, or in a state after which the
            scheduler may requeue a job (``REVERSIBLE_STATES``, e.g. Slurm
            ``PREEMPTED``) reported again at least ``polling_interval``
            seconds after its first report.
        text : str
            The job and its state, or why it is not over.
        """
        manager, scheduler = self.manager, self.manager.scheduler
        if group not in manager.groups:
            return False, 'job not submitted yet'
        info = manager.groups[group]
        if info['refused'] is not None:
            return False, (f'submission of the job refused: {info["refused"]}; fix '
                           f'jobs/{group}/jobscript.sh (keep its last line, the analysis) and the '
                           f'jobscript of [execution.analysis], then "triboworkflow clear-refusal '
                           f'{group}"')
        if info['unknown']:
            return False, ('submission of the job of outcome not known: look for the job in the '
                           f'queue, then "triboworkflow resolve-submission {group} --job-id ID" '
                           'or "--no-job"')
        job_ids = manager.group_jobs(group)
        if not job_ids or manager.missing_jobs(group):
            return False, 'job to submit'
        reversible = REVERSIBLE_STATES.get(scheduler.scheduler_name, ())
        text = None
        for job_id in job_ids:
            job = manager.jobs[job_id]
            state = job['state']
            reported = job['reported'] is True and job['error'] is None
            if reported and scheduler.is_final(state):
                text = f'job {job_id} {state}'
                continue
            if reported and job['active'] is False and scheduler.state_name(state) in reversible:
                waited = (datetime.datetime.fromisoformat(job['checked'])
                          - datetime.datetime.fromisoformat(job['state_since'])).total_seconds()
                if waited >= self.execution.polling_interval:
                    text = f'job {job_id} {state}, not requeued'
                    continue
                return False, (f'job {job_id} {state}: the scheduler may requeue it; to be '
                               'confirmed by a later query')
            status = manager.job_status(job_id)
            if status == 'held':
                return False, f'job {job_id} {state}: {HOLD_MESSAGES[job["hold"]["reason"]]}'
            if job['error'] is not None:
                return False, f'job {job_id} {state}; the last query failed: {job["error"]}'
            if status == 'attention':
                return False, (f'job {job_id}: not reported by the scheduler, or in a state not '
                               f'known (last known state {state}); check it')
            return False, f'job {job_id} {state or "submitted, not reported yet"}'
        return True, text

    def _outcome(self, group):
        """What the job of an analysis wrote in ``outcome.json``, in words."""
        path = job_dir(self.run_dir, group) / OUTCOME_FILE
        if not path.is_file():
            return NO_OUTCOME
        try:
            data = read_json(path)
            result, error = data['result'], data['error']
        except (OSError, ValueError, KeyError, TypeError) as error:
            return f'{OUTCOME_FILE} not read: {error}'
        return f'{result}: {error}' if error else str(result)

    def _follow_analysis(self, system, name, entry):
        """Follow the job of the analysis of a step; the step is never analysed here.

        A step whose entries are done and current is ``completed``, whatever
        its job. While the job may run the step stays ``running``, with the
        state of the job. Once the job is over without the step done, the
        step is ``blocked`` if its prerequisites are no longer current, and
        ``failed`` otherwise, with the state of the job and its outcome: it
        is never submitted again automatically, since the same job would
        fail in the same way (memory, walltime), and ``retry`` lets the next
        pass submit a new job.
        """
        group = entry['job']
        step = self.steps[name]
        try:
            if step.done(self.run_dir, system):
                self._set(entry, 'completed', f'analysed in the job of group {group}')
                return []
        except Exception as error:
            self._fail(entry, 'checking', error)
            return []
        if group not in self.manager.groups and not is_analysis(job_dir(self.run_dir, group)):
            logger.warning('%s/%s: group %s of its analysis not found: a new one is written',
                           system, name, group)
            self._set(entry, 'waiting', f'group {group} of its analysis not found',
                      calculations=None, job=None)
            return []
        over, text = self._analysis_job(group)
        if not over:
            self._set(entry, 'running', text)
            return []
        sdir = system_dir(self.run_dir, system)
        try:
            check_prerequisites(read_system(sdir), step.expected(self.run_dir, system), sdir)
        except PrerequisiteError as error:
            self._set(entry, 'blocked', str(error), calculations=None, job=None)
            return []
        except Exception as error:
            self._fail(entry, 'checking', error)
            return []
        outcome = self._outcome(group)
        reason = (f'{text} without completing the step: {outcome}; output of the job in '
                  f'jobs/{group}; give the jobscript of [execution.analysis] more memory or time '
                  f'if it needs them, then "triboworkflow retry {name}"')
        logger.error('%s/%s: %s', system, name, reason)
        self._set(entry, 'failed', reason, error=f'AnalysisJobFailed: {text}: {outcome}')
        return []

    def _fail(self, entry, doing, error):
        logger.error('Step failed while %s: %s', doing, error)
        self._set(entry, 'failed', f'error while {doing}: {error}',
                  error=f'{type(error).__name__}: {error}')

    def _still_completed(self, system, name, entry):
        """Whether a completed step is still done and current (``Step.done``).

        A completed step no longer current goes back to waiting, with nothing
        prepared, and is evaluated again; an error of ``done`` fails it.
        """
        if entry['status'] != 'completed':
            return False
        try:
            if self.steps[name].done(self.run_dir, system):
                return True
        except Exception as error:
            self._fail(entry, 'checking', error)
            return False
        self._set(entry, 'waiting', 'entries no longer current', calculations=None, calcs={},
                  attention=[], job=None)
        return False

    def _stale(self, system, name, entry):
        """Differences between the plan of a step and its calculations prepared.

        Returns
        -------
        list of str
            ``'<calc_id>: <reason>'`` for each calculation of the plan not
            prepared, each one prepared and no longer in the plan, and each
            record whose fingerprint differs from the plan. A record that
            cannot be read is left to the reconciliation.
        """
        plan = self.steps[name].plan(self.run_dir, system)
        prepared = list(entry['calculations'])
        sdir = system_dir(self.run_dir, system)
        found = [f'{calc_id}: in the settings, not prepared'
                 for calc_id in plan if calc_id not in prepared]
        for calc_id in prepared:
            if calc_id not in plan:
                found.append(f'{calc_id}: prepared, no longer in the settings')
                continue
            try:
                record = read_record(sdir, calc_id)
            except (OSError, ValueError):
                continue
            if record.fingerprint != plan[calc_id]:
                found.append(f'{calc_id}: fingerprint differs from the settings')
        return found

    def _prepared_current(self, system, name, entry):
        """Whether the calculations prepared still match the settings (``_stale``).

        If not, the step has failed until ``retry`` and none of them is used.
        """
        try:
            found = self._stale(system, name, entry)
        except Exception as error:
            self._fail(entry, 'checking the calculations', error)
            return False
        if found:
            differences = '; '.join(found)
            logger.error('%s/%s: settings changed since the preparation: %s', system, name,
                         differences)
            self._set(entry, 'failed', f'settings changed since the preparation: {differences}',
                      error=f'StalePreparation: {differences}')
            return False
        return True

    def _follow(self, reconciler, system, name, entry, analyse=True):
        """Reconcile, decide and relaunch the calculations of a running step.

        A calculation completed counts as such only when its results can be
        used (``_unusable``); otherwise it needs the user, with the reason.
        With ``analyse`` False, a step whose calculations are all completed
        stays running, to be analysed by ``analyse_step``.
        """
        if not self._prepared_current(system, name, entry):
            return []
        states = reconciler.reconcile([(system, calc_id) for calc_id in entry['calculations']])
        calcs, attention, launches = Counter(), [], []

        def note(outcome, state, reason):
            calcs[outcome] += 1
            if outcome in ('uncertain', 'manual'):
                attention.append({'calc_id': state.calc_id, 'state': state.state,
                                  'reason': reason})

        sdir = system_dir(self.run_dir, system)
        setup = self._launch_setup()
        for state in states:
            decision = decide(self.run_dir, state, setup)
            if decision.action == 'done':
                try:
                    why = _unusable(read_record(sdir, state.calc_id))
                except (OSError, ValueError) as error:
                    note('uncertain', state, f'record not read: {error}')
                    continue
                note('completed' if why is None else 'manual', state, why)
            elif decision.action == 'wait':
                note(state.state, state, decision.reason)
            elif decision.action != 'launch':
                note(decision.action, state, decision.reason)
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
        if calcs['completed'] == len(states) and not analyse:
            self._set(entry, 'running', 'calculations completed: run analyse')
        elif calcs['completed'] == len(states):
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

    # --- phases of one step ------------------------------------------------------

    def prepare_step(self, system, name):
        """Prepare one step, as a pass would start it, without the scheduler.

        A completed step still current, a step already prepared and a step
        failed for an error of its own are left as they are; for a step
        already prepared and running, the calculations that differ from the
        settings now are logged (``check_calculations``). A step with
        ``Step.manual_analysis`` has nothing to prepare: without
        ``[execution.analysis]`` it is left to the user as in a pass
        (``_leave_to_user``); with it, a step whose job is known is left as it
        is, and any other is checked as at a start (``_ready``) and left
        waiting for its job, which ``submit_step`` or a pass submits.

        Returns
        -------
        StepReport

        Raises
        ------
        OrchestratorError
            If a replacement of calculations is not completed.
        """
        self._check_replacements()
        entry = self.entry(system, name)
        if self._still_completed(system, name, entry) or entry['error'] is not None:
            pass
        elif self.steps[name].manual_analysis:
            if entry['job'] is not None:
                pass
            elif self.analysis_jobscript is None:
                self._leave_to_user(system, name, entry)
            elif self._ready(system, name, entry):
                self._set(entry, 'waiting', 'ready: nothing to prepare; its analysis runs in a '
                          f'job of its own, submitted by "submit {name}" or by the orchestrator',
                          calculations=None)
        elif entry['calculations'] is None:
            self._start(system, name, entry)
        elif entry['status'] == 'running':
            self._report_differences(system, name)
        self._save()
        return self.step_report(system, name)

    def _report_differences(self, system, name):
        """Log the calculations of a prepared step that differ from the settings now."""
        try:
            checks = check_calculations(self.run_dir, system,
                                        self.steps[name].calculations(self.run_dir, system))
        except Exception as error:
            logger.error('%s/%s: calculations not compared with the settings: %s', system, name,
                         error)
            return
        differ = [check for check in checks if check.kind in ('kept', 'changed')]
        for check in differ:
            logger.warning('%s/%s: prepared, differs from the settings: %s', system,
                           check.calc_id, '; '.join(check.reasons + check.warnings))
        if differ:
            logger.warning('To apply the settings to them: "triboworkflow prepare %s --replace"',
                           name)

    def replace_step(self, system, name, command=None):
        """Replace the calculations of a step that differ from the settings, then prepare it.

        The calculations ``kept`` or ``changed`` (``check_calculations``)
        move to ``<run>/replaced/`` with their attempts and outputs
        (``preparation.py``): every check comes first, and a calculation
        that a worker or a job may still use stops the whole replacement,
        with nothing changed. Then the step goes back to waiting, as with
        ``retry``, the journal is marked done and the step is prepared. A
        replacement of this step not completed is completed first, with the
        calculations of its journal.

        Parameters
        ----------
        system, name : str
            The step.
        command : str or None, optional
            Command that asked for it, written in the journal.

        Returns
        -------
        report : StepReport
            Of the step after its preparation.
        replacement : triboworkflow.execution.replacement.Replacement or None
            The replacement made or completed; None if no calculation
            differs.

        Raises
        ------
        OrchestratorError
            If a replacement of another step is not completed.
        ReplacementError
            If the replacement is refused (nothing changed) or cannot be
            completed.
        """
        self._check_replacements(allow=(system, name))
        entry = self.entry(system, name)
        found = [r for r in unfinished(self.run_dir) if (r.system, r.step) == (system, name)]
        for replacement in found:
            logger.warning('Replacement %s not completed: completed now', replacement.path)
        if not found:
            calcs = self.steps[name].calculations(self.run_dir, system)
            at = self.manager.now().isoformat(timespec='seconds')
            replacement = start_replacement(self.run_dir, system, name, calcs, self.manager,
                                            command, at)
            found = [replacement] if replacement is not None else []
        done = None
        for replacement in found:
            complete_replacement(self.run_dir, replacement, self.manager)
            self._set(entry, 'waiting', 'calculations replaced', error=None, calculations=None,
                      calcs={}, attention=[])
            self._save()
            done = mark_done(replacement)
            logger.info('%s/%s: %d calculations replaced, kept in %s', system, name,
                        len(done.calculations), done.path)
        return self.prepare_step(system, name), done

    def submit_step(self, system, name):
        """Submit the calculations of a prepared step: one pass on that step, without analysis.

        The jobs of the run are recovered and asked for first
        (``JobManager.cycle``); the calculations are then reconciled and
        launched as in a pass. When they are all completed the step stays
        running, with that reason, until ``analyse_step``.

        A step with ``Step.manual_analysis`` is analysed in a job of its own
        (``[execution.analysis]``): the job is written and submitted, or
        followed if it is known, as in a pass (``_analyse_in_job``), without
        waiting for the other steps of the system. Once the job is over the
        step is completed from its entries, or failed.

        Returns
        -------
        report : StepReport
        written : list of str
            Groups of jobs written.

        Raises
        ------
        OrchestratorError
            If the step is not running (not prepared, completed or failed),
            or it is analysed by hand only, without ``[execution.analysis]``.
        NotOnHPCError
            If the commands of the scheduler are not found.
        """
        self._check_replacements()
        entry = self.entry(system, name)
        if self.steps[name].manual_analysis:
            if self.analysis_jobscript is None and entry['job'] is None:
                raise OrchestratorError(
                    f'{system}/{name} has no calculations to submit: it is analysed by hand, with '
                    f'"analyse {name}", or in a job of its own with [execution.analysis]')
            return self._submit_analysis(system, name, entry)
        if entry['status'] != 'running':
            raise OrchestratorError(f'{system}/{name} is {entry["status"]}: only a prepared step '
                                    f'is submitted; run "prepare {name}" first')
        held = self._held_back()
        self.manager.cycle(held)
        self._register_written_groups()
        reconciler = Reconciler(self.run_dir, self.manager, self.execution.polling_interval)
        written = self._follow(reconciler, system, name, entry, analyse=False)
        if written:
            self.manager.submit(held)
        self._save()
        return self.step_report(system, name), written

    def _submit_analysis(self, system, name, entry):
        """``submit_step`` of a step analysed in a job of its own."""
        held = self._held_back()
        self.manager.cycle(held)
        self._register_written_groups()
        if self._still_completed(system, name, entry) or entry['error'] is not None:
            written = []
        else:
            written = self._analyse_in_job(system, name, entry, wait=False)
        if written:
            self.manager.submit(held - set(written))
        self._save()
        return self.step_report(system, name), written

    def analyse_step(self, system, name):
        """Analyse a step from the files of its calculations, without the scheduler.

        A step without calculations is started (``Step.done``, prerequisites,
        ``prepare``), saved as running, and analysed. A step analysed only by
        hand (``Step.manual_analysis``) is analysed only here: from ``manual``
        or any state before it, or from running when an analysis by hand
        stopped before its end, started again from the beginning; the passes
        then find it completed from its entries, without analysing it again.
        Such a step whose analysis was given to a job of its own is analysed
        here only if no job of its group was ever submitted (the group is
        then withdrawn) or its job is over according to the last query of the
        job manager; while the job may still run nothing changes and an error
        says so. A step with calculations must be prepared;
        each calculation must be done according to its files alone
        (``_not_done``), otherwise nothing changes and the error says, for each
        one, what is missing.

        Returns
        -------
        StepReport

        Raises
        ------
        OrchestratorError
            If the step is not prepared, a calculation is not done according
            to its files, or the job of its analysis may still run.
        """
        self._check_replacements()
        entry = self.entry(system, name)
        if self._still_completed(system, name, entry) or entry['error'] is not None:
            self._save()
            return self.step_report(system, name)
        if self.steps[name].manual_analysis and entry['job'] is not None:
            self._take_from_job(system, name, entry)
        if entry['calculations'] is None:
            try:
                planned = self.steps[name].plan(self.run_dir, system)
            except Exception as error:
                raise OrchestratorError(f'{system}/{name}: its calculations are not known: '
                                        f'{error}') from error
            if planned:
                raise OrchestratorError(f'{system}/{name} is not prepared: run "prepare {name}" '
                                        f'and "submit {name}" first')
            if not self._start(system, name, entry):
                self._save()
                return self.step_report(system, name)
        elif not self._prepared_current(system, name, entry):
            self._save()
            return self.step_report(system, name)
        missing = self._not_done(system, entry['calculations'])
        if missing:
            raise OrchestratorError(
                f'{system}/{name} not analysed: calculations not done according to their files: '
                f'{"; ".join(missing)}. The end of a job is read from the scheduler on the HPC '
                f'machine, by "submit {name}" or by the orchestrator')
        try:
            self.steps[name].analyse(self.run_dir, system, list(entry['calculations']))
        except Exception as error:
            self._fail(entry, 'analysing', error)
        else:
            self._set(entry, 'completed')
        self._save()
        return self.step_report(system, name)

    def _take_from_job(self, system, name, entry):
        """Take back from its job the analysis of a step, to analyse it by hand.

        Only the files are read: the registry of the jobs, as the last query
        left it, and ``submitted.json``. A group none of whose jobs was ever
        submitted is withdrawn, so that it is never submitted later; a group
        whose job is over is left as it is. The step is then as one not
        started.

        Raises
        ------
        OrchestratorError
            If the job of the group may still run; nothing changes.
        """
        group = entry['job']
        if self.manager.withdrawal_problem(group) is None:
            if is_analysis(job_dir(self.run_dir, group)):
                self.manager.withdraw(group, f'{system}/{name} analysed by hand')
        else:
            over, text = self._analysis_job(group)
            if not over:
                raise OrchestratorError(
                    f'{system}/{name} not analysed by hand: its analysis runs in a job of its own '
                    f'(group {group}, {text}). Wait for the job, or cancel it; its end is read '
                    f'from the scheduler on the HPC machine, by "submit {name}" or by the '
                    'orchestrator')
        self._set(entry, entry['status'], entry['reason'], calculations=None, job=None)

    def _not_done(self, system, calc_ids):
        """``'<calc_id>: <what is missing>'`` for each calculation not done according to its files.

        Done: the record has attempts, any claim is the one of the last
        attempt, no group lists a next attempt, the last attempt is not
        recorded as running and ended completed (``Status.summary``), and the
        results of the record can be used (``_unusable``): its output was
        parsed, its energy model is fully known and the output agrees with
        it. Only files are read: claims, records and manifests.
        """
        sdir = system_dir(self.run_dir, system)
        launches = run_launches(self.run_dir)
        missing = []
        for calc_id in calc_ids:
            try:
                record = read_record(sdir, calc_id)
            except (OSError, ValueError) as error:
                missing.append(f'{calc_id}: record not read: {error}')
                continue
            k = len(record.attempts)
            claim_path = calc_dir(self.run_dir, system, calc_id) / CLAIM_FILE
            claim = read_claim(claim_path) if claim_path.exists() else False
            listed = [group for group, _ in launches.get((system, calc_id), {}).get(k + 1, [])]
            if claim is None:
                why = 'empty claim: a worker is taking the calculation, or stopped right after'
            elif claim and claim['n'] != k:
                why = f'claim of attempt {claim["n"]}, the record has {k} attempts'
            elif listed:
                why = f'attempt {k + 1} listed in group {", ".join(listed)}, not taken yet'
            elif not k:
                why = 'never run'
            elif record.attempts[-1].status.process == 'running':
                why = f'attempt {k} recorded as running: its job may still run'
            elif record.attempts[-1].status.summary != 'completed':
                why = f'attempt {k} {record.attempts[-1].status.summary}'
            elif (why := _unusable(record)) is None:
                continue
            missing.append(f'{calc_id}: {why}')
        return missing

    # --- actions of the user -----------------------------------------------------

    def _step_of(self, system, calc_id):
        """Name of the step that prepared a calculation."""
        name = next((step for step in self.order
                     if calc_id in (self.entry(system, step)['calculations'] or ())), None)
        if name is None:
            raise OrchestratorError(f'{system}/{calc_id} is not a calculation of a step of the '
                                    'run')
        return name

    def relaunch(self, system, calc_id, group=None, level=None, from_scratch=False):
        """Relaunch a calculation by hand (``policy.manual_launch``) in a new group.

        The step of the calculation is followed again at the next pass.

        Parameters
        ----------
        system, calc_id : str
            The calculation.
        group : str or None, optional
            Group whose launch of the last attempt is taken, when the record
            does not have it and the groups listing it give different ones.
        level : int or None, optional
            Level of the remedies of the new attempt; that of the last attempt
            by default.
        from_scratch : bool, optional
            Start from scratch even when the last attempt could be continued
            from the files of its output directory.

        Returns
        -------
        str
            Name of the group written.

        Raises
        ------
        PolicyError
            If its worker may still run, there is nothing to run again, or
            the launch of the last attempt is not known.
        OrchestratorError
            If no step of the system has the calculation, or the settings
            changed since its step was prepared; nothing is written.
        """
        self._check_replacements()
        name = self._step_of(system, calc_id)
        entry = self.entry(system, name)
        found = self._stale(system, name, entry)
        if found:
            raise OrchestratorError(f'{system}/{name}: settings changed since the preparation: '
                                    f'{"; ".join(found)}; nothing relaunched')
        held = self._held_back()
        self.manager.cycle(held)
        reconciler = Reconciler(self.run_dir, self.manager, self.execution.polling_interval)
        launch = manual_launch(reconciler, system, calc_id, self._launch_setup(), group, level,
                               from_scratch)
        archive_attempt(self.run_dir, system, calc_id)  # nothing to do if already archived
        written = self._write_group(system, name, [launch])
        self.manager.submit(held)
        if entry['status'] == 'failed' and entry['error'] is None:
            self._set(entry, 'running', 'relaunched by hand')
        self._save()
        return written

    def release_claim(self, system, calc_id, group=None):
        """Archive the attempt of an empty claim, once the user knows its worker is over.

        An empty claim is left by a worker that stopped right after creating
        it, and stays uncertain: nothing tells whether that worker runs. The
        attempt gets the number the worker would have used, the attempts in
        the record plus one, and its launch from the groups that list it
        (``launch_for``); its process stays unknown, so the calculation then
        needs the user and is relaunched with ``relaunch``. The scheduler is
        not asked.

        A release interrupted leaves the claim with the number of the attempt
        alone (``released_in_part``) and is completed by releasing the claim
        again: before the record has the attempt, as a first release; once it
        has it, by archiving that attempt, with no other attempt added and
        the launch already recorded (``group``, if given, must give the same).

        Parameters
        ----------
        system, calc_id : str
            The calculation.
        group : str or None, optional
            Group whose launch is taken when the groups listing the attempt
            give different ones.

        Returns
        -------
        int
            Number of the attempt archived.

        Raises
        ------
        OrchestratorError
            If no step of the system has the calculation, it has no claim or
            its claim is neither empty nor the one of a release interrupted,
            the launch of the attempt is not known, or ``group`` gives another
            launch than the one recorded.
        """
        self._check_replacements()
        self._step_of(system, calc_id)
        path = calc_dir(self.run_dir, system, calc_id) / CLAIM_FILE
        where = f'{system}/{calc_id}'
        if not path.is_file():
            raise OrchestratorError(f'{where}: no claim to release')
        record = read_record(system_dir(self.run_dir, system), calc_id)
        part = released_in_part(read_claim(path), record) if path.stat().st_size else 'empty'
        if part is None:
            raise OrchestratorError(f'{where}: the claim is not empty; its job tells when its '
                                    'worker is over, and the reconciliation follows it')
        if part == 'record':
            return self._complete_release(where, system, calc_id, record.attempts[-1], group)
        n = len(record.attempts) + 1
        launch = self._launch_for(where, system, calc_id, n, group)
        archive_attempt(self.run_dir, system, calc_id, launch=launch)
        logger.info('%s: empty claim released as attempt %d (%s, level %d)', where, n,
                    launch.kind, launch.level)
        return n

    def _launch_for(self, where, system, calc_id, n, group):
        try:
            return launch_for(self.run_dir, system, calc_id, n, group)
        except LaunchError as error:
            raise OrchestratorError(f'{where}: {error}') from None

    def _complete_release(self, where, system, calc_id, attempt, group):
        """Archive the attempt recorded by a release interrupted; its number."""
        if group is not None:
            launch = self._launch_for(where, system, calc_id, attempt.n, group)
            if launch.remedies() != attempt.remedies:
                raise OrchestratorError(
                    f'{where}: attempt {attempt.n} is recorded with the launch '
                    f'{attempt.remedies.get("kind")}, level {attempt.remedies.get("level")}; '
                    f'group {group} gives {launch.kind}, level {launch.level}: nothing changed')
        archive_attempt(self.run_dir, system, calc_id)
        logger.info('%s: release of the empty claim of attempt %d completed', where, attempt.n)
        return attempt.n

    def retry(self, system, step):
        """Let a step failed for an error of its own start again at the next pass.

        A step whose analysis failed in a job of its own gets a new job at the
        next pass, unless the job of its last group may still run, which is
        then followed again.
        """
        self._check_replacements()
        entry = self.entry(system, step)
        self._set(entry, 'waiting', 'retry asked', error=None, calculations=None, job=None)
        self._save()

    # --- report ------------------------------------------------------------------

    def step_report(self, system, name):
        """``StepReport`` of the current state of one step."""
        entry = self.entry(system, name)
        return StepReport(system, name, entry['status'], entry['reason'], dict(entry['calcs']),
                          list(entry['attention']), entry['job'])

    def report(self, submitted=()):
        """``Report`` of the current state."""
        steps = [self.step_report(system, name) for system in self.systems for name in self.order]
        return Report(steps, self.manager.summary(), list(submitted))

    @staticmethod
    def summary(report):
        """One line: steps by status and jobs waiting."""
        counts = Counter(step.status for step in report.steps)
        steps = ', '.join(f'{counts[s]} {s}' for s in STEP_STATES if counts[s])
        return (f'steps {steps}; jobs {len(report.jobs.waiting)} waiting, '
                f'{len(report.jobs.held)} held, {len(report.jobs.attention)} to check; '
                f'groups written {len(report.submitted)}')
