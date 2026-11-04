# Author: Vito Foderà
# Created: 2026-10-29

"""Policy of the relaunches: what to do next with a reconciled calculation.

``decide`` turns the ``CalcState`` of a calculation (``reconcile.py``) into a
``Decision``:

=============  ==============================================================
``action``     When
=============  ==============================================================
``launch``     ``launch`` is to run in a new group of jobs: the first attempt,
               a relaunch, or an attempt that no worker took
``wait``       a group for the next attempt is not over, an attempt runs, or
               the job of its claim is held
``done``       the calculation is completed
``manual``     the user decides; ``reason`` says why
``uncertain``  what happened is not established (``reason``); the
               calculation is looked at again at the next pass
=============  ==============================================================

Relaunches:

- an attempt that no worker took runs in a new group with the same launch:
  no relaunch is used, since no attempt was made;
- an attempt that crashed, was killed or stopped at ``max_seconds`` runs
  again with the input of its level (``'unchanged'``), at most
  ``MAX_UNCHANGED`` times in a row at a level, then the user decides. The
  count comes from the launches recorded in the attempts of the record, so a
  restart of the workflow does not reset it; a new level of the remedies or
  a manual relaunch starts it again;
- an attempt that pw.x closed without convergence (``scf_not_converged``,
  ``max_steps``) runs at the next level of the remedies (``'remedy'``), up
  to ``MAX_LEVEL``; then the user decides.

Levels of the remedies (``level_values``). Level 1 is ``input.pwi`` as
prepared, with the values of the user (recommended ``nstep = 150``,
``mixing_beta = 0.4``). Levels 2 and 3 change ``nstep`` and ``mixing_beta``
together, whichever of the two causes stopped the attempt: ``mixing_beta``
becomes ``min(beta0, 0.2)`` and then ``min(beta0, 0.1)``, so it is never
raised; ``nstep``, for a relaxation only, becomes 200 if ``n0 <= 200``, 250 if
``200 < n0 <= 250``, ``n0`` beyond, and then ``max(n0, 250)``, so it is never
lowered. ``beta0`` and ``n0`` are the effective values at level 1: those pw.x
prints in the summary of the run (``PwStatus.mixing_beta``,
``PwStatus.nstep``), given in the input or default. When ``input.pwi``
writes them they must agree with the summary (``mixing_beta`` within the 4
decimals pw.x prints), otherwise the decision is left to the user. They are
kept in ``initial_values`` of the launches above level 1, and so in the
record.

A launch above level 1 replaces values of ``input.pwi``: the worker writes
them in ``run.pwi`` of the attempt.

Restart (``can_restart``). A relaunch continues the last attempt from the
files that pw.x left in its output directory (``Launch.restart``, written by
the worker as ``restart_mode = 'restart'``) only when pw.x closed that
attempt (process ``ended``) and:

- it stopped at ``max_seconds``: everything goes on, the interrupted SCF
  included;
- it stopped at ``max_steps``: pw.x writes the new positions and the
  wavefunctions in a form readable with any parallelization, and keeps the
  history of the optimizer;
- it stopped with ``scf_not_converged`` in a relaxation: the positions and
  the history of the optimizer go on, the SCF starts again from the last
  charge density. An scf calculation starts again from scratch;
- it stopped by request of the user (``user_request``, an exit file), for a
  relaunch asked for by the user only: such a stop never gets an automatic
  relaunch;

and ``<outdir>/<prefix>.save/`` holds ``data-file-schema.xml`` and the
charge density (``prefix`` and ``outdir`` of ``input.pwi``). After a stop of
``PER_PROCESS_STOPS`` (all but ``max_steps``) pw.x leaves files written once
per MPI process (wavefunctions, iteration of the SCF, history of the mixing),
read correctly only with the same parallelization, which pw.x does not
check. The parallelization of the new attempt is known only once it runs;
before, the restart needs what fixes it to be the same as for the last
attempt (``LaunchSetup``): the command (``Attempt.run_command``) and the
jobscript, which sets the resources of the job
(``Attempt.jobscript_digest``); and the parallelization of the last attempt
must be printed in its output, to be compared with that of the new one.
Otherwise the relaunch starts from scratch, with the reason. A restart that
then runs with another parallelization, or one not known, is not reused
(``core.store.restart_problem``): the user decides. Every other relaunch,
after a crash or a kill above all, starts from scratch.

``manual_launch`` gives the launch of a relaunch asked for by the user.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from ase.io.espresso import read_fortran_namelist

from triboworkflow.core.store import read_record
from triboworkflow.core.store.record import PER_PROCESS_STOPS
from triboworkflow.engines.qe.status import PwOutputError, read_parallelization, read_pw_status

from .archive import read_claim
from .jobs import LAUNCH_KINDS, JobError, Launch
from .layout import CLAIM_FILE, INPUT_FILE, calc_dir, system_dir
from .reconcile import LaunchError, launch_for
from .worker import WorkerError, read_input

logger = logging.getLogger(__name__)

ACTIONS = ('launch', 'wait', 'done', 'manual', 'uncertain')

MAX_UNCHANGED = 2
MAX_LEVEL = 3

# mixing_beta at the levels of the remedies, never above its value at level 1.
MIXING_BETA_LEVELS = {2: 0.2, 3: 0.1}
# nstep of a relaxation at level 2: n0 rounded up to the first of these steps
# (n0 itself above them); at level 3: at least the last one.
NSTEP_STEPS = (200, 250)
# Decimals of mixing beta in the summary of pw.x.
MIXING_BETA_DECIMALS = 4

# Reasons of the stop after which a relaunch may continue the last attempt;
# user_request gets no automatic relaunch, only one asked for by the user.
RESTART_STOPS = ('max_seconds', 'max_steps', 'scf_not_converged', 'user_request')
# Files of <outdir>/<prefix>.save/ that a restart needs: the data file with
# the positions, and the charge density as pw.x writes it without or with
# HDF5.
RESTART_FILES = (('data-file-schema.xml',), ('charge-density.dat', 'charge-density.hdf5'))


class PolicyError(ValueError):
    """A calculation whose history or input does not allow the decision asked for."""


@dataclass(frozen=True)
class LaunchSetup:
    """What fixes the parallelization of the next attempt, compared for a restart.

    Attributes
    ----------
    run_command : str
        Command of pw.x (``run_command`` of ``[execution]``).
    jobscript_digest : str or None
        ``file_digest`` of the jobscript of ``[execution]``, which sets the
        resources of the jobs; None when it cannot be read.
    """

    run_command: str
    jobscript_digest: str | None


@dataclass(frozen=True)
class Decision:
    """What to do next with a calculation.

    Attributes
    ----------
    system, calc_id : str
        The calculation.
    action : str
        One of ``ACTIONS`` (see the module docstring).
    launch : triboworkflow.execution.jobs.Launch or None
        For ``launch``, the attempt to run.
    reason : str or None
        Why: what ended the last attempt, what is awaited, what the user has
        to look at.
    released : bool
        False if the last attempt is to be archived before ``launch`` runs.
    """

    system: str
    calc_id: str
    action: str
    launch: Launch | None = None
    reason: str | None = None
    released: bool = False


def level_values(level, calculation, initial):
    """Values of the parameters of the remedies at a level.

    Parameters
    ----------
    level : int
        From 1 to ``MAX_LEVEL``.
    calculation : str
        ``'scf'`` or ``'relax'``; ``nstep`` is changed for ``'relax'`` only.
    initial : dict
        Effective values at level 1: ``mixing_beta``, and ``nstep`` for a
        relaxation.

    Returns
    -------
    dict
        ``{}`` at level 1; ``mixing_beta`` (and ``nstep``) above.

    Examples
    --------
    >>> level_values(2, 'relax', {'mixing_beta': 0.4, 'nstep': 220})
    {'mixing_beta': 0.2, 'nstep': 250}
    >>> level_values(3, 'scf', {'mixing_beta': 0.05})
    {'mixing_beta': 0.05}
    """
    if type(level) is not int or not 1 <= level <= MAX_LEVEL:
        raise ValueError(f'level: {level!r} is not an integer from 1 to {MAX_LEVEL}')
    if level == 1:
        return {}
    values = {'mixing_beta': min(initial['mixing_beta'], MIXING_BETA_LEVELS[level])}
    if calculation == 'relax':
        n0 = initial['nstep']
        if level == 2:
            values['nstep'] = next((step for step in NSTEP_STEPS if n0 <= step), n0)
        else:
            values['nstep'] = max(n0, NSTEP_STEPS[-1])
    return values


def _with(reason, more):
    """``reason; more``, or ``reason`` alone when there is nothing more."""
    return reason if more is None else f'{reason}; {more}'


def unchanged_relaunches(attempts):
    """Unchanged relaunches at the end of the history, since the last launch of another kind."""
    count = 0
    for attempt in reversed(attempts):
        if attempt.remedies.get('kind') != 'unchanged':
            break
        count += 1
    return count


def initial_values(calc_path, record, calculation):
    """Effective values of the parameters of the remedies at level 1.

    They are those recorded in the last attempt above level 1 or, without
    one, those printed in the summary of the output of the last attempt.

    Raises
    ------
    PolicyError
        If they cannot be read, or disagree with the values written in
        ``input.pwi``.
    """
    for attempt in reversed(record.attempts):
        if attempt.remedies.get('initial_values'):
            return dict(attempt.remedies['initial_values'])
    last = record.attempts[-1]
    if last.output is None or not (calc_path / last.output.path).is_file():
        raise PolicyError(f'attempt {last.n} has no output to read the effective values from')
    output = calc_path / last.output.path
    pw = read_pw_status(output, calculation)
    values = {'mixing_beta': pw.mixing_beta}
    if calculation == 'relax':
        values['nstep'] = pw.nstep
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise PolicyError(f'{output}: the summary of pw.x does not print {", ".join(missing)}')
    with open(calc_path / INPUT_FILE) as f:
        data, _ = read_fortran_namelist(f)
    given = {'mixing_beta': data.get('electrons', {}).get('mixing_beta'),
             'nstep': data.get('control', {}).get('nstep')}
    beta = given['mixing_beta']
    if beta is not None and abs(beta - values['mixing_beta']) > 0.5 * 10 ** -MIXING_BETA_DECIMALS:
        raise PolicyError(f'mixing_beta = {beta} in {INPUT_FILE}, {values["mixing_beta"]} in the '
                          f'summary of {output}')
    if 'nstep' in values and given['nstep'] is not None and given['nstep'] != values['nstep']:
        raise PolicyError(f'nstep = {given["nstep"]} in {INPUT_FILE}, {values["nstep"]} in the '
                          f'summary of {output}')
    return values


def _per_process_problem(calc_path, attempt, setup):
    """Why the files written per process by ``attempt`` may not suit the next attempt, or None."""
    n = attempt.n
    if attempt.run_command is None:
        return f'the run command of attempt {n} is not recorded'
    if attempt.run_command != setup.run_command:
        return f'attempt {n} ran with {attempt.run_command!r}, the next one with ' \
               f'{setup.run_command!r}'
    if attempt.jobscript_digest is None:
        return f'the jobscript of attempt {n} is not recorded'
    if setup.jobscript_digest is None:
        return 'the jobscript of the next attempt cannot be read'
    if attempt.jobscript_digest != setup.jobscript_digest:
        return f'attempt {n} ran with another jobscript'
    lines = ()
    if attempt.output is not None:
        try:
            lines = read_parallelization(calc_path / attempt.output.path)
        except (OSError, PwOutputError):
            pass
    if not lines:
        return f'the output of attempt {n} does not print its parallelization'
    return None


def can_restart(calc_path, attempt, calculation, setup):
    """Whether the next attempt continues ``attempt`` from the files of its output directory.

    Parameters
    ----------
    calc_path : pathlib.Path
        Directory of the calculation, with ``input.pwi``.
    attempt : triboworkflow.core.store.Attempt
        The last attempt.
    calculation : str
        ``'scf'`` or ``'relax'``.
    setup : LaunchSetup
        Command and jobscript of the next attempt.

    Returns
    -------
    restart : bool
    reason : str or None
        After a stop of ``RESTART_STOPS`` that pw.x closed, where the attempt
        continues from or why it starts from scratch; None after any other
        end, which always starts from scratch.
    """
    stop = attempt.stop_reason
    if attempt.status.process != 'ended' or stop not in RESTART_STOPS:
        return False, None
    if stop == 'scf_not_converged' and calculation != 'relax':
        return False, 'from scratch: an scf calculation without convergence is not continued'
    if stop in PER_PROCESS_STOPS:
        problem = _per_process_problem(calc_path, attempt, setup)
        if problem is not None:
            return False, ('from scratch: files written per process, read correctly only with '
                           f'the same parallelization: {problem}')
    with open(calc_path / INPUT_FILE) as f:
        data, _ = read_fortran_namelist(f)
    control = data.get('control', {})
    prefix, outdir = control.get('prefix'), control.get('outdir')
    if not isinstance(prefix, str) or not isinstance(outdir, str):
        return False, f'from scratch: prefix and outdir are not both given in {INPUT_FILE}'
    save = Path(outdir) / f'{prefix}.save'
    missing = [' or '.join(names) for names in RESTART_FILES
               if not any((calc_path / save / name).is_file() for name in names)]
    if missing:
        return False, f'from scratch: {", ".join(missing)} not found in {save.as_posix()}'
    return True, f'restart from {save.as_posix()}'


def decide(run_dir, state, setup):
    """Decision for a reconciled calculation (see the module docstring).

    Parameters
    ----------
    run_dir : path-like
        Directory of the run.
    state : triboworkflow.execution.reconcile.CalcState
        The calculation after the reconciliation.
    setup : LaunchSetup
        Command and jobscript of the attempt that a launch would run,
        compared with those of the last attempt for a restart
        (``can_restart``).

    Returns
    -------
    Decision
    """
    system, calc_id = state.system, state.calc_id

    def make(action, launch=None, reason=state.reason):
        return Decision(system, calc_id, action, launch, reason, state.released)

    if state.state in ('pending', 'running', 'held'):
        return make('wait')
    if state.state == 'completed':
        return make('done')
    if state.state in ('manual', 'uncertain'):
        return make(state.state)
    if state.state == 'new':
        return make('launch', Launch.initial(system, calc_id))
    if state.state == 'not_taken':
        return make('launch', state.launch, f'attempt {state.launch.n} not taken by the groups '
                                            f'{", ".join(state.groups)}')
    path = calc_dir(run_dir, system, calc_id)
    try:
        record = read_record(system_dir(run_dir, system), calc_id)
        last = record.attempts[-1]
        kind, level = last.remedies.get('kind'), last.remedies.get('level')
        if kind not in LAUNCH_KINDS or type(level) is not int:
            return make('manual', reason=f'{state.reason}; the launch of attempt {last.n} is '
                                         'not recorded')
        n = len(record.attempts) + 1
        if state.state == 'unchanged':
            made = unchanged_relaunches(record.attempts)
            if made >= MAX_UNCHANGED:
                return make('manual', reason=f'{state.reason}; {made} unchanged relaunches at '
                                             f'level {level} already made')
            calculation, _ = read_input(path / INPUT_FILE)
            restart, why = can_restart(path, last, calculation, setup)
            return make('launch', Launch(system, calc_id, n, 'unchanged', level,
                                         last.remedies.get('values', {}),
                                         last.remedies.get('initial_values', {}), restart),
                        _with(state.reason, why))
        if level >= MAX_LEVEL:
            return make('manual', reason=f'{state.reason} at level {level}, the last of the '
                                         'remedies')
        calculation, _ = read_input(path / INPUT_FILE)
        initial = initial_values(path, record, calculation)
        restart, why = can_restart(path, last, calculation, setup)
        return make('launch', Launch(system, calc_id, n, 'remedy', level + 1,
                                     level_values(level + 1, calculation, initial), initial,
                                     restart),
                    _with(state.reason, why))
    except (OSError, ValueError, JobError) as error:
        return make('manual', reason=f'{state.reason}; no relaunch decided: {error}')


def manual_launch(reconciler, system, calc_id, setup, group=None, level=None,
                  from_scratch=False):
    """Launch of a relaunch asked for by the user.

    It is allowed only once the worker of the last attempt is surely over:
    the attempt is not running, and the calculation has no claim, or the run
    of the job of its claim is over (``Reconciler.run_over``). The new attempt
    is of kind ``'manual'``, at the level of the last attempt with its input,
    or at ``level`` with the values of that level (``level_values``, from the
    effective values at level 1, as for a remedy), and keeps the history;
    the count of the unchanged relaunches starts again, so up to
    ``MAX_UNCHANGED`` unchanged relaunches follow it. It continues the last
    attempt from the files of its output directory on the conditions of a
    relaunch of the workflow (``can_restart``), also after a stop by request
    of the user, unless ``from_scratch``. A next attempt
    that groups listed and no worker took is launched again as it was; a
    calculation without attempts gets its first one; neither takes another
    level.

    When the last attempt has no launch recorded, its level and values come
    from the groups that list it (``launch_for``): the launch they all give,
    or the one of ``group``; never level 1 by default.

    Parameters
    ----------
    reconciler : triboworkflow.execution.reconcile.Reconciler
        Reconciliation of the run, after the cycle of its job manager.
    system, calc_id : str
        The calculation.
    setup : LaunchSetup
        Command and jobscript of the new attempt.
    group : str or None, optional
        Group whose launch of the last attempt is taken, when that launch is
        not recorded and the groups listing it give different ones.
    level : int or None, optional
        Level of the remedies of the new attempt, from 1 to ``MAX_LEVEL``;
        the level of the last attempt by default.
    from_scratch : bool, optional
        Start from scratch even when the last attempt could be continued.

    Returns
    -------
    triboworkflow.execution.jobs.Launch

    Raises
    ------
    PolicyError
        If the calculation is completed, never launched, or its worker may
        still run; if the launch of the last attempt is not recorded and the
        groups do not give it; if ``group`` is given and not needed; if
        ``level`` is not a level, cannot be computed, or is asked for an
        attempt that runs as it was listed. The message says the state found.
    """
    where = f'{system}/{calc_id}'
    if level is not None and (type(level) is not int or not 1 <= level <= MAX_LEVEL):
        raise PolicyError(f'{where}: level {level!r} is not an integer from 1 to {MAX_LEVEL}')
    state = reconciler.calculation(system, calc_id)
    if state.state == 'completed':
        raise PolicyError(f'{where} is completed: nothing to run again')
    if state.state == 'new':
        raise PolicyError(f'{where} was never launched: the workflow launches it')
    if state.state in ('pending', 'running', 'held', 'uncertain'):
        raise PolicyError(f'{where} is {state.state} ({state.reason}): its worker may still run')
    run_dir = Path(reconciler.run_dir)
    if not state.released:
        path = calc_dir(run_dir, system, calc_id)
        claim = read_claim(path / CLAIM_FILE)
        if claim is None or claim['job_id'] is None:
            raise PolicyError(f'{where}: the claim gives no job: whether its worker is over is '
                              'not known')
        verdict, detail = reconciler.run_over(claim['job_id'], claim)
        if verdict != 'over':
            raise PolicyError(f'{where}: job {claim["job_id"]} of the claim is {verdict} '
                              f'({detail})')
    record = read_record(system_dir(run_dir, system), calc_id)
    last = record.attempts[-1] if record.attempts else None
    recorded = last is None or type(last.remedies.get('level')) is int
    if group is not None and (state.launch is not None or recorded):
        raise PolicyError(f'{where}: group {group} not used: the launch is known')
    if state.launch is not None or last is None:
        launch = state.launch or Launch.initial(system, calc_id)
        if level not in (None, launch.level) or (from_scratch and launch.restart):
            raise PolicyError(f'{where}: attempt {launch.n} runs as listed, at level '
                              f'{launch.level}{", restart" if launch.restart else ""}: '
                              'no other level or start is taken')
        return launch
    if recorded:
        current, values = last.remedies['level'], last.remedies.get('values', {})
        initial = last.remedies.get('initial_values', {})
    else:
        try:
            known = launch_for(run_dir, system, calc_id, last.n, group)
        except LaunchError as error:
            raise PolicyError(f'{where}: the launch of attempt {last.n} is not recorded: '
                              f'{error}') from None
        current, values, initial = known.level, known.values, known.initial_values
    path = calc_dir(run_dir, system, calc_id)
    try:
        calculation, _ = read_input(path / INPUT_FILE)
        if level is not None and level != current:
            if level != 1:
                initial = initial or initial_values(path, record, calculation)
            values = level_values(level, calculation, initial)
            initial = initial if level > 1 else {}
        restart, why = (False, 'from scratch, as asked') if from_scratch else \
            can_restart(path, last, calculation, setup)
    except (OSError, ValueError, WorkerError) as error:
        raise PolicyError(f'{where}: no relaunch at level {level or current}: {error}') from None
    logger.info('%s: manual attempt %d at level %d, %s', where, len(record.attempts) + 1,
                level or current, why or 'from scratch')
    return Launch(system, calc_id, len(record.attempts) + 1, 'manual', level or current, values,
                  initial, restart)
