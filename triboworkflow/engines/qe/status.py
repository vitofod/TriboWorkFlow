# Author: Vito Foderà
# Created: 2026-10-26

"""States of a pw.x output.

A pw.x output tells whether a ``scf`` or ``relax`` calculation is over and,
if it is not, why it stopped. This module reads only that: the end of the
process, the outcome of the last SCF cycle, the outcome of the geometry
optimization, whether the final coordinates were written in full, and the
reason of the stop. Energies, forces and structures are read by the parser.

Only the last run of a file is read. A file may hold several runs, one after
the other, when a calculation is restarted and its output is appended: each
run begins with the line ``Program PWSCF v.<version> starts on ...``.

The states are decided from these lines of the last run:

==================================================  ==========================================
Line                                                Meaning
==================================================  ==========================================
``JOB DONE.``                                       pw.x closed the run, whatever the outcome
``Error in routine <name> (<code>):``               pw.x stopped on an error
``Self-consistent Calculation``                     an SCF cycle begins
``convergence has been achieved in``                the SCF cycle converged
``convergence NOT achieved after``                  the SCF cycle did not converge
``bfgs converged in``                               the optimizer declares convergence
``bfgs failed after``                               the optimizer gives up
``The maximum number of steps has been reached.``   the ionic steps reached ``nstep``
``Begin final coordinates``                         the final coordinates begin
``End final coordinates``                           the final coordinates end
``Maximum CPU time exceeded``                       the run reached ``max_seconds``
``Program stopped by user request``                 an exit file or a signal stopped the run
==================================================  ==========================================

pw.x writes the final coordinates when the optimizer stops, on its criteria
or because it gives up; they do not prove convergence, which only the
optimizer declares. Whether they were written in full is kept apart, because
reading the final structure needs them.

The summary that pw.x prints at the beginning of a run also gives the values
it uses, given in the input or default: ``mixing beta`` and, for a
relaxation, ``nstep``. They are read from the last run as well
(``PwStatus.mixing_beta``, ``PwStatus.nstep``), and so is the error that
stopped a run (``PwError``).

A state that the output does not establish is None. An output that ends
before any closing line is truncated: the job may still be running, or it
was stopped from outside, and only the scheduler can tell which. A truncated
output is never taken as not converged.

``record_status`` turns the states of an output into those of the record of
the calculation. What the scheduler knows about the job is passed to it
explicitly, and is used only where the output says nothing.

Notes
-----
The workflow is built for versions 7.1.x, 7.2.x and 7.3.x
(``SUPPORTED_VERSIONS``). The outputs of any other version are read with the
same lines; ``version_warning`` gives the warning the callers log for them,
since the results are not guaranteed. The lines were checked on real outputs
of 7.1 and 7.2. Those of the step limit, of ``max_seconds``, of a stop by
request and of a failed optimizer come from the source code of 7.2 and were
not checked on real outputs.

The checks start from those of Xsorb (``xsorb/calculations/results.py``,
geometry optimizations) and SCS (``labelling/QuantumEspresso/qe.py``, SCF).
"""

import re
from dataclasses import dataclass
from pathlib import Path

from triboworkflow.core.store import Status

SUPPORTED_VERSIONS = ('7.1', '7.2', '7.3')
VALIDATED_VERSIONS = ('7.1', '7.2')
CALCULATIONS = ('scf', 'relax')

FINAL_COORDINATES = ('complete', 'incomplete', 'absent', 'n/a')
STOP_REASONS = ('completed', 'scf_not_converged', 'max_steps', 'optimizer_failed',
                'max_seconds', 'user_request', 'error', 'truncated')
JOB_STATES = ('running', 'killed', 'finished')

_RUN_START = 'Program PWSCF v.'
_JOB_DONE = 'JOB DONE.'
_ERROR = 'Error in routine'
_SCF_START = 'Self-consistent Calculation'
_SCF_CONVERGED = 'convergence has been achieved in'
_SCF_NOT_CONVERGED = 'convergence NOT achieved after'
_BFGS_CONVERGED = 'bfgs converged in'
_BFGS_FAILED = 'bfgs failed after'
_MAX_STEPS = 'The maximum number of steps has been reached.'
_COORDINATES_BEGIN = 'Begin final coordinates'
_COORDINATES_END = 'End final coordinates'
_MAX_SECONDS = 'Maximum CPU time exceeded'
_USER_REQUEST = 'Program stopped by user request'
_SUMMARY_END = 'crystal axes:'
_NSTEP = 'nstep'
_MIXING_BETA = 'mixing beta'
_STOPPING = 'stopping ...'
_ERROR_HEAD = re.compile(r'(\S+?)\s*\(\s*([^)]*?)\s*\)\s*:?$')


class PwOutputError(ValueError):
    """Text that is not a pw.x output this module can read."""


@dataclass(frozen=True)
class PwError:
    """An error printed by pw.x before it stops.

    pw.x prints it between two lines of ``%``::

        Error in routine cdiaghg (309):
        problems computing cholesky

    Parameters
    ----------
    routine : str
        Routine that raised the error, as printed.
    code : str or None
        Code in parentheses after the routine, as printed; None if absent.
    message : str
        Lines of the message, stripped and joined by a space; empty if none.
    """

    routine: str
    code: str | None
    message: str

    def __str__(self):
        code = f' ({self.code})' if self.code is not None else ''
        return f'error in routine {self.routine}{code}: {self.message}'


@dataclass(frozen=True)
class PwStatus:
    """States of the last run of a pw.x output.

    Parameters
    ----------
    calculation : str
        ``'scf'`` or ``'relax'``, as given by the caller.
    version : str
        Version of pw.x printed in the header of the run (e.g. ``'7.2'``).
    runs : int
        Number of runs in the file; only the last one is read.
    process : str or None
        ``'ended'`` (``JOB DONE.``), ``'crashed'`` (an error without
        ``JOB DONE.``) or None (truncated output).
    scf : str or None
        Outcome of the last SCF cycle: ``'converged'``, ``'not_converged'``
        or None (no cycle, or a cycle without outcome).
    geometry : str or None
        ``'n/a'`` for ``scf``. For ``relax``: ``'converged'`` (declared by
        the optimizer), ``'max_steps'``, ``'not_converged'`` (the run is over
        without a declared convergence) or None (truncated output).
    final_coordinates : str
        ``'n/a'`` for ``scf``. For ``relax``: ``'complete'``,
        ``'incomplete'`` (begun, not closed) or ``'absent'``.
    stop_reason : str or None
        Why the run stopped: ``'completed'``, ``'scf_not_converged'``,
        ``'max_steps'``, ``'optimizer_failed'``, ``'max_seconds'``,
        ``'user_request'``, ``'error'``, ``'truncated'``; None when the run
        ended for a reason this module does not recognize.
    error : PwError or None, optional
        The last error of the run, as printed, when ``process`` is
        ``'crashed'``; None otherwise.
    mixing_beta : float or None, optional
        ``mixing beta`` printed in the summary of the run, i.e. the value
        pw.x uses, given or default; None if the summary does not print it.
    nstep : int or None, optional
        ``nstep`` printed in the summary of the run (relax only); None
        otherwise.
    """

    calculation: str
    version: str
    runs: int
    process: str | None
    scf: str | None
    geometry: str | None
    final_coordinates: str
    stop_reason: str | None
    error: PwError | None = None
    mixing_beta: float | None = None
    nstep: int | None = None

    @property
    def clean_stop(self):
        """True when pw.x closed the run (``JOB DONE.``).

        It is required, not sufficient, for an exact restart: whether the
        restart files are on disk is not written in the output.
        """
        return self.process == 'ended'

    @property
    def version_supported(self):
        """True when the version is one the workflow is built for (``SUPPORTED_VERSIONS``)."""
        return _release(self.version) in SUPPORTED_VERSIONS

    @property
    def version_validated(self):
        """True when the version is one checked on real outputs."""
        return _release(self.version) in VALIDATED_VERSIONS


def version_warning(version):
    """Warning for the output of a version of pw.x the workflow is not built for.

    Parameters
    ----------
    version : str
        Version printed by pw.x (``PwStatus.version``).

    Returns
    -------
    str or None
        The message, None for a ``7.1.x``, ``7.2.x`` or ``7.3.x``.
    """
    if _release(version) in SUPPORTED_VERSIONS:
        return None
    return (f'pw.x {version}: the workflow is built to work with QE '
            f'{", ".join(v + ".x" for v in SUPPORTED_VERSIONS)}; with version {version} the '
            'results are not guaranteed and may hold errors: check them')


def _release(version):
    """``'<major>.<minor>'`` of a version string, or None if it has none."""
    parts = version.split('.')
    if len(parts) < 2 or not parts[0].isdigit():
        return None
    minor = ''
    for char in parts[1]:
        if not char.isdigit():
            break
        minor += char
    return f'{parts[0]}.{minor}' if minor else None


def _starts(line, marker):
    return line.lstrip().startswith(marker)


def _last_index(lines, marker, start=0):
    """Index of the last line beginning with ``marker``, or None."""
    for i in range(len(lines) - 1, start - 1, -1):
        if _starts(lines[i], marker):
            return i
    return None


def _has(lines, marker):
    return any(_starts(line, marker) for line in lines)


def _last_run(text):
    """Lines of the last run, its version and the number of runs."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if _starts(line, _RUN_START)]
    if not starts:
        raise PwOutputError(f'no "{_RUN_START}" line: not a pw.x output')
    run = lines[starts[-1]:]
    version = run[0].lstrip()[len(_RUN_START):].split(' starts')[0].strip()
    if not version:
        raise PwOutputError(f'no version after "{_RUN_START}" in the header of the last run')
    return run, version, len(starts)


def _check_calculation(run, calculation):
    """Check the kind of calculation against the summary of the run.

    pw.x prints ``nstep`` in its summary only for an ionic relaxation. The
    check is made once the summary is complete (the ``crystal axes:`` line
    follows it); a run truncated before that is not checked.
    """
    end = _last_index(run, _SUMMARY_END)
    if end is None:
        return
    has_nstep = any(line.split('=')[0].strip() == _NSTEP for line in run[:end])
    if has_nstep != (calculation == 'relax'):
        printed = 'prints' if has_nstep else 'does not print'
        raise PwOutputError(f'calculation {calculation!r}, but the summary of the run '
                            f'{printed} nstep')


def _summary_value(run, name, kind):
    """Value printed as ``<name> = <value>`` in the summary of the run, or None.

    The summary ends at the ``crystal axes:`` line; a run truncated before it
    is read up to its end.

    Raises
    ------
    PwOutputError
        If the value is not a number of the type ``kind``.
    """
    end = _last_index(run, _SUMMARY_END)
    for line in run[:end] if end is not None else run:
        key, sep, value = line.partition('=')
        if sep and key.strip() == name:
            try:
                return kind(value.split()[0])
            except (IndexError, ValueError):
                raise PwOutputError(f'"{line.strip()}" in the summary of the run: not a '
                                    f'{kind.__name__}') from None
    return None


def _process(run):
    if _has(run, _JOB_DONE):
        return 'ended'
    if _has(run, _ERROR):
        return 'crashed'
    return None


def _error(run):
    """The last error of the run (``PwError``), or None."""
    start = _last_index(run, _ERROR)
    if start is None:
        return None
    head = run[start].strip()[len(_ERROR):].strip()
    match = _ERROR_HEAD.match(head)
    routine, code = (match.group(1), match.group(2)) if match else (head.rstrip(':'), None)
    message = []
    for line in run[start + 1:]:
        if line.strip().startswith('%') or _starts(line, _STOPPING):
            break
        if line.strip():
            message.append(line.strip())
    return PwError(routine, code, ' '.join(message))


def _scf(run):
    start = _last_index(run, _SCF_START)
    if start is None:
        return None
    for line in run[start:]:
        if _starts(line, _SCF_CONVERGED):
            return 'converged'
        if _starts(line, _SCF_NOT_CONVERGED):
            return 'not_converged'
    return None


def _geometry(run, process):
    if _has(run, _BFGS_CONVERGED):
        return 'converged'
    if _has(run, _MAX_STEPS):
        return 'max_steps'
    return 'not_converged' if process is not None else None


def _final_coordinates(run):
    begin = _last_index(run, _COORDINATES_BEGIN)
    if begin is None:
        return 'absent'
    return 'complete' if _has(run[begin:], _COORDINATES_END) else 'incomplete'


def _stop_reason(run, calculation, process, scf, geometry):
    if process is None:
        return 'truncated'
    if process == 'crashed':
        return 'error'
    if _has(run, _USER_REQUEST):
        return 'user_request'
    if _has(run, _MAX_SECONDS):
        return 'max_seconds'
    if scf == 'not_converged':
        return 'scf_not_converged'
    if calculation == 'scf':
        return 'completed' if scf == 'converged' else None
    if _has(run, _BFGS_FAILED):
        return 'optimizer_failed'
    if geometry == 'converged':
        return 'completed'
    if geometry == 'max_steps':
        return 'max_steps'
    return None


def pw_status(text, calculation):
    """States of the last run of a pw.x output.

    Parameters
    ----------
    text : str
        Content of the output.
    calculation : str
        ``'scf'`` or ``'relax'``: the kind of calculation of the input. It
        is not read from the output, which does not print it.

    Returns
    -------
    PwStatus
        States of the last run.

    Raises
    ------
    ValueError
        If ``calculation`` is not ``'scf'`` or ``'relax'``.
    PwOutputError
        If the text has no pw.x header, the header of the last run has no
        version, or the summary of the run disagrees with ``calculation``.
        A version outside ``SUPPORTED_VERSIONS`` is read
        (``version_warning``).
    """
    if calculation not in CALCULATIONS:
        raise ValueError(f'calculation: {calculation!r} is not one of {", ".join(CALCULATIONS)}')
    run, version, runs = _last_run(text)
    _check_calculation(run, calculation)
    process = _process(run)
    scf = _scf(run)
    if calculation == 'scf':
        geometry, coordinates = 'n/a', 'n/a'
    else:
        geometry, coordinates = _geometry(run, process), _final_coordinates(run)
    return PwStatus(calculation=calculation, version=version, runs=runs, process=process,
                    scf=scf, geometry=geometry, final_coordinates=coordinates,
                    stop_reason=_stop_reason(run, calculation, process, scf, geometry),
                    error=_error(run) if process == 'crashed' else None,
                    mixing_beta=_summary_value(run, _MIXING_BETA, float),
                    nstep=_summary_value(run, _NSTEP, int))


def read_pw_status(path, calculation):
    """States of the last run of a pw.x output file.

    The file is read as UTF-8; bytes that are not UTF-8 are replaced. The
    lines this module looks for are ASCII, so a replaced byte does not change
    a state.

    Parameters
    ----------
    path : path-like
        Output file of pw.x.
    calculation : str
        ``'scf'`` or ``'relax'``.

    Returns
    -------
    PwStatus
        States of the last run (see ``pw_status``).
    """
    try:
        return pw_status(Path(path).read_text(encoding='utf-8', errors='replace'), calculation)
    except PwOutputError as err:
        raise PwOutputError(f'{path}: {err}') from None


def record_status(pw, job=None):
    """States of the record of a calculation from its output and its job.

    The output decides whenever it can: once pw.x closed the run or stopped
    on an error, the job does not change the states (a job may go on after
    pw.x, or be killed later). For a truncated output only the process comes
    from the job:

    ==============  =============  =====================
    ``job``         ``process``    ``Status.summary``
    ==============  =============  =====================
    None            None           ``'unknown'``
    ``'running'``   ``'running'``  ``'running'``
    ``'killed'``    ``'killed'``   ``'crashed'``
    ``'finished'``  ``'crashed'``  ``'crashed'``
    ==============  =============  =====================

    ``'killed'`` is a job stopped by the scheduler (e.g. at the walltime);
    ``'finished'`` is a job the scheduler reports as over, without saying it
    was killed. The SCF and the geometry stay as read: what the output does
    not tell stays None. ``parsing`` is left None, for the parser to set.

    Parameters
    ----------
    pw : PwStatus
        States of the output.
    job : str or None, optional
        What the scheduler reports about the job: ``'running'``,
        ``'killed'``, ``'finished'``, or None when nothing is known.

    Returns
    -------
    triboworkflow.core.store.Status
        States for the record.

    Raises
    ------
    ValueError
        If ``job`` is not one of ``JOB_STATES`` or None.
    """
    if job is not None and job not in JOB_STATES:
        raise ValueError(f'job: {job!r} is not one of {", ".join(JOB_STATES)} or None')
    process = pw.process
    if process is None and job is not None:
        process = {'running': 'running', 'killed': 'killed', 'finished': 'crashed'}[job]
    return Status(process=process, scf=pw.scf, geometry=pw.geometry)
