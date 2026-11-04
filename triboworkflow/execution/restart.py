# Author: Vito Foderà
# Created: 2026-11-03

"""What an attempt that continues the previous one read from its files.

An attempt launched to continue the previous one (``Launch.restart``) runs
pw.x with ``restart_mode = 'restart'``: pw.x reads the files that the previous
attempt left in the output directory. It reads each part on its own and,
when one cannot be read, computes it again from the input with a message,
without an error; its output tells what was read (``PwRestart``). The result
of such an attempt stands as any other, the thresholds of convergence being
the same and only the starting point different, but for one case: after a
stop that leaves files written per process (``PER_PROCESS_STOPS``), it is
not reused unless the attempt ran with the parallelization of the previous
one (``core.store.restart_problem``), and the user decides.

``restart_outcome`` turns what the output tells into ``Attempt.restart``,
one value per part of ``RESTART_PARTS``, and ``restart_warnings`` gives the
warnings logged with it:

- the positions, the charge density or the wavefunctions not read: they are
  always there after a stop from which a restart is decided
  (``policy.can_restart``), so the attempt started again further back than
  expected;
- a parallelization other than that of the previous attempt: pw.x writes some
  files once per MPI process, and reads them correctly only with the same
  parallelization, which it does not check (the warning comes with any
  restart; whether the result stands is decided as above);
- the iteration of the SCF and the history of the optimizer are recorded
  without a warning: they are not continued, rightly, when the previous
  attempt did not stop inside the SCF, or stopped before the first step of
  the optimizer.
"""

from triboworkflow.core.store.record import RESTART_PARTS
from triboworkflow.engines.qe.status import PwOutputError, read_parallelization

# Parts that a restart always finds once it is decided: one not read gives a
# warning.
WARNED_PARTS = ('positions', 'density', 'wavefunctions')


def previous_parallelization(calc_path, record):
    """Parallelization of the attempt before the last one, from its archived output.

    Parameters
    ----------
    calc_path : pathlib.Path
        Directory of the calculation.
    record : triboworkflow.core.store.Record
        Record whose last attempt continues the one before.

    Returns
    -------
    tuple of str or None
        ``PwStatus.parallelization`` of the previous attempt; None when it
        has no output that can be read, or prints no parallelization.
    """
    if len(record.attempts) < 2 or record.attempts[-2].output is None:
        return None
    try:
        lines = read_parallelization(calc_path / record.attempts[-2].output.path)
    except (OSError, PwOutputError):
        return None
    return lines or None


def restart_outcome(pw, previous):
    """``Attempt.restart`` of an attempt that continued the previous one.

    Parameters
    ----------
    pw : triboworkflow.engines.qe.status.PwStatus
        States of the output of the attempt.
    previous : tuple of str or None
        Parallelization of the previous attempt (``previous_parallelization``).

    Returns
    -------
    dict
        One value per part of ``RESTART_PARTS``: what was read from the
        files (true, false, or null when the output does not tell) and
        whether the parallelization is the same (null when either is not
        known).
    """
    read = pw.restart
    same = None if previous is None or not pw.parallelization else previous == pw.parallelization
    values = {'same_parallelization': same}
    return {part: values[part] if part in values else getattr(read, part)
            for part in RESTART_PARTS}


def restart_warnings(outcome):
    """Warnings of an ``Attempt.restart`` (see the module docstring).

    Returns
    -------
    list of str
    """
    missing = [part for part in WARNED_PARTS if outcome[part] is False]
    notes = []
    if missing:
        notes.append(f'restart asked for, {", ".join(missing)} not read from the files of the '
                     'previous attempt: computed again from the input')
    if outcome['same_parallelization'] is False:
        notes.append('restart with a parallelization other than that of the previous attempt: '
                     'the files written per process may have been read incorrectly')
    return notes
