# Author: Vito Foderà
# Created: 2026-10-27

"""Guard against running calculations away from the HPC machine.

Calculations run only on an HPC machine with the scheduler of
``[execution]``. Inputs are prepared and outputs analysed anywhere, but a
submission needs the commands of that scheduler. ``hpc_only`` turns the
error of a missing scheduler command into ``NotOnHPCError``, which says so;
every other error passes unchanged. ``require_scheduler`` looks for the
submission command before work that submits later, away from the terminal
(the orchestrator in the background). Nothing else is checked: no machine
names, no environment variables. A worker started by hand runs where it is
started.
"""

import contextlib
import shutil

from triboworkflow.execution.scheduler import SCHEDULER_CONFIG, JobSchedulerError


class NotOnHPCError(RuntimeError):
    """A submission where the commands of the configured scheduler are missing."""


@contextlib.contextmanager
def hpc_only(scheduler_name):
    """Context of the scheduler commands that submit calculations.

    Parameters
    ----------
    scheduler_name : str
        Name of the configured scheduler, written in the message.

    Raises
    ------
    NotOnHPCError
        If a scheduler command in the block is not found; the original
        error is kept as ``__cause__``.

    Examples
    --------
    >>> with hpc_only('slurm'):                         # doctest: +SKIP
    ...     job_id = scheduler.submit_job('job.sh')
    """
    try:
        yield
    except JobSchedulerError as error:
        if isinstance(error.__cause__, FileNotFoundError):
            raise NotOnHPCError(f'{error}: {_ADVICE.format(scheduler_name)}') from error
        raise


_ADVICE = ('calculations run only on an HPC machine with the scheduler {!r} of [execution]; '
           'inputs can be prepared and outputs analysed here')


def require_scheduler(scheduler_name, which=shutil.which):
    """Check that the submission command of a scheduler is found.

    Parameters
    ----------
    scheduler_name : str
        A scheduler of ``SCHEDULER_CONFIG``.
    which : callable, optional
        Lookup of a command in the ``PATH``; ``shutil.which`` by default.

    Raises
    ------
    NotOnHPCError
        If the command is not found.
    """
    command = SCHEDULER_CONFIG[scheduler_name.lower()]['submit'][0]
    if which(command) is None:
        raise NotOnHPCError(f"Command '{command}' not found: {_ADVICE.format(scheduler_name)}")
