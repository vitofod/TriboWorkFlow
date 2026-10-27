# Author: Vito Foderà
# Created: 2026-10-27

"""Guard against running calculations away from the HPC machine.

Calculations run only on an HPC machine with the scheduler of
``[execution]``. Inputs are prepared and outputs analysed anywhere, but a
submission needs the commands of that scheduler. ``hpc_only`` turns the
error of a missing scheduler command into ``NotOnHPCError``, which says so;
every other error passes unchanged. Nothing else is checked: no machine
names, no environment variables. A worker started by hand runs where it is
started.
"""

import contextlib

from triboworkflow.execution.scheduler import JobSchedulerError


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
            raise NotOnHPCError(
                f'{error}: calculations run only on an HPC machine with the scheduler '
                f'{scheduler_name!r} of [execution]; inputs can be prepared and '
                f'outputs analysed here') from error
        raise
