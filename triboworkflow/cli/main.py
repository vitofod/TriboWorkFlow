# Author: Vito Foderà
# Created: 2026-10-30

"""Entry point of the command line ``triboworkflow``.

A command goes through these steps, each one before the next:

1. its arguments are parsed; an error of use exits with code 2;
2. the run is opened, read only: ``<run>/config.toml`` read and validated;
3. a command that submits looks for the submission command of the scheduler
   (``core.guard``): away from the HPC machine it stops here;
4. a command that changes the run takes the lock of the run
   (``execution.lock``): a run held by another command, or by the
   orchestrator in the background, stops it here;
5. only then ``<run>/cli.log`` is opened, for the commands that change the
   run, and the command runs.

So a command refused at steps 1 to 4 writes nothing in the run. Messages go
to the standard error; the results of ``check`` and ``status`` to the
standard output. An error prints ``Error: <message>`` and exits with code 1;
with ``-T`` its traceback is shown instead.

Notes
-----
From Xsorb (``xsorb/cli/xsorb_main.py`` and ``xsorb/cli/logger.py``,
CC BY 4.0): the parser, the error printed unless ``-T`` is given, the log
on the console (the message only) and in a file (time, level, message).
Unlike Xsorb: an error exits with code 1 and not 0; without a command the
help is printed and the code is 2; the log file is ``<run>/cli.log`` and is
written only by the commands that change the run, after the guard and the
lock.
"""

import logging
import shlex
import sys
from pathlib import Path

from triboworkflow.core.guard import require_scheduler
from triboworkflow.execution.lock import RunLock

from .parser import build_parser
from .run import execution_of, load_run, run_log

logger = logging.getLogger('triboworkflow')


class _Console(logging.Filter):
    """Leaves out of the console the records meant for the log file only."""

    def filter(self, record):
        return not getattr(record, 'file_only', False)


def main(argv=None):
    """Run a command of the command line.

    Parameters
    ----------
    argv : list of str or None, optional
        Arguments after the program name; ``sys.argv[1:]`` by default.

    Returns
    -------
    int
        Exit code: 0 done, 1 error, 2 error of use.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return 2
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter('%(message)s'))
    console.addFilter(_Console())
    level, propagate = logger.level, logger.propagate
    logger.addHandler(console)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        return _execute(args, argv)
    except Exception as error:
        if args.traceback:
            raise
        logger.error('Error: %s', error)
        return 1
    finally:
        logger.removeHandler(console)
        logger.setLevel(level)
        logger.propagate = propagate


def _execute(args, argv):
    command = args.cli_command
    args.argv = argv
    run_dir = Path(args.run)
    settings = load_run(run_dir) if command.loads_run else None
    if command.is_hpc(args):
        require_scheduler(execution_of(settings).scheduler)
    if not command.locks_run:
        return command.run(args, run_dir, settings) or 0
    with RunLock(run_dir, f'triboworkflow {shlex.join(argv)}'), run_log(run_dir, argv):
        try:
            return command.run(args, run_dir, settings) or 0
        except Exception as error:  # logged here, to be in the log of the run too
            if args.traceback:
                raise
            logger.error('Error: %s', error)
            return 1


if __name__ == '__main__':
    sys.exit(main())
