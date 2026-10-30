# Author: Vito Foderà
# Created: 2026-10-30

"""Base of the commands of the command line, and helpers for their arguments.

Each command is a module with a class ``CLICommand``: its docstring is the
help of the command (the first line in the list of the commands),
``add_arguments`` adds its arguments, ``run`` does its work and
``bind_function`` ties it to its parser. Three class attributes tell
``main`` what to do before ``run``:

- ``loads_run``: open the run, i.e. read and validate ``<run>/config.toml``
  (``check`` reads the settings itself, ``stop`` needs only the pid file);
- ``hpc``: look for the submission command of the scheduler of
  ``[execution]`` (``core.guard``): away from the HPC machine the command
  stops there, before writing anything;
- ``locks_run``: hold the lock of the run (``execution.lock``) for the whole
  command, and log also in ``<run>/cli.log``.

Notes
-----
From Xsorb (``xsorb/cli/command.py``, CC BY 4.0): one class per command
with ``add_arguments``, ``run`` and ``bind_function``, the check of the
positive integers and the formatter of the help. Unlike Xsorb, ``run``
receives the directory and the settings of the run, opened by ``main``, and
returns the exit code; the attributes above are new.
"""

import argparse
from abc import ABC, abstractmethod


class CLICommandBase(ABC):
    """Base class of the commands of the command line."""

    loads_run = True
    hpc = False
    locks_run = False

    @abstractmethod
    def add_arguments(self, parser):
        """Add the arguments of the command to its parser."""

    @abstractmethod
    def run(self, args, run_dir, settings):
        """Do the work of the command.

        Parameters
        ----------
        args : argparse.Namespace
        run_dir : pathlib.Path
            Directory of the run (``-r/--run``).
        settings : Settings or None
            Settings of the run; None when ``loads_run`` is False.

        Returns
        -------
        int or None
            Exit code; None means 0.
        """

    def bind_function(self, parser):
        """Tie the command to its parser."""
        parser.set_defaults(cli_command=self)

    def is_hpc(self, args):
        """Whether the command, with these arguments, needs the scheduler (``hpc``)."""
        return self.hpc


def positive_int(value):
    """Argument that must be an integer > 0."""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f'{value} is not positive')
    return ivalue


def add_run_argument(parser):
    """``-r/--run DIR``: the folder of the run, the current one by default."""
    parser.add_argument('-r', '--run', default='.', metavar='DIR',
                        help='folder of the run, with its config.toml')


class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                      argparse.ArgumentDefaultsHelpFormatter):
    """Help with the docstring as written and the default of each argument."""
