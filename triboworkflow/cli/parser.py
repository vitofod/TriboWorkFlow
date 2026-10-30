# Author: Vito Foderà
# Created: 2026-10-30

"""Parser of the command line: the list of the commands and their modules.

Notes
-----
From Xsorb (``xsorb/cli/xsorb_parser.py``, CC BY 4.0): the main parser with
``-v/--version`` and ``-T/--traceback``, and one subparser per command, built
from the ``CLICommand`` of its module. Every command also takes
``-r/--run``.
"""

import argparse
from importlib import import_module

import triboworkflow

from .command import CustomFormatter, add_run_argument

# Name and module of each command, in the order of the help.
COMMANDS = [
    ('check', 'triboworkflow.cli.check'),
    ('status', 'triboworkflow.cli.status'),
    ('prepare', 'triboworkflow.cli.prepare'),
    ('submit', 'triboworkflow.cli.submit'),
    ('analyse', 'triboworkflow.cli.analyse'),
    ('start', 'triboworkflow.cli.start'),
    ('stop', 'triboworkflow.cli.stop'),
    ('relaunch', 'triboworkflow.cli.relaunch'),
    ('retry', 'triboworkflow.cli.retry'),
    ('resolve-submission', 'triboworkflow.cli.resolve'),
    ('clear-refusal', 'triboworkflow.cli.refusal'),
    ('release-claim', 'triboworkflow.cli.release'),
]


def build_parser():
    """Parser of the command line ``triboworkflow``.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(prog='triboworkflow', description=triboworkflow.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument('-v', '--version', action='version',
                        version=f'%(prog)s-{triboworkflow.__version__}')
    parser.add_argument('-T', '--traceback', action='store_true',
                        help='print the traceback of an error')
    subparsers = parser.add_subparsers(title='commands', dest='command', metavar='<command>')
    for name, module in COMMANDS:
        command = import_module(module).CLICommand()
        doc = command.__doc__ or ''
        subparser = subparsers.add_parser(name, formatter_class=CustomFormatter,
                                          help=doc.split('\n')[0], description=doc,
                                          allow_abbrev=False)
        add_run_argument(subparser)
        command.add_arguments(subparser)
        command.bind_function(subparser)
    return parser
