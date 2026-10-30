# Author: Vito Foderà
# Created: 2026-10-30

"""Command ``clear-refusal``: let a group refused by the scheduler be submitted again."""

from .command import CLICommandBase


class CLICommand(CLICommandBase):
    """let a group whose submission the scheduler refused be submitted again

A refused submission (e.g. a wrong partition in the jobscript) is not
repeated by itself. Each group submits its own copy of the jobscript,
jobs/<group>/jobscript.sh: fix the cause there, keeping its last line (the
worker), and in the jobscript of the run for the next groups. Then this
command lets the next "submit" or pass of the orchestrator submit the jobs
of the group still missing.

Example:
 $ triboworkflow clear-refusal mp-30_mp-134.pes.1
    """

    locks_run = True

    def add_arguments(self, parser):
        parser.add_argument('group', help='group of jobs')

    def run(self, args, run_dir, settings):
        from .run import RunError, job_manager

        manager = job_manager(run_dir, settings)
        if args.group in manager.groups and manager.groups[args.group]['refused'] is None:
            raise RunError(f'group {args.group} has no submission refused')
        manager.clear_refusal(args.group)
        print(f'group {args.group}: it is submitted again by the next "submit" or pass of the '
              'orchestrator')
