# Author: Vito Foderà
# Created: 2026-10-30

"""Command ``clear-refusal``: let a group refused by the scheduler be submitted again."""

from .command import CLICommandBase


class CLICommand(CLICommandBase):
    """let a group whose submission the scheduler refused be submitted again

A refused submission (e.g. a wrong partition in the jobscript, or a walltime
above the limit of the queue) is not repeated by itself. Each group submits
its own copy of the jobscript, jobs/<group>/jobscript.sh: fix the cause
there, keeping its last line (the worker), and in the jobscript of the run
for the next groups; a walltime is fixed in the section of the step,
[execution.<step>]. Then this command lets the next "submit" or pass of the
orchestrator submit the jobs of the group still missing. When none of its jobs
was submitted, the group takes the walltime of its step in [execution.<step>]
as it is now.

Example:
 $ triboworkflow clear-refusal Cu111_Al111.pes.1
    """

    locks_run = True

    def add_arguments(self, parser):
        parser.add_argument('group', help='group of jobs')

    def run(self, args, run_dir, settings):
        from triboworkflow.core.settings.schema import DEFAULT_WALLTIME, WALLTIME_STEPS
        from triboworkflow.execution.jobs import group_walltime, set_group_walltime
        from triboworkflow.execution.layout import job_dir

        from .run import RunError, execution_of, job_manager

        manager = job_manager(run_dir, settings)
        if args.group in manager.groups and manager.groups[args.group]['refused'] is None:
            raise RunError(f'group {args.group} has no submission refused')
        # <system>.<step>.<k>: the system may hold dots, the step does not.
        parts = args.group.rsplit('.', 2)
        step = parts[1] if len(parts) == 3 else None
        path = job_dir(run_dir, args.group)
        if (step in WALLTIME_STEPS and args.group in manager.groups
                and manager.withdrawal_problem(args.group) is None):
            execution = execution_of(settings)
            if execution.walltime_of(step) != group_walltime(path):
                set_group_walltime(path, execution.walltime_of(step))
                value = execution.walltime_given(step) or f'{DEFAULT_WALLTIME} (default)'
                print(f'group {args.group}: walltime {value}, from [execution.{step}] now')
        manager.clear_refusal(args.group)
        print(f'group {args.group}: it is submitted again by the next "submit" or pass of the '
              'orchestrator')
