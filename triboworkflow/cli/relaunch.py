# Author: Vito Foderà
# Created: 2026-10-30

"""Command ``relaunch``: relaunch one calculation by hand."""

from .command import CLICommandBase, positive_int


class CLICommand(CLICommandBase):
    """relaunch a calculation by hand, at the level of its last attempt

Allowed only once the worker of the last attempt is surely over: no claim,
or the run of the job of its claim is over; otherwise the state found is
printed. The new attempt is a manual one, with the input and the level of
the last attempt, in a new group of jobs that is submitted; its step is
followed again. When the launch of the last attempt is not recorded and the
groups that list it give different launches, choose one with --group. A
different level comes with a later version.

Example:
 $ triboworkflow relaunch adhesion/interface_max
 $ triboworkflow relaunch pes/c0 --group mp-30_mp-134.pes.2
    """

    hpc = True
    locks_run = True

    def add_arguments(self, parser):
        parser.add_argument('calc_id', help='calculation, as <step>/<label>')
        parser.add_argument('--groups', type=positive_int, metavar='M',
                            help='jobs (workers) per group; [execution].groups by default')
        parser.add_argument('--group', metavar='G',
                            help='group whose launch of the last attempt is taken, when it is '
                                 'not recorded')

    def run(self, args, run_dir, settings):
        from .run import (print_failed_submissions, resolve_groups, run_orchestrator,
                          submissions, system_name, terminal_ask)

        groups = resolve_groups(settings, args.groups, terminal_ask())
        orch = run_orchestrator(run_dir, settings, groups)
        before = submissions(orch.manager)
        group = orch.relaunch(system_name(settings), args.calc_id, args.group)
        print(f'{system_name(settings)}/{args.calc_id}: relaunched by hand')
        print(f'group {group}: {len(orch.manager.group_jobs(group))} jobs submitted')
        return 1 if print_failed_submissions(orch.manager, before) else 0
