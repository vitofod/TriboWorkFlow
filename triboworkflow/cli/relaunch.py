# Author: Vito Foderà
# Created: 2026-10-30

"""Command ``relaunch``: relaunch one calculation by hand."""

from .command import CLICommandBase, positive_int


class CLICommand(CLICommandBase):
    """relaunch a calculation by hand, at the level of its last attempt or another

Allowed only once the worker of the last attempt is surely over: no claim,
or the run of the job of its claim is over; otherwise the state found is
printed. The new attempt is a manual one, with the input and the level of
the last attempt, or at the level given with --level (1 is the input as
prepared; 2 and 3 lower mixing_beta and, for a relaxation, raise nstep), in
a new group of jobs that is submitted; its step is followed again. It
continues the last attempt from the files that pw.x left in the output
directory when the workflow would (a run closed by pw.x at max_seconds with
the same run command, at the step limit, or without SCF convergence in a
relaxation, with the files on disk); --from-scratch starts it from scratch
anyway. When the launch of the last attempt is not recorded and the groups
that list it give different launches, choose one with --group.

Example:
 $ triboworkflow relaunch adhesion/interface_max
 $ triboworkflow relaunch adhesion/interface_max --level 2 --from-scratch
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
        parser.add_argument('--level', type=positive_int, metavar='L',
                            help='level of the remedies of the new attempt, from 1 to 3; that '
                                 'of the last attempt by default')
        parser.add_argument('--from-scratch', action='store_true',
                            help='start from scratch even when the last attempt could be '
                                 'continued from its output directory')

    def run(self, args, run_dir, settings):
        from triboworkflow.execution.jobs import MANIFEST_FILE, read_manifest
        from triboworkflow.execution.layout import job_dir

        from .run import (print_failed_submissions, resolve_groups, run_orchestrator,
                          submissions, system_name, terminal_ask)

        groups = resolve_groups(settings, args.groups, terminal_ask())
        orch = run_orchestrator(run_dir, settings, groups)
        before = submissions(orch.manager)
        group = orch.relaunch(system_name(settings), args.calc_id, args.group, args.level,
                              args.from_scratch)
        manifest, _ = read_manifest(job_dir(run_dir, group) / MANIFEST_FILE)
        (launch,) = manifest.calculations
        start = 'restart from the output directory' if launch.restart else 'from scratch'
        print(f'{system_name(settings)}/{args.calc_id}: relaunched by hand, attempt {launch.n} '
              f'at level {launch.level}, {start}')
        print(f'group {group}: {len(orch.manager.group_jobs(group))} jobs submitted')
        return 1 if print_failed_submissions(orch.manager, before) else 0
