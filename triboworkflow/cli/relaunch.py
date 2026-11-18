# Author: Vito Foderà
# Created: 2026-10-30

"""Command ``relaunch``: relaunch the calculations of a step by hand."""

from .command import CLICommandBase, positive_int


class CLICommand(CLICommandBase):
    """relaunch by hand the calculations of a step, at the level of their last attempt or another

--step names the step (pes, ppes, …): all its calculations that can be
relaunched are, in one new group of jobs that is submitted, and the others are
listed with the reason. --calc, with the labels of one or more calculations of
the step, relaunches those only (the labels are in "status --calcs", e.g.
min_site_1.5 of ppes); all of them must be relaunchable, or nothing is
written.

Allowed only once the worker of the last attempt of a calculation is surely
over: no claim, or the run of the job of its claim is over; otherwise the
state found is printed. The new attempt is a manual one, with the input and
the level of the last attempt, or at the level given with --level (1 is the
input as prepared; 2 and 3 lower mixing_beta and, for a relaxation, raise
nstep); its step is followed again. It continues the last attempt from the
files that pw.x left in the output directory when the workflow would (a run
closed by pw.x at max_seconds with the same run command, at the step limit, or
without SCF convergence in a relaxation, with the files on disk); --from-scratch
starts it from scratch anyway. A completed calculation is not run again. When
the launch of the last attempt of a calculation is not recorded and the groups
that list it give different launches, choose one with --from-group (for a
single calculation).

Example:
 $ triboworkflow relaunch --step ppes
 $ triboworkflow relaunch --step adhesion --calc interface_max --level 2 --from-scratch
 $ triboworkflow relaunch --step ppes --calc min_site_1.5 min_site_2.0
 $ triboworkflow relaunch --step pes --calc c0 --from-group mp-30_mp-134.pes.2
    """

    hpc = True
    locks_run = True

    def add_arguments(self, parser):
        parser.add_argument('--step', required=True, metavar='STEP',
                            help='step whose calculations are relaunched, e.g. ppes')
        parser.add_argument('--calc', nargs='+', metavar='LABEL',
                            help='labels of the calculations of the step to relaunch, as in '
                                 '"status --calcs"; by default all that can be relaunched')
        parser.add_argument('--groups', type=positive_int, metavar='M',
                            help='jobs (workers) of the new group; by default those of '
                                 '[execution.<step>], else of [execution]')
        parser.add_argument('--from-group', metavar='G',
                            help='group whose launch of the last attempt is taken, when it is '
                                 'not recorded (one calculation, given with --calc)')
        parser.add_argument('--level', type=positive_int, metavar='L',
                            help='level of the remedies of the new attempts, from 1 to 3; that '
                                 'of the last attempt of each by default')
        parser.add_argument('--from-scratch', action='store_true',
                            help='start from scratch even when the last attempt could be '
                                 'continued from its output directory')

    def run(self, args, run_dir, settings):
        from triboworkflow.execution.jobs import MANIFEST_FILE, read_manifest
        from triboworkflow.execution.layout import job_dir

        from .run import (print_failed_submissions, resolve_groups, run_orchestrator,
                          submissions, system_name, terminal_ask)

        groups, override = resolve_groups(settings, args.groups, terminal_ask(),
                                          steps=[args.step])
        orch = run_orchestrator(run_dir, settings, groups, override)
        before = submissions(orch.manager)
        system = system_name(settings)
        group, _, left = orch.relaunch_step(system, args.step, args.calc, args.from_group,
                                            args.level, args.from_scratch)
        manifest, _ = read_manifest(job_dir(run_dir, group) / MANIFEST_FILE)
        for launch in manifest.calculations:
            start = 'restart from the output directory' if launch.restart else 'from scratch'
            print(f'{system}/{launch.calc_id}: relaunched by hand, attempt {launch.n} at level '
                  f'{launch.level}, {start}')
        for calc_id, reason in left.items():
            print(f'{system}/{calc_id}: not relaunched: {reason}')
        print(f'group {group}: {len(orch.manager.group_jobs(group))} jobs submitted')
        return 1 if print_failed_submissions(orch.manager, before) else 0
