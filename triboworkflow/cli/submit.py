# Author: Vito Foderà
# Created: 2026-10-30

"""Command ``submit``: the second phase of a step, on the HPC machine."""

from triboworkflow.core.settings.schema import INTERFACE_STEPS

from .command import CLICommandBase, positive_int


class CLICommand(CLICommandBase):
    """submit the calculations of a prepared step, and relaunch those that need it

The jobs of the run are asked to the scheduler, the calculations of the step
reconciled with their claims, records and jobs, and what has to run (a first
attempt, or a relaunch of a crashed calculation or of one stopped at
max_seconds) goes in a new group of jobs, which is submitted. Then the
command exits. Running it again follows the step: it may start relaunches,
and never submits the same attempt twice. When every calculation is
completed, run "analyse". The settings must not have changed since the
preparation.

Example:
 $ triboworkflow submit pes
 $ triboworkflow submit pes --groups 4
    """

    hpc = True
    locks_run = True

    def add_arguments(self, parser):
        parser.add_argument('step', choices=INTERFACE_STEPS, help='step to submit')
        parser.add_argument('--groups', type=positive_int, metavar='M',
                            help='jobs (workers) per group; [execution].groups by default')

    def run(self, args, run_dir, settings):
        from .run import (print_failed_submissions, print_step, resolve_groups,
                          step_orchestrator, submissions, system_name, terminal_ask)

        groups = resolve_groups(settings, args.groups, terminal_ask())
        orch = step_orchestrator(run_dir, settings, args.step, groups)
        before = submissions(orch.manager)
        report, written = orch.submit_step(system_name(settings), args.step)
        code = print_step(report)
        for name in written:
            print(f'group {name}: {len(orch.manager.group_jobs(name))} jobs submitted')
        if not written:
            print('no group written')
        if print_failed_submissions(orch.manager, before):
            code = 1
        if report.status == 'running':
            print(f'Run "triboworkflow submit {args.step}" again to follow the step: it '
                  'relaunches what needs it, never twice the same attempt')
        return code
