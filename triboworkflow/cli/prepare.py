# Author: Vito Foderà
# Created: 2026-10-30

"""Command ``prepare``: the first phase of a step, without the scheduler."""

from triboworkflow.core.settings.schema import INTERFACE_STEPS

from .command import CLICommandBase


class CLICommand(CLICommandBase):
    """prepare the calculations of a step: directories, inputs and records

The entries the step reads must be done and current: a missing or stale one
blocks the step, with what is missing, and nothing is computed. A step whose
entries are already done and current is completed without running. A step
of analysis only (match, plotpes, shear) has nothing to prepare: run
"analyse" for it. Nothing is submitted: it works on any machine.

Example:
 $ triboworkflow prepare pes
    """

    locks_run = True

    def add_arguments(self, parser):
        parser.add_argument('step', choices=INTERFACE_STEPS, help='step to prepare')

    def run(self, args, run_dir, settings):
        from .run import print_step, step_orchestrator, system_name

        orch = step_orchestrator(run_dir, settings, args.step)
        system = system_name(settings)
        report = orch.prepare_step(system, args.step)
        code = print_step(report)
        calculations = orch.entry(system, args.step)['calculations']
        if report.status == 'running' and calculations is not None:
            print(f'{len(calculations)} calculations prepared')
        return code
