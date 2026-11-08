# Author: Vito Foderà
# Created: 2026-10-30

"""Command ``retry``: let a step failed for an error of its own start again."""

from triboworkflow.core.settings.schema import INTERFACE_STEPS

from .command import CLICommandBase


class CLICommand(CLICommandBase):
    """let a step failed for an error of its own start again

A step fails for an error of its own while it is prepared or analysed, when
the settings changed after its preparation, or when the job of its analysis
(plotpes, plotpes_scf with [execution.analysis]) ended without completing
it; it stays failed until this command, once the cause is fixed (e.g. more
memory or time in the jobscript of [execution.analysis]). It then starts
again at the next pass of the orchestrator, or with "prepare" ("submit" for
a new job of an analysis). A step failed because its calculations need the
user is not retried: relaunch them.

Example:
 $ triboworkflow retry ppes
 $ triboworkflow retry plotpes
    """

    locks_run = True

    def add_arguments(self, parser):
        parser.add_argument('step', choices=INTERFACE_STEPS, help='step to start again')

    def run(self, args, run_dir, settings):
        from .run import RunError, print_step, step_orchestrator, system_name

        system = system_name(settings)
        orch = step_orchestrator(run_dir, settings, args.step)
        entry = orch.entry(system, args.step)
        if entry['status'] != 'failed' or entry['error'] is None:
            reason = 'its calculations need the user: relaunch them' \
                if entry['status'] == 'failed' else 'retry is only for a failed step'
            raise RunError(f'{system}/{args.step} is {entry["status"]}: {reason}')
        orch.retry(system, args.step)
        print_step(orch.step_report(system, args.step))
