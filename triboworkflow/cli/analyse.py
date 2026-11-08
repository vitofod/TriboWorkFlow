# Author: Vito Foderà
# Created: 2026-10-30

"""Command ``analyse``: the third phase of a step, from the files only."""

from triboworkflow.core.settings.schema import INTERFACE_STEPS

from .command import CLICommandBase


class CLICommand(CLICommandBase):
    """analyse a step and write its entries in system.json

Only the files of the run are read, never the scheduler: every calculation
of the step must be done according to its claim, its record and the groups
of jobs, with the fingerprint the settings give now; otherwise nothing is
written and the error says, for each calculation, what is missing. A step of
analysis only (match, plotpes, plotpes_scf, shear) is done here whole. It
works on any machine.

plotpes and plotpes_scf are never analysed by the orchestrator in the
background: the interpolation of a large PES needs much memory. With
[execution.analysis] the orchestrator, or "submit", analyses them in a job of
their own on a compute node; without it the orchestrator leaves them manual
once ready, and this command analyses them, where there is enough memory,
with the orchestrator stopped or ended (it holds the run). It is refused
while the job of the step may still run, according to the last query of the
scheduler. The step is saved as running while it is analysed; an analysis
stopped before its end writes nothing, and the next one starts it again from
the beginning.

Example:
 $ triboworkflow analyse match
 $ triboworkflow analyse pes
 $ triboworkflow analyse plotpes
    """

    locks_run = True

    def add_arguments(self, parser):
        parser.add_argument('step', choices=INTERFACE_STEPS, help='step to analyse')

    def run(self, args, run_dir, settings):
        from .run import print_step, step_orchestrator, system_name

        orch = step_orchestrator(run_dir, settings, args.step)
        return print_step(orch.analyse_step(system_name(settings), args.step))
