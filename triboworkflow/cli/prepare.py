# Author: Vito Foderà
# Created: 2026-10-30

"""Command ``prepare``: the first phase of a step, without the scheduler."""

import shlex

from triboworkflow.core.settings.schema import INTERFACE_STEPS

from .command import CLICommandBase


class CLICommand(CLICommandBase):
    """prepare the calculations of a step: directories, inputs and records

The entries the step reads must be done and current: a missing or stale one
blocks the step, with what is missing, and nothing is computed. A step whose
entries are already done and current is completed without running. A step
of analysis only (match, plotpes, plotpes_scf, shear) has nothing to
prepare: run "analyse" for it; plotpes, plotpes_scf and shear are left manual,
with that command. Nothing is submitted: it works on any machine.

A calculation already prepared is never overwritten. One that differs from
the settings stops the preparation, with the list and why; one that differs
only in keys of the attempts or of the resources (e.g. mixing_beta,
max_seconds) is kept and its values are logged. With --replace, every such
calculation moves with its attempts to replaced/ and is prepared again;
groups of jobs never submitted that list one are withdrawn. A calculation
that a worker or a job may still use stops the replacement, and nothing
changes.

Example:
 $ triboworkflow prepare pes
 $ triboworkflow prepare pes --replace
    """

    locks_run = True

    def add_arguments(self, parser):
        parser.add_argument('step', choices=INTERFACE_STEPS, help='step to prepare')
        parser.add_argument('--replace', action='store_true',
                            help='move the calculations that differ from the settings to '
                                 'replaced/ and prepare them again')

    def run(self, args, run_dir, settings):
        from .run import print_step, step_orchestrator, system_name

        orch = step_orchestrator(run_dir, settings, args.step)
        system = system_name(settings)
        if args.replace:
            argv = getattr(args, 'argv', None)
            command = f'triboworkflow {shlex.join(argv)}' if argv is not None else None
            report, replacement = orch.replace_step(system, args.step, command)
            if replacement is None:
                print('no calculation differs from the settings: nothing replaced')
            else:
                print(f'{len(replacement.calculations)} calculations replaced: the old ones are '
                      f'in {replacement.path}')
                for name in replacement.withdrawn:
                    print(f'group {name} withdrawn: none of its jobs was submitted')
        else:
            report = orch.prepare_step(system, args.step)
        code = print_step(report)
        calculations = orch.entry(system, args.step)['calculations']
        if report.status == 'running' and calculations is not None:
            print(f'{len(calculations)} calculations prepared')
        return code
