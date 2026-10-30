# Author: Vito Foderà
# Created: 2026-10-30

"""Command ``release-claim``: release the empty claim of a calculation."""

from .command import CLICommandBase


class CLICommand(CLICommandBase):
    """release the empty claim of a calculation, once its worker is surely over

A worker that stops right after taking a calculation leaves an empty claim,
and whether it still runs cannot be told from the files: the calculation
stays uncertain. Once you have checked that its job no longer runs, this
command records the attempt with the launch of the group that listed it and
an unknown end, and removes the claim; then relaunch the calculation. When
the groups that list the attempt give different launches, choose one with
--group. No command of the scheduler runs. A release interrupted ("empty
claim released in part" in status) is completed by the same command again.

Example:
 $ triboworkflow release-claim pes/c0 --worker-over
 $ triboworkflow release-claim pes/c0 --worker-over --group mp-30_mp-134.pes.2
    """

    locks_run = True

    def add_arguments(self, parser):
        parser.add_argument('calc_id', help='calculation, as <step>/<label>')
        parser.add_argument('--worker-over', action='store_true', required=True,
                            help='confirm that the worker of the claim no longer runs')
        parser.add_argument('--group', metavar='G',
                            help='group whose launch of the attempt is taken')

    def run(self, args, run_dir, settings):
        from .run import run_orchestrator, system_name

        system = system_name(settings)
        n = run_orchestrator(run_dir, settings).release_claim(system, args.calc_id, args.group)
        print(f'{system}/{args.calc_id}: empty claim released as attempt {n}, with its end not '
              f'known; relaunch it with "triboworkflow relaunch {args.calc_id}"')
