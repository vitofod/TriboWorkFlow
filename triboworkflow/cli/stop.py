# Author: Vito Foderà
# Created: 2026-10-30

"""Command ``stop``: stop the orchestrator in the background."""

from .command import CLICommandBase


def _seconds(value):
    seconds = float(value)
    if seconds <= 0:
        raise ValueError(value)
    return seconds


class CLICommand(CLICommandBase):
    """stop the orchestrator in the background at the end of its pass; the jobs go on

The orchestrator ends the pass under way, saves its state and exits. The
jobs already submitted keep running on the HPC machine: a later "start"
finds them again, without submitting anything twice. Run it on the machine
where the orchestrator runs.

Example:
 $ triboworkflow stop
 $ triboworkflow stop --wait 60
    """

    loads_run = False

    def add_arguments(self, parser):
        parser.add_argument('--wait', type=_seconds, metavar='S',
                            help='seconds to wait for the orchestrator to exit')

    def run(self, args, run_dir, settings):
        from triboworkflow.execution import daemon

        exited = daemon.stop(run_dir, timeout=args.wait)
        if exited:
            print('Orchestrator stopped. The jobs already submitted go on; "triboworkflow '
                  'start" follows them again.')
        elif args.wait is None:
            print('Stop asked: the orchestrator ends its pass, saves its state and exits. The '
                  'jobs already submitted go on.')
        else:
            print(f'Stop asked, the orchestrator has not exited within {args.wait} s: see '
                  '"triboworkflow status". The jobs already submitted go on.')
            return 1
        return 0
