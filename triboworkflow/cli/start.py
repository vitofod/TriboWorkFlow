# Author: Vito Foderà
# Created: 2026-10-30

"""Command ``start``: the orchestrator of the steps in the background."""

import logging

from .command import CLICommandBase, positive_int

logger = logging.getLogger(__name__)

FACTORY = 'triboworkflow.cli.run:daemon_factory'


class CLICommand(CLICommandBase):
    """start the orchestrator of the steps of [interface].steps in the background

The orchestrator runs on this machine (the login node of the HPC machine),
detached from the terminal, and holds the run until it ends: every
polling_interval it asks the scheduler for the jobs, checks again the steps
completed, prepares, submits, follows and relaunches the calculations of the
steps that are ready, and analyses them. It ends when no step waits or runs.
Its log is orchestrator.log; "status" shows where it stands, "stop" stops it
at the end of its pass, while the jobs go on.

--dry-run makes a limited check of the first pass on a temporary copy of a
run that has submitted nothing yet: the first steps ready are prepared and
the commands of the scheduler are printed, not run. Nothing is left in the
run, and it works on any machine.

Example:
 $ triboworkflow start
 $ triboworkflow start --groups 4
 $ triboworkflow start --dry-run
    """

    hpc = True

    def add_arguments(self, parser):
        parser.add_argument('--groups', type=positive_int, metavar='M',
                            help='jobs (workers) per group; [execution].groups by default')
        parser.add_argument('--dry-run', action='store_true',
                            help='check the first pass on a copy of the run, submitting nothing')

    def is_hpc(self, args):
        return not args.dry_run

    def run(self, args, run_dir, settings):
        from triboworkflow.execution import daemon

        from .run import execution_of, resolve_groups, run_log, terminal_ask

        groups = resolve_groups(settings, args.groups, terminal_ask())
        if args.dry_run:
            return _dry_run(run_dir, groups)
        started = daemon.start(run_dir, FACTORY, execution_of(settings).scheduler,
                               factory_args=(str(groups),))
        print(f'Orchestrator: {started.message}')
        # Written only once the start is confirmed: a start refused leaves nothing.
        with run_log(run_dir, args.argv):
            logger.info('Orchestrator: %s', started.message, extra={'file_only': True})
        return 0


def _dry_run(run_dir, groups):
    """First pass on a temporary copy of the run, with the scheduler in a dry run."""
    import shlex
    import shutil
    import tempfile
    from pathlib import Path

    from triboworkflow.execution.layout import JOBS_DIR
    from triboworkflow.execution.lock import LOCK_FILE, RunLock

    from .run import CLI_LOG, RunError, build_orchestrator, print_step

    with tempfile.TemporaryDirectory(prefix='triboworkflow-dry-run-') as tmp:
        copy = Path(tmp) / Path(run_dir).resolve().name
        with RunLock(run_dir, 'triboworkflow start --dry-run'):
            if (Path(run_dir) / JOBS_DIR).exists():
                raise RunError(f'{Path(run_dir) / JOBS_DIR} exists: the dry run checks only a '
                               'run that has submitted nothing yet; see "status"')
            shutil.copytree(run_dir, copy, ignore=shutil.ignore_patterns(LOCK_FILE, CLI_LOG))
        orch = build_orchestrator(copy, groups=groups, dry_run=True)
        report = orch.run_pass()
        print('Dry run on a copy of the run, first pass only: nothing is submitted and nothing '
              'is left in the run; later passes, jobs and relaunches are not simulated')
        code = max((print_step(step) for step in report.steps), default=0)
        commands = orch.manager.scheduler.commands
        print(f'Commands of the scheduler of the first pass: {len(commands)}')
        for command, _, _ in commands:
            print('  ' + shlex.join(str(part).replace(str(copy), str(run_dir))
                                    for part in command))
        return code
