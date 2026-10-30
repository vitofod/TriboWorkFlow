# Author: Vito Foderà
# Created: 2026-10-30

"""Command ``resolve-submission``: tell the outcome of a submission not known."""

from .command import CLICommandBase


class CLICommand(CLICommandBase):
    """tell the outcome of a submission whose answer was lost (e.g. sbatch timed out)

Such a submission may have created a job, and it is never repeated by
itself: look for the job in the queue of the scheduler (e.g. squeue -u
$USER), then give its id, or say that there is none, so that its place in
the group is submitted again. The oldest submission of the group is
resolved first.

Example:
 $ triboworkflow resolve-submission mp-30_mp-134.pes.1 --job-id 123456
 $ triboworkflow resolve-submission mp-30_mp-134.pes.1 --no-job
    """

    locks_run = True

    def add_arguments(self, parser):
        parser.add_argument('group', help='group of jobs')
        found = parser.add_mutually_exclusive_group(required=True)
        found.add_argument('--job-id', metavar='ID', help='id of the job that it created')
        found.add_argument('--no-job', action='store_true', help='it created no job')

    def run(self, args, run_dir, settings):
        from .run import job_manager

        job_manager(run_dir, settings).resolve_submission(args.group, args.job_id)
        if args.job_id is not None:
            print(f'group {args.group}: job {args.job_id} registered; it is followed as the '
                  'others')
        else:
            print(f'group {args.group}: no job; its place is submitted again by the next '
                  '"submit" or pass of the orchestrator')
