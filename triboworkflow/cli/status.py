# Author: Vito Foderà
# Created: 2026-10-30

"""Command ``status``: where a run stands, from its files only.

Notes
-----
As ``status`` of Xsorb (``xsorb/cli/status.py``, CC BY 4.0): the state of the
calculations in one command; here also the orchestrator in the background,
the lock of the run, the steps and the jobs.
"""

from .command import CLICommandBase

# Hints for the calculations and groups that need the user.
_EMPTY_CLAIM = ('check that its job no longer runs, then "triboworkflow release-claim {calc_id} '
                '--worker-over"')


class CLICommand(CLICommandBase):
    """show the orchestrator, the lock, the steps, the calculations and the jobs of a run

Only the files of the run are read: this is the state last saved, with the
time of its last update, and no command of the scheduler runs. The jobs are
asked to the scheduler by "submit", "start" and "relaunch" on the HPC
machine.

Example:
 $ triboworkflow status
 $ triboworkflow status --calcs
    """

    def add_arguments(self, parser):
        parser.add_argument('--calcs', action='store_true',
                            help='list every calculation of the steps with its state')

    def run(self, args, run_dir, settings):
        from triboworkflow.core.store import read_record
        from triboworkflow.execution import daemon
        from triboworkflow.execution.layout import system_dir
        from triboworkflow.execution.lock import holder
        from triboworkflow.execution.orchestrator import Step
        from triboworkflow.execution.replacement import unfinished

        from .run import build_orchestrator, run_steps, system_name

        system = system_name(settings)
        print(f'Run {run_dir.resolve()}, system {system}')
        print(f'Orchestrator: {daemon.status(run_dir).message}')
        print(f'Lock: {_lock(holder(run_dir))}')
        for replacement in unfinished(run_dir):
            print(f'Replacement of {replacement.system}/{replacement.step} not completed '
                  f'({replacement.path}): run "triboworkflow prepare {replacement.step} '
                  '--replace" to complete it')

        def named(name):
            step = Step()  # nothing is run: the state is only read
            step.name = name
            return step

        names = run_steps(run_dir, settings)
        orch = build_orchestrator(run_dir, dry_run=True, steps=names,
                                  step_objects={name: named(name) for name in names})
        report = orch.report()
        entries = {name: orch.entry(system, name) for name in orch.order}
        saved = max((e['updated'] for e in entries.values() if e['updated']), default='never')
        print(f'Steps (state saved {saved})')
        for step in report.steps:
            entry = entries[step.step]
            calcs = ', '.join(f'{n} {outcome}' for outcome, n in step.calcs.items())
            detail = '; '.join(filter(None, [step.reason, calcs and f'calculations: {calcs}',
                                             step.job and f'job of the analysis: jobs/{step.job}']))
            print(f'  {step.step:<11} {step.status:<10} {entry["updated"] or "-":<26} {detail}')
        attention = [(step.step, item) for step in report.steps for item in step.attention]
        if attention:
            print('Needs the user')
            for _, item in attention:
                hint = _EMPTY_CLAIM.format(calc_id=item['calc_id']) \
                    if (item['reason'] or '').startswith('empty claim') else ''
                print(f'  {item["calc_id"]:<24} {item["state"]:<11} {item["reason"]}'
                      + (f'; {hint}' if hint else ''))
        jobs, registry = report.jobs, orch.manager.jobs
        checked = max((job['checked'] for job in registry.values() if job['checked']),
                      default='never')
        print(f'Jobs (last query {checked}): {len(jobs.waiting)} waiting, {len(jobs.held)} held, '
              f'{len(jobs.attention)} to check, {jobs.to_submit} to submit')
        for job_id, held in jobs.held.items():
            print(f'  job {job_id} held ({held["state"]}): {held["message"]}')
        for name, error in jobs.refused.items():
            print(f'  group {name}: submission refused: {error}; fix jobs/{name}/jobscript.sh '
                  f'(keep its last line, the worker), then "triboworkflow clear-refusal {name}"')
        for name, count in jobs.unknown.items():
            print(f'  group {name}: {count} submissions of outcome not known; look for the job '
                  f'in the queue, then "triboworkflow resolve-submission {name} --job-id ID" '
                  'or "--no-job"')
        for name in jobs.withdrawn:
            print(f'  group {name}: withdrawn before any of its jobs was submitted')
        if args.calcs:
            sdir = system_dir(run_dir, system)
            print('Calculations')
            for name in orch.order:
                for calc_id in entries[name]['calculations'] or ():
                    try:
                        state = read_record(sdir, calc_id).status.summary
                    except (OSError, ValueError) as error:
                        state = f'record not read: {error}'
                    print(f'  {calc_id:<24} {state}')


def _lock(found):
    """The holder of the lock of the run, in words."""
    if found is None:
        return 'free'
    text = (f'held by {found["command"]} (pid {found["pid"]} on {found["host"]} since '
            f'{found["since"]})')
    if found['stale']:
        text += '; its process no longer runs: the next command replaces it'
    elif found['command'] is None:
        text = 'run.lock not readable: a command may be starting, or stopped while writing it'
    return text
