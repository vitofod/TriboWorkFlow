# Author: Vito Foderà
# Created: 2026-11-08

"""The analysis of one step in a job of its own, on a compute node.

Some analyses need more memory and time than the login node, where the
orchestrator runs, can give (``Step.manual_analysis``, e.g. the
interpolation of a potential energy surface). With ``[execution.analysis]``
the orchestrator writes for such a step a group of jobs with
``analysis.json`` (``execution/jobs.py``) and submits it; the last line of
its jobscript is::

    <python> -m triboworkflow.cli.analysis_job <run>/jobs/<group>/analysis.json

This module is that line. It:

1. reads the group and the run (``<run>/config.toml``) and takes the step
   from the registry of the steps;
2. exits without analysing if the group was withdrawn, if a later group of
   the same step exists (a ``retry`` replaced it), or if the step is already
   done and current (``Step.done``), e.g. analysed by hand meanwhile;
3. checks the entries the step reads (``check_prerequisites``);
4. runs ``Step.analyse``, the same analysis as by hand (``triboworkflow
   analyse <step>``): the step writes ``system.json`` under its own lock,
   after checking again its prerequisites and the entries it read;
5. writes its outcome in ``jobs/<group>/outcome.json``, atomically.

It never takes the lock of the run, which the orchestrator holds for its whole
life, and writes neither the state of the orchestrator nor the registry of the
jobs: like the workers of the calculations, it writes only what belongs to it.
The orchestrator follows the job from the scheduler and, once the job is over,
decides from the entries of the step (``Step.done``), never from the exit
code; ``outcome.json`` gives the reason when the step is not done. A job
stopped by the scheduler (walltime, memory) writes no outcome.

Exit codes: 0 when the step is done (now or before), or the group was
withdrawn or replaced; 1 when the analysis did not run or failed; 2 when the
group cannot be read.
"""

import argparse
import datetime
import logging
import os
import socket
import sys
from pathlib import Path

from triboworkflow.core.store import PrerequisiteError, check_prerequisites, read_system
from triboworkflow.core.store.atomic import write_json
from triboworkflow.execution.jobs import OUTCOME_FILE, is_withdrawn, read_analysis
from triboworkflow.execution.layout import JOBS_DIR, system_dir
from triboworkflow.execution.worker import job_id_from_environment

logger = logging.getLogger(__name__)

OUTCOME_VERSION = 1
# Results of a job: the step analysed now; found already done; the group
# withdrawn or replaced by a later one, nothing analysed; prerequisites missing
# or stale, nothing analysed; an error of the step.
RESULTS = ('analysed', 'done', 'withdrawn', 'replaced', 'blocked', 'failed')
# Results after which the job exits with code 0.
SUCCESS = ('analysed', 'done', 'withdrawn', 'replaced')


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def later_groups(run_dir, analysis):
    """Groups of the same step and system with a larger number, which replace this one."""
    prefix = f'{analysis.system}.{analysis.step}.'
    root = Path(run_dir) / JOBS_DIR
    if not root.is_dir():
        return []
    later = [(int(p.name[len(prefix):]), p.name) for p in root.iterdir()
             if p.name.startswith(prefix) and p.name[len(prefix):].isdigit()
             and int(p.name[len(prefix):]) > analysis.number]
    return [name for _, name in sorted(later)]


def _default_step(name):
    # Imported here: the registry imports the steps and their dependencies.
    from triboworkflow.workflows.interface.registry import get_step

    return get_step(name)


def run_analysis(path, get_step=None, environ=None, now=None):
    """Analyse the step of a group of an analysis and write ``outcome.json``.

    Parameters
    ----------
    path : path-like
        ``analysis.json`` of the group.
    get_step : callable or None, optional
        Step from its name; the registry of the steps by default.
    environ : mapping or None, optional
        Environment, for the id of the job; ``os.environ`` by default.
    now : callable or None, optional
        Current time as an ISO 8601 string; the local time by default.

    Returns
    -------
    dict
        The outcome written: ``result`` (one of ``RESULTS``) and ``error``
        (None, or why the step was not analysed), with group, system, step,
        job id, host, process and the times of start and end.

    Raises
    ------
    JobError, StoreError, OSError
        If the group cannot be read; nothing is written.
    """
    path = Path(path)
    analysis, run_dir = read_analysis(path)
    stamp = now or _now
    get_step = get_step or _default_step
    system, name = analysis.system, analysis.step
    job_id = job_id_from_environment(analysis.scheduler, environ)
    outcome = {'schema_version': OUTCOME_VERSION, 'group': analysis.name, 'system': system,
               'step': name, 'job_id': job_id, 'host': socket.gethostname(),
               'pid': os.getpid(), 'started': stamp(), 'ended': None, 'result': None,
               'error': None}
    logger.info('Analysis of %s/%s, group %s, job %s on %s', system, name, analysis.name,
                job_id, outcome['host'])
    later = later_groups(run_dir, analysis)
    if is_withdrawn(path.parent):
        result, error = 'withdrawn', 'the group was withdrawn: nothing analysed'
    elif later:
        result, error = 'replaced', f'replaced by the group {later[-1]}: nothing analysed'
    else:
        result, error = _analyse(run_dir, system, name, get_step)
    outcome.update(result=result, error=error, ended=stamp())
    write_json(path.parent / OUTCOME_FILE, outcome)
    log = logger.info if result in SUCCESS else logger.error
    log('%s/%s: %s%s', system, name, result, f' ({error})' if error else '')
    return outcome


def _analyse(run_dir, system, name, get_step):
    """``(result, error)`` of the analysis of a step, run unless it is done."""
    try:
        step = get_step(name)
        if not step.manual_analysis:
            return 'failed', f'{name} is not a step analysed in a job of its own'
        if step.done(run_dir, system):
            return 'done', None
        sdir = system_dir(run_dir, system)
        check_prerequisites(read_system(sdir), step.expected(run_dir, system), sdir)
        step.analyse(run_dir, system, [])
    except PrerequisiteError as error:
        return 'blocked', str(error)
    except Exception as error:  # the job reports it, the orchestrator gives it to the user
        logger.exception('%s/%s: the analysis failed', system, name)
        return 'failed', f'{type(error).__name__}: {error}'
    return 'analysed', None


def main(argv=None):
    """Entry point: ``python -m triboworkflow.cli.analysis_job <analysis.json>``."""
    parser = argparse.ArgumentParser(description='Analyse one step in a job of its own.')
    parser.add_argument('analysis', help='analysis.json of the group')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        read_analysis(args.analysis)
    except (OSError, ValueError) as error:  # JobError, and StoreError of the JSON
        logger.error('Group not read: %s', error)
        return 2
    outcome = run_analysis(args.analysis)
    return 0 if outcome['result'] in SUCCESS else 1


if __name__ == '__main__':
    sys.exit(main())
