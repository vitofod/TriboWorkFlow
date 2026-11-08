# Author: Vito Foderà
# Created: 2026-10-27

"""Directories and file names of a run on HPC.

A run keeps the systems and the jobs that run their calculations::

    <run>/
    ├── systems/<system>/
    │   ├── system.json
    │   └── <step>/<label>/          one directory per calculation
    │       ├── input.pwi            input prepared for the calculation
    │       ├── structure_in.extxyz  its structure, with the constraints of the input
    │       ├── input.ppi            pp.x input, if the calculation gives the charge density
    │       ├── run.pwi              input of the current attempt
    │       ├── output.pwo           output of the current attempt
    │       ├── structure_out.extxyz final structure, once the output is parsed
    │       ├── output.ppo           pp.x output of the current attempt
    │       ├── density.filplot      charge density written by pp.x
    │       ├── claim                taken by the worker running the attempt
    │       ├── record.json
    │       ├── OUT/                 outdir of pw.x
    │       └── attempts/<n>/        the files of attempt n: run.pwi, output.pwo,
    │                                structure_out.extxyz, and input.ppi, output.ppo and
    │                                density.filplot of the charge density
    ├── jobs/<name>/                 one group of jobs
    │   ├── manifest.json            calculations of the group and how to run them
    │   ├── analysis.json            in place of manifest.json: the step analysed in the job
    │   ├── outcome.json             outcome of that analysis, written by its job
    │   ├── jobscript.sh             jobscript submitted for each job
    │   ├── submitted.json           ids of the submitted jobs
    │   └── withdrawn.json           only for a group withdrawn before any submission
    ├── replaced/<system>/<step>/<k>/  one replacement of calculations of a step
    │   ├── replacement.json         what was replaced, why, and the groups concerned
    │   └── <step>/<label>/          the directory of a calculation replaced, as it was
    └── jobs.json                    registry of the job manager: jobs asked, states

pw.x runs inside the calculation directory, so that the relative ``outdir``
of every calculation is its own ``OUT/``. The ``prefix`` is made of step and
label. ``input.pwi`` is never changed after preparation: each attempt runs
``run.pwi``, a copy with the time limit of the job written in it. pp.x runs
in the same directory on ``input.ppi``, as prepared. A
calculation whose settings changed is replaced only on request: its whole
directory moves to ``replaced/`` and is prepared again.
"""

from pathlib import Path

from triboworkflow.core.store.record import RECORD_FILE, check_name, split_calc_id

SYSTEMS_DIR = 'systems'
JOBS_DIR = 'jobs'
JOBS_REGISTRY_FILE = 'jobs.json'

REPLACED_DIR = 'replaced'

INPUT_FILE = 'input.pwi'
STRUCTURE_IN_FILE = 'structure_in.extxyz'
RUN_INPUT_FILE = 'run.pwi'
OUTPUT_FILE = 'output.pwo'
STRUCTURE_OUT_FILE = 'structure_out.extxyz'
PP_INPUT_FILE = 'input.ppi'
PP_OUTPUT_FILE = 'output.ppo'
DENSITY_FILE = 'density.filplot'
CLAIM_FILE = 'claim'
OUTDIR = 'OUT'
ATTEMPTS_DIR = 'attempts'
REPLACEMENT_FILE = 'replacement.json'

__all__ = ['ATTEMPTS_DIR', 'CLAIM_FILE', 'DENSITY_FILE', 'INPUT_FILE', 'JOBS_DIR',
           'JOBS_REGISTRY_FILE', 'OUTDIR', 'OUTPUT_FILE', 'PP_INPUT_FILE', 'PP_OUTPUT_FILE',
           'RECORD_FILE', 'REPLACED_DIR', 'REPLACEMENT_FILE',
           'RUN_INPUT_FILE', 'STRUCTURE_IN_FILE', 'STRUCTURE_OUT_FILE', 'SYSTEMS_DIR',
           'attempt_dir', 'calc_dir',
           'calc_prefix', 'job_dir', 'replaced_dir', 'system_dir']


def system_dir(run_dir, system):
    """Directory of a system: ``<run>/systems/<system>``."""
    return Path(run_dir) / SYSTEMS_DIR / check_name(system, 'system')


def calc_dir(run_dir, system, calc_id):
    """Directory of a calculation: ``<run>/systems/<system>/<step>/<label>``.

    Parameters
    ----------
    run_dir : path-like
        Directory of the run.
    system : str
        Name of the system.
    calc_id : str
        ``<step>/<label>``.
    """
    step, label = split_calc_id(calc_id)
    return system_dir(run_dir, system) / step / label


def calc_prefix(calc_id):
    """``prefix`` of pw.x for a calculation: ``<step>_<label>``.

    Each calculation also has its own ``outdir`` (``OUTDIR``, relative to
    its directory), so the prefix only makes the files recognizable.
    """
    step, label = split_calc_id(calc_id)
    return f'{step}_{label}'


def attempt_dir(calc_path, n):
    """Archive of attempt ``n`` (counted from 1) of a calculation: ``attempts/<n>``."""
    if type(n) is not int or n < 1:
        raise ValueError(f'attempt number: {n!r} is not an integer >= 1')
    return Path(calc_path) / ATTEMPTS_DIR / str(n)


def job_dir(run_dir, name):
    """Directory of a group of jobs: ``<run>/jobs/<name>``."""
    return Path(run_dir) / JOBS_DIR / check_name(name, 'job name')


def replaced_dir(run_dir, system, step):
    """Replacements of the calculations of a step: ``<run>/replaced/<system>/<step>``.

    ``step`` is the step of the workflow, whose calculations may have
    ``calc_id`` of other names.
    """
    return Path(run_dir) / REPLACED_DIR / check_name(system, 'system') / check_name(step, 'step')
