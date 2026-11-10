# Author: Vito Foderà
# Created: 2026-11-09

"""A run of the nine steps of the interface workflow, with a stand-in for pw.x and pp.x.

``build_chain`` makes the run and takes it to its end; the fixture ``chain`` of
``conftest.py`` calls it once for the whole session. The tests of the run are in
``test_interface_chain.py``, those of the report on it in ``test_interface_check.py``.

The orchestrator runs in this process and a simulated Slurm takes the jobs. Each job is run
as its line in the jobscript would run it, in a process of its own: the worker for the groups
of calculations, with ``tests/qe_check_fake.py pw --emt`` as pw.x and ``... pp`` as pp.x; the
analysis of ``plotpes``, ``plotpes_scf`` and ``shear`` for the groups of an analysis
(``[execution.analysis]``). The logs of the jobs are left in the folders of the groups, as
the jobscript of a real run leaves them (``job.<id>.err``).

The stand-in pw.x gives the energy, the forces and the stress of ASE's EMT, so the surfaces
that ``plotpes`` and ``shear`` work on vary from point to point; the pp.x writes a uniform
density, so the charge displacement is zero. What the run proves is that the steps hand each
other the right data, not that the numbers are those of pw.x. The slabs are two Cu(111): EMT
covers Cu, and the interface of the example (Cu(111)/Al(111)) has an interpolation of 88 GiB.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from run_helpers import Slurm
from test_pes import edit_config, pes_run
from triboworkflow.cli import run as cli_run
from triboworkflow.cli.run import build_orchestrator
from triboworkflow.core.store import read_system
from triboworkflow.execution import layout
from triboworkflow.execution.jobs import ANALYSIS_FILE
from triboworkflow.execution.scheduler import JobScheduler

CUCU = 'mp-30_mp-30'
FAKE = Path(__file__).resolve().parent / 'qe_check_fake.py'
STEPS = ('match', 'pes', 'pes_scf', 'plotpes', 'plotpes_scf', 'shear', 'adhesion', 'ppes',
         'charge')
SHIFTS = [-0.25, 0.0, 0.5, 1.0]

posix_only = pytest.mark.skipif(os.name != 'posix' or not Path('/bin/bash').exists(),
                                reason='the worker launches pw.x and pp.x with /bin/bash')


def chain_run(tmp_path):
    """A run of the example with two Cu(111) slabs, the nine steps and the stand-in programs."""
    run_dir = pes_run(tmp_path, name='run', steps=STEPS)
    edit_config(run_dir, 'file = "slabs/Al111.json"\nmid = "mp-134"',
                'file = "slabs/Cu111.json"\nmid = "mp-30"')
    pseudo = tmp_path / 'pseudo'
    pseudo.mkdir()
    lines = []
    for name in ('Cu.UPF', 'Al.UPF'):
        (pseudo / name).write_text(f'fake {name}\n')
        lines.append(f'{hashlib.md5((pseudo / name).read_bytes()).hexdigest()}  {name}')
    (run_dir / 'pseudo.md5').write_text('\n'.join(lines) + '\n')
    edit_config(run_dir, 'pseudo_dir = "/path/to/pseudo"', f'pseudo_dir = "{pseudo}"')
    edit_config(run_dir, 'run_command = "mpirun -np 32 pw.x -nk 4"',
                f'run_command = "{sys.executable} {FAKE} pw --emt"')
    edit_config(run_dir, '# pp_command = "mpirun -np 32 pp.x"',
                f'pp_command = "{sys.executable} {FAKE} pp"')
    # A few shifts of the top slab, and short strings: the run must be quick.
    edit_config(run_dir, '# shifts = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, '
                '1.75, 2.0, 2.25, 2.75, 3.0]', f'shifts = {SHIFTS}')
    edit_config(run_dir, 'npts = 101', 'npts = 6')
    edit_config(run_dir, 'nstepmax = 99999', 'nstepmax = 2')
    (run_dir / 'analysis.sh').write_text('#!/bin/bash\n#SBATCH --mem=8G\n')
    with open(run_dir / 'config.toml', 'a', encoding='utf-8') as f:
        f.write('\n[execution.analysis]\njobscript = "analysis.sh"\n')
    return run_dir


def run_the_jobs(run_dir, slurm):
    """Run the jobs submitted and not yet run, each in a process of its own; the jobs over.

    Returns
    -------
    list of str
        The ids of the jobs run.
    """
    ran = []
    for submitted in sorted((run_dir / 'jobs').glob('*/submitted.json')):
        folder = submitted.parent
        for job_id in json.loads(submitted.read_text())['job_ids']:
            if slurm.states.get(job_id) != 'PENDING':
                continue
            if (folder / ANALYSIS_FILE).is_file():
                command = [sys.executable, '-m', 'triboworkflow.cli.analysis_job',
                           str(folder / ANALYSIS_FILE)]
            else:
                command = [sys.executable, '-m', 'triboworkflow.execution.worker',
                           str(folder / 'manifest.json')]
            done = subprocess.run(command, env={**os.environ, 'SLURM_JOB_ID': job_id},
                                  capture_output=True, text=True, timeout=1800)
            (folder / f'job.{job_id}.err').write_text(done.stdout + done.stderr)
            assert done.returncode == 0, f'{folder.name}: {done.stdout}{done.stderr}'
            slurm.states[job_id] = 'COMPLETED'
            ran.append(job_id)
    return ran


class Chain:
    """The run after the nine steps, its orchestrator and its simulated Slurm."""

    def __init__(self, run_dir, orch, slurm, passes, report):
        self.run_dir, self.orch, self.slurm = run_dir, orch, slurm
        self.passes, self.report = passes, report

    @property
    def document(self):
        return read_system(layout.system_dir(self.run_dir, CUCU)).document


def build_chain(tmp_path):
    """The run of ``chain_run`` to its end: passes of the orchestrator, the jobs run between them.

    The simulated Slurm replaces the scheduler only while the orchestrator is built: the
    orchestrator keeps it afterwards, and no other test sees it.

    Returns
    -------
    Chain
    """
    slurm = Slurm()
    run_dir = chain_run(tmp_path)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cli_run, 'make_scheduler',
                      lambda settings, dry_run=False: JobScheduler(settings.execution.scheduler,
                                                                   dry_run=True)
                      if dry_run else slurm)
        orch = build_orchestrator(run_dir, groups=2)
    for passes in range(1, 61):
        report = orch.run_pass()
        if report.finished:
            break
        ran = run_the_jobs(run_dir, slurm)
        assert ran, f'pass {passes}: nothing to run, steps {report.steps}'
    return Chain(run_dir, orch, slurm, passes, report)
