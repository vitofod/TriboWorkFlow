# Author: Vito Foderà
# Created: 2026-10-30

"""A run folder and fictitious steps, for the tests of the command line.

``write_run`` writes a run as the user would: ``config.toml`` (the example
of the interface workflow, with the steps and the groups of the test) and a
jobscript. The steps are fictitious, as in the tests of the orchestrator:
each one prepares a few calculations and writes one entry of
``system.json``; ``fake_steps`` puts them in the registry of the steps,
which has only ``match``, ``pes``, ``pes_scf``, ``plotpes`` and
``plotpes_scf`` in this version. No
scheduler is called: ``fake_slurm`` replaces the scheduler of the commands
with the simulated Slurm of ``run_helpers``, and
``fake_scheduler_bin`` puts fictitious ``sbatch``, ``sacct`` and ``squeue``
first in the ``PATH``, which record their calls and run nothing.
"""

import json
import os
import re
from pathlib import Path

import pytest

from run_helpers import FP, SCF_DONE, SINCE, Slurm, parsed_fields
from triboworkflow.cli import run as cli_run
from triboworkflow.core import store
from triboworkflow.core.settings.schema import INTERFACE_STEPS
from triboworkflow.core.store import Provenance, SystemWriter, is_done, read_system
from triboworkflow.execution import layout
from triboworkflow.execution.jobs import Launch
from triboworkflow.execution.orchestrator import INTERFACE_GRAPH, Step
from triboworkflow.execution.scheduler import JobScheduler
from triboworkflow.workflows.interface import registry

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples' / 'interface.toml'
SYSTEM = 'mp-30_mp-134'
# Calculations of each fictitious step; match, plotpes and shear only analyse.
CALCS = {'pes': 2, 'adhesion': 2, 'ppes': 1, 'charge': 1}


class FakeStep(Step):
    """A step that prepares ``CALCS[name]`` calculations and writes the entry ``[name]``.

    ``fingerprint`` is the one its settings give to each calculation.
    """

    name = None
    fingerprint = FP

    def done(self, run_dir, system):
        sdir = layout.system_dir(run_dir, system)
        return is_done(read_system(sdir), [self.name], Provenance.build(self.name, {}), sdir)

    def expected(self, run_dir, system):
        return {(dep,): Provenance.build(dep, {}) for dep in INTERFACE_GRAPH[self.name]}

    def plan(self, run_dir, system):
        return {f'{self.name}/c{i}': self.fingerprint for i in range(CALCS.get(self.name, 0))}

    def prepare(self, run_dir, system):
        sdir = layout.system_dir(run_dir, system)
        for calc_id, fingerprint in self.plan(run_dir, system).items():
            path = layout.calc_dir(run_dir, system, calc_id)
            if path.exists():
                continue
            path.mkdir(parents=True)
            (path / layout.INPUT_FILE).write_text("&CONTROL\n  calculation = 'scf'\n/\n")
            step, label = calc_id.split('/')
            store.write_record(sdir, store.Record(
                system=system, step=step, label=label, calc_type='slab_from_scratch',
                engine=store.Engine('qe', 'pw.x'), energy_model={'ecutwfc': 40.0},
                fingerprint=fingerprint, files={'input': 'input.pwi', 'output': 'output.pwo'}))
        return list(self.plan(run_dir, system))

    def analyse(self, run_dir, system, calculations):
        write_entry(run_dir, system, self.name)


def write_entry(run_dir, system, name):
    with SystemWriter(layout.system_dir(run_dir, system)) as writer:
        if not writer.document.exists:
            writer.document.create({})
        writer.document.set([name], {'done': True})
        writer.document.set_provenance([name], Provenance.build(name, {}))
        writer.save()


def complete(run_dir, calc_id, job_id='900', **status):
    """What a worker leaves after running the next attempt of ``calc_id``.

    Claim, ``run.pwi``, ``output.pwo`` and the attempt in the record, ended
    and converged unless other ``status`` fields are given.
    """
    sdir = layout.system_dir(run_dir, SYSTEM)
    path = layout.calc_dir(run_dir, SYSTEM, calc_id)
    record = store.read_record(sdir, calc_id)
    n = len(record.attempts) + 1
    (path / layout.CLAIM_FILE).write_text(json.dumps(
        {'n': n, 'job_id': job_id, 'restart_count': None, 'host': 'node01', 'pid': 77,
         'since': SINCE}) + '\n')
    (path / layout.RUN_INPUT_FILE).write_text((path / layout.INPUT_FILE).read_text())
    (path / layout.OUTPUT_FILE).write_text(SCF_DONE)
    fields = {'status': store.Status('ended', 'converged', 'n/a', 'ok'),
              'stop_reason': 'completed', 'clean_stop': True, **status}
    attempt = store.Attempt(n, job_id=job_id, host='node01', started=SINCE, ended=SINCE,
                            exit_code=0, input=store.AttemptFile('run.pwi'),
                            output=store.AttemptFile('output.pwo'),
                            remedies=Launch.initial(SYSTEM, calc_id).remedies() if n == 1 else {},
                            **fields)
    store.write_record(sdir, store.Record(**{**record.__dict__,
                                             'attempts': record.attempts + (attempt,),
                                             'status': attempt.status,
                                             **parsed_fields(attempt.status)}))


@pytest.fixture
def fake_steps(monkeypatch):
    """The registry of the steps filled with fictitious steps."""
    steps = {name: type(f'Fake_{name}', (FakeStep,), {'name': name}) for name in INTERFACE_STEPS}
    monkeypatch.setattr(registry, 'STEPS', steps)
    return steps


def write_run(tmp_path, steps=('match', 'pes'), groups=10, name='run'):
    """A run folder: ``config.toml`` from the example, with ``steps`` and ``groups``; a jobscript.

    ``groups=None`` leaves ``groups`` out of ``[execution]``.
    """
    text = EXAMPLE.read_text(encoding='utf-8')
    listed = ', '.join(f'"{step}"' for step in steps)
    text = re.sub(r'^steps = .*$', f'steps = [{listed}]', text, count=1, flags=re.M)
    line = '' if groups is None else f'groups = {groups}'
    text = re.sub(r'^groups = \d+.*$', line, text, count=1, flags=re.M)
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / cli_run.CONFIG_FILE).write_text(text, encoding='utf-8')
    (run_dir / 'jobscript.sh').write_text('#!/bin/bash\n')
    return run_dir


@pytest.fixture
def fake_slurm(monkeypatch):
    """Slurm simulated by ``run_helpers.Slurm`` for every command; dry runs stay dry."""
    slurm = Slurm()

    def make(settings, dry_run=False):
        return JobScheduler(settings.execution.scheduler, dry_run=True) if dry_run else slurm

    monkeypatch.setattr(cli_run, 'make_scheduler', make)
    return slurm


@pytest.fixture
def fake_scheduler_bin(tmp_path, monkeypatch):
    """``sbatch``, ``sacct`` and ``squeue`` first in the ``PATH``; the file where they write their calls."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    calls = tmp_path / 'scheduler_calls'
    for command in ('sbatch', 'sacct', 'squeue'):
        path = bin_dir / command
        path.write_text(f'#!/bin/sh\necho "{command} $*" >> "{calls}"\nexit 1\n')
        path.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    return calls
