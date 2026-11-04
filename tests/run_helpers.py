# Author: Vito Foderà
# Created: 2026-10-29

"""A run on a simulated HPC machine, for the tests of the execution.

No scheduler is called and pw.x never runs. Slurm is simulated at the level
of its commands: ``sbatch`` numbers the jobs, ``sacct`` prints the state and
the restarts set by the test, or nothing for a job it does not report. The
job manager is the real one, with a clock that the tests move forward. The
calculation directories are written by hand in the state a worker leaves
(claim, attempt in the record, ``run.pwi``, ``output.pwo``), with synthetic
pw.x outputs built from the lines pw.x prints.
"""

import datetime
import json
import subprocess

from triboworkflow.core import store
from triboworkflow.execution import layout
from triboworkflow.execution.jobs import Launch, Manifest, write_job
from triboworkflow.execution.manager import JobManager
from triboworkflow.execution.reconcile import Reconciler
from triboworkflow.execution.scheduler import JobScheduler, JobSchedulerError

SYSTEM = 'Cu111_Al111'
FP = 'sha256:' + 'a' * 64
POLL, WAIT = 300, 3600
START = datetime.datetime(2026, 10, 29, 10, 0,
                          tzinfo=datetime.timezone(datetime.timedelta(hours=1)))
SINCE = '2026-10-29T10:00:00+01:00'

# Value of a claim key that is left out of the claim.
DROP = object()

# Jobscript of the user, and its digest as the manifests and the attempts hold it.
JOBSCRIPT = '#!/bin/bash\n'
JOBSCRIPT_DIGEST = store.bytes_digest(JOBSCRIPT.encode())

# Lines of the parallelization of a run on 4 MPI processes.
PARALLEL = ['     Number of MPI processes:                 4',
            '     R & G space division:  proc/nbgrp/npool/nimage =       4']


def header(mixing_beta=None, nstep=None, parallel=()):
    """Beginning of a run, with ``mixing beta`` and ``nstep`` in its summary if given.

    ``parallel`` are lines of the parallelization, written after the first one.
    """
    lines = ['', '     Program PWSCF v.7.2 starts on  1Jan2026 at 10: 0: 0 ', *parallel, '',
             '     bravais-lattice index     =            0']
    if mixing_beta is not None:
        lines.append(f'     mixing beta               = {mixing_beta:12.4f}')
    if nstep is not None:
        lines.append(f'     nstep                     = {nstep:12d}')
    return lines + ['', '     crystal axes: (cart. coord. in units of alat)', '',
                    '     Self-consistent Calculation', '',
                    '     total energy              =     -31.00000000 Ry']


HEADER = header()
CONVERGED = ['', '     End of self-consistent calculation', '',
             '!    total energy              =     -31.10000000 Ry', '',
             '     convergence has been achieved in   7 iterations', '']
NOT_CONVERGED = ['', '     End of self-consistent calculation', '',
                 '     convergence NOT achieved after 100 iterations: stopping', '']
MAX_SECONDS = ['', '     Maximum CPU time exceeded', '',
               '     max_seconds     =     100.00', '     elapsed seconds =     101.00']
MAX_STEPS = ['', '     The maximum number of steps has been reached.', '',
             '     End of BFGS Geometry Optimization', '']
JOB_DONE = ['', '=' + '-' * 78 + '=', '   JOB DONE.', '=' + '-' * 78 + '=']


def text(*parts):
    return '\n'.join(line for part in parts for line in part) + '\n'


SCF_DONE = text(HEADER, CONVERGED, JOB_DONE)
SCF_MAX_SECONDS = text(HEADER, MAX_SECONDS, JOB_DONE)
SCF_TRUNCATED = text(HEADER)

# What the parser leaves in the record of an output read with parsing 'ok'.
RESULTS = {'nsites': 2, 'input': {'structure': {}}, 'output': {'structure': {}, 'energy': -423.1}}


def parsed_fields(status):
    """Results and ``observed`` of a record whose last attempt ended with ``status``."""
    if status.parsing == 'ok':
        return {'results': RESULTS, 'observed': {}}
    return {'results': None, 'observed': None}


def pwi(calculation='scf', **values):
    """Input of a calculation; ``nstep`` in ``&CONTROL``, ``mixing_beta`` in ``&ELECTRONS``."""
    control = [f"  calculation = '{calculation}'"]
    if 'nstep' in values:
        control.append(f"  nstep = {values['nstep']}")
    electrons = [f"  mixing_beta = {values['mixing_beta']}"] if 'mixing_beta' in values else []
    return '\n'.join(['&CONTROL', *control, '/', '&SYSTEM', '  ibrav = 0', '/', '&ELECTRONS',
                      *electrons, '/', 'K_POINTS gamma', ''])


PWI = pwi()


class Clock:
    """Current time of the job manager, moved forward by the tests."""

    def __init__(self):
        self.time = START

    def __call__(self):
        return self.time


class Slurm(JobScheduler):
    """Slurm whose commands are simulated.

    ``sbatch`` numbers the jobs and sets them PENDING; ``sacct`` prints
    ``states`` or ``restarts`` (0 by default) of a job, nothing for a job
    missing from ``states``. ``fail`` maps a field of ``sacct`` to the error
    its query raises; ``refuse`` makes the next submissions fail with an
    error and create no job. ``submitted`` counts the submissions,
    ``queries`` the calls of ``sacct``.
    """

    def __init__(self):
        super().__init__('slurm')
        self.next_id = 1000
        self.states, self.restarts, self.fail = {}, {}, {}
        self.submitted = self.queries = self.refuse = 0

    def _run_command(self, cmd, input_data=None, cwd=None):
        if cmd[0] == 'sbatch' and self.refuse:
            self.refuse -= 1
            raise JobSchedulerError('Command failed with return code 1: invalid account') \
                from subprocess.CalledProcessError(1, cmd, output='', stderr='invalid account')
        if cmd[0] == 'sbatch':
            job_id = str(self.next_id)
            self.next_id += 1
            self.submitted += 1
            self.states[job_id] = 'PENDING'
            return subprocess.CompletedProcess(cmd, 0, stdout=f'Submitted batch job {job_id}\n',
                                               stderr='')
        assert cmd[0] == 'sacct', cmd
        self.queries += 1
        field, job_id = cmd[cmd.index('-o') + 1], cmd[-1]
        if field in self.fail:
            raise self.fail[field]
        if job_id not in self.states:
            out = ''
        elif field == 'Restarts':
            out = f'{self.restarts.get(job_id, 0):>8}'
        else:
            out = f'{self.states[job_id]:>10}'
        return subprocess.CompletedProcess(cmd, 0, stdout=out + '\n', stderr='')


class Run:
    """A run with calculations, groups of jobs, a simulated Slurm and its job manager."""

    def __init__(self, tmp_path):
        self.dir = tmp_path / 'run'
        self.dir.mkdir()
        self.jobscript = tmp_path / 'job.sh'
        self.jobscript.write_text(JOBSCRIPT)
        self.slurm, self.clock = Slurm(), Clock()
        self.manager = JobManager(self.dir, self.slurm, WAIT, now=self.clock)

    def calc(self, label='aa', input_text=PWI, step='pes_scf', fingerprint=FP):
        """A prepared calculation: directory, ``input.pwi`` and record."""
        calc_id = f'{step}/{label}'
        path = self.path(calc_id)
        path.mkdir(parents=True)
        (path / layout.INPUT_FILE).write_text(input_text)
        store.write_record(self.sdir, store.Record(
            system=SYSTEM, step=step, label=label, calc_type='slab_from_scratch',
            engine=store.Engine('qe', 'pw.x'), energy_model={'ecutwfc': 40.0},
            fingerprint=fingerprint, files={'input': 'input.pwi', 'output': 'output.pwo'}))
        return calc_id

    @property
    def sdir(self):
        return layout.system_dir(self.dir, SYSTEM)

    def path(self, calc_id):
        return layout.calc_dir(self.dir, SYSTEM, calc_id)

    def record(self, calc_id):
        return store.read_record(self.sdir, calc_id)

    def group(self, name, *launches):
        """Write, register and submit a group; the ids of its jobs."""
        write_job(self.dir, Manifest(name=name, scheduler='slurm', run_command='pw.x',
                                     walltime=3600, max_seconds_margin=60,
                                     calculations=launches, pp_command='pp.x'), self.jobscript)
        self.manager.add_group(name, len(launches))
        self.manager.cycle()
        return self.manager.group_jobs(name)

    def claim(self, calc_id, n, job_id, **changes):
        content = {'n': n, 'job_id': job_id, 'restart_count': None, 'host': 'node01',
                   'pid': 77, 'since': SINCE, **changes}
        content = {k: v for k, v in content.items() if v is not DROP}
        (self.path(calc_id) / layout.CLAIM_FILE).write_text(json.dumps(content) + '\n')

    def start(self, calc_id, job_id, output=SCF_TRUNCATED, kind='initial', launch=None, **claim):
        """What a worker leaves while pw.x runs: claim, running attempt, run.pwi, output.

        ``launch`` is the launch of the attempt; without it, an initial launch
        or one of ``kind`` at level 1.
        """
        record = self.record(calc_id)
        n = len(record.attempts) + 1
        if launch is None:
            launch = Launch.initial(SYSTEM, calc_id) if kind == 'initial' else \
                Launch(SYSTEM, calc_id, n, kind, 1)
        self.claim(calc_id, n, job_id, **claim)
        attempt = store.Attempt(n, job_id=job_id, host='node01', started=SINCE, max_seconds=3540,
                                run_command='pw.x', jobscript_digest=JOBSCRIPT_DIGEST,
                                input=store.AttemptFile('run.pwi'),
                                output=store.AttemptFile('output.pwo'),
                                status=store.Status('running'), remedies=launch.remedies())
        store.write_record(self.sdir, store.Record(**{
            **record.__dict__, 'attempts': record.attempts + (attempt,),
            'status': store.Status('running')}))
        (self.path(calc_id) / layout.RUN_INPUT_FILE).write_text(
            (self.path(calc_id) / layout.INPUT_FILE).read_text())
        if output is not None:
            (self.path(calc_id) / layout.OUTPUT_FILE).write_text(output)

    def finish(self, calc_id, **fields):
        """End of the last attempt as the worker writes it, results included."""
        record = self.record(calc_id)
        done = store.Attempt(**{**record.attempts[-1].__dict__, 'ended': SINCE, 'exit_code': 0,
                                **fields})
        store.write_record(self.sdir, store.Record(**{
            **record.__dict__, 'attempts': record.attempts[:-1] + (done,),
            'status': done.status, **parsed_fields(done.status)}))

    def poll(self, seconds=0, **states):
        """Move the clock, set job states, run a cycle of the manager."""
        self.clock.time += datetime.timedelta(seconds=seconds)
        self.slurm.states.update(states)
        self.manager.cycle()

    def reconciler(self):
        return Reconciler(self.dir, self.manager, POLL)

    def reconcile(self, calc_id):
        return self.reconciler().calculation(SYSTEM, calc_id)

    def files(self, calc_id):
        """Files of the calculation directory, the record apart."""
        path = self.path(calc_id)
        return {p.name: p.read_bytes() for p in sorted(path.iterdir())
                if p.is_file() and p.name != layout.RECORD_FILE}
