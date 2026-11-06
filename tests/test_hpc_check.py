# Author: Vito Foderà
# Created: 2026-10-27

"""Local run of ``tests/hpc/hpc_check.py``, so that the HPC check works when it is used.

No job is submitted: the run is prepared in dry run and its workers are
started here, on the fake engine. POSIX only, since the worker launches the
engine with bash.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / 'hpc' / 'hpc_check.py'
RANK_VARIABLES = ('OMPI_COMM_WORLD_RANK', 'PMIX_RANK', 'PMI_RANK', 'SLURM_PROCID')

pytestmark = pytest.mark.skipif(os.name != 'posix' or not Path('/bin/bash').exists(),
                                reason='the worker launches the engine with /bin/bash')


def check(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True,
                          text=True, timeout=120)


def fake_run(tmp_path, calcs):
    """A run of ``calcs`` fake calculations, in one group, not submitted."""
    jobscript = tmp_path / 'job.sh'
    jobscript.write_text('#!/bin/bash\n')
    run = tmp_path / 'run'
    assert check('prepare', run, '--jobscript', jobscript, '--calcs', calcs, '--groups', 1,
                 '--sleep', 0).returncode == 0
    return run


def engine_dir(run, label, rank_wait):
    """Directory of calculation ``label``, ready for the fake engine started by hand."""
    calc = run / 'systems' / 'check' / 'fake_scf' / label
    spec = json.loads((calc / 'fake.json').read_text())
    (calc / 'fake.json').write_text(json.dumps({**spec, 'rank_wait': rank_wait}))
    (calc / 'run.pwi').write_text((calc / 'input.pwi').read_text())
    return calc


def start_engine(run, calc, rank_env):
    """The fake engine in ``calc``, as one rank of a launcher with ``rank_env``."""
    env = {k: v for k, v in os.environ.items() if k not in RANK_VARIABLES}
    return subprocess.Popen([sys.executable, str(run / 'fake_pw.py'), '-in', 'run.pwi'],
                            cwd=calc, env={**env, **rank_env}, stdout=subprocess.DEVNULL)


@pytest.mark.parametrize('ranks', [
    [{'SLURM_PROCID': '0', 'OMPI_COMM_WORLD_RANK': str(i), 'PMIX_RANK': str(i), 'PMI_RANK': ''}
     for i in range(4)],
    [{'SLURM_PROCID': str(i)} for i in range(4)],
    [{'SLURM_PROCID': str(i), 'PMI_RANK': str(i)} for i in range(4)],
    [{}],
], ids=['mpirun in a Slurm job', 'srun', 'srun with PMI', 'no launcher'])
def test_fake_engine_logs_once_per_run(tmp_path, ranks):
    run = fake_run(tmp_path, 2)
    calc = engine_dir(run, 'c001', 0.5)
    engines = [start_engine(run, calc, env) for env in ranks]
    assert [e.wait(timeout=60) for e in engines] == [2] * len(ranks)
    logs = list((run / 'fake_log').iterdir())
    assert [p.name.split('.')[0] for p in logs] == ['c001']
    assert json.loads(logs[0].read_text())['exit'] == 2


def test_other_ranks_wait_before_an_exit_code_other_than_0(tmp_path):
    run = fake_run(tmp_path, 2)
    calc = engine_dir(run, 'c001', 1.0)
    started = time.time()
    assert start_engine(run, calc, {'OMPI_COMM_WORLD_RANK': '1'}).wait(timeout=60) == 2
    assert time.time() - started >= 1.0
    assert not list((run / 'fake_log').iterdir())


def test_claims_and_exit_codes(tmp_path):
    jobscript = tmp_path / 'job.sh'
    jobscript.write_text('#!/bin/bash\n#SBATCH --time=00:30:00\n')
    run = tmp_path / 'run'
    done = check('prepare', run, '--jobscript', jobscript, '--calcs', 8, '--groups', 3,
                 '--sleep', 0.1)
    assert done.returncode == 0, done.stderr
    assert 'Not submitted (no --submit)' in done.stdout
    assert done.stdout.count('sbatch ') == 3
    manifest = run / 'jobs' / 'check' / 'manifest.json'
    workers = [subprocess.Popen([sys.executable, '-m', 'triboworkflow.execution.worker',
                                 str(manifest)], env={**os.environ, 'SLURM_JOB_ID': str(i)},
                                stderr=subprocess.DEVNULL)
               for i in range(3)]
    assert [w.wait(timeout=120) for w in workers] == [0, 0, 0]
    done = check('report', run)
    assert done.returncode == 0, done.stdout + done.stderr
    assert '8 calculations' in done.stdout
    for code in (0, 2, 3, 255):
        assert f'  {code} -> {code}: 2 calculations\n' in done.stdout
    assert 'RESULT: each calculation ran at most once' in done.stdout


def test_report_finds_a_calculation_run_twice(tmp_path):
    jobscript = tmp_path / 'job.sh'
    jobscript.write_text('#!/bin/bash\n')
    run = tmp_path / 'run'
    assert check('prepare', run, '--jobscript', jobscript, '--calcs', 2, '--groups', 1,
                 '--sleep', 0).returncode == 0
    manifest = run / 'jobs' / 'check' / 'manifest.json'
    subprocess.run([sys.executable, '-m', 'triboworkflow.execution.worker', str(manifest)],
                   check=True, stderr=subprocess.DEVNULL)
    log = next((run / 'fake_log').glob('c000.*.json'))
    (log.parent / 'c000.copy.json').write_text(log.read_text())
    done = check('report', run)
    assert done.returncode == 1
    assert 'c000: RUN 2 TIMES' in done.stdout


def test_report_names_a_log_that_cannot_be_read(tmp_path):
    run = fake_run(tmp_path, 2)
    manifest = run / 'jobs' / 'check' / 'manifest.json'
    subprocess.run([sys.executable, '-m', 'triboworkflow.execution.worker', str(manifest)],
                   check=True, stderr=subprocess.DEVNULL)
    (run / 'fake_log' / 'c001.999.json').write_text('')
    done = check('report', run)
    assert done.returncode == 1, done.stderr
    assert 'Logs that cannot be read (1), e.g. of a rank 0 stopped while writing: ' \
           'c001.999.json' in done.stdout
    assert 'RESULT: a log of the fake engine cannot be read' in done.stdout


@pytest.mark.parametrize('method', ['excl', 'mkdir'])
def test_locks(tmp_path, method):
    start = time.time() + 1
    command = [sys.executable, str(SCRIPT), 'locks', str(tmp_path), '--method', method,
               '--names', '200', '--start-at', str(start)]
    contenders = [subprocess.Popen(command) for _ in range(6)]
    assert [c.wait(timeout=60) for c in contenders] == [0] * 6
    done = check('locks-report', tmp_path)
    assert done.returncode == 0, done.stdout
    assert f'{method}: 6 contenders on 1 hosts' in done.stdout
    assert '200 locks, 200 taken, 0 with more than one owner' in done.stdout
    results = [json.loads(p.read_text()) for p in (tmp_path / 'results').glob('*.json')]
    assert sorted(i for r in results for i in r['taken']) == list(range(200))
