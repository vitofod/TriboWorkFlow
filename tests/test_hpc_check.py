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

pytestmark = pytest.mark.skipif(os.name != 'posix' or not Path('/bin/bash').exists(),
                                reason='the worker launches the engine with /bin/bash')


def check(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True,
                          text=True, timeout=120)


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
