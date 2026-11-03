# Author: Vito Foderà
# Created: 2026-10-27

"""Tests of the worker of a group of jobs (``triboworkflow.execution.worker``).

pw.x never runs: the run command is a small Python program written by the
tests, which prints a synthetic pw.x output (the one given in ``fake.json``
of the calculation directory) and exits with the code given there. The
launch needs bash, so the tests that run the worker are POSIX only.
"""

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import pwo_helpers as pwo
from triboworkflow.core import store
from triboworkflow.core.structures import extxyz_text, read_structure
from triboworkflow.execution import layout
from triboworkflow.execution.archive import archive_attempt
from triboworkflow.execution.jobs import MANIFEST_FILE, Launch, Manifest, read_manifest, write_job
from triboworkflow.execution.worker import (Worker, WorkerError, job_id_from_environment,
                                            read_input, restart_count_from_environment,
                                            with_max_seconds)

posix_only = pytest.mark.skipif(os.name != 'posix' or not Path('/bin/bash').exists(),
                                reason='the worker launches the engine with /bin/bash')

SYSTEM = 'Cu111_Al111'
FP = 'sha256:' + 'a' * 64
WALLTIME, MARGIN = 86400, 600

FAKE_ENGINE = '''\
import json, os, sys, time
args = sys.argv[1:]
assert args[args.index('-in') + 1] == 'run.pwi'
with open('fake.json') as f:
    spec = json.load(f)
with open(spec['log'], 'a') as f:
    f.write(os.path.basename(os.getcwd()) + ' ' + os.environ.get('SLURM_JOB_ID', '-') + '\\n')
time.sleep(spec.get('sleep', 0))
sys.stdout.write(spec['output'])
sys.exit(spec.get('exit', 0))
'''

# Synthetic pw.x outputs, from the lines pw.x prints.
HEADER = ['', '     Program PWSCF v.7.2 starts on  1Jan2026 at 10: 0: 0 ', '',
          '     bravais-lattice index     =            0',
          '', '     crystal axes: (cart. coord. in units of alat)', '',
          '     Self-consistent Calculation', '',
          '     total energy              =     -31.00000000 Ry']
CONVERGED = ['', '     End of self-consistent calculation', '',
             '!    total energy              =     -31.10000000 Ry', '',
             '     convergence has been achieved in   7 iterations', '']
MAX_SECONDS = ['', '     Maximum CPU time exceeded', '',
               '     max_seconds     =     100.00', '     elapsed seconds =     101.00']
JOB_DONE = ['', '=' + '-' * 78 + '=', '   JOB DONE.', '=' + '-' * 78 + '=']


def text(*parts):
    return '\n'.join(line for part in parts for line in part) + '\n'


SCF_DONE = text(HEADER, CONVERGED, JOB_DONE)
SCF_MAX_SECONDS = text(HEADER, MAX_SECONDS, JOB_DONE)
SCF_TRUNCATED = text(HEADER)


def pwi(calculation='scf', max_seconds=None):
    lines = ['&CONTROL', f"  calculation = '{calculation}'", "  prefix = 'p'"]
    if max_seconds is not None:
        lines.append(f'  max_seconds = {max_seconds}')
    lines += ['/', '&SYSTEM', '  ibrav = 0', '/', 'K_POINTS gamma', '']
    return '\n'.join(lines)


class Clock:
    """Clock that returns the given times, then keeps the last one."""

    def __init__(self, *times):
        self.times = list(times)

    def __call__(self):
        return self.times.pop(0) if len(self.times) > 1 else self.times[0]


@pytest.fixture
def run(tmp_path):
    """Builder of a run: calculations with input, record and fake output, and a manifest."""
    fake = tmp_path / 'fake_engine.py'
    fake.write_text(FAKE_ENGINE)
    log = tmp_path / 'engine.log'
    jobscript = tmp_path / 'job.sh'
    jobscript.write_text('#!/bin/bash\n')
    run_dir = tmp_path / 'run'

    def build(labels=('aa',), output=SCF_DONE, exit=0, input_text=None, sleep=0, name='pes_scf',
              run_command=None, record=True, n=1, kind='initial', level=1, values=None,
              initial_values=None):
        calcs = []
        for label in labels:
            calc_id = f'pes_scf/{label}'
            path = layout.calc_dir(run_dir, SYSTEM, calc_id)
            path.mkdir(parents=True)
            (path / layout.INPUT_FILE).write_text(input_text or pwi())
            (path / 'fake.json').write_text(json.dumps({'log': str(log), 'output': output,
                                                        'exit': exit, 'sleep': sleep}))
            if record:
                store.write_record(layout.system_dir(run_dir, SYSTEM), store.Record(
                    system=SYSTEM, step='pes_scf', label=label, calc_type='slab_from_scratch',
                    engine=store.Engine('qe', 'pw.x'), energy_model={'ecutwfc': 40.0},
                    fingerprint=FP, files={'input': 'input.pwi', 'output': 'output.pwo'}))
            calcs.append(Launch(SYSTEM, calc_id, n, kind, level, values or {},
                                initial_values or {}))
        command = run_command or f'{shlex.quote(sys.executable)} {shlex.quote(str(fake))}'
        manifest = Manifest(name=name, scheduler='slurm', run_command=command, walltime=WALLTIME,
                            max_seconds_margin=MARGIN, calculations=tuple(calcs))
        return write_job(run_dir, manifest, jobscript) / MANIFEST_FILE

    build.run_dir, build.log = run_dir, log
    return build


def calc_path(run, label):
    return layout.calc_dir(run.run_dir, SYSTEM, f'pes_scf/{label}')


def record_of(run, label):
    return store.read_record(layout.system_dir(run.run_dir, SYSTEM), f'pes_scf/{label}')


# --- max_seconds in the input ----------------------------------------------------------

def test_max_seconds_added_after_control():
    new, replaced = with_max_seconds(pwi(), 85800)
    assert not replaced
    assert new.splitlines()[:2] == ['&CONTROL', '  max_seconds = 85800']
    assert new.replace('  max_seconds = 85800\n', '') == pwi()


@pytest.mark.parametrize('line, expected', [
    ('  max_seconds = 84600', '  max_seconds = 1000'),
    ('    MAX_SECONDS=8.46d4   ! cap', '    max_seconds = 1000'),
    ('max_seconds = 84600.0', 'max_seconds = 1000'),
])
def test_max_seconds_line_replaced(line, expected):
    source = f"&control\n  calculation = 'scf'\n{line}\n/\n"
    new, replaced = with_max_seconds(source, 1000)
    assert replaced
    assert new == f"&control\n  calculation = 'scf'\n{expected}\n/\n"


def test_max_seconds_keeps_the_line_endings_and_ignores_comments():
    source = "&CONTROL\r\n! max_seconds = 5, nstep = 3\r\n  max_seconds = 9\r\n/\r\n"
    assert with_max_seconds(source, 7) == ("&CONTROL\r\n! max_seconds = 5, nstep = 3\r\n"
                                           "  max_seconds = 7\r\n/\r\n", True)


@pytest.mark.parametrize('source, message', [
    ("&CONTROL\n  max_seconds = 1\n  max_seconds = 2\n/\n", 'alone on one line'),
    ("&CONTROL\n  nstep = 50, max_seconds = 100\n/\n", 'alone on one line'),
    ("&CONTROL\n  max_seconds = 100, nstep = 50\n/\n", 'alone on one line'),
    ("&SYSTEM\n  ibrav = 0\n/\n", 'no &CONTROL'),
    ("&CONTROL\n  calculation = 'scf'\n", 'not closed'),
])
def test_max_seconds_in_other_forms(source, message):
    with pytest.raises(WorkerError, match=message):
        with_max_seconds(source, 100)


@pytest.mark.parametrize('source, expected', [
    (pwi(), ('scf', None)),
    (pwi('relax', 84600), ('relax', 84600)),
    (pwi('scf', '8.46d4'), ('scf', 84600.0)),
    ("&CONTROL\n/\n", ('scf', None)),
])
def test_read_input(tmp_path, source, expected):
    path = tmp_path / 'input.pwi'
    path.write_text(source)
    assert read_input(path) == expected


def test_read_input_other_calculations(tmp_path):
    path = tmp_path / 'input.pwi'
    path.write_text(pwi('vc-relax'))
    with pytest.raises(WorkerError, match="calculation = 'vc-relax'; only scf, relax run"):
        read_input(path)


@pytest.mark.parametrize('scheduler, environ, job_id', [
    ('slurm', {'SLURM_JOB_ID': '123'}, '123'),
    ('pbs', {'PBS_JOBID': '45.server'}, '45.server'),
    ('torque', {'PBS_JOBID': '46'}, '46'),
    ('lsf', {'LSB_JOBID': '7'}, '7'),
    ('sge', {'JOB_ID': '8'}, '8'),
    ('slurm', {}, None),
    ('slurm', {'SLURM_JOB_ID': ''}, None),
    ('htcondor', {'SLURM_JOB_ID': '1'}, None),
])
def test_job_id_from_environment(scheduler, environ, job_id):
    assert job_id_from_environment(scheduler, environ) == job_id


@pytest.mark.parametrize('scheduler, environ, count', [
    ('slurm', {'SLURM_RESTART_COUNT': '2'}, 2),
    ('slurm', {'SLURM_RESTART_COUNT': '0'}, 0),
    # Set by Slurm only after a restart.
    ('slurm', {}, None),
    ('slurm', {'SLURM_RESTART_COUNT': ''}, None),
    # Not an integer: kept as written.
    ('slurm', {'SLURM_RESTART_COUNT': ' 2'}, ' 2'),
    ('slurm', {'SLURM_RESTART_COUNT': '-1'}, '-1'),
    ('pbs', {'SLURM_RESTART_COUNT': '2'}, None),
])
def test_restart_count_from_environment(scheduler, environ, count):
    assert restart_count_from_environment(scheduler, environ) == count


# --- one worker ------------------------------------------------------------------------

@posix_only
def test_attempt_recorded(run):
    manifest = run()
    worker = Worker(manifest, clock=Clock(0.0, 100.4), environ={'SLURM_JOB_ID': '4521'})
    assert worker.run() == [(SYSTEM, 'pes_scf/aa', 'ran')]
    path = calc_path(run, 'aa')
    assert (path / layout.INPUT_FILE).read_text() == pwi()
    assert '  max_seconds = 85699\n' in (path / layout.RUN_INPUT_FILE).read_text()
    assert (path / layout.OUTPUT_FILE).read_text() == SCF_DONE
    claim = json.loads((path / layout.CLAIM_FILE).read_text())
    assert list(claim) == ['n', 'job_id', 'restart_count', 'host', 'pid', 'since']
    assert claim['n'] == 1 and claim['job_id'] == '4521' and claim['pid'] == os.getpid()
    assert claim['restart_count'] is None
    record = record_of(run, 'aa')
    (attempt,) = record.attempts
    assert attempt.remedies == {'kind': 'initial', 'level': 1, 'values': {}, 'initial_values': {}}
    assert record.status == attempt.status == store.Status('ended', 'converged', 'n/a', 'failed')
    assert (attempt.n, attempt.job_id, attempt.max_seconds, attempt.max_seconds_cap) == \
        (1, '4521', WALLTIME - MARGIN - 101, None)
    assert attempt.host == worker.host and attempt.started <= attempt.ended
    assert attempt.input == store.AttemptFile('run.pwi',
                                              store.file_digest(path / layout.RUN_INPUT_FILE))
    assert attempt.output == store.AttemptFile('output.pwo',
                                               store.file_digest(path / layout.OUTPUT_FILE))
    assert (attempt.exit_code, attempt.job_state, attempt.stop_reason, attempt.clean_stop) == \
        (0, None, 'completed', True)
    # A synthetic output without header and no input structure: not parsed.
    assert record.status.parsing == 'failed'
    assert attempt.parse_error.startswith('input structure not read: ')
    assert record.results is None and record.observed is None
    assert record.engine.version == '7.2'


@posix_only
@pytest.mark.parametrize('cap, expected', [(1000, 1000), (90000, WALLTIME - MARGIN),
                                           ('1.5d3', 1500.0)])
def test_cap_of_the_input(run, cap, expected):
    manifest = run(input_text=pwi('scf', cap))
    Worker(manifest, clock=Clock(0.0)).run()
    attempt = record_of(run, 'aa').attempts[0]
    assert attempt.max_seconds == expected
    assert f'  max_seconds = {expected}\n' in \
        (calc_path(run, 'aa') / layout.RUN_INPUT_FILE).read_text()
    assert attempt.max_seconds_cap == (1500.0 if cap == '1.5d3' else cap)


@posix_only
def test_no_more_calculations_without_time(run):
    manifest = run(labels=('aa', 'ab', 'ac'))
    # Start at 0; the first calculation starts at 0, the second when only the margin is left.
    worker = Worker(manifest, clock=Clock(0.0, 0.0, WALLTIME - MARGIN))
    assert worker.run() == [(SYSTEM, 'pes_scf/aa', 'ran')]
    for label in ('ab', 'ac'):
        assert not (calc_path(run, label) / layout.CLAIM_FILE).exists()
        assert record_of(run, label).attempts == ()


@posix_only
def test_time_limit_shrinks_with_the_elapsed_time(run):
    manifest = run(labels=('aa', 'ab'))
    Worker(manifest, clock=Clock(10.0, 10.0, 3610.9)).run()
    assert record_of(run, 'aa').attempts[0].max_seconds == WALLTIME - MARGIN
    assert record_of(run, 'ab').attempts[0].max_seconds == WALLTIME - MARGIN - 3601


@posix_only
def test_claimed_calculation_is_skipped(run):
    manifest = run(labels=('aa', 'ab'))
    (calc_path(run, 'aa') / layout.CLAIM_FILE).write_text('{}\n')
    assert Worker(manifest).run() == [(SYSTEM, 'pes_scf/aa', 'taken'), (SYSTEM, 'pes_scf/ab', 'ran')]
    assert record_of(run, 'aa').attempts == ()
    assert not (calc_path(run, 'aa') / layout.RUN_INPUT_FILE).exists()


@posix_only
@pytest.mark.parametrize('case', ['no record', 'vc-relax', 'output left', 'run input left',
                                  'max_seconds twice', 'cap zero'])
def test_errors_before_the_run_leave_the_calculation_free(run, case, caplog):
    manifest = run(record=case != 'no record',
                   input_text={'vc-relax': pwi('vc-relax'), 'cap zero': pwi('scf', 0),
                               'max_seconds twice': pwi('scf', '1, max_seconds = 2')}.get(case))
    path = calc_path(run, 'aa')
    left = {'output left': layout.OUTPUT_FILE, 'run input left': layout.RUN_INPUT_FILE}.get(case)
    if left:
        (path / left).write_text('previous attempt\n')
    assert Worker(manifest).run() == [(SYSTEM, 'pes_scf/aa', 'error')]
    assert not (path / layout.CLAIM_FILE).exists()
    assert 'pes_scf/aa: not run' in caplog.text
    assert not run.log.exists()
    if case != 'no record':
        assert record_of(run, 'aa').attempts == ()


@posix_only
def test_truncated_output_with_an_exit_code(run):
    Worker(run(output=SCF_TRUNCATED, exit=137)).run()
    attempt = record_of(run, 'aa').attempts[0]
    assert attempt.status == store.Status('crashed', None, 'n/a')
    assert (attempt.exit_code, attempt.stop_reason, attempt.clean_stop) == (137, 'truncated', False)


@posix_only
def test_stop_at_max_seconds(run):
    Worker(run(output=SCF_MAX_SECONDS, exit=255)).run()
    attempt = record_of(run, 'aa').attempts[0]
    assert attempt.status == store.Status('ended', None, 'n/a')
    assert (attempt.exit_code, attempt.stop_reason, attempt.clean_stop) == (255, 'max_seconds', True)


@posix_only
def test_command_not_found(run, caplog):
    Worker(run(run_command='surely_not_a_pw_x_command')).run()
    attempt = record_of(run, 'aa').attempts[0]
    assert attempt.status == store.Status('crashed')
    assert (attempt.exit_code, attempt.stop_reason, attempt.clean_stop) == (127, None, None)
    assert (calc_path(run, 'aa') / layout.OUTPUT_FILE).read_text() == ''


@posix_only
def test_output_that_is_not_pw_output(run, caplog):
    Worker(run(output='Segmentation fault\n', exit=139)).run()
    attempt = record_of(run, 'aa').attempts[0]
    assert attempt.status == store.Status() and record_of(run, 'aa').status == store.Status()
    assert (attempt.exit_code, attempt.stop_reason, attempt.clean_stop) == (139, None, None)
    assert 'pes_scf/aa' in caplog.text


# --- results of the output -------------------------------------------------------------

FORCES = ((0.001, -0.002, 0.003), (-0.001, 0.002, -0.003))


def with_structure(run, s, label='aa'):
    """``structure_in.extxyz`` of ``s`` in the directory of a calculation built by ``run``."""
    (calc_path(run, label) / layout.STRUCTURE_IN_FILE).write_text(extxyz_text(s))


@posix_only
@pytest.mark.parametrize('calculation', ['scf', 'relax'])
def test_results_read_at_the_end_of_the_attempt(run, calculation, caplog):
    s = pwo.structure(if_pos=[[1, 1, 1], [1, 1, 1]])
    manifest = run(output=pwo.pw_output(s, calculation, forces=FORCES),
                   input_text=pwi(calculation))
    with_structure(run, s)
    caplog.set_level('WARNING')
    Worker(manifest, clock=Clock(0.0)).run()
    record = record_of(run, 'aa')
    geometry = 'converged' if calculation == 'relax' else 'n/a'
    assert record.status == record.attempts[-1].status == \
        store.Status('ended', 'converged', geometry, 'ok')
    assert record.attempts[-1].parse_error is None
    assert record.results['nsites'] == 2 and len(record.results['output']['forces']) == 2
    assert record.observed['version'] == record.engine.version == '7.2'
    assert record.observed['species'][0]['md5'] == pwo.MD5_FE
    final = read_structure(calc_path(run, 'aa') / layout.STRUCTURE_OUT_FILE)
    expected = pwo.relaxed(s) if calculation == 'relax' else pwo.structure_in(s).cart_coords
    assert np.allclose(final.cart_coords, expected, atol=1e-8)
    assert final.labels == ['Fe1', 'H']
    assert caplog.records == []


@posix_only
def test_output_that_cannot_be_parsed_leaves_the_reason(run, caplog):
    s = pwo.structure()
    manifest = run(output=pwo.pw_output(s, 'relax'), input_text=pwi('relax'))  # no forces
    with_structure(run, s)
    Worker(manifest, clock=Clock(0.0)).run()
    record = record_of(run, 'aa')
    assert record.status == store.Status('ended', 'converged', 'converged', 'failed')
    assert record.attempts[-1].parse_error == 'no forces in the last SCF cycle of a relaxation'
    assert record.results is None and record.observed['version'] == '7.2'
    assert not (calc_path(run, 'aa') / layout.STRUCTURE_OUT_FILE).exists()
    assert 'pes_scf/aa: output not parsed: no forces' in caplog.text


@posix_only
def test_output_not_converged_is_not_parsed(run):
    s = pwo.structure()
    text = pwo.pw_output(s).replace('     convergence has been achieved in   7 iterations',
                                    '     convergence NOT achieved after 100 iterations: stopping')
    manifest = run(output=text)
    with_structure(run, s)
    Worker(manifest, clock=Clock(0.0)).run()
    record = record_of(run, 'aa')
    assert record.status == store.Status('ended', 'not_converged', 'n/a', None)
    assert (record.results, record.observed, record.attempts[-1].parse_error) == (None,) * 3
    assert record.engine.version == '7.2'


@posix_only
def test_version_outside_those_supported_is_read_with_a_warning(run, caplog):
    s = pwo.structure()
    manifest = run(output=pwo.pw_output(s, version='8.0', forces=FORCES))
    with_structure(run, s)
    Worker(manifest, clock=Clock(0.0)).run()
    record = record_of(run, 'aa')
    assert record.status.parsing == 'ok' and record.engine.version == '8.0'
    assert ('pes_scf/aa: pw.x 8.0: the workflow is built to work with QE 7.1.x, 7.2.x, 7.3.x; '
            'with version 8.0 the results are not guaranteed and may hold errors: check them'
            in caplog.text)


@posix_only
def test_new_attempt_clears_the_results_and_archives_the_final_structure(run):
    s = pwo.structure()
    manifest = run(output=pwo.pw_output(s, forces=FORCES))
    with_structure(run, s)
    Worker(manifest, clock=Clock(0.0)).run()
    path = calc_path(run, 'aa')
    final = (path / layout.STRUCTURE_OUT_FILE).read_bytes()
    assert archive_attempt(run.run_dir, SYSTEM, 'pes_scf/aa') == 1
    assert (path / 'attempts' / '1' / layout.STRUCTURE_OUT_FILE).read_bytes() == final
    assert not (path / layout.STRUCTURE_OUT_FILE).exists()
    # The second attempt runs a truncated output: nothing of the first one stays.
    (path / 'fake.json').write_text(json.dumps({'log': str(run.log), 'output': SCF_TRUNCATED}))
    second = write_job(run.run_dir, Manifest(
        name='again', scheduler='slurm', run_command=read_manifest(manifest)[0].run_command,
        walltime=WALLTIME, max_seconds_margin=MARGIN,
        calculations=(Launch(SYSTEM, 'pes_scf/aa', 2, 'unchanged', 1, {}, {}),)),
        run.run_dir.parent / 'job.sh') / MANIFEST_FILE
    Worker(second, clock=Clock(0.0)).run()
    record = record_of(run, 'aa')
    assert record.status == store.Status('crashed', None, 'n/a', None)
    assert (record.results, record.observed, record.engine.version) == (None, None, '7.2')


def add_attempts(run, label, count):
    """Give the record of a calculation ``count`` killed attempts."""
    attempts = tuple(store.Attempt(i + 1, job_id=str(i + 1), status=store.Status('killed'))
                     for i in range(count))
    store.write_record(layout.system_dir(run.run_dir, SYSTEM), store.Record(**{
        **record_of(run, label).__dict__, 'attempts': attempts,
        'status': store.Status('killed')}))
    return attempts


@posix_only
def test_second_attempt_after_archiving(run):
    manifest = run(n=2, kind='unchanged')
    path = calc_path(run, 'aa')
    (first,) = add_attempts(run, 'aa', 1)
    Worker(manifest, environ={'SLURM_JOB_ID': '2', 'SLURM_RESTART_COUNT': '3'}).run()
    record = record_of(run, 'aa')
    assert [a.n for a in record.attempts] == [1, 2]
    assert record.attempts[0] == first
    assert record.attempts[1].job_id == '2' and record.status.process == 'ended'
    assert record.attempts[1].remedies == {'kind': 'unchanged', 'level': 1, 'values': {},
                                           'initial_values': {}}
    claim = json.loads((path / layout.CLAIM_FILE).read_text())
    assert (claim['n'], claim['restart_count']) == (2, 3)


@posix_only
@pytest.mark.parametrize('made', [1, 2])
def test_attempt_already_made_is_skipped(run, made, caplog):
    # A late job of a replaced group: the record already has attempt n.
    caplog.set_level('INFO')
    manifest = run()
    attempts = add_attempts(run, 'aa', made)
    assert Worker(manifest).run() == [(SYSTEM, 'pes_scf/aa', 'skipped')]
    assert record_of(run, 'aa').attempts == attempts
    assert not (calc_path(run, 'aa') / layout.CLAIM_FILE).exists()
    assert not run.log.exists()
    assert 'attempt 1 already made' in caplog.text


@posix_only
def test_attempt_made_between_the_first_look_and_the_claim(run, monkeypatch):
    manifest = run(labels=('aa', 'ab'))
    worker = Worker(manifest)
    real, calls = Worker._next_attempt, []

    def next_attempt(self, launch):
        calls.append(launch.calc_id)
        if calls == ['pes_scf/aa', 'pes_scf/aa']:
            add_attempts(run, 'aa', 1)  # made and archived by another worker meanwhile
        return real(self, launch)

    monkeypatch.setattr(Worker, '_next_attempt', next_attempt)
    assert worker.run() == [(SYSTEM, 'pes_scf/aa', 'skipped'), (SYSTEM, 'pes_scf/ab', 'ran')]
    assert not (calc_path(run, 'aa') / layout.CLAIM_FILE).exists()
    assert len(record_of(run, 'aa').attempts) == 1


@posix_only
def test_attempt_ahead_of_the_record(run, caplog):
    manifest = run(n=3, kind='unchanged')
    add_attempts(run, 'aa', 1)
    assert Worker(manifest).run() == [(SYSTEM, 'pes_scf/aa', 'error')]
    assert 'the manifest asks for attempt 3, the record has 1 attempts' in caplog.text
    assert not (calc_path(run, 'aa') / layout.CLAIM_FILE).exists()
    assert not run.log.exists()


@posix_only
@pytest.mark.parametrize('kind', ['remedy', 'unchanged'])
def test_launches_above_level_one_are_not_run(run, kind, caplog):
    manifest = run(n=2, kind=kind, level=2, values={'mixing_beta': 0.2, 'nstep': 200},
                   initial_values={'mixing_beta': 0.4, 'nstep': 150})
    add_attempts(run, 'aa', 1)
    assert Worker(manifest).run() == [(SYSTEM, 'pes_scf/aa', 'error')]
    assert ('attempt 2 runs at level 2 with mixing_beta, nstep, which this version does not '
            'write in the input') in caplog.text
    assert not (calc_path(run, 'aa') / layout.CLAIM_FILE).exists()
    assert not (calc_path(run, 'aa') / layout.RUN_INPUT_FILE).exists()
    assert len(record_of(run, 'aa').attempts) == 1


@posix_only
def test_attempt_recorded_as_running_while_the_engine_runs(run):
    manifest = run(sleep=3)
    worker = subprocess.Popen([sys.executable, '-m', 'triboworkflow.execution.worker', str(manifest)],
                              stderr=subprocess.PIPE, text=True)
    try:
        output = calc_path(run, 'aa') / layout.OUTPUT_FILE
        deadline = time.monotonic() + 20
        while not output.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        record = record_of(run, 'aa')
        assert record.status == store.Status('running')
        assert record.attempts[0].status == store.Status('running')
        assert record.attempts[0].ended is None and record.attempts[0].exit_code is None
    finally:
        assert worker.wait(timeout=60) == 0


# --- several workers -----------------------------------------------------------------------

@posix_only
def test_concurrent_workers_run_each_calculation_once(run):
    labels = [f'c{i:02d}' for i in range(12)]
    manifest = run(labels=labels, sleep=0.2)
    workers = [subprocess.Popen([sys.executable, '-m', 'triboworkflow.execution.worker',
                                 str(manifest)],
                                env={**os.environ, 'SLURM_JOB_ID': str(100 + i)},
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
               for i in range(4)]
    results = [(w.wait(timeout=120), w.stderr.read()) for w in workers]
    for w in workers:
        w.stdout.close()
        w.stderr.close()
    assert [code for code, _ in results] == [0, 0, 0, 0], results
    assert all('Worker done:' in err for _, err in results)
    runs = [line.split() for line in run.log.read_text().splitlines()]
    assert sorted(label for label, _ in runs) == labels
    for label, job_id in runs:
        (attempt,) = record_of(run, label).attempts
        assert attempt.job_id == job_id and attempt.status.process == 'ended'
        claim = json.loads((calc_path(run, label) / layout.CLAIM_FILE).read_text())
        assert claim['job_id'] == job_id and claim['n'] == 1
