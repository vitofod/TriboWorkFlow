# Author: Vito Foderà
# Created: 2026-10-27

"""Tests of the archiving of attempts (``triboworkflow.execution.archive``).

The calculation directories are written by hand in the state a worker
leaves after each of its steps. Archiving is interrupted at each of its own
steps and run again: the final state must be the one of an archiving that
was never interrupted.
"""

import json
import os
import shlex
import sys
from pathlib import Path

import pytest

from triboworkflow.core import store
from triboworkflow.execution import archive as archive_module
from triboworkflow.execution import layout
from triboworkflow.execution.archive import ArchiveError, archive_attempt, read_claim
from triboworkflow.execution.jobs import Launch

SYSTEM, CALC_ID = 'Cu111_Al111', 'pes_scf/aa'
FP = 'sha256:' + 'a' * 64
RUN_PWI = "&CONTROL\n  max_seconds = 85800\n/\n"
OUTPUT = "     Program PWSCF v.7.2 starts on  1Jan2026 at 10: 0: 0 \n"
CLAIM = {'n': 1, 'job_id': '4521', 'host': 'node01', 'pid': 77,
         'since': '2026-10-27T10:00:00+01:00'}


class Interrupt(Exception):
    """Stop of the archiving at a given point."""


def base_record(attempts=(), status=store.Status()):
    return store.Record(system=SYSTEM, step='pes_scf', label='aa', calc_type='slab_from_scratch',
                        engine=store.Engine('qe', 'pw.x'), energy_model={'ecutwfc': 40.0},
                        fingerprint=FP, status=status, attempts=attempts,
                        files={'input': 'input.pwi', 'output': 'output.pwo'})


@pytest.fixture
def calc(tmp_path):
    """Calculation in the state the worker leaves; ``calc(stage)`` writes it."""
    run_dir = tmp_path / 'run'
    path = layout.calc_dir(run_dir, SYSTEM, CALC_ID)
    sdir = layout.system_dir(run_dir, SYSTEM)

    def write(stage='finished', previous=0):
        path.mkdir(parents=True, exist_ok=True)
        (path / layout.INPUT_FILE).write_text("&CONTROL\n  outdir = 'OUT'\n/\n")
        (path / layout.OUTDIR).mkdir(exist_ok=True)
        (path / layout.OUTDIR / 'p.save').write_text('restart data\n')
        done = tuple(store.Attempt(i + 1, job_id=str(i + 1), status=store.Status('killed'))
                     for i in range(previous))
        claim = {**CLAIM, 'n': previous + 1}
        if stage == 'empty claim':
            (path / layout.CLAIM_FILE).write_text('')
            store.write_record(sdir, base_record(done))
            return
        (path / layout.CLAIM_FILE).write_text(json.dumps(claim) + '\n')
        if stage == 'claim':
            store.write_record(sdir, base_record(done))
            return
        (path / layout.RUN_INPUT_FILE).write_text(RUN_PWI)
        (path / layout.OUTPUT_FILE).write_text(OUTPUT)
        status = store.Status('running') if stage == 'running' else store.Status('crashed')
        attempt = store.Attempt(previous + 1, job_id='4521', host='node01',
                                started=CLAIM['since'], max_seconds=85800,
                                input=store.AttemptFile('run.pwi'),
                                output=store.AttemptFile('output.pwo'), status=status)
        if stage == 'finished':
            attempt = store.Attempt(**{
                **attempt.__dict__, 'ended': '2026-10-27T11:00:00+01:00', 'exit_code': 137,
                'input': store.AttemptFile('run.pwi', store.file_digest(path / 'run.pwi')),
                'output': store.AttemptFile('output.pwo', store.file_digest(path / 'output.pwo')),
                'stop_reason': 'truncated', 'clean_stop': False})
        store.write_record(sdir, base_record(done + (attempt,), status))

    write.run_dir, write.path, write.sdir = run_dir, path, sdir
    return write


def state(calc):
    """Record and files of the calculation directory."""
    files = {str(p.relative_to(calc.path)): p.read_bytes() for p in sorted(calc.path.rglob('*'))
             if p.is_file() and p.name != layout.RECORD_FILE}
    return store.read_record(calc.sdir, CALC_ID).to_dict(), files


def archive(calc):
    return archive_attempt(calc.run_dir, SYSTEM, CALC_ID)


# --- archiving --------------------------------------------------------------------

def test_archive_a_finished_attempt(calc):
    calc('finished')
    before = store.read_record(calc.sdir, CALC_ID).attempts[0]
    assert archive(calc) == 1
    path = calc.path
    assert not (path / layout.CLAIM_FILE).exists()
    assert not (path / layout.RUN_INPUT_FILE).exists() and not (path / layout.OUTPUT_FILE).exists()
    assert (path / 'attempts' / '1' / 'run.pwi').read_text() == RUN_PWI
    assert (path / 'attempts' / '1' / 'output.pwo').read_text() == OUTPUT
    assert (path / layout.INPUT_FILE).read_text() == "&CONTROL\n  outdir = 'OUT'\n/\n"
    assert (path / layout.OUTDIR / 'p.save').read_text() == 'restart data\n'
    record = store.read_record(calc.sdir, CALC_ID)
    (attempt,) = record.attempts
    assert attempt.input == store.AttemptFile('attempts/1/run.pwi', before.input.digest)
    assert attempt.output == store.AttemptFile('attempts/1/output.pwo', before.output.digest)
    assert attempt == store.Attempt(**{**before.__dict__, 'input': attempt.input,
                                       'output': attempt.output})
    assert record.status == store.Status('crashed')


def test_nothing_to_archive(calc):
    calc('finished')
    archive(calc)
    assert archive(calc) is None


@pytest.mark.parametrize('name', [layout.RUN_INPUT_FILE, layout.OUTPUT_FILE])
def test_files_without_a_claim(calc, name):
    calc('finished')
    os.unlink(calc.path / layout.CLAIM_FILE)
    with pytest.raises(ArchiveError, match='without a claim'):
        archive(calc)
    other = ({layout.RUN_INPUT_FILE, layout.OUTPUT_FILE} - {name}).pop()
    os.unlink(calc.path / other)
    with pytest.raises(ArchiveError, match=f'{name} without a claim'):
        archive(calc)


def test_claim_without_attempt(calc):
    calc('claim', previous=1)
    assert archive(calc) == 2
    record = store.read_record(calc.sdir, CALC_ID)
    assert [a.n for a in record.attempts] == [1, 2]
    assert record.attempts[1] == store.Attempt(2, job_id='4521', host='node01',
                                               started=CLAIM['since'])
    assert not (calc.path / layout.CLAIM_FILE).exists()
    assert not (calc.path / 'attempts').exists()


def test_empty_claim(calc):
    calc('empty claim', previous=2)
    assert archive(calc) == 3
    record = store.read_record(calc.sdir, CALC_ID)
    assert record.attempts[2] == store.Attempt(3)
    assert not (calc.path / layout.CLAIM_FILE).exists()


def test_archive_gives_the_launch_to_the_new_attempt(calc):
    calc('empty claim', previous=2)
    with pytest.raises(ArchiveError, match='launch of attempt 2 given for attempt 3'):
        archive_attempt(calc.run_dir, SYSTEM, CALC_ID, launch=Launch(SYSTEM, CALC_ID, 2, 'manual', 1))
    assert (calc.path / layout.CLAIM_FILE).read_text() == ''  # nothing changed
    launch = Launch(SYSTEM, CALC_ID, 3, 'unchanged', 1)
    assert archive_attempt(calc.run_dir, SYSTEM, CALC_ID, launch=launch) == 3
    record = store.read_record(calc.sdir, CALC_ID)
    assert record.attempts[2] == store.Attempt(3, remedies=launch.remedies())


def test_running_attempt_is_not_archived(calc):
    calc('running')
    before = state(calc)
    with pytest.raises(ArchiveError, match='attempt 1 is recorded as running'):
        archive(calc)
    assert state(calc) == before


def test_file_in_both_places(calc):
    calc('finished')
    archived = calc.path / 'attempts' / '1'
    archived.mkdir(parents=True)
    (archived / 'output.pwo').write_text('another output\n')
    before = state(calc)
    with pytest.raises(ArchiveError, match='both exist; nothing was moved or deleted'):
        archive(calc)
    assert state(calc) == before


@pytest.mark.parametrize('name', [layout.RUN_INPUT_FILE, layout.OUTPUT_FILE])
def test_file_changed_after_the_attempt(calc, name):
    calc('finished')
    (calc.path / name).write_text('changed\n')
    before = state(calc)
    with pytest.raises(ArchiveError, match='differs from the one recorded for attempt 1'):
        archive(calc)
    assert state(calc) == before


@pytest.mark.parametrize('claim_n, previous', [(3, 0), (1, 2)])
def test_claim_and_record_disagree(calc, claim_n, previous):
    calc('finished', previous=previous)
    (calc.path / layout.CLAIM_FILE).write_text(json.dumps({**CLAIM, 'n': claim_n}))
    with pytest.raises(ArchiveError, match=f'the claim is for attempt {claim_n}'):
        archive(calc)


@pytest.mark.parametrize('text', ['', '{', '[]', '{"n": 0}', '{"n": "1"}'])
def test_unreadable_claims(tmp_path, text):
    path = tmp_path / 'claim'
    path.write_text(text)
    assert read_claim(path) is None


@pytest.mark.parametrize('content, expected', [
    ({**CLAIM, 'restart_count': 2}, {**CLAIM, 'restart_count': 2}),
    ({**CLAIM, 'restart_count': None}, {**CLAIM, 'restart_count': None}),
    ({**CLAIM, 'restart_count': 'x1'}, {**CLAIM, 'restart_count': 'x1'}),
    # Written without the restart count: the key is not there, not None.
    (CLAIM, CLAIM),
    ({'n': 3}, {'n': 3, 'job_id': None, 'host': None, 'pid': None, 'since': None}),
])
def test_read_claim(tmp_path, content, expected):
    path = tmp_path / 'claim'
    path.write_text(json.dumps(content))
    assert read_claim(path) == expected


def test_empty_claim_rewritten_without_restart_count(calc, monkeypatch):
    calc('empty claim', previous=1)
    with monkeypatch.context() as patch:
        patch.setattr(archive_module, 'write_json', failing(archive_module.write_json, 1, True))
        with pytest.raises(Interrupt):
            archive(calc)
    assert read_claim(calc.path / layout.CLAIM_FILE) == {
        'n': 2, 'job_id': None, 'host': None, 'pid': None, 'since': None}


# --- interrupted archiving ---------------------------------------------------------------

def failing(function, at, after=False):
    """``function`` that raises Interrupt at call number ``at`` (from 1), before or after it."""
    calls = []

    def wrapper(*args, **kwargs):
        calls.append(1)
        if len(calls) == at and not after:
            raise Interrupt
        result = function(*args, **kwargs)
        if len(calls) == at and after:
            raise Interrupt
        return result
    return wrapper


# (stage, function of the archive module that stops, call, after it)
INTERRUPTIONS = [
    ('empty claim', 'write_json', 1, True),       # number written in the claim
    ('empty claim', 'write_record', 1, True),     # attempt added to the record
    ('claim', 'write_record', 1, False),          # before adding the attempt
    ('claim', 'write_record', 1, True),           # attempt added
    ('finished', 'replace', 1, True),             # run.pwi moved
    ('finished', 'replace', 2, True),             # both files moved
    ('finished', 'write_record', 1, True),        # record updated
    ('finished', 'unlink', 1, False),             # before removing the claim
]


@pytest.mark.parametrize('stage, name, at, after', INTERRUPTIONS,
                         ids=[f'{s}-{n}{a}-{"after" if b else "before"}'
                              for s, n, a, b in INTERRUPTIONS])
def test_interrupted_archiving_is_completed(tmp_path, calc, monkeypatch, stage, name, at, after):
    calc(stage, previous=1)
    archive(calc)
    expected = state(calc)

    for path in sorted(calc.run_dir.rglob('*'), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    calc(stage, previous=1)
    target = archive_module.os if name in ('replace', 'unlink') else archive_module
    with monkeypatch.context() as patch:
        patch.setattr(target, name, failing(getattr(target, name), at, after))
        with pytest.raises(Interrupt):
            archive(calc)
    assert (calc.path / layout.CLAIM_FILE).exists()
    assert archive(calc) == 2
    assert state(calc) == expected


# --- with the worker ------------------------------------------------------------------

FAKE_ENGINE = '''\
import sys
sys.stdout.write("     Program PWSCF v.7.2 starts on  1Jan2026 at 10: 0: 0 \\n")
sys.exit(1)
'''


@pytest.mark.skipif(os.name != 'posix' or not Path('/bin/bash').exists(),
                    reason='the worker launches the engine with /bin/bash')
def test_new_attempt_after_archiving(calc, tmp_path):
    from triboworkflow.execution.jobs import MANIFEST_FILE, Launch, Manifest, write_job
    from triboworkflow.execution.worker import Worker

    calc('finished')
    fake = tmp_path / 'fake.py'
    fake.write_text(FAKE_ENGINE)
    jobscript = tmp_path / 'job.sh'
    jobscript.write_text('#!/bin/bash\n')
    manifest = Manifest(name='pes_scf', scheduler='slurm',
                        run_command=f'{shlex.quote(sys.executable)} {shlex.quote(str(fake))}',
                        walltime=86400, max_seconds_margin=600,
                        calculations=(Launch(SYSTEM, CALC_ID, 2, 'unchanged', 1),))
    manifest_path = write_job(calc.run_dir, manifest, jobscript) / MANIFEST_FILE
    environ = {'SLURM_JOB_ID': '4522', 'SLURM_RESTART_COUNT': '1'}
    assert Worker(manifest_path, environ=environ).run() == [(SYSTEM, CALC_ID, 'taken')]
    assert archive(calc) == 1
    assert Worker(manifest_path, environ=environ).run() == [(SYSTEM, CALC_ID, 'ran')]
    record = store.read_record(calc.sdir, CALC_ID)
    assert [a.n for a in record.attempts] == [1, 2]
    assert record.attempts[0].output.path == 'attempts/1/output.pwo'
    assert record.attempts[1].output.path == 'output.pwo'
    assert record.attempts[1].exit_code == 1
    assert record.attempts[1].remedies == {'kind': 'unchanged', 'level': 1, 'values': {},
                                           'initial_values': {}, 'restart': False}
    claim = read_claim(calc.path / layout.CLAIM_FILE)
    assert (claim['n'], claim['job_id'], claim['restart_count']) == (2, '4522', 1)
