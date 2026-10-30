# Author: Vito Foderà
# Created: 2026-10-30

"""Tests of the lock of a run (``triboworkflow.execution.lock``).

One holder at a time: a second one is refused and names the first; a lock
left by a process gone from this host is replaced, one written on another
host is never touched, and a holder removes only its own lock.
"""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from run_helpers import SYSTEM, WAIT, Run
from triboworkflow.execution.jobs import Launch, Manifest, write_job
from triboworkflow.execution.lock import LOCK_FILE, TAKEOVER_FILE, RunLock, RunLockError, holder
from triboworkflow.execution.manager import JobManager

posix_only = pytest.mark.skipif(os.name != 'posix', reason='process checks are POSIX only')


def lock_file(run_dir, **content):
    data = {'pid': 4242, 'host': socket.gethostname(), 'since': '2026-10-30T10:00:00+01:00',
            'command': 'triboworkflow submit pes', 'token': 'abc', **content}
    (run_dir / LOCK_FILE).write_text(json.dumps(data) + '\n')
    return data


def read_lock(run_dir):
    return json.loads((run_dir / LOCK_FILE).read_text())


def gone_pid():
    """Pid of a process that has ended."""
    process = subprocess.Popen([sys.executable, '-c', 'pass'])
    process.wait()
    return process.pid


def test_lock_created_and_removed(tmp_path):
    with RunLock(tmp_path, 'triboworkflow submit pes') as lock:
        data = read_lock(tmp_path)
        assert (data['pid'], data['host'], data['command'], data['token']) == \
               (os.getpid(), socket.gethostname(), 'triboworkflow submit pes', lock.token)
        assert data['since']
        assert holder(tmp_path) == {**data, 'stale': False}
    assert not (tmp_path / LOCK_FILE).exists()
    assert holder(tmp_path) is None


def test_second_holder_is_refused(tmp_path):
    with RunLock(tmp_path, 'triboworkflow submit pes'):
        first = read_lock(tmp_path)
        with pytest.raises(RunLockError) as error:
            RunLock(tmp_path, 'triboworkflow relaunch pes_scf/aa').acquire()
        message = str(error.value)
        assert 'triboworkflow submit pes' in message
        assert f'pid {os.getpid()}' in message and socket.gethostname() in message
        assert read_lock(tmp_path) == first


def test_lock_released_after_an_exception(tmp_path):
    with pytest.raises(KeyError):
        with RunLock(tmp_path, 'triboworkflow retry pes'):
            raise KeyError('pes')
    assert not (tmp_path / LOCK_FILE).exists()


@posix_only
def test_dead_holder_on_this_host_is_replaced(tmp_path, caplog):
    lock_file(tmp_path, pid=gone_pid())
    assert holder(tmp_path)['stale'] is True
    with RunLock(tmp_path, 'triboworkflow submit pes') as lock:
        assert read_lock(tmp_path)['token'] == lock.token
    assert 'no longer runs' in caplog.text and 'replaced' in caplog.text


@posix_only
def test_replacement_of_a_stale_lock_admits_one_holder(tmp_path, monkeypatch):
    # B runs whole at the moment A removes the stale lock, after A checked it.
    lock_file(tmp_path, pid=gone_pid())
    path = tmp_path / LOCK_FILE
    original, injected, outcomes = Path.unlink, [], {}

    def unlink(self, *args, **kwargs):
        if self == path and not injected:
            injected.append(True)
            outcomes['B'] = attempt(RunLock(tmp_path, 'triboworkflow relaunch pes_scf/aa'))
        return original(self, *args, **kwargs)

    def attempt(lock):
        try:
            return lock.acquire()
        except RunLockError as error:
            return error
    monkeypatch.setattr(Path, 'unlink', unlink)
    outcomes['A'] = attempt(RunLock(tmp_path, 'triboworkflow submit pes'))
    assert injected
    holders = [lock for lock in outcomes.values() if isinstance(lock, RunLock)]
    assert len(holders) == 1 and read_lock(tmp_path)['token'] == holders[0].token
    assert not (tmp_path / TAKEOVER_FILE).exists()


@posix_only
def test_replacement_left_unfinished_is_never_taken_over(tmp_path):
    data = lock_file(tmp_path, pid=gone_pid())
    (tmp_path / TAKEOVER_FILE).write_text('')  # a command stopped while replacing the lock
    with pytest.raises(RunLockError, match=f'{TAKEOVER_FILE} exists.*by hand'):
        RunLock(tmp_path, 'triboworkflow submit pes').acquire()
    assert read_lock(tmp_path) == data and (tmp_path / TAKEOVER_FILE).exists()


def test_replacement_left_unfinished_does_not_stop_a_free_run(tmp_path):
    (tmp_path / TAKEOVER_FILE).write_text('')
    with RunLock(tmp_path, 'triboworkflow submit pes') as lock:
        assert read_lock(tmp_path)['token'] == lock.token


@posix_only
def test_pid_reused_by_another_program_counts_as_dead(tmp_path):
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    try:
        lock_file(tmp_path, pid=other.pid)
        assert holder(tmp_path)['stale'] is True
        with RunLock(tmp_path, 'triboworkflow submit pes') as lock:
            assert read_lock(tmp_path)['token'] == lock.token
    finally:
        other.kill()
        other.wait()


def test_holder_on_another_host_is_never_touched(tmp_path):
    data = lock_file(tmp_path, host='login02')
    assert holder(tmp_path)['stale'] is False
    with pytest.raises(RunLockError, match='login02.*by hand'):
        RunLock(tmp_path, 'triboworkflow submit pes').acquire()
    assert read_lock(tmp_path) == data


def test_unreadable_lock_is_never_touched(tmp_path):
    (tmp_path / LOCK_FILE).write_text('')
    assert holder(tmp_path)['stale'] is False
    with pytest.raises(RunLockError, match='not readable.*by hand'):
        RunLock(tmp_path, 'triboworkflow submit pes').acquire()
    assert (tmp_path / LOCK_FILE).read_text() == ''


def test_lock_of_another_token_is_not_removed(tmp_path):
    with RunLock(tmp_path, 'triboworkflow submit pes'):
        other = lock_file(tmp_path, token='other')
    assert read_lock(tmp_path) == other


def test_adopt_writes_the_pid_of_the_new_holder(tmp_path):
    lock = RunLock(tmp_path, 'orchestrator', token='t0k3n')
    lock.acquire()
    lock_file(tmp_path, **{**read_lock(tmp_path), 'pid': 4242})  # as written by the parent
    adopted = RunLock.adopt(tmp_path, 't0k3n')
    assert read_lock(tmp_path)['pid'] == os.getpid()
    assert (adopted.token, adopted.command) == ('t0k3n', 'orchestrator')
    adopted.release()
    assert not (tmp_path / LOCK_FILE).exists()


def test_adopt_refuses_the_lock_of_another_token(tmp_path):
    data = lock_file(tmp_path, token='other')
    with pytest.raises(RunLockError, match='not the one'):
        RunLock.adopt(tmp_path, 't0k3n')
    assert read_lock(tmp_path) == data


def test_adopt_refused_while_the_lock_is_replaced(tmp_path):
    RunLock(tmp_path, 'orchestrator', token='t0k3n').acquire()
    before = read_lock(tmp_path)
    (tmp_path / TAKEOVER_FILE).write_text('')
    with pytest.raises(RunLockError, match=f'{TAKEOVER_FILE} exists.*by hand'):
        RunLock.adopt(tmp_path, 't0k3n')
    assert read_lock(tmp_path) == before


def test_adopt_without_lock_is_refused(tmp_path):
    with pytest.raises(RunLockError, match='not the one'):
        RunLock.adopt(tmp_path, 't0k3n')
    assert not (tmp_path / LOCK_FILE).exists()


def test_lost_update_of_two_managers_is_prevented(tmp_path):
    # Two managers that load jobs.json at the same time lose the update of the
    # first that saves; under the lock the second is refused before loading.
    run = Run(tmp_path)
    for name, label in (('g1', 'aa'), ('g2', 'bb')):
        calc_id = run.calc(label)
        write_job(run.dir, Manifest(name=name, scheduler='slurm', run_command='pw.x',
                                    walltime=3600, max_seconds_margin=60,
                                    calculations=(Launch.initial(SYSTEM, calc_id),)),
                  run.jobscript)
    with RunLock(run.dir, 'triboworkflow submit pes'):
        first = JobManager(run.dir, run.slurm, WAIT)
        with pytest.raises(RunLockError):
            with RunLock(run.dir, 'triboworkflow relaunch pes_scf/bb'):
                JobManager(run.dir, run.slurm, WAIT).add_group('g2', 1)
        assert holder(run.dir)['command'] == 'triboworkflow submit pes'
        first.add_group('g1', 1)
    with RunLock(run.dir, 'triboworkflow relaunch pes_scf/bb'):
        JobManager(run.dir, run.slurm, WAIT).add_group('g2', 1)
    assert sorted(JobManager(run.dir, run.slurm, WAIT).groups) == ['g1', 'g2']
