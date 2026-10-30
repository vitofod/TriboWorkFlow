# Author: Vito Foderà
# Created: 2026-10-29

"""Tests of the orchestrator in the background (``triboworkflow.execution.daemon``).

The background process runs a fictitious orchestrator (``daemon_fake``) that
submits nothing; the submission command of the scheduler is looked for in a
``PATH`` written by the tests. Signals and sessions are POSIX only. A failed
assertion on a background process shows its status, the files of the run and
its log.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from triboworkflow.core.guard import NotOnHPCError, require_scheduler
from triboworkflow.execution import daemon
from triboworkflow.execution.daemon import DaemonError, start, status, stop

posix_only = pytest.mark.skipif(os.name != 'posix', reason='signals and sessions are POSIX only')

FACTORY = 'daemon_fake:build'
TESTS = Path(__file__).resolve().parent


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """A run on a machine where ``sbatch`` is found; the fictitious factory importable."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    sbatch = bin_dir / 'sbatch'
    sbatch.write_text('#!/bin/sh\nexit 1\n')
    sbatch.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join(filter(None, [str(TESTS),
                                                                 os.environ.get('PYTHONPATH')])))
    path = tmp_path / 'run'
    path.mkdir()
    yield path
    found = status(path)
    if found.running:  # never leave a process behind
        os.kill(found.pid, 9)


def wait_for(condition, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def read(path):
    """Text of path, None if it does not exist."""
    return path.read_text() if path.exists() else None


def report(run_dir):
    """Status of the orchestrator of run_dir, the files of the run and its log."""
    lines = [f'status: {status(run_dir).message}']
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name != daemon.LOG_FILE:
            lines.append(f'{path.name}: {path.read_text(errors="replace").strip()}')
    log = read(run_dir / daemon.LOG_FILE)
    lines.append(f'--- {daemon.LOG_FILE}')
    lines.append('(missing)' if log is None else log or '(empty)')
    return '\n'.join(lines)


def pid_file(run_dir, **content):
    data = {'pid': 4242, 'host': socket.gethostname(), 'since': '2026-10-29T10:00:00+01:00',
            'command': 'python -m triboworkflow.execution.daemon serve', 'token': 'abc',
            **content}
    (run_dir / daemon.PID_FILE).write_text(json.dumps(data))


def gone_pid():
    """Pid of a process that has ended."""
    process = subprocess.Popen([sys.executable, '-c', 'pass'])
    process.wait()
    return process.pid


# --- guard ---------------------------------------------------------------------------------

def test_require_scheduler_looks_for_the_submission_command():
    require_scheduler('slurm', which=lambda name: f'/usr/bin/{name}')
    seen = []
    with pytest.raises(NotOnHPCError, match="Command 'qsub' not found: calculations run only "
                                            "on an HPC machine with the scheduler 'pbs'"):
        require_scheduler('pbs', which=lambda name: seen.append(name))
    assert seen == ['qsub']


def test_nothing_starts_away_from_hpc(tmp_path, monkeypatch):
    monkeypatch.setenv('PATH', str(tmp_path))
    with pytest.raises(NotOnHPCError, match="Command 'sbatch' not found"):
        start(tmp_path, FACTORY, 'slurm')
    assert not (tmp_path / daemon.PID_FILE).exists()


# --- start, status, stop -------------------------------------------------------------------

@posix_only
def test_start_status_and_stop(run_dir):
    started = start(run_dir, FACTORY, 'slurm')
    assert started.running is True and started.host == socket.gethostname(), report(run_dir)
    # The pid was written by the process itself, in the file created for its start.
    content = json.loads((run_dir / daemon.PID_FILE).read_text())
    assert content['pid'] == started.pid and content['command'].endswith(
        f'-m triboworkflow.execution.daemon serve {run_dir.resolve()} {FACTORY} '
        f'{content["token"]}'), report(run_dir)
    assert wait_for(lambda: (run_dir / 'passes').exists()), report(run_dir)
    assert status(run_dir).running is True, report(run_dir)
    assert stop(run_dir, timeout=20) is True, report(run_dir)
    # The pass under way ended and the state was saved, then the pid file removed.
    saved = read(run_dir / 'saved')
    assert saved is not None and saved == read(run_dir / 'passes'), report(run_dir)
    assert not (run_dir / daemon.PID_FILE).exists(), report(run_dir)
    assert status(run_dir).running is False, report(run_dir)
    assert 'SIGTERM received' in read(run_dir / daemon.LOG_FILE), report(run_dir)


@posix_only
def test_stop_right_after_start_ends_the_pass(run_dir):
    """start() returns once SIGTERM is handled: a stop at once still saves the state."""
    start(run_dir, FACTORY, 'slurm')
    assert stop(run_dir, timeout=20) is True, report(run_dir)
    saved = read(run_dir / 'saved')
    assert saved is not None and saved == read(run_dir / 'passes'), report(run_dir)
    assert not (run_dir / daemon.PID_FILE).exists(), report(run_dir)
    assert 'SIGTERM received' in read(run_dir / daemon.LOG_FILE), report(run_dir)


@posix_only
def test_orchestrator_done_removes_its_pid_file(run_dir):
    (run_dir / 'limit').write_text('2')
    start(run_dir, FACTORY, 'slurm')
    assert wait_for(lambda: not (run_dir / daemon.PID_FILE).exists()), report(run_dir)
    assert read(run_dir / 'saved') == '2', report(run_dir)
    assert 'Orchestrator done' in read(run_dir / daemon.LOG_FILE), report(run_dir)


@posix_only
def test_orchestrator_done_before_start_goes_on(run_dir, monkeypatch):
    """A process that ends before start() looks at it leaves no pid file behind."""

    class LateParent(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.wait(timeout=20)  # start() goes on only once the process has ended

    monkeypatch.setattr(subprocess, 'Popen', LateParent)
    (run_dir / 'limit').write_text('1')
    started = start(run_dir, FACTORY, 'slurm')
    assert started.running is False and 'exited with code 0' in started.message, report(run_dir)
    assert not (run_dir / daemon.PID_FILE).exists(), report(run_dir)
    assert read(run_dir / 'saved') == '1', report(run_dir)
    assert 'Orchestrator done' in read(run_dir / daemon.LOG_FILE), report(run_dir)


@posix_only
def test_failed_start_is_reported(run_dir):
    with pytest.raises(DaemonError, match=r'not running: pid \d+ exited with code 1 at its '
                                          r'start; see .*orchestrator\.log'):
        start(run_dir, 'daemon_fake:broken', 'slurm')
    assert not (run_dir / daemon.PID_FILE).exists(), report(run_dir)
    assert 'the factory failed' in read(run_dir / daemon.LOG_FILE), report(run_dir)


@posix_only
def test_start_not_confirmed_in_time(run_dir):
    (run_dir / 'delay').write_text('30')
    with pytest.raises(DaemonError, match=r'not started: pid (\d+) did not confirm its start '
                                          r'within 0\.5 s and was killed'):
        start(run_dir, 'daemon_fake:slow', 'slurm', timeout=0.5)
    assert not (run_dir / daemon.PID_FILE).exists(), report(run_dir)
    assert not (run_dir / 'passes').exists(), report(run_dir)


def test_process_of_another_start_never_takes_the_pid_file(run_dir):
    pid_file(run_dir, pid=None, token='of-another-start')
    done = subprocess.run([sys.executable, '-m', daemon.MODULE, 'serve', str(run_dir), FACTORY,
                           'mine'], cwd=run_dir, capture_output=True, text=True, timeout=60)
    assert done.returncode == 1 and 'not the one of this start' in done.stderr, done.stderr
    content = json.loads((run_dir / daemon.PID_FILE).read_text())
    assert (content['pid'], content['token']) == (None, 'of-another-start')
    assert not (run_dir / 'passes').exists()


@posix_only
def test_second_instance_is_refused(run_dir):
    first = start(run_dir, FACTORY, 'slurm')
    with pytest.raises(DaemonError, match=f'not started: running: pid {first.pid}'):
        start(run_dir, FACTORY, 'slurm')
    assert status(run_dir).pid == first.pid
    stop(run_dir, timeout=20)


@posix_only
def test_stale_pid_file_is_replaced(run_dir, caplog):
    pid_file(run_dir, pid=gone_pid())
    found = status(run_dir)
    assert (found.running, found.stale) == (False, True)
    assert 'no longer runs' in found.message
    started = start(run_dir, FACTORY, 'slurm')
    assert started.running is True and 'the pid file is replaced' in caplog.text
    stop(run_dir, timeout=20)


@posix_only
def test_pid_of_another_program_counts_as_gone(run_dir):
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    try:
        pid_file(run_dir, pid=other.pid)
        assert status(run_dir).stale is True
    finally:
        other.kill()
        other.wait()


def test_pid_file_of_another_host_is_never_touched(run_dir):
    pid_file(run_dir, host='login02')
    found = status(run_dir)
    assert found.running is None and 'written on host login02' in found.message
    with pytest.raises(DaemonError, match='not started: .* written on host login02'):
        start(run_dir, FACTORY, 'slurm')
    with pytest.raises(DaemonError, match='nothing to stop: .* written on host login02'):
        stop(run_dir)
    assert json.loads((run_dir / daemon.PID_FILE).read_text())['host'] == 'login02'


def test_pid_not_yet_written(run_dir):
    pid_file(run_dir, pid=None)
    found = status(run_dir)
    assert found.running is None and 'without a pid' in found.message
    with pytest.raises(DaemonError, match='not started'):
        start(run_dir, FACTORY, 'slurm')


def test_stop_without_orchestrator(run_dir):
    with pytest.raises(DaemonError, match='nothing to stop: not running: no '):
        stop(run_dir)


# --- stop request --------------------------------------------------------------------------

def test_stop_request_never_blocks_the_wait():
    """The handler of SIGTERM runs in the main thread between two of its lines, also while
    it waits between two passes: at none of them may the handler block, as Event.set()
    blocks while Event.wait() holds the lock of the event."""

    def wait_with_request_at(line):
        """A wait with the handler called at its line event ``line``; the line events seen."""
        request = daemon.StopRequest(step=0.001)
        seen = 0

        def tracer(frame, event, arg):
            nonlocal seen
            if event == 'line':
                seen += 1
                if seen == line:
                    request.request(signal.SIGTERM, None)
            return tracer

        sys.settrace(tracer)
        try:
            request.wait(0.002)
        finally:
            sys.settrace(None)
        return seen

    total = wait_with_request_at(0)
    assert total > 0
    for line in range(1, total + 1):
        thread = threading.Thread(target=wait_with_request_at, args=(line,), daemon=True)
        thread.start()
        thread.join(5)
        assert not thread.is_alive(), f'the handler blocked at line event {line} of {total}'


def test_wait_ends_at_the_stop_request():
    request = daemon.StopRequest(step=0.01)
    assert request.wait(0.05) is False and request.asked() is False
    threading.Timer(0.1, request.request, args=(signal.SIGTERM, None)).start()
    begin = time.monotonic()
    assert request.wait(30) is True and time.monotonic() - begin < 5
    assert request.asked() is True and request.wait(30) is True


def test_factory_names():
    assert daemon.load_factory('daemon_fake:build').__name__ == 'build'
    for factory in ('daemon_fake', ':build', 'daemon_fake:'):
        with pytest.raises(DaemonError, match='is not "module:function"'):
            daemon.load_factory(factory)
