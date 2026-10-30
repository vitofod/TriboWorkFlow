# Author: Vito Foderà
# Created: 2026-10-29

"""The orchestrator of a run in the background, on the login node.

``start`` launches the orchestrator of a run as a process of its own,
detached from the terminal (a new session, the standard output and error in
``<run>/orchestrator.log``), and returns once that process has confirmed its
start or has ended. ``status`` reads ``<run>/orchestrator.pid`` and tells
whether the process runs; ``stop`` sends it SIGTERM: the orchestrator ends
the pass under way, saves its state and exits, removing the file. The jobs
already submitted go on; a later start recovers them.

The start goes in two steps. ``start`` creates the pid file with ``O_EXCL``,
so that a run has one instance: it holds the host, the start time, the
command and a token of this start, but no pid yet. The background process
handles SIGTERM, builds the orchestrator and only then writes its own pid in
that file, provided the file still holds its token; otherwise it exits
without running. ``start`` waits for that pid. A process that ends first, or
does not confirm its start within ``timeout`` seconds and is then killed,
leaves no pid file behind, and ``start`` reports it.

A pid file left by a process that no longer runs on this host is replaced,
with a warning; one written on another host is never touched, since whether
its process runs cannot be seen from here. A pid that now belongs to another
program (its command line, where the system shows it, does not run this
module) counts as a process that no longer runs.

The orchestrator is built in the background process by a factory, given as
``"module:function"`` and called with the directory of the run; it returns
an object with the ``run`` method of ``Orchestrator``. Before anything
starts, the submission command of the scheduler is looked for
(``require_scheduler``): away from the HPC machine nothing starts.

The handler of SIGTERM only records the request (``StopRequest``). Python
runs it in the main thread between two instructions, also while that thread
holds a lock or writes to the log, so the handler takes no lock and writes
nothing. The orchestrator sees the request after each pass and while it
waits between two passes, which it does in short steps.

Notes
-----
As in SCS (``example/Complete_setups/DP/start_scs.sh``, ``stop_scs.sh``,
``is_running.sh``, CC BY 4.0): the orchestrator started in the background as
with ``nohup``, its pid in a file, stopped with a signal, checked from the pid.
Unlike SCS the pid file also holds the host, the time and the command, a
second instance is refused, the process confirms its start, and SIGTERM is
handled so that the state is saved at the end of the pass.
"""

import argparse
import datetime
import importlib
import json
import logging
import os
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from triboworkflow.core.guard import require_scheduler
from triboworkflow.core.store.atomic import write_json

logger = logging.getLogger(__name__)

PID_FILE = 'orchestrator.pid'
LOG_FILE = 'orchestrator.log'
MODULE = 'triboworkflow.execution.daemon'

# Seconds start() waits for the process to confirm its start.
START_TIMEOUT = 120.0
# Longest sleep, in seconds, between two checks of a stop request.
STOP_STEP = 0.2


class DaemonError(RuntimeError):
    """An orchestrator that cannot be started, stopped or found."""


@dataclass(frozen=True)
class DaemonStatus:
    """What the pid file of a run tells.

    Attributes
    ----------
    running : bool or None
        True if the process runs, False if it does not (no pid file, or one
        left by a process gone), None if it cannot be told from here (another
        host, or a start not yet confirmed).
    stale : bool
        True for a pid file left by a process that no longer runs here.
    pid : int or None
    host, since, command : str or None
        As written in the pid file.
    message : str
        The status in words.
    """

    running: bool | None
    stale: bool
    pid: int | None
    host: str | None
    since: str | None
    command: str | None
    message: str


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def _alive(pid):
    """Whether process ``pid`` runs this module, as far as this host shows it."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # it exists, owned by another user
    try:
        cmdline = Path(f'/proc/{pid}/cmdline').read_bytes()
    except OSError:
        return True  # the command line is not shown here
    return MODULE.encode() in cmdline


def _read(path):
    """Content of a pid file, None if it is missing or not readable."""
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _remove_own(path, token):
    """Remove the pid file if it holds ``token``: never the file of another start."""
    data = _read(path)
    if isinstance(data, dict) and data.get('token') == token:
        path.unlink(missing_ok=True)


def _running(pid, host, since, command):
    return DaemonStatus(True, False, pid, host, since, command,
                        f'running: pid {pid} on {host} since {since}')


def status(run_dir):
    """Status of the orchestrator of a run, from its pid file.

    Returns
    -------
    DaemonStatus
    """
    path = Path(run_dir) / PID_FILE
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return DaemonStatus(False, False, None, None, None, None, f'not running: no {path}')
    except (OSError, ValueError) as error:
        return DaemonStatus(None, False, None, None, None, None, f'{path} not readable: {error}')
    pid, host = data.get('pid'), data.get('host')
    fields = dict(pid=pid, host=host, since=data.get('since'), command=data.get('command'))
    if host != socket.gethostname():
        return DaemonStatus(None, False, **fields,
                            message=f'{path} written on host {host}: check there whether pid '
                                    f'{pid} runs')
    if type(pid) is not int:
        return DaemonStatus(None, False, **fields,
                            message=f'{path} without a pid: the orchestrator is starting, or its '
                                    'start stopped before confirming it')
    if _alive(pid):
        return _running(**fields)
    return DaemonStatus(False, True, **fields,
                        message=f'not running: pid {pid} of {path} no longer runs on {host}')


def start(run_dir, factory, scheduler, python=None, timeout=START_TIMEOUT, poll=0.05):
    """Start the orchestrator of a run in the background.

    Parameters
    ----------
    run_dir : path-like
        Directory of the run.
    factory : str
        ``"module:function"`` building the orchestrator from the directory.
    scheduler : str
        Scheduler of ``[execution]``, whose submission command must be found.
    python : str or None, optional
        Interpreter of the background process; the one running this code by
        default.
    timeout : float, optional
        Seconds to wait for the process to confirm its start; past them it is
        killed.
    poll : float, optional
        Seconds between two checks while waiting.

    Returns
    -------
    DaemonStatus
        Of the process started: running once it has confirmed its start, or
        not running if it ended with code 0 before its confirmation was seen
        (an orchestrator with nothing left to do).

    Raises
    ------
    NotOnHPCError
        If the submission command of the scheduler is not found.
    DaemonError
        If an orchestrator of the run may be running, or the process started
        exited with an error or did not confirm its start in time.
    """
    require_scheduler(scheduler)
    run_dir = Path(run_dir).resolve()
    path = run_dir / PID_FILE
    log_path = run_dir / LOG_FILE
    token = secrets.token_hex(8)
    command = [python or sys.executable, '-m', MODULE, 'serve', str(run_dir), factory, token]
    shown = ' '.join(shlex.quote(part) for part in command)
    owner = {'pid': None, 'host': socket.gethostname(), 'since': _now(), 'command': shown,
             'token': token}
    for _ in range(2):
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            break
        except FileExistsError:
            found = status(run_dir)
            if not found.stale:
                raise DaemonError(f'orchestrator not started: {found.message}') from None
            logger.warning('%s; the pid file is replaced', found.message)
            path.unlink(missing_ok=True)
    else:
        raise DaemonError(f'orchestrator not started: {path} was created again meanwhile')
    with os.fdopen(fd, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps(owner) + '\n')
    try:
        with open(log_path, 'ab') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, cwd=run_dir,
                                       start_new_session=True)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    deadline = time.monotonic() + timeout
    while process.poll() is None:
        data = _read(path)
        if isinstance(data, dict) and data.get('token') == token \
                and data.get('pid') == process.pid:
            logger.info('Orchestrator of %s started: pid %d', run_dir, process.pid)
            return _running(process.pid, data.get('host'), data.get('since'), data.get('command'))
        if time.monotonic() >= deadline:
            process.kill()
            process.wait()
            _remove_own(path, token)
            raise DaemonError(f'orchestrator not started: pid {process.pid} did not confirm its '
                              f'start within {timeout} s and was killed; see {log_path}')
        time.sleep(poll)
    _remove_own(path, token)
    if process.returncode != 0:
        raise DaemonError(f'orchestrator not running: pid {process.pid} exited with code '
                          f'{process.returncode} at its start; see {log_path}')
    logger.info('Orchestrator of %s started and done: pid %d exited with code 0', run_dir,
                process.pid)
    return DaemonStatus(False, False, process.pid, owner['host'], owner['since'], shown,
                        f'not running: pid {process.pid} exited with code 0 at its start, '
                        f'nothing left to do; see {log_path}')


def stop(run_dir, timeout=None, poll=0.2):
    """Ask the orchestrator of a run to stop at the end of its pass (SIGTERM).

    Parameters
    ----------
    run_dir : path-like
        Directory of the run.
    timeout : float or None, optional
        Seconds to wait for the process to exit; it is not waited for by
        default.
    poll : float, optional
        Seconds between two checks while waiting.

    Returns
    -------
    bool
        Whether the process exited (always False without ``timeout``).

    Raises
    ------
    DaemonError
        If no orchestrator of the run runs on this host.
    """
    found = status(run_dir)
    if found.running is not True:
        raise DaemonError(f'nothing to stop: {found.message}')
    os.kill(found.pid, signal.SIGTERM)
    logger.info('SIGTERM sent to the orchestrator of %s (pid %d)', run_dir, found.pid)
    if timeout is None:
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(found.pid):
            return True
        time.sleep(poll)
    return not _alive(found.pid)


class StopRequest:
    """A stop asked by SIGTERM, recorded by a handler that takes no lock.

    ``request`` is the handler and only sets an attribute. Python runs it in
    the main thread between two instructions, also while that thread holds a
    lock or writes to the log: ``threading.Event.set`` there would wait
    forever for the lock that ``Event.wait`` holds, and a message would be
    written in the middle of another. The main thread waits in steps of
    ``step`` seconds and logs the request when it first sees it.

    Parameters
    ----------
    step : float, optional
        Longest sleep, in seconds, between two checks of the request.
    """

    def __init__(self, step=STOP_STEP):
        self.step = step
        self._asked = False
        self._logged = False

    def request(self, signum, frame):
        """Handler of SIGTERM: record the request."""
        self._asked = True

    def asked(self):
        """Whether a stop was asked; logged the first time it is seen."""
        if self._asked and not self._logged:
            self._logged = True
            logger.info('SIGTERM received: stopping at the end of the pass')
        return self._asked

    def wait(self, seconds):
        """Sleep ``seconds``, less if a stop is asked meanwhile.

        Returns
        -------
        bool
            Whether a stop was asked.
        """
        deadline = time.monotonic() + seconds
        while not self.asked():
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            time.sleep(min(left, self.step))
        return True


def load_factory(factory):
    """Function named by ``"module:function"``."""
    module, sep, name = factory.partition(':')
    if not sep or not module or not name:
        raise DaemonError(f'factory {factory!r} is not "module:function"')
    return getattr(importlib.import_module(module), name)


def serve(run_dir, factory, token):
    """Body of the background process: run the orchestrator until it is done or stopped.

    SIGTERM is handled first; then the orchestrator is built and the pid of
    this process written in the pid file, which confirms the start to
    ``start``. The pid file is removed at the end, if it still holds ``token``.

    Raises
    ------
    DaemonError
        If the pid file does not hold ``token``: it is not the one of this
        start, and the orchestrator does not run.
    """
    run_dir = Path(run_dir)
    path = run_dir / PID_FILE
    request = StopRequest()
    signal.signal(signal.SIGTERM, request.request)
    try:
        orchestrator = load_factory(factory)(run_dir)
        data = _read(path)
        if not isinstance(data, dict) or data.get('token') != token:
            raise DaemonError(f'{path} is not the one of this start: the orchestrator does not '
                              'run')
        write_json(path, {**data, 'pid': os.getpid()})
        logger.info('Orchestrator of %s running: pid %d', run_dir, os.getpid())
        report = orchestrator.run(sleep=request.wait, stop=request.asked)
        logger.info('Orchestrator %s', 'stopped by a signal' if request.asked() else 'done')
        return report
    finally:
        _remove_own(path, token)


def main(argv=None):
    """Entry point of the background process: ``python -m ... serve <run> <factory> <token>``."""
    parser = argparse.ArgumentParser(description='Orchestrator of a run in the background.')
    sub = parser.add_subparsers(dest='action', required=True)
    serve_parser = sub.add_parser('serve', help='run the orchestrator (started by start())')
    serve_parser.add_argument('run_dir')
    serve_parser.add_argument('factory', help='"module:function" building the orchestrator')
    serve_parser.add_argument('token', help='token of the pid file written by start()')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    serve(args.run_dir, args.factory, args.token)
    return 0


if __name__ == '__main__':
    sys.exit(main())
