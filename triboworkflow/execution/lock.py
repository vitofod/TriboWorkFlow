# Author: Vito Foderà
# Created: 2026-10-30

"""Lock of a run: one process at a time changes it.

The state of a run is in files that the code reads once and writes back
whole: ``orchestrator.json`` (state of the steps), ``jobs.json`` (registry of
the groups and jobs), and the groups, archived attempts and prepared
calculations under ``jobs/`` and ``systems/``. Two processes that load them
together lose the changes of the first that saves, or submit the same work
twice. Every command that changes the run, and the orchestrator in the
background for its whole life, therefore holds ``<run>/run.lock`` before
loading anything; a second holder is refused, with the name of the first.

The lock is created with ``O_EXCL`` and holds the pid, the host, the start
time, the command and a token. A lock left by a process that no longer runs
on this host is replaced, with a warning; one written on another host, or one
that cannot be read, is never touched, since whether its process runs cannot
be seen from here: the message says to remove it by hand after checking. A
pid that now belongs to another program (its command line, where the system
shows it, does not run this package) counts as a process that no longer runs.
A holder removes the lock only while it still holds its token.

A lock is replaced only by the process holding ``<run>/run.lock.takeover``,
also created with ``O_EXCL``: under it the lock is read again and removed
only if it is still the one left, then created anew with ``O_EXCL``. Two
processes that find the same lock left therefore never both hold the run. A
takeover file left by a process stopped while replacing the lock is never
removed here: while it is there a lock left is not replaced, and the message
says to remove it by hand after checking; a free run is taken as usual.

``RunLock.adopt`` hands a lock over to another process: the orchestrator in
the background takes the lock that ``daemon.start`` created for it with the
token of its start, and writes its own pid in it, holding the takeover file.

Examples
--------
>>> with RunLock(run_dir, 'triboworkflow relaunch pes_scf/aa'):    # doctest: +SKIP
...     orchestrator = build_orchestrator(run_dir)
...     orchestrator.relaunch(system, 'pes_scf/aa')
"""

import contextlib
import datetime
import json
import logging
import os
import secrets
import socket
import time
from pathlib import Path

from triboworkflow.core.store.atomic import write_json

logger = logging.getLogger(__name__)

LOCK_FILE = 'run.lock'
# Held by the one process that replaces or rewrites the lock.
TAKEOVER_FILE = 'run.lock.takeover'
# Seconds a process waits for another one to end its replacement of the lock.
TAKEOVER_WAIT = 1.0
# Text found in the command line of every process of the package.
PACKAGE = 'triboworkflow'
FIELDS = ('pid', 'host', 'since', 'command', 'token')


class RunLockError(RuntimeError):
    """A run held by another process, or a lock that is not this process's."""


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def _alive(pid):
    """Whether process ``pid`` runs this package, as far as this host shows it."""
    if pid == os.getpid():
        return True
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
    return PACKAGE.encode() in cmdline


def _read(path):
    """Content of a lock: dict, None if missing, ``{}`` if not readable."""
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _stale(data):
    return data.get('host') == socket.gethostname() and type(data.get('pid')) is int \
        and not _alive(data['pid'])


def holder(run_dir):
    """Holder of the lock of a run.

    Parameters
    ----------
    run_dir : path-like

    Returns
    -------
    dict or None
        ``{'pid', 'host', 'since', 'command', 'token', 'stale'}`` (values None
        for a lock that cannot be read); ``stale`` is True for a lock left by
        a process that no longer runs on this host. None if the run is free.
    """
    data = _read(Path(run_dir) / LOCK_FILE)
    if data is None:
        return None
    return {**{key: data.get(key) for key in FIELDS}, 'stale': _stale(data)}


def _refusal(path, data):
    """Message of a lock held by another process."""
    if not data:
        return (f'{path} not readable: a command of the run may be starting, or one stopped '
                f'while writing it; remove {path} by hand only after checking that no command '
                'of the run runs')
    who = f'{data.get("command")} (pid {data.get("pid")} on {data.get("host")} since ' \
          f'{data.get("since")})'
    if data.get('host') != socket.gethostname():
        return (f'the run is in use by {who}, on another host: remove {path} by hand only after '
                'checking that no command of the run runs there')
    return (f'the run is in use by {who}: wait for it to end; the orchestrator in the '
            'background ends with "triboworkflow stop"')


@contextlib.contextmanager
def _takeover(run_dir, poll=0.05):
    """Hold ``<run>/run.lock.takeover`` while replacing or rewriting the lock of a run.

    The file is created with ``O_EXCL``, so one process at a time changes a
    lock it does not create. One held for more than ``TAKEOVER_WAIT`` seconds
    belongs to a process that stopped while replacing the lock: it is never
    removed here, since whether that process runs is not known.

    Raises
    ------
    RunLockError
        If the file is still there after ``TAKEOVER_WAIT`` seconds.
    """
    path = Path(run_dir) / TAKEOVER_FILE
    deadline = time.monotonic() + TAKEOVER_WAIT
    while True:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RunLockError(
                    f'{path} exists: another command of the run is replacing its lock, or one '
                    f'stopped while doing it; remove {path} by hand only after checking that '
                    'no command of the run runs') from None
            time.sleep(poll)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            fh.write(json.dumps({'pid': os.getpid(), 'host': socket.gethostname(),
                                 'since': _now()}) + '\n')
        yield
    finally:
        path.unlink(missing_ok=True)


class RunLock:
    """Lock of a run, held by one process at a time.

    Parameters
    ----------
    run_dir : path-like
        Directory of the run.
    command : str
        Command of the holder, shown to a process refused.
    token : str or None, optional
        Token of this holder; a new one by default.

    Attributes
    ----------
    path : pathlib.Path
        ``<run>/run.lock``.
    token : str

    Notes
    -----
    Use it as a context manager, or call ``acquire`` and ``release``.
    """

    def __init__(self, run_dir, command, token=None):
        self.path = Path(run_dir) / LOCK_FILE
        self.command = command
        self.token = token or secrets.token_hex(8)

    def acquire(self):
        """Create the lock.

        Returns
        -------
        RunLock
            This lock.

        Raises
        ------
        RunLockError
            If another process holds the run, or a lock cannot be read.
        """
        owner = {'pid': os.getpid(), 'host': socket.gethostname(), 'since': _now(),
                 'command': self.command, 'token': self.token}
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
                break
            except FileExistsError:
                data = _read(self.path)
                if data is None:
                    continue  # removed meanwhile
                if not _stale(data):
                    raise RunLockError(_refusal(self.path, data)) from None
                with _takeover(self.path.parent):
                    # Read again: another process may have replaced it meanwhile.
                    if _read(self.path) == data:
                        logger.warning('%s left by %s: pid %s no longer runs on %s; the lock is '
                                       'replaced', self.path, data.get('command'), data.get('pid'),
                                       data.get('host'))
                        self.path.unlink(missing_ok=True)
        else:
            raise RunLockError(f'{self.path} was created again meanwhile: another command of '
                               'the run is starting')
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            fh.write(json.dumps(owner) + '\n')
        return self

    def release(self):
        """Remove the lock if it still holds the token of this holder."""
        data = _read(self.path)
        if data and data.get('token') == self.token:
            self.path.unlink(missing_ok=True)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False

    @classmethod
    def adopt(cls, run_dir, token):
        """Take over the lock created for this process with ``token``.

        The pid of this process replaces the one of the creator.

        Parameters
        ----------
        run_dir : path-like
        token : str
            Token given to the lock by its creator.

        Returns
        -------
        RunLock

        Raises
        ------
        RunLockError
            If the lock is missing or holds another token, or another
            process is replacing it.
        """
        lock = cls(run_dir, None, token)
        with _takeover(run_dir):
            data = _read(lock.path)
            if not data or data.get('token') != token:
                raise RunLockError(f'{lock.path} is not the one created for this process')
            lock.command = data.get('command')
            write_json(lock.path, {**data, 'pid': os.getpid()})
        return lock
