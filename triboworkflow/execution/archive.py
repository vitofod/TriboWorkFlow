# Author: Vito Foderà
# Created: 2026-10-27

"""Archiving of the last attempt of a calculation, before a new one.

An attempt leaves in the calculation directory its claim, ``run.pwi`` and
``output.pwo`` (``worker.py``). Before the calculation can run again, the
attempt is archived:

1. the number ``n`` of the attempt comes from the claim; an empty claim (a
   worker stopped right after creating it) gets the number the worker would
   have used, the attempts in the record plus one, written in the claim;
2. a record without attempt ``n`` (a worker stopped before writing it) gets
   it, with what the claim tells (job, host, time) and the rest null;
3. ``run.pwi`` and ``output.pwo`` are moved to ``attempts/<n>/`` with
   ``os.replace``, after checking them against the digests recorded;
4. attempt ``n`` in the record gets the archived paths and their digests;
5. the claim is removed, last.

Each step can be repeated, so an archiving interrupted at any point is
completed by running it again, with nothing lost and the claim kept until
the record is up to date. A file found both in the calculation directory
and in the archive is an error, and nothing is moved or deleted. ``OUT/`` is
not touched: what to keep of it for a restart is decided elsewhere.

The caller makes sure that no worker is running the attempt. An attempt
still recorded as ``running`` is not archived: its state is set first, from
the scheduler.
"""

import dataclasses
import json
import os

from triboworkflow.core.store import Attempt, AttemptFile, file_digest, read_record, write_record
from triboworkflow.core.store.atomic import write_json

from .layout import (ATTEMPTS_DIR, CLAIM_FILE, OUTPUT_FILE, RUN_INPUT_FILE, attempt_dir,
                     calc_dir, system_dir)

_CLAIM_KEYS = ('n', 'job_id', 'host', 'pid', 'since')


class ArchiveError(RuntimeError):
    """An attempt whose files or record do not allow archiving it safely."""


def read_claim(path):
    """Content of a claim file, or None for an empty or unreadable claim."""
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or type(data.get('n')) is not int or data['n'] < 1:
        return None
    return {key: data.get(key) for key in _CLAIM_KEYS}


def _check_digest(path, recorded, where):
    if recorded is not None and file_digest(path) != recorded:
        raise ArchiveError(f'{path}: its digest differs from the one recorded for {where}; '
                           'nothing was moved')


def archive_attempt(run_dir, system, calc_id):
    """Archive the last attempt of a calculation and free it for a new one.

    Parameters
    ----------
    run_dir : path-like
        Directory of the run.
    system, calc_id : str
        The calculation.

    Returns
    -------
    int or None
        Number of the attempt archived; None when there is no claim and no
        file of an attempt, i.e. nothing to archive.

    Raises
    ------
    ArchiveError
        If the attempt is still recorded as running, the files are not
        those recorded, a file is both in place and in the archive, there
        are files of an attempt without a claim, or the claim and the record
        disagree on the number of the attempt.
    """
    path = calc_dir(run_dir, system, calc_id)
    sdir = system_dir(run_dir, system)
    claim_path = path / CLAIM_FILE
    if not claim_path.exists():
        left = [name for name in (RUN_INPUT_FILE, OUTPUT_FILE) if (path / name).exists()]
        if left:
            raise ArchiveError(f'{path}: {", ".join(left)} without a claim: the attempt they '
                               'belong to is unknown')
        return None

    record = read_record(sdir, calc_id)
    # 1. Number of the attempt.
    claim = read_claim(claim_path)
    if claim is None:
        claim = dict.fromkeys(_CLAIM_KEYS)
        claim['n'] = len(record.attempts) + 1
        write_json(claim_path, claim)
    n = claim['n']

    # 2. Attempt n in the record.
    if len(record.attempts) == n - 1:
        attempt = Attempt(n=n, job_id=claim['job_id'], host=claim['host'],
                          started=claim['since'])
        record = dataclasses.replace(record, attempts=record.attempts + (attempt,))
        write_record(sdir, record)
    elif len(record.attempts) != n:
        raise ArchiveError(f'{path}: the claim is for attempt {n}, the record has '
                           f'{len(record.attempts)} attempts')
    attempt = record.attempts[-1]
    if attempt.status.process == 'running':
        raise ArchiveError(f'{path}: attempt {n} is recorded as running; set its state from '
                           'the job before archiving it')

    # 3. Files to the archive, once all of them are checked.
    archive = attempt_dir(path, n)
    recorded = {RUN_INPUT_FILE: attempt.input, OUTPUT_FILE: attempt.output}
    for name in (RUN_INPUT_FILE, OUTPUT_FILE):
        source, target = path / name, archive / name
        if source.exists() and target.exists():
            raise ArchiveError(f'{source} and {target} both exist; nothing was moved or deleted')
        digest = recorded[name].digest if recorded[name] is not None else None
        for found in (source, target):
            if found.exists():
                _check_digest(found, digest, f'attempt {n}')
    for name in (RUN_INPUT_FILE, OUTPUT_FILE):
        if (path / name).exists():
            archive.mkdir(parents=True, exist_ok=True)
            os.replace(path / name, archive / name)

    # 4. Archived files in the record.
    files = {}
    for name in (RUN_INPUT_FILE, OUTPUT_FILE):
        target = archive / name
        files[name] = (AttemptFile(f'{ATTEMPTS_DIR}/{n}/{name}', file_digest(target))
                       if target.exists() else None)
    done = dataclasses.replace(attempt, input=files[RUN_INPUT_FILE], output=files[OUTPUT_FILE])
    if done != attempt:
        write_record(sdir, dataclasses.replace(record, attempts=record.attempts[:-1] + (done,)))

    # 5. The calculation is free.
    os.unlink(claim_path)
    return n
