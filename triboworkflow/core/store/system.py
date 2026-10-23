# Author: Vito Foderà
# Created: 2026-10-23

"""Aggregated document of a system, provenance of its entries, single writer.

``system.json`` holds::

    {"schema_version": 1,
     "document": {...},
     "provenance": [{"path": [...], "step": "...", "parameters": {...},
                     "calculations": {...}, "inputs": {...}, "files": {...},
                     "results": {...}, "fingerprint": "sha256:..."}]}

``document`` is the aggregated data of the system, with the historic keys of
the workflows (``structure``, ``comp_params``, ``inter_params``, ``highsym``,
``pes``, ``adhesion``, ...); it is null until the document is created.
``provenance`` tells, for each entry written by an analysis, what the entry
was computed from (``Provenance``).

Updates merge leaves. Setting a dict at a path sets each of its leaves
(values that are not dicts; lists are leaves) and keeps the other keys met on
the way, so an empty dict changes nothing; setting a leaf replaces what was at
its path. Writing an entry drops the provenance of the entries it overlaps
(the same path, a parent or a child): an entry is valid again only once its
step records a new provenance.

An entry is done (``is_done``) when it exists, is not empty and its recorded
provenance is the one expected now, with every calculation it used still
backed by a complete record with the same fingerprint and the same results it
consumed. Changing a calculation or its results, an input read from another
entry or a file, or a processing parameter (an interpolation setting, a MEP
setting, a seed) therefore invalidates the analysis that used it; nothing is
deleted. Since inputs count by content, the entries computed from it are
invalidated as soon as it is computed again with a different result.

Content alone cannot tell whether an entry read by an analysis is up to date
with the current settings of the step that computes it. An analysis therefore
states the entries it reads as prerequisites, with the provenance their steps
expect now, and ``check_prerequisites`` stops it when one is missing or not
current. Nothing is recomputed in cascade: the user, or the orchestrator, runs
the step that computes the prerequisite.

One process at a time writes ``system.json`` (``SystemWriter``, with a lock
file created with ``O_EXCL``), whichever phase it runs; readers call
``read_system`` at any time and see the last saved version, never a partial
one.
"""

import copy
import dataclasses
import datetime
import json
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path

from .atomic import StoreError, check_json_value, read_json, write_json
from .fingerprint import (canonical_json, check_fingerprint, content_digest, file_digest,
                          fingerprint)
from .record import can_reuse, check_keys, check_name, read_record, split_calc_id

SCHEMA_VERSION = 1
SYSTEM_FILE = 'system.json'
LOCK_FILE = 'system.json.lock'

_SYSTEM_KEYS = ('schema_version', 'document', 'provenance')
_PROVENANCE_KEYS = ('path', 'step', 'parameters', 'calculations', 'inputs', 'files',
                    'results', 'fingerprint')


class LockError(StoreError):
    """``system.json`` is locked by another writer, or by a stale lock file."""


class PrerequisiteError(StoreError):
    """An entry that an analysis needs is missing or not current."""


def entry_path(path):
    """Path of an entry as a tuple of keys.

    Parameters
    ----------
    path : str or sequence of str
        A top-level key, or the keys from the top level down; ``()`` is the
        whole document.

    Returns
    -------
    tuple of str
    """
    if isinstance(path, str):
        return (path,)
    if isinstance(path, (list, tuple)) and all(isinstance(key, str) for key in path):
        return tuple(path)
    raise StoreError(f'{path!r} is not a key or a sequence of keys')


def _show(path):
    return '.'.join(path) if path else '<document>'


def _overlap(a, b):
    n = min(len(a), len(b))
    return a[:n] == b[:n]


@dataclass(frozen=True)
class Provenance:
    """What an entry of the document was computed from.

    Parameters
    ----------
    step : str
        Step that computed the entry.
    parameters : dict
        Processing parameters of the analysis (e.g. interpolation and MEP
        settings, seeds), stored in full so that a change can be read.
    calculations : dict
        {calc_id: fingerprint} of the calculations whose results are used.
    inputs : dict
        {name: digest} of the data read from other entries.
    files : dict
        {name: digest} of the files read.
    results : dict, optional
        {calc_id: digest of the results} actually consumed, for every
        calculation in ``calculations``; empty in an expected provenance,
        which is known before the results.

    Notes
    -----
    Use ``Provenance.build`` for the provenance expected now, which computes
    the digests from the data and the files themselves, and ``with_results``
    for the provenance recorded after the analysis. Names of inputs and files
    are chosen by the step; only the digests count, so a file moved elsewhere
    with the same content gives the same provenance.
    """

    step: str
    parameters: dict
    calculations: dict
    inputs: dict
    files: dict
    results: dict = field(default_factory=dict)

    def __post_init__(self):
        check_name(self.step, 'provenance.step')
        if not isinstance(self.parameters, dict):
            raise StoreError('provenance.parameters: a JSON object is required')
        check_json_value(self.parameters, 'provenance.parameters')
        canonical_json(self.parameters, 'provenance.parameters')
        for name in ('calculations', 'inputs', 'files', 'results'):
            mapping = getattr(self, name)
            if not isinstance(mapping, dict):
                raise StoreError(f'provenance.{name}: a JSON object is required')
            for key, digest in mapping.items():
                if name in ('calculations', 'results'):
                    split_calc_id(key)
                elif not isinstance(key, str) or not key:
                    raise StoreError(f'provenance.{name}: {key!r} is not a name')
                check_fingerprint(digest, f'provenance.{name}[{key!r}]')
            object.__setattr__(self, name, dict(mapping))
        unknown = sorted(set(self.results) - set(self.calculations))
        if unknown:
            raise StoreError(f'provenance.results: {", ".join(unknown)} not in calculations')
        object.__setattr__(self, 'parameters', copy.deepcopy(self.parameters))

    @classmethod
    def build(cls, step, parameters=None, calculations=None, inputs=None, files=None):
        """Provenance of an entry from what it was computed from.

        Parameters
        ----------
        step : str
            Step that computes the entry.
        parameters : dict, optional
            Processing parameters of the analysis.
        calculations : dict, optional
            {calc_id: fingerprint expected now} of the calculations used.
        inputs : dict, optional
            {name: data} read from other entries; digested here.
        files : dict, optional
            {name: path} of the files read; digested here.

        Returns
        -------
        Provenance
            Without results: the provenance expected now.
        """
        return cls(step=step, parameters=parameters or {},
                   calculations=dict(calculations or {}),
                   inputs={name: content_digest(value, f'inputs[{name!r}]')
                           for name, value in (inputs or {}).items()},
                   files={name: file_digest(path) for name, path in (files or {}).items()})

    def with_results(self, records):
        """The same provenance with the digests of the results consumed.

        Parameters
        ----------
        records : iterable of Record
            Records read by the analysis, exactly one for each calculation in
            ``calculations``.

        Returns
        -------
        Provenance

        Raises
        ------
        StoreError
            If the records are not those of ``calculations``, or one of them
            cannot be reused with its expected fingerprint.
        """
        results = {}
        for record in records:
            expected = self.calculations.get(record.calc_id)
            if expected is None or record.calc_id in results:
                raise StoreError(f'record {record.calc_id}: not a calculation of this '
                                 'provenance, or given twice')
            if not can_reuse(record, expected):
                raise StoreError(f'record {record.calc_id}: not complete, without results or '
                                 'with another fingerprint; its results cannot be consumed')
            results[record.calc_id] = record.results_digest
        missing = sorted(set(self.calculations) - set(results))
        if missing:
            raise StoreError(f'no record given for {", ".join(missing)}')
        return dataclasses.replace(self, results=results)

    @property
    def fingerprint(self):
        """Fingerprint of what the entry is computed from (results excluded)."""
        return fingerprint({'step': self.step, 'parameters': self.parameters,
                            'calculations': self.calculations, 'inputs': self.inputs,
                            'files': self.files})

    def to_dict(self, path):
        """JSON object stored in ``system.json`` for the entry at ``path``."""
        return {'path': list(path), 'step': self.step,
                'parameters': copy.deepcopy(self.parameters),
                'calculations': dict(self.calculations), 'inputs': dict(self.inputs),
                'files': dict(self.files), 'results': dict(self.results),
                'fingerprint': self.fingerprint}

    @classmethod
    def from_dict(cls, data):
        """Path and provenance from their JSON object; the fingerprint is checked."""
        check_keys(data, _PROVENANCE_KEYS, 'provenance')
        path = entry_path(data['path'])
        provenance = cls(step=data['step'], parameters=data['parameters'],
                         calculations=data['calculations'], inputs=data['inputs'],
                         files=data['files'], results=data['results'])
        if provenance.fingerprint != data['fingerprint']:
            raise StoreError(f'provenance of {_show(path)}: the stored fingerprint does not '
                             'match its content')
        return path, provenance


class SystemDocument:
    """Aggregated document of a system and the provenance of its entries, in memory.

    Parameters
    ----------
    document : dict or None, optional
        Data of the system; None if the document does not exist yet.
    provenance : dict, optional
        {entry path: Provenance}.

    Notes
    -----
    Values are copied in and out: changing what ``get`` returns, or what was
    passed to ``set``, does not change the document.
    """

    def __init__(self, document=None, provenance=None):
        if document is not None:
            if not isinstance(document, dict):
                raise StoreError('document: a JSON object or null is required')
            check_json_value(document, 'document')
        self._document = copy.deepcopy(document)
        self._provenance = {}
        for path, item in (provenance or {}).items():
            if not isinstance(item, Provenance):
                raise StoreError('provenance: Provenance values are required')
            self._provenance[entry_path(path)] = item

    @property
    def exists(self):
        """True once the document has been created."""
        return self._document is not None

    @property
    def document(self):
        """Copy of the whole document, or None."""
        return copy.deepcopy(self._document)

    def _require(self):
        if self._document is None:
            raise StoreError('the system has no document yet: create it first')

    def get(self, path=()):
        """Copy of the value at a path, or None if a key is missing.

        Raises
        ------
        StoreError
            If the path goes through a value that is not a dict.
        """
        path = entry_path(path)
        value = self._document
        for i, key in enumerate(path):
            if value is None:
                return None
            if not isinstance(value, dict):
                raise StoreError(f'{_show(path[:i])} holds a {type(value).__name__}, not a dict: '
                                 f'{_show(path)} cannot be read')
            value = value.get(key)
        return copy.deepcopy(value)

    def create(self, document):
        """Create the document of the system.

        Raises
        ------
        StoreError
            If the document already exists or is not a JSON object.
        """
        if self._document is not None:
            raise StoreError('the system already has a document')
        if not isinstance(document, dict):
            raise StoreError('document: a JSON object is required')
        check_json_value(document, 'document')
        self._document = copy.deepcopy(document)
        self._provenance.clear()

    def set(self, path, value):
        """Merge a value into the document, leaf by leaf.

        Parameters
        ----------
        path : str or sequence of str
            Entry where the value goes; ``()`` for the top level, in which
            case the value must be a dict.
        value : object
            A dict sets each of its leaves below ``path`` (an empty dict sets
            nothing); any other value replaces what is at ``path``.

        Raises
        ------
        StoreError
            If the document does not exist, the value is not JSON, or a key
            on the way to a leaf holds a value that is not a dict (null
            included). Nothing is changed then.
        """
        path = entry_path(path)
        self._require()
        check_json_value(value, _show(path))
        if not path and not isinstance(value, dict):
            raise StoreError('only a dict can be merged into the whole document')
        leaves = []
        self._leaves(path, copy.deepcopy(value), leaves)
        # Leaves of one dict never lie below one another: checking them all
        # first makes the update all or nothing.
        for leaf, _ in leaves:
            self._check_parents(leaf)
        for leaf, item in leaves:
            self._put(leaf, item)
        self._drop_provenance([leaf for leaf, _ in leaves])

    def replace(self, path, value):
        """Put a value at a path as a whole, replacing what was there.

        Raises
        ------
        StoreError
            As ``set``; the path cannot be empty.
        """
        path = entry_path(path)
        self._require()
        if not path:
            raise StoreError('the whole document cannot be replaced: create it instead')
        check_json_value(value, _show(path))
        self._check_parents(path)
        self._put(path, copy.deepcopy(value))
        self._drop_provenance([path])

    def _leaves(self, path, value, leaves):
        if isinstance(value, dict):
            for key, item in value.items():
                self._leaves(path + (key,), item, leaves)
        else:
            leaves.append((path, value))

    def _check_parents(self, path):
        # Every key above the leaf is missing or holds a dict.
        parent = self._document
        for i, key in enumerate(path[:-1]):
            if key not in parent:
                return
            if not isinstance(parent[key], dict):
                raise StoreError(f'{_show(path[:i + 1])} holds a {type(parent[key]).__name__}, '
                                 f'not a dict: {_show(path)} cannot be set')
            parent = parent[key]

    def _put(self, path, value):
        parent = self._document
        for key in path[:-1]:
            parent = parent.setdefault(key, {})
        parent[path[-1]] = value

    def _drop_provenance(self, written):
        for stored in list(self._provenance):
            if any(_overlap(stored, path) for path in written):
                del self._provenance[stored]

    def provenance(self, path):
        """Provenance recorded for an entry, or None."""
        return self._provenance.get(entry_path(path))

    def set_provenance(self, path, provenance):
        """Record what an entry was computed from, after writing the entry.

        Parameters
        ----------
        path : str or sequence of str
            Entry written by the analysis.
        provenance : Provenance
            With the results consumed from every calculation
            (``Provenance.with_results``).

        Raises
        ------
        StoreError
            If the entry does not exist, or results of a calculation are
            missing from the provenance.
        """
        path = entry_path(path)
        if not isinstance(provenance, Provenance):
            raise StoreError('provenance: a Provenance is required')
        if not path or self.get(path) is None:
            raise StoreError(f'{_show(path)}: no such entry, its provenance cannot be recorded')
        missing = sorted(set(provenance.calculations) - set(provenance.results))
        if missing:
            raise StoreError(f'{_show(path)}: the results consumed from {", ".join(missing)} '
                             'are not in the provenance (Provenance.with_results)')
        self._provenance[path] = provenance

    def to_dict(self):
        """JSON object of ``system.json``."""
        return {'schema_version': SCHEMA_VERSION, 'document': copy.deepcopy(self._document),
                'provenance': [item.to_dict(path) for path, item in self._provenance.items()]}

    @classmethod
    def from_dict(cls, data, source=None):
        """Document from the JSON object of ``system.json``."""
        try:
            check_keys(data, _SYSTEM_KEYS, SYSTEM_FILE)
            version = data['schema_version']
            if type(version) is not int or version != SCHEMA_VERSION:
                raise StoreError(f'schema_version: {version!r}; this version reads '
                                 f'{SCHEMA_VERSION}')
            if not isinstance(data['provenance'], list):
                raise StoreError('provenance: a JSON array is required')
            provenance = dict(Provenance.from_dict(item) for item in data['provenance'])
            return cls(data['document'], provenance)
        except StoreError as err:
            raise StoreError(f'{source}: {err}' if source else str(err)) from None


def read_system(system_dir):
    """Last saved document of a system; an empty one if ``system.json`` is missing."""
    path = Path(system_dir) / SYSTEM_FILE
    try:
        data = read_json(path)
    except FileNotFoundError:
        return SystemDocument()
    return SystemDocument.from_dict(data, source=path)


class SystemWriter:
    """Exclusive writer of ``system.json``.

    Parameters
    ----------
    system_dir : path-like
        Directory of the system, created if missing.

    Notes
    -----
    Use it as a context manager. Entering creates the lock file
    ``system.json.lock`` with ``O_EXCL`` and loads the document into
    ``document``; leaving removes the lock. Changes reach the disk only with
    ``save``, as one atomic write: leaving without saving discards them.

    A lock left by a writer that stopped without leaving is never removed
    automatically: the error names the file and its owner, and the file is
    removed by hand once no writer is running.

    Examples
    --------
    >>> with SystemWriter(system_dir) as writer:                # doctest: +SKIP
    ...     writer.document.set(['pes', 'energy'], data)
    ...     writer.document.set_provenance(['pes', 'energy'], provenance)
    ...     writer.save()
    """

    def __init__(self, system_dir):
        self.system_dir = Path(system_dir)
        self.document = None
        self._locked = False

    @property
    def lock_path(self):
        """Lock file of the system."""
        return self.system_dir / LOCK_FILE

    def __enter__(self):
        self.system_dir.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            try:
                owner = self.lock_path.read_text(encoding='utf-8').strip() or 'unknown'
            except OSError:
                owner = 'unknown'
            raise LockError(f'{self.lock_path} exists (owner: {owner}): another process is '
                            f'writing {SYSTEM_FILE}, or a writer stopped without removing the '
                            'lock. Remove the file only after checking that no writer is '
                            'running.') from None
        self._locked = True
        try:
            owner = {'pid': os.getpid(), 'host': socket.gethostname(),
                     'since': datetime.datetime.now().astimezone().isoformat(timespec='seconds')}
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                fh.write(json.dumps(owner) + '\n')
            self.document = read_system(self.system_dir)
        except BaseException:
            self._release()
            raise
        return self

    def save(self):
        """Write ``document`` to ``system.json`` atomically."""
        if not self._locked:
            raise StoreError(f'{self.system_dir / SYSTEM_FILE} can be saved only inside the '
                             'writer context')
        write_json(self.system_dir / SYSTEM_FILE, self.document.to_dict())

    def _release(self):
        if self._locked:
            self._locked = False
            os.unlink(self.lock_path)

    def __exit__(self, *exc):
        self._release()
        return False


def entry_is_current(document, path, expected, system_dir):
    """Whether the provenance recorded for an entry is the one expected now.

    Parameters
    ----------
    document : SystemDocument
        Document of the system.
    path : str or sequence of str
        Entry.
    expected : Provenance
        Provenance the entry would have if computed now.
    system_dir : path-like
        Directory of the system, where the records are.

    Returns
    -------
    bool
        True if the recorded provenance equals ``expected`` and every
        calculation in it has a record that can be reused with its
        fingerprint and still holds the results that were consumed.
    """
    if not isinstance(expected, Provenance):
        raise StoreError('expected: a Provenance is required')
    stored = document.provenance(path)
    if stored is None or stored.fingerprint != expected.fingerprint:
        return False
    for calc_id, digest in expected.calculations.items():
        try:
            record = read_record(system_dir, calc_id)
        except FileNotFoundError:
            return False
        if not can_reuse(record, digest) or record.results_digest != stored.results[calc_id]:
            return False
    return True


def is_done(document, path, expected, system_dir):
    """Whether an entry exists, is not empty and is current (``entry_is_current``)."""
    return bool(document.get(path)) and entry_is_current(document, path, expected, system_dir)


def check_prerequisites(document, expected, system_dir):
    """Stop an analysis whose prerequisites are missing or not current.

    Parameters
    ----------
    document : SystemDocument
        Document of the system.
    expected : dict
        {entry: Provenance expected now by the step that computes the entry}
        for every entry the analysis reads; an entry is a key or a sequence
        of keys.
    system_dir : path-like
        Directory of the system, where the records are.

    Raises
    ------
    PrerequisiteError
        Listing the entries that are not done (``is_done``). Nothing is
        recomputed: the steps that compute them have to run first.
    """
    stale = [_show(entry_path(path)) for path, provenance in expected.items()
             if not is_done(document, path, provenance, system_dir)]
    if stale:
        raise PrerequisiteError(f'entries missing or not current: {", ".join(stale)}; run the '
                                'steps that compute them first')
