# Author: Vito Foderà
# Created: 2026-10-23

"""Historic database calls on top of the file store.

The ported workflow code reads and writes the aggregated document of a system
through four calls with the meaning of the historic document-database API:
``query_db``, ``update_db``, ``insert_db`` and ``is_done``. This module is the
only place that knows those calls; the rest of the package uses
``SystemDocument`` and the records directly.

The calls keep the historic arguments that have an effect. Those that select
a database server (``db_file``, ``database``), ``upsert`` and the log
``message`` have no counterpart in the store and are not accepted. Every
argument after the collection is keyword-only: a historic call that passes
them by position fails, instead of giving its values to other parameters, and
the ported call sites name their arguments.

Only the forms used by the interface workflows are accepted. Any other form
is an error, never a guess:

- ``fltr`` is always given. It selects a document by equality of its
  top-level fields, each with a value (never None). The store holds one
  document per system, so a filter that does not match the document of the
  system is an error, not an empty result; before the document exists,
  queries return nothing;
- the collection is either the one given for the system's documents or
  ``'tasks'``, a read-only view of the records of the system: one task
  document per complete record, found by ``task_label``, which is the
  ``calc_id`` of the record;
- updates use ``'$set'`` only. With ``to_mongodb=True`` (the default) the
  data are merged leaf by leaf, as dotted ``$set`` keys would merge them, so
  an empty dict changes nothing; with ``to_mongodb=False`` the top-level key
  of the entry is replaced as a whole;
- ``entry`` is a key, a list of nested keys, or a list of such lists with one
  value per list. Keys are strings without dots and not starting with ``$``,
  in entries, filters and data alike, because a document database would read
  them as paths or operators;
- ``insert_db`` creates the document of the system. Inserting again a
  document whose fields all equal those of the existing document does
  nothing, so a repeated step does not fail; any other insertion is an error,
  since a system has one document. Records and their attempts are never
  written through these calls.

Writes change the in-memory ``SystemDocument``; whoever holds the
``SystemWriter`` saves it. ``is_done`` applies the historic condition (the
entry exists and is not empty) and requires, in addition, the provenance
recorded for the entry to be the one expected now (``system.is_done``).

Notes
-----
To remove: the historic behaviour is that of TribChem's ``FireTaskTribChem``
methods (``highput/firetasks/core.py`` 163-368) over
``highput/database/query.py``, ``dbtools.py`` (``write_multiple_entry``,
``convert_dict_to_mongodb``) and ``navigator.py`` (``insert_data``, duplicate
check). The task documents follow atomate's ``tasks`` collection, from which
``FT_MoveTagResults`` copies the results. atomate's ``dir_name`` is
``<host>:<absolute path>``, which the charge step splits with
``split(':')[1]``; here it is the directory relative to the system, so that
code fails instead of reading a wrong path, and the port of charge reads the
density file from the record. ``FT_StartShearStrength.get_pes`` and
``get_main_folder`` (``run.py`` 1253-1257, 1613-1626) read the database
through ``Navigator`` directly, outside these calls (A47).
"""

import copy
import logging
from pathlib import Path

from triboworkflow.vendored.highput.database.dbtools import read_multiple_entry

from .atomic import StoreError, check_json_value
from .record import read_record
from .system import Provenance, SystemDocument, entry_is_current, entry_path

logger = logging.getLogger(__name__)

TASKS = 'tasks'
TASK_KEYS = ('task_label', 'task_id', 'dir_name')

_MISSING = object()


def _check_key(key, where):
    if not isinstance(key, str) or not key or '.' in key or key.startswith('$'):
        raise StoreError(f'{where}: {key!r} is not a key (a non-empty string without dots, not '
                         'starting with "$")')


def _check_keys(value, where):
    if isinstance(value, dict):
        for key, item in value.items():
            _check_key(key, where)
            _check_keys(item, f'{where}[{key!r}]')
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _check_keys(item, f'{where}[{i}]')


def _path(entry):
    # One entry: a key or a non-empty list of keys.
    if isinstance(entry, str):
        path = (entry,)
    elif isinstance(entry, list) and entry and all(isinstance(key, str) for key in entry):
        path = tuple(entry)
    else:
        raise StoreError(f'entry: {entry!r} is not a key or a list of keys')
    for key in path:
        _check_key(key, 'entry')
    return path


def _is_list_of_entries(entry):
    return isinstance(entry, list) and bool(entry) and all(isinstance(e, list) for e in entry)


def _check_entry(entry):
    # Raise unless entry is None, one entry or a list of entries.
    if _is_list_of_entries(entry):
        for e in entry:
            _path(e)
    elif entry is not None:
        _path(entry)


def _updates(data, entry):
    # (path, value) pairs of an update, as the historic call builds them.
    if _is_list_of_entries(entry):
        if not isinstance(data, list) or len(data) != len(entry):
            raise StoreError('entry is a list of lists: data must be a list with one value '
                             'per entry')
        return [(_path(e), value) for e, value in zip(entry, data)]
    if entry is None:
        if not isinstance(data, dict):
            raise StoreError('without entry, data must be a dict of top-level keys')
        return [((), data)]
    return [(_path(entry), data)]


def _same(stored, wanted, where):
    # Equality of a stored value with a wanted one, as a document filter tests it.
    if wanted is None:
        return stored is None or stored is _MISSING
    if stored is _MISSING:
        return False
    if isinstance(wanted, dict):
        return (isinstance(stored, dict) and list(stored) == list(wanted)
                and all(_same(stored[k], wanted[k], where) for k in wanted))
    if isinstance(wanted, (list, tuple)):
        return (isinstance(stored, list) and len(stored) == len(wanted)
                and all(_same(s, w, where) for s, w in zip(stored, wanted)))
    if isinstance(stored, list):
        raise StoreError(f'{where}: matching a single value against an array is not supported')
    if isinstance(wanted, bool) or isinstance(stored, bool):
        return type(stored) is type(wanted) and stored == wanted
    if isinstance(wanted, (int, float)) and isinstance(stored, (int, float)):
        return stored == wanted
    return type(stored) is type(wanted) and stored == wanted


class LegacyStore:
    """Historic database calls on the document and the records of one system.

    Parameters
    ----------
    system_dir : path-like
        Directory of the system, with its records.
    document : SystemDocument
        Document read and changed by the calls: ``SystemWriter.document``
        to save the changes, or ``read_system(system_dir)`` to read only.
    collection : str
        Name of the collection that the ported code uses for the documents of
        this kind of system (e.g. the interfaces).
    expected : dict, optional
        {entry: Provenance} expected now for the entries checked by
        ``is_done``; an entry is a key or a sequence of keys.
    """

    def __init__(self, system_dir, document, collection, expected=None):
        if not isinstance(document, SystemDocument):
            raise StoreError('document: a SystemDocument is required')
        if not isinstance(collection, str) or not collection or collection == TASKS:
            raise StoreError(f'collection: {collection!r} cannot hold the system documents')
        self.system_dir = Path(system_dir)
        self.document = document
        self.collection = collection
        self.expected = {}
        for entry, provenance in (expected or {}).items():
            if not isinstance(provenance, Provenance):
                raise StoreError(f'expected[{entry!r}]: a Provenance is required')
            self.expected[entry_path(entry)] = provenance

    # -- helpers ---------------------------------------------------------

    def _collection(self, collection, write):
        if collection == TASKS:
            if write:
                raise StoreError(f'{TASKS!r} is a read-only view of the records')
            return TASKS
        if collection != self.collection:
            raise StoreError(f'collection {collection!r} is not in this store: it holds '
                             f'{self.collection!r} and the read-only view {TASKS!r}')
        return collection

    def _match(self, fltr):
        # Raise unless the filter matches the document of the system.
        if not isinstance(fltr, dict):
            raise StoreError(f'fltr: a dict is required, not {fltr!r}')
        check_json_value(fltr, 'fltr')
        _check_keys(fltr, 'fltr')
        top = self.document.document
        for key, wanted in fltr.items():
            if wanted is None:
                raise StoreError(f'fltr[{key!r}] is None: a filter gives values only')
            stored = top.get(key, _MISSING)
            if not _same(stored, wanted, f'fltr[{key!r}]'):
                shown = 'missing' if stored is _MISSING else repr(stored)
                raise StoreError(f'fltr {fltr!r} does not match the document of system '
                                 f'{self.system_dir.name!r} ({key}: {shown})')

    def _field(self, fltr):
        if fltr is None:
            raise StoreError('fltr is required')
        if not self.document.exists:
            return None
        self._match(fltr)
        return self.document.document

    def _task(self, fltr):
        if not isinstance(fltr, dict) or list(fltr) != ['task_label'] \
                or not isinstance(fltr['task_label'], str):
            raise StoreError(f'fltr {fltr!r}: the {TASKS!r} view is searched by task_label only')
        try:
            record = read_record(self.system_dir, fltr['task_label'])
        except FileNotFoundError:
            return None
        return task_document(record)

    # -- historic calls --------------------------------------------------

    def is_done(self, collection, *, entry=None, fltr=None):
        """Whether an entry of the document exists, is not empty and is current.

        Returns
        -------
        bool
            False without ``entry`` or before the document exists; otherwise
            the historic condition (the entry is not empty) and the expected
            provenance (``system.is_done``).

        Raises
        ------
        StoreError
            Without ``fltr``, for a filter that does not match, for an entry
            without an expected provenance, and for a list of entries.
        """
        if self._collection(collection, write=False) == TASKS:
            raise StoreError(f'is_done does not apply to the {TASKS!r} view')
        if entry is None:
            return False
        _check_entry(entry)
        field = self._field(fltr)
        if field is None or not bool(read_multiple_entry(field, entry)):
            return False
        if _is_list_of_entries(entry):
            raise StoreError('is_done on a list of entries is not supported: provenance is '
                             'recorded per entry')
        path = _path(entry)
        if path not in self.expected:
            raise StoreError(f'is_done({".".join(path)}): no expected provenance was given for '
                             'this entry')
        return entry_is_current(self.document, path, self.expected[path], self.system_dir)

    def query_db(self, collection, *, entry=None, fltr=None):
        """Document (or task document) matching the filter, and the entry read from it.

        Returns
        -------
        field : dict or None
            Copy of the document; None before it exists or for a missing
            task.
        field_entry : object
            Value of ``entry`` in ``field``: a list for a list of entries,
            None for a missing key or without ``entry``.
        """
        _check_entry(entry)
        if self._collection(collection, write=False) == TASKS:
            field = self._task(fltr)
        else:
            field = self._field(fltr)
        if field is None:
            return None, None
        return field, read_multiple_entry(field, entry)

    def update_db(self, data, collection, *, entry=None, fltr=None, update_op='$set',
                  to_mongodb=True):
        """Set data at one or more entries of the document.

        Raises
        ------
        StoreError
            Without ``fltr``, before the document exists, for an operator
            other than ``'$set'``, for data or entries of an unsupported form.
        """
        self._collection(collection, write=True)
        if fltr is None:
            raise StoreError('update_db needs fltr')
        if update_op != '$set':
            raise StoreError(f'update_op {update_op!r} is not supported: only "$set"')
        if not self.document.exists:
            raise StoreError('update_db: the system has no document yet; create it with '
                             'insert_db')
        self._match(fltr)
        updates = _updates(data, entry)
        for path, value in updates:
            check_json_value(value, f'data at {".".join(path) or "<document>"}')
            _check_keys(value, 'data')
        for path, value in updates:
            if to_mongodb:
                self.document.set(path, value)
            elif path:
                for key in reversed(path[1:]):
                    value = {key: value}
                self.document.replace(path[:1], value)
            else:
                for key, item in value.items():
                    self.document.replace((key,), item)

    def insert_db(self, field, collection, *, duplicates=False):
        """Create the document of the system.

        Raises
        ------
        StoreError
            For anything but one dict, and when the system already has a
            document whose fields differ from those given (or any document,
            with ``duplicates=True``).
        """
        self._collection(collection, write=True)
        if not isinstance(field, dict):
            raise StoreError('insert_db takes one document (a dict)')
        check_json_value(field, 'field')
        _check_keys(field, 'field')
        if not self.document.exists:
            self.document.create(field)
            return
        top = self.document.document
        if not duplicates and all(_same(top.get(k, _MISSING), v, f'field[{k!r}]')
                                  for k, v in field.items()):
            logger.warning('%s: the fields given equal those of the document, which is not '
                           'inserted again', self.system_dir.name)
            return
        raise StoreError(f'system {self.system_dir.name!r} already has a document: a second '
                         'one cannot be inserted')


def task_document(record):
    """Task document of a complete record, as the ``tasks`` view returns it.

    Parameters
    ----------
    record : Record
        Complete record (hence with results).

    Returns
    -------
    dict
        Copy of ``record.results`` with ``task_label``, ``task_id`` and
        ``dir_name``, all equal to the ``calc_id``: ``dir_name`` is the
        directory of the calculation relative to the system, which stays true
        when the run is copied elsewhere.

    Raises
    ------
    StoreError
        If the record is not complete, or its results already hold one of
        those three keys.
    """
    if not record.is_complete or record.results is None:
        raise StoreError(f'record {record.calc_id} is not complete ({record.status}): its '
                         'results cannot be used')
    clash = [key for key in TASK_KEYS if key in record.results]
    if clash:
        raise StoreError(f'record {record.calc_id}: results cannot hold {", ".join(clash)}')
    document = copy.deepcopy(record.results)
    document.update({'task_label': record.calc_id, 'task_id': record.calc_id,
                     'dir_name': record.calc_id})
    return document
