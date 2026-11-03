# Author: Vito Foderà
# Created: 2026-10-23

"""Atomic JSON and text files, and the values JSON files can hold.

A file written by ``write_json`` or ``write_text`` is never seen
half-written. The text goes to
a temporary file in the same directory (``.<name>.<random>.tmp``), is flushed
to disk and is then renamed over the target with ``os.replace``. A reader
opens either the old file or the new one. A write interrupted by a crash
leaves the old file and, at most, an orphan temporary file that no reader
opens. The directory is then synchronized too, so that the rename survives a
power loss; a filesystem that does not support this is reported once in the
log, since only that durability is lost, never the atomicity seen by readers.

Files are standard JSON. Floats that are not finite, which standard JSON
cannot write, are stored as a tagged object and read back as the same float::

    nan   -> {"@float": "NaN"}
    inf   -> {"@float": "Infinity"}
    -inf  -> {"@float": "-Infinity"}

A missing value stays a missing key, and None stays null. A dict whose only
key is ``"@float"`` would read back as a float, so it cannot be stored.

Notes
-----
The rename is atomic within one POSIX filesystem. Whether the parallel
filesystems of the HPC machines keep this guarantee across nodes is checked
there, together with the locks. On Windows ``os.replace`` fails while another
process has the target open.
"""

import contextlib
import errno
import json
import logging
import math
import os
import secrets
from pathlib import Path

logger = logging.getLogger(__name__)

FLOAT_TAG = '@float'
NONFINITE = {'NaN': math.nan, 'Infinity': math.inf, '-Infinity': -math.inf}

_UNSYNCED = set()


class StoreError(ValueError):
    """Data, files or calls that the store does not accept."""


def nonfinite_name(value):
    """Name of a non-finite float in the tagged form: ``'NaN'``, ``'Infinity'``, ``'-Infinity'``."""
    if math.isnan(value):
        return 'NaN'
    return 'Infinity' if value > 0 else '-Infinity'


def is_float_tag(value):
    """True for a dict that has ``"@float"`` as its only key."""
    return isinstance(value, dict) and list(value) == [FLOAT_TAG]


def check_json_value(value, where='value'):
    """Check that a value can be written as JSON and read back unchanged.

    Parameters
    ----------
    value : object
        Dicts with string keys, lists, tuples, strings, numbers, booleans and
        None, nested at will. Tuples are read back as lists, as a document
        database stores them. Non-finite floats are allowed and stored in the
        tagged form of the module notes.
    where : str, optional
        Location of the value, for the error messages.

    Raises
    ------
    StoreError
        For keys that are not strings, which ``json`` would turn into strings,
        for dicts whose only key is ``"@float"``, and for values of any other
        type (sets, paths, numpy arrays, numpy integers and booleans, ...).
    """
    if isinstance(value, dict):
        if is_float_tag(value):
            raise StoreError(f'{where}: a dict with the only key {FLOAT_TAG!r} is reserved for '
                             'non-finite floats')
        for key, item in value.items():
            if not isinstance(key, str):
                raise StoreError(f'{where}: key {key!r} is not a string')
            check_json_value(item, f'{where}[{key!r}]')
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            check_json_value(item, f'{where}[{i}]')
    elif not isinstance(value, (str, int, float, type(None))):
        raise StoreError(f'{where}: a {type(value).__name__} cannot be stored as JSON')


def _encode(value):
    # Tagged objects in place of the non-finite floats.
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {FLOAT_TAG: nonfinite_name(value)}
    return value


def _decode(obj, path):
    if is_float_tag(obj):
        name = obj[FLOAT_TAG]
        if name not in NONFINITE:
            raise StoreError(f'{path}: {name!r} is not one of {", ".join(NONFINITE)}')
        return NONFINITE[name]
    return obj


def _reject_constant(name, path):
    raise StoreError(f'{path}: {name} is not standard JSON; non-finite floats are written as '
                     f'{{"{FLOAT_TAG}": "{name}"}}')


def write_json(path, value):
    """Write a JSON file atomically.

    Parameters
    ----------
    path : path-like
        Target file; its directory must exist.
    value : object
        Data accepted by ``check_json_value``.

    Raises
    ------
    StoreError
        If the data cannot be stored as JSON; the target is not touched.
    """
    path = Path(path)
    check_json_value(value, path.name)
    text = json.dumps(_encode(value), indent=1, ensure_ascii=False, allow_nan=False)
    _write_bytes(path, (text + '\n').encode('utf-8'))


def write_text(path, text):
    """Write a text file atomically, in UTF-8 and with the line ends of ``text``.

    Parameters
    ----------
    path : path-like
        Target file; its directory must exist.
    text : str
    """
    if not isinstance(text, str):
        raise StoreError(f'{Path(path).name}: a str is required, not {type(text).__name__}')
    _write_bytes(Path(path), text.encode('utf-8'))


def _write_bytes(path, data):
    tmp = path.with_name(f'.{path.name}.{secrets.token_hex(8)}.tmp')
    # Mode 0o666 lets the umask decide the permissions, as for any new file.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0), 0o666)
    try:
        with os.fdopen(fd, 'wb') as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    _sync_directory(path.parent)


def _sync_directory(directory):
    # Makes the rename itself survive a power loss. Not available on Windows.
    if os.name != 'posix':
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as err:
        if err.errno not in (errno.EINVAL, errno.ENOTSUP):
            raise
        key = os.path.abspath(directory)
        if key not in _UNSYNCED:
            _UNSYNCED.add(key)
            logger.warning('%s: the filesystem does not synchronize directories (%s): files are '
                           'still replaced atomically, but the last renames may be lost after '
                           'a power failure', key, os.strerror(err.errno))
    finally:
        os.close(fd)


def read_json(path):
    """Read a JSON file written by ``write_json``.

    Parameters
    ----------
    path : path-like
        File to read.

    Returns
    -------
    object
        The data, with the tagged non-finite floats read back as floats.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    StoreError
        If the file is not valid standard JSON (bare ``NaN`` or ``Infinity``
        included) or holds an unknown ``"@float"`` tag.
    """
    path = Path(path)
    try:
        return json.loads(path.read_bytes().decode('utf-8'),
                          object_hook=lambda obj: _decode(obj, path),
                          parse_constant=lambda name: _reject_constant(name, path))
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise StoreError(f'{path}: not a valid JSON file ({err})') from err
