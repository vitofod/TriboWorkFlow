# Author: Vito Foderà
# Created: 2026-10-23

"""Content digests: fingerprints of inputs and data, digests of files.

A fingerprint is the SHA-256 of the canonical JSON text of a value, written
``sha256:<64 hex digits>``. Two values have the same fingerprint when they
have the same canonical form:

- dict keys are sorted, and must be strings;
- tuples and numpy arrays count as lists, numpy scalars as the Python value
  they hold;
- floats are written with the shortest text that reads back to the same
  float, so they are compared exactly. ``1`` and ``1.0`` differ, and so do
  ``True`` and ``1``: choosing the types of an input is up to the caller;
- NaN and infinities have no fingerprint: an input that holds them is an
  error.

What enters the fingerprint of a calculation (normalized input and energy
model) is decided by each engine; this module only computes digests.

``content_digest`` identifies data rather than inputs: the results of a
calculation or an entry read by an analysis. It follows the same rules but
accepts non-finite floats, written in the tagged form of the stored files
(``{"@float": "NaN"}``, ...), because results may legitimately hold them (for
instance a fit whose uncertainty cannot be estimated).
"""

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from .atomic import FLOAT_TAG, StoreError, is_float_tag, nonfinite_name

PREFIX = 'sha256:'
PATTERN = re.compile(r'sha256:[0-9a-f]{64}')


def canonical(value, where='value', nonfinite=False):
    """Canonical form of a value: plain dicts, lists, strings and numbers.

    Parameters
    ----------
    value : object
        Dicts with string keys, lists, tuples, numpy arrays and scalars,
        strings, numbers, booleans and None.
    where : str, optional
        Location of the value, for the error messages.
    nonfinite : bool, optional
        If True, non-finite floats become their tagged form; otherwise they
        are an error.

    Returns
    -------
    object
        The same data with tuples and arrays as lists and numpy scalars as
        Python values.

    Raises
    ------
    StoreError
        For keys that are not strings, non-finite floats (unless allowed),
        dicts whose only key is ``"@float"`` and values of any other type.
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        if is_float_tag(value):
            raise StoreError(f'{where}: a dict with the only key {FLOAT_TAG!r} is reserved for '
                             'non-finite floats')
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise StoreError(f'{where}: key {key!r} is not a string')
            out[key] = canonical(item, f'{where}[{key!r}]', nonfinite)
        return out
    if isinstance(value, (list, tuple)):
        return [canonical(item, f'{where}[{i}]', nonfinite) for i, item in enumerate(value)]
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if nonfinite:
            return {FLOAT_TAG: nonfinite_name(value)}
        raise StoreError(f'{where}: {value!r} has no fingerprint')
    if isinstance(value, (str, int, type(None))):
        return value
    raise StoreError(f'{where}: a {type(value).__name__} has no canonical JSON form')


def canonical_json(value, where='value', nonfinite=False):
    """Canonical JSON text of a value (sorted keys, no spaces)."""
    return json.dumps(canonical(value, where, nonfinite), sort_keys=True,
                      separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def fingerprint(value, where='value'):
    """Fingerprint of an input: ``sha256:`` and the digest of its canonical JSON.

    Parameters
    ----------
    value : object
        Data accepted by ``canonical``, without non-finite floats.
    where : str, optional
        Location of the value, for the error messages.

    Returns
    -------
    str

    Examples
    --------
    >>> fingerprint({'b': (1, 2.5), 'a': 'x'}) == fingerprint({'a': 'x', 'b': [1, 2.5]})
    True
    """
    text = canonical_json(value, where)
    return PREFIX + hashlib.sha256(text.encode('utf-8')).hexdigest()


def content_digest(value, where='value'):
    """Digest of data, written like a fingerprint; non-finite floats allowed.

    Parameters
    ----------
    value : object
        Data accepted by ``canonical`` with ``nonfinite=True``.
    where : str, optional
        Location of the value, for the error messages.

    Returns
    -------
    str
        Equal to ``fingerprint(value)`` when the value has only finite floats.
    """
    text = canonical_json(value, where, nonfinite=True)
    return PREFIX + hashlib.sha256(text.encode('utf-8')).hexdigest()


def file_digest(path):
    """Digest of the bytes of a file, written like a fingerprint."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            digest.update(block)
    return PREFIX + digest.hexdigest()


def check_fingerprint(value, where='fingerprint'):
    """Check the form ``sha256:<64 lowercase hex digits>`` and return the value."""
    if not isinstance(value, str) or not PATTERN.fullmatch(value):
        raise StoreError(f'{where}: {value!r} is not "sha256:" followed by 64 hex digits')
    return value
