# Author: Vito Foderà
# Created: 2026-10-23

"""Reading of the TOML settings into the dataclasses of ``schema.py``.

The file is read with ``tomllib``, every table is converted by dacite into
its dataclass, and then the ``check`` methods of the schema validate the
result. Every error is a ``SettingsError`` that names the file and the TOML
path of the field.

Notes
-----
As in Xsorb (``xsorb/settings/settings.py:78-109``, CC BY 4.0) each table is
converted by dacite. Differences:

- dacite runs with ``strict=True`` on every table of the schema (Xsorb only
  on ``Structure``); unknown keys are found first by ``_check_unknown_keys``,
  which gives their full path and the allowed keys. The pw.x namelists are
  free tables: their keys are those of pw.x;
- in the tables of the schema integers are accepted for real numbers, and
  booleans are not accepted for numbers;
- after dacite, ``validate`` runs the ``check`` methods of the schema, then
  logs as warnings the ``advice`` of the tables that give it.

A file with ``[structure]`` and without ``[generation]`` gets the defaults of
``[generation]`` (``GenerationParams()``), as documented for its keys; in the
workflow ``[generation]`` is not read and stays as given.
"""

import logging
import tomllib
import types
import typing
from dataclasses import fields, is_dataclass
from pathlib import Path

from dacite import (Config, DaciteError, MissingValueError, WrongTypeError,
                    from_dict)

from .schema import (KEY_HINTS, SCHEMA_VERSION, GenerationParams, Settings, SettingsError, join,
                     walk)

logger = logging.getLogger(__name__)


def load_settings(path):
    """Read, convert and validate a TOML settings file.

    Parameters
    ----------
    path : str or pathlib.Path
        The settings file.

    Returns
    -------
    Settings
        With ``source`` set to ``path``.

    Raises
    ------
    SettingsError
        If the file cannot be read, is not valid TOML or does not follow the
        schema.
    """
    path = Path(path)
    try:
        with open(path, 'rb') as f:
            data = tomllib.load(f)
    except OSError as error:
        raise SettingsError('', f'cannot read the settings: {error}', path) from None
    except tomllib.TOMLDecodeError as error:
        raise SettingsError('', f'invalid TOML: {error}', path) from None
    try:
        settings = settings_from_dict(data)
    except SettingsError as error:
        raise SettingsError(error.path, error.message, path) from None
    settings.source = path
    return settings


def settings_from_dict(data):
    """Convert and validate the dict of a TOML settings file.

    Parameters
    ----------
    data : dict
        The settings as read by ``tomllib``.

    Returns
    -------
    Settings
        With ``source`` left to None.

    Raises
    ------
    SettingsError
        If ``data`` does not follow the schema.
    """
    version = data.get('schema_version')
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        found = 'missing' if version is None else f'unsupported value {version!r}'
        raise SettingsError('schema_version', f'{found}; supported: {SCHEMA_VERSION}')
    _check_unknown_keys(Settings, data, '')
    config = Config(strict=True, type_hooks={float: _real, int: _integer})
    try:
        settings = from_dict(Settings, data, config)
    except WrongTypeError as error:
        raise SettingsError(error.field_path, f'expected {_type_name(error.field_type)}, '
                                              f'got {error.value!r}') from None
    except MissingValueError as error:
        raise SettingsError(error.field_path, 'required key missing') from None
    except DaciteError as error:
        raise SettingsError('', str(error)) from None
    if settings.structure is not None and settings.generation is None:
        settings.generation = GenerationParams()
    validate(settings)
    return settings


def validate(settings):
    """Run every check of the schema, innermost tables first, and log its advice.

    Parameters
    ----------
    settings : Settings

    Raises
    ------
    SettingsError
        At the first check that fails.
    """
    for path, node in walk(settings):
        if hasattr(node, 'check'):
            node.check(path)
    for path, node in walk(settings):
        for note in node.advice(path) if hasattr(node, 'advice') else ():
            logger.warning('%s', note)


def _real(value):
    if isinstance(value, bool):
        raise WrongTypeError(field_type=float, value=value)
    return float(value) if isinstance(value, int) else value


def _integer(value):
    if isinstance(value, bool):
        raise WrongTypeError(field_type=int, value=value)
    return value


def _without_none(type_):
    if isinstance(type_, types.UnionType):
        args = [arg for arg in typing.get_args(type_) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return type_


def _type_name(type_):
    type_ = _without_none(type_)
    if typing.get_origin(type_) is list:
        return f'a list of {_type_name(typing.get_args(type_)[0])}'
    if is_dataclass(type_) or typing.get_origin(type_) is dict:
        return 'a table'
    return getattr(type_, '__name__', str(type_))


def _check_unknown_keys(cls, data, path):
    """Raise SettingsError for a key that is not a field of ``cls`` (recursively)."""
    if not isinstance(data, dict):
        return  # dacite reports the wrong type
    hints = typing.get_type_hints(cls)
    known = {f.name: _without_none(hints[f.name]) for f in fields(cls) if f.init}
    for key, value in data.items():
        child = join(path, key)
        if key not in known:
            reason = KEY_HINTS.get(key, f'allowed keys: {", ".join(known)}')
            raise SettingsError(child, f'unknown key ({reason})')
        type_ = known[key]
        if is_dataclass(type_):
            _check_unknown_keys(type_, value, child)
        elif typing.get_origin(type_) is list and isinstance(value, list):
            item = typing.get_args(type_)[0]
            if is_dataclass(item):
                for i, entry in enumerate(value):
                    _check_unknown_keys(item, entry, f'{child}[{i}]')
