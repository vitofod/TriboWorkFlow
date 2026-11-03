# Author: Vito Foderà
# Created: 2026-10-23

"""Record of one calculation.

A record is the JSON file ``<system>/<step>/<label>/record.json`` of one
calculation. It tells what the calculation is (system, step, label,
``calc_type``), which engine runs it and with which energy model, the
fingerprint of its input, its states, the attempts made, the results and the
names of its files. The identifier of a calculation is ``<step>/<label>``,
its directory relative to the system.

States are kept apart (``Status``): end of the process, electronic (SCF)
convergence, geometric convergence and parsing of the output. Each one is
``None`` until it is known. A record is complete when the process ended, the
SCF converged, the geometry converged or does not apply, and the output was
parsed; a parsed output always leaves its results in the record. For whoever
looks at a calculation, ``Status.summary`` sums the states up in one word.
The results of a record stand for a calculation only if the record is
complete, has results, its fingerprint equals the one expected, its energy
model is fully known and its output agrees with it (``can_reuse``). Which results each kind of
calculation must have is checked by the parser of its engine.

The energy model is written by the engine. A null value in it marks a part
that is not known, e.g. the expected checksum of a pseudopotential not
given: the inputs can still be prepared and run, but such a record is never
reused and its energies are never combined (``check_energy_model``).

What the output tells of the energy model is written by the parser in
``observed``, with the keys and the form of the energy model (e.g. the
version of the engine, the checksum of each pseudopotential). Each value of
``observed`` whose key is also in the energy model is compared with it
(``identity_differences``): an output that differs is kept, with its results
and its states, but it does not stand for its energy model, so it is never
reused and its energies are never combined. A parsed output always leaves
``observed``, empty if the engine observes nothing.

The states, the results and ``observed`` describe the last run. Each run is
also kept as an ``Attempt``: where and when it ran, the time limit, the input
and the output with their digests, the exit code, the state of the job, the
reason of the stop, why its output could not be parsed and the states at its
end.

Notes
-----
The engine version read from the output (``Engine.version``) is recorded as
printed; the engine writes in ``observed`` the form that is compared with
the version expected.
"""

import copy
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

from .atomic import StoreError, check_json_value, read_json, write_json
from .fingerprint import canonical_json, check_fingerprint, content_digest

SCHEMA_VERSION = 3
RECORD_FILE = 'record.json'

PROCESS_STATES = ('pending', 'running', 'ended', 'crashed', 'killed')
SCF_STATES = ('converged', 'not_converged')
GEOMETRY_STATES = ('converged', 'not_converged', 'max_steps', 'n/a')
PARSING_STATES = ('ok', 'failed')
SUMMARY_STATES = ('pending', 'running', 'completed', 'not_converged', 'crashed', 'unknown')

_RECORD_KEYS = ('schema_version', 'calc_id', 'system', 'step', 'label', 'calc_type', 'engine',
                'energy_model', 'fingerprint', 'status', 'attempts', 'results', 'observed',
                'files')
_STATUS_KEYS = ('process', 'scf', 'geometry', 'parsing')
_ATTEMPT_KEYS = ('n', 'job_id', 'host', 'started', 'ended', 'max_seconds', 'max_seconds_cap',
                 'input', 'output', 'exit_code', 'job_state', 'stop_reason', 'clean_stop',
                 'parse_error', 'status', 'remedies')


def check_name(value, where):
    """Check a name that is also a directory: system, step or label.

    Parameters
    ----------
    value : object
        Name to check.
    where : str
        Field of the name, for the error message.

    Returns
    -------
    str
        The name.

    Raises
    ------
    StoreError
        Unless the name is a non-empty string other than ``.`` and ``..``,
        without ``/``, ``\\`` and NUL.
    """
    if (not isinstance(value, str) or value in ('', '.', '..')
            or any(c in value for c in '/\\\0')):
        raise StoreError(f'{where}: {value!r} is not a valid name (a non-empty string without '
                         f'"/", "\\" and NUL, other than "." and "..")')
    return value


def split_calc_id(calc_id):
    """Step and label of a calculation identifier ``<step>/<label>``."""
    if not isinstance(calc_id, str) or calc_id.count('/') != 1:
        raise StoreError(f'calc_id: {calc_id!r} is not "<step>/<label>"')
    step, label = calc_id.split('/')
    return check_name(step, 'calc_id'), check_name(label, 'calc_id')


def system_name(system_dir):
    """Name of a system: the name of its directory."""
    return Path(os.path.abspath(system_dir)).name


def _text(value, where):
    if not isinstance(value, str) or not value:
        raise StoreError(f'{where}: a non-empty string is required, not {value!r}')
    return value


def _optional_text(value, where):
    return None if value is None else _text(value, where)


def _relative_file(name, where):
    """Check a path relative to the calculation directory and inside it."""
    if (not isinstance(name, str) or not name or PurePosixPath(name).is_absolute()
            or PureWindowsPath(name).is_absolute() or '..' in PurePosixPath(name).parts
            or '\\' in name):
        raise StoreError(f'{where}: {name!r} is not a relative path inside the calculation '
                         'directory')
    return name


def check_keys(value, keys, where):
    """Check that a value is a JSON object with exactly the given keys, and return it."""
    if not isinstance(value, dict):
        raise StoreError(f'{where}: a JSON object is required, not {type(value).__name__}')
    missing = [k for k in keys if k not in value]
    unknown = [k for k in value if k not in keys]
    if missing or unknown:
        parts = ([f'missing {", ".join(missing)}'] if missing else []) + \
                ([f'unknown {", ".join(map(repr, unknown))}'] if unknown else [])
        raise StoreError(f'{where}: {"; ".join(parts)}')
    return value


@dataclass(frozen=True)
class Engine:
    """Program that runs a calculation.

    Parameters
    ----------
    name : str
        Engine (e.g. ``'qe'``).
    program : str
        Executable (e.g. ``'pw.x'``).
    version : str or None, optional
        Version of the program, read from its output; None until known.
    """

    name: str
    program: str
    version: str | None = None

    def __post_init__(self):
        _text(self.name, 'engine.name')
        _text(self.program, 'engine.program')
        if self.version is not None:
            _text(self.version, 'engine.version')


@dataclass(frozen=True)
class Status:
    """Separate states of a calculation, each None until known.

    Parameters
    ----------
    process : str or None, optional
        End of the process: ``'pending'``, ``'running'``, ``'ended'``,
        ``'crashed'`` or ``'killed'``.
    scf : str or None, optional
        Electronic convergence: ``'converged'`` or ``'not_converged'``.
    geometry : str or None, optional
        Geometric convergence: ``'converged'``, ``'not_converged'`` (the run
        is over without convergence), ``'max_steps'`` or ``'n/a'`` (no
        relaxation).
    parsing : str or None, optional
        Reading of the output: ``'ok'`` or ``'failed'``.
    """

    process: str | None = None
    scf: str | None = None
    geometry: str | None = None
    parsing: str | None = None

    def __post_init__(self):
        for name, allowed in (('process', PROCESS_STATES), ('scf', SCF_STATES),
                              ('geometry', GEOMETRY_STATES), ('parsing', PARSING_STATES)):
            value = getattr(self, name)
            if value is not None and value not in allowed:
                raise StoreError(f'status.{name}: {value!r} is not one of '
                                 f'{", ".join(allowed)} or null')

    @property
    def complete(self):
        """True when the process ended, the SCF and the geometry converged
        (or the geometry does not apply) and the output was parsed."""
        return (self.process == 'ended' and self.scf == 'converged'
                and self.geometry in ('converged', 'n/a') and self.parsing == 'ok')

    @property
    def summary(self):
        """One word for whoever looks at the calculation.

        ``'pending'`` (queued), ``'running'``, ``'completed'`` (ended with
        the SCF and the geometry converged, or the geometry does not apply),
        ``'not_converged'`` (ended without convergence, whichever part did
        not converge), ``'crashed'`` (crashed or killed) or ``'unknown'``
        (end of the process not known). It describes the calculation, not the
        reading of its output: ``parsing`` does not enter it. The separate
        states tell the details.
        """
        if self.process is None:
            return 'unknown'
        if self.process in ('crashed', 'killed'):
            return 'crashed'
        if self.process == 'ended':
            converged = self.scf == 'converged' and self.geometry in ('converged', 'n/a')
            return 'completed' if converged else 'not_converged'
        return self.process


def _status_dict(status):
    return {'process': status.process, 'scf': status.scf, 'geometry': status.geometry,
            'parsing': status.parsing}


@dataclass(frozen=True)
class AttemptFile:
    """A file of an attempt.

    Parameters
    ----------
    path : str
        Path relative to the calculation directory (e.g. ``'output.pwo'``,
        or ``'attempts/1/output.pwo'`` once archived).
    digest : str or None, optional
        ``file_digest`` of its bytes; None until computed.
    """

    path: str
    digest: str | None = None

    def __post_init__(self):
        _relative_file(self.path, 'attempts.file.path')
        if self.digest is not None:
            check_fingerprint(self.digest, 'attempts.file.digest')

    def to_dict(self):
        """JSON object of the file."""
        return {'path': self.path, 'digest': self.digest}

    @classmethod
    def from_dict(cls, data):
        """File from its JSON object, or None from null."""
        if data is None:
            return None
        return cls(**check_keys(data, ('path', 'digest'), 'attempts.file'))


def _time(value, where):
    if value is None:
        return None
    try:
        datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise StoreError(f'{where}: {value!r} is not an ISO 8601 time') from None
    return value


def _seconds(value, where):
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise StoreError(f'{where}: a finite number of seconds is required, not {value!r}')
    return value


@dataclass(frozen=True)
class Attempt:
    """One run of a calculation.

    The fields that nothing has established yet stay None. Each one has a
    single source: the worker that ran the attempt (host, times, time limit,
    files, exit code), the scheduler (state of the job), the output of the
    engine (stop reason, clean stop, states).

    Parameters
    ----------
    n : int
        Number of the attempt, counted from 1.
    job_id : str or None, optional
        Id of the job, as given by the scheduler; None outside a job.
    host : str or None, optional
        Machine the attempt ran on.
    started, ended : str or None, optional
        ISO 8601 times of the start and of the end of the run.
    max_seconds : int or float or None, optional
        Time limit given to the engine for this attempt, in seconds.
    max_seconds_cap : int or float or None, optional
        Upper limit of ``max_seconds`` written in the prepared input; None if
        the input has none.
    input, output : AttemptFile or None, optional
        Input run and output written by the attempt.
    exit_code : int or None, optional
        Exit status of the command that ran the engine, as seen by the
        worker; None when the worker did not see the command end.
    job_state : str or None, optional
        State of the job as reported by the scheduler.
    stop_reason : str or None, optional
        Why the run stopped, as read from the output.
    clean_stop : bool or None, optional
        Whether the engine closed the run, as read from the output.
    parse_error : str or None, optional
        Why the output could not be parsed, written by the parser; given if
        and only if ``status.parsing`` is ``'failed'``.
    status : Status, optional
        States of the calculation at the end of this attempt.
    remedies : dict, optional
        How the attempt was launched, written by the worker: ``kind`` (why
        it ran), ``level`` (of the remedies) and ``values`` (the parameters
        changed with respect to the input). Empty when not known.
    """

    n: int
    job_id: str | None = None
    host: str | None = None
    started: str | None = None
    ended: str | None = None
    max_seconds: float | None = None
    max_seconds_cap: float | None = None
    input: AttemptFile | None = None
    output: AttemptFile | None = None
    exit_code: int | None = None
    job_state: str | None = None
    stop_reason: str | None = None
    clean_stop: bool | None = None
    parse_error: str | None = None
    status: Status = field(default_factory=Status)
    remedies: dict = field(default_factory=dict)

    def __post_init__(self):
        if type(self.n) is not int or self.n < 1:
            raise StoreError(f'attempts.n: {self.n!r} is not an integer >= 1')
        _optional_text(self.job_id, 'attempts.job_id')
        _optional_text(self.host, 'attempts.host')
        _time(self.started, 'attempts.started')
        _time(self.ended, 'attempts.ended')
        _seconds(self.max_seconds, 'attempts.max_seconds')
        _seconds(self.max_seconds_cap, 'attempts.max_seconds_cap')
        for name in ('input', 'output'):
            value = getattr(self, name)
            if value is not None and not isinstance(value, AttemptFile):
                raise StoreError(f'attempts.{name}: an AttemptFile or None is required')
        if self.exit_code is not None and type(self.exit_code) is not int:
            raise StoreError(f'attempts.exit_code: an integer or null is required, '
                             f'not {self.exit_code!r}')
        _optional_text(self.job_state, 'attempts.job_state')
        _optional_text(self.stop_reason, 'attempts.stop_reason')
        if self.clean_stop is not None and type(self.clean_stop) is not bool:
            raise StoreError(f'attempts.clean_stop: true, false or null is required, '
                             f'not {self.clean_stop!r}')
        if not isinstance(self.status, Status):
            raise StoreError('attempts.status: a Status is required')
        _optional_text(self.parse_error, 'attempts.parse_error')
        if (self.parse_error is not None) != (self.status.parsing == 'failed'):
            raise StoreError(f'attempts.parse_error: {self.parse_error!r} with status.parsing '
                             f'{self.status.parsing!r}; the reason is given if and only if the '
                             'parsing failed')
        if not isinstance(self.remedies, dict):
            raise StoreError('attempts.remedies: a JSON object is required')
        check_json_value(self.remedies, 'attempts.remedies')
        object.__setattr__(self, 'remedies', copy.deepcopy(self.remedies))

    def to_dict(self):
        """JSON object of the attempt, with the keys in the order of the file."""
        return {'n': self.n, 'job_id': self.job_id, 'host': self.host,
                'started': self.started, 'ended': self.ended,
                'max_seconds': self.max_seconds, 'max_seconds_cap': self.max_seconds_cap,
                'input': None if self.input is None else self.input.to_dict(),
                'output': None if self.output is None else self.output.to_dict(),
                'exit_code': self.exit_code, 'job_state': self.job_state,
                'stop_reason': self.stop_reason, 'clean_stop': self.clean_stop,
                'parse_error': self.parse_error, 'status': _status_dict(self.status),
                'remedies': copy.deepcopy(self.remedies)}

    @classmethod
    def from_dict(cls, data):
        """Attempt from its JSON object, with exactly the keys of ``to_dict``."""
        check_keys(data, _ATTEMPT_KEYS, 'attempts')
        fields = dict(data)
        fields['input'] = AttemptFile.from_dict(data['input'])
        fields['output'] = AttemptFile.from_dict(data['output'])
        fields['status'] = Status(**check_keys(data['status'], _STATUS_KEYS, 'attempts.status'))
        return cls(**fields)


@dataclass(frozen=True)
class Record:
    """Record of one calculation.

    Parameters
    ----------
    system : str
        System, i.e. the name of its directory.
    step : str
        Step of the workflow (e.g. ``'pes_scf'``), the first directory level.
    label : str
        Label of the calculation within the step, the second level.
    calc_type : str
        Kind of calculation, which fixes its constraints (e.g.
        ``'slab_from_scratch'``).
    engine : Engine
        Program that runs the calculation.
    energy_model : dict
        Everything that makes energies comparable, as written by the engine
        (pseudopotentials with their digests or the MLIP model, cutoffs,
        k-points, smearing, spin, vdW). Not empty.
    fingerprint : str
        Fingerprint of the normalized input and of the energy model.
    status : Status, optional
        States; all None by default.
    attempts : tuple of Attempt, optional
        Runs made so far, numbered 1, 2, ... in order.
    results : dict or None, optional
        Results read from the output; None until read.
    observed : dict or None, optional
        What the output tells of the energy model, with its keys and form,
        written by the parser (module notes); None until read, a dict once
        ``status.parsing`` is ``'ok'``.
    files : dict, optional
        {role: file name relative to the calculation directory}, e.g.
        ``{'input': 'input.pwi', 'output': 'output.pwo'}``.

    Raises
    ------
    StoreError
        If a field is not valid.
    """

    system: str
    step: str
    label: str
    calc_type: str
    engine: Engine
    energy_model: dict
    fingerprint: str
    status: Status = field(default_factory=Status)
    attempts: tuple = ()
    results: dict | None = None
    observed: dict | None = None
    files: dict = field(default_factory=dict)

    def __post_init__(self):
        check_name(self.system, 'system')
        check_name(self.step, 'step')
        check_name(self.label, 'label')
        _text(self.calc_type, 'calc_type')
        if not isinstance(self.engine, Engine):
            raise StoreError('engine: an Engine is required')
        if not isinstance(self.energy_model, dict) or not self.energy_model:
            raise StoreError('energy_model: a non-empty JSON object is required')
        check_json_value(self.energy_model, 'energy_model')
        canonical_json(self.energy_model, 'energy_model')
        check_fingerprint(self.fingerprint)
        if not isinstance(self.status, Status):
            raise StoreError('status: a Status is required')
        attempts = tuple(self.attempts)
        if not all(isinstance(a, Attempt) for a in attempts):
            raise StoreError('attempts: a sequence of Attempt is required')
        numbers = [a.n for a in attempts]
        if numbers != list(range(1, len(attempts) + 1)):
            raise StoreError(f'attempts: numbered {numbers}, not 1, 2, ... in order')
        if self.results is not None:
            if not isinstance(self.results, dict):
                raise StoreError('results: a JSON object or null is required')
            check_json_value(self.results, 'results')
        elif self.status.parsing == 'ok':
            raise StoreError('status.parsing is "ok" but results is null: a parsed output must '
                             'leave its results in the record')
        if self.observed is not None:
            if not isinstance(self.observed, dict):
                raise StoreError('observed: a JSON object or null is required')
            check_json_value(self.observed, 'observed')
        elif self.status.parsing == 'ok':
            raise StoreError('status.parsing is "ok" but observed is null: a parsed output must '
                             'leave what it tells of the energy model in the record')
        if not isinstance(self.files, dict):
            raise StoreError('files: a JSON object is required')
        for role, name in self.files.items():
            _text(role, 'files')
            _relative_file(name, f'files[{role!r}]')
        object.__setattr__(self, 'energy_model', copy.deepcopy(self.energy_model))
        object.__setattr__(self, 'attempts', attempts)
        object.__setattr__(self, 'results', copy.deepcopy(self.results))
        object.__setattr__(self, 'observed', copy.deepcopy(self.observed))
        object.__setattr__(self, 'files', dict(self.files))

    @property
    def calc_id(self):
        """Identifier of the calculation: ``<step>/<label>``."""
        return f'{self.step}/{self.label}'

    @property
    def is_complete(self):
        """True when the states allow the results to be used (``Status.complete``)."""
        return self.status.complete

    @property
    def results_digest(self):
        """Digest of the results (``content_digest``), or None without results.

        It identifies the results an analysis consumed: a calculation run
        again with the same input but different results changes it.
        """
        return None if self.results is None else content_digest(self.results, 'results')

    def to_dict(self):
        """JSON object of the record, with the keys in the order of the file."""
        return {
            'schema_version': SCHEMA_VERSION,
            'calc_id': self.calc_id,
            'system': self.system,
            'step': self.step,
            'label': self.label,
            'calc_type': self.calc_type,
            'engine': {'name': self.engine.name, 'program': self.engine.program,
                       'version': self.engine.version},
            'energy_model': copy.deepcopy(self.energy_model),
            'fingerprint': self.fingerprint,
            'status': _status_dict(self.status),
            'attempts': [a.to_dict() for a in self.attempts],
            'results': copy.deepcopy(self.results),
            'observed': copy.deepcopy(self.observed),
            'files': dict(self.files),
        }

    @classmethod
    def from_dict(cls, data, source=None):
        """Record from its JSON object.

        Parameters
        ----------
        data : dict
            JSON object with exactly the keys written by ``to_dict``.
        source : path-like, optional
            File of the record, prefixed to the error messages.

        Raises
        ------
        StoreError
            For missing or unknown keys, another schema version, a
            ``calc_id`` other than ``<step>/<label>`` or an invalid field.
        """
        try:
            check_keys(data, _RECORD_KEYS, 'record')
            version = data['schema_version']
            if type(version) is not int or version != SCHEMA_VERSION:
                raise StoreError(f'schema_version: {version!r}; this version reads '
                                 f'{SCHEMA_VERSION}')
            engine = check_keys(data['engine'], ('name', 'program', 'version'), 'engine')
            status = check_keys(data['status'], _STATUS_KEYS, 'status')
            if not isinstance(data['attempts'], list):
                raise StoreError('attempts: a JSON array is required')
            attempts = [Attempt.from_dict(a) for a in data['attempts']]
            record = cls(system=data['system'], step=data['step'], label=data['label'],
                         calc_type=data['calc_type'], engine=Engine(**engine),
                         energy_model=data['energy_model'], fingerprint=data['fingerprint'],
                         status=Status(**status), attempts=tuple(attempts),
                         results=data['results'], observed=data['observed'],
                         files=data['files'])
            if data['calc_id'] != record.calc_id:
                raise StoreError(f'calc_id: {data["calc_id"]!r} is not "<step>/<label>" '
                                 f'= {record.calc_id!r}')
            return record
        except StoreError as err:
            raise StoreError(f'{source}: {err}' if source else str(err)) from None


def record_path(system_dir, calc_id):
    """File of the record of a calculation: ``<system>/<step>/<label>/record.json``."""
    step, label = split_calc_id(calc_id)
    return Path(system_dir) / step / label / RECORD_FILE


def write_record(system_dir, record):
    """Write the record of a calculation atomically.

    Parameters
    ----------
    system_dir : path-like
        Directory of the system; the calculation directory is created if
        missing.
    record : Record
        Record to write; its system must be the name of ``system_dir``.

    Returns
    -------
    pathlib.Path
        File written.
    """
    if record.system != system_name(system_dir):
        raise StoreError(f'record {record.calc_id} belongs to system {record.system!r}, '
                         f'not to {system_dir}')
    path = record_path(system_dir, record.calc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, record.to_dict())
    return path


def read_record(system_dir, calc_id):
    """Read the record of a calculation.

    Raises
    ------
    FileNotFoundError
        If the calculation has no record.
    StoreError
        If the file is not a valid record of that calculation of that system.
    """
    path = record_path(system_dir, calc_id)
    record = Record.from_dict(read_json(path), source=path)
    if record.calc_id != calc_id or record.system != system_name(system_dir):
        raise StoreError(f'{path}: record of {record.system}/{record.calc_id}, found in place '
                         f'of {system_name(system_dir)}/{calc_id}')
    return record


def iter_records(system_dir):
    """Records of all the calculations of a system, sorted by ``calc_id``."""
    system_dir = Path(system_dir)
    for path in sorted(system_dir.glob(f'*/*/{RECORD_FILE}')):
        yield read_record(system_dir, f'{path.parent.parent.name}/{path.parent.name}')


def unknown_parts(energy_model):
    """Parts of an energy model that are not known: its null values.

    Parameters
    ----------
    energy_model : dict

    Returns
    -------
    list of str
        Path of each null value, e.g. ``['species[0].md5', 'version']``;
        empty when the whole model is known.
    """
    found = []

    def visit(value, path):
        if value is None:
            found.append(path)
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f'{path}.{key}' if path else key)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                visit(item, f'{path}[{i}]')

    visit(energy_model, '')
    return found


def identity_differences(record):
    """Where the output of a record differs from its energy model.

    Each value of ``record.observed`` is compared with the value at the same
    path of ``record.energy_model``; keys that the energy model does not have
    are not compared. Two lists of different length differ as a whole.

    Parameters
    ----------
    record : Record

    Returns
    -------
    list of str
        One sentence per difference, e.g. ``"species[0].md5: expected 'ab...',
        observed 'cd...'"``; empty when nothing differs or nothing was
        observed.
    """
    found = []

    def compare(expected, observed, path):
        if isinstance(observed, dict) and isinstance(expected, dict):
            for key, item in observed.items():
                if key in expected:
                    compare(expected[key], item, f'{path}.{key}' if path else key)
        elif isinstance(observed, list) and isinstance(expected, list):
            if len(observed) != len(expected):
                found.append(f'{path}: expected {len(expected)} entries, observed '
                             f'{len(observed)}')
            else:
                for i, (e, o) in enumerate(zip(expected, observed)):
                    compare(e, o, f'{path}[{i}]')
        elif observed != expected:
            found.append(f'{path}: expected {expected!r}, observed {observed!r}')

    if record.observed is not None:
        compare(record.energy_model, record.observed, '')
    return found


def reuse_problem(record, fingerprint):
    """Why a record does not stand for a calculation with the given fingerprint.

    Parameters
    ----------
    record : Record or None
        Record found for the calculation, None if there is none.
    fingerprint : str
        Fingerprint expected now for the calculation.

    Returns
    -------
    str or None
        The reason, or None when the record can be reused (``can_reuse``).
    """
    check_fingerprint(fingerprint, 'expected fingerprint')
    if record is None:
        return 'no record'
    if record.fingerprint != fingerprint:
        return 'another fingerprint'
    if not record.is_complete or record.results is None:
        return 'not complete or without results'
    unknown = unknown_parts(record.energy_model)
    if unknown:
        return f'energy model not known ({", ".join(unknown)})'
    differences = identity_differences(record)
    if differences:
        return f'output of another energy model ({"; ".join(differences)})'
    return None


def can_reuse(record, fingerprint):
    """Whether a record stands for a calculation with the given fingerprint.

    Parameters
    ----------
    record : Record or None
        Record found for the calculation, None if there is none.
    fingerprint : str
        Fingerprint expected now for the calculation.

    Returns
    -------
    bool
        True only for a complete record with that fingerprint, with results,
        with an energy model fully known (``unknown_parts`` empty) and an
        output that agrees with it (``identity_differences`` empty).
    """
    return reuse_problem(record, fingerprint) is None


def check_energy_model(records):
    """Check that records share one energy model, as any energy difference requires.

    The energy model of a record is its engine (name and program) together
    with its ``energy_model``. A model with parts not known
    (``unknown_parts``) is never shared, not even with an identical one, and
    neither is a record whose output differs from its model
    (``identity_differences``).

    Raises
    ------
    StoreError
        If a record has parts of its energy model not known or an output of
        another model, or the records have more than one energy model; the
        message lists the calculations and the parts not known or the
        differences, or the calculations of each model.
    """
    records = list(records)
    unknown = []
    for record in records:
        parts = unknown_parts(record.energy_model)
        if parts:
            unknown.append(f'{record.calc_id} ({", ".join(parts)})')
    if unknown:
        raise StoreError('energies of an energy model not known cannot be combined: '
                         f'{"; ".join(unknown)}')
    other = []
    for record in records:
        differences = identity_differences(record)
        if differences:
            other.append(f'{record.calc_id} ({"; ".join(differences)})')
    if other:
        raise StoreError('energies of outputs of another energy model cannot be combined: '
                         f'{"; ".join(other)}')
    groups = {}
    for record in records:
        key = (record.engine.name, record.engine.program, canonical_json(record.energy_model))
        groups.setdefault(key, []).append(record.calc_id)
    if len(groups) > 1:
        models = '; '.join(f'{name} {program} {model}: {", ".join(ids)}'
                           for (name, program, model), ids in groups.items())
        raise StoreError(f'energies of different energy models cannot be combined: {models}')
