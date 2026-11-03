# Author: Vito Foderà
# Created: 2026-11-02

"""Journal of a replacement of calculations, and the launches it retires.

A replacement moves the directories of some calculations of one step to
``<run>/replaced/<system>/<step>/<k>/`` and prepares them again
(``preparation.py``). Moving several directories is not one atomic
operation, so the journal ``replacement.json`` is written first, in the
same folder, and marked ``done`` last:

- ``calculations``: for each calculation replaced, why, the fingerprints
  before and after, and the groups of jobs, all complete, whose launches of
  it are retired, each with whether one of its workers took the attempt of
  the calculation replaced;
- ``withdrawn``: the groups none of whose jobs was ever submitted, withdrawn
  because they list a calculation replaced.

From the moment the journal is written, the launches it retires no longer
count for the calculation (``retired_launches``): the calculation prepared
again starts from attempt 1, and an attempt of an old group with the same
number is not one of its attempts. The other calculations of a group keep
its launches, and whether the group made progress stays the one recorded. A
replacement not done is completed before anything else changes the run
(``unfinished``).
"""

from dataclasses import dataclass
from pathlib import Path

from triboworkflow.core.store.atomic import StoreError, read_json, write_json
from triboworkflow.core.store.record import check_keys

from .layout import REPLACED_DIR, REPLACEMENT_FILE, replaced_dir

JOURNAL_VERSION = 1

_JOURNAL_KEYS = ('schema_version', 'system', 'step', 'at', 'command', 'done', 'calculations',
                 'withdrawn')
_CALC_KEYS = ('reasons', 'old_fingerprint', 'new_fingerprint', 'retired')


class ReplacementJournalError(StoreError):
    """A journal of a replacement that cannot be read."""


@dataclass(frozen=True)
class Replacement:
    """One replacement of calculations of a step, as its journal tells.

    Attributes
    ----------
    path : pathlib.Path
        Its folder, ``<run>/replaced/<system>/<step>/<k>``.
    data : dict
        Content of ``replacement.json``.
    """

    path: Path
    data: dict

    @property
    def system(self):
        return self.data['system']

    @property
    def step(self):
        return self.data['step']

    @property
    def done(self):
        return self.data['done']

    @property
    def calculations(self):
        """``{calc_id: {'reasons', 'old_fingerprint', 'new_fingerprint', 'retired'}}``."""
        return self.data['calculations']

    @property
    def withdrawn(self):
        """Groups withdrawn by the replacement."""
        return self.data['withdrawn']

    def old_dir(self, calc_id):
        """Where the directory of a calculation replaced is kept: ``<k>/<step>/<label>``."""
        return self.path / Path(*calc_id.split('/'))


def read_replacement(path):
    """The replacement of folder ``path``.

    Raises
    ------
    ReplacementJournalError
        If ``replacement.json`` is missing or not valid.
    """
    path = Path(path)
    file = path / REPLACEMENT_FILE
    try:
        data = check_keys(read_json(file), _JOURNAL_KEYS, 'replacement')
        if data['schema_version'] != JOURNAL_VERSION:
            raise StoreError(f'schema_version: {data["schema_version"]!r}; this version reads '
                             f'{JOURNAL_VERSION}')
        if type(data['done']) is not bool:
            raise StoreError(f'done: {data["done"]!r} is not true or false')
        if not isinstance(data['calculations'], dict) or not isinstance(data['withdrawn'], list):
            raise StoreError('calculations must be an object and withdrawn an array')
        for calc_id, entry in data['calculations'].items():
            check_keys(entry, _CALC_KEYS, f'calculations.{calc_id}')
    except (OSError, ValueError) as error:
        raise ReplacementJournalError(f'{file}: {error}') from None
    return Replacement(path, data)


def new_replacement(run_dir, system, step, data):
    """Write the journal of a new replacement in the next free folder ``<k>``.

    Parameters
    ----------
    run_dir : path-like
    system, step : str
    data : dict
        Content of the journal, without ``schema_version``, ``system`` and
        ``step``.

    Returns
    -------
    Replacement
    """
    root = replaced_dir(run_dir, system, step)
    root.mkdir(parents=True, exist_ok=True)
    taken = [int(p.name) for p in root.iterdir() if p.name.isdigit()]
    path = root / str(max(taken, default=0) + 1)
    path.mkdir()
    content = {'schema_version': JOURNAL_VERSION, 'system': system, 'step': step, **data}
    write_json(path / REPLACEMENT_FILE, content)
    return Replacement(path, check_keys(content, _JOURNAL_KEYS, 'replacement'))


def mark_done(replacement):
    """Write the journal of a replacement as done; return it."""
    data = {**replacement.data, 'done': True}
    write_json(replacement.path / REPLACEMENT_FILE, data)
    return Replacement(replacement.path, data)


def replacements(run_dir):
    """Every replacement of a run, in the order they were made for each step.

    A folder whose journal was never written (a replacement interrupted
    before it) is skipped: nothing of the run was changed.
    """
    root = Path(run_dir) / REPLACED_DIR
    found = []
    if root.is_dir():
        for path in root.glob(f'*/*/*/{REPLACEMENT_FILE}'):
            if path.parent.name.isdigit():
                found.append(path.parent)
    found.sort(key=lambda p: (p.parent.parent.name, p.parent.name, int(p.name)))
    return [read_replacement(path) for path in found]


def unfinished(run_dir):
    """The replacements of a run not marked done."""
    return [r for r in replacements(run_dir) if not r.done]


def retired_launches(run_dir):
    """Launches retired by the replacements of a run.

    Returns
    -------
    dict
        ``{(system, calc_id): {group: progress}}``: the groups whose launches
        no longer count for the calculation, each with whether a worker of
        the group took the attempt of the calculation replaced. A group
        retired by two replacements of a calculation keeps the first value.
    """
    retired = {}
    for replacement in replacements(run_dir):
        for calc_id, entry in replacement.calculations.items():
            groups = retired.setdefault((replacement.system, calc_id), {})
            for name, progress in entry['retired'].items():
                groups.setdefault(name, progress)
    return retired
