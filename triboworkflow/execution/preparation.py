# Author: Vito Foderà
# Created: 2026-11-02

"""Preparation of the calculations of a step in their directories, and their replacement.

A calculation is given as the settings make it now, nothing written (e.g.
``QECalculation``): its ``calc_id``, the texts of its input and of the extxyz
of its structure, its record before any attempt, and the changes made to the
settings. ``prepare_calculations`` writes, in
``<run>/systems/<system>/<step>/<label>/``, ``input.pwi``,
``structure_in.extxyz`` and, last, ``record.json``: a directory with the
first two files and no record is a preparation interrupted, written again.

What is already on disk is compared first (``check_calculations``), and
nothing is overwritten:

- ``current``: same fingerprint, same input, same structure;
- ``kept``: same fingerprint, and the input differs only in keys of the
  attempts or of the resources (e.g. ``mixing_beta``, ``max_seconds``), or
  the extxyz only in data that the input does not take (e.g. the moments):
  the files are kept and the differences logged, with the value kept and the
  one requested;
- ``changed``: another fingerprint, a record that cannot be read, an input
  that differs elsewhere (e.g. changed by hand), an extxyz whose cell,
  positions, labels or constraints differ from the input, a file missing, or
  a directory without record that holds other files. The preparation stops
  with the list of these calculations, and writes nothing.

The changes made to the settings in a new input (e.g. ``nstep = 1`` in an
scf) are logged, one line per calculation and change, with the values.

A replacement, asked by the user, moves the directories of the calculations
``kept`` and ``changed`` to ``<run>/replaced/`` (``replacement.py``), so that
they are prepared again. Every check comes before any change, and the
replacement is refused, with the reason of each calculation, if:

- the calculation has a claim, even empty: a worker holds it, or stopped
  right after taking it;
- a group of jobs listing it may still run or submit a job: a job not in a
  final state confirmed by the last query, a submission of outcome not
  known or interrupted, jobs still to submit. A group submitted in part
  (some jobs over, others refused or held back) is one of these: the message
  says so and how to go on, since this version does not handle it.

A group listing it that never had a job submitted is withdrawn; a complete
group keeps its launches for the other calculations, and its launches of
the calculation are retired. Then the journal is written, the groups
withdrawn and the directories moved; ``Orchestrator.replace_step`` sets the
step back to waiting, marks the journal done and prepares the step. A
replacement interrupted is completed by asking it again: what is already
done is not done twice, and nothing is deleted.
"""

import dataclasses
import logging
import os
from dataclasses import dataclass

from triboworkflow.core.store import read_record, write_record, write_text
from triboworkflow.core.store.record import RECORD_FILE, split_calc_id
from triboworkflow.core.structures import StructureError
from triboworkflow.engines.qe.prepare import qe_calculation

from .layout import (CLAIM_FILE, INPUT_FILE, OUTDIR, STRUCTURE_IN_FILE, calc_dir, calc_prefix,
                     system_dir)
from .reconcile import read_groups
from .replacement import new_replacement, retired_launches, unfinished

logger = logging.getLogger(__name__)

CHECK_KINDS = ('new', 'interrupted', 'current', 'kept', 'changed')
REPLACED_KINDS = ('kept', 'changed')

# Files that a preparation writes before the record.
_PREPARED_FILES = (INPUT_FILE, STRUCTURE_IN_FILE)


class PreparationError(ValueError):
    """Calculations that differ from the settings, or a preparation that cannot run.

    Attributes
    ----------
    checks : tuple of CalcCheck
        The checks of the calculations, when the error comes from them.
    """

    def __init__(self, message, checks=()):
        super().__init__(message)
        self.checks = tuple(checks)


class ReplacementError(ValueError):
    """A replacement refused, with the calculations and why, or one that cannot be completed."""


@dataclass(frozen=True)
class CalcCheck:
    """A calculation of the settings compared with its directory.

    Attributes
    ----------
    calc_id : str
    kind : str
        One of ``CHECK_KINDS`` (see the module docstring).
    reasons : tuple of str
        Why a calculation is ``changed``.
    warnings : tuple of str
        What differs in a calculation ``kept``.
    """

    calc_id: str
    kind: str
    reasons: tuple = ()
    warnings: tuple = ()


def workflow_calculation(system, calc_id, calc_type, structure, kpoints, settings):
    """A pw.x calculation of the workflows, with its own ``prefix`` and ``outdir``.

    ``qe_calculation`` with ``prefix = calc_prefix(calc_id)`` and ``outdir =
    OUTDIR``; the other parameters are the same.
    """
    return qe_calculation(system, calc_id, calc_type, structure, kpoints, settings,
                          calc_prefix(calc_id), OUTDIR)


def _is_temporary(name):
    return name.startswith('.') and name.endswith('.tmp')


def _check(run_dir, system, calc):
    calc_id = calc.calc_id
    path = calc_dir(run_dir, system, calc_id)
    if not (path / RECORD_FILE).exists():
        if not path.exists():
            return CalcCheck(calc_id, 'new')
        other = sorted(p.name for p in path.iterdir()
                       if p.name not in _PREPARED_FILES and not _is_temporary(p.name))
        if not other:
            return CalcCheck(calc_id, 'interrupted')
        return CalcCheck(calc_id, 'changed', (f'no {RECORD_FILE}, and the directory holds '
                                              f'{", ".join(other)}',))
    try:
        record = read_record(system_dir(run_dir, system), calc_id)
    except (OSError, ValueError) as error:
        return CalcCheck(calc_id, 'changed', (f'{RECORD_FILE} not read: {error}',))
    if record.fingerprint != calc.record.fingerprint:
        return CalcCheck(calc_id, 'changed', ('fingerprint differs from the settings',))
    reasons, warnings = [], []
    if not (path / INPUT_FILE).is_file():
        reasons.append(f'{INPUT_FILE} missing')
    else:
        difference = calc.input_differences((path / INPUT_FILE).read_text(encoding='utf-8'))
        if difference.kind == 'other':
            reasons.append(f'{INPUT_FILE} differs from the one the settings give beyond the keys '
                           'of the attempts and resources (changed by hand?)')
        warnings += [f'{INPUT_FILE}: {message}' for message in difference.messages()]
    if not (path / STRUCTURE_IN_FILE).is_file():
        reasons.append(f'{STRUCTURE_IN_FILE} missing')
    else:
        text = (path / STRUCTURE_IN_FILE).read_text(encoding='utf-8')
        if text != calc.structure_text:
            try:
                fields, other = calc.structure_differences(text)
            except StructureError as error:
                reasons.append(f'{STRUCTURE_IN_FILE} not read: {error}')
            else:
                if fields:
                    reasons.append(f'{STRUCTURE_IN_FILE} does not match the input: '
                                   f'{", ".join(fields)} differ')
                else:
                    warnings.append(f'{STRUCTURE_IN_FILE}: {", ".join(other) or "its text"} '
                                    'differ from the structure of the settings, not written in '
                                    'the input')
    if reasons:
        return CalcCheck(calc_id, 'changed', tuple(reasons), tuple(warnings))
    return CalcCheck(calc_id, 'kept' if warnings else 'current', (), tuple(warnings))


def _checked_list(system, calcs):
    calcs = list(calcs)
    seen = set()
    for calc in calcs:
        if calc.calc_id in seen:
            raise PreparationError(f'{system}/{calc.calc_id} is given twice')
        seen.add(calc.calc_id)
        if (calc.record.system, calc.record.calc_id) != (system, calc.calc_id):
            raise PreparationError(f'{system}/{calc.calc_id}: its record is of '
                                   f'{calc.record.system}/{calc.record.calc_id}')
    return calcs


def check_calculations(run_dir, system, calcs):
    """Compare each calculation of the settings with its directory; nothing is written.

    Parameters
    ----------
    run_dir : path-like
        Directory of the run.
    system : str
    calcs : iterable
        The calculations as the settings make them now (e.g.
        ``QECalculation``).

    Returns
    -------
    list of CalcCheck
        In the order of ``calcs``.

    Raises
    ------
    PreparationError
        If a calculation is given twice or its record is of another one.
    """
    return [_check(run_dir, system, calc) for calc in _checked_list(system, calcs)]


def _unfinished_message(found):
    return '; '.join(f'replacement of {r.system}/{r.step} ({r.path}) not completed: run '
                     f'"triboworkflow prepare {r.step} --replace" to complete it' for r in found)


def _changed_message(system, changed, step):
    listed = '; '.join(f'{c.calc_id}: {", ".join(c.reasons)}' for c in changed)
    how = f'"triboworkflow prepare {step} --replace"' if step else 'a replacement'
    return (f'{len(changed)} calculations of {system} differ from the settings and are not '
            f'overwritten: {listed}. Restore the settings, or replace them with {how}: their '
            'directories are kept in replaced/')


def prepare_calculations(run_dir, system, calcs, step=None):
    """Write the calculations of a step that are not on disk yet.

    Parameters
    ----------
    run_dir : path-like
        Directory of the run.
    system : str
    calcs : iterable
        The calculations as the settings make them now (e.g.
        ``QECalculation``).
    step : str or None, optional
        Step of the workflow, named in the messages.

    Returns
    -------
    list of str
        ``calc_id`` of every calculation, in order.

    Raises
    ------
    PreparationError
        If a calculation is ``changed`` (nothing is written then; the error
        lists them, with ``checks``), or a replacement of the run is not
        completed.
    """
    found = unfinished(run_dir)
    if found:
        raise PreparationError(_unfinished_message(found))
    calcs = _checked_list(system, calcs)
    checks = [_check(run_dir, system, calc) for calc in calcs]
    changed = [check for check in checks if check.kind == 'changed']
    if changed:
        raise PreparationError(_changed_message(system, changed, step), checks)
    kept = False
    for calc, check in zip(calcs, checks):
        for warning in check.warnings:
            logger.warning('%s/%s: kept as prepared, %s', system, calc.calc_id, warning)
            kept = True
        if check.kind in ('new', 'interrupted'):
            _write(run_dir, system, calc)
    if kept:
        how = f'"triboworkflow prepare {step} --replace"' if step else 'a replacement'
        logger.warning('The calculations kept as prepared run with the values of their input; '
                       'to apply the settings, replace them with %s', how)
    return [calc.calc_id for calc in calcs]


def _write(run_dir, system, calc):
    """Write input, extxyz and, last, the record of a calculation; log its changes."""
    path = calc_dir(run_dir, system, calc.calc_id)
    path.mkdir(parents=True, exist_ok=True)
    write_text(path / INPUT_FILE, calc.input_text)
    write_text(path / STRUCTURE_IN_FILE, calc.structure_text)
    record = dataclasses.replace(calc.record, files={'input': INPUT_FILE,
                                                     'structure': STRUCTURE_IN_FILE})
    write_record(system_dir(run_dir, system), record)
    for change in calc.changes:
        logger.warning('%s/%s: %s', system, calc.calc_id, change)


# --- replacement -----------------------------------------------------------------

def start_replacement(run_dir, system, step, calcs, manager, command, at):
    """Check a replacement and write its journal; nothing else is changed.

    Parameters
    ----------
    run_dir : path-like
        Directory of the run.
    system, step : str
        System and step of the workflow.
    calcs : iterable
        The calculations of the step as the settings make them now.
    manager : triboworkflow.execution.manager.JobManager
        Job manager of the run, read only.
    command : str or None
        Command that asked for the replacement, for the journal.
    at : str
        Time of the replacement, ISO 8601.

    Returns
    -------
    triboworkflow.execution.replacement.Replacement or None
        None when no calculation is ``kept`` or ``changed``.

    Raises
    ------
    ReplacementError
        If a replacement of the run is not completed, or a calculation to
        replace has a claim or a group that may still run it: the message
        lists them all, and nothing is written.
    """
    found = unfinished(run_dir)
    if found:
        raise ReplacementError(_unfinished_message(found))
    calcs = _checked_list(system, calcs)
    targets = [(calc, check) for calc, check in zip(calcs, (_check(run_dir, system, c)
                                                            for c in calcs))
               if check.kind in REPLACED_KINDS]
    if not targets:
        return None
    groups = read_groups(run_dir)
    retired_before = retired_launches(run_dir)
    sdir = system_dir(run_dir, system)
    entries, withdraw, problems = {}, set(), []
    for calc, check in targets:
        calc_id = calc.calc_id
        blockers = []
        if (calc_dir(run_dir, system, calc_id) / CLAIM_FILE).exists():
            blockers.append('it has a claim: a worker holds it, or stopped right after taking it '
                            '("triboworkflow status" tells)')
        try:
            old = read_record(sdir, calc_id)
        except (OSError, ValueError):
            old = None
        made = len(old.attempts) if old is not None else 0
        before = retired_before.get((system, calc_id), {})
        retired = {}
        for name, manifest in groups.items():
            listed = [launch.n for launch in manifest.calculations
                      if launch.calc == (system, calc_id)]
            if not listed or name in before:
                continue
            reason = manager.open_reason(name)
            if reason is None:
                retired[name] = any(made >= n for n in listed)
            elif manager.withdrawal_problem(name) is None:
                withdraw.add(name)
            elif manager.partly_submitted(name) is not None:
                blockers.append(_partly_submitted(name, step, *manager.partly_submitted(name)))
            else:
                blockers.append(f'group {name}: {reason}')
        if blockers:
            problems.append(f'{calc_id}: {"; ".join(blockers)}')
        entries[calc_id] = {'reasons': list(check.reasons + check.warnings),
                            'old_fingerprint': old.fingerprint if old is not None else None,
                            'new_fingerprint': calc.record.fingerprint, 'retired': retired}
    if problems:
        raise ReplacementError(f'{system}/{step}: nothing replaced, since jobs or workers may '
                               f'still use these calculations: {"; ".join(problems)}')
    replacement = new_replacement(run_dir, system, step, {
        'at': at, 'command': command, 'done': False, 'calculations': entries,
        'withdrawn': sorted(withdraw)})
    logger.info('Replacement of %s/%s: journal %s', system, step, replacement.path)
    return replacement


def _partly_submitted(name, step, submitted, missing, refused):
    """Why a group submitted in part stops a replacement, and how to go on."""
    why = 'its last submission was refused' if refused else 'held back since the settings changed'
    clear = f' and "triboworkflow clear-refusal {name}"' if refused else ''
    return (f'group {name}: {submitted} of its jobs were submitted and are over, {missing} were '
            f'never submitted ({why}). A group submitted in part is neither complete nor can it '
            'be withdrawn, since its other jobs may still be submitted, and this version does not '
            f'replace a calculation it lists. To go on: restore the settings, "triboworkflow retry '
            f'{step}" and "triboworkflow prepare {step}"{clear}, let the group end on the HPC '
            f'machine ("triboworkflow submit {step}" or "start"), then change the settings again '
            'and replace')


def complete_replacement(run_dir, replacement, manager):
    """Withdraw the groups and move the directories of a replacement; again if interrupted.

    A group already withdrawn and a directory already moved are left as
    they are. The claim of a calculation is checked again before its
    directory moves.

    Raises
    ------
    ReplacementError
        If a directory and its copy both exist or are both missing, or a
        calculation not moved yet has a claim now.
    JobManagerError
        If a group to withdraw has a job now.
    """
    reason = f'calculations of {replacement.system}/{replacement.step} replaced ' \
             f'({replacement.path})'
    for name in replacement.withdrawn:
        manager.withdraw(name, reason)
    for calc_id in replacement.calculations:
        split_calc_id(calc_id)
        source = calc_dir(run_dir, replacement.system, calc_id)
        target = replacement.old_dir(calc_id)
        if target.exists():
            if source.exists():
                raise ReplacementError(f'{calc_id}: both {source} and its copy {target} exist; '
                                       'check them by hand')
            continue
        if not source.exists():
            raise ReplacementError(f'{calc_id}: neither {source} nor its copy {target} exists')
        if (source / CLAIM_FILE).exists():
            raise ReplacementError(f'{calc_id}: a claim appeared in {source}; not moved')
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, target)
        logger.info('%s/%s: directory moved to %s', replacement.system, calc_id, target)
