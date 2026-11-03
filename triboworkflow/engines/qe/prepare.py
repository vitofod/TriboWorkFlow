# Author: Vito Foderà
# Created: 2026-11-02

"""A pw.x calculation, as prepared, and compared with one on disk.

``qe_calculation`` gives everything the directory of a calculation of the
workflows starts with, without writing it: the text of the input
(``workflow_input``), the text of the extxyz of its structure with the
constraints written in the input (``move_mask``, for a look in OVITO), and
its record, with the energy model and the fingerprint (``model.py``) and no
attempt. The changes made to the settings (``PwInput.changes``) come with
it, to be logged.

``single_input`` gives the input and the extxyz of the single calculation of
``[structure]``: structure and constraints of the file and of the settings
(``load_structure``), ``K_POINTS`` of ``[k_points]`` and every ``&CONTROL``
key as written in the settings, ``calculation``, ``prefix`` and ``outdir``
included.

``input_differences`` compares an input prepared earlier with the one the
settings give now: equal, different only in keys of class ``'attempt'`` or
``'resource'`` (``keys.py``), whose values are listed, or different elsewhere.
Each key is alone on its line (``pwi.py``), so the comparison goes line by
line.
"""

import re
from dataclasses import dataclass

from triboworkflow.core.store import Engine, Record
from triboworkflow.core.store.record import split_calc_id
from triboworkflow.core.structures import (calc_type_if_pos, extxyz_text, load_structure,
                                           structure_differences, with_if_pos)

from .keys import key_class
from .model import ENGINE, PROGRAM, energy_model, input_fingerprint
from .pwi import kpoints_card, pw_input, workflow_input

# Classes of the keys that a difference of input leaves to the attempts.
ATTEMPT_CLASSES = ('attempt', 'resource')

# A key line of a namelist: key, value.
_ENTRY = re.compile(r'\s*([A-Za-z][A-Za-z0-9_]*(?:\([^)]*\))?)\s*=\s*(.*?)\s*')


@dataclass(frozen=True)
class InputDifference:
    """How an input on disk differs from the one prepared now.

    Attributes
    ----------
    kind : {'same', 'attempt', 'other'}
        Equal; different only in keys of ``ATTEMPT_CLASSES``; different
        elsewhere.
    keys : tuple of (str, str or None, str or None)
        For ``'attempt'``: each key that differs (``namelists.<NAME>.<key>``),
        the value on disk and the value requested now, as written in the
        files, None where the key is absent.
    """

    kind: str
    keys: tuple = ()

    def messages(self):
        """One sentence per key of ``keys``, e.g. ``... max_seconds: requested 3600, kept 300``."""
        return tuple(f'{key}: requested {"absent" if new is None else new}, kept '
                     f'{"absent" if kept is None else kept}' for key, kept, new in self.keys)


def _split(text):
    """Lines of an input without the keys of ``ATTEMPT_CLASSES``, and those keys apart."""
    lines, apart, namelist = [], {}, None
    for line in text.splitlines():
        stripped = line.strip()
        if namelist is None and stripped.startswith('&'):
            namelist = stripped[1:].upper()
        elif namelist is not None and stripped == '/':
            namelist = None
        elif namelist is not None:
            match = _ENTRY.fullmatch(line)
            if match and key_class(match.group(1)) in ATTEMPT_CLASSES:
                apart[f'namelists.{namelist}.{match.group(1)}'] = match.group(2)
                continue
        lines.append(line)
    return lines, apart


def input_differences(kept, new):
    """How the input ``kept`` differs from ``new``.

    Parameters
    ----------
    kept, new : str
        Texts of an input prepared earlier and of the one prepared now.

    Returns
    -------
    InputDifference
    """
    if kept == new:
        return InputDifference('same')
    kept_lines, kept_keys = _split(kept)
    new_lines, new_keys = _split(new)
    if kept_lines != new_lines:
        return InputDifference('other')
    keys = tuple((key, kept_keys.get(key), new_keys.get(key))
                 for key in dict.fromkeys([*new_keys, *kept_keys])
                 if kept_keys.get(key) != new_keys.get(key))
    return InputDifference('attempt', keys) if keys else InputDifference('other')


@dataclass(frozen=True)
class QECalculation:
    """A pw.x calculation as the settings give it now, nothing written.

    Attributes
    ----------
    calc_id : str
        ``<step>/<label>``.
    input_text : str
        Text of its input.
    structure_text : str
        Text of the extxyz of its structure, with the ``if_pos`` of the input
        as ``move_mask``.
    record : triboworkflow.core.store.Record
        Its record before any attempt; ``files`` is set where it is written.
    changes : tuple of str
        Changes made to the settings in the input (``PwInput.changes``).
    """

    calc_id: str
    input_text: str
    structure_text: str
    record: Record
    changes: tuple = ()

    def input_differences(self, kept):
        """``input_differences`` of an input on disk from this one."""
        return input_differences(kept, self.input_text)

    def structure_differences(self, kept):
        """``structure_differences`` of an extxyz on disk from this one."""
        return structure_differences(kept, self.structure_text)


def qe_calculation(system, calc_id, calc_type, structure, kpoints, settings, prefix, outdir):
    """A pw.x calculation of the workflows, from a structure of ``VaspInputSet``.

    Parameters
    ----------
    system, calc_id : str
        System and ``<step>/<label>`` of the calculation.
    calc_type : str
        One of ``pwi.CALC_TYPES``.
    structure, kpoints
        Structure and k-points of the vendored ``VaspInputSet`` for
        ``calc_type`` (``workflow_input``).
    settings : triboworkflow.core.settings.Settings
    prefix, outdir : str
        ``prefix`` and ``outdir`` of the calculation.

    Returns
    -------
    QECalculation

    Raises
    ------
    PwInputError
        As ``workflow_input``.
    SettingsError
        As ``energy_model``.
    StructureError
        If the structure cannot be written as extxyz.
    """
    pw = workflow_input(structure, kpoints, calc_type, settings, prefix, outdir)
    model = energy_model(settings)
    step, label = split_calc_id(calc_id)
    record = Record(system=system, step=step, label=label, calc_type=calc_type,
                    engine=Engine(ENGINE, PROGRAM), energy_model=model,
                    fingerprint=input_fingerprint(pw, model, calc_type))
    shown = with_if_pos(structure, calc_type_if_pos(structure))
    return QECalculation(calc_id, pw.text, extxyz_text(shown), record, pw.changes)


@dataclass(frozen=True)
class SingleInput:
    """The input of the single calculation of ``[structure]`` and the extxyz of its structure.

    Attributes
    ----------
    pw : triboworkflow.engines.qe.pwi.PwInput
        The input, with the changes made to the settings.
    structure_text : str
        Text of the extxyz: cell, positions and labels of the input, its
        ``if_pos`` as ``move_mask``.
    structure : triboworkflow.core.structures.StructureInput
        The structure read and the constraints applied, with their sources.
    """

    pw: object
    structure_text: str
    structure: object


def single_input(settings):
    """The pw.x input of the single calculation of ``[structure]``, nothing written.

    Parameters
    ----------
    settings : triboworkflow.core.settings.Settings
        With ``[structure]``, every placeholder filled.

    Returns
    -------
    SingleInput

    Raises
    ------
    SettingsError
        If a placeholder is left (``require_complete``).
    StructureError
        If the structure cannot be read or its constraints resolved.
    PwInputError
        As ``pw_input`` and ``kpoints_card`` (``kdens`` with
        ``material_type='auto'``).
    """
    settings.require_complete()
    loaded = load_structure(settings)
    card = kpoints_card(settings.k_points, loaded.structure)
    pw = pw_input(loaded.structure, settings.atomic_species, settings.namelists, card,
                  loaded.constraints)
    shown = with_if_pos(loaded.structure, loaded.constraints)
    return SingleInput(pw, extxyz_text(shown), loaded)
