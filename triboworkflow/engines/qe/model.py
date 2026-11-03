# Author: Vito Foderà
# Created: 2026-11-02

"""Energy model and fingerprint of a pw.x calculation.

The energy model (``energy_model``) is what two energies must share to be
combined, e.g. in an adhesion energy:

- the version of pw.x expected (``[qe] version``);
- every species, in the order of ``[[atomic_species]]``: label, UPF file,
  its expected MD5 checksum (``[qe] upf_md5``) and its starting
  magnetization when given;
- the ``[k_points]`` settings, not the grid written: with the same density a
  bulk and a slab have different grids, and their energies are combined;
- the namelist keys of class ``'model'`` (``keys.py``), with their values.

It is the same for every calculation made from the same settings.

The fingerprint (``input_fingerprint``) identifies the input of one
calculation: engine and program, calc_type (in the workflows), energy model,
every namelist key written except those of class ``'attempt'`` and
``'resource'``, and the cards as written (species with their masses,
``K_POINTS``, cell, positions with the ``if_pos`` flags). Keys are taken in
lower case and without blanks, in the order written; values keep the type of
the settings, so ``40`` and ``40.0`` differ: at worst a calculation is run
again, a result is never reused wrongly.

A part of the energy model that is not known is None: the version without
``[qe] version``, the checksums without ``[qe] upf_md5``. The input is still
generated, but the results of the calculation are never reused and its
energies never combined automatically (``core.store.can_reuse`` and
``check_energy_model``). The path of the manifest does not enter: only the
checksums read from it.
"""

import re
from dataclasses import asdict
from pathlib import Path

from triboworkflow.core.settings.schema import NAMELISTS, SettingsError
from triboworkflow.core.store.fingerprint import fingerprint

from .keys import key_class, normal_key

ENGINE, PROGRAM = 'qe', 'pw.x'

# A line of md5sum: checksum, then two blanks (text mode) or a blank and an
# asterisk (binary mode), then the file name.
_MD5_LINE = re.compile(r'([0-9a-f]{32}) [ *](.+)')


def read_upf_md5(path):
    """Expected MD5 checksums of UPF files, from a manifest in the format of ``md5sum``.

    Parameters
    ----------
    path : path-like
        Manifest, e.g. made with ``md5sum *.UPF > pseudo.md5`` in the
        ``pseudo_dir``: one line ``<checksum>  <file>`` per file. Blank lines
        are skipped.

    Returns
    -------
    dict
        ``{file name: checksum}``, the checksum in 32 lower-case hex digits,
        as pw.x prints it.

    Raises
    ------
    OSError
        If the file cannot be read.
    ValueError
        For a line in another format or a file given twice.
    """
    checksums, lines = {}, {}
    text = Path(path).read_text(encoding='utf-8')
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        match = _MD5_LINE.fullmatch(line)
        if not match:
            raise ValueError(f'{path}, line {n}: {line!r} is not "<checksum>  <file>" with a '
                             'checksum of 32 lower-case hex digits (format of md5sum)')
        checksum, name = match.groups()
        if name in checksums:
            raise ValueError(f'{path}, line {n}: {name} is already at line {lines[name]}')
        checksums[name], lines[name] = checksum, n
    return checksums


def expected_md5(settings):
    """Expected MD5 checksum of the UPF file of each species.

    Parameters
    ----------
    settings : triboworkflow.core.settings.Settings
        ``[qe] upf_md5``, relative to the folder of ``settings.source``.

    Returns
    -------
    tuple of str or None
        One per species, in the order of ``[[atomic_species]]``; all None
        without a manifest.

    Raises
    ------
    SettingsError
        For a manifest that cannot be read or is not valid
        (``read_upf_md5``), or without the file of a species.
    """
    species = settings.atomic_species
    if settings.qe.upf_md5 is None:
        return (None,) * len(species)
    path = Path(settings.qe.upf_md5)
    if not path.is_absolute() and settings.source is not None:
        path = Path(settings.source).parent / path
    try:
        checksums = read_upf_md5(path)
    except OSError as error:
        raise SettingsError('qe.upf_md5', f'cannot read the manifest: {error}',
                            settings.source) from None
    except ValueError as error:
        raise SettingsError('qe.upf_md5', str(error), settings.source) from None
    missing = [s.pseudopotential for s in species if s.pseudopotential not in checksums]
    if missing:
        raise SettingsError('qe.upf_md5', f'{path}: no checksum for '
                            f'{", ".join(dict.fromkeys(missing))}', settings.source)
    return tuple(checksums[s.pseudopotential] for s in species)


def energy_model(settings):
    """Energy model of the calculations made from the settings.

    Parameters
    ----------
    settings : triboworkflow.core.settings.Settings

    Returns
    -------
    dict
        ``version`` (None if not given), ``species`` (``label``,
        ``pseudopotential``, ``md5``, None without a manifest, and
        ``starting_magnetization`` when given), ``k_points`` (the keys of
        ``[k_points]`` that are set) and ``namelists`` (``{namelist: [[key,
        value], ...]}`` with the keys of class ``'model'``).

    Raises
    ------
    SettingsError
        As ``expected_md5``.
    """
    species = []
    for s, md5 in zip(settings.atomic_species, expected_md5(settings)):
        entry = {'label': s.label, 'pseudopotential': s.pseudopotential, 'md5': md5}
        if s.starting_magnetization is not None:
            entry['starting_magnetization'] = s.starting_magnetization
        species.append(entry)
    return {
        'version': settings.qe.version,
        'species': species,
        'k_points': {k: v for k, v in asdict(settings.k_points).items() if v is not None},
        'namelists': {name: [[normal_key(k), v] for k, v in getattr(settings.namelists, name).items()
                             if key_class(k) == 'model'] for name in NAMELISTS},
    }


def fingerprint_data(pw, model, calc_type=None):
    """What the fingerprint of a calculation is computed from.

    Parameters
    ----------
    pw : triboworkflow.engines.qe.pwi.PwInput
        The input written.
    model : dict
        Its energy model (``energy_model``).
    calc_type : str, optional
        calc_type of a calculation of the workflows.

    Returns
    -------
    dict
        ``engine``, ``program``, ``energy_model``, ``namelists`` (``{namelist:
        [[key, value], ...]}`` with the keys written, except those of class
        ``'attempt'`` and ``'resource'``), ``cards`` (``{card: [line, ...]}``)
        and, when given, ``calc_type``.
    """
    data = {
        'engine': ENGINE, 'program': PROGRAM, 'energy_model': model,
        'namelists': {name: [[normal_key(k), v] for k, v in entries
                             if key_class(k) not in ('attempt', 'resource')]
                      for name, entries in pw.namelists},
        'cards': {name: list(lines) for name, lines in pw.cards},
    }
    if calc_type is not None:
        data['calc_type'] = calc_type
    return data


def input_fingerprint(pw, model, calc_type=None):
    """Fingerprint of a calculation (``fingerprint_data``): ``sha256:...``."""
    return fingerprint(fingerprint_data(pw, model, calc_type), 'pw.x input')
