# Author: Vito Foderà
# Created: 2026-11-02

"""Energy model and fingerprint of a pw.x calculation.

The energy model (``energy_model``) is what two energies must share to be
combined, e.g. in an adhesion energy:

- the version of pw.x expected (``[qe] version``);
- every species, in the order of ``[[atomic_species]]``: label, name of the
  UPF file (the name identifies the pseudopotential: the content of the file
  is not checked) and its starting magnetization when given;
- the ``[k_points]`` settings, not the grid written: with the same density a
  bulk and a slab have different grids, and their energies are combined;
- the namelist keys of class ``'model'`` (``keys.py``), with their values.

It is the same for every calculation made from the same settings.

The fingerprint (``input_fingerprint``) identifies the input of one
calculation: engine and program, calc_type (in the workflows), energy model,
every namelist key written except those of class ``'attempt'`` and
``'resource'``, and the cards as written (species with their masses,
``K_POINTS``, cell, positions with the ``if_pos`` flags); for a calculation
that also gives the charge density, the post-processing too (pp.x and the
keys of its input, without those naming files, ``ppi.py``). Keys are taken in
lower case and without blanks, in the order written; values keep the type of
the settings, so ``40`` and ``40.0`` differ: at worst a calculation is run
again, a result is never reused wrongly.

The version is the only part of the energy model that can be unknown, None
without ``[qe] version``. The input is still generated, but the results of the
calculation are never reused and its energies never combined automatically
(``core.store.can_reuse`` and ``check_energy_model``).
"""

from dataclasses import asdict

from triboworkflow.core.settings.schema import NAMELISTS
from triboworkflow.core.store.fingerprint import fingerprint

from .keys import key_class, normal_key

ENGINE, PROGRAM = 'qe', 'pw.x'


def energy_model(settings):
    """Energy model of the calculations made from the settings.

    Parameters
    ----------
    settings : triboworkflow.core.settings.Settings

    Returns
    -------
    dict
        ``version`` (None if not given), ``species`` (``label``,
        ``pseudopotential`` and ``starting_magnetization`` when given),
        ``k_points`` (the keys of ``[k_points]`` that are set) and
        ``namelists`` (``{namelist: [[key, value], ...]}`` with the keys of
        class ``'model'``).
    """
    species = []
    for s in settings.atomic_species:
        entry = {'label': s.label, 'pseudopotential': s.pseudopotential}
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


def fingerprint_data(pw, model, calc_type=None, pp=None):
    """What the fingerprint of a calculation is computed from.

    Parameters
    ----------
    pw : triboworkflow.engines.qe.pwi.PwInput
        The input written.
    model : dict
        Its energy model (``energy_model``).
    calc_type : str, optional
        calc_type of a calculation of the workflows.
    pp : triboworkflow.engines.qe.ppi.PpInput, optional
        The pp.x input of a calculation that also gives the charge density.

    Returns
    -------
    dict
        ``engine``, ``program``, ``energy_model``, ``namelists`` (``{namelist:
        [[key, value], ...]}`` with the keys written, except those of class
        ``'attempt'`` and ``'resource'``), ``cards`` (``{card: [line, ...]}``)
        and, when given, ``calc_type`` and ``post_processing``
        (``PpInput.fingerprint_data``).
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
    if pp is not None:
        data['post_processing'] = pp.fingerprint_data()
    return data


def input_fingerprint(pw, model, calc_type=None, pp=None):
    """Fingerprint of a calculation (``fingerprint_data``): ``sha256:...``."""
    return fingerprint(fingerprint_data(pw, model, calc_type, pp), 'pw.x input')
