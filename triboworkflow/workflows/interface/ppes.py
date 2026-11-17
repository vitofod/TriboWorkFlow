# Author: Vito Foderà
# Created: 2026-11-08

"""Step ``ppes`` of the interface workflow: the perpendicular potential energy surface.

The perpendicular PES (PPES) is the energy of an interface as the two slabs
move apart or closer along z, rigidly. The step starts from the interface that
the step ``adhesion`` relaxed at the site of lowest energy (``min_site``) or at
the site of highest energy (``max_site``) of its surface, moves the slab on top
by each of the shifts of ``[interface.ppes]`` and runs one pw.x calculation
without relaxation (``scf``, ``calc_type`` ``slab_from_scratch``) for each:
``ppes/<site>_<shift>``, with the shift in angstrom as it is written in the
settings (``min_site_-0.75``). The kind of interface, ``regular`` or ``short``,
is the one of ``[interface.adhesion]`` (D57); a run covers one site and one
kind, and the results of the other site or kind stay in the document. The
default shifts (``PPES_SHIFTS``) are 15 values from -0.75 to 3 angstrom with a
step of 0.25 that jumps from 2.25 to 2.75: there is no 2.5 (A90); the fit
(``fit = "uber"``, off by default) uses the shifts that are zero or positive.

The atoms to move are those of the slab on top, each told by the slab it belongs
to and not by its position in the cell: the site property ``interface_region``
(``bot_slab`` or ``top_slab``) that ``match`` gives to every atom of an
interface and that the pw.x adapter returns with the final structure. Slabs of
any thickness, an interface whose origin is anywhere and coordinates wrapped or
not therefore move the slab on top as a whole, by the same shift in z, and the
cell and the atoms of the bottom slab stay where they are. The slabs are never
guessed from the elements, which may be the same in both. The map is checked
before any calculation is prepared (``slab_on_top``): the interface of the
document must carry it, and its atoms, their species in order and their slabs
must be those of the input of the calculation of the adhesion that gave it,
which stays in the record ``adhesion/interface_min`` or
``adhesion/interface_max``. If the map or the record is missing,
``ppes_inputs`` raises ``PrerequisiteError``; if the map is not valid or the two
disagree, ``StoreError``; nothing is prepared in either case (D58). The inputs
(``ppes_inputs``) come from the document of the system, which
``check_prerequisites`` requires current together with the whole chain of steps
behind it (``chain.check_upstream``). The k-point grid of each calculation is
that of the vendored workflow code (``VaspInputSet``), with the grid from
``[k_points].kdens`` (``mode = "kdens"``); the pw.x input is written from it
(``preparation.workflow_calculation``).

The analysis (``PpesStep.analyse``) uses the records of the calculations only
when each one is complete with the fingerprint of the settings and they share
one energy model (``check_energy_model``). With ``<kind>`` and ``<site>`` as
above it writes:

- ``ppes.<kind>.data.<site>.shift_<shift>``: the results of each calculation
  (``move_results``), the final and the input structures among them; the key
  has a comma for the point of the shift (``shift_-0,75``);
- ``ppes.<kind>.energy.<site>`` (``arrange_data``): ``ppes_ev`` and
  ``ppes_jm2``, a ``[shift, energy]`` pair for each shift, the total energy of
  the cell in eV and per area of the cell in J/m^2, not referred to the lowest
  one; with ``fit = "uber"`` also ``uber``: ``g``, the depth of the curve in eV
  per cell (the fit receives eV), ``l``, its length in angstrom, and their
  standard deviations ``sigma_g`` and ``sigma_l`` in the same units. The fit
  uses the shifts that are zero or positive and needs at least three of them;
  with exactly three the deviations are infinite, with fewer it fails and
  nothing is saved;
- in the folder of the system, the text file ``ppes/ppes_<kind>_<site>.output``
  (``ppes_output``) and the figure ``ppes/ppes_<kind>_<site>_<name>.png`` and
  ``.pdf`` (``draw_ppes``): the energy of each shift against the distance
  between the slabs, referred so that the shift 0 has the adhesion energy of
  the site (``ppes_curve``), with the UBER curve when fitted. They are renamed
  into place once the entries are saved, and the step is done only with them
  carrying the stamp of the provenance saved (``outputs``).

The two entries of the kind and site run (``ppes_entries``) are emptied before
being written, and nothing else is touched: the other sites, the other kind and
the other steps keep their results (DEV-43). Each entry written
(``ppes_leaves``) gets the provenance of the step: the parameters, the fit
among them, the calculations with their results, and the entry read. Nothing is
saved when a part fails.

Notes
-----
To remove: port of TribChem's ``FT_StartPPES`` (``highput/firetasks/run.py
:1738-1835``, with ``FireTaskInterface.query_interface``, ``core.py:493-556``),
of ``InterfaceWF.calculate_ppes`` (``highput/workflows/interface.py:357-468``)
and of ``generate_ppes_inputs`` (``highput/utils/manipulate_struct.py:609-670``,
vendored and no longer called: its loop over the shifts is repeated in
``ppes_inputs`` with the atoms of the slab on top, DEV-44). TribChem's
``set_ppes`` (``core/cli/calc_interface_cli.py:232-248``) fixes ``calc_type`` to
``slab_from_scratch`` and ``ppes_type`` to ``energy``. ``comp_params`` is
``{'kdens': [k_points].kdens}`` (DEV-35); the ids are those of the document
written by ``match`` (A80). Not ported: ``override``, ``run_vasp``,
``get_main_folder`` (A47), ``FT_SaveVASPOutputs``, ``bot_slab_entry``,
``top_slab_entry``, ``database`` and ``db_file``. The analysis ports
``FT_MoveTagResults`` (``highput/firetasks/dbtask.py:166-374``) with the entries
of ``GeneralTribchemWF._select_entry('ppes')`` (``highput/workflows/wfcore.py
:114-128``), then ``FT_FitPPES`` (``highput/firetasks/pes.py:265-375``) with
``ppes_type`` ``energy``; ``FT_FitPPES`` runs the fit only ``if p['fit']``,
which TribChem's command line never gave (A88). The original chooses the atoms
to move by their fractional z, once wrapped into the cell, below one half
(``'middle'`` of ``generate_ppes_inputs``, with the print "ELISA site to shift"
for each atom moved) and nothing checks which atoms it picks: it is the slab on
top only while each slab lies in one half of the cell, a slab thicker than half
the cell is deformed, and an interface whose origin is moved by half a cell gets
the bottom slab moved (A91). The step chooses by the slab each atom belongs to
instead (DEV-44, D58) and repeats the original operations for each shift
(the clean-up of the site properties, the translation of the atoms in cartesian
coordinates without wrapping them, the vendored ``VaspInputSet``): where the
original selection was right the structures are the same as the original ones
and so are their fingerprints, where it was not the calculations change and the
results computed with the wrong atoms are no longer reused (they are replaced
with ``prepare ppes --replace``).
"""

import contextlib
import copy
import io
import logging
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.sets import BadInputSetWarning

from triboworkflow.core.settings import SettingsError
from triboworkflow.core.store import (PrerequisiteError, Provenance, StoreError, SystemWriter,
                                      check_energy_model, check_prerequisites, is_done,
                                      read_record, read_system)
from triboworkflow.core.store.legacy import TASKS, LegacyStore
from triboworkflow.execution.layout import system_dir
from triboworkflow.execution.orchestrator import Step
from triboworkflow.execution.preparation import workflow_calculation
from triboworkflow.vendored.highput.database.dbtools import read_one_entry
from triboworkflow.vendored.highput.utils.manipulate_struct import clean_up_site_properties
from triboworkflow.vendored.highput.utils.tasktools import read_default_params, select_struct_func
from triboworkflow.vendored.highput.utils.vasp_tools import VaspInputSet
from triboworkflow.vendored.physics.base.solidstate import get_cell_info
from triboworkflow.vendored.physics.base.units import Converter
from triboworkflow.vendored.physics.tribology.ppes import PPES, PPESModel, fit_ppes

from .adhesion import STEP as ADHESION
from .adhesion import AdhesionStep
from .chain import check_upstream, upstream_done, upstream_entries_done
from .figures import figure_paths, ppes_figure
from .match import COLLECTION, DEFAULTS, FILTER_KEYS, clear_entries, match_parameters, set_filter
from .outputs import analysis_stamp, header, place, slab_distance, staged, written_by
from .pes import ID_KEYS, TRANSFER_FROM
from .plotpes import interface_name

logger = logging.getLogger(__name__)

STEP = 'ppes'
# The calculations are static (no relaxation) and the quantity is the energy.
# To remove: set_ppes of TribChem's command line (calc_interface_cli.py:232-248).
CALC_TYPE = 'slab_from_scratch'
PPES_TYPE = 'energy'
# The slab each atom of an interface belongs to: a site property of the structures of match,
# kept by the adapter of pw.x in the final structure of a calculation.
SLAB_PROPERTY = 'interface_region'
TOP_SLAB = 'top_slab'
BOTTOM_SLAB = 'bot_slab'
# Entry of the interface of each site, in structure.opt.<kind>.
INTERFACE_OF = {'min_site': 'interface_min', 'max_site': 'interface_max'}


@dataclass
class PpesInputs:
    """What the step reads from the settings and from the document of the system.

    Attributes
    ----------
    type : str
        ``regular`` or ``short``, from ``[interface.adhesion]``.
    site : str
        ``min_site`` or ``max_site``.
    shifts : list of float
        Shifts of the slab on top, in angstrom, in the order of the settings.
    fit : str
        ``none`` or ``uber``, from ``[interface.ppes]``.
    settings : triboworkflow.core.settings.Settings
    ids : dict
        ``mid``, ``formula``, ``name`` and ``miller`` of the document, which
        select it in the writes.
    read : dict
        The entry of the document read, as stored: ``interface``, the final
        structure of the adhesion at the site.
    parameters : dict
        Parameters of the step, recorded in the provenance.
    top : list of int
        Indices, in the interface, of the atoms of the slab on top: the ones
        each shift moves.
    points : list of tuple
        ``(label, shift, structure, kpoints)`` of each calculation, in the
        order of ``shifts``: the structure moved and the k-points of
        ``VaspInputSet``.
    printed : str
        What the vendored code printed.
    """

    type: str
    site: str
    shifts: list
    fit: str
    settings: object
    ids: dict
    read: dict
    parameters: dict
    top: list = field(default_factory=list)
    points: list = field(default_factory=list)
    printed: str = ''


# --- inputs ---------------------------------------------------------------------------

def interface_entry(kind, site):
    """Entry of the document that holds the interface the step starts from.

    Parameters
    ----------
    kind : str
        ``regular`` or ``short``.
    site : str
        ``min_site`` or ``max_site``.

    Returns
    -------
    tuple of str
        ``structure.opt.<kind>.interface_min`` or ``.interface_max``: written
        by the step ``adhesion``.
    """
    return ('structure', 'opt', kind, INTERFACE_OF[site])


def shift_label(site, shift):
    """Label of the calculation of a shift: ``<site>_<shift>``, the shift as written."""
    return f'{site}_{shift}'


def interface_slabs(structure, where):
    """The slab each atom of an interface belongs to.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        With the site property ``interface_region``.
    where : str
        What the structure is, for the messages (e.g. the entry of the document).

    Returns
    -------
    list of str
        ``bot_slab`` or ``top_slab`` for each atom, in the order of the atoms.

    Raises
    ------
    PrerequisiteError
        If the atoms carry no slab.
    StoreError
        If a slab is neither ``bot_slab`` nor ``top_slab``, or no atom
        belongs to one of the two.
    """
    slabs = structure.site_properties.get(SLAB_PROPERTY)
    if slabs is None:
        raise PrerequisiteError(f'{where}: the atoms carry no {SLAB_PROPERTY!r} (the slab each '
                                'belongs to); run adhesion again')
    for i, slab in enumerate(slabs):
        if slab not in (BOTTOM_SLAB, TOP_SLAB):
            raise StoreError(f'{where}: atom {i} is of the slab {slab!r}, not {BOTTOM_SLAB!r} or '
                             f'{TOP_SLAB!r}')
    for name in (BOTTOM_SLAB, TOP_SLAB):
        if name not in slabs:
            raise StoreError(f'{where}: no atom belongs to the slab {name!r}')
    return list(slabs)


def slab_on_top(system, sdir, interface, kind, site):
    """Indices of the atoms of the slab on top of the interface of a site, checked.

    The slab of each atom is the one the interface carries. It is checked against
    the input of the calculation of the adhesion that gave the interface, which
    the document does not keep but the record of that calculation does: the
    interface must have the same number of atoms, the same species in the same
    order and the same slab for each atom. The positions are not compared, since
    the calculation moved the atoms.

    Parameters
    ----------
    system : str
    sdir : path-like
        Directory of the system.
    interface : pymatgen.core.structure.Structure
        The interface of the document at the site.
    kind : str
        ``regular`` or ``short``.
    site : str
        ``min_site`` or ``max_site``.

    Returns
    -------
    list of int
        Indices of the atoms of the slab on top, in increasing order.

    Raises
    ------
    PrerequisiteError
        If the interface or the input carries no slab for its atoms, or the
        record of the calculation or the input in it is missing.
    StoreError
        If a slab is not valid, or the interface and the input differ.
    """
    entry = '.'.join(interface_entry(kind, site))
    calc_id = f'{ADHESION}/{INTERFACE_OF[site]}'
    where = f'the input of {calc_id}'
    slabs = interface_slabs(interface, entry)
    try:
        record = read_record(sdir, calc_id)
    except FileNotFoundError:
        raise PrerequisiteError(f'{system}: the record {calc_id} is missing: the input that {entry}'
                                ' was relaxed from cannot be checked; run adhesion again') from None
    given = ((record.results or {}).get('input') or {}).get('structure')
    if not given:
        raise PrerequisiteError(f'{system}: the record {calc_id} has no input structure: the '
                                f'atoms of {entry} cannot be checked; run adhesion again')
    given = select_struct_func(given['@class']).from_dict(copy.deepcopy(given))
    given_slabs = interface_slabs(given, where)
    differ = 'the interface does not come from that input; run adhesion again'
    if len(interface) != len(given):
        raise StoreError(f'{entry}: the interface has {len(interface)} atoms and {where} has '
                         f'{len(given)}; {differ}')
    if [atom.species_string for atom in interface] != [atom.species_string for atom in given]:
        raise StoreError(f'{entry}: the species of the atoms of the interface, in order, differ '
                         f'from those of {where}; {differ}')
    for i, (slab, slab_given) in enumerate(zip(slabs, given_slabs)):
        if slab != slab_given:
            raise StoreError(f'{entry}: atom {i} is of the slab {slab!r} in the interface and of '
                             f'the slab {slab_given!r} in {where}; {differ}')
    return [i for i, slab in enumerate(slabs) if slab == TOP_SLAB]


def ppes_inputs(run_dir, system, settings):
    """Everything the step reads, and the structure and k-points of each calculation.

    Parameters
    ----------
    run_dir : path-like
        Directory of the run.
    system : str
    settings : triboworkflow.core.settings.Settings
        With an ``[interface]`` table.

    Returns
    -------
    PpesInputs

    Raises
    ------
    SettingsError
        If the k-points are not given by their density (``[k_points]
        mode = "kdens"``): the cell of the interface is known only after the
        matching.
    PrerequisiteError
        If the document of the system or the interface of the adhesion at the
        site is missing, or the slab of its atoms cannot be told: the
        interface or the input it was relaxed from carries none, or the record
        of the adhesion that holds the input is missing (``slab_on_top``).
    StoreError
        If the document is of another interface, or the slabs of the atoms
        are not valid or are not those of the input (``slab_on_top``).

    Notes
    -----
    To remove: ``FT_StartPPES.query_interface`` (``run.py:1803-1822`` with
    ``core.py:493-556``) up to ``generate_ppes_inputs`` (``interface.py
    :429-431``), whose loop over the shifts (``manipulate_struct.py:655-675``)
    is repeated here with the indices of the slab on top (DEV-44). The
    document is found with ``mid`` and ``miller``, as there; ``comp_params``
    comes from the settings instead of the document (DEV-35). The vendored
    code changes in place the structures that it is given, and prints: the
    entry read is copied first, so that ``read`` (the provenance) is what the
    document holds.
    """
    ppes = settings.interface.ppes
    kind, site = settings.interface.adhesion.type, ppes.site
    if settings.k_points.mode != 'kdens':
        raise SettingsError('k_points.mode', f'the step {STEP} takes the k-point grid from its '
                            'density (mode = "kdens"): the cell of the interface is known only '
                            'after the matching', settings.source)
    params, _, _ = match_parameters(settings)
    sdir = system_dir(run_dir, system)
    store = LegacyStore(sdir, read_system(sdir), COLLECTION)
    fltr = set_filter({'mid': '_'.join(params['mid']), 'miller': params['miller']})
    field_, _ = store.query_db(COLLECTION, fltr=fltr)
    if field_ is None:
        raise PrerequisiteError(f'{system}: the system has no document yet: run match first')
    ids = set_filter({key: field_.get(key) for key in ID_KEYS}, FILTER_KEYS)

    entry = interface_entry(kind, site)
    interface = read_one_entry(field_, list(entry))
    if not interface:
        raise PrerequisiteError(f'{system}: entries missing: {".".join(entry)}; run adhesion '
                                'first')
    read = {'interface': interface}

    # The vendored code prints, and warns about the POTCAR functional of the VASP input
    # set, which pw.x does not use.
    printed = io.StringIO()
    comp_params = {'kdens': settings.k_points.kdens}
    shifts = list(ppes.shifts)
    with contextlib.redirect_stdout(printed), warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Overriding the POTCAR functional', BadInputSetWarning)
        structure = select_struct_func(interface['@class']).from_dict(copy.deepcopy(interface))
        top = slab_on_top(system, sdir, structure, kind, site)
        points = []
        for shift in shifts:
            # Each shift starts from the interface read, with the slab on top translated in
            # cartesian coordinates and no atom wrapped back into the cell.
            moved = clean_up_site_properties(structure.copy())
            moved.translate_sites(indices=top, vector=[0, 0, shift], frac_coords=False,
                                  to_unit_cell=False)
            vis = VaspInputSet(moved, comp_params, CALC_TYPE).get_vasp_settings()
            points.append((shift_label(site, shift), shift, moved, vis.kpoints))
    parameters = {'type': kind, 'site': site, 'shifts': shifts, 'fit': ppes.fit,
                  'calc_type': CALC_TYPE, 'ppes_type': PPES_TYPE, 'comp_params': comp_params}
    return PpesInputs(type=kind, site=site, shifts=shifts, fit=ppes.fit, settings=settings,
                      ids=ids, read=read, parameters=parameters, top=top, points=points,
                      printed=printed.getvalue())


def ppes_calculations(system, inputs):
    """The pw.x calculations, as ``preparation.workflow_calculation`` makes them.

    Returns
    -------
    list of QECalculation
        ``ppes/<site>_<shift>`` in the order of the shifts.
    """
    return [workflow_calculation(system, f'{STEP}/{label}', CALC_TYPE, structure, kpoints,
                                 inputs.settings)
            for label, _, structure, kpoints in inputs.points]


def ppes_provenance(inputs, calculations):
    """Provenance that the entries of the step have when computed now.

    Parameters
    ----------
    inputs : PpesInputs
    calculations : dict
        ``{calc_id: fingerprint}`` of the calculations of the shifts.

    Returns
    -------
    Provenance
        The parameters of the step (the fit among them), the calculations and
        the content of the entry of the document read.
    """
    return Provenance.build(STEP, parameters=inputs.parameters, calculations=calculations,
                            inputs=inputs.read)


# --- the entries ----------------------------------------------------------------------

def data_entry(kind, site):
    """Entry that holds the results of the calculations of a site: ``ppes.<kind>.data.<site>``."""
    return (STEP, kind, 'data', site)


def energy_entry(kind, site):
    """Entry that holds the energies and the fit of a site: ``ppes.<kind>.energy.<site>``."""
    return (STEP, kind, PPES_TYPE, site)


def ppes_entries(kind, site):
    """The entries that the step empties before writing: those of the kind and site run.

    Returns
    -------
    list of tuple of str
        ``data_entry`` and ``energy_entry``; the other sites and kinds, and the
        other steps, are not touched (DEV-43).
    """
    return [data_entry(kind, site), energy_entry(kind, site)]


def ppes_leaves(kind, site):
    """The entries that the step writes, each with the provenance of the step.

    Returns
    -------
    list of tuple of str
        The same as ``ppes_entries``.
    """
    return ppes_entries(kind, site)


# --- analysis -------------------------------------------------------------------------

def shift_key(shift):
    """Key of a shift in the document: ``shift_<shift>`` with a comma for the point."""
    return 'shift_' + str(shift).replace('.', ',')


def transfer_entries(kind, site, shift):
    """Keys read in the task document of a shift and entries where they are written.

    Parameters
    ----------
    kind : str
        ``regular`` or ``short``.
    site : str
    shift : float

    Returns
    -------
    entry_from, entry_to : list of list of str
        One entry per key, in the same order: those of ``TRANSFER_FROM``, each
        to ``ppes.<kind>.data.<site>.<shift_key>.output.<last part>``, then the
        final and the input structures.

    Notes
    -----
    To remove: ``GeneralTribchemWF._select_entry('ppes')`` (``wfcore.py
    :114-128``) for one shift.
    """
    base = [STEP, kind, 'data', site, shift_key(shift)]
    entry_from = [list(path) for path in TRANSFER_FROM] + [['output', 'structure'],
                                                           ['input', 'structure']]
    entry_to = [base + ['output', path[-1]] for path in TRANSFER_FROM]
    entry_to += [base + ['output', 'structure'], base + ['input', 'structure']]
    return entry_from, entry_to


def move_results(store, ids, kind, site, shift, calc_id):
    """Copy the results of the calculation of a shift into the document.

    Parameters
    ----------
    store : LegacyStore
        On the document of the system.
    ids : dict
        Ids of the document (``PpesInputs.ids``).
    kind, site : str
    shift : float
    calc_id : str
        ``ppes/<site>_<shift>``, the ``task_label`` of its task document.

    Raises
    ------
    StoreError
        If the calculation has no task document: no record, or a record
        whose results cannot be used.

    Notes
    -----
    To remove: ``FT_MoveTagResults.query_db`` and ``update_db``
    (``dbtask.py:333-374``) with the defaults of ``MoveTagResults``
    (``defaults.json``: ``tag_key`` ``task_label``, ``update_op`` ``$set``,
    ``to_mongodb`` true).
    """
    defaults = read_default_params(str(DEFAULTS), 'MoveTagResults', {})
    entry_from, entry_to = transfer_entries(kind, site, shift)
    field_, field_entry = store.query_db(TASKS, entry=entry_from,
                                         fltr={defaults['tag_key']: calc_id})
    if field_ is None:
        raise StoreError(f'{calc_id}: no record of the calculation')
    store.update_db(field_entry, COLLECTION, entry=entry_to, fltr=set_filter(ids),
                    update_op=defaults['update_op'], to_mongodb=defaults['to_mongodb'])


def historic_fit(fit):
    """The fit as the original code takes it: ``None`` for no fit, else its name.

    Parameters
    ----------
    fit : str
        ``none`` or ``uber``, as in ``[interface.ppes]``.

    Returns
    -------
    str or None
        The string ``'none'`` is true in a condition: it has to become None
        before the code that runs a fit only if it is given.
    """
    return None if fit == 'none' else fit


def fit_energies(values, fit):
    """Fit the energies of the shifts, with the vendored function.

    Parameters
    ----------
    values : list
        ``[shift, energy in eV]`` for each shift.
    fit : str
        Name of the fit, ``uber``.

    Returns
    -------
    dict
        ``g`` (the depth of the curve, in eV per cell), ``l`` (its length, in
        angstrom) and their standard deviations ``sigma_g`` and ``sigma_l``,
        in the same units. Infinite when the covariance of the fit cannot be
        estimated, as with exactly three shifts that are zero or positive.

    Raises
    ------
    StoreError
        If the fit fails, as it does with fewer than three shifts that are
        zero or positive, the only ones it uses.
    """
    try:
        return fit_ppes(values, fit)
    except Exception as error:
        used = sum(1 for shift, _ in values if shift >= 0)
        raise StoreError(f'the fit {fit!r} failed with {used} shifts that are zero or positive '
                         f'({type(error).__name__}: {error}); it uses those only and needs at '
                         'least three of them') from error


def arrange_data(store, ids, kind, site, shifts, fit):
    """Collect the energies of the shifts and write them, in eV and J/m^2, with the fit.

    Parameters
    ----------
    store : LegacyStore
        On the document of the system, with the results of every shift
        (``move_results``).
    ids : dict
        Ids of the document.
    kind, site : str
    shifts : list of float
        In the order of the settings.
    fit : str or None
        ``uber`` to fit the energies, None for no fit (``historic_fit``).

    Returns
    -------
    dict
        What is written in ``ppes.<kind>.energy.<site>``: ``ppes_ev`` and
        ``ppes_jm2`` (``[shift, energy]`` for each shift) and, with a fit, the
        fit under its name (``fit_energies``). The energy per area is the
        total energy of the cell, not referred to the lowest one.

    Raises
    ------
    StoreError
        If a calculation gave no energy, or the fit fails. Nothing is
        written in either case.

    Notes
    -----
    To remove: ``FT_FitPPES.arrange_data`` (``pes.py:327-375``) with
    ``ppes_type`` ``energy`` (the other values of ``ppes_type`` are not
    ported); the fit receives the shifts and the energies in eV, as there.
    """
    # Retrieve the document with all the data
    fltr = set_filter(ids)
    field_, _ = store.query_db(COLLECTION, fltr=fltr)

    # Recover the values to perform the fit, and convert them to J/m^2
    values = []
    values_jm2 = []
    for shift in shifts:
        base = [STEP, kind, 'data', site, shift_key(shift), 'output']
        energy = read_one_entry(field_, base + [PPES_TYPE])
        if energy is None:
            raise StoreError(f'no energy in the results of the shift {shift} (ppes/'
                             f'{shift_label(site, shift)})')
        values.append([shift, energy])
        struct = Structure.from_dict(read_one_entry(field_, base + ['structure']))
        area, _ = get_cell_info(struct.lattice.matrix)
        values_jm2.append([shift, Converter.eV_to_Jm2(energy, area * 1e-20)])

    # Try to fit the PPES
    fit_dict = None
    if fit:
        fit_dict = fit_energies(values, fit)

    # Create the final data dictionary
    values_dict = {'ppes_ev': values, 'ppes_jm2': values_jm2}
    if fit_dict is not None:
        values_dict[fit] = fit_dict

    # Update them in the document
    store.update_db({PPES_TYPE: {site: values_dict}}, COLLECTION, entry=[STEP, kind], fltr=fltr)
    return values_dict


def run_ppes_analysis(inputs, store, calc_ids):
    """Copy the results of every shift and arrange the energies, without the provenance.

    Parameters
    ----------
    inputs : PpesInputs
    store : LegacyStore
        On the document of the system, which is changed and not saved.
    calc_ids : list of str
        ``calc_id`` of the calculations, in the order of the shifts.

    Returns
    -------
    dict
        What ``arrange_data`` wrote.
    """
    for shift, calc_id in zip(inputs.shifts, calc_ids):
        move_results(store, inputs.ids, inputs.type, inputs.site, shift, calc_id)
    return arrange_data(store, inputs.ids, inputs.type, inputs.site, inputs.shifts,
                        historic_fit(inputs.fit))


def uber_parameters(values, fit):
    """``g``, ``l`` and ``s0`` of the UBER curve fitted, as the vendored fit finds them.

    The fit gives back ``g`` and ``l`` only; ``s0``, the shift of the
    minimum of the curve, is needed to draw it. The same fit is run again on
    the same data, and its ``g`` and ``l`` must be those written.

    Parameters
    ----------
    values : dict
        As ``arrange_data`` returns it, with the fit ``uber``.
    fit : str
        ``uber``.

    Returns
    -------
    tuple of float

    Raises
    ------
    StoreError
        If the fit run again gives another ``g`` or ``l``.

    Notes
    -----
    To remove: ``fit_ppes`` (``physics/tribology/ppes.py:214-228``) with ``fit='uber'``,
    whose ``popt`` holds the three parameters.
    """
    data = np.array(values['ppes_ev'])
    fit_data = data[data[:, 0] >= 0.0, :]
    # The warnings of the fit were given when the step fitted.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        popt, _ = PPES(fit_data).fit(method=fit, p0=[-min(fit_data[:, 1]), 0.75, 0])
    written = values[fit]
    if not (math.isclose(popt[0], written['g'], rel_tol=1e-9)
            and math.isclose(popt[1], written['l'], rel_tol=1e-9)):
        raise StoreError(f'the fit {fit!r} run again gives g = {popt[0]!r}, l = {popt[1]!r}, not '
                         f'those written ({written["g"]!r}, {written["l"]!r})')
    return float(popt[0]), float(popt[1]), float(popt[2])


def ppes_curve(inputs, values, document):
    """What the text file and the figure of the step show.

    Parameters
    ----------
    inputs : PpesInputs
    values : dict
        As ``arrange_data`` returns it.
    document : SystemDocument
        The document of the system, with the adhesion energy of the site.

    Returns
    -------
    dict
        ``shift``, ``energy_ev`` and ``energy_jm2``: arrays, one value per
        shift; ``distance``: the distance between the slabs of the interface
        read, in angstrom, or None (``outputs.slab_distance``);
        ``adhesion``: the adhesion energy of the site in J/m^2
        (``adhesion.<kind>.<site>``), or None; ``adhesion_jm2``:
        ``energy_jm2`` minus its value at the shift 0 plus the adhesion
        energy, or None without the shift 0 or the adhesion energy;
        ``uber``: ``(g, l, s0)`` of the fit, or None; ``fit_values``: the fit
        as written, with its deviations, or None; ``jm2_per_ev``: J/m^2 of one
        eV on the area of the cell.

    Notes
    -----
    To remove: the energies of TribChem's ``plot_ppes`` (``get_shifts``,
    ``highput/utils/plotting.py:694-709``): the energy of the shift 0 is the
    adhesion energy of the site.
    """
    shift = np.array([row[0] for row in values['ppes_ev']], dtype=float)
    energy_ev = np.array([row[1] for row in values['ppes_ev']], dtype=float)
    energy_jm2 = np.array([row[1] for row in values['ppes_jm2']], dtype=float)
    adhesion = document.get((ADHESION, inputs.type, inputs.site, 'adhesion_energy_jm2'))
    at_zero = np.flatnonzero(shift == 0.0)
    adhesion_jm2 = None
    if adhesion is not None and len(at_zero):
        adhesion_jm2 = energy_jm2 - energy_jm2[at_zero[0]] + adhesion
    fitted = inputs.fit != 'none'
    area, _ = get_cell_info(Structure.from_dict(inputs.read['interface']).lattice.matrix)
    return {'shift': shift, 'energy_ev': energy_ev, 'energy_jm2': energy_jm2,
            'distance': slab_distance(inputs.read['interface']), 'adhesion': adhesion,
            'adhesion_jm2': adhesion_jm2,
            'uber': uber_parameters(values, inputs.fit) if fitted else None,
            'fit_values': values[inputs.fit] if fitted else None,
            'jm2_per_ev': Converter.eV_to_Jm2(1.0, area * 1e-20)}


def output_path(sdir, kind, site):
    """Text file of the step: ``ppes/ppes_<kind>_<site>.output`` in the folder of the system."""
    return sdir / STEP / f'{STEP}_{kind}_{site}.output'


def figure_path(sdir, kind, site, name):
    """PNG of the figure: ``ppes/ppes_<kind>_<site>_<name>.png``; the PDF is next to it."""
    return sdir / STEP / f'{STEP}_{kind}_{site}_{name}.png'


def ppes_output(system, inputs, curve, now=None, stamp=None):
    """Text of ``ppes_<kind>_<site>.output``: parameters, energies of the shifts and fit.

    Parameters
    ----------
    system : str
    inputs : PpesInputs
    curve : dict
        As ``ppes_curve`` returns it.
    now : datetime.datetime or None, optional
    stamp : str or None, optional
        ``outputs.analysis_stamp`` of the analysis.

    Returns
    -------
    str
    """
    distance, adhesion = curve['distance'], curve['adhesion']
    lines = header('Perpendicular PES of the interface', system, STEP, now, stamp) + [
        '#', '# Parameters',
        f'#   [interface.ppes]      site = "{inputs.site}", fit = "{inputs.fit}", shifts = '
        f'{inputs.shifts}',
        f'#   [interface.adhesion]  type = "{inputs.type}"',
        f'#   distance between the slabs of the interface relaxed by the adhesion at '
        f'{inputs.site}: ' + ('not told' if distance is None else f'{distance:.6f} angstrom'),
        f'#   adhesion energy at {inputs.site} (adhesion.{inputs.type}.{inputs.site}): '
        + ('not written' if adhesion is None else f'{adhesion:.8f} J/m^2'),
        '#', '# One row per shift of the slab on top:',
        '#   shift         shift of the slab on top along z from the relaxed interface, angstrom',
        '#   distance      distance between the slabs, angstrom (nan: not told)',
        f'#   energy_eV     total energy of the cell, eV (ppes.{inputs.type}.energy.'
        f'{inputs.site}.ppes_ev)',
        '#   energy_Jm2    the same per area of the cell, J/m^2 (ppes_jm2)',
        '#   adhesion_Jm2  energy_Jm2 minus its value at the shift 0 plus the adhesion energy',
        '#                 (nan without the shift 0 or the adhesion energy)',
        f'# {"shift":>10} {"distance":>12} {"energy_eV":>18} {"energy_Jm2":>18} '
        f'{"adhesion_Jm2":>14}']
    for i, shift in enumerate(curve['shift']):
        lines.append(
            f'  {shift:>10.4f} '
            f'{"nan" if distance is None else f"{distance + shift:.6f}":>12} '
            f'{curve["energy_ev"][i]:>18.8f} {curve["energy_jm2"][i]:>18.8f} '
            + ('{:>14}'.format('nan') if curve['adhesion_jm2'] is None
               else f'{curve["adhesion_jm2"][i]:>14.8f}'))
    if curve['uber'] is not None:
        g, l, s0 = curve['uber']
        fit, written = inputs.fit, curve['fit_values']
        lines += ['#', f'# Fit {fit} on the shifts >= 0 (ppes.{inputs.type}.energy.{inputs.site}.'
                       f'{fit}): energy_eV minus its lowest value',
                  '#   = g (1 - (1 + (shift - s0)/l) exp(-(shift - s0)/l))',
                  f'#   g = {g:.8f} eV per cell ({g * curve["jm2_per_ev"]:.8f} J/m^2), '
                  f'l = {l:.8f} angstrom, s0 = {s0:.8f} angstrom',
                  f'#   sigma_g = {written["sigma_g"]:.4g} eV per cell, '
                  f'sigma_l = {written["sigma_l"]:.4g} angstrom']
    return '\n'.join(lines) + '\n'


def draw_ppes(curve, png, pdf, stamp=None):
    """The figure of the PPES: energy against the distance between the slabs.

    The energy is the adhesion energy (``adhesion_jm2``) when the shift 0
    and the adhesion energy are there, else the energy above the lowest one;
    the distance is that between the slabs when told, else the shift. With a
    fit, its UBER curve, from the shift 0 to the largest. ``stamp`` goes in
    the metadata of the files.

    Notes
    -----
    To remove: TribChem's ``plot_ppes`` (``highput/utils/plotting.py:631-661``) drew the
    energies against the shift minus the z of the highest atom below the contact plane
    (``estimate_interface(...)[0]``), the distance of the plane from the bottom slab; here
    the abscissa is the distance between the two slabs, from the same structure.
    """
    if curve['adhesion_jm2'] is not None:
        energy, ylabel = curve['adhesion_jm2'], r'adhesion energy (J/m$^2$)'
        offset = curve['adhesion_jm2'][0] - curve['energy_jm2'][0]
    else:
        energy, ylabel = curve['energy_jm2'] - curve['energy_jm2'].min(), r'$\Delta E$ (J/m$^2$)'
        offset = -curve['energy_jm2'].min()
    if curve['distance'] is not None:
        x, xlabel = curve['distance'] + curve['shift'], r'distance between the slabs ($\AA$)'
        start = curve['distance']
    else:
        x, xlabel = curve['shift'], r'shift of the slab on top ($\AA$)'
        start = 0.0
    fitted = None
    if curve['uber'] is not None:
        g, l, s0 = curve['uber']
        s = np.linspace(0.0, curve['shift'].max(), 200)
        lowest = curve['energy_ev'][curve['shift'] >= 0.0].min()
        fit_ev = lowest + PPESModel.uber(s, g, l, s0)
        fitted = (start + s, fit_ev * curve['jm2_per_ev'] + offset)
    ppes_figure(x, energy, png, pdf, xlabel, ylabel, fitted, stamp)


def describe(inputs, values):
    """Lines of the message of the step: shifts, energies and fit.

    Parameters
    ----------
    inputs : PpesInputs
    values : dict
        As ``arrange_data`` returns it.

    Returns
    -------
    list of str
    """
    ev, jm2 = values['ppes_ev'], values['ppes_jm2']
    (low, e_low), (high, e_high) = (min(jm2, key=lambda row: row[1]),
                                    max(jm2, key=lambda row: row[1]))
    shifts = [shift for shift, _ in ev]
    lines = [f'{STEP}: {inputs.type} PPES at {inputs.site}, {len(ev)} shifts of the slab on top '
             f'from {min(shifts)} to {max(shifts)} angstrom ({inputs.parameters["calc_type"]})',
             f'total energy per area: lowest {e_low:.6f} J/m^2 at the shift {low}, highest '
             f'{e_high:.6f} J/m^2 at the shift {high}']
    if inputs.fit == 'none':
        lines.append('no fit (fit = "none")')
    else:
        fit = values[inputs.fit]
        lines.append(f'fit {inputs.fit}: g = {fit["g"]:.6g} +- {fit["sigma_g"]:.3g} eV per cell, '
                     f'l = {fit["l"]:.6g} +- {fit["sigma_l"]:.3g} angstrom')
    return lines


# --- step -----------------------------------------------------------------------------

def _load_run(run_dir):
    # Imported here: the command line imports the registry of the steps.
    from triboworkflow.cli.run import load_run

    return load_run(run_dir)


class PpesStep(Step):
    """Step ``ppes``: one calculation per shift of the slab on top of the relaxed interface."""

    name = STEP

    def done(self, run_dir, system):
        """Whether its entries are done (``entries_done``) and its text file and the two
        files of its figure were written by the analysis saved (``outputs.written_by``)."""
        current = self._current(run_dir, system, upstream_done)
        return current is not None and written_by(*self._exports(run_dir, system, *current))

    def entries_done(self, run_dir, system):
        """Whether its entries (``ppes_leaves``) are written and current (``provenance``).

        The adhesion, the surface and ``match``, which it reads, have to be
        done too: the provenance holds the content of the entries read, not
        whether they are still current. Their entries only, not their files
        (``upstream_entries_done``).
        """
        return self._current(run_dir, system, upstream_entries_done) is not None

    def exports(self, run_dir, system):
        """Files of the analysis saved and the stamp each carries: ``(paths, stamp)``.

        Only for entries done (``entries_done``): the files are named after
        the interface that the inputs read.
        """
        return self._exports(run_dir, system,
                             *self._current(run_dir, system, upstream_entries_done))

    def _current(self, run_dir, system, upstream):
        """The document and the inputs when the entries are written and current, else None.

        ``upstream`` tells whether the steps upstream are done: ``upstream_done`` or
        ``upstream_entries_done``.
        """
        sdir = system_dir(run_dir, system)
        document = read_system(sdir)
        settings = _load_run(run_dir)
        kind, site = settings.interface.adhesion.type, settings.interface.ppes.site
        entries = ppes_leaves(kind, site)
        # Before any analysis the inputs are not read: the adhesion may not have run.
        if not all(document.get(entry) for entry in entries):
            return None
        if not upstream(self, run_dir, system):
            return None
        try:
            inputs = self.inputs(run_dir, system)
        except PrerequisiteError:
            return None
        expected = ppes_provenance(inputs, {calc.calc_id: calc.record.fingerprint
                                            for calc in ppes_calculations(system, inputs)})
        if not all(is_done(document, entry, expected, sdir) for entry in entries):
            return None
        return document, inputs

    @staticmethod
    def _exports(run_dir, system, document, inputs):
        sdir = system_dir(run_dir, system)
        paths = [output_path(sdir, inputs.type, inputs.site),
                 *figure_paths(figure_path(sdir, inputs.type, inputs.site,
                                           interface_name(inputs.ids)))]
        stamp = analysis_stamp(document.provenance(ppes_leaves(inputs.type, inputs.site)[0]))
        return paths, stamp

    def analyse(self, run_dir, system, calculations):
        """Read the calculations, write the energies and the fit with their provenance.

        Only the entries of the kind and site of the run are replaced, in the
        same saving: the results of the other sites and kinds stay (DEV-43).

        Raises
        ------
        PrerequisiteError
            If the interface read, or a step behind it, is missing or not
            current.
        ValueError
            If ``calculations`` are not those that the settings give now.
        StoreError
            If a record is missing, not complete, of another fingerprint, or
            the records do not share one energy model, or a calculation gave
            no energy, or the fit fails. Nothing is saved.
        """
        sdir = system_dir(run_dir, system)
        with SystemWriter(sdir) as writer:
            check_prerequisites(writer.document, self.expected(run_dir, system), sdir)
            inputs = self.inputs(run_dir, system)
            plan = {calc.calc_id: calc.record.fingerprint
                    for calc in ppes_calculations(system, inputs)}
            if sorted(calculations) != sorted(plan):
                raise ValueError(f'{system}/{self.name}: the calculations given '
                                 f'({", ".join(sorted(calculations))}) are not those of the '
                                 f'settings ({", ".join(plan)})')
            records = []
            for calc_id in plan:
                try:
                    records.append(read_record(sdir, calc_id))
                except FileNotFoundError:
                    raise StoreError(f'{system}/{calc_id}: no record of the calculation') from None
            provenance = ppes_provenance(inputs, plan).with_results(records)
            check_energy_model(records)
            # Nothing of an earlier analysis of this kind and site is left (DEV-43).
            clear_entries(writer.document, ppes_entries(inputs.type, inputs.site))
            values = run_ppes_analysis(inputs, LegacyStore(sdir, writer.document, COLLECTION),
                                       list(plan))
            for entry in ppes_leaves(inputs.type, inputs.site):
                writer.document.set_provenance(entry, provenance)
            curve = ppes_curve(inputs, values, writer.document)
            (output, *figure), stamp = self._exports(run_dir, system, writer.document, inputs)
            with staged([output, *figure]) as partials:
                partials[output].write_text(ppes_output(system, inputs, curve, stamp=stamp),
                                            encoding='utf-8')
                draw_ppes(curve, *(partials[path] for path in figure), stamp)
                writer.save()
                place(partials)
        fit = values.get(inputs.fit)
        if fit is not None and not all(math.isfinite(fit[key]) for key in ('sigma_g', 'sigma_l')):
            logger.warning('%s/%s: the covariance of the fit could not be estimated: the '
                           'deviations of g and l are infinite', system, self.name)
        for line in describe(inputs, values):
            logger.info('%s: %s', system, line)
        logger.info('%s: table of the PPES in %s, figure in %s', system, output,
                    ' and '.join(path.name for path in figure))

    def inputs(self, run_dir, system):
        """``ppes_inputs`` of the step, from the settings of the run."""
        inputs = ppes_inputs(run_dir, system, _load_run(run_dir))
        if inputs.printed:
            logger.debug('%s/%s: output of the vendored code:\n%s', system, self.name,
                         inputs.printed)
        return inputs

    def calculations(self, run_dir, system):
        """One calculation per shift, nothing written."""
        return ppes_calculations(system, self.inputs(run_dir, system))

    def expected(self, run_dir, system):
        """The interface read, with the provenance of the adhesion now.

        Raises
        ------
        PrerequisiteError
            If a step behind the adhesion (its surface or ``match``) is not
            done or not current: the provenance of the adhesion holds the
            content that it read and would not tell.
        """
        settings = _load_run(run_dir)
        adhesion = AdhesionStep()
        check_upstream(adhesion, run_dir, system)
        entry = interface_entry(settings.interface.adhesion.type, settings.interface.ppes.site)
        return {entry: adhesion.provenance(run_dir, system)}

    def provenance(self, run_dir, system):
        """Provenance expected now for the entries that the step writes."""
        inputs = self.inputs(run_dir, system)
        calcs = {calc.calc_id: calc.record.fingerprint
                 for calc in ppes_calculations(system, inputs)}
        return ppes_provenance(inputs, calcs)
