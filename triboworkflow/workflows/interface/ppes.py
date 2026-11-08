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

The atoms to move are chosen from the final structure of the adhesion
(``'middle'`` of ``generate_ppes_inputs``) by their position, not by the slab
they belong to: the sites whose fractional z, once wrapped into the cell, is
below one half. They are the slab on top only while each slab lies in one half
of the cell, the one on top below the middle (the contact plane of the
interface at the origin) and the bottom one above it, as the structures of
``match`` have them when the slabs are thin compared with the cell and as a
relaxation leaves them (the positions are not wrapped). Otherwise the slab is
deformed and not moved as a whole: a slab on top thicker than half the cell
keeps its upper atoms in place, a bottom slab thicker than half the cell has its
lower atoms moved, and with the origin moved by half a cell the bottom slab is
moved instead (``Notes``). The inputs (``ppes_inputs``)
come from the document of the system, which ``check_prerequisites`` requires
current together with the whole chain of steps behind it
(``chain.check_upstream``). The k-point grid of each calculation is that of the
vendored workflow code (``VaspInputSet``), with the grid from ``[k_points]
.kdens`` (``mode = "kdens"``); the pw.x input is written from it
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
  nothing is saved.

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
vendored and called unchanged, with its debugging ``print``). TribChem's
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
which TribChem's command line never gave (A88). The choice of the slab by the
fractional z, with the print "ELISA site to shift" for each atom moved, is the
original one and nothing checks which atoms it picks: a slab thicker than half
the cell is deformed, and an interface whose origin is moved by half a cell gets
the bottom slab moved (A91, ``test_ppes``). The atoms of the interface carry the
slab they belong to (the site property ``interface_region`` of ``match``), which
the selection does not use.
"""

import contextlib
import copy
import io
import logging
import math
import warnings
from dataclasses import dataclass, field

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
from triboworkflow.vendored.highput.utils.manipulate_struct import generate_ppes_inputs
from triboworkflow.vendored.highput.utils.tasktools import read_default_params, select_struct_func
from triboworkflow.vendored.physics.base.solidstate import get_cell_info
from triboworkflow.vendored.physics.base.units import Converter
from triboworkflow.vendored.physics.tribology.ppes import fit_ppes

from .adhesion import AdhesionStep
from .chain import check_upstream, upstream_done
from .match import COLLECTION, DEFAULTS, FILTER_KEYS, clear_entries, match_parameters, set_filter
from .pes import ID_KEYS, TRANSFER_FROM

logger = logging.getLogger(__name__)

STEP = 'ppes'
# The calculations are static (no relaxation) and the quantity is the energy.
# To remove: set_ppes of TribChem's command line (calc_interface_cli.py:232-248).
CALC_TYPE = 'slab_from_scratch'
PPES_TYPE = 'energy'
# Part of the slab on top: the sites with a fractional z below this are moved.
INTER_TYPE = 'middle'
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
        site is missing.
    StoreError
        If the document is of another interface.

    Notes
    -----
    To remove: ``FT_StartPPES.query_interface`` (``run.py:1803-1822`` with
    ``core.py:493-556``) up to ``generate_ppes_inputs`` (``interface.py
    :429-431``). The document is found with ``mid`` and ``miller``, as there;
    ``comp_params`` comes from the settings instead of the document (DEV-35).
    The vendored code changes in place the structures that it is given, and
    prints which sites it moves: the entry read is copied first, so that
    ``read`` (the provenance) is what the document holds.
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

    # The vendored code prints which sites it moves, and warns about the POTCAR
    # functional of the VASP input set, which pw.x does not use.
    printed = io.StringIO()
    comp_params = {'kdens': settings.k_points.kdens}
    shifts = list(ppes.shifts)
    with contextlib.redirect_stdout(printed), warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Overriding the POTCAR functional', BadInputSetWarning)
        structure = select_struct_func(interface['@class']).from_dict(copy.deepcopy(interface))
        structures, vis, _ = generate_ppes_inputs(structure, shifts, comp_params, INTER_TYPE, None,
                                                  CALC_TYPE)
    points = [(shift_label(site, shift), shift, moved, v.kpoints)
              for shift, moved, v in zip(shifts, structures, vis)]
    parameters = {'type': kind, 'site': site, 'shifts': shifts, 'fit': ppes.fit,
                  'calc_type': CALC_TYPE, 'ppes_type': PPES_TYPE, 'comp_params': comp_params}
    return PpesInputs(type=kind, site=site, shifts=shifts, fit=ppes.fit, settings=settings,
                      ids=ids, read=read, parameters=parameters, points=points,
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
        """Whether its entries (``ppes_leaves``) are written and current (``provenance``).

        The adhesion, the surface and ``match``, which it reads, have to be
        done too: the provenance holds the content of the entries read, not
        whether they are still current.
        """
        sdir = system_dir(run_dir, system)
        document = read_system(sdir)
        settings = _load_run(run_dir)
        entries = ppes_leaves(settings.interface.adhesion.type, settings.interface.ppes.site)
        # Before any analysis the inputs are not read: the adhesion may not have run.
        if not all(document.get(entry) for entry in entries):
            return False
        if not upstream_done(self, run_dir, system):
            return False
        try:
            expected = self.provenance(run_dir, system)
        except PrerequisiteError:
            return False
        return all(is_done(document, entry, expected, sdir) for entry in entries)

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
            writer.save()
        fit = values.get(inputs.fit)
        if fit is not None and not all(math.isfinite(fit[key]) for key in ('sigma_g', 'sigma_l')):
            logger.warning('%s/%s: the covariance of the fit could not be estimated: the '
                           'deviations of g and l are infinite', system, self.name)
        for line in describe(inputs, values):
            logger.info('%s: %s', system, line)

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
