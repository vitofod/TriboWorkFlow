# Author: Vito Foderà
# Created: 2026-11-09

"""Step ``charge`` of the interface workflow: the charge displacement of the interface.

The step starts from the interface that the step ``adhesion`` relaxed at the site of
lowest energy (``min_site``) or of highest energy (``max_site``) of its surface. It
runs three static calculations (``scf``, ``calc_type`` ``slab_from_scratch``) in the
same cell, each of which also gives the charge density (pw.x, then pp.x, see
``docs/qe_charge.md``): the interface, its bottom slab and its top slab, the slabs with
their atoms where they are in the interface. Their ids are ``charge/<site>_<part>``,
with ``<part>`` one of ``interface``, ``bot_slab`` and ``top_slab`` (``LABELS``), so that
the two sites live side by side. The kind of interface, ``regular`` or ``short``, is the
one of ``[interface.adhesion]`` (D56); a run covers one site and one kind, and the
results of the other site or kind stay in the document. Everything is moved by half a
cell along z, so that the contact plane of the slabs, at z = 0 in the interface, is in
the middle of the cell. The k-point grid of each calculation is that of the vendored
workflow code (``VaspInputSet``), with the grid from ``[k_points].kdens``
(``mode = "kdens"``); the pw.x and pp.x inputs are written from it
(``preparation.workflow_calculation`` with ``density=True``).

The two slabs are told by the slab each atom belongs to and not by the sign of z: the
site property ``interface_region`` (``bot_slab`` or ``top_slab``) that ``match`` gives to
every atom of an interface and that the pw.x adapter returns with the final structure,
checked before any calculation is prepared against the input of the calculation of the
adhesion that gave the interface (``ppes.slab_on_top``): the interface of the document
must carry it, with the atoms, species in order and slabs of that input (D58, D59).
Without the map or the record the inputs raise ``PrerequisiteError``; if the map is not
valid or the two disagree, ``StoreError``. The interface itself is brought around z = 0
as the vendored workflow does, with ``ManipulateStruct.fix_interface``: an interface
that comes back from a relaxation can have its bottom slab in the upper half of the
cell, and the function subtracts one cell from every atom beyond the middle. That moves
by a lattice vector, so it does not change a periodic calculation, but it does not tell
the slabs. The range of integration is the region between the two slabs
(``ManipulateStruct.estimate_interface``, half a cell added). It assumes the contact
plane at z = 0 (``INTER_TYPE`` ``zero``): if an atom lies exactly at z = 0, which the
vendored decomposition into slabs refuses too, or the highest atom below z = 0 is not of
the bottom slab or the lowest above it is not of the top slab, ``charge_inputs`` raises
``StoreError`` and prepares nothing, since the range would be the wrong region.

The analysis (``ChargeStep.analyse``) reads the records of the three calculations and
the files of their densities, and uses them only when each record is complete with the
fingerprint of the settings and its density is ``ok``, the three share one energy model
(``check_energy_model``) and the densities lie on the same grid in the same cell
(``density.check_comparable``). Each density is turned into the object the charge
analysis works on (``density.to_chgcar``: the density times the volume of the cell, in
the units of a CHGCAR). With ``<kind>`` and ``<site>`` as above it writes:

- ``charge.<kind>.data.<site>.<part>``: the results of each calculation
  (``move_results``), with the final structure in ``structure``;
- ``charge.<kind>.average.<site>.<part>.<axis>`` (``average_charge``): ``[x, density]``
  for the planar average of the density along x, y and z, ``x`` in angstrom from the
  origin of the cell and the density in e/angstrom^3; for z the ``x`` is moved back by
  half the cell, so that the contact plane of the slabs is at 0;
- ``charge.<kind>.displacement.<site>.z`` (``displace_charge``): the density of the
  interface combined with those of the slabs by the operations of
  ``[interface.charge]`` (the interface minus the bottom slab minus the top slab by
  default), averaged in the plane and integrated in absolute value: ``interface`` over
  the range between the slabs and ``all`` over the whole cell, each with ``integral``
  (e/angstrom^3, the integral divided by the width) and ``width`` (angstrom), then
  ``grid`` (the averaged curve, ``x`` as above) and ``rescale``, the number of points of
  the integration (``RESCALE``).

The keys are fixed, so nothing is emptied before they are written. Each of the three
entries of the kind and site run (``charge_leaves``) gets the provenance of the step:
the parameters (the operations among them), the calculations with their results, the
entry read, and the digests of the three files of the densities, which the results of
the records do not cover. Nothing is saved when a part fails.

Notes
-----
To remove: port of TribChem's ``FT_StartChargeDisplacement`` (``highput/firetasks/run.py
:1838-1896``, with ``FireTaskInterface.query_interface``, ``core.py:493-556``), of
``ElectronicsWF.charge_displacement`` (``highput/workflows/electronics.py:41-127``) and of
``generate_chargedisp_inputs`` (``highput/utils/manipulate_struct.py:567-607``, vendored
and no longer called: its translation of the structures and its input sets are repeated
in ``charge_inputs`` with the atoms of each slab, DEV-45). The original tells the slabs
with ``decompose_structure('zero')``, by the sign of z once ``fix_interface`` has been
called: with the top slab thicker than half the cell some of its atoms get a negative z
and land in the bottom fragment, so the density of the "slab" has an atom of another
material and the difference does not remove the charge of the slabs, with no error
(A92). The analysis ports ``FT_MoveTagResults`` (``highput/firetasks/dbtask.py
:166-374``) with the entries of ``GeneralTribchemWF._select_entry('chgdisp')``
(``highput/workflows/wfcore.py:130-143``), ``FT_AverageCharge``
(``highput/firetasks/charge.py:39-160``) and ``FT_ChargeDisplacement`` (``charge.py
:162-260``), which read the CHGCAR files of VASP from ``dir_name`` (``split(':')[1]``)
and here read the files of pp.x of the records. TribChem's command line fixes
``calc_type`` to ``slab_from_scratch`` (``calc_interface_cli.py``), the axis to 2 and
``inter_type`` to ``zero``, and ``rescale`` is 1000 in ``defaults.json`` (the signatures
say 10000, A93); ``integrate_in`` is always estimated, as a value given makes the
workflow fail (A32). ``comp_params`` is ``{'kdens': [k_points].kdens}`` (DEV-35). The
original considers the step done when ``charge.<kind>`` has anything in it, whatever the
site (A59). Not ported: ``to_plot``, ``override``, ``run_vasp``, ``get_main_folder``
(A47), ``FT_SaveVASPOutputs``, the gunzip and ``rm`` of the CHGCAR files, ``database``
and ``db_file``.
"""

import contextlib
import copy
import io
import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
from monty.json import jsanitize
from pymatgen.io.vasp.sets import BadInputSetWarning

from triboworkflow.core.settings import SettingsError
from triboworkflow.core.store import (PrerequisiteError, Provenance, StoreError, SystemWriter,
                                      check_energy_model, check_prerequisites, is_done,
                                      read_record, read_system)
from triboworkflow.core.store.legacy import TASKS, LegacyStore
from triboworkflow.core.structures import StructureError, read_structure
from triboworkflow.engines.qe import density
from triboworkflow.execution.layout import STRUCTURE_OUT_FILE, calc_dir, system_dir
from triboworkflow.execution.orchestrator import Step
from triboworkflow.execution.preparation import workflow_calculation
from triboworkflow.vendored.highput.database.dbtools import read_one_entry
from triboworkflow.vendored.highput.utils.manipulate_struct import ManipulateStruct
from triboworkflow.vendored.highput.utils.tasktools import read_default_params, select_struct_func
from triboworkflow.vendored.highput.utils.vasp_tools import VaspInputSet
from triboworkflow.vendored.physics.electronics.charge.chgcar_integrator import (
    charge_displacement, chgcar_algebra_from_list, chgcar_average_along_axis)

from .adhesion import AdhesionStep
from .chain import check_upstream, upstream_done
from .match import COLLECTION, DEFAULTS, FILTER_KEYS, match_parameters, set_filter
from .pes import ID_KEYS, TRANSFER_FROM
# The slab each atom of an interface belongs to and the entries of the interface of each site are
# those of the step ppes, which starts from the same interface (D58).
from .ppes import (BOTTOM_SLAB, INTERFACE_OF, SLAB_PROPERTY, TOP_SLAB,  # noqa: F401 (re-exported)
                   interface_entry, slab_on_top)

logger = logging.getLogger(__name__)

STEP = 'charge'
# The calculations are static (no relaxation), the contact plane of the slabs is at z = 0 in the
# interface and the quantity is integrated along z, the third vector of the cell.
# To remove: the values that TribChem's command line fixes (calc_interface_cli.py), and rescale
# of defaults.json (A93).
CALC_TYPE = 'slab_from_scratch'
INTER_TYPE = 'zero'
AXIS = 2
RESCALE = 1000
# The three calculations, in the order of the operations: the interface, then the slabs.
LABELS = ('interface', 'bot_slab', 'top_slab')
AXIS_NAMES = {0: 'x', 1: 'y', 2: 'z'}


@dataclass
class ChargeInputs:
    """What the step reads from the settings and from the document of the system.

    Attributes
    ----------
    type : str
        ``regular`` or ``short``, from ``[interface.adhesion]``.
    site : str
        ``min_site`` or ``max_site``.
    operations : list of str
        ``+`` or ``-`` twice, from ``[interface.charge]``.
    settings : triboworkflow.core.settings.Settings
    ids : dict
        ``mid``, ``formula``, ``name`` and ``miller`` of the document, which
        select it in the writes.
    read : dict
        The entry of the document read, as stored: ``interface``, the final
        structure of the adhesion at the site.
    parameters : dict
        Parameters of the step, recorded in the provenance.
    top, bottom : list of int
        Indices, in the interface, of the atoms of the top slab and of the
        bottom slab.
    integrate_in : list of float
        Lower and upper limit, in angstrom, of the region of the interface, in
        the cell of the calculations (half a cell above the estimate).
    points : list of tuple
        ``(label, part, structure, kpoints)`` of each calculation, in the order
        of ``LABELS``: the id of the calculation without the step, the part of
        the interface it is, its structure and the k-points of
        ``VaspInputSet``.
    printed : str
        What the vendored code printed.
    """

    type: str
    site: str
    operations: list
    settings: object
    ids: dict
    read: dict
    parameters: dict
    top: list = field(default_factory=list)
    bottom: list = field(default_factory=list)
    integrate_in: list = field(default_factory=list)
    points: list = field(default_factory=list)
    printed: str = ''


# --- inputs ---------------------------------------------------------------------------

def charge_label(site, part):
    """Label of the calculation of a part of the interface: ``<site>_<part>``."""
    return f'{site}_{part}'


def fragment(structure, indices):
    """The atoms of a structure with the given indices, as a structure of the same kind.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
    indices : list of int
        Indices of the atoms to keep, in the order wanted.

    Returns
    -------
    pymatgen.core.structure.Structure
        Same cell, the sites and their properties as they are in ``structure``.
    """
    data = structure.as_dict()
    data['sites'] = [data['sites'][i] for i in indices]
    return type(structure).from_dict(data)


def contact_plane_problem(fixed, top, bottom):
    """Why the region between the two slabs of an interface cannot be told from z = 0, or None.

    Parameters
    ----------
    fixed : pymatgen.core.structure.Structure
        The interface once brought around z = 0 (``fix_interface``).
    top, bottom : list of int
        Indices of the atoms of the top slab and of the bottom slab.

    Returns
    -------
    str or None
        What is wrong when an atom lies exactly at z = 0 (it belongs to neither side of the
        plane), or the highest atom below z = 0 is not of the bottom slab, or the lowest atom
        above it is not of the top slab, or no atom lies below or above; None when the
        contact plane of the slabs is at z = 0.
    """
    z = np.array([site.coords[2] for site in fixed])
    on_plane = np.flatnonzero(z == 0)
    if len(on_plane):
        atoms = ', '.join(str(int(i)) for i in on_plane)
        return f'the plane z = 0 passes through the atoms {atoms}, on neither side of it'
    below, above = np.flatnonzero(z < 0), np.flatnonzero(z > 0)
    if not len(below) or not len(above):
        return 'no atom lies below z = 0' if not len(below) else 'no atom lies above z = 0'
    lowest_above = int(above[np.argmin(z[above])])
    highest_below = int(below[np.argmax(z[below])])
    wrong = []
    if highest_below not in set(bottom):
        wrong.append(f'the highest atom below z = 0 (atom {highest_below}) is of the slab '
                     f'{TOP_SLAB!r}')
    if lowest_above not in set(top):
        wrong.append(f'the lowest atom above z = 0 (atom {lowest_above}) is of the slab '
                     f'{BOTTOM_SLAB!r}')
    return '; '.join(wrong) or None


def charge_inputs(run_dir, system, settings):
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
    ChargeInputs

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
        are not valid or are not those of the input (``slab_on_top``), or the
        contact plane of the slabs cannot be placed at z = 0 once the interface is
        brought around it: an atom lies exactly at z = 0, or the atoms next to it
        are not those of the two slabs (``contact_plane_problem``).

    Notes
    -----
    To remove: ``FT_StartChargeDisplacement.query_interface`` (``run.py:1865-1884`` with
    ``core.py:493-556``) up to ``ElectronicsWF.charge_displacement`` (``electronics.py
    :78-93``), whose ``generate_chargedisp_inputs`` (``manipulate_struct.py:567-607``)
    is repeated here with the atoms of each slab (DEV-45): the translation by half a cell
    of the three structures, without wrapping the atoms, and their input sets. The
    document is found with ``mid`` and ``miller``, as there; ``comp_params`` comes from the
    settings instead of the document (DEV-35). The vendored code changes in place the
    structures that it is given, and prints: the entry read is copied first, so that
    ``read`` (the provenance) is what the document holds.
    """
    options = settings.interface.charge
    kind, site = settings.interface.adhesion.type, options.site
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

    # The vendored code prints, and warns about the POTCAR functional of the VASP input set,
    # which pw.x does not use, and about the interface it brings around z = 0 without the
    # interface it started from.
    printed = io.StringIO()
    comp_params = {'kdens': settings.k_points.kdens}
    with contextlib.redirect_stdout(printed), warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Overriding the POTCAR functional', BadInputSetWarning)
        warnings.filterwarnings('ignore', 'Interface_init not provided', UserWarning)
        structure = select_struct_func(interface['@class']).from_dict(copy.deepcopy(interface))
        top = slab_on_top(system, sdir, structure, kind, site)
        bottom = [i for i in range(len(structure)) if i not in set(top)]
        # The interface around z = 0, with the same atoms in the same order.
        fixed = ManipulateStruct(structure).fix_interface(structure)
        problem = contact_plane_problem(fixed, top, bottom)
        if problem is None:
            try:
                low, up = ManipulateStruct(fixed).estimate_interface(INTER_TYPE, AXIS)
            except ValueError as error:  # pragma: no cover (contact_plane_problem tells first)
                problem = str(error)
        if problem is not None:
            raise StoreError(f'{".".join(entry)}: the contact plane of the slabs cannot be placed '
                             f'at z = 0 once the interface is brought around it ({problem}): the '
                             'region between the slabs cannot be told')
        half = structure.lattice.c / 2
        integrate_in = [float(low) + half, float(up) + half]
        parts = {'interface': fixed, 'bot_slab': fragment(fixed, bottom),
                 'top_slab': fragment(fixed, top)}
        points = []
        for part in LABELS:
            piece = parts[part]
            piece.translate_sites(indices=np.arange(len(piece.sites)), vector=[0, 0, half],
                                  frac_coords=False, to_unit_cell=False)
            vis = VaspInputSet(piece, comp_params, CALC_TYPE).get_vasp_settings()
            points.append((charge_label(site, part), part, piece, vis.kpoints))
    parameters = {'type': kind, 'site': site, 'operations': list(options.operations),
                  'calc_type': CALC_TYPE, 'inter_type': INTER_TYPE, 'axis': AXIS,
                  'rescale': RESCALE, 'comp_params': comp_params}
    return ChargeInputs(type=kind, site=site, operations=list(options.operations),
                        settings=settings, ids=ids, read=read, parameters=parameters, top=top,
                        bottom=bottom, integrate_in=integrate_in, points=points,
                        printed=printed.getvalue())


def charge_calculations(system, inputs):
    """The pw.x calculations that also give the charge density, as ``preparation`` makes them.

    Returns
    -------
    list of QECalculation
        ``charge/<site>_<part>`` in the order of ``LABELS``.

    Raises
    ------
    SettingsError
        If ``[execution] pp_command`` is missing or ``disk_io`` is a value with which pw.x
        writes no charge density (``docs/qe_charge.md``).
    """
    return [workflow_calculation(system, f'{STEP}/{label}', CALC_TYPE, structure, kpoints,
                                 inputs.settings, density=True)
            for label, _, structure, kpoints in inputs.points]


def charge_provenance(inputs, calculations, files=None):
    """Provenance that the entries of the step have when computed now.

    Parameters
    ----------
    inputs : ChargeInputs
    calculations : dict
        ``{calc_id: fingerprint}`` of the three calculations.
    files : dict, optional
        ``{name: path}`` of the files of the densities read (``density_files``); their digests
        are part of the provenance. Without it, the provenance of the calculations and the
        entry read only.

    Returns
    -------
    Provenance
        The parameters of the step (the operations among them), the calculations and the
        content of the entry of the document read.
    """
    return Provenance.build(STEP, parameters=inputs.parameters, calculations=calculations,
                            inputs=inputs.read, files=files)


# --- the entries ----------------------------------------------------------------------

def data_entry(kind, site):
    """Entry that holds the results of the calculations of a site: ``charge.<kind>.data.<site>``."""
    return (STEP, kind, 'data', site)


def average_entry(kind, site):
    """Entry that holds the planar averages of a site: ``charge.<kind>.average.<site>``."""
    return (STEP, kind, 'average', site)


def displacement_entry(kind, site):
    """Entry that holds the charge displacement of a site: ``charge.<kind>.displacement.<site>``."""
    return (STEP, kind, 'displacement', site)


def charge_leaves(kind, site):
    """The entries that the step writes, each with the provenance of the step.

    Returns
    -------
    list of tuple of str
        ``data_entry``, ``average_entry`` and ``displacement_entry``; the other sites and
        kinds, and the other steps, are not touched.
    """
    return [data_entry(kind, site), average_entry(kind, site), displacement_entry(kind, site)]


# --- the densities --------------------------------------------------------------------

def density_files(run_dir, system, records):
    """The files of the density and of the structure of the three calculations.

    Parameters
    ----------
    run_dir : path-like
    system : str
    records : list of triboworkflow.core.store.Record
        The records of the three calculations, in the order of ``LABELS``.

    Returns
    -------
    dict
        ``{part: (filplot, structure)}``: the file that pp.x wrote in the last attempt of the
        calculation and the final structure written next to it (``structure_out.extxyz``).

    Raises
    ------
    PrerequisiteError
        If a calculation has no attempt that wrote a file of the density, or the file is gone.
    """
    files = {}
    for part, record in zip(LABELS, records):
        attempt = record.attempts[-1] if record.attempts else None
        run = None if attempt is None else attempt.density
        where = f'{system}/{record.calc_id}'
        if run is None or run.density is None:
            raise PrerequisiteError(f'{where}: no file of the charge density in the last attempt '
                                    '(pp.x did not write it); run the calculation again')
        path = calc_dir(run_dir, system, record.calc_id) / run.density.path
        if not path.is_file():
            raise PrerequisiteError(f'{where}: the file of the charge density {run.density.path} '
                                    'is gone; run the calculation again')
        files[part] = (path, path.parent / STRUCTURE_OUT_FILE)
    return files


def read_chgcars(files, records):
    """The three densities as ``Chgcar`` objects, checked to be combinable.

    Parameters
    ----------
    files : dict
        ``{part: (filplot, structure)}`` (``density_files``).
    records : list of triboworkflow.core.store.Record
        The records of the three calculations.

    Returns
    -------
    list of pymatgen.io.vasp.outputs.Chgcar
        In the order of ``LABELS``: ``data['total']`` is the density times the volume of the
        cell, as in a CHGCAR (``density.to_chgcar``).

    Raises
    ------
    triboworkflow.engines.qe.density.DensityError
        If a file cannot be read or disagrees with the structure of its calculation, or the
        three densities are not on the same grid, in the same cell, from the same cutoffs
        and the same energy model (``density.check_comparable``).
    StoreError
        If the final structure of a calculation cannot be read.
    """
    filplots = {part: density.read_filplot(path) for part, (path, _) in files.items()}
    density.check_comparable(filplots, records)
    chgcars = []
    for part, (_, structure_path) in files.items():
        try:
            structure = read_structure(structure_path, format='extxyz')
        except (OSError, StructureError) as error:
            raise StoreError(f'{structure_path}: the final structure of the calculation is not '
                             f'read: {error}') from error
        chgcars.append(density.to_chgcar(filplots[part], structure))
    return chgcars


# --- analysis -------------------------------------------------------------------------

def transfer_entries(kind, site, part):
    """Keys read in the task document of a calculation and entries where they are written.

    Parameters
    ----------
    kind : str
        ``regular`` or ``short``.
    site : str
    part : str
        One of ``LABELS``.

    Returns
    -------
    entry_from, entry_to : list of list of str
        One entry per key, in the same order: those of ``TRANSFER_FROM``, each to
        ``charge.<kind>.data.<site>.<part>.<last part>``, then the final structure to
        ``.structure``.

    Notes
    -----
    To remove: ``GeneralTribchemWF._select_entry('chgdisp')`` (``wfcore.py:130-143``) for
    one calculation.
    """
    base = [STEP, kind, 'data', site, part]
    entry_from = [list(path) for path in TRANSFER_FROM] + [['output', 'structure']]
    entry_to = [base + [path[-1]] for path in TRANSFER_FROM] + [base + ['structure']]
    return entry_from, entry_to


def move_results(store, ids, kind, site, part, calc_id):
    """Copy the results of the calculation of a part into the document.

    Parameters
    ----------
    store : LegacyStore
        On the document of the system.
    ids : dict
        Ids of the document (``ChargeInputs.ids``).
    kind, site, part : str
    calc_id : str
        ``charge/<site>_<part>``, the ``task_label`` of its task document.

    Raises
    ------
    StoreError
        If the calculation has no task document: no record, or a record whose results cannot
        be used.

    Notes
    -----
    To remove: ``FT_MoveTagResults.query_db`` and ``update_db`` (``dbtask.py:333-374``) with the
    defaults of ``MoveTagResults`` (``defaults.json``: ``tag_key`` ``task_label``,
    ``update_op`` ``$set``, ``to_mongodb`` true).
    """
    defaults = read_default_params(str(DEFAULTS), 'MoveTagResults', {})
    entry_from, entry_to = transfer_entries(kind, site, part)
    field_, field_entry = store.query_db(TASKS, entry=entry_from,
                                         fltr={defaults['tag_key']: calc_id})
    if field_ is None:
        raise StoreError(f'{calc_id}: no record of the calculation')
    store.update_db(field_entry, COLLECTION, entry=entry_to, fltr=set_filter(ids),
                    update_op=defaults['update_op'], to_mongodb=defaults['to_mongodb'])


def average_charge(store, ids, kind, site, chgcars, c_2):
    """Write the planar average of each density along x, y and z.

    Parameters
    ----------
    store : LegacyStore
        On the document of the system.
    ids : dict
        Ids of the document.
    kind, site : str
    chgcars : list of pymatgen.io.vasp.outputs.Chgcar
        The densities of the interface, of the bottom slab and of the top slab.
    c_2 : float
        Half the third vector of the cell of the interface, in angstrom: the shift that puts the
        contact plane at z = 0 in the curve along z.

    Returns
    -------
    dict
        What is written in ``charge.<kind>.average``: for each site and part, the curve ``[x,
        density]`` of each axis, ``x`` in angstrom from the origin of the cell (for z, from the
        contact plane) and the density in e/angstrom^3.

    Notes
    -----
    To remove: ``FT_AverageCharge._make_average_and_update`` (``charge.py:106-160``).
    """
    data = {'average': {site: {}}}
    for chgcar, part in zip(chgcars, LABELS):
        data['average'][site][part] = {}
        for axis in (0, 1, 2):
            curve = chgcar_average_along_axis(chgcar, axis, to_plot=False)
            if axis == 2:
                curve[0] = list(np.array(curve[0]) - c_2)
            data['average'][site][part][AXIS_NAMES[axis]] = curve
    data = jsanitize(data)
    store.update_db(data, COLLECTION, entry=[STEP, kind], fltr=set_filter(ids))
    return data


def displace_charge(store, ids, kind, site, chgcars, operations, integrate_in, c_2):
    """Combine the three densities and write the integral of the planar average.

    Parameters
    ----------
    store : LegacyStore
        On the document of the system.
    ids : dict
        Ids of the document.
    kind, site : str
    chgcars : list of pymatgen.io.vasp.outputs.Chgcar
        The densities of the interface, of the bottom slab and of the top slab.
    operations : list of str
        ``+`` or ``-`` twice: the interface with the bottom slab, then with the top slab.
    integrate_in : list of float
        Limits of the region of the interface along z, in angstrom.
    c_2 : float
        Half the third vector of the cell, in angstrom.

    Returns
    -------
    dict
        What is written in ``charge.<kind>.displacement``.

    Notes
    -----
    To remove: ``FT_ChargeDisplacement.calculate_charge_displacement`` (``charge.py:190-260``)
    with ``rescale`` 1000 and the axis 2.
    """
    # The densities are of different structures by construction (the atoms of the interface and
    # of each slab), which pymatgen warns about when it combines them.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Structures are different', UserWarning)
        combined = chgcar_algebra_from_list(chgcars, operations, AXIS)

    # The integral in the region of the interface, and the curve it is taken from.
    integral, width, curve, _ = charge_displacement(combined, AXIS, integrate_in[0],
                                                    integrate_in[1], RESCALE, False)
    # Shift the x of the curve back on zero, so the interface is z = 0.
    curve[0] = list(np.array(curve[0]) - c_2)
    # The same over the whole cell.
    integral_all, width_all, _, _ = charge_displacement(combined, AXIS, rescale=RESCALE,
                                                        to_plot=False)
    data = {'displacement': {site: {AXIS_NAMES[AXIS]: {
        'all': {'integral': float(integral_all), 'width': float(width_all)},
        'interface': {'integral': float(integral), 'width': float(width)},
        'grid': jsanitize(curve),
        'rescale': RESCALE}}}}
    store.update_db(data, COLLECTION, entry=[STEP, kind], fltr=set_filter(ids))
    return data


def run_charge_analysis(inputs, store, calc_ids, chgcars):
    """Copy the results of the three calculations, write the averages and the displacement.

    Parameters
    ----------
    inputs : ChargeInputs
    store : LegacyStore
        On the document of the system, which is changed and not saved.
    calc_ids : list of str
        ``calc_id`` of the three calculations, in the order of ``LABELS``.
    chgcars : list of pymatgen.io.vasp.outputs.Chgcar
        Their densities (``read_chgcars``).

    Returns
    -------
    dict
        What ``displace_charge`` wrote.
    """
    for part, calc_id in zip(LABELS, calc_ids):
        move_results(store, inputs.ids, inputs.type, inputs.site, part, calc_id)
    c_2 = inputs.read['interface']['lattice']['c'] / 2
    average_charge(store, inputs.ids, inputs.type, inputs.site, chgcars, c_2)
    return displace_charge(store, inputs.ids, inputs.type, inputs.site, chgcars,
                           inputs.operations, inputs.integrate_in, c_2)


def describe(inputs, values):
    """Lines of the message of the step: the site, the operations and the integrals.

    Parameters
    ----------
    inputs : ChargeInputs
    values : dict
        As ``displace_charge`` returns it.

    Returns
    -------
    list of str
    """
    z = values['displacement'][inputs.site][AXIS_NAMES[AXIS]]
    first, second = inputs.operations
    return [f'{STEP}: {inputs.type} charge displacement at {inputs.site}: interface {first} '
            f'bottom slab {second} top slab ({inputs.parameters["calc_type"]})',
            f'integral of the planar average of the difference, over the region of the '
            f'interface ({z["interface"]["width"]:.4g} angstrom): '
            f'{z["interface"]["integral"]:.6g} e/angstrom^3; over the cell: '
            f'{z["all"]["integral"]:.6g} e/angstrom^3']


# --- step -----------------------------------------------------------------------------

def _load_run(run_dir):
    # Imported here: the command line imports the registry of the steps.
    from triboworkflow.cli.run import load_run

    return load_run(run_dir)


class ChargeStep(Step):
    """Step ``charge``: the charge density of the relaxed interface and of its two slabs."""

    name = STEP

    def done(self, run_dir, system):
        """Whether its entries (``charge_leaves``) are written and current (``provenance``).

        The adhesion, the surface and ``match``, which it reads, have to be done too: the
        provenance holds the content of the entries read, not whether they are still current.
        The files of the densities must be there, since the provenance holds their digests.
        """
        sdir = system_dir(run_dir, system)
        document = read_system(sdir)
        settings = _load_run(run_dir)
        entries = charge_leaves(settings.interface.adhesion.type, settings.interface.charge.site)
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
        """Read the calculations and their densities, write the averages and the displacement.

        Only the entries of the kind and site of the run are written, in the same saving: the
        results of the other sites and kinds stay.

        Raises
        ------
        PrerequisiteError
            If the interface read, or a step behind it, is missing or not current, or a
            calculation has no file of the density.
        ValueError
            If ``calculations`` are not those that the settings give now.
        StoreError
            If a record is missing, not complete, of another fingerprint, or its density is not
            ``ok``, or the records do not share one energy model. Nothing is saved.
        triboworkflow.engines.qe.density.DensityError
            If a file of a density is not valid, or the three cannot be combined. Nothing is
            saved.
        """
        sdir = system_dir(run_dir, system)
        with SystemWriter(sdir) as writer:
            check_prerequisites(writer.document, self.expected(run_dir, system), sdir)
            inputs = self.inputs(run_dir, system)
            plan = {calc.calc_id: calc.record.fingerprint
                    for calc in charge_calculations(system, inputs)}
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
            # The first problem of a record is the one to tell, before its files.
            charge_provenance(inputs, plan).with_results(records)
            files = density_files(run_dir, system, records)
            provenance = charge_provenance(
                inputs, plan, {f'density_{part}': path for part, (path, _) in files.items()}
            ).with_results(records)
            check_energy_model(records)
            chgcars = read_chgcars(files, records)
            values = run_charge_analysis(inputs, LegacyStore(sdir, writer.document, COLLECTION),
                                         list(plan), chgcars)
            for entry in charge_leaves(inputs.type, inputs.site):
                writer.document.set_provenance(entry, provenance)
            writer.save()
        for line in describe(inputs, values):
            logger.info('%s: %s', system, line)

    def inputs(self, run_dir, system):
        """``charge_inputs`` of the step, from the settings of the run."""
        inputs = charge_inputs(run_dir, system, _load_run(run_dir))
        if inputs.printed:
            logger.debug('%s/%s: output of the vendored code:\n%s', system, self.name,
                         inputs.printed)
        return inputs

    def calculations(self, run_dir, system):
        """The interface and its two slabs, nothing written."""
        return charge_calculations(system, self.inputs(run_dir, system))

    def expected(self, run_dir, system):
        """The interface read, with the provenance of the adhesion now.

        Raises
        ------
        PrerequisiteError
            If a step behind the adhesion (its surface or ``match``) is not done or not
            current: the provenance of the adhesion holds the content that it read and would
            not tell.
        """
        settings = _load_run(run_dir)
        adhesion = AdhesionStep()
        check_upstream(adhesion, run_dir, system)
        entry = interface_entry(settings.interface.adhesion.type, settings.interface.charge.site)
        return {entry: adhesion.provenance(run_dir, system)}

    def provenance(self, run_dir, system):
        """Provenance expected now for the entries that the step writes.

        Raises
        ------
        PrerequisiteError
            If a calculation has no record or no file of its density: the digests of the files
            are part of the provenance.
        """
        inputs = self.inputs(run_dir, system)
        plan = {calc.calc_id: calc.record.fingerprint
                for calc in charge_calculations(system, inputs)}
        sdir = system_dir(run_dir, system)
        records = []
        for calc_id in plan:
            try:
                records.append(read_record(sdir, calc_id))
            except FileNotFoundError:
                raise PrerequisiteError(f'{system}/{calc_id}: no record of the calculation; '
                                        'run it first') from None
        files = density_files(run_dir, system, records)
        return charge_provenance(inputs, plan,
                                 {f'density_{part}': path for part, (path, _) in files.items()})
