# Author: Vito Foderà
# Created: 2026-11-08

"""Step ``adhesion`` of the interface workflow: adhesion energy and corrugation.

The adhesion energy of an interface is the energy that binds its two slabs:
the energy of the interface minus those of the two isolated slabs. The step
computes it at two lateral positions of the top slab on the bottom one, the
site of lowest energy and the site of highest energy of a potential energy
surface (PES), and takes the difference between the two as the corrugation.
Four pw.x calculations, ``adhesion/<label>``:

- ``bot_slab`` and ``top_slab``: the two slabs alone, with all the atoms free
  (``calc_type`` ``slab_pos_relax``);
- ``interface_min``: the interface at the site of lowest energy, all the atoms
  free (``slab_pos_relax``);
- ``interface_max``: the interface at the site of highest energy, relaxed
  along z only, the hydrogen atoms in every direction (``interface_z_relax``).

``[interface.adhesion]`` chooses, for the whole run, the PES that gives the two
sites and the kind of adhesion (``AdhesionParams``); the graph of the
orchestrator makes the step wait for that PES (``interface_graph``):

- ``pes`` and ``regular``: the interfaces are the final structures of the
  minimum and of the maximum of the relaxed PES (``structure.opt.pes.min`` and
  ``.max``), and the slabs are those of ``match``;
- ``pes_scf`` and ``regular``: the interfaces are made by stacking the complete
  slabs of ``match`` at the sites of lowest and highest energy of the static
  PES (``min_site`` and ``max_site`` of ``pes_scf.energy``, with the shifts of
  ``highsym.inter_unique``), at the distance of 2 angstrom of
  ``apply_interface_shifts``, whatever ``interface_distance`` is;
- ``pes_scf`` and ``short``: the interfaces are the input structures of the
  static PES at those two sites, which need ``short = true`` in
  ``[interface.pes_scf]``, and the slabs are the two parts of the interface
  of the minimum, taken as slabs again.

The inputs (``adhesion_inputs``) come from the document of the system, which
``check_prerequisites`` requires current: the slabs of ``match`` and the
entries of the PES. The structures, the constraints and the k-point grid of
each calculation are those of the vendored workflow code (``VaspInputSet``),
with the grid from ``[k_points].kdens``; the pw.x input is written from them
(``preparation.workflow_calculation``).

The analysis (``AdhesionStep.analyse``) uses the records of the calculations
only when each one is complete with the fingerprint of the settings and the
four share one energy model (``check_energy_model``). It writes, with ``<kind>``
the kind of adhesion and ``<data>`` the entry ``data`` (``data.short`` with
``short``):

- ``<data>.<label>``: the results of each calculation (``move_results``), one
  key for each of ``TRANSFER_FROM`` (None when the calculation does not give
  it), without the input structure;
- ``structure.opt.<kind>.<label>``: the final structure of each calculation,
  with the two slabs given back their Miller indices and the other keys of a
  slab (``change_slabs``);
- ``adhesion.<kind>`` (``calculate_adhesion``): for the sites of lowest and
  highest energy the adhesion energy, the energy of the interface minus those
  of the two slabs, in eV and in J/m^2 on the area of the cell of the relaxed
  interface at the lowest site; the corrugation, the energy at the highest
  site minus the energy at the lowest one;
- with ``short``, ``structure.init.short`` (``write_short_init``).

Each entry written (``adhesion_leaves``) gets the provenance of the step: the
parameters, the calculations with their results, and the entries read. The
writes replace the same keys at every analysis, so nothing is emptied first,
and the entries of the other kind are not touched. Nothing is saved when a
part fails.

Notes
-----
To remove: port of TribChem's ``FT_StartAdhesion`` (``highput/firetasks/run.py
:1312-1468``, with ``FireTaskInterface.query_interface``, ``core.py:493-556``)
and of ``InterfaceWF.adhesion_energy`` (``highput/workflows/interface.py
:246-354``). TribChem's ``set_adh`` (``core/cli/calc_interface_cli.py
:201-230``) fixes the ``calc_type`` of the four calculations and reads the
surface and the kind of adhesion as two options (D56, ``AdhesionParams``);
the other ``calc_type`` that the workflow function allows at one's own risk
are not ported. ``comp_params`` is ``{'kdens': [k_points].kdens}`` (DEV-35);
the ids are those of the document written by ``match`` (A80). The distance of
the interface of ``pes_scf`` with ``regular`` is the default of
``apply_interface_shifts`` (A86). Not ported: ``override``, ``run_vasp``,
``get_main_folder`` (A47) and ``FT_SaveVASPOutputs``. TribChem never passed
``check_entry``, so the adhesion was never found done (A57): here it is done
when its entries are written and their provenance is current (DEV-02), and
``structure.init.short`` is written with the results, not before the
calculations. The analysis ports ``FT_MoveTagResults`` (``highput/firetasks/
dbtask.py:166-374``) with the entries of ``GeneralTribchemWF._select_entry
('adhesion')`` (``highput/workflows/wfcore.py:61-111``), then
``FT_AdhesionEnergy`` (``highput/firetasks/adhesion.py:22-171``): its writes
follow the same order, and the lines of ``_change_slabs`` that moved the slabs
back to z = 0 with ``fix_interface`` are commented out there and not ported.
"""

import contextlib
import copy
import io
import logging
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
from triboworkflow.vendored.highput.database.dbtools import read_multiple_entry, read_one_entry
from triboworkflow.vendored.highput.utils.manipulate_struct import (ManipulateStruct,
                                                                     apply_interface_shifts,
                                                                     structure_to_slab)
from triboworkflow.vendored.highput.utils.tasktools import read_default_params, select_struct_func
from triboworkflow.vendored.highput.utils.vasp_tools import VaspInputSet
from triboworkflow.vendored.physics.base.solidstate import get_cell_info
from triboworkflow.vendored.physics.base.units import Converter

from .match import (COLLECTION, DEFAULTS, FILTER_KEYS, match_parameters, match_provenance,
                    set_filter)
from .pes import ID_KEYS, TRANSFER_FROM, RelaxedPesStep, StaticPesStep

logger = logging.getLogger(__name__)

STEP = 'adhesion'
# The four calculations, in the order of the energies of the original workflow
# for the structures: the two slabs, then the interface at the minimum and at
# the maximum.
LABELS = ('bot_slab', 'top_slab', 'interface_min', 'interface_max')
# calc_type of each calculation: all the atoms free, except the interface at
# the maximum, which relaxes along z only.
# To remove: set_adh of TribChem's command line (calc_interface_cli.py:201-230)
# and c_max of InterfaceWF.adhesion_energy (interface.py:313-319).
CALC_TYPES = {'bot_slab': 'slab_pos_relax', 'top_slab': 'slab_pos_relax',
              'interface_min': 'slab_pos_relax', 'interface_max': 'interface_z_relax'}
# The PES steps that give the two sites.
PES_STEPS = {'pes': RelaxedPesStep, 'pes_scf': StaticPesStep}
# Entries of the document read from match.
BOT_SLAB = ('structure', 'init', 'bot_slab')
TOP_SLAB = ('structure', 'init', 'top_slab')
HIGHSYM = ('highsym',)
SHIFTS = ('highsym', 'inter_unique')
# Structures written in structure.init.short with the kind short.
SHORT_INIT = ('bot_slab', 'top_slab', 'interface')


@dataclass
class AdhesionInputs:
    """What the step reads from the settings and from the document of the system.

    Attributes
    ----------
    pes, type : str
        ``[interface.adhesion]``: the PES step that gives the sites, and the
        kind of adhesion, ``regular`` or ``short``.
    settings : triboworkflow.core.settings.Settings
    ids : dict
        ``mid``, ``formula``, ``name`` and ``miller`` of the document, which
        select it in the writes.
    read : dict
        The entries of the document read, as stored, by name: ``bot_slab`` and
        ``top_slab`` of ``match``; then ``min`` and ``max`` (the structures of
        the relaxed PES), or ``min_site`` and ``max_site`` with ``shifts``
        (``regular``) or with ``min_input`` and ``max_input`` (``short``) of
        the static PES.
    sites : tuple of str or None
        The sites of lowest and highest energy of the static PES; None with
        the relaxed PES.
    init : dict
        The structures of the slabs before the calculations, by ``bot_slab``
        and ``top_slab`` (and ``interface`` with ``short``): those of
        ``match``, or the reduced ones with ``short``. With ``short`` they are
        written in ``structure.init.short``.
    parameters : dict
        Parameters of the step (``pes``, ``type``, ``calc_types``,
        ``comp_params``), recorded in the provenance.
    points : list of tuple
        ``(label, calc_type, structure, kpoints)`` of each calculation, in the
        order of ``LABELS``: the structure with the ``selective_dynamics`` of
        the ``calc_type`` and the k-points of ``VaspInputSet``.
    printed : str
        What the vendored code printed.
    """

    pes: str
    type: str
    settings: object
    ids: dict
    read: dict
    sites: tuple | None
    init: dict
    parameters: dict
    points: list = field(default_factory=list)
    printed: str = ''


# --- inputs ---------------------------------------------------------------------------

def _structure(data):
    return select_struct_func(data['@class']).from_dict(data)


def _fetch(document, system, wanted, producer):
    """Entries of the document by name, or the ``PrerequisiteError`` naming those missing."""
    read = {name: read_one_entry(document, list(entry)) for name, entry in wanted.items()}
    missing = ['.'.join(entry) for name, entry in wanted.items() if not read[name]]
    if missing:
        raise PrerequisiteError(f'{system}: entries missing: {", ".join(missing)}; run '
                                f'{producer} first')
    return read


def entries_read(pes, kind):
    """The entries of the document that the step reads first, by name and by producer.

    Parameters
    ----------
    pes : str
        ``pes`` or ``pes_scf``.
    kind : str
        ``regular`` or ``short``.

    Returns
    -------
    match, surface : dict
        ``{name: entry}`` read from ``match`` (the two slabs and, with the
        static PES and ``regular``, the shifts of the points) and from the PES
        step: the final structures of its minimum and maximum, or its sites of
        lowest and highest energy. With ``short`` the input structures at
        those sites are read next (``site_inputs``).
    """
    match = {'bot_slab': BOT_SLAB, 'top_slab': TOP_SLAB}
    if pes == 'pes':
        surface = {'min': ('structure', 'opt', 'pes', 'min'),
                   'max': ('structure', 'opt', 'pes', 'max')}
    else:
        surface = {'min_site': ('pes_scf', 'energy', 'min_site'),
                   'max_site': ('pes_scf', 'energy', 'max_site')}
        if kind == 'regular':
            match['shifts'] = SHIFTS
    return match, surface


def site_inputs(sites):
    """Entries of the static PES that hold the input structures at two sites.

    Parameters
    ----------
    sites : tuple of str
        The sites of lowest and highest energy.

    Returns
    -------
    dict
        ``{name: entry}`` with ``min_input`` and ``max_input``.
    """
    return {'min_input': ('pes_scf', 'data', sites[0], 'input', 'structure'),
            'max_input': ('pes_scf', 'data', sites[1], 'input', 'structure')}


def adhesion_expected(run_dir, system, settings):
    """The entries that the step reads, each with the provenance its producer gives it now.

    Parameters
    ----------
    run_dir : path-like
    system : str
    settings : triboworkflow.core.settings.Settings

    Returns
    -------
    dict
        ``{entry: Provenance}``: the slabs (and ``highsym``, with the static
        PES and ``regular``) with the provenance of ``match``, and the
        entries of the PES with that of its step.

    Raises
    ------
    PrerequisiteError
        If the entries of ``match`` that the PES step needs are missing.
    """
    adhesion = settings.interface.adhesion
    leaves = {'pes': [('structure', 'opt', 'pes', 'min'), ('structure', 'opt', 'pes', 'max')],
              'pes_scf': [('pes_scf', 'energy', 'min_site'), ('pes_scf', 'energy', 'max_site')]}
    from_match = [BOT_SLAB, TOP_SLAB]
    surface = list(leaves[adhesion.pes])
    if adhesion.pes == 'pes_scf':
        if adhesion.type == 'regular':
            from_match.append(HIGHSYM)
        else:
            surface.append(('pes_scf', 'data'))
    provenance = match_provenance(settings)
    expected = {entry: provenance for entry in from_match}
    provenance = PES_STEPS[adhesion.pes]().provenance(run_dir, system)
    expected.update({entry: provenance for entry in surface})
    return expected


def adhesion_inputs(run_dir, system, settings):
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
    AdhesionInputs

    Raises
    ------
    SettingsError
        If the k-points are not given by their density (``[k_points]
        mode = "kdens"``): the cell of the interface is known only after the
        matching.
    PrerequisiteError
        If the document of the system, an entry of ``match`` or an entry of
        the PES is missing, or a site of the static PES has no shifts.
    StoreError
        If the document is of another interface.

    Notes
    -----
    To remove: ``FT_StartAdhesion.query_interface`` (``run.py:1375-1455``,
    with ``FireTaskInterface.query_interface``, ``core.py:493-556``) up to the
    ``VaspInputSet`` of ``InterfaceWF.adhesion_energy`` (``interface.py
    :321-323``). The document is found with ``mid`` and ``miller``, as there;
    ``comp_params`` comes from the settings instead of the document (DEV-35).
    The vendored code changes in place the dictionaries and the structures
    that it is given: the entries read are copied first, so that ``read``
    (the provenance) is what the document holds.
    """
    adhesion = settings.interface.adhesion
    pes, kind = adhesion.pes, adhesion.type
    if (pes, kind) == ('pes', 'short'):
        raise SettingsError('interface.adhesion.type', "'short' needs pes = 'pes_scf'",
                            settings.source)
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

    from_match, surface = entries_read(pes, kind)
    read = _fetch(field_, system, from_match, 'match')
    read.update(_fetch(field_, system, surface, pes))
    sites = None
    if pes == 'pes_scf':
        sites = (read['min_site'], read['max_site'])
        if kind == 'short':
            read.update(_fetch(field_, system, site_inputs(sites), pes))

    # The vendored code prints, warns about the POTCAR functional of the VASP
    # input set, which pw.x does not use, and about the interface that
    # fix_interface is given without a reference (the case it is used for).
    printed = io.StringIO()
    comp_params = {'kdens': settings.k_points.kdens}
    with contextlib.redirect_stdout(printed), warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Overriding the POTCAR functional', BadInputSetWarning)
        warnings.filterwarnings('ignore', 'Interface_init not provided')
        # The structures share their site properties with the dictionaries that they
        # are made from or give, and VaspInputSet changes them in place: everything
        # that is kept, or read for the provenance, is a copy.
        bot_slab = _structure(copy.deepcopy(read['bot_slab']))
        top_slab = _structure(copy.deepcopy(read['top_slab']))
        init = {'bot_slab': copy.deepcopy(read['bot_slab']),
                'top_slab': copy.deepcopy(read['top_slab'])}
        if pes == 'pes':
            interfaces = [copy.deepcopy(read['min']), copy.deepcopy(read['max'])]
        elif kind == 'regular':
            for site in sites:
                if site not in read['shifts']:
                    raise PrerequisiteError(f'{system}: the site {site} of the static PES is not '
                                            'among the points of highsym.inter_unique')
            # Without a distance: apply_interface_shifts takes its own (A86).
            interfaces = [apply_interface_shifts(bot_slab, top_slab, read['shifts'][site]).as_dict()
                          for site in sites]
        else:
            min_input, max_input = (copy.deepcopy(read['min_input']),
                                    copy.deepcopy(read['max_input']))
            interfaces = [ManipulateStruct(i).fix_interface(i, as_dict=True)
                          for i in (min_input, max_input)]
            # The two slabs are the parts of the interface of the minimum.
            bot_short, top_short = ManipulateStruct(interfaces[0]).decompose_structure(
                inter_type='zero', as_dict=True)
            bot_slab = structure_to_slab(bot_short, bot_slab, as_dict=False)
            top_slab = structure_to_slab(top_short, top_slab, as_dict=False)
            init = {'bot_slab': copy.deepcopy(bot_slab.as_dict()),
                    'top_slab': copy.deepcopy(top_slab.as_dict()),
                    'interface': copy.deepcopy(interfaces[0])}
        structures = [bot_slab, top_slab] + [_structure(i) for i in interfaces]
        points = []
        for label, structure in zip(LABELS, structures):
            vis = VaspInputSet(structure, comp_params, CALC_TYPES[label]).get_vasp_settings()
            points.append((label, CALC_TYPES[label], structure, vis.kpoints))
    parameters = {'pes': pes, 'type': kind, 'calc_types': dict(CALC_TYPES),
                  'comp_params': comp_params}
    return AdhesionInputs(pes=pes, type=kind, settings=settings, ids=ids, read=read, sites=sites,
                          init=init, parameters=parameters, points=points,
                          printed=printed.getvalue())


def adhesion_calculations(system, inputs):
    """The pw.x calculations, as ``preparation.workflow_calculation`` makes them.

    Returns
    -------
    list of QECalculation
        ``adhesion/<label>`` in the order of ``LABELS``.
    """
    return [workflow_calculation(system, f'{STEP}/{label}', calc_type, structure, kpoints,
                                 inputs.settings)
            for label, calc_type, structure, kpoints in inputs.points]


def adhesion_provenance(inputs, calculations):
    """Provenance that the entries of the step have when computed now.

    Parameters
    ----------
    inputs : AdhesionInputs
    calculations : dict
        ``{calc_id: fingerprint}`` of the four calculations.

    Returns
    -------
    Provenance
        The parameters of the step, the calculations and the content of the
        entries of the document read.
    """
    return Provenance.build(STEP, parameters=inputs.parameters, calculations=calculations,
                            inputs=inputs.read)


# --- the entries ----------------------------------------------------------------------

def data_entry(kind):
    """Entry of the document that holds the results of the four calculations.

    Returns
    -------
    tuple of str
        ``data`` for ``regular``, ``data.short`` for ``short``.
    """
    return ('data',) if kind == 'regular' else ('data', 'short')


def adhesion_leaves(kind):
    """The entries that the step writes, each with the provenance of the step.

    Parameters
    ----------
    kind : str
        ``regular`` or ``short``.

    Returns
    -------
    list of tuple of str
        ``adhesion.<kind>``, the results of the four calculations
        (``data_entry``), their final structures (``structure.opt.<kind>``)
        and, with ``short``, the reduced structures of ``structure.init.short``.
    """
    leaves = [(STEP, kind)]
    leaves += [data_entry(kind) + (label,) for label in LABELS]
    leaves += [('structure', 'opt', kind, label) for label in LABELS]
    if kind == 'short':
        leaves += [('structure', 'init', 'short', key) for key in SHORT_INIT]
    return leaves


# --- analysis -------------------------------------------------------------------------

def transfer_entries(kind, label):
    """Keys read in the task document of a calculation and entries where they are written.

    Parameters
    ----------
    kind : str
        ``regular`` or ``short``.
    label : str
        One of ``LABELS``.

    Returns
    -------
    entry_from, entry_to : list of list of str
        One entry per key, in the same order: those of ``TRANSFER_FROM``, each
        to ``<data>.<label>.<last part>`` (``data_entry``), then the final
        structure to ``structure.opt.<kind>.<label>``.

    Notes
    -----
    To remove: ``GeneralTribchemWF._select_entry('adhesion')`` (``wfcore.py
    :100-111``).
    """
    entry_from = [list(path) for path in TRANSFER_FROM] + [['output', 'structure']]
    data = list(data_entry(kind)) + [label]
    entry_to = [data + [path[-1]] for path in TRANSFER_FROM]
    entry_to.append(['structure', 'opt', kind, label])
    return entry_from, entry_to


def move_results(store, ids, kind, label, calc_id):
    """Copy the results of a calculation into the document.

    Parameters
    ----------
    store : LegacyStore
        On the document of the system.
    ids : dict
        Ids of the document (``AdhesionInputs.ids``).
    kind : str
    label : str
    calc_id : str
        ``adhesion/<label>``, the ``task_label`` of its task document.

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
    entry_from, entry_to = transfer_entries(kind, label)
    field_, field_entry = store.query_db(TASKS, entry=entry_from,
                                         fltr={defaults['tag_key']: calc_id})
    if field_ is None:
        raise StoreError(f'{calc_id}: no record of the calculation')
    store.update_db(field_entry, COLLECTION, entry=entry_to, fltr=set_filter(ids),
                    update_op=defaults['update_op'], to_mongodb=defaults['to_mongodb'])


def write_short_init(store, ids, inputs):
    """Write the reduced slabs and the interface of the minimum, with ``short``.

    Parameters
    ----------
    store : LegacyStore
    ids : dict
    inputs : AdhesionInputs
        Of the kind ``short``: ``init`` holds the structures written in
        ``structure.init.short``.

    Notes
    -----
    To remove: ``FT_StartAdhesion.query_interface`` (``run.py:1426-1432``),
    which wrote them before the calculations; they depend only on the PES
    (N4 of D56).
    """
    store.update_db({'init': {'short': inputs.init}}, COLLECTION, entry='structure',
                    fltr=set_filter(ids))


def calculate_adhesion(store, ids, kind):
    """Adhesion energy at the lowest and at the highest site, and the corrugation.

    Reads the energies written by ``move_results``. The adhesion energy is the
    energy of the interface minus those of the two slabs, negative for an
    interface that binds; the corrugation is the energy of the interface at
    the highest site minus that at the lowest one. The area of the J/m^2 is
    that of the cell of the relaxed interface at the lowest site.

    Parameters
    ----------
    store : LegacyStore
        On the document of the system, with the results of the four
        calculations. Not changed.
    ids : dict
    kind : str

    Returns
    -------
    dict
        ``{kind: {'min_site': {...}, 'max_site': {...}, 'corrugation_ev': ...,
        'corrugation_jm2': ...}}``, the entry ``adhesion`` of the document:
        each site with ``adhesion_energy_ev`` and ``adhesion_energy_jm2``.

    Raises
    ------
    StoreError
        If a calculation gave no energy.

    Notes
    -----
    To remove: ``FT_AdhesionEnergy.calculate_adhesion`` (``adhesion.py
    :48-105``).
    """
    fltr = set_filter(ids)
    field_, _ = store.query_db(COLLECTION, fltr=fltr)

    # Retrieve the energies from the document, in the order of the formulas
    names = ['interface_min', 'interface_max', 'top_slab', 'bot_slab']
    read_ene = [list(data_entry(kind)) + [label, 'energy'] for label in names]
    ene = read_multiple_entry(field_, read_ene)
    missing = [label for label, energy in zip(names, ene) if energy is None]
    if missing:
        raise StoreError(f'no energy in the results of {", ".join(missing)}')

    inter_min = read_one_entry(field_, ['structure', 'opt', kind, 'interface_min'])
    area, _ = get_cell_info(Structure.from_dict(inter_min).lattice.matrix)

    # Calculate adhesion and energy corrugation for min and max sites
    adhesion = {}
    for i, s in enumerate(['min_site', 'max_site']):
        # Calculate adhesion energy and convert it to J/m^2
        adhesion_ev = ene[i] - ene[2] - ene[3]
        adhesion_jm2 = Converter.eV_to_Jm2(adhesion_ev, area * 1e-20)
        adhesion.update({s: {'adhesion_energy_ev': adhesion_ev,
                             'adhesion_energy_jm2': adhesion_jm2}})

    # Calculate energetic corrugation and convert it to J/m^2
    corrugation_ev = ene[1] - ene[0]
    corrugation_jm2 = Converter.eV_to_Jm2(corrugation_ev, area * 1e-20)
    adhesion.update({'corrugation_ev': corrugation_ev, 'corrugation_jm2': corrugation_jm2})

    # Assign a further key depending on the type of calculation
    return {kind: adhesion}


def update_adhesion(store, ids, adhesion):
    """Write the adhesion energy and the corrugation in the entry ``adhesion``.

    Parameters
    ----------
    store : LegacyStore
    ids : dict
    adhesion : dict
        As ``calculate_adhesion`` gives it.

    Notes
    -----
    To remove: ``FT_AdhesionEnergy.update_db`` (``adhesion.py:162-170``).
    """
    store.update_db(adhesion, COLLECTION, entry=STEP, fltr=set_filter(ids))


def change_slabs(store, ids, kind):
    """Give back to the two slabs their type of slab, in ``structure.opt``.

    A calculation gives a plain structure: the final structures of the slabs
    take from the slabs they started from (``structure.init``, or
    ``structure.init.short`` with ``short``) the Miller indices and the other
    keys of a slab. The final structures of the two interfaces are written
    again as they are.

    Parameters
    ----------
    store : LegacyStore
        On the document of the system, with the final structures of the four
        calculations (``move_results``).
    ids : dict
    kind : str

    Notes
    -----
    To remove: ``FT_AdhesionEnergy._change_slabs`` (``adhesion.py:107-160``);
    the lines that shifted the slabs and the interface back to z = 0 with
    ``fix_interface`` are commented out there (147-155) and not ported.
    """
    fltr = set_filter(ids)
    field_, _ = store.query_db(COLLECTION, fltr=fltr)

    # Read input information for slabs
    init = ['structure', 'init'] if kind == 'regular' else ['structure', 'init', 'short']
    init_bot = read_one_entry(field_, init + ['bot_slab'])
    init_top = read_one_entry(field_, init + ['top_slab'])

    # Read corresponding output structures
    opt_bot = read_one_entry(field_, ['structure', 'opt', kind, 'bot_slab'])
    opt_top = read_one_entry(field_, ['structure', 'opt', kind, 'top_slab'])
    inter_min_opt = read_one_entry(field_, ['structure', 'opt', kind, 'interface_min'])
    inter_max_opt = read_one_entry(field_, ['structure', 'opt', kind, 'interface_max'])

    # Convert the slabs from Structures to Slabs
    opt_bot = structure_to_slab(opt_bot, init_bot, as_dict=True)
    opt_top = structure_to_slab(opt_top, init_top, as_dict=True)

    # Update them in the document
    slabs = [opt_bot, opt_top, inter_min_opt, inter_max_opt]
    for label, structure in zip(LABELS, slabs):
        store.update_db(structure, COLLECTION, entry=['structure', 'opt', kind, label],
                        fltr=fltr)


def run_adhesion_analysis(inputs, store, calc_ids):
    """Write the results of the four calculations, the adhesion energy and the slabs.

    Parameters
    ----------
    inputs : AdhesionInputs
    store : LegacyStore
        On the document of the system, which is changed and not saved.
    calc_ids : list of str
        ``calc_id`` of the calculations, in the order of ``LABELS``.

    Returns
    -------
    dict
        What ``calculate_adhesion`` gave and ``update_adhesion`` wrote.

    Notes
    -----
    The writes follow the order of the original: the reduced structures, the
    results of the four calculations, the slabs, then the adhesion energy.
    """
    if inputs.type == 'short':
        write_short_init(store, inputs.ids, inputs)
    for label, calc_id in zip(LABELS, calc_ids):
        move_results(store, inputs.ids, inputs.type, label, calc_id)
    adhesion = calculate_adhesion(store, inputs.ids, inputs.type)
    change_slabs(store, inputs.ids, inputs.type)
    update_adhesion(store, inputs.ids, adhesion)
    return adhesion


def describe(inputs, adhesion, sites):
    """Lines of the message of the step: surface, sites, adhesion energy and corrugation.

    Parameters
    ----------
    inputs : AdhesionInputs
    adhesion : dict
        As ``calculate_adhesion`` returns it.
    sites : tuple of str
        The sites of lowest and highest energy of the surface.

    Returns
    -------
    list of str
    """
    surface = 'relaxed along z' if inputs.pes == 'pes' else 'static'
    kind = 'on the complete slabs' if inputs.type == 'regular' else 'on the reduced slabs'
    values = adhesion[inputs.type]
    lines = [f'{STEP}: {inputs.type} adhesion {kind}, from the PES {inputs.pes} ({surface}), '
             f'at {sites[0]} (lowest energy) and {sites[1]} (highest energy)']
    for site, name in zip(sites, ('min_site', 'max_site')):
        lines.append(f'adhesion energy at {site}: {values[name]["adhesion_energy_ev"]:.6f} eV, '
                     f'{values[name]["adhesion_energy_jm2"]:.6f} J/m^2')
    lines.append(f'corrugation (maximum - minimum): {values["corrugation_ev"]:.6f} eV, '
                 f'{values["corrugation_jm2"]:.6f} J/m^2')
    return lines


# --- step -----------------------------------------------------------------------------

def _load_run(run_dir):
    # Imported here: the command line imports the registry of the steps.
    from triboworkflow.cli.run import load_run

    return load_run(run_dir)


class AdhesionStep(Step):
    """Step ``adhesion``: four calculations and the adhesion energy of the interface."""

    name = STEP

    def done(self, run_dir, system):
        """Whether its entries (``adhesion_leaves``) are written and current (``provenance``)."""
        sdir = system_dir(run_dir, system)
        document = read_system(sdir)
        entries = adhesion_leaves(_load_run(run_dir).interface.adhesion.type)
        # Before any analysis the inputs are not read: the PES may not have run.
        if not all(document.get(entry) for entry in entries):
            return False
        try:
            expected = self.provenance(run_dir, system)
        except PrerequisiteError:
            return False
        return all(is_done(document, entry, expected, sdir) for entry in entries)

    def analyse(self, run_dir, system, calculations):
        """Read the four calculations, write the adhesion energy and the entries with their provenance.

        Raises
        ------
        PrerequisiteError
            If an entry of ``match`` or of the PES is missing or not current.
        ValueError
            If ``calculations`` are not those that the settings give now.
        StoreError
            If a record is missing, not complete, of another fingerprint, or
            the records do not share one energy model, or a calculation gave
            no energy. Nothing is saved.
        """
        sdir = system_dir(run_dir, system)
        with SystemWriter(sdir) as writer:
            check_prerequisites(writer.document, self.expected(run_dir, system), sdir)
            inputs = self.inputs(run_dir, system)
            plan = {calc.calc_id: calc.record.fingerprint
                    for calc in adhesion_calculations(system, inputs)}
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
            provenance = adhesion_provenance(inputs, plan).with_results(records)
            check_energy_model(records)
            adhesion = run_adhesion_analysis(inputs, LegacyStore(sdir, writer.document, COLLECTION),
                                             list(plan))
            for entry in adhesion_leaves(inputs.type):
                writer.document.set_provenance(entry, provenance)
            sites = tuple(writer.document.get((inputs.pes, 'energy', name))
                          for name in ('min_site', 'max_site'))
            writer.save()
        for line in describe(inputs, adhesion, sites):
            logger.info('%s: %s', system, line)

    def inputs(self, run_dir, system):
        """``adhesion_inputs`` of the step, from the settings of the run."""
        inputs = adhesion_inputs(run_dir, system, _load_run(run_dir))
        if inputs.printed:
            logger.debug('%s/%s: output of the vendored code:\n%s', system, self.name,
                         inputs.printed)
        return inputs

    def calculations(self, run_dir, system):
        """The four calculations, nothing written."""
        return adhesion_calculations(system, self.inputs(run_dir, system))

    def expected(self, run_dir, system):
        """The entries read, with the provenance their step gives them now."""
        return adhesion_expected(run_dir, system, _load_run(run_dir))

    def provenance(self, run_dir, system):
        """Provenance expected now for the entries that the step writes."""
        inputs = self.inputs(run_dir, system)
        calcs = {calc.calc_id: calc.record.fingerprint
                 for calc in adhesion_calculations(system, inputs)}
        return adhesion_provenance(inputs, calcs)
