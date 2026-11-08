# Author: Vito Foderà
# Created: 2026-11-07

"""Steps ``pes`` and ``pes_scf`` of the interface workflow: potential energy surfaces.

The potential energy surface (PES) of an interface is its energy as a
function of the lateral position of the top slab on the bottom one. It is
sampled at the high-symmetry points of the interface written by ``match``
(``highsym.inter_unique``): one pw.x calculation per point, with the top slab
shifted laterally by the point and placed at ``interface_distance`` (of
``[interface.inter_params]``) from the bottom one. Two steps compute it, each
run when listed in ``[interface].steps``, also both, neither by default:

- ``pes``: the PES relaxed along z. Each point is a ``relax`` in which the
  atoms move only along z, the hydrogen atoms in every direction (``calc_type``
  ``interface_z_relax``). Calculations ``pes/<point>``, entry ``pes`` of the
  document;
- ``pes_scf``: the static PES, one ``scf`` per point (``calc_type``
  ``slab_from_scratch``). Calculations ``pes_scf/<point>``, entry ``pes_scf``.
  With ``short = true`` in ``[interface.pes_scf]`` the two slabs are first
  reduced to their layers next to the interface: ``layers`` of the bottom
  and of the top slab, by default ``SHORT_LAYERS_111`` for a (111) slab and
  ``SHORT_LAYERS_OTHER`` for the others.

The inputs (``pes_inputs``) come from the entries of ``match`` in the
document of the system (``structure.init.interface``, ``.bot_slab``,
``.top_slab``, ``highsym``): ``check_prerequisites`` stops the step when they
are missing or not current. The structure, the constraints and the k-point
grid of each point are those of the vendored workflow code
(``generate_pes_inputs``, which stacks the slabs with ``apply_interface_shifts``
and sets up the calculation with ``VaspInputSet``), with the grid from
``[k_points].kdens``; the pw.x input is written from them
(``preparation.workflow_calculation``).

The analysis (``PesStep.analyse``) uses the records of the calculations only
when each one is complete with the fingerprint of the settings and all share
one energy model. It writes, in the entry ``<name>`` of the document:

- ``data.<point>``: the results of the calculation of each point
  (``move_results``): ``output.<key>`` for each key of ``TRANSFER_FROM``
  (None when the calculation does not give it), ``output.structure`` and
  ``input.structure``;
- ``energy`` (``arrange_data``): ``grid.hs_unique_ev``, the list
  ``[point, x, y, energy]`` of the points sorted by ``output.energy`` (eV),
  with ``x, y`` the first lateral shift of the point; ``grid.hs_unique_jm2``,
  the same with the energies in J/m^2 on the area of the cell of the
  minimum; ``min_site`` and ``max_site``, the points of lowest and highest
  energy;

and the final structures of the two points in ``structure.opt.<name>.min``
and ``.max``. Before these writes the three entries, when present, are
emptied: the writes merge the new data into the old ones key by key, and the
points of an earlier interface would stay in ``data``. The entries of the
other PES and those of ``match`` are not touched. Each entry written
(``pes_leaves``: ``data``, the two grids, ``min_site``, ``max_site`` and the
two structures) gets the provenance of the step: the parameters, the
calculations with their results and the entries of ``match`` read. The
provenance stays on these entries, not on the whole ``<name>``, so that the
interpolation of the PES, written into ``<name>.energy`` by ``plotpes`` and
``plotpes_scf``, leaves them current; emptying ``<name>`` again removes the
interpolation too. Nothing is saved when a part fails, the emptying included.

Notes
-----
To remove: port of TribChem's ``FT_StartPES`` (``highput/firetasks/run.py``
:1178-1242, with ``FireTaskInterface.query_interface``, ``core.py:486-556``)
and of ``InterfaceWF.calculate_pes`` (``highput/workflows/interface.py``
:125-217). The two steps are the two values of ``-pt`` of TribChem's command
line (``set_pes``, ``core/cli/calc_interface_cli.py:172-199``), which ran one
PES per invocation, ``pes`` by default (DEV-37). Differences: ``comp_params``
is ``{'kdens': [k_points].kdens}`` (DEV-35); the layers of the reduced slabs
are those of the bottom and of the top slab, and the default rule takes the
Miller indices of the slab below and of the one above, where TribChem's
command line took them in the order of the slabs given (DEV-38, A81); the
writes use the ids of the document written by ``match``, where TribChem made
them again from the bottom and the top slab, which changes the formula when
the alphabetical order swaps the slabs (A80). Not ported: ``override``,
``run_vasp``, ``to_file``, ``get_main_folder`` (A47) and
``FT_SaveVASPOutputs`` (the files of each calculation stay in its directory).
The analysis ports ``FT_MoveTagResults`` (``highput/firetasks/dbtask.py``
:166-374) with the entries of ``GeneralTribchemWF._select_entry('pes')`` and
``InterfaceWF.pes_entry_name`` (``highput/workflows/wfcore.py:30-97``,
``interface.py:470-498``), then ``FT_PotentialEnergySurface.arrange_data`` and
``_update_opt_structure`` (``highput/firetasks/pes.py:57-138``); the
interpolation, commented out there (``pes.py:52-53``), is not part of it.
Not ported: the ``check_entry`` of ``FT_MoveTagResults``, null in the PES (the
provenance decides when the step runs again). Deviation: DEV-40 (the three
entries emptied before being written again, as DEV-36 for ``match``).
"""

import contextlib
import io
import logging
import warnings
from dataclasses import dataclass, field
from operator import itemgetter

from pymatgen.io.vasp.sets import BadInputSetWarning

from triboworkflow.core.settings import SettingsError
from triboworkflow.core.settings.schema import SHORT_LAYERS_111, SHORT_LAYERS_OTHER
from triboworkflow.core.store import (PrerequisiteError, Provenance, StoreError, SystemWriter,
                                      check_energy_model, check_prerequisites, is_done,
                                      read_record, read_system)
from triboworkflow.core.store.legacy import TASKS, LegacyStore
from triboworkflow.execution.layout import system_dir
from triboworkflow.execution.orchestrator import Step
from triboworkflow.execution.preparation import workflow_calculation
from triboworkflow.vendored.highput.database.dbtools import read_one_entry
from triboworkflow.vendored.highput.utils.manipulate_struct import (ManipulateStruct,
                                                                     generate_pes_inputs)
from triboworkflow.vendored.highput.utils.tasktools import read_default_params, select_struct_func
from triboworkflow.vendored.highput.utils.vasp_tools import VaspInputSet
from triboworkflow.vendored.physics.base.solidstate import get_cell_info
from triboworkflow.vendored.physics.base.units import Converter

from .chain import upstream_done
from .match import (COLLECTION, DEFAULTS, FILTER_KEYS, clear_entries, match_parameters,
                    match_provenance, set_filter)

logger = logging.getLogger(__name__)

# The two steps and their calculations.
# To remove: set_pes of TribChem's command line (calc_interface_cli.py:191-197).
CALC_TYPES = {'pes': 'interface_z_relax', 'pes_scf': 'slab_from_scratch'}
# Output of the calculations that the PES is made of: the total energy.
# To remove: 'pes_type' of FT_StartPES and FT_PotentialEnergySurface, whose
# only value in TribChem's workflows is 'energy' (defaults.json).
PES_TYPE = 'energy'
# Keys of the document that identify the interface.
ID_KEYS = ('mid', 'formula', 'name', 'miller')
# Results copied from the task document of each calculation, each to the key
# of its last part below <name>.data.<point>.output; then the final and the
# input structures. The keys that only VASP gave are copied as None.
# To remove: transfer_data_from and transfer_data_to of GeneralTribchemWF
# (wfcore.py:30-58), whose last parts are the same.
TRANSFER_FROM = (('nsites',), ('output', 'density'), ('output', 'energy'),
                 ('output', 'energy_per_atom'), ('output', 'bandgap'), ('output', 'cbm'),
                 ('output', 'vbm'), ('output', 'is_gap_direct'), ('output', 'forces'),
                 ('output', 'stress'), ('input', 'pseudo_potential'), ('dir_name',),
                 ('task_id',), ('task_label',))


@dataclass
class PesInputs:
    """What a PES step reads from the settings and from the entries of ``match``.

    Attributes
    ----------
    name : str
        ``pes`` or ``pes_scf``: the step, the entry of the document and the
        first part of the ``calc_id`` of its calculations.
    calc_type : str
    settings : triboworkflow.core.settings.Settings
    ids : dict
        ``mid``, ``formula``, ``name`` and ``miller`` of the document, which
        select it in the writes.
    read : dict
        The entries of ``match`` read, as stored: ``interface``,
        ``bot_slab``, ``top_slab`` and ``shifts`` (``highsym.inter_unique``).
    interface, bot_slab, top_slab
        The interface and the two slabs used, reduced with ``short``.
    shifts : dict
        ``{point: [[x, y], ...]}``, the lateral shifts of the top slab.
    parameters : dict
        Parameters of the step (``calc_type``, ``pes_type``, ``comp_params``,
        ``interface_distance``, ``layers``), recorded in the provenance.
    points : list of tuple
        ``(point, structure, kpoints)`` of each calculation, in the order of
        ``shifts``: the structure with the ``selective_dynamics`` of the
        ``calc_type`` and the k-points of ``VaspInputSet``.
    printed : str
        What the vendored code printed.
    """

    name: str
    calc_type: str
    settings: object
    ids: dict
    read: dict
    interface: object
    bot_slab: object
    top_slab: object
    shifts: dict
    parameters: dict
    points: list = field(default_factory=list)
    printed: str = ''


# --- inputs ---------------------------------------------------------------------------

def start_defaults():
    """Defaults of the start of the PES: entries of ``match`` read, ``layers``, ...

    Returns
    -------
    dict
    """
    return read_default_params(str(DEFAULTS), 'StartPES', {})


def match_entries_read(start=None):
    """The entries of ``match`` that the PES steps read.

    Parameters
    ----------
    start : dict, optional
        As ``start_defaults``, read when not given.

    Returns
    -------
    list of tuple of str
        The interface, the two slabs and ``highsym``, whose ``inter_unique``
        gives the points.
    """
    start = start_defaults() if start is None else start
    return [tuple(start['inter_entry']), tuple(start['bot_slab_entry']),
            tuple(start['top_slab_entry']), tuple(start['hs_entry'][:1])]


def short_layers(bot_slab, top_slab, given=None):
    """Layers kept in the bottom and in the top slab of a reduced static PES.

    Parameters
    ----------
    bot_slab, top_slab : pymatgen.core.surface.Slab
        The slabs of the interface, as stacked by ``match``.
    given : list of int, optional
        ``layers`` of ``[interface.pes_scf]``.

    Returns
    -------
    list of int
        ``given``; otherwise, for each slab, ``SHORT_LAYERS_111`` if its
        Miller indices are (1, 1, 1), else ``SHORT_LAYERS_OTHER``.

    Notes
    -----
    To remove: TribChem's command line applied the rule to the slabs in the
    order given (``calc_interface.py:151``), and ``shorten_interface_slabs``
    used the values for the bottom and the top slab (A81, DEV-38).
    """
    if given is not None:
        return list(given)
    layers = []
    for slab in (bot_slab, top_slab):
        miller = getattr(slab, 'miller_index', None)
        if miller is None:
            raise PrerequisiteError('the slabs of the interface have no Miller indices: give '
                                    'layers in [interface.pes_scf]')
        layers.append(SHORT_LAYERS_111 if tuple(miller) == (1, 1, 1) else SHORT_LAYERS_OTHER)
    return layers


def _structure(data):
    return select_struct_func(data['@class']).from_dict(data)


def pes_inputs(run_dir, system, settings, name):
    """Everything a PES step reads, and the structure and k-points of each point.

    Parameters
    ----------
    run_dir : path-like
        Directory of the run.
    system : str
    settings : triboworkflow.core.settings.Settings
        With an ``[interface]`` table.
    name : str
        ``pes`` or ``pes_scf``.

    Returns
    -------
    PesInputs

    Raises
    ------
    SettingsError
        If the k-points are not given by their density (``[k_points]
        mode = "kdens"``): the cell of the interface is known only after the
        matching.
    PrerequisiteError
        If the document of the system or an entry of ``match`` is missing.
    StoreError
        If the document is of another interface.

    Notes
    -----
    To remove: ``FT_StartPES.query_interface`` (``run.py:1220-1242``, with
    ``FireTaskInterface.query_interface``, ``core.py:493-556``) and
    ``InterfaceWF.calculate_pes`` up to ``generate_pes_inputs``
    (``interface.py:165-177``). The document is found with ``mid`` and
    ``miller``, as there (``core.py:500-503``); ``comp_params`` comes from
    the settings instead of the document (DEV-35).
    """
    calc_type = CALC_TYPES[name]
    if settings.k_points.mode != 'kdens':
        raise SettingsError('k_points.mode', f'the step {name} takes the k-point grid from its '
                            'density (mode = "kdens"): the cell of the interface is known only '
                            'after the matching', settings.source)
    params, inter_params, _ = match_parameters(settings)
    start = start_defaults()
    sdir = system_dir(run_dir, system)
    store = LegacyStore(sdir, read_system(sdir), COLLECTION)
    fltr = set_filter({'mid': '_'.join(params['mid']), 'miller': params['miller']})
    field_, _ = store.query_db(COLLECTION, fltr=fltr)
    if field_ is None:
        raise PrerequisiteError(f'{system}: the system has no document yet: run match first')
    entries = {'interface': start['inter_entry'], 'bot_slab': start['bot_slab_entry'],
               'top_slab': start['top_slab_entry'], 'shifts': start['hs_entry']}
    read = {key: read_one_entry(field_, entry) for key, entry in entries.items()}
    missing = ['.'.join(entries[key]) for key, value in read.items() if not value]
    if missing:
        raise PrerequisiteError(f'{system}: entries missing: {", ".join(missing)}; run match '
                                'first')
    ids = set_filter({key: field_.get(key) for key in ID_KEYS}, FILTER_KEYS)

    interface = _structure(read['interface'])
    bot_slab, top_slab = _structure(read['bot_slab']), _structure(read['top_slab'])
    layers = None
    if name == 'pes_scf' and settings.interface.pes_scf.short:
        layers = short_layers(bot_slab, top_slab, settings.interface.pes_scf.layers)
    comp_params = {'kdens': settings.k_points.kdens}
    interface_distance = inter_params['interface_distance']

    # The vendored code prints, and pymatgen warns about the POTCAR functional
    # of the VASP input set, which pw.x does not use.
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed), warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Overriding the POTCAR functional', BadInputSetWarning)
        if calc_type in VaspInputSet.static_types and layers is not None:
            interface, bot_slab, top_slab = ManipulateStruct.shorten_interface_slabs(
                interface, layers[0], layers[1])
        structures, vis, _ = generate_pes_inputs(bot_slab, top_slab, read['shifts'], comp_params,
                                                 calc_type=calc_type,
                                                 interface_distance=interface_distance)
    points = [(point, structure, v.kpoints)
              for point, structure, v in zip(read['shifts'], structures, vis)]
    parameters = {'calc_type': calc_type, 'pes_type': PES_TYPE, 'comp_params': comp_params,
                  'interface_distance': interface_distance, 'layers': layers}
    return PesInputs(name=name, calc_type=calc_type, settings=settings, ids=ids, read=read,
                     interface=interface, bot_slab=bot_slab, top_slab=top_slab,
                     shifts=read['shifts'], parameters=parameters, points=points,
                     printed=printed.getvalue())


def pes_calculations(system, inputs):
    """The pw.x calculations of the points, as ``preparation.workflow_calculation`` makes them.

    Returns
    -------
    list of QECalculation
        ``<name>/<point>`` in the order of the points.
    """
    return [workflow_calculation(system, f'{inputs.name}/{point}', inputs.calc_type, structure,
                                 kpoints, inputs.settings)
            for point, structure, kpoints in inputs.points]


def pes_provenance(inputs, calculations):
    """Provenance that the entries of a PES step have when computed now.

    Parameters
    ----------
    inputs : PesInputs
    calculations : dict
        ``{calc_id: fingerprint}`` of the calculations of the points.

    Returns
    -------
    Provenance
        The parameters of the step, the calculations and the content of the
        entries of ``match`` read.
    """
    return Provenance.build(inputs.name, parameters=inputs.parameters, calculations=calculations,
                            inputs=inputs.read)


# --- analysis -----------------------------------------------------------------------------

def pes_entry(calc_type):
    """Entry of the document of a PES of ``calc_type``: ``pes_scf`` if static, else ``pes``.

    Notes
    -----
    To remove: as ``InterfaceWF.pes_entry_name`` (``interface.py:470-498``)
    and ``FT_PotentialEnergySurface.arrange_data`` (``pes.py:74``) choose it.
    """
    return 'pes_scf' if calc_type in VaspInputSet.static_types else 'pes'


def pes_entries(name):
    """The entries that a PES step empties before writing: ``<name>`` and its two structures.

    Returns
    -------
    list of tuple of str
        ``<name>`` and the structures of its minimum and maximum.
    """
    return [(name,), ('structure', 'opt', name, 'min'), ('structure', 'opt', name, 'max')]


def pes_leaves(name):
    """The entries that a PES step writes, each with the provenance of the step.

    Returns
    -------
    list of tuple of str
        ``<name>.data``, ``<name>.energy.grid.hs_unique_ev`` and
        ``.hs_unique_jm2``, ``<name>.energy.min_site`` and ``.max_site``, and
        the structures of the minimum and of the maximum.

    Notes
    -----
    The interpolation of the PES writes other entries of ``<name>.energy``:
    a provenance on the whole ``<name>`` would be dropped by those writes.
    """
    energy = (name, PES_TYPE)
    return [(name, 'data'), energy + ('grid', 'hs_unique_ev'), energy + ('grid', 'hs_unique_jm2'),
            energy + ('min_site',), energy + ('max_site',),
            ('structure', 'opt', name, 'min'), ('structure', 'opt', name, 'max')]


def transfer_entries(calc_type, point):
    """Keys read in the task document of a point and entries where they are written.

    Returns
    -------
    entry_from, entry_to : list of list of str
        One entry per key, in the same order: those of ``TRANSFER_FROM``,
        then the final and the input structures.

    Notes
    -----
    To remove: ``GeneralTribchemWF._select_entry('pes')`` (``wfcore.py
    :86-97``) for one point, with the entry of ``pes_entry_name``.
    """
    name = pes_entry(calc_type)
    entry_from = [list(path) for path in TRANSFER_FROM]
    entry_to = [[name, 'data', str(point), 'output', path[-1]] for path in TRANSFER_FROM]
    entry_from += [['output', 'structure'], ['input', 'structure']]
    entry_to += [[name, 'data', str(point), 'output', 'structure'],
                 [name, 'data', str(point), 'input', 'structure']]
    return entry_from, entry_to


def move_results(store, ids, calc_type, point, calc_id):
    """Copy the results of the calculation of a point into the document.

    Parameters
    ----------
    store : LegacyStore
        On the document of the system.
    ids : dict
        Ids of the document (``PesInputs.ids``).
    calc_type : str
    point : str
        High-symmetry point of the calculation.
    calc_id : str
        ``<name>/<point>``, the ``task_label`` of its task document.

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
    entry_from, entry_to = transfer_entries(calc_type, point)
    field_, field_entry = store.query_db(TASKS, entry=entry_from,
                                         fltr={defaults['tag_key']: calc_id})
    if field_ is None:
        raise StoreError(f'{calc_id}: no record of the calculation')
    store.update_db(field_entry, COLLECTION, entry=entry_to, fltr=set_filter(ids),
                    update_op=defaults['update_op'], to_mongodb=defaults['to_mongodb'])


def arrange_data(store, ids, calc_type, pes_type=PES_TYPE):
    """Sort the points by energy and write the grid, the minimum and the maximum of the PES.

    Parameters
    ----------
    store : LegacyStore
        On the document of the system, with the results of every point
        (``move_results``).
    ids : dict
        Ids of the document.
    calc_type : str
    pes_type : str, optional
        Output of the calculations that the PES is made of.

    Returns
    -------
    all_shifts : dict
        ``highsym.inter_all``.
    values : list
        ``[hs_unique_ev, hs_unique_jm2]``: the lists ``[point, x, y, value]``
        sorted by the value, in eV and in J/m^2.
    cell : numpy.ndarray
        Lattice matrix of the final structure of the minimum.

    Notes
    -----
    To remove: ``FT_PotentialEnergySurface.arrange_data`` (``pes.py:57-118``)
    with ``p['collection']`` the collection of the interfaces.
    """
    # Retrieve the document with all the data
    fltr = set_filter(ids)
    field_, _ = store.query_db(COLLECTION, fltr=fltr)

    # Recover the shift names and values
    shifts = read_one_entry(field_, ['highsym', 'inter_unique'])
    all_shifts = read_one_entry(field_, ['highsym', 'inter_all'])

    # Set the name for the pes based on calc_type
    pes_calc = pes_entry(calc_type)

    # Collect all values associated to shifts and specified by pes_type
    values = []
    for s, d in shifts.items():
        x, y = d[0]
        v = read_one_entry(field_, [pes_calc, 'data', str(s), 'output', pes_type])
        values.append([s, x, y, v])
    sorted_values = sorted(values, key=itemgetter(3))

    # Find out the minimum and maximum stacking for values
    min_stack = sorted_values[0][0]
    max_stack = sorted_values[-1][0]
    struct_min = read_one_entry(field_, [pes_calc, 'data', str(min_stack), 'output', 'structure'])

    # Prepare the grid energy values to be saved in J/m^2
    area, _ = get_cell_info(struct_min['lattice']['matrix'])
    sorted_values_jm2 = [[s, x, y, Converter.eV_to_Jm2(v, area * 1e-20)]
                         for s, x, y, v in sorted_values]

    # Prepare the data to be stored in the Database
    data = {
        pes_type: {
            'grid': {
                'hs_unique_ev': sorted_values,
                'hs_unique_jm2': sorted_values_jm2
            },
            'max_site': str(max_stack),
            'min_site': str(min_stack)
        }
    }

    # Store information associated to pes_type for min and max stackings
    store.update_db(data, COLLECTION, entry=pes_calc, fltr=fltr)

    # Store the minimum energy structure in the high layer of the field
    update_opt_structure(store, min_stack, max_stack, field_, pes_calc, fltr, pes_type)

    # Take out the cell of the lattice matrix
    func = select_struct_func(struct_min['@class'])
    cell = func.from_dict(struct_min).lattice.matrix

    return all_shifts, [sorted_values, sorted_values_jm2], cell


def update_opt_structure(store, min_stack, max_stack, field_, pes_calc, fltr, pes_type):
    """Write the final structures of the minimum and of the maximum, for an energy PES.

    Notes
    -----
    To remove: ``FT_PotentialEnergySurface._update_opt_structure``
    (``pes.py:120-138``).
    """
    # Make the update only in the case of energy pes_type
    if pes_type == 'energy':

        # Insert both the minimum and the maximum energy stack
        for s, n in zip([min_stack, max_stack], ['min', 'max']):
            data = read_one_entry(field_, [pes_calc, 'data', str(s), 'output', 'structure'])
            entry = ['structure', 'opt', pes_calc, n]

            # Make the update
            store.update_db(data, COLLECTION, entry=entry, fltr=fltr)


def run_pes_analysis(inputs, store, calc_ids):
    """Copy the results of every point and arrange the PES, without the provenance.

    Parameters
    ----------
    inputs : PesInputs
    store : LegacyStore
        On the document of the system, which is changed and not saved.
    calc_ids : list of str
        ``calc_id`` of the calculations, in the order of the points.

    Returns
    -------
    list
        ``[hs_unique_ev, hs_unique_jm2]`` of ``arrange_data``.
    """
    for point, calc_id in zip(inputs.shifts, calc_ids):
        move_results(store, inputs.ids, inputs.calc_type, point, calc_id)
    _, values, _ = arrange_data(store, inputs.ids, inputs.calc_type)
    return values


def describe(inputs, values):
    """Lines of the message of the step: points, minimum, maximum and corrugation.

    Parameters
    ----------
    inputs : PesInputs
    values : list
        ``[hs_unique_ev, hs_unique_jm2]``.

    Returns
    -------
    list of str
    """
    ev, jm2 = values
    if inputs.name == 'pes':
        kind = 'relaxed along z'
    elif inputs.parameters['layers'] is None:
        kind = 'static, on the complete slabs'
    else:
        bottom, top = inputs.parameters['layers']
        kind = f'static, on the slabs reduced to {bottom} (bottom) and {top} (top) layers'
    (low, *_, e_low), (high, *_, e_high) = ev[0], ev[-1]
    return [
        f'{inputs.name}: {len(ev)} high-symmetry points, {kind} ({inputs.calc_type})',
        f'minimum at {low}: {e_low:.6f} eV; maximum at {high}: {e_high:.6f} eV',
        f'corrugation (maximum - minimum): {e_high - e_low:.6f} eV, '
        f'{jm2[-1][3] - jm2[0][3]:.6f} J/m^2',
    ]


# --- steps --------------------------------------------------------------------------------

def _load_run(run_dir):
    # Imported here: the command line imports the registry of the steps.
    from triboworkflow.cli.run import load_run

    return load_run(run_dir)


class PesStep(Step):
    """A PES step: one calculation per high-symmetry point of the interface."""

    name = None

    def done(self, run_dir, system):
        """Whether its entries (``pes_leaves``) are written and current (``provenance``).

        ``match``, which it reads, has to be done too: the provenance holds the
        content of the entries read, not whether they are still current.
        """
        sdir = system_dir(run_dir, system)
        document = read_system(sdir)
        entries = pes_leaves(self.name)
        # Before any analysis the inputs are not read: match may not have run.
        if not all(document.get(entry) for entry in entries):
            return False
        if not upstream_done(self, run_dir, system):
            return False
        expected = self.provenance(run_dir, system)
        return all(is_done(document, entry, expected, sdir) for entry in entries)

    def analyse(self, run_dir, system, calculations):
        """Read the results of the points, write the entries of the PES with their provenance.

        Raises
        ------
        PrerequisiteError
            If an entry of ``match`` is missing or not current.
        ValueError
            If ``calculations`` are not those that the settings give now.
        StoreError
            If a record is missing, not complete, of another fingerprint, or
            the records do not share one energy model.
        """
        sdir = system_dir(run_dir, system)
        with SystemWriter(sdir) as writer:
            check_prerequisites(writer.document, self.expected(run_dir, system), sdir)
            inputs = self.inputs(run_dir, system)
            plan = {calc.calc_id: calc.record.fingerprint
                    for calc in pes_calculations(system, inputs)}
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
            provenance = pes_provenance(inputs, plan).with_results(records)
            check_energy_model(records)
            # Nothing of an earlier analysis of this PES is left (DEV-40).
            clear_entries(writer.document, pes_entries(self.name))
            values = run_pes_analysis(inputs, LegacyStore(sdir, writer.document, COLLECTION),
                                      list(plan))
            for entry in pes_leaves(self.name):
                writer.document.set_provenance(entry, provenance)
            writer.save()
        for line in describe(inputs, values):
            logger.info('%s: %s', system, line)

    def inputs(self, run_dir, system):
        """``pes_inputs`` of the step, from the settings of the run."""
        inputs = pes_inputs(run_dir, system, _load_run(run_dir), self.name)
        if inputs.printed:
            logger.debug('%s/%s: output of the vendored code:\n%s', system, self.name,
                         inputs.printed)
        return inputs

    def calculations(self, run_dir, system):
        """One calculation per point of ``highsym.inter_unique``, nothing written."""
        return pes_calculations(system, self.inputs(run_dir, system))

    def expected(self, run_dir, system):
        """The entries of ``match`` read, with the provenance ``match`` gives them now."""
        provenance = match_provenance(_load_run(run_dir))
        return {entry: provenance for entry in match_entries_read()}

    def provenance(self, run_dir, system):
        """Provenance expected now for the entries that the step writes."""
        inputs = self.inputs(run_dir, system)
        calcs = {calc.calc_id: calc.record.fingerprint
                 for calc in pes_calculations(system, inputs)}
        return pes_provenance(inputs, calcs)


class RelaxedPesStep(PesStep):
    """Step ``pes``: the PES relaxed along z."""

    name = 'pes'


class StaticPesStep(PesStep):
    """Step ``pes_scf``: the static PES, on complete or reduced slabs."""

    name = 'pes_scf'
