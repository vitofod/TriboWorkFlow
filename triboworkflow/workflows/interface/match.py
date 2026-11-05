# Author: Vito Foderà
# Created: 2026-11-05

"""Step ``match`` of the interface workflow: the interface of two slabs.

The step runs no calculation. From the two slabs of ``[interface]`` it builds
the interface cell and the high-symmetry points of the two facing surfaces,
and writes them in the document of the system:

- ``structure.init.interface``: the interface, with vacuum around it;
- ``structure.init.bot_slab``, ``structure.init.top_slab``: the two slabs
  strained to the common cell and placed as in the interface;
- ``inter_params``: the matching parameters used, after the enlargements;
- ``highsym``: the high-symmetry points of each surface (``bot_unique``,
  ``bot_all``, ``top_unique``, ``top_all``) and the lateral shifts of the
  interface (``inter_unique``, ``inter_all``), which the PES step uses.

The work goes in this order:

1. inputs (``match_inputs``): the slabs (``read_workflow_slab``); the
   matching parameters: the keys of ``[interface.inter_params]``, then the
   defaults of the workflow (``INTER_PARAMS_DEFAULTS``: ``max_area`` 160.0,
   ``interface_distance`` 2.0, ``max_angle_tol`` 0.4), then those of the
   matching; with ``set_bm`` the strain weights are the bulk moduli of the
   two slabs; the ids of the interface: ``mid`` is ``mid1_mid2``, ``formula``
   and ``name`` join those of the settings, or are made from the slabs (e.g.
   ``Cu_Al`` and ``Cu111-Al111``);
2. the document of the system is created when missing, with the ids and an
   empty ``structure``; an existing document is kept, and must have the same
   ids;
3. the five entries of the step already in the document are emptied, so that
   nothing of an earlier interface is left among the new data;
4. matching (``generate_interface``): the slab whose formula and Miller
   indices come first alphabetically goes on top. When no match is found,
   ``max_area``, ``max_length_tol`` and ``max_area_ratio_tol`` are multiplied
   by 1.01, then by 1.02, and so on, the products accumulating, until a match
   is found or the factor reaches 1.5, which is an error; ``max_angle_tol``
   stays as given. Vacuum is set around the interface; the interface, the
   aligned slabs and the final parameters are written;
5. high-symmetry points: the matching is done again from the parameters of
   step 1, without vacuum, and ``get_hspts`` computes the points of the
   aligned slabs; they are written;
6. ``MatchStep.analyse`` writes ``match/interface.extxyz`` and
   ``match/interface.json`` in the folder of the system, records the
   provenance of the five entries and saves the document. Nothing is saved
   when a part fails. The message of the step gives the slab on top, the
   attempts, the parameters used and the strain of the two slabs; what the
   matching code prints goes to the log at debug level.

Notes
-----
To remove: port of TribChem's ``FT_StartInterfaceMatch`` (inputs and
document, ``highput/firetasks/run.py:985-1176``), ``FT_GenerateInterface``
(``highput/firetasks/slabs.py:347-443``) and ``FT_GetHighSymmetryPts``
(``slabs.py:485-502``, ``559-679``), chained as ``InterfaceWF.match_interface``
does (``highput/workflows/interface.py:34-123``), each Firework with its own
copy of the parameters. The database calls go through ``core/store/legacy.py``
with the arguments of the original calls. Deviations: DEV-01
(``max_angle_tol`` of the user at every attempt, instead of 0.4 forced),
DEV-33 (the recursion of ``FT_GenerateInterface`` works: its debug prints
raised ``UnboundLocalError`` at the first failed attempt, A73), DEV-34
(``best_match`` only ``'area'``, checked in the settings), DEV-35 (no
``comp_params`` in the document), DEV-36 (entries emptied before being
written again). Not ported: ``override`` and ``override_inter_db`` (the
provenance decides when the step runs again, the document is created only
when missing as by default in TribChem, A45); the ``is_done`` check on
``structure.init.interface`` alone (``MatchStep.done`` checks the five
entries); the prints. ``decompose_structure`` is called as in TribChem and its
result is not used.
"""

import contextlib
import copy
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

from monty.json import jsanitize

import triboworkflow.vendored.highput as highput
from triboworkflow.core.store import Provenance, SystemWriter, is_done, read_system
from triboworkflow.core.store.atomic import write_json
from triboworkflow.core.store.legacy import LegacyStore
from triboworkflow.core.structures import write_structure
from triboworkflow.core.structures.inputs import load_workflow_slabs, structure_path
from triboworkflow.execution.layout import system_dir
from triboworkflow.execution.orchestrator import Step
from triboworkflow.vendored.highput.utils.errors import InterfaceMatchError
from triboworkflow.vendored.highput.utils.manipulate_struct import (ManipulateStruct,
                                                                     generate_interface_ids,
                                                                     slab_to_structure)
from triboworkflow.vendored.highput.utils.tasktools import get_miller_str, read_default_params
from triboworkflow.vendored.physics.tribology.highsym import (fix_hs_dicts, get_interface_hs,
                                                               get_slab_hs, pbc_hs_points)
from triboworkflow.vendored.physics.tribology.interface_matcher import (InterfaceMatcher,
                                                                         flip_slab)

logger = logging.getLogger(__name__)

STEP = 'match'
# Defaults of the matching code: matching parameters ('inter_params') and
# start of the step ('StartInterfaceMatch': entries, vacuum, update options).
DEFAULTS = Path(highput.__file__).parent / 'firetasks' / 'defaults.json'
# Collection of the interface documents in the historic database calls.
# To remove: default of TribChem's command line (-ci,
# entrypoints/parsers/utils.py:673-674).
COLLECTION = 'PBE.interface_elements'
# Keys of the filter that selects the document of the interface.
FILTER_KEYS = ('mid', 'formula', 'name', 'miller', 'adsorbate')
# Files of the step in the folder of the system.
OUTPUT_DIR = 'match'
INTERFACE_FILE = 'interface.extxyz'
INTERFACE_JSON = 'interface.json'


@dataclass
class InterfaceMatch:
    """A match of two slabs.

    Attributes
    ----------
    interface : pymatgen.core.structure.Structure
        The interface; with vacuum around it when asked.
    bot_slab, top_slab : pymatgen.core.surface.Slab
        The aligned slabs, placed as in the interface.
    matcher : InterfaceMatcher
        The matcher of the attempt that found the match.
    attempts : int
        Attempts made, the last one included.
    """

    interface: object
    bot_slab: object
    top_slab: object
    matcher: object
    attempts: int


@dataclass
class MatchInputs:
    """What the step reads from the settings.

    Attributes
    ----------
    slabs : list of WorkflowSlab
        The two slabs, in the order of ``[[interface.slabs]]``.
    files : list of pathlib.Path
        Their files.
    params : dict
        ``mid``, ``miller``, ``formula`` and ``name`` of the two slabs, as
        lists; ``formula`` and ``name`` are None when not given.
    inter_params : dict
        Complete matching parameters, with the strain weights.
    set_bm : bool
        Whether the strain weights are the bulk moduli.
    start : dict
        Defaults of the start of the step: ``inter_entry``, ``hs_entry``,
        ``vacuum``, ``update_op``, ``to_mongodb``, ...
    ids : dict
        ``mid``, ``formula``, ``name`` and ``miller`` of the interface.
    """

    slabs: list
    files: list
    params: dict
    inter_params: dict
    set_bm: bool
    start: dict
    ids: dict = field(default_factory=dict)

    def provenance(self):
        """Provenance of the entries of the step (``match_provenance``)."""
        return _provenance(self.params, self.inter_params, self.set_bm, self.start, self.files)


@dataclass
class MatchResult:
    """What ``run_match`` computed and wrote.

    Attributes
    ----------
    match : InterfaceMatch
        The match of the interface, written in ``structure.init``.
    inter_params : dict
        The matching parameters of that match, written in ``inter_params``.
    hs_match : InterfaceMatch
        The match from which the high-symmetry points were computed.
    hs : dict
        The high-symmetry points, written in ``highsym``.
    """

    match: InterfaceMatch
    inter_params: dict
    hs_match: InterfaceMatch
    hs: dict


# --- inputs ---------------------------------------------------------------------------

def start_defaults():
    """Defaults of the start of the step: entries, vacuum and update options.

    Returns
    -------
    dict
    """
    return read_default_params(str(DEFAULTS), 'StartInterfaceMatch', {})


def match_entries(start=None):
    """The five entries that the step writes.

    Parameters
    ----------
    start : dict, optional
        As ``start_defaults``, read when not given.

    Returns
    -------
    list of tuple of str
    """
    start = start_defaults() if start is None else start
    inter = tuple(start['inter_entry'])
    return [inter + ('interface',), inter + ('bot_slab',), inter + ('top_slab',),
            ('inter_params',), tuple(start['hs_entry'])]


def _ids_of(slabs, key):
    # The values of one id of the two slabs, or None when no slab gives it.
    values = [getattr(slab, key) for slab in slabs]
    return None if all(v is None for v in values) else values


def match_parameters(settings):
    """Parameters of the matching from the settings, without reading the slabs.

    Parameters
    ----------
    settings : triboworkflow.core.settings.Settings
        With an ``[interface]`` table.

    Returns
    -------
    params : dict
        ``mid``, ``miller``, ``formula`` and ``name`` of the two slabs.
    inter_params : dict
        Complete matching parameters, in the order of the defaults of the
        matching, with the strain weights.
    start : dict
        As ``start_defaults``.

    Raises
    ------
    InterfaceMatchError
        If ``formula`` or ``name`` is given without the other.

    Notes
    -----
    To remove: ``FT_StartInterfaceMatch.run_task`` checks the inputs with
    ``InterfaceMatchError.check_inputs`` (``run.py:991``), then fills
    ``inter_params`` with ``read_default_params`` (``run.py:1085-1087``) and
    sets the strain weights (``set_bulk_modulus``, ``run.py:1098-1150``). The
    bulk moduli are those of ``[[interface.slabs]]`` instead of the database
    (``bm_custom`` not ported). ``check_inputs`` refuses formula without
    name, and the reverse, although ``generate_interface_ids`` could make the
    name from the formula (A75); the settings check it earlier, with the
    names of the fields.
    """
    interface = settings.interface
    params = {'mid': [slab.mid for slab in interface.slabs],
              'miller': [list(slab.miller) for slab in interface.slabs],
              'formula': _ids_of(interface.slabs, 'formula'),
              'name': _ids_of(interface.slabs, 'name')}
    InterfaceMatchError.check_inputs(params)
    inter_params = read_default_params(str(DEFAULTS), 'inter_params',
                                       interface.inter_params.requested())
    if interface.set_bm:
        inter_params['strain_weight_1'] = interface.slabs[0].bulk_modulus
        inter_params['strain_weight_2'] = interface.slabs[1].bulk_modulus
    return params, inter_params, start_defaults()


def slab_files(settings):
    """Paths of the two slab files."""
    return [structure_path(settings, slab.file, f'interface.slabs[{i}].file')
            for i, slab in enumerate(settings.interface.slabs)]


def _provenance(params, inter_params, set_bm, start, files):
    return Provenance.build(STEP, parameters={
        'mid': params['mid'], 'miller': params['miller'], 'formula': params['formula'],
        'name': params['name'], 'inter_params': inter_params, 'set_bm': set_bm,
        'vacuum': start['vacuum']},
        files={f'slab_{i + 1}': path for i, path in enumerate(files)})


def match_provenance(settings):
    """Provenance that the entries of the step have when computed now.

    The same for the five entries: the ids and the miller indices of the two
    slabs, the complete matching parameters, ``set_bm``, the vacuum and the
    content of the two slab files.

    Parameters
    ----------
    settings : triboworkflow.core.settings.Settings

    Returns
    -------
    Provenance
    """
    params, inter_params, start = match_parameters(settings)
    return _provenance(params, inter_params, settings.interface.set_bm, start,
                       slab_files(settings))


def match_inputs(settings):
    """Everything the step reads from the settings.

    Parameters
    ----------
    settings : triboworkflow.core.settings.Settings
        With an ``[interface]`` table.

    Returns
    -------
    MatchInputs

    Raises
    ------
    InterfaceMatchError
        As ``match_parameters``.
    StructureError
        If a slab cannot be read or used by the workflow.

    Notes
    -----
    To remove: ``FT_StartInterfaceMatch.query_structure``
    (``run.py:1014-1096``) without the database and without ``comp_params``
    (DEV-35). The ids are made by ``generate_interface_ids`` from the slabs
    in the order of the settings, as TribChem's slab 1 and slab 2.
    """
    params, inter_params, start = match_parameters(settings)
    slabs = load_workflow_slabs(settings)
    mid, formula, name, miller = generate_interface_ids(params, [s.slab for s in slabs])
    return MatchInputs(slabs=slabs, files=slab_files(settings), params=params,
                       inter_params=inter_params, set_bm=settings.interface.set_bm,
                       start=start,
                       ids={'mid': mid, 'formula': formula, 'name': name, 'miller': miller})


# --- matching ---------------------------------------------------------------------------

def generate_interface(slabs, inter_params, vacuum=None, factor=1, delta_f=0.01, attempt=1):
    """Match two slabs, enlarging the tolerances until a match is found.

    Parameters
    ----------
    slabs : list of pymatgen.core.surface.Slab
        The two slabs, as ``slab_1`` and ``slab_2`` of the matcher.
    inter_params : dict
        Complete matching parameters. Changed in place: at the end they hold
        the values of the attempt that found the match.
    vacuum : float, optional
        Vacuum around the interface, in angstrom; none is set when not given.
    factor : float, optional
        Factor that multiplies ``max_area``, ``max_length_tol`` and
        ``max_area_ratio_tol`` at this attempt.
    delta_f : float, optional
        Increase of the factor at each new attempt.
    attempt : int, optional
        Number of this attempt.

    Returns
    -------
    InterfaceMatch

    Raises
    ------
    InterfaceMatchError
        When the factor reaches 1.5 without a match.

    Notes
    -----
    To remove: port of ``FT_GenerateInterface.generate_interface``
    (``slabs.py:360-411``, with ``vacuum``) and of the effective
    ``FT_GetHighSymmetryPts.generate_interface`` (``slabs.py:559-597``,
    without), apart from ``max_angle_tol``, which is neither enlarged nor
    forced to 0.4 (DEV-01), and the prints, which made the recursion of
    ``FT_GenerateInterface`` fail (A73, DEV-33). The save of
    ``FT_GenerateInterface`` (``slabs.py:410``) is done by ``write_interface``
    after the return.
    """
    # Raise an error if the matching has not converged after the recursions.
    if factor >= 1.5:
        raise InterfaceMatchError('Error in generating the interface. '
                                  'Matching not converged')

    # Enlarge the tolerances; the factors of the attempts accumulate.
    inter_params['max_area'] *= factor
    inter_params['max_length_tol'] *= factor
    inter_params['max_area_ratio_tol'] *= factor

    # Try to generate the interface.
    matcher = InterfaceMatcher(slab_1=slabs[0], slab_2=slabs[1], **inter_params)
    top_aligned, bot_aligned = matcher.get_centered_slabs()
    interface = matcher.get_interface()
    interface = slab_to_structure(interface)

    # Add the vacuum.
    if vacuum is not None and interface is not None:
        manipulate = ManipulateStruct(structure=interface)
        interface = manipulate.make_vacuum_around(vacuum=vacuum)
        manipulate.decompose_structure()

    # Try again with larger tolerances.
    if any([t is None for t in [bot_aligned, top_aligned, interface]]):
        return generate_interface(slabs, inter_params, vacuum, factor + delta_f, delta_f,
                                  attempt + 1)
    return InterfaceMatch(interface, bot_aligned, top_aligned, matcher, attempt)


def get_hspts(bot_slab, top_slab):
    """High-symmetry points of the two facing surfaces and lateral shifts of the interface.

    Parameters
    ----------
    bot_slab, top_slab : pymatgen.core.surface.Slab
        The aligned slabs of a match.

    Returns
    -------
    dict
        ``bot_unique``, ``bot_all``, ``top_unique``, ``top_all``: the points
        of each surface; ``inter_unique``, ``inter_all``: the shifts of the
        top slab, as JSON data.

    Notes
    -----
    To remove: ``FT_GetHighSymmetryPts.get_hspts`` (``slabs.py:600-679``)
    without the prints.
    """
    # The top slab is mirrored to find the points of its surface at the interface.
    flipped_top = flip_slab(top_slab)
    top_hsp_unique, top_hsp_all = get_slab_hs(flipped_top)
    bottom_hsp_unique, bottom_hsp_all = get_slab_hs(bot_slab)

    cell = bot_slab.lattice.matrix
    hsp_unique = get_interface_hs(bottom_hsp_unique, top_hsp_unique, cell)
    hsp_all = get_interface_hs(bottom_hsp_all, top_hsp_all, cell)
    c_hsp_u, c_hsp_a = fix_hs_dicts(hsp_unique, hsp_all, top_slab, bot_slab)

    b_hsp_u = pbc_hs_points(bottom_hsp_unique, cell)
    b_hsp_a = pbc_hs_points(bottom_hsp_all, cell)
    t_hsp_u = pbc_hs_points(top_hsp_unique, cell)
    t_hsp_a = pbc_hs_points(top_hsp_all, cell)

    return {
        'bot_unique': b_hsp_u,
        'bot_all': b_hsp_a,
        'top_unique': t_hsp_u,
        'top_all': t_hsp_a,
        'inter_unique': jsanitize(c_hsp_u),
        'inter_all': jsanitize(c_hsp_a)}


# --- document ---------------------------------------------------------------------------

def set_filter(params, keys=FILTER_KEYS):
    """Filter of the document: the ids among ``keys`` that are not None.

    Notes
    -----
    To remove: ``FireTaskTribChem.set_filter`` (``core.py:136-161``).
    """
    return {k: copy.deepcopy(params[k]) for k in keys if params.get(k) is not None}


def set_interface_document(store, ids):
    """Create the document of the interface when missing.

    Parameters
    ----------
    store : LegacyStore
    ids : dict
        ``mid``, ``formula``, ``name`` and ``miller`` of the interface.

    Raises
    ------
    StoreError
        If the system has a document with other ids.

    Notes
    -----
    To remove: ``FT_StartInterfaceMatch.set_interface_document``
    (``run.py:1152-1176``) without ``comp_params`` (DEV-35) and without
    ``override_inter_db`` (A45).
    """
    fltr = set_filter(ids)
    field_, _ = store.query_db(COLLECTION, fltr=fltr)
    if field_ is None:
        inter_slab = {
            'mid': ids['mid'],
            'formula': ids['formula'],
            'miller': ids['miller'],
            'structure': {},
        }
        if ids['name'] is not None:
            inter_slab.update({'name': ids['name']})
        store.insert_db(inter_slab, COLLECTION)


def clear_entries(document, entries):
    """Empty the entries that exist, so that a new write leaves nothing of the old one.

    Parameters
    ----------
    document : SystemDocument
    entries : list of tuple of str

    Notes
    -----
    To remove: DEV-36. The historic writes merge the new data into the old
    ones key by key; the points of ``highsym`` are keyed by their labels,
    which change with the interface.
    """
    for entry in entries:
        if document.get(entry) is not None:
            document.replace(entry, {})


def write_interface(store, ids, match, inter_params, start):
    """Write the interface, its aligned slabs and the matching parameters.

    Notes
    -----
    To remove: ``FT_GenerateInterface.update_db`` (``slabs.py:413-443``).
    """
    fltr = set_filter(ids)
    data = {
        'interface': match.interface.as_dict(),
        'bot_slab': match.bot_slab.as_dict(),
        'top_slab': match.top_slab.as_dict()
    }
    for d, entry in zip([data, inter_params], [start['inter_entry'], 'inter_params']):
        store.update_db(d, COLLECTION, entry=entry, fltr=fltr, update_op=start['update_op'],
                        to_mongodb=start['to_mongodb'])


def write_highsym(store, ids, hs, start):
    """Write the high-symmetry points.

    Notes
    -----
    To remove: ``FT_GetHighSymmetryPts.run_task`` (``slabs.py:498-500``),
    with the defaults of ``FireTaskTribChem.update_db`` (``core.py:274-276``).
    """
    store.update_db(hs, COLLECTION, entry=start['hs_entry'], fltr=set_filter(ids))


def run_match(inputs, store):
    """Match the slabs and write the entries of the step, without their provenance.

    Parameters
    ----------
    inputs : MatchInputs
    store : LegacyStore
        On the document of the system, which is changed and not saved.

    Returns
    -------
    MatchResult

    Raises
    ------
    InterfaceMatchError
        If the matching does not converge.
    StoreError
        If the document of the system has other ids.
    """
    set_interface_document(store, inputs.ids)
    clear_entries(store.document, match_entries(inputs.start))

    # The interface: each part starts from its own copy of the parameters.
    slabs = [workflow_slab.slab for workflow_slab in inputs.slabs]
    inter_params = copy.deepcopy(inputs.inter_params)
    match = generate_interface(copy.deepcopy(slabs), inter_params,
                               vacuum=inputs.start['vacuum'])
    write_interface(store, inputs.ids, match, inter_params, inputs.start)

    # The high-symmetry points, from the same matching without vacuum.
    hs_match = generate_interface(copy.deepcopy(slabs), copy.deepcopy(inputs.inter_params))
    hs = get_hspts(hs_match.bot_slab, hs_match.top_slab)
    write_highsym(store, inputs.ids, hs, inputs.start)
    return MatchResult(match, inter_params, hs_match, hs)


# --- message ------------------------------------------------------------------------------

def _label(slab):
    return f'{slab.composition.reduced_formula}({get_miller_str(slab.miller_index)})'


def top_index(matcher, slabs):
    """Index in ``slabs`` of the slab that the matcher put on top."""
    top = matcher.top_slab
    same = (top.composition.reduced_formula == slabs[0].composition.reduced_formula
            and list(top.miller_index) == list(slabs[0].miller_index))
    return 0 if same else 1


def _strain(values):
    if isinstance(values, dict):
        return f'a {values["a"]:+.3f} %, b {values["b"]:+.3f} %, gamma {values["gamma"]:+.3f} deg'
    return f'a {values[0]:+.3f} %, b {values[1]:+.3f} %'


def describe(inputs, result):
    """Lines of the message of the step: slabs, attempts, parameters, strain, points.

    Parameters
    ----------
    inputs : MatchInputs
    result : MatchResult

    Returns
    -------
    list of str
    """
    slabs = [workflow_slab.slab for workflow_slab in inputs.slabs]
    top = top_index(result.match.matcher, slabs)
    bot = 1 - top
    p = result.inter_params
    attempts = result.match.attempts
    lines = [
        f'interface {inputs.ids["mid"]}: {_label(slabs[top])} (slab {top + 1}, '
        f'{inputs.params["mid"][top]}) on top of {_label(slabs[bot])} (slab {bot + 1}, '
        f'{inputs.params["mid"][bot]})',
        f'match found at attempt {attempts}' + (
            '' if attempts == 1 else
            '; max_area, max_length_tol and max_area_ratio_tol enlarged at each attempt'),
        f'parameters used: max_area = {p["max_area"]:.6g} angstrom^2, '
        f'max_length_tol = {p["max_length_tol"]:.6g}, '
        f'max_area_ratio_tol = {p["max_area_ratio_tol"]:.6g}, '
        f'max_angle_tol = {p["max_angle_tol"]:.6g} (never enlarged), '
        f'interface_distance = {p["interface_distance"]}, '
        f'strain weights = {p["strain_weight_1"]}, {p["strain_weight_2"]} '
        f'(slab 1, slab 2)',
    ]
    strain = result.match.matcher.get_strain()
    lines.append(f'strain of the top slab: {_strain(strain["top"])}; of the bottom slab: '
                 f'{_strain(strain["bot"])}')
    lines.append(f'high-symmetry points: {len(result.hs["inter_unique"])} lateral shifts '
                 f'(inter_unique), {len(result.hs["inter_all"])} in all (inter_all)')
    return lines


# --- step ---------------------------------------------------------------------------------

def _load_run(run_dir):
    # Imported here: the command line imports the registry of the steps.
    from triboworkflow.cli.run import load_run

    return load_run(run_dir)


class MatchStep(Step):
    """Step ``match``: interface and high-symmetry points, no calculation."""

    name = STEP

    def done(self, run_dir, system):
        """Whether the five entries are written and current (``match_provenance``)."""
        sdir = system_dir(run_dir, system)
        document = read_system(sdir)
        expected = match_provenance(_load_run(run_dir))
        return all(is_done(document, entry, expected, sdir) for entry in match_entries())

    def analyse(self, run_dir, system, calculations):
        """Match the slabs, write the files and the entries with their provenance."""
        inputs = match_inputs(_load_run(run_dir))
        if inputs.ids['mid'] != system:
            raise ValueError(f'system {system!r}: the slabs of the settings make the interface '
                             f'{inputs.ids["mid"]!r}')
        for i, workflow_slab in enumerate(inputs.slabs):
            changed = []
            if workflow_slab.rotated:
                changed.append('rotated to the frame of the workflow')
            if workflow_slab.dropped:
                changed.append(f'{", ".join(workflow_slab.dropped)} dropped')
            if changed:
                logger.warning('%s: slab %d (%s): %s', system, i + 1, inputs.files[i],
                               '; '.join(changed))
        sdir = system_dir(run_dir, system)
        with SystemWriter(sdir) as writer:
            # The matching code prints its progress: kept in the log, at debug level.
            printed = io.StringIO()
            try:
                with contextlib.redirect_stdout(printed):
                    result = run_match(inputs, LegacyStore(sdir, writer.document, COLLECTION))
            finally:
                if printed.getvalue():
                    logger.debug('%s: output of the matching code:\n%s', system,
                                 printed.getvalue())
            out = sdir / OUTPUT_DIR
            out.mkdir(exist_ok=True)
            write_structure(result.match.interface, out / INTERFACE_FILE)
            write_json(out / INTERFACE_JSON, result.match.interface.as_dict())
            provenance = inputs.provenance()
            for entry in match_entries(inputs.start):
                writer.document.set_provenance(entry, provenance)
            writer.save()
        for line in describe(inputs, result):
            logger.info('%s: %s', system, line)
        logger.info('%s: interface written in %s and %s', system,
                    Path(OUTPUT_DIR) / INTERFACE_FILE, Path(OUTPUT_DIR) / INTERFACE_JSON)
