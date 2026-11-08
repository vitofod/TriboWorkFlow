# Author: Vito Foderà
# Created: 2026-11-07

"""Steps ``plotpes`` and ``plotpes_scf`` of the interface workflow: the PES interpolated.

Each step interpolates one of the two potential energy surfaces (PES) of the
interface over the whole plane, from the energies of its high-symmetry
points: ``plotpes`` the PES relaxed along z (entry ``pes``), ``plotpes_scf``
the static one (entry ``pes_scf``). A run requests one, the other or both in
``[interface].steps``; neither is a default. They run no calculation: the
analysis reads the document of the system and writes into it.

The analysis reads, from the entries written by ``match`` and by the PES step:

- ``highsym.inter_unique``: the unique high-symmetry points, with their first
  lateral shift ``x, y``;
- ``highsym.inter_all``: every replica of each point in the cell;
- ``<pes>.data.<point>.output.energy`` of each point (eV);
- the final structure of the point of lowest energy: its cell, and the area of
  the cell for the energies in J/m^2.

The points are sorted by energy (``[point, x, y, energy]``, ties in the order
of the points) and their energies converted to J/m^2 on the area of the cell
(``Converter.eV_to_Jm2``). ``get_pes`` then gives every replica the energy of
its point, removes the duplicates, brings the points into the cell, sets the
minimum to zero and interpolates them with radial basis functions (thin-plate
spline, smoothing 1e-3, points replicated 6 x 6 around the cell, grid of
density 50): once with the energies in eV, once in J/m^2, the second time
drawing the figure. The analysis writes, in the entry ``<pes>.energy``:

- ``grid.hs_all_ev`` and ``grid.hs_all_jm2``: ``[point, x, y, energy]`` of
  every replica;
- ``pes_ev`` and ``pes_jm2``: ``[x, y, energy]`` of the points interpolated,
  the minimum at zero;

each with the provenance of the step: its parameters and the content of the
entries read. Writing into ``<pes>.energy`` leaves the entries of the PES
step current: they have their provenance each (``pes.pes_leaves``). The
figure (energies in J/m^2) is the file ``<step>/pes_<name>.png`` in the
folder of the system, ``<name>`` the name of the interface in the document;
it is not stored in the document. The two steps write in two folders, so
their figures never overwrite each other.

Cost: the interpolation solves a dense linear system on all the points
replicated (144 times the points left after the duplicates), whose matrix
grows with the square of their number: 20 736 points (an interface of 28
atoms) take about two minutes and 3.2 GiB for the matrix alone; 108 864
points (64 atoms) would need about 88 GiB.

Where it runs: only by hand (``Step.manual_analysis``), with ``triboworkflow
analyse plotpes`` or ``analyse plotpes_scf`` on a machine with enough memory.
The orchestrator in the background, on the login node, never interpolates:
once the step is ready it leaves it ``manual``, with that command, and the
steps after it wait for it. The analysis takes the lock of the run, so it
runs with the orchestrator stopped or ended.

All or nothing: the entries are read, the interpolation runs and the figure is
written to a temporary file without locking the document; then, with the
document locked, the prerequisites and the entries read are checked again,
the entries written, the figure renamed and the document saved once. An
interruption leaves the document as it was; a figure without its entries does
not make the step done, and the next analysis starts from the beginning. A
step done is recognised from its entries, their provenance and the figure,
without interpolating again.

Notes
-----
To remove: port of TribChem's ``FT_StartInterpolation``
(``highput/firetasks/run.py:1272-1309``) and ``FT_PESInterpolation``
(``highput/firetasks/pes.py:140-262``: ``arrange_data``, ``compute_pes``),
with the document found as ``FireTaskInterface.query_interface``
(``core.py:493-556``) finds it. The two steps are the two values of ``-pt``
of TribChem's command line (``set_pes``, ``core/cli/calc_interface_cli.py
:171-199``), which interpolated one PES per invocation, ``pes`` by default
(DEV-41). The figure stays a file and the key ``image`` (the bytes of the PNG
encoded again with PIL) is not written (DEV-42); the figure is closed once
saved (A84). The interpolation is computed twice, as there (A07). Not ported:
``override``, ``check_entry`` (the provenance decides), ``to_file`` (A83),
``db_file``, ``database``; ``pes_type`` is always ``energy``. The writes use
the ids of the document written by ``match`` (DEV-39). The docstring of
``get_pes`` describes a cubic function on 3 x 3 replicas; the code is what
counts (A85). The analysis by hand stands until the job of its own (D54).
"""

import contextlib
import io
import logging
import os
import time
from dataclasses import dataclass
from operator import itemgetter

import matplotlib
import matplotlib.pyplot as plt
from monty.json import jsanitize

from triboworkflow.core.store import (PrerequisiteError, Provenance, StoreError, SystemWriter,
                                      check_prerequisites, is_done, read_system)
from triboworkflow.core.store.legacy import LegacyStore
from triboworkflow.execution.layout import system_dir
from triboworkflow.execution.orchestrator import Step
from triboworkflow.vendored.highput.database.dbtools import read_one_entry
from triboworkflow.vendored.highput.utils.tasktools import get_miller_str, select_struct_func
from triboworkflow.vendored.physics.base.solidstate import get_cell_info
from triboworkflow.vendored.physics.base.units import Converter
from triboworkflow.vendored.physics.tribology.potensurf import get_pes

from .match import COLLECTION, FILTER_KEYS, match_parameters, match_provenance, set_filter
from .pes import CALC_TYPES, ID_KEYS, PES_TYPE, RelaxedPesStep, StaticPesStep, pes_entry

logger = logging.getLogger(__name__)

# The PES that each step interpolates, and the step that computes it.
PES_OF = {'plotpes': 'pes', 'plotpes_scf': 'pes_scf'}
PES_STEPS = {'pes': RelaxedPesStep, 'pes_scf': StaticPesStep}
# Entries of highsym read (written by match).
# To remove: FT_PESInterpolation.arrange_data (pes.py:186-187).
HIGHSYM = ('highsym',)
INTER_UNIQUE = ['highsym', 'inter_unique']
INTER_ALL = ['highsym', 'inter_all']
# Entries written below <pes>.energy, each with the provenance of the step.
INTERPOLATION = (('grid', 'hs_all_ev'), ('grid', 'hs_all_jm2'), ('pes_ev',), ('pes_jm2',))


@dataclass
class PlotPesInputs:
    """What an interpolation step reads from the document of the system.

    Attributes
    ----------
    name : str
        ``plotpes`` or ``plotpes_scf``: the step, and the folder of its figure.
    pes : str
        ``pes`` or ``pes_scf``: the entry of the PES interpolated.
    calc_type : str
        ``calc_type`` of the calculations of the PES.
    ids : dict
        Ids of the document of ``match``, which select it in the writes.
    title : str
        Name of the interface: title and file name of the figure.
    all_shifts : dict
        ``highsym.inter_all``: ``{point: [[x, y], ...]}``.
    values : list
        ``[hs_unique_ev, hs_unique_jm2]``: ``[point, x, y, energy]`` sorted by
        the energy, in eV and in J/m^2.
    cell : numpy.ndarray
        Lattice matrix of the final structure of the minimum.
    """

    name: str
    pes: str
    calc_type: str
    ids: dict
    title: str
    all_shifts: dict
    values: list
    cell: object

    def read(self):
        """The content of the entries read, as given to the interpolation (for the provenance)."""
        return jsanitize({'inter_all': self.all_shifts, 'values': self.values,
                          'cell': self.cell}, strict=True)


@dataclass
class Interpolation:
    """What the two calls to ``get_pes`` give, ready to be written.

    Attributes
    ----------
    data : dict
        ``{'energy': {'grid': {'hs_all_ev', 'hs_all_jm2'}, 'pes_ev',
        'pes_jm2'}}``, through ``jsanitize``.
    unfolded : int
        Points of every replica in the cell.
    points : int
        Points left after removing the duplicates.
    seconds : tuple of float
        Time of the interpolation in eV and of that in J/m^2 with the figure.
    """

    data: dict
    unfolded: int
    points: int
    seconds: tuple


# --- inputs ---------------------------------------------------------------------------

def interpolation_leaves(pes):
    """The entries that an interpolation of the PES ``pes`` writes.

    Returns
    -------
    list of tuple of str
        ``<pes>.energy.grid.hs_all_ev``, ``.hs_all_jm2``,
        ``<pes>.energy.pes_ev`` and ``.pes_jm2``.
    """
    return [(pes, PES_TYPE, *leaf) for leaf in INTERPOLATION]


def interface_name(ids):
    """Name of the interface, for the title and the file of the figure.

    Parameters
    ----------
    ids : dict
        Ids of the document: ``name``, else ``formula`` and ``miller``.

    Returns
    -------
    str
        ``name``; without it, formula and Miller indices of the two slabs
        (``Cu111Al111`` for ``Cu_Al`` and ``[[1, 1, 1], [1, 1, 1]]``).

    Notes
    -----
    To remove: ``FT_PESInterpolation.compute_pes`` (``pes.py:224-229``).
    """
    if ids.get('name') is not None:
        name = str(ids['name'])
    elif ids.get('formula') is not None:
        formula = ids['formula'].split('_')
        name = (formula[0] + get_miller_str(ids['miller'][0]) + formula[1]
                + get_miller_str(ids['miller'][1]))
    else:
        raise StoreError('the document of the interface has neither a name nor a formula')
    if '/' in name or os.sep in name or name in ('', '.', '..'):
        raise StoreError(f'the name of the interface {name!r} cannot name a file')
    return name


def figure_path(sdir, name, title):
    """Figure of the step ``name``: ``<step>/pes_<title>.png`` in the folder of the system."""
    return sdir / name / f'pes_{title}.png'


def partial_path(figure):
    """Temporary file of the figure, renamed to ``figure`` when the entries are saved.

    It ends in ``.png``, from which matplotlib takes the format.
    """
    return figure.with_name(f'.{figure.stem}.partial.png')


def arrange_data(store, ids, calc_type, pes_type=PES_TYPE):
    """The points sorted by energy, in eV and in J/m^2, and the cell of the minimum.

    Parameters
    ----------
    store : LegacyStore
        On the document of the system.
    ids : dict
        Ids of the document.
    calc_type : str
        ``calc_type`` of the calculations of the PES: it chooses the entry.
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
    To remove: ``FT_PESInterpolation.arrange_data`` (``pes.py:173-216``),
    which reads only: nothing is written here. Its print of the values sorted
    goes to the log, at debug level.
    """
    # Retrieve the document with all the data
    fltr = set_filter(ids)
    field_, _ = store.query_db(COLLECTION, fltr=fltr)

    # Recover the shift names and values
    shifts = read_one_entry(field_, INTER_UNIQUE)
    all_shifts = read_one_entry(field_, INTER_ALL)

    # Set the name for the pes based on calc_type
    pes_calc = pes_entry(calc_type)
    # Collect all values associated to shifts and specified by pes_type
    values = []
    for s, d in shifts.items():
        x, y = d[0]
        v = read_one_entry(field_, [pes_calc, 'data', str(s), 'output', pes_type])
        values.append([s, x, y, v])
    sorted_values = sorted(values, key=itemgetter(3))
    logger.debug('points sorted by %s: %s', pes_type, sorted_values)
    # Find out the minimum and maximum stacking for values
    min_stack = sorted_values[0][0]
    struct_min = read_one_entry(field_, [pes_calc, 'data', str(min_stack), 'output', 'structure'])

    # Prepare the grid energy values to be saved in J/m^2
    area, _ = get_cell_info(struct_min['lattice']['matrix'])
    sorted_values_jm2 = [[s, x, y, Converter.eV_to_Jm2(v, area * 1e-20)]
                         for s, x, y, v in sorted_values]

    # Take out the cell of the lattice matrix
    func = select_struct_func(struct_min['@class'])
    cell = func.from_dict(struct_min).lattice.matrix

    return all_shifts, [sorted_values, sorted_values_jm2], cell


def _missing(field_, pes):
    # Entries that arrange_data needs and the document lacks.
    if not read_one_entry(field_, INTER_UNIQUE) or not read_one_entry(field_, INTER_ALL):
        return ['highsym.inter_unique', 'highsym.inter_all']
    missing = []
    for s in read_one_entry(field_, INTER_UNIQUE):
        output = read_one_entry(field_, [pes, 'data', str(s), 'output'])
        if not output or output.get(PES_TYPE) is None or not output.get('structure'):
            missing.append(f'{pes}.data.{s}.output')
    return missing


def plotpes_inputs(sdir, document, settings, name):
    """Everything an interpolation step reads from the document.

    Parameters
    ----------
    sdir : pathlib.Path
        Folder of the system.
    document : SystemDocument
        Document of the system.
    settings : triboworkflow.core.settings.Settings
        Settings of the run: the slabs find the document.
    name : str
        ``plotpes`` or ``plotpes_scf``.

    Returns
    -------
    PlotPesInputs

    Raises
    ------
    PrerequisiteError
        If the document, the high-symmetry points or the results of a point
        of the PES are missing.

    Notes
    -----
    To remove: the document is found with ``mid`` and ``miller``, as
    ``FireTaskInterface.query_interface`` finds it (``core.py:500-503``);
    the ids are those of the document (DEV-39).
    """
    pes = PES_OF[name]
    calc_type = CALC_TYPES[pes]
    params, _, _ = match_parameters(settings)
    store = LegacyStore(sdir, document, COLLECTION)
    field_, _ = store.query_db(COLLECTION, fltr=set_filter({'mid': '_'.join(params['mid']),
                                                            'miller': params['miller']}))
    if field_ is None:
        raise PrerequisiteError(f'{sdir.name}: the system has no document yet: run match first')
    missing = _missing(field_, pes)
    if missing:
        raise PrerequisiteError(f'{sdir.name}: entries missing: {", ".join(missing)}; run '
                                f'{"match" if missing[0].startswith("highsym") else pes} first')
    ids = set_filter({key: field_.get(key) for key in ID_KEYS}, FILTER_KEYS)
    all_shifts, values, cell = arrange_data(store, ids, calc_type)
    return PlotPesInputs(name=name, pes=pes, calc_type=calc_type, ids=ids,
                         title=interface_name(ids), all_shifts=all_shifts, values=values,
                         cell=cell)


def plotpes_provenance(inputs):
    """Provenance that the entries of an interpolation step have when computed now.

    Parameters
    ----------
    inputs : PlotPesInputs

    Returns
    -------
    Provenance
        The parameters of the step and the content of the entries read, as
        given to the interpolation.
    """
    return Provenance.build(inputs.name,
                            parameters={'calc_type': inputs.calc_type, 'pes_type': PES_TYPE},
                            inputs=inputs.read())


# --- analysis -------------------------------------------------------------------------

def interpolate(inputs, figure):
    """The two interpolations of the PES, the second drawing the figure.

    Parameters
    ----------
    inputs : PlotPesInputs
    figure : pathlib.Path
        File where the figure of the PES in J/m^2 is saved; it must end in
        ``.png``.

    Returns
    -------
    Interpolation

    Notes
    -----
    To remove: ``FT_PESInterpolation.compute_pes`` (``pes.py:231-250``) up to
    the dict written, without ``image`` (DEV-42). matplotlib draws without a
    display (backend Agg); the figure is closed once saved (A84). What
    ``get_pes`` prints goes to the log, at debug level.
    """
    matplotlib.use('Agg')
    hs_all, unique_values, cell = inputs.all_shifts, inputs.values, inputs.cell
    pes_name = inputs.title
    opened = set(plt.get_fignums())
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(printed):
            # Obtain PES data in eV and J/m^2 interpolating with RBF
            start = time.perf_counter()
            _, v_list, data, _ = get_pes(hs_all, unique_values[0], cell,
                                         title=pes_name, to_fig=None)
            middle = time.perf_counter()
            logger.info('%s: interpolation in eV: %.1f s', inputs.name, middle - start)
            _, v_list_jm2, data_jm2, _ = get_pes(hs_all, unique_values[1], cell,
                                                 title=pes_name, to_fig=str(figure))
            end = time.perf_counter()
            logger.info('%s: interpolation in J/m^2 and figure: %.1f s', inputs.name, end - middle)
    finally:
        for number in set(plt.get_fignums()) - opened:
            plt.close(number)
        if printed.getvalue():
            logger.debug('%s: output of the vendored code:\n%s', inputs.name, printed.getvalue())

    # Prepare data do be stored in the database
    data_written = {
        PES_TYPE: {
            'grid':
                {
                    'hs_all_ev': jsanitize(v_list),
                    'hs_all_jm2': jsanitize(v_list_jm2)
                    },
            'pes_ev': jsanitize(data),
            'pes_jm2': jsanitize(data_jm2),
            }
        }
    return Interpolation(data=data_written, unfolded=len(v_list), points=len(data),
                         seconds=(middle - start, end - middle))


def write_interpolation(store, inputs, interpolation):
    """Write the entries of the interpolation into the document.

    Parameters
    ----------
    store : LegacyStore
        On the document of the system, which is changed and not saved.
    inputs : PlotPesInputs
    interpolation : Interpolation

    Notes
    -----
    To remove: ``FT_PESInterpolation.compute_pes`` (``pes.py:252-257``).
    """
    # Set the name for the pes based on calc_type
    pes_calc = pes_entry(inputs.calc_type)
    store.update_db(interpolation.data, COLLECTION, entry=pes_calc, fltr=set_filter(inputs.ids))


def describe(inputs, interpolation, figure):
    """Lines of the message of the step: points, times and figure.

    Returns
    -------
    list of str
    """
    ev = interpolation.seconds[0]
    jm2 = interpolation.seconds[1]
    return [
        f'{inputs.name}: PES {inputs.pes} ({inputs.calc_type}) interpolated: '
        f'{len(inputs.values[0])} high-symmetry points, {interpolation.unfolded} in the cell, '
        f'{interpolation.points} after removing the duplicates',
        f'interpolation in eV {ev:.1f} s, in J/m^2 with the figure {jm2:.1f} s',
        f'figure written in {figure.parent.name}/{figure.name}',
    ]


# --- steps --------------------------------------------------------------------------------

def _load_run(run_dir):
    # Imported here: the command line imports the registry of the steps.
    from triboworkflow.cli.run import load_run

    return load_run(run_dir)


class PlotPesStep(Step):
    """An interpolation step: the PES of one entry interpolated, no calculation.

    Analysed only by hand (``manual_analysis``): its memory grows with the
    square of the points interpolated.
    """

    name = None
    manual_analysis = True

    def done(self, run_dir, system):
        """Whether the four entries are written and current and the figure exists.

        Nothing is interpolated: the entries read are compared with those
        recorded in the provenance.
        """
        sdir = system_dir(run_dir, system)
        document = read_system(sdir)
        leaves = interpolation_leaves(PES_OF[self.name])
        # Before any analysis the inputs are not read: the PES may not have run.
        if not all(document.get(leaf) for leaf in leaves):
            return False
        try:
            inputs = plotpes_inputs(sdir, document, _load_run(run_dir), self.name)
        except PrerequisiteError:
            return False
        if not figure_path(sdir, self.name, inputs.title).is_file():
            return False
        expected = plotpes_provenance(inputs)
        return all(is_done(document, leaf, expected, sdir) for leaf in leaves)

    def expected(self, run_dir, system):
        """``highsym`` with the provenance of ``match``, ``<pes>.data`` with that of the PES step."""
        pes = PES_OF[self.name]
        return {HIGHSYM: match_provenance(_load_run(run_dir)),
                (pes, 'data'): PES_STEPS[pes]().provenance(run_dir, system)}

    def provenance(self, run_dir, system):
        """Provenance expected now for the entries that the step writes."""
        sdir = system_dir(run_dir, system)
        return plotpes_provenance(plotpes_inputs(sdir, read_system(sdir), _load_run(run_dir),
                                                 self.name))

    def analyse(self, run_dir, system, calculations):
        """Interpolate the PES, save the figure and write the entries with their provenance.

        Raises
        ------
        PrerequisiteError
            If ``highsym`` or the data of the PES are missing or not current.
        StoreError
            If the entries read change during the interpolation: nothing is
            written.
        """
        sdir = system_dir(run_dir, system)
        settings = _load_run(run_dir)
        check_prerequisites(read_system(sdir), self.expected(run_dir, system), sdir)
        inputs = plotpes_inputs(sdir, read_system(sdir), settings, self.name)
        provenance = plotpes_provenance(inputs)
        figure = figure_path(sdir, self.name, inputs.title)
        partial = partial_path(figure)
        figure.parent.mkdir(parents=True, exist_ok=True)
        try:
            interpolation = interpolate(inputs, partial)
            with SystemWriter(sdir) as writer:
                check_prerequisites(writer.document, self.expected(run_dir, system), sdir)
                again = plotpes_inputs(sdir, writer.document, settings, self.name)
                if plotpes_provenance(again) != provenance:
                    raise StoreError(f'{system}/{self.name}: the entries read changed during the '
                                     'interpolation; nothing written')
                write_interpolation(LegacyStore(sdir, writer.document, COLLECTION), inputs,
                                    interpolation)
                for leaf in interpolation_leaves(inputs.pes):
                    writer.document.set_provenance(leaf, provenance)
                os.replace(partial, figure)
                writer.save()
        finally:
            partial.unlink(missing_ok=True)
        for line in describe(inputs, interpolation, figure):
            logger.info('%s: %s', system, line)


class RelaxedPlotPesStep(PlotPesStep):
    """Step ``plotpes``: the PES relaxed along z (entry ``pes``) interpolated."""

    name = 'plotpes'


class StaticPlotPesStep(PlotPesStep):
    """Step ``plotpes_scf``: the static PES (entry ``pes_scf``) interpolated."""

    name = 'plotpes_scf'
