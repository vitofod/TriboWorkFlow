# Author: Vito Foderà
# Created: 2026-11-09

"""Step ``shear`` of the interface workflow: the shear strength of the interface.

The step finds, on the potential energy surface (PES) of the interface relaxed along
z and interpolated by ``plotpes``, the paths of minimum energy (MEP) that one slab
follows when it slides on the other, and the shear strength along each of them: the
largest force that opposes the sliding, in GPa. It runs no calculation: the analysis
reads the document of the system and writes into it.

The analysis reads, from the entries written by ``match`` and by ``plotpes``:

- ``pes.energy.pes_jm2``: ``[x, y, energy]`` of the points of the PES, in J/m^2, the
  minimum at zero;
- ``structure.init.interface.lattice.matrix``: the cell of the interface.

The PES is interpolated again with radial basis functions (thin-plate spline,
smoothing 1e-3, points replicated 5 x 5 around the cell, grid of density 20 that
this step does not use), once for the three paths. For each of the starting strings
``bs_line`` (the straight line, among the orientations in steps of one degree, along
which the largest derivative of the PES is the smallest, with a small random noise),
``x`` and ``y`` (the axes of the cell) the string method at zero temperature evolves
the string to the MEP. The force along the path is minus the derivative of the PES
along its tangent, and the shear strength is its largest absolute value, in GPa
(J/m^2 per angstrom times 10). The parameters of the string method are the settings
``[interface.shear.mep]``, and the step of the derivative of the force and the seed of
the noise ``[interface.shear]``. The analysis writes, in the entry ``shearstr``:

- ``along_<string>.nstep`` and ``.tol``: the index of the last step of the string
  method (the steps done minus one) and the mean displacement of the string in it;
- ``along_<string>.shear_strength_GPa``: the shear strength, computed on the whole path;
- ``along_<string>.lxy``, ``.dvx``, ``.dvy``, ``.vz``, ``.ve`` and ``.force``: six numbers
  each. The code that computes the shear strength builds a table with a row for each point
  of the path and the columns length along the path, derivatives of the PES in x and y,
  integral of the force, energy of the PES (the last two with the minimum at zero) and
  force, and writes its first six rows under the names of the six columns. They are the
  values at each of the first six points of the path, not the profiles along it, so a
  string needs at least six points;

each of the three ``along_<string>`` with the provenance of the step: its parameters
and the content of the entries read. A string that reaches ``nstepmax`` without its
displacement falling below ``tol`` is written like one that converged: compare ``tol`` of
the entry with the one of the settings.

The profiles along the paths are in the text file ``shear/shear.output`` of the folder of
the system (``shear_output``): the parameters, the shear strength, ``nstep`` and ``tol`` of
each string, and for each string the whole table, one row for each point of the path,
with its position ``x``, ``y`` in angstrom and the six columns above: ``lxy`` in angstrom,
``dvx``, ``dvy`` and ``force`` in J/m^2 per angstrom, ``vz`` and ``ve`` in J/m^2. The
table is the one the shear strength is computed from, not computed again.

The noise of ``bs_line`` is drawn with the global generator of NumPy, which is seeded
with ``[interface.shear] seed`` before each string and restored at the end, so that
the analysis can be repeated and does not change the numbers that the program draws
elsewhere.

Cost: the interpolation solves a dense linear system on all the points replicated
(100 times the points of the PES), and every step of the string method asks the
interpolation for the value of the PES at the points of the string, four times. With
the recommended parameters a string takes many thousands of steps; ``tests/shear_cost.py``
measures it.

Where it runs: never in the orchestrator, on the login node
(``Step.manual_analysis``). The orchestrator submits it in a job of its own on a compute
node, with the jobscript of ``[execution.analysis]`` or else of ``[execution]``, and
follows the job, which runs ``analyse`` of this step (``cli/analysis_job.py``) without
taking the lock of the run. The user may also run ``triboworkflow analyse shear`` on a
machine with enough memory; that command takes the lock of the run, so it runs with the
orchestrator stopped or ended.

Two figures go next to ``shear.output`` (``draw_paths``): ``mep_<name>.png`` and ``.pdf``,
the map of the PES of the interpolation with the three paths drawn on it, and
``mep_profile_<name>.png`` and ``.pdf``, Delta E and the shear stress along each path.

All or nothing: the entries are read, the three strings are computed without locking the
document and ``shear.output`` and the figures written to temporary files; then, with the
document locked, the prerequisites and the entries read are checked again, the entries
written, the document saved once and the files renamed. An interruption before the save
leaves the document as it was: the step is never done with a string missing. Each file
carries the stamp of the provenance saved (``outputs.analysis_stamp``), so files left by
an interruption after the save do not make the step done. A step done is recognised from
its entries, their provenance, ``shear.output`` and the figures with that stamp, without
computing again.

Notes
-----
To remove: port of TribChem's ``FT_StartShearStrength``
(``highput/firetasks/run.py:1471-1735``: ``get_pes``) and ``FT_CalcShearStrength``
(``highput/firetasks/shearstr.py:34-161``), through ``ShearstrWF.calculate_shearstr``.
The PES is the one of ``plotpes`` and not the grid of ``pes`` (A58): the grid and the
computational parameters, which ``get_pes`` read and the calculation did not use, are
not read (A96). The three ``update_db`` of the original, one after each string, are
kept, with a single save at the end, so that the step is done only when the three
strings are written (A95); a string stopped by ``nstepmax`` before it converged is written
and counts as well, as there (A99). The options of the string method of the command line never reached it,
and it ran with the defaults of its own file, which are the defaults here (A94, DEV-47);
the noise was not seeded (DEV-46, Q02). ``get_shear_strength`` shifts the arrays of the path
in place and shifts them back, an effect of the order of the rounding (A97). The warning for
a cell that is not a square is Q24. The vendored code prints the coordinates of the path
and the shear strength: that goes to the log, at debug level. The entry is fixed
(``shearstr``, ``-ess`` of the command line); the document is found with ``mid`` and
``miller`` as there, which keep the order of the user, so the ids that A80 changes are not
involved.
"""

import contextlib
import io
import logging
import math
import time
from dataclasses import dataclass

import matplotlib
import numpy as np
from monty.json import jsanitize

from triboworkflow.core.settings.schema import SHEAR_DELTA, SHEAR_MEP
from triboworkflow.core.store import (PrerequisiteError, Provenance, StoreError, SystemWriter,
                                      check_prerequisites, is_done, read_system)
from triboworkflow.core.store.legacy import LegacyStore
from triboworkflow.execution.layout import system_dir
from triboworkflow.execution.orchestrator import Step
from triboworkflow.vendored.highput.database.dbtools import read_one_entry
from triboworkflow.vendored.physics.tribology.mep import MEP
from triboworkflow.vendored.physics.tribology.pes import PES

from .chain import check_upstream, upstream_done, upstream_entries_done
from .figures import figure_paths, pes_figure, profile_figure
from .match import COLLECTION, FILTER_KEYS, match_parameters, match_provenance, set_filter
from .outputs import analysis_stamp, header, place, staged, written_by
from .pes import ID_KEYS
from .plotpes import RelaxedPlotPesStep, interface_name

logger = logging.getLogger(__name__)

# Entry written, and the three strings, in the order in which they are computed and written.
ENTRY = 'shearstr'
OPTIMIZATIONS = ('bs_line', 'x', 'y')
# Method of the string method: the only one.
METHOD = 'zerotemp'
# Interpolation of the PES: replicas around the cell, density of the grid and tolerance on
# the points brought into the cell.
REPLICATE_OF = (5, 5)
DENSITY = 20
PES_TOL = 1e-4
# Entries read: the interpolated PES of plotpes and the interface of match.
PES_JM2 = ('pes', 'energy', 'pes_jm2')
INTERFACE = ('structure', 'init', 'interface')
CELL = INTERFACE + ('lattice', 'matrix')
# A cell is a square with its sides along x and y if its sides differ and its off-diagonal
# elements are at most this fraction of its largest side.
SQUARE_TOL = 1e-3
# Text file of the step, in the folder of the system.
OUTPUT = ('shear', 'shear.output')
# Columns of a profile: the position of the point, then those of the table from which the
# vendored code computes the shear strength, in its order.
PROFILE_COLUMNS = ('x', 'y', 'lxy', 'dvx', 'dvy', 'vz', 've', 'force')


@dataclass
class ShearInputs:
    """What the step reads from the document of the system.

    Attributes
    ----------
    ids : dict
        Ids of the document of ``match``, which select it in the writes.
    pes_jm2 : list
        ``pes.energy.pes_jm2``: ``[x, y, energy]`` of the points of the PES.
    cell : list
        ``structure.init.interface.lattice.matrix``: the cell of the interface.
    """

    ids: dict
    pes_jm2: list
    cell: list

    def read(self):
        """The content of the entries read, as given to the calculation (for the provenance)."""
        return jsanitize({'pes_jm2': self.pes_jm2, 'cell': self.cell}, strict=True)


@dataclass
class ShearResult:
    """What the three strings give, ready to be written.

    Attributes
    ----------
    data : dict
        ``{string: {'along_<string>': {...}}}``, through ``jsanitize``: what each string
        writes in the entry ``shearstr``.
    points : int
        Points of the PES interpolated.
    centres : int
        Centres of the radial basis functions.
    seconds : dict
        ``interpolation`` and the time of each string, in seconds.
    profiles : dict
        ``{string: array of shape (npts, 8)}``: the columns ``PROFILE_COLUMNS`` for each
        point of the path.
    solver : triboworkflow.vendored.physics.tribology.pes.PES or None
        The interpolation on which the strings were found, for the figure of the paths.
    """

    data: dict
    points: int
    centres: int
    seconds: dict
    profiles: dict
    solver: object = None


# --- inputs ---------------------------------------------------------------------------

def shear_leaves():
    """The entries that the step writes, one for each string, in the order of the strings.

    Returns
    -------
    list of tuple of str
        ``shearstr.along_bs_line``, ``shearstr.along_x`` and ``shearstr.along_y``.
    """
    return [(ENTRY, f'along_{opt}') for opt in OPTIMIZATIONS]


def mep_arguments(params):
    """The parameters of the string method, as given to it.

    Parameters
    ----------
    params : ShearParams
        ``[interface.shear]``.

    Returns
    -------
    dict
        ``npts``, ``extent``, ``h``, ``nstepmax``, ``tol`` and ``delta`` of
        ``[interface.shear.mep]``, ``extent`` as a new list.
    """
    mep = params.mep
    return {'npts': mep.npts, 'extent': list(mep.extent), 'h': mep.h, 'nstepmax': mep.nstepmax,
            'tol': mep.tol, 'delta': mep.delta}


def not_recommended(params):
    """The parameters of the analysis other than the recommended ones.

    Parameters
    ----------
    params : ShearParams
        ``[interface.shear]``.

    Returns
    -------
    list of str
        ``<key> = <value> (recommended <default>)`` for ``delta`` and for each parameter of
        the string method that differs from the default, in the order of the settings. The
        seed has no recommended value.
    """
    told = []
    if params.delta != SHEAR_DELTA:
        told.append(f'delta = {params.delta!r} (recommended {SHEAR_DELTA!r})')
    given = mep_arguments(params)
    for key, default in SHEAR_MEP.items():
        recommended = list(default) if key == 'extent' else default
        if given[key] != recommended:
            told.append(f'mep.{key} = {given[key]!r} (recommended {recommended!r})')
    return told


def cell_warning(cell):
    """The warning for a cell that is not a square with its sides along x and y.

    The starting strings are built on ``cell[0][0]`` and ``cell[1][1]`` only, which are the
    two lengths of the cell for a square cell with its sides along the axes.

    Parameters
    ----------
    cell : array-like
        Lattice matrix of the interface.

    Returns
    -------
    str or None
        None for a square cell along x and y (within ``SQUARE_TOL``); otherwise the text
        of the warning, with the lengths a and b of the two lateral vectors and the
        angle gamma between them.
    """
    plane = np.asarray(cell, dtype=float)[:2, :2]
    a, b = np.linalg.norm(plane[0]), np.linalg.norm(plane[1])
    gamma = math.degrees(math.acos(max(-1.0, min(1.0, float(plane[0] @ plane[1]) / (a * b)))))
    tolerance = SQUARE_TOL * max(a, b)
    if (abs(plane[0, 1]) <= tolerance and abs(plane[1, 0]) <= tolerance
            and abs(plane[0, 0] - plane[1, 1]) <= tolerance):
        return None
    return (f'the cell of the interface is not a square (a = {a:.4f}, b = {b:.4f} angstrom, '
            f'gamma = {gamma:.2f} degrees): the strings are built on the diagonal elements of '
            'the cell only')


def shear_inputs(sdir, document, settings):
    """Everything the step reads from the document.

    Parameters
    ----------
    sdir : pathlib.Path
        Folder of the system.
    document : SystemDocument
        Document of the system.
    settings : triboworkflow.core.settings.Settings
        Settings of the run: the slabs find the document.

    Returns
    -------
    ShearInputs

    Raises
    ------
    PrerequisiteError
        If the document, the interpolated PES or the cell are missing.

    Notes
    -----
    To remove: the document is found with ``mid`` and ``miller``, as
    ``FT_StartShearStrength.get_pes`` finds it (``run.py:1552-1553``), which read it with a
    ``Navigator`` outside the compatibility with the database (A47).
    """
    params, _, _ = match_parameters(settings)
    store = LegacyStore(sdir, document, COLLECTION)
    field_, _ = store.query_db(COLLECTION, fltr=set_filter({'mid': '_'.join(params['mid']),
                                                            'miller': params['miller']}))
    if field_ is None:
        raise PrerequisiteError(f'{sdir.name}: the system has no document yet: run match first')
    pes_jm2 = read_one_entry(field_, list(PES_JM2))
    if not pes_jm2:
        raise PrerequisiteError(f'{sdir.name}: entries missing: {".".join(PES_JM2)}; run '
                                'plotpes first')
    cell = read_one_entry(field_, list(CELL))
    if not cell:
        raise PrerequisiteError(f'{sdir.name}: entries missing: {".".join(CELL)}; run match '
                                'first')
    ids = set_filter({key: field_.get(key) for key in ID_KEYS}, FILTER_KEYS)
    return ShearInputs(ids=ids, pes_jm2=pes_jm2, cell=cell)


def shear_provenance(inputs, params):
    """Provenance that the entries of the step have when computed now.

    Parameters
    ----------
    inputs : ShearInputs
    params : ShearParams
        ``[interface.shear]``.

    Returns
    -------
    Provenance
        The parameters of the step, the seed and the content of the entries read, as given
        to the calculation.
    """
    parameters = {'method': METHOD, 'delta': params.delta, 'seed': params.seed,
                  'mep': mep_arguments(params), 'replicate_of': list(REPLICATE_OF),
                  'density': DENSITY, 'tol': PES_TOL, 'optimizations': list(OPTIMIZATIONS)}
    return Provenance.build('shear', parameters=parameters, inputs=inputs.read())


# --- analysis -------------------------------------------------------------------------

def compute_shear(inputs, params):
    """The three strings and the shear strength along each.

    Parameters
    ----------
    inputs : ShearInputs
    params : ShearParams
        ``[interface.shear]``.

    Returns
    -------
    ShearResult

    Notes
    -----
    To remove: ``FT_CalcShearStrength.run_task`` (``shearstr.py:109-130``), with the
    parameters of the string method passed to it (A94, DEV-47) and the seed set before each
    string (DEV-46); the state of the global generator of NumPy is put back at the end. What
    the vendored code prints goes to the log, at debug level.
    """
    cell = np.array(inputs.cell)
    mep_params = mep_arguments(params)
    data, seconds, profiles = {}, {}, {}
    printed = io.StringIO()
    state = np.random.get_state()
    try:
        with contextlib.redirect_stdout(printed):
            start = time.perf_counter()
            solver = PES(np.array(inputs.pes_jm2), cell)
            solver.make_pes(replicate_of=REPLICATE_OF, density=DENSITY, tol=PES_TOL)
            rbf = solver._rbf
            seconds['interpolation'] = time.perf_counter() - start
            logger.info('interpolation of the PES: %.1f s, %d centres', seconds['interpolation'],
                        len(rbf.y))
            for opt in OPTIMIZATIONS:
                start = time.perf_counter()
                np.random.seed(params.seed)
                string = MEP(rbf=rbf, cell=cell)
                string.get_mep(method=METHOD, optimization=opt, **mep_params)
                _, table = string.get_shear_strenght(delta=params.delta, npts=mep_params['npts'],
                                                     optimization=opt)
                data[opt] = jsanitize(string.mep_data)
                # The path after the call, at which the table was evaluated (A97).
                profiles[opt] = np.column_stack((string.mep, table))
                seconds[opt] = time.perf_counter() - start
                logger.info('string %s: %d steps, %.1f s', opt,
                            data[opt][f'along_{opt}']['nstep'] + 1, seconds[opt])
    finally:
        np.random.set_state(state)
        if printed.getvalue():
            logger.debug('output of the vendored code:\n%s', printed.getvalue())
    return ShearResult(data=data, points=len(inputs.pes_jm2), centres=len(rbf.y),
                       seconds=seconds, profiles=profiles, solver=solver)


def write_shear(store, inputs, result):
    """Write the entries of the three strings into the document.

    Parameters
    ----------
    store : LegacyStore
        On the document of the system, which is changed and not saved.
    inputs : ShearInputs
    result : ShearResult

    Notes
    -----
    To remove: ``FT_CalcShearStrength.run_task`` (``shearstr.py:130-142``): one write for each
    string, which merges the key ``along_<string>`` into the entry. The document is found with
    ``mid`` and ``miller`` only, as there (``shearstr.py:132-133``).
    """
    fltr = set_filter({'mid': inputs.ids['mid'], 'miller': inputs.ids['miller']})
    for opt in OPTIMIZATIONS:
        store.update_db(result.data[opt], COLLECTION, entry=ENTRY, fltr=fltr)


def output_path(sdir):
    """Text file of the step: ``shear/shear.output`` in the folder of the system."""
    return sdir.joinpath(*OUTPUT)


def figure_files(sdir, name):
    """The two figures of the step, PNG and PDF of each, in ``shear/`` of the folder of the system.

    ``mep_<name>``: the map of the PES with the three paths; ``mep_profile_<name>``: Delta E
    and the shear stress along them.
    """
    folder = sdir.joinpath(*OUTPUT[:-1])
    return [*figure_paths(folder / f'mep_{name}.png'),
            *figure_paths(folder / f'mep_profile_{name}.png')]


def draw_paths(result, files, stamp=None):
    """Draw the two figures of the paths into ``files`` (``figure_files``, or their partials).

    ``stamp``, the ``outputs.analysis_stamp`` of the analysis, goes in the metadata of the
    files.

    Notes
    -----
    To remove: TribChem's ``PES.plot(mep=...)`` (``physics/tribology/pes.py:326-327``) drew one
    path on the map; the step drew none.
    """
    matplotlib.use('Agg')
    paths = {opt: result.profiles[opt][:, :2] for opt in OPTIMIZATIONS}
    pes_figure(result.solver, files[0], files[1], paths=paths, stamp=stamp)
    profile_figure({opt: result.profiles[opt] for opt in OPTIMIZATIONS}, files[2], files[3],
                   stamp)


def shear_output(system, params, result, now=None, stamp=None):
    """Text of ``shear.output``: parameters, shear strengths and the profiles along the paths.

    Parameters
    ----------
    system : str
        Name of the system.
    params : ShearParams
        ``[interface.shear]``.
    result : ShearResult
    now : datetime.datetime or None, optional
        Time written in the header; the current local time by default.
    stamp : str or None, optional
        ``outputs.analysis_stamp`` of the analysis, on a line of the header.

    Returns
    -------
    str
        Lines starting with ``#`` describe the parameters, the units and the columns; the
        others are the table of the shear strengths and, for each string, its profile, one
        row per point of the path, readable with ``numpy.loadtxt`` once split.
    """
    mep = mep_arguments(params)
    lines = [*header('Shear strength of the interface', system, 'shear', now, stamp),
             '#',
             '# Parameters',
             f'#   [interface.shear]      delta = {params.delta!r}, seed = {params.seed!r}',
             '#   [interface.shear.mep]  ' + ', '.join(f'{k} = {v!r}' for k, v in mep.items()),
             f'#   string method {METHOD}; PES pes.energy.pes_jm2 of {result.points} points, '
             f'interpolated with radial basis functions on {result.centres} centres (the points '
             f'replicated {REPLICATE_OF[0]} x {REPLICATE_OF[1]})',
             '#',
             '# Shear strength along each string, as in shearstr.along_<string> of system.json',
             f'# {"string":<8} {"shear_strength_GPa":>20} {"nstep":>8} {"tol":>12}']
    for opt in OPTIMIZATIONS:
        entry = result.data[opt][f'along_{opt}']
        lines.append(f'  {opt:<8} {entry["shear_strength_GPa"]:>20.10f} {entry["nstep"]:>8d} '
                     f'{entry["tol"]:>12.4e}')
    lines += ['#',
              '# Profile along the minimum energy path of each string, one row per point:',
              '#   x, y       position of the point in the plane of the interface, angstrom',
              '#   lxy        length along the path from its first point, angstrom',
              '#   dvx, dvy   derivatives of the PES along x and y, J/m^2 per angstrom',
              '#   vz         integral of the force along the path, J/m^2, minimum at zero',
              '#   ve         energy of the PES at the point, J/m^2, minimum at zero',
              '#   force      force along the path (minus the derivative of the PES along its',
              '#              tangent), J/m^2 per angstrom; the shear strength is 10 times its',
              '#              largest absolute value, in GPa',
              '# The keys lxy, dvx, dvy, vz, ve and force of shearstr.along_<string> in',
              '# system.json are the first six rows of the profile, columns lxy to force: lxy',
              '# holds the six values of the first point, dvx those of the second, and so on.']
    for opt in OPTIMIZATIONS:
        profile = result.profiles[opt]
        lines += ['#', f'# Profile along {opt}: {len(profile)} points',
                  '#' + ' '.join(f'{name:>17}' for name in PROFILE_COLUMNS)]
        lines += [' ' + ' '.join(f'{value: .10e}' for value in row) for row in profile]
    return '\n'.join(lines) + '\n'


def describe(inputs, params, result):
    """Lines of the message of the step: the shear strengths, the times and the parameters.

    Parameters
    ----------
    inputs : ShearInputs
    params : ShearParams
    result : ShearResult

    Returns
    -------
    list of str
        The shear strength along each string; the points of the PES and the time of each
        part; and, if there are any, the parameters other than the recommended ones.
    """
    strengths = ', '.join(
        f'along {opt} {result.data[opt][f"along_{opt}"]["shear_strength_GPa"]:.4f} GPa'
        for opt in OPTIMIZATIONS)
    times = ', '.join(
        f'{opt} {result.seconds[opt]:.1f} s '
        f'({result.data[opt][f"along_{opt}"]["nstep"] + 1} steps)'
        for opt in OPTIMIZATIONS)
    lines = [f'shear strength {strengths}',
             f'PES of {result.points} points, {result.centres} centres: interpolation '
             f'{result.seconds["interpolation"]:.1f} s; strings {times}']
    told = not_recommended(params)
    if told:
        lines.append('parameters other than the recommended ones: ' + ', '.join(told))
    return lines


# --- step -----------------------------------------------------------------------------

def _load_run(run_dir):
    # Imported here: the command line imports the registry of the steps.
    from triboworkflow.cli.run import load_run

    return load_run(run_dir)


class ShearStep(Step):
    """The shear strength along three paths of minimum energy, no calculation.

    Never analysed by the orchestrator (``manual_analysis``), but in a job of its own or by
    hand: the string method takes long and the interpolation needs memory.
    """

    name = 'shear'
    manual_analysis = True

    def done(self, run_dir, system):
        """Whether the entries are done (``entries_done``) and ``shear.output`` and the
        figures of the paths were written by the analysis saved (``outputs.written_by``)."""
        current = self._current(run_dir, system, upstream_done)
        return current is not None and written_by(*self._exports(run_dir, system, *current))

    def entries_done(self, run_dir, system):
        """Whether the three entries are written and current.

        Nothing is computed: the entries read are compared with those recorded in the
        provenance. ``plotpes`` and the steps it reads from have to be done too: the
        provenance holds the content of the entries read, not whether they are still
        current. Their entries only, not their files (``upstream_entries_done``).
        """
        return self._current(run_dir, system, upstream_entries_done) is not None

    def exports(self, run_dir, system):
        """Files of the analysis saved and the stamp each carries: ``(paths, stamp)``.

        Only for entries done (``entries_done``): the figures are named after the interface
        that the inputs read.
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
        # Before any analysis the inputs are not read: plotpes may not have run.
        if not all(document.get(leaf) for leaf in shear_leaves()):
            return None
        if not upstream(self, run_dir, system):
            return None
        settings = _load_run(run_dir)
        try:
            inputs = shear_inputs(sdir, document, settings)
        except PrerequisiteError:
            return None
        expected = shear_provenance(inputs, settings.interface.shear)
        if not all(is_done(document, leaf, expected, sdir) for leaf in shear_leaves()):
            return None
        return document, inputs

    @staticmethod
    def _exports(run_dir, system, document, inputs):
        sdir = system_dir(run_dir, system)
        paths = [output_path(sdir), *figure_files(sdir, interface_name(inputs.ids))]
        return paths, analysis_stamp(document.provenance(shear_leaves()[0]))

    def expected(self, run_dir, system):
        """The interpolated PES with the provenance of ``plotpes``, the interface with that of ``match``.

        Raises
        ------
        PrerequisiteError
            If the PES is not there to be interpolated, or the steps behind ``plotpes``
            are not done or not current.
        """
        step = RelaxedPlotPesStep()
        try:
            pes_provenance = step.provenance(run_dir, system)
        except PrerequisiteError as error:
            raise PrerequisiteError(f'{error}; shear reads {".".join(PES_JM2)}, which plotpes '
                                    'writes from pes') from None
        check_upstream(step, run_dir, system)
        return {PES_JM2: pes_provenance, INTERFACE: match_provenance(_load_run(run_dir))}

    def provenance(self, run_dir, system):
        """Provenance expected now for the entries that the step writes."""
        sdir = system_dir(run_dir, system)
        settings = _load_run(run_dir)
        return shear_provenance(shear_inputs(sdir, read_system(sdir), settings),
                                settings.interface.shear)

    def analyse(self, run_dir, system, calculations):
        """Compute the shear strength, write the entries with their provenance and ``shear.output``.

        Raises
        ------
        PrerequisiteError
            If the interpolated PES or the interface are missing or not current.
        StoreError
            If the entries read change during the calculation: nothing is written.
        """
        sdir = system_dir(run_dir, system)
        settings = _load_run(run_dir)
        params = settings.interface.shear
        check_prerequisites(read_system(sdir), self.expected(run_dir, system), sdir)
        inputs = shear_inputs(sdir, read_system(sdir), settings)
        provenance = shear_provenance(inputs, params)
        warning = cell_warning(inputs.cell)
        if warning:
            logger.warning('%s: %s', system, warning)
        result = compute_shear(inputs, params)
        output = output_path(sdir)
        figures = figure_files(sdir, interface_name(inputs.ids))
        stamp = analysis_stamp(provenance)
        with staged([output, *figures]) as partials:
            partials[output].write_text(shear_output(system, params, result, stamp=stamp),
                                        encoding='utf-8')
            draw_paths(result, [partials[path] for path in figures], stamp)
            with SystemWriter(sdir) as writer:
                check_prerequisites(writer.document, self.expected(run_dir, system), sdir)
                again = shear_inputs(sdir, writer.document, settings)
                if shear_provenance(again, params) != provenance:
                    raise StoreError(f'{system}/{self.name}: the entries read changed during '
                                     'the analysis; nothing written')
                write_shear(LegacyStore(sdir, writer.document, COLLECTION), inputs, result)
                for leaf in shear_leaves():
                    writer.document.set_provenance(leaf, provenance)
                writer.save()
                place(partials)
        logger.info('%s: profiles along the strings in %s; figures of the paths: %s', system,
                    output, ', '.join(path.name for path in figures))
        for line in describe(inputs, params, result):
            logger.info('%s: %s', system, line)
