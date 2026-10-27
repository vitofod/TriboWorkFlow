# Author: Vito Foderà
# Created: 2026-10-23

"""Schema of the TOML settings (schema version 1): one dataclass per table.

A settings file describes either one calculation (``[structure]``) or the
interface workflow (``[interface]``), together with the pw.x template
(``[namelists]``, ``[[atomic_species]]``, ``[k_points]``) and, optionally,
the execution on HPC (``[execution]``). ``loader.py`` fills the dataclasses
from the TOML tables with dacite and then runs their ``check`` methods.

The namelists are passed to pw.x as given: pw.x applies its own defaults to
the keys left out and judges the values. The checks only cover what the code
needs to work: the layout of the file, the keys the input generator writes
itself, and the conventions of the project (``ibrav = 0``, angstrom, ``scf``
and ``relax`` only, no Hubbard corrections). Values that a step does not use
are ignored, as pw.x ignores keys that do not apply.

Placeholders, i.e. empty strings and ``ecutwfc``/``ecutrho`` equal to 0, are
accepted when a file is loaded and rejected when an input is generated
(``Settings.missing_values``, ``Settings.require_complete``).

Notes
-----
The layout follows Xsorb (``xsorb/settings/input_settings.py`` and
``xsorb/dft_codes/input_settings.py``, CC BY 4.0). Unlike Xsorb, the checks
do not run in ``__post_init__`` but in a separate pass over the filled
dataclasses, so that every message carries the full TOML path of the field,
list indices included.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

NAMELISTS = ('CONTROL', 'SYSTEM', 'ELECTRONS', 'IONS')
# Scheduler names of Xsorb's SCHEDULER_CONFIG (xsorb/io/scheduler.py:13-71).
SCHEDULERS = ('slurm', 'pbs', 'torque', 'lsf', 'sge', 'htcondor', 'condor')
ENGINES = ('qe',)
# Interface steps that run calculations; match, plotpes and shear only analyse.
STEPS_WITH_ENGINE = ('pes', 'adhesion', 'ppes', 'charge')

# &CONTROL keys of a single calculation: in the workflow the code sets them
# for every calculation (own prefix and outdir, calculation from calc_type).
PER_CALCULATION = ('calculation', 'title', 'prefix', 'restart_mode', 'outdir')

_CELL = 'ibrav = 0 always: the cell is written in CELL_PARAMETERS from the structure'
# &SYSTEM keys written by the generator, or excluded by the project.
GENERATED_SYSTEM_KEYS = {
    'ibrav': _CELL, 'celldm': _CELL, 'a': _CELL, 'b': _CELL, 'c': _CELL,
    'cosab': _CELL, 'cosac': _CELL, 'cosbc': _CELL,
    'nat': 'set by the generator from the structure',
    'ntyp': 'set by the generator from [[atomic_species]]',
    'starting_magnetization': 'set it for each species in [[atomic_species]]',
    'lda_plus_u': 'Hubbard corrections are not supported',
    'lda_plus_u_kind': 'Hubbard corrections are not supported',
    'u_projection_type': 'Hubbard corrections are not supported',
}

# Hints for keys that are not part of the other tables.
KEY_HINTS = {
    'london_c6': 'write "london_c6(i)" in [namelists.SYSTEM], i = position of the species '
                 'in [[atomic_species]], counted from 1',
    'london_rvdw': 'write "london_rvdw(i)" in [namelists.SYSTEM], i = position of the species '
                   'in [[atomic_species]], counted from 1',
}


class SettingsError(ValueError):
    """Invalid settings.

    Parameters
    ----------
    path : str
        Dotted TOML path of the field (e.g. ``structure.cell.vectors``); empty
        for errors about the whole file.
    message : str
        What is wrong.
    source : pathlib.Path or None, optional
        Settings file, prefixed to the message when given.
    """

    def __init__(self, path, message, source=None):
        self.path, self.message, self.source = path, message, source
        text = f'{path}: {message}' if path else message
        super().__init__(f'{source}: {text}' if source else text)


def _choice(path, value, allowed, hint=''):
    if value not in allowed:
        raise SettingsError(path, f'{value!r} is not allowed; allowed values: '
                                  f'{", ".join(map(repr, allowed))}{hint}')


def _three(path, value):
    if len(value) != 3:
        raise SettingsError(path, f'give three values, got {value}')


def _positive(path, value):
    if value <= 0:
        raise SettingsError(path, f'must be > 0, got {value}')


def _required(path, value, reason):
    if value is None:
        raise SettingsError(path, f'required {reason}')


def key_name(key):
    """Name of a namelist key without index and case.

    Parameters
    ----------
    key : str
        Key as written in the settings, e.g. ``"London_C6(2)"``.

    Returns
    -------
    str
        The lower-case name without the index, e.g. ``"london_c6"``.
    """
    return key.split('(')[0].strip().lower()


# --- [structure]: one calculation -------------------------------------------

@dataclass
class CellParams:
    """``[structure.cell]``: cell from the structure file or from the settings.

    Attributes
    ----------
    source : {'file', 'config'}
        Where the cell comes from.
    units : {'angstrom'}
        Units of ``vectors``.
    vectors : list of list of float
        Three row vectors, read only with ``source = "config"``.
    """

    source: str = 'file'
    units: str = 'angstrom'
    vectors: list[list[float]] = field(default_factory=list)  # read only with source = "config"

    def check(self, path):
        """Validate the table; errors name ``path``."""
        _choice(f'{path}.source', self.source, ('file', 'config'))
        _choice(f'{path}.units', self.units, ('angstrom',))
        if self.source == 'config' and (len(self.vectors) != 3 or any(len(v) != 3 for v in self.vectors)):
            raise SettingsError(f'{path}.vectors', 'with source = "config" give three '
                                                   'row vectors of three numbers')


@dataclass
class Override:
    """One entry of ``structure.constraints.overrides``.

    Attributes
    ----------
    atoms : list of int
        Atoms, counted from 1 in the order of the structure file.
    if_pos : list of int
        The three pw.x ``if_pos`` flags given to those atoms.
    """

    atoms: list[int]
    if_pos: list[int]

    def check(self, path):
        """Validate the entry; errors name ``path``."""
        for atom in self.atoms:
            if atom < 1:
                raise SettingsError(f'{path}.atoms', f'atoms are counted from 1, got {atom}')
        _three(f'{path}.if_pos', self.if_pos)


@dataclass
class ConstraintsParams:
    """``[structure.constraints]``: precedence override > file > default_if_pos.

    Attributes
    ----------
    read_from_file : bool
        Whether the constraints of the structure file are used.
    default_if_pos : list of int
        Flags of the atoms without override and without constraints in the file.
    overrides : list of Override
        Flags given to selected atoms; an atom may appear in one entry only.
    """

    read_from_file: bool = True
    default_if_pos: list[int] = field(default_factory=lambda: [1, 1, 1])
    overrides: list[Override] = field(default_factory=list)

    def check(self, path):
        """Validate the table; errors name ``path``."""
        _three(f'{path}.default_if_pos', self.default_if_pos)
        seen = {}
        for i, override in enumerate(self.overrides):
            for atom in override.atoms:
                if seen.get(atom, i) != i:
                    raise SettingsError(f'{path}.overrides[{i}].atoms',
                                        f'atom {atom} is already in overrides[{seen[atom]}]')
                seen[atom] = i


@dataclass
class StructureParams:
    """``[structure]``: the structure of the single calculation.

    Attributes
    ----------
    file : str
        Structure file, relative to the folder of the settings file.
    format : str
        ``auto`` or one of the formats of ``core.structures``.
    frame : int
        Structure to read, counted from 0, in files with several.
    xyz_units : {'angstrom'}
        Units of the coordinates of a plain XYZ file.
    preserve_atom_order, preserve_species_labels : bool
        Only ``True``: atom indices and species labels refer to the file.
    cell : CellParams
    constraints : ConstraintsParams
    """

    file: str
    format: str = 'auto'
    frame: int = 0
    xyz_units: str = 'angstrom'
    preserve_atom_order: bool = True
    preserve_species_labels: bool = True
    cell: CellParams = field(default_factory=CellParams)
    constraints: ConstraintsParams = field(default_factory=ConstraintsParams)

    def check(self, path):
        """Validate the table; errors name ``path``."""
        _choice(f'{path}.format', self.format, ('auto', 'cif', 'xyz', 'extxyz', 'pwi', 'poscar', 'json'))
        _choice(f'{path}.xyz_units', self.xyz_units, ('angstrom',))
        if not self.preserve_atom_order:
            raise SettingsError(f'{path}.preserve_atom_order', 'only true is supported: atom '
                                'indices refer to the order of the structure file')
        if not self.preserve_species_labels:
            raise SettingsError(f'{path}.preserve_species_labels', 'only true is supported: '
                                'species labels (e.g. Fe1, Fe2) are kept as in the file')


# --- pw.x input ---------------------------------------------------------------

@dataclass
class Namelists:
    """``[namelists]``: the pw.x namelists, written as given.

    Attributes
    ----------
    CONTROL, SYSTEM, ELECTRONS, IONS : dict
        pw.x keys and values of each namelist; ``IONS`` is written only for
        ``relax``. Arrays are written one key per element, e.g.
        ``"london_c6(1)"``.
    """

    CONTROL: dict[str, Any] = field(default_factory=dict)
    SYSTEM: dict[str, Any] = field(default_factory=dict)
    ELECTRONS: dict[str, Any] = field(default_factory=dict)
    IONS: dict[str, Any] = field(default_factory=dict)  # written only for relax

    def get(self, namelist, name, default=None):
        """Value of a key of a namelist, whatever its case.

        Parameters
        ----------
        namelist : str
            ``CONTROL``, ``SYSTEM``, ``ELECTRONS`` or ``IONS``.
        name : str
            Lower-case key name without index.
        default : optional
            Returned when the key is not given.
        """
        for key, value in getattr(self, namelist).items():
            if key_name(key) == name and '(' not in key:
                return value
        return default

    def check(self, path):
        """Validate the namelists; errors name ``path``."""
        for namelist in NAMELISTS:
            for key, value in getattr(self, namelist).items():
                if not isinstance(value, (str, int, float)):
                    raise SettingsError(f'{path}.{namelist}.{key}', 'a pw.x value is a string, a '
                                        'number or a boolean; for an array write one key per '
                                        'element, e.g. "london_c6(1)"')
        for key in self.SYSTEM:
            name = key_name(key)
            if name in GENERATED_SYSTEM_KEYS or name.startswith('hubbard_'):
                reason = GENERATED_SYSTEM_KEYS.get(name, 'Hubbard corrections are not supported')
                raise SettingsError(f'{path}.SYSTEM.{key}', f'not allowed: {reason}')
        calculation = self.get('CONTROL', 'calculation')
        if calculation is not None:
            _choice(f'{path}.CONTROL.calculation', calculation, ('scf', 'relax'))


@dataclass
class Species:
    """One ``[[atomic_species]]`` entry; the order of the entries is the pw.x order.

    Attributes
    ----------
    label : str
        Species label as in ``ATOMIC_POSITIONS`` (e.g. ``Fe1``).
    mass : float
        Atomic mass in amu.
    pseudopotential : str
        UPF file name.
    starting_magnetization : float or None
        Written as ``starting_magnetization(i)``, i = position of the entry.
    """

    label: str
    mass: float
    pseudopotential: str
    starting_magnetization: float | None = None


@dataclass
class KPoints:
    """``[k_points]``: fixed grid, Gamma point, or grid from a density.

    Attributes
    ----------
    mode : {'automatic', 'gamma', 'kdens'}
    grid, offset : list of int or None
        Monkhorst-Pack grid and offsets, with ``mode = "automatic"``.
    kdens : float or None
        k-points per 1/angstrom of each reciprocal vector, with ``mode = "kdens"``.
    gamma_for_unshifted_1x1x1 : bool
        Whether a final grid 1 1 1 with offset 0 0 0 is written as ``K_POINTS gamma``.
    """

    mode: str
    grid: list[int] | None = None      # mode = "automatic"
    offset: list[int] | None = None    # mode = "automatic"
    kdens: float | None = None         # mode = "kdens"
    gamma_for_unshifted_1x1x1: bool = True

    def check(self, path):
        """Validate the table; errors name ``path``."""
        _choice(f'{path}.mode', self.mode, ('automatic', 'gamma', 'kdens'))
        reason = f'with mode = {self.mode!r}'
        if self.mode == 'automatic':
            _required(f'{path}.grid', self.grid, reason)
            _three(f'{path}.grid', self.grid)
            _required(f'{path}.offset', self.offset, reason)
            _three(f'{path}.offset', self.offset)
        elif self.mode == 'kdens':
            _required(f'{path}.kdens', self.kdens, reason)


@dataclass
class GenerationParams:
    """``[generation]``: the input file of the single calculation.

    Attributes
    ----------
    file_prefix : str
        Name of the generated input file.
    overwrite : bool
        Whether an existing file may be replaced.
    atomic_positions_units, cell_parameters_units : {'angstrom'}
    """

    file_prefix: str = 'input.pwi'
    overwrite: bool = False
    atomic_positions_units: str = 'angstrom'
    cell_parameters_units: str = 'angstrom'

    def check(self, path):
        """Validate the table; errors name ``path``."""
        _choice(f'{path}.atomic_positions_units', self.atomic_positions_units, ('angstrom',))
        _choice(f'{path}.cell_parameters_units', self.cell_parameters_units, ('angstrom',))


# --- execution on HPC -----------------------------------------------------------

@dataclass
class ExecutionParams:
    """``[execution]``: scheduler, jobscript and groups of calculations.

    Attributes
    ----------
    scheduler : str
        One of ``SCHEDULERS``.
    jobscript : str
        Jobscript copied for every group of jobs, with the launch line of the
        worker added; relative to the folder of the settings file.
    run_command : str
        Command up to the input file, e.g. ``mpirun -np 32 pw.x -nk 4``.
    groups : int
        Jobs (one worker each) over which the calculations of a group are
        spread; never more jobs than calculations are submitted.
    walltime, polling_interval : int
        Seconds. ``walltime`` must match the one of the jobscript.
    max_seconds_margin : int
        Seconds kept free at the end of the walltime. Before each run the
        worker sets ``max_seconds`` to the walltime left minus this margin,
        capped by the ``max_seconds`` of ``[namelists.CONTROL]`` if given,
        and takes no more calculations once nothing is left.
    """

    scheduler: str
    jobscript: str
    run_command: str
    groups: int
    walltime: int
    polling_interval: int
    max_seconds_margin: int = 600

    def check(self, path):
        """Validate the table; errors name ``path``."""
        _choice(f'{path}.scheduler', self.scheduler, SCHEDULERS)
        _positive(f'{path}.groups', self.groups)
        _positive(f'{path}.walltime', self.walltime)
        _positive(f'{path}.polling_interval', self.polling_interval)
        _positive(f'{path}.max_seconds_margin', self.max_seconds_margin)
        if self.max_seconds_margin >= self.walltime:
            raise SettingsError(f'{path}.max_seconds_margin',
                                f'must be < walltime ({self.walltime}), '
                                f'got {self.max_seconds_margin}')


# --- interface workflow -----------------------------------------------------------

@dataclass
class SlabParams:
    """One ``[[interface.slabs]]`` entry: a slab and its identifiers.

    Attributes
    ----------
    file : str
        Slab structure, relative to the folder of the settings file.
    mid : str
        Material id; the interface id is ``mid1_mid2``.
    miller : list of int
        Miller indices of the surface.
    formula, name : str or None
        Further identifiers (``name`` tells apart homonyms).
    bulk_modulus : float or None
        Bulk modulus of the material in GPa, used with ``set_bm = true``.
    """

    file: str
    mid: str
    miller: list[int]
    formula: str | None = None
    name: str | None = None
    bulk_modulus: float | None = None  # used with set_bm = true

    def check(self, path):
        """Validate the entry; errors name ``path``."""
        _three(f'{path}.miller', self.miller)


@dataclass
class InterParams:
    """``[interface.inter_params]``: parameters of the interface matching.

    Missing keys take the defaults of the matching (``given`` returns only the
    keys set here); the matching itself checks the values.

    Attributes
    ----------
    max_area : float or None
        Largest area of the interface cell.
    interface_distance : float or None
        Distance between the two slabs.
    max_angle_tol, max_length_tol, max_area_ratio_tol : float or None
        Tolerances of the lattice match.
    strain_weight_1, strain_weight_2 : float or None
        Weights of the strain of the two slabs.
    best_match : str or None
        Criterion to choose among the matches.

    Notes
    -----
    To remove: these are the keys of TribChem's ``inter_params``; the defaults
    are those of ``highput/firetasks/defaults.json``, filled by the vendored
    ``read_default_params`` as ``FT_StartInterfaceMatch`` does, and the values
    are checked by the vendored ``InterfaceMatcher``.
    """

    max_area: float | None = None
    interface_distance: float | None = None
    max_angle_tol: float | None = None
    max_length_tol: float | None = None
    max_area_ratio_tol: float | None = None
    strain_weight_1: float | None = None
    strain_weight_2: float | None = None
    best_match: str | None = None

    def given(self):
        """The keys set in the settings.

        Returns
        -------
        dict
            ``{key: value}`` of the parameters that are not None.
        """
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class InterfaceParams:
    """``[interface]``: inputs of the interface workflow.

    Attributes
    ----------
    slabs : list of SlabParams
        Bottom slab, then top slab.
    set_bm : bool
        Whether the strain weights are the bulk moduli of the two materials.
    engine : str
        Engine of the steps that run calculations.
    engines : dict
        Engine per step, overriding ``engine``.
    inter_params : InterParams
    """

    slabs: list[SlabParams]
    set_bm: bool = True
    engine: str = 'qe'
    engines: dict[str, str] = field(default_factory=dict)
    inter_params: InterParams = field(default_factory=InterParams)

    def engine_for(self, step):
        """Engine of a step that runs calculations.

        Parameters
        ----------
        step : str
            One of ``STEPS_WITH_ENGINE``.

        Returns
        -------
        str
        """
        return self.engines.get(step, self.engine)

    def check(self, path):
        """Validate the table; errors name ``path``."""
        if len(self.slabs) != 2:
            raise SettingsError(f'{path}.slabs', f'give two slabs, bottom then top; got {len(self.slabs)}')
        later = '; MLIP and CP2K engines are not available yet'
        _choice(f'{path}.engine', self.engine, ENGINES, later)
        for step, engine in self.engines.items():
            _choice(f'{path}.engines', step, STEPS_WITH_ENGINE)
            _choice(f'{path}.engines.{step}', engine, ENGINES, later)
        # With set_bm the strain weights are the bulk moduli of the two
        # materials, which replace those of inter_params.
        if self.set_bm:
            for i, slab in enumerate(self.slabs):
                _required(f'{path}.slabs[{i}].bulk_modulus', slab.bulk_modulus,
                          'with set_bm = true (the strain weights are the bulk moduli)')


# --- whole file ---------------------------------------------------------------------

@dataclass
class Settings:
    """The whole settings file.

    Attributes
    ----------
    schema_version : int
        Must equal ``SCHEMA_VERSION``.
    atomic_species : list of Species
    k_points : KPoints
    namelists : Namelists
    structure : StructureParams or None
        Given for a single calculation.
    interface : InterfaceParams or None
        Given for the interface workflow.
    generation : GenerationParams or None
        Used with ``structure``.
    execution : ExecutionParams or None
    source : pathlib.Path or None
        File the settings were read from, set by the loader.
    """

    schema_version: int
    atomic_species: list[Species]
    k_points: KPoints
    namelists: Namelists = field(default_factory=Namelists)
    structure: StructureParams | None = None
    interface: InterfaceParams | None = None
    generation: GenerationParams | None = None  # used with [structure]
    execution: ExecutionParams | None = None
    # File the settings were read from (set by the loader).
    source: Path | None = field(default=None, init=False, repr=False, compare=False)

    def check(self, path=''):
        """Validate the checks that involve several tables."""
        if self.schema_version != SCHEMA_VERSION:
            raise SettingsError('schema_version', f'unsupported value {self.schema_version}; '
                                f'supported: {SCHEMA_VERSION}')
        if (self.structure is None) == (self.interface is None):
            raise SettingsError('', 'give either [structure] (one calculation) '
                                    'or [interface] (interface workflow)')
        if self.interface is not None:
            for key in self.namelists.CONTROL:
                if key_name(key) in PER_CALCULATION:
                    raise SettingsError(f'namelists.CONTROL.{key}', 'not allowed in a workflow: '
                                        'it is set for every calculation')
        if not self.atomic_species:
            raise SettingsError('atomic_species', 'give at least one species')
        seen = {}
        for i, species in enumerate(self.atomic_species):
            if species.label and species.label in seen:
                raise SettingsError(f'atomic_species[{i}].label', f'{species.label!r} is already '
                                    f'atomic_species[{seen[species.label]}]')
            seen[species.label] = i

    def missing_values(self):
        """TOML paths of the placeholders.

        Returns
        -------
        list of str
            Paths of the empty strings and of ``ecutwfc``/``ecutrho`` equal to 0.
        """
        missing = []
        for path, node in walk(self, children_first=False):
            for f in fields(node):
                if f.init and getattr(node, f.name) == '':
                    missing.append(join(path, f.name))
        for namelist in NAMELISTS:
            for key, value in getattr(self.namelists, namelist).items():
                cutoff = key_name(key) in ('ecutwfc', 'ecutrho') and not isinstance(value, (str, bool))
                if value == '' or (cutoff and value == 0):
                    missing.append(f'namelists.{namelist}.{key}')
        return missing

    def require_complete(self):
        """Check, before generating an input, that no placeholder is left.

        Raises
        ------
        SettingsError
            Listing every placeholder.
        """
        missing = self.missing_values()
        if missing:
            raise SettingsError('', 'fill in before generating an input: ' + ', '.join(missing),
                                self.source)


def join(path, name):
    """Dotted TOML path of ``name`` inside ``path``."""
    return f'{path}.{name}' if path else name


def walk(obj, path='', children_first=True):
    """Every dataclass inside a dataclass, with its TOML path.

    Parameters
    ----------
    obj : dataclass instance
    path : str, optional
        TOML path of ``obj``.
    children_first : bool, optional
        Whether inner tables come before the table that holds them.

    Yields
    ------
    tuple of (str, dataclass instance)
        ``(path, node)`` for ``obj`` and every dataclass inside it, list items
        included.
    """
    if not children_first:
        yield path, obj
    for f in fields(obj):
        if not f.init:
            continue
        value, child = getattr(obj, f.name), join(path, f.name)
        if is_dataclass(value):
            yield from walk(value, child, children_first)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if is_dataclass(item):
                    yield from walk(item, f'{child}[{i}]', children_first)
    if children_first:
        yield path, obj
