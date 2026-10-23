"""Schema of the TOML settings (schema version 1): one dataclass per table.

The layout follows Xsorb (``xsorb/settings/input_settings.py`` and
``xsorb/dft_codes/input_settings.py``, CC BY 4.0): dacite fills the
dataclasses from the TOML tables (``loader.py``). Unlike Xsorb, the checks
do not run in ``__post_init__`` but in a separate pass over the filled
dataclasses (``check`` methods, called by ``loader.validate``), so that every
message carries the full TOML path of the field, list indices included.

A file describes either one calculation (``[structure]``) or the interface
workflow (``[interface]``). The namelists are passed to ``pw.x`` as given:
``pw.x`` applies its own defaults to the keys left out and judges the
values. The checks only cover what the code needs to work: the layout of
the file, the keys the generator writes itself, and the conventions of the
project (``ibrav = 0``, angstrom, ``scf`` and ``relax`` only, no Hubbard).
Values that a step does not use are ignored, as ``pw.x`` ignores keys that
do not apply.

Placeholders, i.e. empty strings and ``ecutwfc``/``ecutrho`` equal to 0, are
accepted when a file is loaded and rejected when an input is generated
(``Settings.missing_values``, ``Settings.require_complete``).
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
    """Invalid settings; ``path`` is the dotted TOML path of the field."""

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
    """Name of a namelist key without index and case: "London_C6(2)" -> "london_c6"."""
    return key.split('(')[0].strip().lower()


# --- [structure]: one calculation -------------------------------------------

@dataclass
class CellParams:
    """``[structure.cell]``: cell from the structure file or from the settings."""

    source: str = 'file'
    units: str = 'angstrom'
    vectors: list[list[float]] = field(default_factory=list)  # read only with source = "config"

    def check(self, path):
        _choice(f'{path}.source', self.source, ('file', 'config'))
        _choice(f'{path}.units', self.units, ('angstrom',))
        if self.source == 'config' and (len(self.vectors) != 3 or any(len(v) != 3 for v in self.vectors)):
            raise SettingsError(f'{path}.vectors', 'with source = "config" give three '
                                                   'row vectors of three numbers')


@dataclass
class Override:
    """One entry of ``structure.constraints.overrides``; atoms counted from 1."""

    atoms: list[int]
    if_pos: list[int]

    def check(self, path):
        for atom in self.atoms:
            if atom < 1:
                raise SettingsError(f'{path}.atoms', f'atoms are counted from 1, got {atom}')
        _three(f'{path}.if_pos', self.if_pos)


@dataclass
class ConstraintsParams:
    """``[structure.constraints]``: precedence override > file > default_if_pos."""

    read_from_file: bool = True
    default_if_pos: list[int] = field(default_factory=lambda: [1, 1, 1])
    overrides: list[Override] = field(default_factory=list)

    def check(self, path):
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
    """``[structure]``: the structure of the single calculation."""

    file: str
    format: str = 'auto'
    frame: int = 0
    xyz_units: str = 'angstrom'
    preserve_atom_order: bool = True
    preserve_species_labels: bool = True
    cell: CellParams = field(default_factory=CellParams)
    constraints: ConstraintsParams = field(default_factory=ConstraintsParams)

    def check(self, path):
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
    """``[namelists]``: the pw.x namelists, written as given."""

    CONTROL: dict[str, Any] = field(default_factory=dict)
    SYSTEM: dict[str, Any] = field(default_factory=dict)
    ELECTRONS: dict[str, Any] = field(default_factory=dict)
    IONS: dict[str, Any] = field(default_factory=dict)  # written only for relax

    def get(self, namelist, name, default=None):
        """Value of a key of ``namelist``, whatever its case (no index)."""
        for key, value in getattr(self, namelist).items():
            if key_name(key) == name and '(' not in key:
                return value
        return default

    def check(self, path):
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
    """One ``[[atomic_species]]`` entry; the order of the entries is the pw.x order."""

    label: str
    mass: float
    pseudopotential: str
    starting_magnetization: float | None = None


@dataclass
class KPoints:
    """``[k_points]``: fixed grid, Gamma point, or grid from a density."""

    mode: str
    grid: list[int] | None = None      # mode = "automatic"
    offset: list[int] | None = None    # mode = "automatic"
    kdens: float | None = None         # mode = "kdens"
    # A final grid 1 1 1 with offset 0 0 0 is written as K_POINTS gamma.
    gamma_for_unshifted_1x1x1: bool = True

    def check(self, path):
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
    """``[generation]``: the input file of the single calculation."""

    file_prefix: str = 'input.pwi'
    overwrite: bool = False
    atomic_positions_units: str = 'angstrom'
    cell_parameters_units: str = 'angstrom'

    def check(self, path):
        _choice(f'{path}.atomic_positions_units', self.atomic_positions_units, ('angstrom',))
        _choice(f'{path}.cell_parameters_units', self.cell_parameters_units, ('angstrom',))


# --- execution on HPC -----------------------------------------------------------

@dataclass
class ExecutionParams:
    """``[execution]``: scheduler, jobscript and groups of calculations."""

    scheduler: str
    jobscript: str
    run_command: str
    groups: int
    walltime: int
    polling_interval: int

    def check(self, path):
        _choice(f'{path}.scheduler', self.scheduler, SCHEDULERS)
        _positive(f'{path}.groups', self.groups)
        _positive(f'{path}.walltime', self.walltime)
        _positive(f'{path}.polling_interval', self.polling_interval)


# --- interface workflow -----------------------------------------------------------

@dataclass
class SlabParams:
    """One ``[[interface.slabs]]`` entry: the identifiers TribChem uses for a slab."""

    file: str
    mid: str
    miller: list[int]
    formula: str | None = None
    name: str | None = None
    bulk_modulus: float | None = None  # used with set_bm = true

    def check(self, path):
        _three(f'{path}.miller', self.miller)


@dataclass
class InterParams:
    """``[interface.inter_params]``: the keys of TribChem's ``inter_params``.

    Missing keys take TribChem's defaults (``highput/firetasks/defaults.json``),
    filled by ``read_default_params`` as in ``FT_StartInterfaceMatch``; the
    values are checked by TribChem's ``InterfaceMatcher``.
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
        """The keys set in the settings, as the ``inter_params`` dict of TribChem."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class InterfaceParams:
    """``[interface]``: inputs of the interface workflow."""

    slabs: list[SlabParams]
    set_bm: bool = True
    engine: str = 'qe'
    engines: dict[str, str] = field(default_factory=dict)
    inter_params: InterParams = field(default_factory=InterParams)

    def engine_for(self, step):
        """Engine of a step that runs calculations."""
        return self.engines.get(step, self.engine)

    def check(self, path):
        if len(self.slabs) != 2:
            raise SettingsError(f'{path}.slabs', f'give two slabs, bottom then top; got {len(self.slabs)}')
        later = '; MLIP and CP2K engines are not available yet'
        _choice(f'{path}.engine', self.engine, ENGINES, later)
        for step, engine in self.engines.items():
            _choice(f'{path}.engines', step, STEPS_WITH_ENGINE)
            _choice(f'{path}.engines.{step}', engine, ENGINES, later)
        # With set_bm the strain weights are the bulk moduli of the two
        # materials (FT_StartInterfaceMatch.set_bulk_modulus).
        if self.set_bm:
            for i, slab in enumerate(self.slabs):
                _required(f'{path}.slabs[{i}].bulk_modulus', slab.bulk_modulus,
                          'with set_bm = true (the strain weights are the bulk moduli)')


# --- whole file ---------------------------------------------------------------------

@dataclass
class Settings:
    """The whole settings file."""

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
        """TOML paths of the placeholders: empty strings, ecutwfc/ecutrho equal to 0."""
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
        """Raise SettingsError if a placeholder is left: before generating inputs."""
        missing = self.missing_values()
        if missing:
            raise SettingsError('', 'fill in before generating an input: ' + ', '.join(missing),
                                self.source)


def join(path, name):
    return f'{path}.{name}' if path else name


def walk(obj, path='', children_first=True):
    """Yield ``(path, dataclass)`` for ``obj`` and every dataclass inside it."""
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
