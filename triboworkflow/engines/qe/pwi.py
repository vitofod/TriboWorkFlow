# Author: Vito Foderà
# Created: 2026-11-02

"""pw.x input files.

``pw_input`` writes the input of one ``scf`` or ``relax`` calculation from a
structure, the species and the namelists of the settings, the ``K_POINTS``
card and the constraints of the atoms. The file holds, in this order:

- ``&CONTROL``, ``&SYSTEM``, ``&ELECTRONS`` and, for ``relax`` only,
  ``&IONS``. The keys of the settings are written as given: spelling, case
  and order are kept, keys left out take the pw.x defaults, and no value is
  checked. The keys set by the caller (``control``: in the workflows
  ``calculation``, ``prefix`` and ``outdir``) open ``&CONTROL``; ``ibrav =
  0``, ``nat`` and ``ntyp`` open ``&SYSTEM``, and the
  ``starting_magnetization(i)`` of the species close it;
- ``ATOMIC_SPECIES``: every species of the settings, in their order, with the
  mass and the UPF file given there, whether the structure has atoms of it or
  not, so that the index ``i`` of ``starting_magnetization(i)`` or
  ``london_c6(i)`` names the same species in every calculation;
- ``K_POINTS``: ``automatic`` with grid and offsets, or ``gamma``
  (``kpoints_card``: from ``[k_points]``, with ``kdens`` the grid of the
  workflows, ``mesh_from_density``; ``vasp_grid``: from the k-points of the
  workflow code);
- ``CELL_PARAMETERS angstrom``: the cell of the structure;
- ``ATOMIC_POSITIONS angstrom``: QE label and cartesian position of every
  atom, in the order of the structure, with the three ``if_pos`` flags only
  for the atoms that have at least one fixed component.

Two changes are made to the settings, and returned in ``PwInput.changes``: in
an ``scf`` calculation ``&IONS`` is not written, and a ``nstep`` of
``&CONTROL`` other than 1 is written as 1.

Values are written as Fortran constants: booleans as ``.true.`` and
``.false.``, integers as they are, floats with the shortest text that reads
back to the same float (Python's ``repr``, e.g. ``1e-06``), strings between
apostrophes. Each key is alone on its line, so that the worker can set
``max_seconds`` in the input it runs. What a pw.x input cannot hold is an
error, never changed: a string with an apostrophe or a line break, a key that
is not a name with an optional index, a non-finite float, a label or a UPF
file name with blanks.

Notes
-----
From Xsorb (``xsorb/ase_custom/espresso.py``, ``write_espresso_in_custom``,
Enrico Pedretti, CC BY 4.0, after ASE's ``write_espresso_in``): the order of
the cards, ``ATOMIC_SPECIES`` written in the order given, the labels of the
atoms, ``ibrav = 0``, the formats of the cell (14 decimals) and of the
positions (10 decimals), ``if_pos`` only for the atoms with a fixed
component, and the ``K_POINTS`` lines. Unlike Xsorb:

- the mass of each species comes from the settings, not from ASE's mass of
  the element;
- the namelists are written by this module, in the order and spelling of the
  settings. Xsorb uses ASE's ``Namelist``, which sorts the keys, writes every
  namelist of pw.x (an empty ``&IONS`` in an ``scf`` too) and writes strings
  with Python's ``repr`` (doubled backslashes, double quotes around an
  apostrophe);
- ``&IONS`` only for ``relax``, and ``nstep`` written as 1 in ``scf``;
- ``K_POINTS gamma`` for a grid 1 1 1 without offsets, unless
  ``gamma_for_unshifted_1x1x1`` is false; Xsorb writes it only when no grid
  is given.
"""

import math
import re
from dataclasses import dataclass
from numbers import Integral

from triboworkflow.core.settings.schema import key_name
from triboworkflow.core.structures.constraints import AppliedConstraints, calc_type_if_pos

CALCULATIONS = ('scf', 'relax')
KPOINTS_MODES = ('automatic', 'gamma')

# pw.x calculation of each calc_type of the interface workflow. With the cell
# fixed, the relaxations of the positions are pw.x relax; the other calc_types
# (cell relaxations, follow-ups, nscf, atoms) are not mapped yet.
CALC_TYPES = {'slab_from_scratch': 'scf', 'slab_pos_relax': 'relax', 'interface_z_relax': 'relax'}

# &SYSTEM keys written from the structure and the species.
SYSTEM_KEYS = ('ibrav', 'nat', 'ntyp', 'starting_magnetization')

# A namelist key: a Fortran name with an optional index, e.g. london_c6(2).
_KEY = re.compile(r'[A-Za-z][A-Za-z0-9_]*(\(\s*[0-9]+(\s*,\s*[0-9]+)*\s*\))?')
# Layout of a key line, as ASE's Namelist.to_string.
_INDENT, _KEY_WIDTH = '   ', 16


class PwInputError(ValueError):
    """Data that cannot be written as a pw.x input."""


@dataclass(frozen=True)
class KPointsCard:
    """The ``K_POINTS`` card.

    Attributes
    ----------
    mode : {'automatic', 'gamma'}
    grid, offset : tuple of int or None
        Monkhorst-Pack grid and offsets, with ``mode = 'automatic'`` only.

    Raises
    ------
    PwInputError
        For another mode, or a grid or offsets that are not three integers.
    """

    mode: str
    grid: tuple | None = None
    offset: tuple | None = None

    def __post_init__(self):
        if self.mode not in KPOINTS_MODES:
            raise PwInputError(f'K_POINTS {self.mode!r}: only {", ".join(KPOINTS_MODES)}')
        if self.mode == 'gamma':
            if self.grid is not None or self.offset is not None:
                raise PwInputError('K_POINTS gamma has no grid and no offsets')
            return
        for name in ('grid', 'offset'):
            value = getattr(self, name)
            if value is None or len(value) != 3 or not all(_integer(v) for v in value):
                raise PwInputError(f'K_POINTS automatic: {name} must be three integers, got {value!r}')
            object.__setattr__(self, name, tuple(int(v) for v in value))

    def lines(self):
        """Lines of the card, without line ends."""
        if self.mode == 'gamma':
            return ('K_POINTS gamma',)
        (k1, k2, k3), (s1, s2, s3) = self.grid, self.offset
        return ('K_POINTS automatic', f'{k1} {k2} {k3}  {s1} {s2} {s3}')


def grid_card(grid, offset, gamma_for_unshifted_1x1x1=True):
    """``K_POINTS`` card of a Monkhorst-Pack grid.

    Parameters
    ----------
    grid, offset : sequence of int
        Three subdivisions and three offsets (0 or 1).
    gamma_for_unshifted_1x1x1 : bool, optional
        Whether the grid 1 1 1 with offsets 0 0 0 is written as ``K_POINTS
        gamma``.

    Returns
    -------
    KPointsCard
    """
    if gamma_for_unshifted_1x1x1 and tuple(grid) == (1, 1, 1) and tuple(offset) == (0, 0, 0):
        return KPointsCard('gamma')
    return KPointsCard('automatic', tuple(grid), tuple(offset))


def kpoints_card(k_points, structure=None, material_type='auto'):
    """``K_POINTS`` card of ``[k_points]``.

    Parameters
    ----------
    k_points : triboworkflow.core.settings.schema.KPoints
        ``automatic``: the grid and offsets given; ``gamma``: ``K_POINTS
        gamma``; ``kdens``: the grid of ``mesh_from_density``.
    structure : pymatgen.core.structure.Structure, optional
        Required with ``mode = 'kdens'``.
    material_type : str, optional
        Passed to ``mesh_from_density`` with ``mode = 'kdens'``.

    Returns
    -------
    KPointsCard
        A final grid 1 1 1 with offsets 0 0 0 is ``K_POINTS gamma`` when
        ``k_points.gamma_for_unshifted_1x1x1`` is true.

    Raises
    ------
    PwInputError
        For another mode, or ``kdens`` without a structure.
    """
    if k_points.mode == 'gamma':
        return KPointsCard('gamma')
    if k_points.mode == 'automatic':
        grid, offset = k_points.grid, k_points.offset
    elif k_points.mode == 'kdens':
        if structure is None:
            raise PwInputError('k_points.mode = "kdens" needs the structure')
        grid, offset = mesh_from_density(structure, k_points.kdens, material_type)
    else:
        raise PwInputError(f'k_points.mode = {k_points.mode!r}: only automatic, gamma, kdens')
    return grid_card(grid, offset, k_points.gamma_for_unshifted_1x1x1)


def mesh_from_density(structure, kdens, material_type='auto'):
    """Grid of a k-point density, as in the workflows.

    The grid is the one of the vendored ``MeshFromDensity`` with
    ``force_gamma=True``: ``round(|b_i| * kdens)`` subdivisions along each
    reciprocal vector ``b_i``, whose length includes the factor 2π (pymatgen's
    ``reciprocal_lattice``), centred on Gamma. The third subdivision is 1 for
    a slab, and the grid is 1 1 1 for an atom or a molecule.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
    kdens : float
        k-points per 1/angstrom of each reciprocal vector.
    material_type : str, optional
        ``'slab'`` (or ``'interface'``), ``'molecule'`` (or ``'atom'``), or
        any other value, e.g. ``'auto'``, to infer it: a slab when the cell is
        at least 6 angstrom longer along z than the atoms, a bulk otherwise.

    Returns
    -------
    grid : tuple of int
    offset : tuple of int
        ``(0, 0, 0)``: a Gamma-centred grid is unshifted in pw.x.

    Notes
    -----
    To remove: TribChem's ``VaspInputSet.get_vasp_settings`` calls
    ``MeshFromDensity`` with ``material_type='slab'`` for the ``slab_*`` and
    ``interface_*`` calc_types, ``'molecule'`` for ``atom`` and ``'auto'``
    otherwise (``highput/utils/vasp_tools.py:439-460``).
    """
    # Imported here: the vendored module takes seconds to import.
    from triboworkflow.vendored.highput.utils.vasp_tools import MeshFromDensity
    return vasp_grid(MeshFromDensity(structure, kdens, material_type=material_type,
                                     force_gamma=True).get_kpoints())


def vasp_grid(kpoints):
    """pw.x grid of a Gamma-centred automatic pymatgen ``Kpoints``.

    Parameters
    ----------
    kpoints : pymatgen.io.vasp.inputs.Kpoints
        E.g. the k-points of the vendored ``VaspInputSet``, or of
        ``MeshFromDensity`` with ``force_gamma=True``.

    Returns
    -------
    grid : tuple of int
    offset : tuple of int
        ``(0, 0, 0)``.

    Raises
    ------
    PwInputError
        For any other kind of ``Kpoints`` (Monkhorst-Pack, shifted, explicit
        list): they are not converted.
    """
    from pymatgen.io.vasp.inputs import Kpoints
    gamma = (kpoints.style == Kpoints.supported_modes.Gamma and kpoints.num_kpts == 0
             and len(kpoints.kpts) == 1)
    if not gamma or any(kpoints.kpts_shift):
        raise PwInputError(f'k-points {kpoints.style.name} {list(kpoints.kpts)} shifted by '
                           f'{list(kpoints.kpts_shift)}: only an automatic Gamma-centred grid '
                           'without shift is converted')
    return tuple(int(k) for k in kpoints.kpts[0]), (0, 0, 0)


@dataclass(frozen=True)
class PwInput:
    """A pw.x input.

    Attributes
    ----------
    text : str
        Content of the input file.
    calculation : str
        ``scf`` or ``relax``.
    changes : tuple of str
        The changes made to the settings, one sentence each; empty if none.
    namelists : tuple
        The namelists as written: ``(name, ((key, value), ...))`` in order.
    cards : tuple
        The cards as written: ``(name, (line, ...))`` in order, the first
        line being the header (e.g. ``K_POINTS automatic``), without line
        ends.
    """

    text: str
    calculation: str
    changes: tuple = ()
    namelists: tuple = ()
    cards: tuple = ()


def fortran_value(value, where='value'):
    """A value of the settings as a Fortran constant of a namelist.

    Parameters
    ----------
    value : bool, int, float or str
    where : str, optional
        TOML path of the value, for the error messages.

    Returns
    -------
    str

    Raises
    ------
    PwInputError
        For a non-finite float, a string with an apostrophe or a line break,
        and values of any other type.

    Examples
    --------
    >>> [fortran_value(v) for v in (True, 150, 1e-06, 0.3, 'relax')]
    ['.true.', '150', '1e-06', '0.3', "'relax'"]
    """
    if isinstance(value, bool):
        return '.true.' if value else '.false.'
    if _integer(value):
        return str(int(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PwInputError(f'{where}: {value!r} cannot be written in a pw.x input')
        return repr(value)
    if isinstance(value, str):
        if "'" in value or '\n' in value or '\r' in value:
            raise PwInputError(f'{where}: {value!r} holds an apostrophe or a line break, which '
                               'a pw.x string between apostrophes cannot hold')
        return f"'{value}'"
    raise PwInputError(f'{where}: a {type(value).__name__} is not a pw.x value')


def namelist_lines(name, entries, where):
    """Lines of a namelist, from ``&NAME`` to ``/``.

    Parameters
    ----------
    name : str
        Name of the namelist, e.g. ``CONTROL``.
    entries : iterable of (str, object)
        Keys and values, written in this order.
    where : str
        TOML path of the namelist, for the error messages.

    Returns
    -------
    list of str

    Raises
    ------
    PwInputError
        For a key that is not a Fortran name with an optional index, or a
        value that ``fortran_value`` cannot write.
    """
    lines = [f'&{name}\n']
    for key, value in entries:
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise PwInputError(f'{where}.{key}: not a pw.x key (a name with an optional index, '
                               'e.g. "london_c6(1)")')
        lines.append(f'{_INDENT}{key:{_KEY_WIDTH}} = {fortran_value(value, f"{where}.{key}")}\n')
    lines.append('/\n')
    return lines


def pw_input(structure, species, namelists, k_points, if_pos, control=None):
    """The pw.x input of one calculation.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        Cell, atoms in the order to write, QE labels in ``site.label``.
    species : sequence of triboworkflow.core.settings.schema.Species
        ``[[atomic_species]]``, in the order of ``ATOMIC_SPECIES``; it must
        hold every label of the structure.
    namelists : triboworkflow.core.settings.schema.Namelists
        ``[namelists]``, written as given.
    k_points : KPointsCard
    if_pos : AppliedConstraints or sequence of tuple of int
        The three pw.x flags of every atom (0 fixed, 1 free).
    control : dict, optional
        ``&CONTROL`` keys set by the caller, written first; none of them may
        be in ``namelists.CONTROL``. ``calculation`` given here decides the
        kind of calculation, otherwise the one of ``namelists.CONTROL``, and
        ``scf``, the pw.x default, when neither gives it.

    Returns
    -------
    PwInput

    Raises
    ------
    PwInputError
        If the calculation is not ``scf`` or ``relax``, a key is given both
        by the caller (or the generator) and in the settings, a label of the
        structure is not a species, ``if_pos`` does not give three integers
        per atom, or a value cannot be written.
    """
    control = dict(control or {})
    given = {key_name(key): key for key in namelists.CONTROL}
    for key in control:
        if key_name(key) in given:
            raise PwInputError(f'namelists.CONTROL.{given[key_name(key)]}: set for this '
                               'calculation, it cannot also be given in the settings')
    for key in namelists.SYSTEM:
        if key_name(key) in SYSTEM_KEYS:
            raise PwInputError(f'namelists.SYSTEM.{key}: written from the structure and the species')
    calculation = control.get('calculation', namelists.get('CONTROL', 'calculation', 'scf'))
    if calculation not in CALCULATIONS:
        raise PwInputError(f'calculation = {calculation!r}: only {", ".join(CALCULATIONS)}')

    labels = [site.label for site in structure]
    _check_species(species, labels)
    flags = _if_pos(if_pos, len(structure))

    changes = []
    control_entries = list(control.items())
    for key, value in namelists.CONTROL.items():
        if calculation == 'scf' and key_name(key) == 'nstep' and '(' not in key and value != 1:
            changes.append(f'namelists.CONTROL.{key} = {value!r} written as 1: an scf '
                           'calculation makes a single step')
            value = 1
        control_entries.append((key, value))
    system_entries = [('ibrav', 0), ('nat', len(structure)), ('ntyp', len(species))]
    system_entries += list(namelists.SYSTEM.items())
    system_entries += [(f'starting_magnetization({i})', s.starting_magnetization)
                       for i, s in enumerate(species, 1) if s.starting_magnetization is not None]

    written = [('CONTROL', control_entries), ('SYSTEM', system_entries),
               ('ELECTRONS', list(namelists.ELECTRONS.items()))]
    if calculation == 'relax':
        written.append(('IONS', list(namelists.IONS.items())))
    elif namelists.IONS:
        changes.append(f'namelists.IONS ({", ".join(namelists.IONS)}) not written: an scf '
                       'calculation does not move the atoms')

    species_lines = ['ATOMIC_SPECIES']
    for i, s in enumerate(species):
        mass = fortran_value(s.mass, f'atomic_species[{i}].mass')
        species_lines.append(f'{s.label} {mass} {s.pseudopotential}')
    cell_lines = ['CELL_PARAMETERS angstrom']
    cell_lines += [f'{a:.14f} {b:.14f} {c:.14f}' for a, b, c in structure.lattice.matrix]
    position_lines = ['ATOMIC_POSITIONS angstrom']
    for label, (x, y, z), f in zip(labels, structure.cart_coords, flags):
        mask = f' {f[0]} {f[1]} {f[2]}' if 0 in f else ''
        position_lines.append(f'{label} {x:.10f} {y:.10f} {z:.10f}{mask}')
    cards = [('ATOMIC_SPECIES', species_lines), ('K_POINTS', k_points.lines()),
             ('CELL_PARAMETERS', cell_lines), ('ATOMIC_POSITIONS', position_lines)]

    lines = []
    for name, entries in written:
        lines += namelist_lines(name, entries, f'namelists.{name}')
    for _, card in cards:
        lines.append('\n')
        lines += [f'{line}\n' for line in card]
    return PwInput(''.join(lines), calculation, tuple(changes),
                   tuple((name, tuple(entries)) for name, entries in written),
                   tuple((name, tuple(card)) for name, card in cards))


def calculation_for(calc_type):
    """pw.x ``calculation`` of a calc_type.

    Parameters
    ----------
    calc_type : str

    Returns
    -------
    str
        ``scf`` or ``relax``, from ``CALC_TYPES``.

    Raises
    ------
    PwInputError
        For a calc_type not in ``CALC_TYPES``.

    Notes
    -----
    To remove: in TribChem ``slab_from_scratch`` is a static type and
    ``slab_pos_relax`` and ``interface_z_relax`` are relaxations
    (``highput/utils/vasp_tools.py:264-274``) with ISIF 2, i.e. with the cell
    fixed (``highput/utils/vis.json:62-66``).
    """
    if calc_type not in CALC_TYPES:
        raise PwInputError(f'calc_type {calc_type!r} has no pw.x calculation yet; mapped: '
                           f'{", ".join(CALC_TYPES)}')
    return CALC_TYPES[calc_type]


def workflow_input(structure, kpoints, calc_type, settings, prefix, outdir):
    """The pw.x input of a calculation of the workflow.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        The structure prepared by the vendored ``VaspInputSet`` for
        ``calc_type``, with the ``selective_dynamics`` of the calc_type.
    kpoints : pymatgen.io.vasp.inputs.Kpoints
        The k-points of the same ``VaspInputSet`` (``vis.kpoints``).
    calc_type : str
        One of ``CALC_TYPES``.
    settings : triboworkflow.core.settings.Settings
        Species, namelists and ``k_points.gamma_for_unshifted_1x1x1``.
    prefix, outdir : str
        ``prefix`` and ``outdir`` of the calculation.

    Returns
    -------
    PwInput
        With ``calculation`` from the calc_type, the constraints of the
        calc_type (``calc_type_if_pos``) and the grid of ``kpoints``.

    Raises
    ------
    PwInputError
        As ``calculation_for``, ``vasp_grid`` and ``pw_input``.
    """
    calculation = calculation_for(calc_type)
    card = grid_card(*vasp_grid(kpoints), settings.k_points.gamma_for_unshifted_1x1x1)
    control = {'calculation': calculation, 'prefix': prefix, 'outdir': outdir}
    return pw_input(structure, settings.atomic_species, settings.namelists, card,
                    calc_type_if_pos(structure), control)


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def _check_species(species, labels):
    seen = set()
    for i, s in enumerate(species):
        for name in ('label', 'pseudopotential'):
            text = getattr(s, name)
            if not isinstance(text, str) or not text or text.split() != [text]:
                raise PwInputError(f'atomic_species[{i}].{name}: {text!r} cannot be written in '
                                   'ATOMIC_SPECIES (empty, or with blanks)')
        if s.label in seen:
            raise PwInputError(f'atomic_species[{i}].label: {s.label!r} is given twice')
        seen.add(s.label)
    unknown = sorted(set(labels) - seen)
    if unknown:
        raise PwInputError(f'labels of the structure not in atomic_species: {", ".join(unknown)}')


def _if_pos(if_pos, n):
    flags = if_pos.if_pos if isinstance(if_pos, AppliedConstraints) else list(if_pos)
    if len(flags) != n:
        raise PwInputError(f'if_pos: {len(flags)} entries for {n} atoms')
    for i, f in enumerate(flags):
        if len(f) != 3 or not all(_integer(x) for x in f):
            raise PwInputError(f'if_pos of atom {i + 1}: three integers required, got {f!r}')
    return [tuple(int(x) for x in f) for f in flags]
