# Author: Vito Foderà
# Created: 2026-11-03

"""Results of a pw.x calculation, read from its output.

``parse_output`` turns the output of a ``scf`` or ``relax`` calculation into
the results of its record and the state ``parsing``, using the structure of
its input (``structure_in.extxyz``). ``pw_values`` reads the values those
results are made of, as pw.x prints them.

Results
-------
The results have the keys and the units of the task documents that the
workflow code reads, as atomate wrote them for VASP:

==========================  ==================================================================
Key                         Value
==========================  ==================================================================
``nsites``                  number of atoms
``input.structure``         structure of the input, as a pymatgen ``Structure`` dict
``output.structure``        final structure, as a pymatgen ``Structure`` dict, in angstrom
``output.density``          density of the final structure, in g/cm^3
``output.energy``           total energy, in eV
``output.energy_per_atom``  ``output.energy`` divided by ``nsites``, in eV
``output.forces``           total force on every atom, in eV/angstrom; only when printed
``output.stress``           stress tensor, in kbar, with the sign pw.x prints; only when printed
==========================  ==================================================================

The energy is ``Final energy`` in a ``relax``, the last ``!    total
energy`` in a ``scf``: the total energy pw.x prints, smearing term included,
converted with the constants of QE and nothing subtracted or extrapolated.
VASP gave the energy extrapolated to zero smearing: the two are not the same
quantity, and only differences between energies of one energy model are
used. A ``relax`` without ``Final energy`` takes the last ``!`` line; with
both, they must agree to the digits printed.

The final structure has the cell and the atoms of the input (labels,
``if_pos`` and the other site properties, except ``magmom``) and the
positions of the final coordinates in a ``relax``, those of the input in a
``scf``. ``magmom`` is an initial moment of the input: it is not a final
moment, which atomate added to the final structure and pw.x prints in
another form, not read here. The positions are kept as printed, not moved
into the cell. The keys of atomate that VASP alone gave (band gap, band
edges, the description of the pseudopotentials) are not written: the
workflow code reads them as None, as a missing key.

Reading
-------
The output is read only when the states of its last run say that the
calculation is over with the SCF converged and, for a ``relax``, the geometry
converged; otherwise ``parsing`` and the results stay None. ``parsing`` is
``'ok'`` when the results required are there and agree with the input:

- in every calculation the total energy of the last SCF cycle; in a
  ``relax`` also the forces and the final coordinates, closed;
- the number of atoms, their labels and the cell of the header, equal to
  those of the input to the digits printed; in a ``scf`` the positions of
  the header too; in a ``relax`` the labels and the ``if_pos`` flags of the
  final coordinates.

Otherwise ``parsing`` is ``'failed'`` with the reason, and there are no
results.

Identity
--------
What the output tells of the energy model goes to ``observed`` of the record
(``observed_of``), with the keys and the form of ``model.energy_model``, and
the store compares it with the energy model expected
(``core.store.identity_differences``):

- ``version``: the version printed, with as many components as the version
  expected (``observed_version``): an expected ``7.3`` stands for every
  ``7.3.x``, an expected ``7.3.1`` for that one only. The whole version
  printed is ``Engine.version``;
- ``species``: for each species of ``ATOMIC_SPECIES``, in order, its
  ``label``, the name of its UPF file (``pseudopotential``, the last part of
  the path printed, which is kept in ``path``) and its ``md5`` (None when
  not printed).

An output whose identity differs keeps its results and ``parsing = 'ok'``;
it is never reused and its energies are never combined.

Values
------
``pw_values`` reads, from the last run of a pw.x output, the values in the
units pw.x prints them: Ry, bohr and units of ``alat``, without conversion.
Whether the run is over and converged is read by ``status.py``.

The values come from these lines of the last run:

=================================================  ===============================================
Line                                               Value
=================================================  ===============================================
``number of atoms/cell = <nat>``                   number of atoms
``celldm(1)= <alat>``                              ``alat``, in bohr
``crystal axes: (cart. coord. in units of alat)``  the cell, on the three lines ``a(i) = ( ... )``
``PseudoPot. # <i> for <element> read from file:`` UPF file of species ``i`` (next line) and its
                                                   checksum (``MD5 check sum: <md5>``)
``atomic species   valence    mass ...``           label of each species, in their order
``positions (alat units)``                         label and position of every atom
``!    total energy = <E> Ry``                     total energy of the last SCF cycle
``Forces acting on atoms (cartesian axes, Ry/au):`` total force on every atom
``total   stress  (Ry/bohr**3)  (kbar)  P=``       stress tensor (columns in Ry/bohr^3)
``Final energy = <E> Ry``                          energy at the end of the optimization
``Begin final coordinates`` ...                    final positions of a relaxation, in angstrom
``End final coordinates``
=================================================  ===============================================

The energy, the forces and the stress are those of the last SCF cycle of the
run, i.e. printed after its last ``Self-consistent Calculation`` line. The
forces are the total ones, the first block after the heading; with high
verbosity pw.x prints their contributions after it (non-local, ionic, ...,
``Dispersion contribution to forces``), which are not read. The total energy
is the one pw.x prints, including the smearing term (``F = E - TS``); the
internal energy and its contributions are not read.

``Final energy`` is printed when the optimizer stops on its criteria or gives
up (``bfgs converged`` or ``bfgs failed``), not when the ionic steps reach
``nstep``: it does not prove convergence, which the states tell. It is the
same energy as the last ``!    total energy`` line, with ten decimals instead
of eight.

A value that is not a finite number (e.g. ``NaN``, or ``****`` where a
number does not fit its format) is an error, never read as a number. A block
that is begun and not ended is not read: the final coordinates without
``End final coordinates`` give None.

Notes
-----
From Xsorb (``xsorb/ase_custom/espresso.py``, ``parse_pwo_start_custom`` and
``read_espresso_out_custom``, Enrico Pedretti, CC BY 4.0, after ASE's
``read_espresso_out``): the lines read, ``celldm(1)`` instead of ``lattice
parameter (alat)`` (it has more digits), the cell from ``crystal axes`` times
``celldm(1)``, the positions of the header with ASE's
``parse_position_line``, the energy as the second-to-last word of the ``!``
line, the forces on the lines after the heading, the stress from the first
three columns, the final positions with ASE's ``get_atomic_positions``.
Unlike Xsorb:

- only the last run of the file is read, and only its last SCF cycle; Xsorb
  reads every run and every configuration, and drops with a message the
  initial positions repeated by a restart;
- the lines are recognized at their beginning, not anywhere in the line;
- the values stay in the units pw.x prints them; Xsorb converts them with
  the constants of ``ase.io.espresso`` (CODATA 2006) and changes the sign of
  the stress to the convention of ASE;
- ``Final energy``, the species table and the checksums are read; Xsorb
  reads magnetic moments, Fermi energy, k-points and bands, not read here;
- the final positions are read only in angstrom, the unit of the inputs of
  this package, and only from a block that is closed;
- every block must have exactly one line per atom, numbered in order, and
  every number must be finite.

The constants of the conversions (``RY_TO_EV``, ``BOHR_TO_ANGSTROM``,
``RY_KBAR``) are those of QE 7.1, 7.2 and 7.3 (``Modules/constants.f90``,
CODATA 2018), computed as there.

The comparisons with the input allow the rounding of the printed values, as
QE 7.1 and 7.2 print them: six decimals for ``celldm(1)`` and the cell, seven
for the positions in units of ``alat``, eight for the ``!`` line, ten for
``Final energy``. The positions of ``structure_in.extxyz`` have eight
decimals in angstrom (those of the input ten): their rounding is allowed
too.

To remove: TribChem received the documents of atomate's ``VaspDrone``
(``atomate/vasp/drones.py``, ``generate_doc`` and ``process_vasprun``), where
``output.energy`` is pymatgen's ``Vasprun.final_energy``; its interface
steps read ``output.energy``, ``output.structure`` (``@class``, lattice) and
``input.structure``.
"""

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.io.espresso import get_atomic_positions, parse_position_line
from monty.json import jsanitize
from pymatgen.core.structure import Structure

from triboworkflow.core.structures import extxyz_text
from triboworkflow.core.structures.model import IF_POS, MAGMOM

from .status import (_COORDINATES_BEGIN, _COORDINATES_END, _SCF_START, CALCULATIONS,
                     PwOutputError, _last_index, _last_run, _starts, _summary_value, pw_status)

# Constants of QE 7.1-7.3 (Modules/constants.f90).
HARTREE_SI = 4.3597447222071E-18     # J
ELECTRONVOLT_SI = 1.602176634E-19    # J
BOHR_RADIUS_SI = 0.529177210903E-10  # m
RY_TO_EV = HARTREE_SI / ELECTRONVOLT_SI / 2.0             # RYTOEV
BOHR_TO_ANGSTROM = BOHR_RADIUS_SI * 100.0 * 1.0E8         # BOHR_RADIUS_ANGS
RY_KBAR = 10.0 * (HARTREE_SI / BOHR_RADIUS_SI ** 3 / 1.0E+9) / 2.0

_NAT = 'number of atoms/cell'
_NTYP = 'number of atomic types'
_ALAT = 'celldm(1)'
_AXES = 'crystal axes:'
_POSITIONS = 'positions (alat units)'
_MD5 = 'MD5 check sum:'
_TOTAL_ENERGY = '!    total energy'
_FINAL_ENERGY = 'Final energy'
_FORCES = 'Forces acting on atoms'
_STRESS = 'total   stress'
_ATOMIC_POSITIONS = 'ATOMIC_POSITIONS'
_CELL_PARAMETERS = 'CELL_PARAMETERS'

# Half of the last digit printed (QE 7.1, 7.2): celldm(1) and the cell, the
# positions in units of alat, the "!" line, "Final energy"; and of the
# positions of structure_in.extxyz, in angstrom.
_HALF_AXES = 0.5E-6
_HALF_POSITION = 0.5E-7
_HALF_TOTAL_ENERGY = 0.5E-8
_HALF_FINAL_ENERGY = 0.5E-10
_HALF_EXTXYZ = 0.5E-8
# Margin on the bounds above, for the arithmetic of the comparison.
_MARGIN = 1.01

_AXIS = re.compile(r'a\(\s*([123])\)\s*=\s*\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)')
_PSEUDO = re.compile(r'PseudoPot\.\s*#\s*(\d+)\s+for\s+(\S+)\s+read from file:')
_SPECIES_TABLE = re.compile(r'atomic species\s+valence\s+mass\s+pseudopotential')
_FORCE = re.compile(r'atom\s+(\d+)\s+type\s+\d+\s+force\s*=\s*(\S+)\s+(\S+)\s+(\S+)')
_CHECKSUM = re.compile(r'[0-9a-f]{32}')
_VERSION = re.compile(r'[0-9]+(\.[0-9]+)+')


@dataclass(frozen=True)
class PwSpecies:
    """A species as the header of a run prints it.

    Parameters
    ----------
    label : str
        Label of the species, from the species table.
    element : str
        Element named in ``PseudoPot. # <i> for <element>``, as the UPF file
        gives it.
    pseudopotential : str
        UPF file read, as printed: the pseudopotential directory followed by
        the file name of ``ATOMIC_SPECIES``.
    md5 : str or None
        Checksum printed on the ``MD5 check sum:`` line, 32 lower-case hex
        digits; None when the line is missing or holds anything else.
    """

    label: str
    element: str
    pseudopotential: str
    md5: str | None


@dataclass(frozen=True)
class PwValues:
    """Values of the last run of a pw.x output, in the units pw.x prints them.

    Parameters
    ----------
    calculation : str
        ``'scf'`` or ``'relax'``, as given by the caller.
    version : str
        Version of pw.x printed in the header of the run.
    runs : int
        Number of runs in the file; only the last one is read.
    nat : int
        Number of atoms.
    alat : float
        ``celldm(1)``, in bohr.
    axes : tuple of tuple of float
        The three cell vectors (rows), in units of ``alat``.
    labels : tuple of str
        Label of every atom, in the order of the input.
    positions : tuple of tuple of float
        Cartesian positions printed in the header, in units of ``alat``.
    species : tuple of PwSpecies
        In the order of ``ATOMIC_SPECIES``.
    energy : float or None
        Total energy of the last SCF cycle (``!    total energy``), in Ry;
        None if the cycle did not print it.
    final_energy : float or None
        ``Final energy`` of a relaxation, in Ry; None if not printed.
    forces : tuple of tuple of float or None
        Total forces of the last SCF cycle, in Ry/bohr; None if not printed
        (e.g. a ``scf`` without ``tprnfor``).
    stress : tuple of tuple of float or None
        Stress tensor of the last SCF cycle, in Ry/bohr^3, with the sign
        pw.x prints; None if not printed (without ``tstress``).
    final_labels : tuple of str or None
        Labels of the final coordinates of a relaxation; None without a
        closed block.
    final_positions : tuple of tuple of float or None
        Final cartesian positions of a relaxation, in angstrom.
    final_if_pos : tuple or None
        Flags printed after each final position (0 fixed, 1 free), a tuple of
        three integers, or None for an atom printed without flags.
    """

    calculation: str
    version: str
    runs: int
    nat: int
    alat: float
    axes: tuple
    labels: tuple
    positions: tuple
    species: tuple
    energy: float | None = None
    final_energy: float | None = None
    forces: tuple | None = None
    stress: tuple | None = None
    final_labels: tuple | None = None
    final_positions: tuple | None = None
    final_if_pos: tuple | None = None


def _number(text, what):
    """A finite float printed by pw.x, or ``PwOutputError``."""
    try:
        value = float(text)
    except ValueError:
        raise PwOutputError(f'{what}: {text!r} is not a number') from None
    if not math.isfinite(value):
        raise PwOutputError(f'{what}: {text!r} is not a finite number')
    return value


def _finite(values, what):
    """The floats of ``values`` as a tuple, or ``PwOutputError`` if one is not finite."""
    values = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in values):
        raise PwOutputError(f'{what}: {values} holds a value that is not a finite number')
    return values


def _required(value, what):
    if value is None:
        raise PwOutputError(f'no "{what}" in the summary of the last run')
    return value


def _energy(line, what):
    """Energy of a line ``<name> = <value> Ry``, in Ry."""
    words = line.partition('=')[2].split()
    if len(words) != 2 or words[1] != 'Ry':
        raise PwOutputError(f'"{line.strip()}": not "<name> = <value> Ry"')
    return _number(words[0], what)


def _axes(run):
    start = _last_index(run, _AXES)
    if start is None:
        raise PwOutputError(f'no "{_AXES}" line in the last run')
    axes = []
    for i, line in enumerate(run[start + 1:start + 4], 1):
        match = _AXIS.search(line)
        if not match or int(match.group(1)) != i:
            raise PwOutputError(f'line {i} after "{_AXES}" is not "a({i}) = ( x y z )": '
                                f'{line.strip()!r}')
        axes.append(tuple(_number(x, f'cell vector a({i})') for x in match.group(2, 3, 4)))
    if len(axes) != 3:
        raise PwOutputError(f'the run ends before the three cell vectors after "{_AXES}"')
    return tuple(axes)


def _positions(run, start, nat):
    """Labels and positions in units of alat of the header, and the index after them."""
    index = next((i for i in range(start, len(run)) if _POSITIONS in run[i]), None)
    if index is None:
        raise PwOutputError(f'no "{_POSITIONS}" line in the last run')
    lines = run[index + 1:index + 1 + nat]
    if len(lines) != nat:
        raise PwOutputError(f'the run ends before the {nat} positions after "{_POSITIONS}"')
    labels, positions = [], []
    for n, line in enumerate(lines, 1):
        try:
            label, x, y, z = parse_position_line(line)
        # parse_position_line asserts the match: AssertionError, or
        # AttributeError when assertions are off.
        except (AssertionError, AttributeError, ValueError):
            raise PwOutputError(f'atom {n} of "{_POSITIONS}": {line.strip()!r} is not '
                                '"<n> <label> tau(<n>) = ( x y z )" with finite numbers') from None
        labels.append(label)
        positions.append(_finite((x, y, z), f'position of atom {n}'))
    return tuple(labels), tuple(positions), index + 1 + nat


def _species(run, end, ntyp):
    """Species of the header, read before line ``end``."""
    pseudos = {}
    for i in range(end):
        match = _PSEUDO.search(run[i])
        if not match:
            continue
        n = int(match.group(1))
        if n in pseudos:
            raise PwOutputError(f'pseudopotential {n} is printed twice in the last run')
        path = run[i + 1].strip() if i + 1 < end else ''
        md5_line = run[i + 2].strip() if i + 2 < end else ''
        md5 = md5_line[len(_MD5):].strip() if md5_line.startswith(_MD5) else None
        pseudos[n] = (match.group(2), path, md5 if md5 and _CHECKSUM.fullmatch(md5) else None)
    if sorted(pseudos) != list(range(1, ntyp + 1)):
        raise PwOutputError(f'pseudopotentials {sorted(pseudos)} printed for {ntyp} species')
    table = next((i for i in range(end) if _SPECIES_TABLE.search(run[i])), None)
    if table is None:
        raise PwOutputError('no species table ("atomic species   valence    mass ...") in the '
                            'last run')
    rows = run[table + 1:table + 1 + ntyp]
    if len(rows) != ntyp or any(not row.split() for row in rows):
        raise PwOutputError(f'the species table has fewer than {ntyp} rows')
    return tuple(PwSpecies(row.split()[0], *pseudos[n]) for n, row in enumerate(rows, 1))


def _forces(cycle, nat):
    """Total forces of the last block of the cycle, in Ry/bohr, or None."""
    start = _last_index(cycle, _FORCES)
    if start is None:
        return None
    i = start + 1
    while i < len(cycle) and not cycle[i].strip():
        i += 1
    lines = cycle[i:i + nat]
    if len(lines) != nat:
        raise PwOutputError(f'the run ends before the {nat} forces after "{_FORCES}"')
    forces = []
    for n, line in enumerate(lines, 1):
        match = _FORCE.search(line)
        if not match or int(match.group(1)) != n:
            raise PwOutputError(f'force of atom {n}: {line.strip()!r} is not '
                                f'"atom {n} type <t> force = x y z"')
        forces.append(tuple(_number(x, f'force on atom {n}') for x in match.group(2, 3, 4)))
    return tuple(forces)


def _stress(cycle):
    """Stress tensor of the last block of the cycle, in Ry/bohr^3, or None."""
    start = _last_index(cycle, _STRESS)
    if start is None:
        return None
    lines = cycle[start + 1:start + 4]
    if len(lines) != 3:
        raise PwOutputError(f'the run ends before the three rows after "{_STRESS}"')
    stress = []
    for i, line in enumerate(lines, 1):
        words = line.split()
        if len(words) < 3:
            raise PwOutputError(f'row {i} of the stress: {line.strip()!r} has fewer than three '
                                'columns')
        stress.append(tuple(_number(x, f'stress, row {i}') for x in words[:3]))
    return tuple(stress)


def _final_coordinates(run, nat):
    """Labels, positions in angstrom and flags of the closed block of final coordinates.

    Returns ``(None, None, None)`` when the block is absent or not closed.
    """
    begin = _last_index(run, _COORDINATES_BEGIN)
    if begin is None:
        return None, None, None
    end = next((i for i in range(begin + 1, len(run)) if _starts(run[i], _COORDINATES_END)), None)
    if end is None:
        return None, None, None
    block = [line for line in run[begin + 1:end] if line.strip()]
    if any(_starts(line, _CELL_PARAMETERS) for line in block):
        raise PwOutputError('the final coordinates hold a cell: only relaxations with a fixed '
                            'cell are read')
    if not block or not _starts(block[0], _ATOMIC_POSITIONS):
        raise PwOutputError(f'the final coordinates do not begin with {_ATOMIC_POSITIONS}')
    if 'angstrom' not in block[0].lower():
        raise PwOutputError(f'final coordinates "{block[0].strip()}": only angstrom is read, '
                            'the unit of the inputs written by this package')
    if len(block) - 1 != nat:
        raise PwOutputError(f'the final coordinates have {len(block) - 1} positions for {nat} '
                            'atoms')
    try:
        entries = get_atomic_positions(block, n_atoms=nat)
    # ASE evaluates each number as an expression: "****" gives a TypeError.
    except (ValueError, IndexError, StopIteration, TypeError, ArithmeticError) as error:
        raise PwOutputError(f'the final coordinates cannot be read: '
                            f'{type(error).__name__}: {error}') from None
    labels, positions, flags = [], [], []
    for n, (label, position, mask) in enumerate(entries, 1):
        labels.append(label)
        positions.append(_finite(position, f'final position of atom {n}'))
        flags.append(None if mask is None else tuple(int(f) for f in mask))
    return tuple(labels), tuple(positions), tuple(flags)


def pw_values(text, calculation):
    """Values of the last run of a pw.x output.

    Parameters
    ----------
    text : str
        Content of the output.
    calculation : str
        ``'scf'`` or ``'relax'``: the kind of calculation of the input.

    Returns
    -------
    PwValues
        The final energy and coordinates are read for ``relax`` only.

    Raises
    ------
    ValueError
        If ``calculation`` is not ``'scf'`` or ``'relax'``.
    PwOutputError
        If the text has no pw.x header, a line of the header is missing, a
        block does not have the expected lines, or a value is not a finite
        number. A version outside ``status.SUPPORTED_VERSIONS`` is read
        (``status.version_warning``).
    """
    if calculation not in CALCULATIONS:
        raise ValueError(f'calculation: {calculation!r} is not one of {", ".join(CALCULATIONS)}')
    run, version, runs = _last_run(text)
    nat = _required(_summary_value(run, _NAT, int), _NAT)
    ntyp = _required(_summary_value(run, _NTYP, int), _NTYP)
    alat = _required(_summary_value(run, _ALAT, float), _ALAT)
    if not math.isfinite(alat) or alat <= 0:
        raise PwOutputError(f'{_ALAT} = {alat!r}: not a positive number')
    axes = _axes(run)
    labels, positions, header_end = _positions(run, _last_index(run, _AXES), nat)
    species = _species(run, header_end, ntyp)

    scf = _last_index(run, _SCF_START, header_end)
    cycle = run[scf:] if scf is not None else []
    energy_line = _last_index(cycle, _TOTAL_ENERGY)
    energy = None if energy_line is None else _energy(cycle[energy_line], 'total energy')
    values = dict(energy=energy, forces=_forces(cycle, nat), stress=_stress(cycle))
    if calculation == 'relax':
        final_line = _last_index(cycle, _FINAL_ENERGY)
        values['final_energy'] = (None if final_line is None
                                  else _energy(cycle[final_line], 'final energy'))
        values['final_labels'], values['final_positions'], values['final_if_pos'] = \
            _final_coordinates(run, nat)
    return PwValues(calculation=calculation, version=version, runs=runs, nat=nat, alat=alat,
                    axes=axes, labels=labels, positions=positions, species=species, **values)


def read_pw_values(path, calculation):
    """Values of the last run of a pw.x output file.

    The file is read as UTF-8; bytes that are not UTF-8 are replaced, as in
    ``status.read_pw_status``.

    Parameters
    ----------
    path : path-like
        Output file of pw.x.
    calculation : str
        ``'scf'`` or ``'relax'``.

    Returns
    -------
    PwValues
        See ``pw_values``.
    """
    try:
        return pw_values(Path(path).read_text(encoding='utf-8', errors='replace'), calculation)
    except PwOutputError as err:
        raise PwOutputError(f'{path}: {err}') from None


@dataclass(frozen=True)
class Parsed:
    """Outcome of the reading of the results of a calculation.

    Parameters
    ----------
    parsing : str or None
        ``'ok'``, ``'failed'``, or None when the output is not read because
        the calculation is not over with the SCF and the geometry converged.
    results : dict or None
        Results of the record (module notes), with ``parsing = 'ok'`` only.
    structure : pymatgen.core.structure.Structure or None
        Final structure, with ``parsing = 'ok'`` only.
    reason : str or None
        Why the reading failed, with ``parsing = 'failed'`` only.
    values : PwValues or None
        The values read, when they could be read, whatever the outcome.
    observed : dict or None
        Identity of the output for ``observed`` of the record
        (``observed_of``), whenever the values could be read.
    """

    parsing: str | None
    results: dict | None = None
    structure: Structure | None = None
    reason: str | None = None
    values: PwValues | None = None
    observed: dict | None = None

    @property
    def structure_text(self):
        """Text of the extxyz of the final structure (``structure_out.extxyz``), or None."""
        return None if self.structure is None else extxyz_text(self.structure)


def observed_version(printed, expected):
    """Version printed by pw.x, with as many components as the version expected.

    Parameters
    ----------
    printed : str
        Version printed (``PwValues.version``).
    expected : str or None
        Version expected (``[qe] version``), ``major.minor`` or
        ``major.minor.patch``.

    Returns
    -------
    str
        ``printed`` cut to the components of ``expected``; ``printed`` as it
        is without ``expected``, or when either is not made of numbers
        separated by dots.

    Examples
    --------
    >>> [observed_version(p, e) for p, e in [('7.3.1', '7.3'), ('7.3.1', '7.3.1'),
    ...                                      ('7.3', '7.3.1'), ('7.4', '7.3')]]
    ['7.3', '7.3.1', '7.3', '7.4']
    """
    if expected is None or not _VERSION.fullmatch(printed) or not _VERSION.fullmatch(expected):
        return printed
    return '.'.join(printed.split('.')[:len(expected.split('.'))])


def observed_of(values, energy_model=None):
    """What an output tells of its energy model, for ``observed`` of the record.

    Parameters
    ----------
    values : PwValues
    energy_model : dict, optional
        Energy model expected (``model.energy_model``): its version gives the
        form of the version observed.

    Returns
    -------
    dict
        ``version`` (``observed_version``) and ``species``: ``label``,
        ``pseudopotential`` (name of the UPF file), ``md5`` and ``path`` (as
        printed) of each species, in the order of ``ATOMIC_SPECIES``.
    """
    expected = (energy_model or {}).get('version')
    return {'version': observed_version(values.version, expected),
            'species': [{'label': s.label, 'pseudopotential': s.pseudopotential.rsplit('/', 1)[-1],
                         'md5': s.md5, 'path': s.pseudopotential} for s in values.species]}


def _check_header(values, structure):
    """Check number of atoms, labels and cell of the header against the input structure."""
    if values.nat != len(structure):
        raise PwOutputError(f'the output has {values.nat} atoms, the input structure '
                            f'{len(structure)}')
    labels = tuple(structure.labels)
    if values.labels != labels:
        raise PwOutputError(f'labels of the output {list(values.labels)} differ from those of the '
                            f'input structure {list(labels)}')
    alat = values.alat
    expected = np.asarray(structure.lattice.matrix) / (BOHR_TO_ANGSTROM * alat)
    tol = _MARGIN * _HALF_AXES * (1 + np.abs(expected) / alat)
    if np.any(np.abs(np.asarray(values.axes) - expected) > tol):
        raise PwOutputError(f'the cell of the output, {values.axes} in units of alat = {alat} '
                            'bohr, differs from that of the input structure beyond the digits '
                            'printed')
    if values.calculation == 'scf':
        expected = np.asarray(structure.cart_coords) / (BOHR_TO_ANGSTROM * alat)
        tol = _MARGIN * (_HALF_POSITION + (_HALF_AXES * np.abs(expected)
                                           + _HALF_EXTXYZ / BOHR_TO_ANGSTROM) / alat)
        far = np.any(np.abs(np.asarray(values.positions) - expected) > tol, axis=1)
        if far.any():
            raise PwOutputError(f'positions of atoms {[int(i) + 1 for i in np.flatnonzero(far)]} '
                                'of the output differ from those of the input structure beyond '
                                'the digits printed')


def _energy_of(values):
    """Total energy of the results, in Ry: ``Final energy`` of a relax, else the last ``!``."""
    if values.energy is None:
        raise PwOutputError('no "!    total energy" in the last SCF cycle')
    if values.final_energy is None:
        return values.energy
    tol = _MARGIN * (_HALF_TOTAL_ENERGY + _HALF_FINAL_ENERGY) + 1E-15 * abs(values.energy)
    if abs(values.final_energy - values.energy) > tol:
        raise PwOutputError(f'Final energy {values.final_energy!r} Ry differs from the last total '
                            f'energy {values.energy!r} Ry beyond the digits printed')
    return values.final_energy


def _final_positions(values, structure):
    """Final positions of a relax, checked against the input structure."""
    if values.final_positions is None:
        raise PwOutputError('no complete final coordinates ("Begin final coordinates" ... '
                            '"End final coordinates")')
    labels = tuple(structure.labels)
    if values.final_labels != labels:
        raise PwOutputError(f'labels of the final coordinates {list(values.final_labels)} differ '
                            f'from those of the input structure {list(labels)}')
    given = structure.site_properties.get(IF_POS) or [None] * len(structure)
    # pw.x prints the flags of the atoms with a fixed component only.
    expected = tuple(tuple(int(f) for f in flags) if flags is not None and 0 in flags else None
                     for flags in given)
    if values.final_if_pos != expected:
        raise PwOutputError(f'if_pos of the final coordinates {list(values.final_if_pos)} differ '
                            f'from those of the input structure {list(expected)}')
    return values.final_positions


def _final_structure(structure, positions):
    props = {k: v for k, v in structure.site_properties.items() if k != MAGMOM}
    return Structure(structure.lattice, structure.species, positions, coords_are_cartesian=True,
                     labels=structure.labels, site_properties=props)


def results_of(values, structure):
    """Results of a calculation from the values of its output.

    Parameters
    ----------
    values : PwValues
        Values of a run that ended with the SCF and the geometry converged.
    structure : pymatgen.core.structure.Structure
        Structure of the input (``structure_in.extxyz``).

    Returns
    -------
    results : dict
        With the keys of the module notes, JSON values only.
    final : pymatgen.core.structure.Structure
        The final structure.

    Raises
    ------
    PwOutputError
        If a result required is missing or the output disagrees with the
        input structure.
    """
    _check_header(values, structure)
    energy = _energy_of(values) * RY_TO_EV
    if values.calculation == 'relax':
        if values.forces is None:
            raise PwOutputError('no forces in the last SCF cycle of a relaxation')
        final = _final_structure(structure, _final_positions(values, structure))
    else:
        final = _final_structure(structure, structure.cart_coords)
    nsites = len(structure)
    output = {'structure': final.as_dict(), 'density': float(final.density), 'energy': energy,
              'energy_per_atom': energy / nsites}
    if values.forces is not None:
        output['forces'] = (np.asarray(values.forces) * (RY_TO_EV / BOHR_TO_ANGSTROM)).tolist()
    if values.stress is not None:
        output['stress'] = (np.asarray(values.stress) * RY_KBAR).tolist()
    results = {'nsites': nsites, 'input': {'structure': structure.as_dict()}, 'output': output}
    return jsanitize(results, strict=False), final


def parse_output(text, calculation, structure, energy_model=None):
    """Results and state ``parsing`` of a calculation from its output.

    Parameters
    ----------
    text : str
        Content of the output.
    calculation : str
        ``'scf'`` or ``'relax'``.
    structure : pymatgen.core.structure.Structure
        Structure of the input (``structure_in.extxyz``).
    energy_model : dict, optional
        Energy model of the record, for the form of ``observed``
        (``observed_of``).

    Returns
    -------
    Parsed
        ``parsing`` None when the last run is not over with the SCF and the
        geometry converged; ``'failed'`` with the reason when a value cannot
        be read, a result required is missing or the output disagrees with
        the input structure; ``'ok'`` with the results and the final
        structure otherwise. ``observed`` is given whenever the values could
        be read.

    Raises
    ------
    PwOutputError
        If the states cannot be read (``status.pw_status``): the text is not
        a pw.x output of a version read.
    """
    pw = pw_status(text, calculation)
    if not (pw.process == 'ended' and pw.scf == 'converged'
            and pw.geometry in ('converged', 'n/a')):
        return Parsed(None)
    try:
        values = pw_values(text, calculation)
    except PwOutputError as error:
        return Parsed('failed', reason=str(error))
    observed = observed_of(values, energy_model)
    try:
        results, final = results_of(values, structure)
    except PwOutputError as error:
        return Parsed('failed', reason=str(error), values=values, observed=observed)
    return Parsed('ok', results, final, values=values, observed=observed)
