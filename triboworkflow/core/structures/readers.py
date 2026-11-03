# Author: Vito Foderà
# Created: 2026-10-23

"""Reading of CIF, XYZ, extxyz, PWI, POSCAR and pymatgen JSON files.

Every reader keeps the cell matrix, the cartesian positions, the order of the
atoms, the labels and the per-atom data present in the file (``model.py``),
and never adds or zeroes data that the file does not give.

Notes
-----
Libraries and choices per format:

- CIF: ASE, which keeps the order of the sites; pymatgen's ``CifParser``
  sorts them (a P1 file with Fe, O, Fe is read as Fe, Fe, O). The QE label is
  the element; ``_atom_site_label`` becomes the site property ``cif_label``.
  Partial occupancies are rejected; the other CIF tags are not kept;
- XYZ: ASE's plain reader. The format has elements and positions only, so the
  cell must come from the settings (``cell``);
- extxyz: ASE. ``move_mask`` holds cartesian flags (``ase/io/extxyz.py:492-500``),
  ``initial_magmoms`` the ``magmom``, ``qe_label`` the QE labels;
- PWI: the parsing functions of ASE's ``read_espresso_in``, as Xsorb does
  (``xsorb/ase_custom/espresso.py:34-102``), with ``ibrav = 0`` only.
  Differently from both, the labels are kept whole and
  ``starting_magnetization(i)`` stays with the i-th species of
  ``ATOMIC_SPECIES``: ASE and Xsorb look it up by element and turn it into
  initial moments, a different quantity (``model.py``);
- POSCAR: pymatgen's reader, as ``Structure.from_file`` uses it;
- JSON: a pymatgen ``Structure`` or ``Slab`` dict, e.g. a slab saved with
  ``Slab.as_dict``. ``Slab.from_dict`` reorients the lattice and drops the
  reconstruction and the labels, so slabs are rebuilt here without
  ``reorient_lattice``.

To remove: the POSCAR reader is the one of TribChem's custom-slab Firetask
(``highput/firetasks/custom_slab.py:120``), and the JSON slabs include the
``structure.opt`` of TribChem slab documents.
"""

import io
import json
import re
from pathlib import Path

import ase.io
import numpy as np
from ase.io.espresso import (get_atomic_positions, get_atomic_species, get_cell_parameters,
                             label_to_symbol, read_fortran_namelist, units)
from pymatgen.core.lattice import Lattice
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab

from .convert import to_structure
from .model import CIF_LABEL, CONSTRAINT_GIVEN, IF_POS, QE_SPECIES, StructureError

FORMATS = ('cif', 'xyz', 'extxyz', 'pwi', 'poscar', 'json')

# Keys of a pymatgen Slab dict that read_json restores.
SLAB_KEYS = {'@module', '@class', '@version', 'charge', 'lattice', 'properties', 'sites',
             'miller_index', 'oriented_unit_cell', 'shift', 'scale_factor', 'reconstruction',
             'energy'}
STRUCTURE_KEYS = {'@module', '@class', '@version', 'charge', 'lattice', 'properties', 'sites'}


def detect_format(path):
    """Format of a structure file from its name.

    Parameters
    ----------
    path : str or pathlib.Path

    Returns
    -------
    str
        One of ``FORMATS``: from the suffix (``.cif``, ``.xyz``, ``.extxyz``,
        ``.pwi``, ``.json``, ``.vasp``, ``.poscar``) or, for POSCAR, from a
        name starting with ``POSCAR`` or ``CONTCAR``.

    Raises
    ------
    StructureError
        If the name tells no format.
    """
    name = Path(path).name
    suffix = Path(path).suffix.lower()
    by_suffix = {'.cif': 'cif', '.xyz': 'xyz', '.extxyz': 'extxyz', '.pwi': 'pwi',
                 '.json': 'json', '.vasp': 'poscar', '.poscar': 'poscar'}
    if suffix in by_suffix:
        return by_suffix[suffix]
    if name.upper().startswith(('POSCAR', 'CONTCAR')):
        return 'poscar'
    raise StructureError(f'{path}: cannot tell the format from the name; give it explicitly '
                         f'({", ".join(FORMATS)})')


def read_structure(path, format='auto', frame=0, cell=None):
    """Read one structure from a file.

    Parameters
    ----------
    path : str or pathlib.Path
        The structure file.
    format : str, optional
        One of ``FORMATS``, or ``'auto'`` to take it from the name
        (``detect_format``).
    frame : int, optional
        Structure to read, counted from 0, in files with several (CIF,
        XYZ, extxyz); must be 0 for the others.
    cell : array_like, shape (3, 3), optional
        Row vectors in angstrom that replace the cell of the file, keeping
        the cartesian positions; required for plain XYZ.

    Returns
    -------
    pymatgen.core.structure.Structure
        A ``Slab`` for a JSON slab.

    Raises
    ------
    StructureError
        If the file cannot be read in that format, holds partial
        occupancies, or lacks data the format requires (e.g. the cell).
    """
    path = Path(path)
    fmt = detect_format(path) if format == 'auto' else format
    if fmt not in FORMATS:
        raise StructureError(f'{path}: unknown format {fmt!r}; allowed: {", ".join(FORMATS)}')
    if not path.is_file():
        raise StructureError(f'{path}: file not found')
    if fmt in ('pwi', 'poscar', 'json') and frame != 0:
        raise StructureError(f'{path}: a {fmt} file holds one structure; frame must be 0, got {frame}')
    try:
        if fmt == 'cif':
            structure = _read_cif(path, frame)
        elif fmt == 'xyz':
            structure = _read_xyz(path, frame, cell)
        elif fmt == 'extxyz':
            structure = _read_extxyz(path, frame, cell)
        elif fmt == 'pwi':
            structure = _read_pwi(path)
        elif fmt == 'poscar':
            structure = Structure.from_str(path.read_text(), fmt='poscar')
        else:
            structure = read_json(path)
    except StructureError as error:
        raise StructureError(f'{path}: {error}') from None
    except (ValueError, KeyError, IndexError, StopIteration) as error:
        raise StructureError(f'{path}: cannot read it as {fmt}: {type(error).__name__}: {error}') from None
    if not structure.is_ordered:
        raise StructureError(f'{path}: partial occupancies are not supported')
    if cell is not None and fmt not in ('xyz', 'extxyz'):
        structure = replace_cell(structure, cell)
    return structure


def replace_cell(structure, cell):
    """A structure with another cell and the same cartesian positions.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        Not a ``Slab``.
    cell : array_like, shape (3, 3)
        Three independent row vectors in angstrom.

    Returns
    -------
    pymatgen.core.structure.Structure
        With the labels and properties of ``structure``.
    """
    if isinstance(structure, Slab):
        raise StructureError('the cell of a JSON slab cannot be replaced')
    matrix = np.asarray(cell, float)
    if matrix.shape != (3, 3) or abs(np.linalg.det(matrix)) < 1e-12:
        raise StructureError(f'the cell must be three independent row vectors, got {cell}')
    return Structure(Lattice(matrix), structure.species, structure.cart_coords,
                     coords_are_cartesian=True, site_properties=structure.site_properties,
                     labels=structure.labels, properties=dict(structure.properties))


def _frame(path, fmt, frame, **kwargs):
    images = ase.io.read(path, index=':', format=fmt, **kwargs)
    if not 0 <= frame < len(images):
        raise StructureError(f'frame {frame} does not exist: the file has {len(images)} '
                             f'structure(s), counted from 0')
    return images[frame]


def _read_cif(path, frame):
    atoms = _frame(path, 'cif', frame, store_tags=True)
    occupancy = atoms.info.get('occupancy') or {}
    partial = sorted({sym for kind in occupancy.values() for sym, occ in kind.items() if occ != 1})
    if partial:
        raise StructureError(f'partial occupancies are not supported (species {", ".join(partial)})')
    site_ids = atoms.info.get('_atom_site_label')
    kinds = atoms.arrays.pop('spacegroup_kinds', None)
    atoms.info = {}  # CIF tags and ASE bookkeeping, not structure data
    structure = to_structure(atoms)
    if site_ids is not None and kinds is not None:
        structure.add_site_property(CIF_LABEL, [str(site_ids[k]) for k in kinds])
    return structure


def _read_xyz(path, frame, cell):
    with open(path) as f:
        f.readline()
        comment = f.readline()
    if re.search(r'\bLattice\s*=', comment):
        raise StructureError('the comment line has a Lattice: this is an extxyz file, read it '
                             'with format = "extxyz"')
    if cell is None:
        raise StructureError('a plain XYZ file has no cell: give it in the settings '
                             '([structure.cell] source = "config")')
    atoms = _frame(path, 'xyz', frame)
    atoms.cell, atoms.pbc = np.asarray(cell, float), True
    return to_structure(atoms)


def read_extxyz_text(text):
    """Structure of the text of an extxyz file, as ``read_structure`` reads the file.

    Parameters
    ----------
    text : str
        One structure, with its periodic cell.

    Returns
    -------
    pymatgen.core.structure.Structure

    Raises
    ------
    StructureError
        If the text cannot be read as extxyz or has no periodic cell.
    """
    try:
        return _read_extxyz(io.StringIO(text), 0, None)
    except StructureError:
        raise
    # OSError: ASE's XYZError, raised for a text that is not extxyz.
    except (ValueError, KeyError, IndexError, StopIteration, OSError) as error:
        raise StructureError(f'cannot read it as extxyz: {type(error).__name__}: {error}') from None


def _read_extxyz(path, frame, cell):
    atoms = _frame(path, 'extxyz', frame)
    if cell is not None:
        atoms.cell, atoms.pbc = np.asarray(cell, float), True
    elif not atoms.pbc.all() or atoms.cell.rank != 3:
        raise StructureError('no periodic cell in the file (Lattice and pbc="T T T"): '
                             'give it in the settings ([structure.cell] source = "config")')
    given = atoms.arrays.pop(CONSTRAINT_GIVEN, None)
    structure = to_structure(atoms)
    if IF_POS in structure.site_properties:
        # A move_mask column gives every atom's flags; constraint_given,
        # written by write_extxyz, marks the atoms whose flags were given.
        if_pos = [[1, 1, 1] if v is None else v for v in structure.site_properties[IF_POS]]
        if given is not None:
            if_pos = [v if g else None for v, g in zip(if_pos, given)]
        structure.add_site_property(IF_POS, if_pos)
    return structure


def _read_pwi(path):
    with open(path) as f:
        data, card_lines = read_fortran_namelist(f)
    system = data.get('system')
    if system is None:
        raise StructureError('&SYSTEM not found')
    for key in ('ibrav', 'nat', 'ntyp'):
        if key not in system:
            raise StructureError(f'{key} missing in &SYSTEM')
    if system['ibrav'] != 0:
        raise StructureError(f'ibrav = {system["ibrav"]}: only ibrav = 0 is supported')
    # As read_espresso_in (ase/io/espresso.py:541-550): celldm(1) in bohr wins over A.
    if 'celldm(1)' in system:
        alat = system['celldm(1)'] * units['Bohr']
    else:
        alat = system.get('A')
    cell, _ = get_cell_parameters(card_lines, alat=alat)
    if cell is None:
        raise StructureError('CELL_PARAMETERS missing (required with ibrav = 0)')
    species = get_atomic_species(card_lines, n_species=system['ntyp'])
    if species is None:
        raise StructureError('ATOMIC_SPECIES missing')
    positions = get_atomic_positions(card_lines, n_atoms=system['nat'], cell=cell, alat=alat)
    if positions is None:
        raise StructureError('ATOMIC_POSITIONS missing')
    elements = {}
    qe_species = []
    for i, (label, mass, pseudo) in enumerate(species):
        elements[label] = label_to_symbol(label)
        qe_species.append({'label': label, 'mass': mass, 'pseudopotential': pseudo,
                           'starting_magnetization': system.get(f'starting_magnetization({i + 1})')})
    unknown = sorted({label for label, _, _ in positions if label not in elements})
    if unknown:
        raise StructureError(f'labels of ATOMIC_POSITIONS not in ATOMIC_SPECIES: {", ".join(unknown)}')
    structure = Structure(Lattice(cell), [elements[label] for label, _, _ in positions],
                          [pos for _, pos, _ in positions], coords_are_cartesian=True,
                          labels=[label for label, _, _ in positions],
                          properties={QE_SPECIES: qe_species})
    flags = [None if f is None else [int(x) for x in f] for _, _, f in positions]
    if any(f is not None for f in flags):
        structure.add_site_property(IF_POS, flags)
    return structure


def read_json(path):
    """A pymatgen Structure or Slab from its JSON dict, without losses.

    Parameters
    ----------
    path : str or pathlib.Path
        JSON file with a ``Structure`` or ``Slab`` dict (``as_dict``).

    Returns
    -------
    pymatgen.core.structure.Structure or pymatgen.core.surface.Slab
        A ``Slab`` keeps lattice matrix, fractional coordinates, labels, site
        and structure properties, ``miller_index``, ``shift``,
        ``scale_factor``, ``reconstruction``, ``energy`` and
        ``oriented_unit_cell``; it is built with ``reorient_lattice=False``.

    Raises
    ------
    StructureError
        For another ``@class``, keys that are not restored, a lattice without
        ``matrix``, or a charged slab.
    """
    with open(path) as f:
        dct = json.load(f)
    if not isinstance(dct, dict) or dct.get('@class') not in ('Structure', 'Slab'):
        raise StructureError('a pymatgen Structure or Slab dict is required ("@class")')
    known = SLAB_KEYS if dct['@class'] == 'Slab' else STRUCTURE_KEYS
    unknown = sorted(set(dct) - known)
    if unknown:
        raise StructureError(f'keys not supported in a {dct["@class"]} dict: {", ".join(unknown)}')
    if 'matrix' not in dct['lattice']:
        raise StructureError('the lattice has no matrix: its orientation is unknown (pymatgen '
                             'would rebuild it with Lattice.from_parameters)')
    if dct['@class'] == 'Structure':
        return Structure.from_dict(dct)
    if dct.get('charge') not in (None, 0):
        raise StructureError(f'charge = {dct["charge"]}: a pymatgen Slab cannot carry a charge')
    lattice = Lattice(dct['lattice']['matrix'])
    sites = [PeriodicSite.from_dict(sd, lattice) for sd in dct['sites']]
    struct = Structure.from_sites(sites)
    slab = Slab(lattice, struct.species_and_occu, struct.frac_coords,
                miller_index=dct['miller_index'],
                oriented_unit_cell=Structure.from_dict(dct['oriented_unit_cell']),
                shift=dct['shift'], scale_factor=np.array(dct['scale_factor']),
                reorient_lattice=False, reconstruction=dct.get('reconstruction'),
                site_properties=struct.site_properties, energy=dct.get('energy'))
    for site, label in zip(slab, struct.labels):
        site.label = label
    slab.properties = dict(dct.get('properties') or {})
    return slab
