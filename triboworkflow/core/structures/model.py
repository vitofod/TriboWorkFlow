# Author: Vito Foderà
# Created: 2026-10-23

"""Data carried by the structures read and written by this package.

A structure is a pymatgen ``Structure`` (a ``Slab`` when it comes from a
pymatgen JSON slab), with the atoms in the order of the file.

Notes
-----
Three names are kept apart for every atom:

- the element: ``site.specie``;
- the QE label of the species, as in ``ATOMIC_SPECIES`` and
  ``ATOMIC_POSITIONS`` (e.g. ``Fe1``, ``Fe2``): ``site.label``, equal to the
  element when the file gives no label;
- the identifier of the CIF site (``_atom_site_label``, e.g. ``O1``): the site
  property ``cif_label``, only for CIF files.

Constraints come in two frames, never mixed in one structure:

- ``if_pos``: pw.x convention, three integers per atom on the *cartesian*
  components (0 fixed, 1 free), ``None`` for an atom without constraints in
  the file. From PWI and extxyz files;
- ``selective_dynamics``: VASP and pymatgen convention, three booleans per
  atom along the *lattice vectors* (True free), as ASE reads POSCAR flags
  (``FixScaled``, ``ase/io/vasp.py:311-323``). From POSCAR and JSON files,
  and from the vendored ``VaspInputSet``, which sets the constraints of each
  ``calc_type``.

Magnetisation:

- the site property ``magmom`` is an initial magnetic moment per atom in
  Bohr magnetons, with the meaning of VASP's MAGMOM and of ASE's initial
  magnetic moments. It is read from extxyz and JSON files and carried for
  the vendored workflow code;
- ``starting_magnetization(i)`` of pw.x is a fraction per species, a
  different quantity. In the inputs it comes from the TOML settings only. The
  values found in a PWI are kept as provenance in
  ``structure.properties['qe_species']`` and never become ``magmom``; neither
  of the two is ever converted into the other.
"""

QE_LABEL = 'qe_label'                  # ASE array and extxyz column of the QE labels
CIF_LABEL = 'cif_label'                # site property: CIF site identifier
IF_POS = 'if_pos'                      # site property: cartesian constraints (pw.x)
SELECTIVE_DYNAMICS = 'selective_dynamics'  # site property: constraints along the lattice vectors
CONSTRAINT_GIVEN = 'constraint_given'  # extxyz column: atoms with constraints in the source
MAGMOM = 'magmom'                      # site property: initial moment per atom (Bohr magnetons)
QE_SPECIES = 'qe_species'              # structure property: ATOMIC_SPECIES of a PWI

CONSTRAINT_KEYS = (IF_POS, SELECTIVE_DYNAMICS)


class StructureError(ValueError):
    """A structure that cannot be read, converted or written as requested."""
