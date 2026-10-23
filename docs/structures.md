# Structure files

`triboworkflow.core.structures` reads structure files into pymatgen objects,
converts them to and from ASE, and writes xyz and extxyz files. Every reader
keeps the cell matrix, the cartesian positions, the order of the atoms, the
labels and the per-atom data present in the file; nothing missing from the
file is added or zeroed. What cannot be represented is an error, never a
silent conversion.

## Data carried by a structure

For every atom three names are kept apart:

| What | Where | Example |
|---|---|---|
| element | `site.specie` | `Fe` |
| QE label of the species (`ATOMIC_SPECIES`, `ATOMIC_POSITIONS`) | `site.label`; the element when the file has no label | `Fe1`, `Fe2` |
| CIF site identifier (`_atom_site_label`) | site property `cif_label`, CIF files only | `O1` |

Constraints come in two frames, never mixed in one structure:

| Site property | Frame | Values | From |
|---|---|---|---|
| `if_pos` | cartesian components, as pw.x applies them to the forces | 3 integers, 0 fixed, 1 free; `None` = no constraint given in the file | PWI, extxyz |
| `selective_dynamics` | lattice vectors, as ASE reads POSCAR flags (`FixScaled`) | 3 booleans, True free | POSCAR, JSON, the vendored `VaspInputSet` |

Flags along the lattice vectors become `if_pos` only when the lattice vectors
they leave free lie along cartesian axes, e.g. `F F T` with c along z gives
`0 0 1`; otherwise the conversion is refused. A lattice-vector component of
at most 1e-6 Å counts as zero.

Magnetisation: two different quantities, never converted into each other.

| Quantity | Where | Meaning |
|---|---|---|
| `magmom` (site property) | extxyz (`initial_magmoms`), JSON | initial moment per atom in Bohr magnetons, as VASP's MAGMOM and ASE's initial moments |
| `starting_magnetization(i)` | `[[atomic_species]]` of the TOML | pw.x fraction per species: the only source for the inputs |
| `starting_magnetization(i)` read from a PWI | `structure.properties['qe_species']` | provenance only: never becomes `magmom`, never replaces the TOML |

## Formats

| Format | Library | Cell | QE labels | Constraints | `magmom` | Notes |
|---|---|---|---|---|---|---|
| `cif` | ASE | yes | element | — | — | keeps the order of the sites (pymatgen's `CifParser` sorts them); partial occupancies rejected; CIF tags not kept |
| `xyz` | ASE (plain) | no: `[structure.cell] source = "config"` | — (elements only) | — | — | a comment line with `Lattice=` is an extxyz file: error |
| `extxyz` | ASE | `Lattice`, `pbc="T T T"` | column `qe_label` | `move_mask` (cartesian) | `initial_magmoms` | other columns become site properties |
| `pwi` | ASE parsing functions | `CELL_PARAMETERS`, `ibrav = 0` only | whole labels | `if_pos` of `ATOMIC_POSITIONS` | — | `qe_species`: label, mass, pseudopotential, `starting_magnetization` |
| `poscar` | pymatgen | yes | element | `selective_dynamics` | — | input only |
| `json` | pymatgen dict | `lattice.matrix` required | `label` of the sites | site properties | site properties | `Structure` or `Slab`; see below |

`format = "auto"` takes the format from the name: `.cif`, `.xyz`,
`.extxyz`, `.pwi`, `.json`, `.vasp`, `.poscar`, names starting with
`POSCAR` or `CONTCAR`. Relative paths in the settings are taken from the
folder of the settings file.

**JSON slabs.** A pymatgen `Slab` dict (`Slab.as_dict`) is read without
losses: lattice matrix, sites (fractional coordinates, labels, properties),
`miller_index`, `shift`, `scale_factor`, `reconstruction`, `energy`,
`oriented_unit_cell`. pymatgen's `Slab.from_dict` alone is not lossless: it
rebuilds the lattice with `Lattice.from_parameters` and drops the
reconstruction and the labels. A dict without `lattice.matrix` has no
orientation and is an error; so are unknown keys and a charge.

To remove: the JSON slabs include the `structure.opt` of TribChem slab
documents; those of TribChem's custom slabs have no `lattice.matrix`.

## Writing

- **xyz**: elements and cartesian positions only. Cell, labels, constraints
  and moments are lost: the format has no place for them.
- **extxyz**: cell, `pbc`, `initial_magmoms`, `qe_label` (when a label
  differs from its element), the other per-atom data, and the constraints as
  `move_mask`, which is cartesian: `selective_dynamics` is converted when
  possible, otherwise error. Atoms without constraints in the source are
  marked by the column `constraint_given` and read back as `None`. Structure
  properties (e.g. `qe_species`) are not written. Every atom gets its own
  `FixCartesian`, because ASE's writer marks as fixed the atoms missing from
  a `FixCartesian` list and keeps only the first kind of a mixed list.

## ASE and pymatgen

`to_atoms` and `to_structure` convert without losses: constraints
(`if_pos` ↔ `FixCartesian`, `selective_dynamics` ↔ `FixScaled`, `FixAtoms`
read as fully fixed), QE labels (ASE array `qe_label`), moments and the other
per-atom data. pymatgen's `AseAtomsAdaptor` alone fails on partial
`selective_dynamics` and drops `FixCartesian` and `FixScaled`. Other ASE
constraints, both frames in one `Atoms`, and an atom in two constraints are
errors. A `Slab` becomes plain `Atoms`.

## Constraints applied

- **Single calculation** (`[structure.constraints]`): override > constraint
  of the file > `default_if_pos`, per atom; `read_from_file = false` ignores
  the file. `load_structure` returns the `if_pos` of every atom with its
  source (`override`, `file`, `default`).
- **Workflows**: the constraints of the `calc_type`, set by the vendored
  `VaspInputSet` (`interface_z_relax`: `0 0 1`, H atoms `1 1 1`;
  `slab_pos_relax`: `1 1 1`), with source `calc_type`. The constraints of
  the slab files are kept in the structure as read, as provenance.

## Slabs of the workflows

The interface workflow (vendored matching and high-symmetry modules) works on
pymatgen `Slab` objects. `load_workflow_slabs` gives, for each slab of
`[interface]`, the structure as read (`original`), the `Slab` given to the
workflow (`slab`) and the transformation between them:

- a JSON slab follows the `database` path: it is used as stored, with
  `Slab.from_dict`; its `miller_index` must equal `miller`;
- any other file follows the `custom_slab` path: `Slab` with
  `oriented_unit_cell` = the structure, `shift = 0`, `scale_factor = (1, 1,
  1)`; only `magmom` among the site properties, from the file or zeros;
  stored as a dict without lattice matrix and read back with
  `Slab.from_dict`.

Both paths rebuild the lattice with `Lattice.from_parameters`, already in the
`Slab` constructor (`reorient_lattice=True`): c along z, a along x. Hence:

- the cell must be right-handed with c perpendicular to a and b (|cos| at
  most 1e-6); otherwise the slab would be mirrored or its surface tilted, and
  the file is refused;
- the rotation from the frame of the file to that of the slab is recorded
  (`rotation`, `rotated`, `provenance()`), with the data dropped (e.g.
  `selective_dynamics`, structure properties) and added (`magmom`);
- QE labels other than the element are refused: the workflow code does not
  carry them, so the workflows cannot guarantee them along the chain. This
  also excludes, for now, magnetic cases with several QE species of the same
  element (e.g. `Fe1`, `Fe2` with opposite `starting_magnetization`). No
  label is ever turned into its element silently.

To remove: the two paths reproduce how TribChem's interface workflow reads
its slabs from the database (`FT_StartInterfaceMatch`) and how
`FT_StoreCustomSlab` stores a custom slab, whose MAGMOM of the INCAR is
replaced by the moments of the file; golden tests compare `custom_slab` with
that Firetask. TribChem transforms the left-handed and tilted cells that are
refused here.
