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
| `pwo` | the pw.x output parser of `engines.qe` | `crystal axes` × `celldm(1)`, any `ibrav` | whole labels | `if_pos` of the final coordinates of a `relax` | — | last run only, over and converged; `qe_species` as for `pwi`; the digits pw.x prints; see below |
| `poscar` | pymatgen | yes | element | `selective_dynamics` | — | input only |
| `json` | pymatgen dict | `lattice.matrix` required | `label` of the sites | site properties | site properties | `Structure` or `Slab`; see below |

`format = "auto"` takes the format from the name: `.cif`, `.xyz`,
`.extxyz`, `.pwi`, `.pwo`, `.json`, `.vasp`, `.poscar`, names starting with
`POSCAR` or `CONTCAR`. Any other name needs the format (an output called
`run.out` is read with `format = "pwo"`). Relative paths in the settings are
taken from the folder of the settings file.

**Structures from a pw.x output (`pwo`).** The structure of a calculation that
was run elsewhere, e.g. a relaxed slab, is read from its output instead of its
input.

- *Which run.* Only the last run of the file is read (a restart appends its
  output to the same file), and only if that run is over with the SCF
  converged and, for a relaxation, the geometry converged and the final
  coordinates closed. Otherwise the file is refused and the error names the
  states (`process`, `SCF`, `geometry`, `final coordinates`, `stop reason`);
  a structure from an unfinished run is never returned. The summary of the run
  tells the kind of calculation: it prints `nstep` for a relaxation only.
- *Positions.* A `scf` gives the positions of the header (the atoms of its
  input); a `relax` the closed block `Begin final coordinates` … `End final
  coordinates`, which must be in angstrom and must not hold a cell: a
  variable-cell relaxation is refused. The cell is the `crystal axes` of the
  header times `celldm(1)`, whatever `ibrav` the input gave.
- *Digits.* The structure has what pw.x prints, not what its input had. The
  axes and `celldm(1)` have six decimals, so the cell is known to a few
  1e-6 Å for the cells of a slab; the header positions of a `scf` have seven decimals in
  units of `alat`; the final coordinates of a `relax` keep their printed
  digits. A PWI file is the input itself and exact. When the exact cell is
  known, `cell` in `[structure]` replaces the printed one and keeps the
  cartesian positions.
- *Labels and species.* Labels are kept whole (`Fe1` stays `Fe1`). The element
  is the symbol the label starts with, as for a PWI file, and must be the
  element of the pseudopotential that pw.x printed for the species; a
  contradiction is an error. `qe_species` has, in the order of the output,
  `label`, `mass`, the name of the UPF file and `starting_magnetization`. The
  last one comes from the table `Starting magnetic structure` that pw.x prints
  for spin-polarized calculations, and is `None` without it; it is provenance
  only and never becomes `magmom`. A table with columns other than the
  magnetization (non-collinear) is an error.
- *Constraints.* A `relax` prints the flags after the final positions of the
  atoms that have a fixed component only, so an atom given `1 1 1` in the
  input reads as an atom without constraint (`None`), and the other flags
  become `if_pos`. A `scf` prints no flags: the structure has no `if_pos`.
- *Slabs.* A structure from a `pwo` follows the rules of any other file (see
  "Slabs of the workflows"). A cell whose c is not perpendicular to a and b is
  refused as a slab: e.g. a relaxed cell printed with
  `a(1) = (0.998727, 0, -0.050443)` and `a(3) = (0, 0, 3.304038)` (in units
  of `alat`) has an angle of about 92.9° between a and c. The reader does not
  straighten it; it must be a slab with c along z before it is used.

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
