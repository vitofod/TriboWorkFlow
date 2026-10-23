# Settings file (TOML)

Reference of the settings read by `triboworkflow.core.settings`
(schema version 1). Examples: `examples/single_input.toml` (one `pw.x`
input, template with placeholders) and `examples/interface.toml` (interface
workflow, complete).

```python
from triboworkflow.core.settings import load_settings

settings = load_settings('settings.toml')   # SettingsError if invalid
settings.require_complete()                 # SettingsError if placeholders are left
```

## General rules

- **Two kinds of file.** `[structure]` describes one calculation;
  `[interface]` describes the interface workflow. A file has exactly one of
  the two.
- **pw.x decides on pw.x.** The namelists are written to the input as
  given: any `pw.x` key can be used, keys left out take the `pw.x` defaults,
  and `pw.x` judges the values and ignores the keys that do not apply. The
  settings never add, change or reject a `pw.x` parameter, except for the
  keys the generator writes itself and the conventions of the project
  (`ibrav = 0`, cell and positions in angstrom, `scf` and `relax` only, no
  Hubbard).
- **The other tables are strict.** In the tables of the project
  (`[structure]`, `[k_points]`, `[interface]`, `[execution]`, …) an unknown
  or misspelt key is an error, with the list of the allowed keys. Their
  values are checked only as far as the code needs them; values that a step
  does not use are ignored.
- **Types.** In the tables of the project integers are accepted for real
  numbers; booleans are never accepted for numbers. Namelist values keep the
  TOML type and are written as such.
- **Placeholders.** Empty strings and `ecutwfc`/`ecutrho` equal to 0 are
  accepted when the file is loaded; `require_complete()` lists them all and
  refuses to go on. Generators call it before writing an input.
- **Errors** name the file and the TOML path of the field:
  `settings.toml: namelists.SYSTEM.ibrav: not allowed: ibrav = 0 always: …`.
  Type errors inside a list of tables give the path without the index
  (`atomic_species.mass`).

`R` = required, `—` = optional.

## `schema_version`

| Key | Type | | Notes |
|---|---|---|---|
| `schema_version` | int | R | `1` |

## `[structure]` (one calculation)

The structure file gives cell, positions, species labels and constraints;
everything else comes from the settings. Formats, data read and limits:
[structures.md](structures.md).

| Key | Type | Default | Allowed / notes |
|---|---|---|---|
| `file` | str | R | structure file, relative to the folder of the settings file; `""` is a placeholder |
| `format` | str | `"auto"` | `auto`, `cif`, `xyz`, `extxyz`, `pwi`, `poscar`, `json` |
| `frame` | int | `0` | counted from 0, for files with several structures |
| `xyz_units` | str | `"angstrom"` | `angstrom` |
| `preserve_atom_order` | bool | `true` | only `true` |
| `preserve_species_labels` | bool | `true` | only `true` (labels such as `Fe1`, `Fe2` stay apart) |

`[structure.cell]`

| Key | Type | Default | Allowed / notes |
|---|---|---|---|
| `source` | str | `"file"` | `file`: cell from the structure file; `config`: from `vectors` |
| `units` | str | `"angstrom"` | `angstrom` |
| `vectors` | 3×3 float | `[]` | three row vectors, read only with `source = "config"` |

No cell and no vacuum are ever created automatically.

`[structure.constraints]`: precedence override > constraint read from the
file > `default_if_pos`, atom by atom. The `if_pos` values are written as
given. They are cartesian, as in pw.x: POSCAR flags, which are along the
lattice vectors, are used only when they amount to cartesian flags (e.g.
`F F T` with c along z); otherwise give those atoms in `overrides`.

| Key | Type | Default | Allowed / notes |
|---|---|---|---|
| `read_from_file` | bool | `true` | read the constraints of the structure file |
| `default_if_pos` | 3 int | `[1, 1, 1]` | |
| `overrides` | list | `[]` | `{ atoms = [6, 7, 8], if_pos = [0, 0, 0] }`; atoms counted from 1; an atom may appear in one override only |

## `[namelists]`

`[namelists.CONTROL]`, `[namelists.SYSTEM]`, `[namelists.ELECTRONS]` and
`[namelists.IONS]` hold `pw.x` keys, written as given in the namelist of the
same name (`&IONS` only for `relax`). Values are strings, numbers or
booleans; array elements are written one key per element, quoted, with the
`pw.x` index: `"london_c6(1)" = 159.927`. Keys are matched without regard to
case, as in `pw.x`.

Keys the settings do not accept:

| Key | Where | Why |
|---|---|---|
| `ibrav`, `celldm(i)`, `A`, `B`, `C`, `cosAB`, `cosAC`, `cosBC` | SYSTEM | `ibrav = 0` always: the cell is written in `CELL_PARAMETERS` from the structure |
| `nat`, `ntyp` | SYSTEM | written by the generator |
| `starting_magnetization(i)` | SYSTEM | given per species in `[[atomic_species]]` |
| `lda_plus_u`, `lda_plus_u_kind`, `U_projection_type`, `Hubbard_*` | SYSTEM | Hubbard corrections are not supported |
| `calculation` other than `scf`, `relax` | CONTROL | only `scf` and `relax` for now |
| `calculation`, `title`, `prefix`, `restart_mode`, `outdir` | CONTROL, with `[interface]` | the workflow sets them for every calculation (own `prefix` and `outdir`, `calculation` from the calculation type) |

`pseudo_dir` and `outdir` are paths on the machine that runs `pw.x`; they
are never resolved locally.

### Dispersion corrections

Most calculations have no dispersion correction. When one is used, it is
written as in a `pw.x` input. For Grimme-D2 `vdw_corr = "grimme-d2"` alone
is enough: `pw.x` then uses its defaults for `london_s6`, `london_rcut` and
for the C6 coefficients and radii of every species. The keys given are
written as they are:

```toml
[namelists.SYSTEM]
vdw_corr = "grimme-d2"
london_s6 = 0.75            # optional
london_rcut = 200.0         # optional, bohr
"london_c6(1)" = 159.927    # optional, Ry*bohr^6, species 1 of [[atomic_species]]
"london_rvdw(1)" = 3.014    # optional, bohr, species 1 of [[atomic_species]]
```

- **Index.** The index of `london_c6(i)` and `london_rvdw(i)` is the
  position of the species in `[[atomic_species]]`, counted from 1, as in
  `ATOMIC_SPECIES`.
- **Units.** `pw.x` copies `london_c6` and `london_rvdw` without conversion
  (`in_c6(:) = london_c6(:)`, `in_rvdw(:) = london_rvdw(:)` in
  `PW/src/input.f90`) into the arrays that hold its own table of Grimme's
  values, which `Modules/mm_dispersion.f90` stores in Ry·bohr⁶ and bohr
  ("1 J*nm^6/mol = 34.69 Ry*Bohr^6"). Values from papers or VASP
  (`VDW_C6` in J·nm⁶·mol⁻¹, `VDW_R0` in Å) must be converted first; the
  settings never convert them.
- **Modified parameters.** In the D2 variant with the parameters of the
  preceding noble gas on metal atoms (Marquis et al., ACS Appl. Nano Mater.
  5, 10516 (2022): C6 = 4.61 J·nm⁶·mol⁻¹ and R0 = 1.595 Å, the values of
  Ar, instead of 10.80 and 1.562 for Ti), the `pw.x` table values of Ar are
  159.927 Ry·bohr⁶ and 3.014 bohr, as in the example above.
- **What pw.x does with the values** (QE 7.3): with a negative C6 or a
  radius ≤ 0 it uses its table value; the D2 keys are read only with
  `grimme-d2`; with `grimme-d3`, `dftd3_version = 2` turns
  `dftd3_threebody` off.

Sources: `pw.x` input description, version 7.5
(<https://www.quantum-espresso.org/Doc/INPUT_PW.html>); `PW/src/input.f90`
and `Modules/mm_dispersion.f90` of QE 7.3. Check them against the QE
version in use.

## `[[atomic_species]]`

One entry per species label of the structures. The order of the entries is
the order of `ATOMIC_SPECIES` and of the per-species `pw.x` indices
(`starting_magnetization(i)`, `london_c6(i)`, …); it is never sorted.

| Key | Type | | Notes |
|---|---|---|---|
| `label` | str | R | label as in `ATOMIC_POSITIONS`; unique; `""` is a placeholder |
| `mass` | float | R | amu |
| `pseudopotential` | str | R | UPF file name; `""` is a placeholder |
| `starting_magnetization` | float | — | written as `starting_magnetization(i)`; never set automatically |

## `[k_points]`

| Key | Type | | Notes |
|---|---|---|---|
| `mode` | str | R | `automatic`, `gamma`, `kdens` |
| `grid` | 3 int | R with `automatic` | `nk1 nk2 nk3` |
| `offset` | 3 int | R with `automatic` | `sk1 sk2 sk3` |
| `kdens` | float | R with `kdens` | |
| `gamma_for_unshifted_1x1x1` | bool | `true` | a final grid 1 1 1 with offset 0 0 0 is written as `K_POINTS gamma` |

- `grid` and `offset` are read only with `automatic`, `kdens` only with
  `kdens`.
- `gamma` writes `K_POINTS gamma` (real wavefunctions, Γ-point routines).
- `kdens` computes the grid when the cell is known, as TribChem does for
  VASP: `MeshFromDensity(structure, kdens, material_type, force_gamma=True)`
  (`highput/utils/vasp_tools.py`), i.e. k-points per Å⁻¹ of each reciprocal
  vector, k3 = 1 for slabs and interfaces, mesh centred on Γ (offset 0 0 0).

## `[generation]` (one calculation)

Read only with `[structure]`; in the workflow the file names are fixed.

| Key | Type | Default | Notes |
|---|---|---|---|
| `file_prefix` | str | `"input.pwi"` | name of the generated input |
| `overwrite` | bool | `false` | if false, an existing file is an error |
| `atomic_positions_units` | str | `"angstrom"` | `angstrom` |
| `cell_parameters_units` | str | `"angstrom"` | `angstrom` |

## `[execution]`

Needed only to run on HPC. No value is tied to a specific machine.

| Key | Type | | Notes |
|---|---|---|---|
| `scheduler` | str | R | `slurm`, `pbs`, `torque`, `lsf`, `sge`, `htcondor`, `condor` (the names of Xsorb) |
| `jobscript` | str | R | user jobscript, copied for every job with the launch line added |
| `run_command` | str | R | command up to the input file, e.g. `mpirun -np 32 pw.x -nk 4` |
| `groups` | int | R | > 0, jobs over which the calculations of a step are spread |
| `walltime` | int | R | > 0 s, walltime of each job |
| `polling_interval` | int | R | > 0 s, time between two checks of the jobs |

## `[interface]` (interface workflow)

| Key | Type | Default | Notes |
|---|---|---|---|
| `slabs` | list | R | two `[[interface.slabs]]` entries: bottom slab, then top slab |
| `set_bm` | bool | `true` | strain weights from the bulk moduli of the two materials |
| `engine` | str | `"qe"` | engine of the steps that run calculations |
| `engines` | table | `{}` | per-step engine for `pes`, `adhesion`, `ppes`, `charge`; only `qe` for now |

`[[interface.slabs]]`: the identifiers TribChem uses for a slab.

| Key | Type | | Notes |
|---|---|---|---|
| `file` | str | R | slab structure, relative to the folder of the settings file; `""` is a placeholder. A pymatgen JSON slab is used as TribChem reads it from its database, any other file as a TribChem custom slab ([structures.md](structures.md)) |
| `mid` | str | R | material id; the interface id is `mid1_mid2` |
| `miller` | 3 int | R | |
| `formula` | str | — | |
| `name` | str | — | tells apart homonyms (e.g. C diamond, C graphene) |
| `bulk_modulus` | float | R with `set_bm = true` | GPa as in TribChem; read only with `set_bm = true` |

With `set_bm = true` TribChem sets `strain_weight_1` and `strain_weight_2`
to the two bulk moduli (`FT_StartInterfaceMatch.set_bulk_modulus`),
replacing any value given in `inter_params`. With `set_bm = false` the
weights come from `inter_params` (TribChem's default 1:1).

`[interface.inter_params]`: the keys of TribChem's `inter_params`; other
keys are errors, as in TribChem's `read_default_params`. Missing keys take
TribChem's defaults (`highput/firetasks/defaults.json`); the values are
checked by TribChem's `InterfaceMatcher`.

| Key | Type | TribChem default |
|---|---|---|
| `max_area` | float | 150 |
| `interface_distance` | float | 2 |
| `max_angle_tol` | float | 0.01 |
| `max_length_tol` | float | 0.02 |
| `max_area_ratio_tol` | float | 0.05 |
| `strain_weight_1`, `strain_weight_2` | float | 1, 1 |
| `best_match` | str | `"area"` |

TribChem's interface generation overwrites `max_angle_tol` with 0.4 before
matching (`highput/firetasks/slabs.py`), whatever its value.
