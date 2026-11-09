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
  keys the generator writes itself, the conventions of the project
  (`ibrav = 0`, cell and positions in angstrom, `scf` and `relax` only), and
  the keys the relaunches change: `max_seconds`, written for
  every run, and `nstep` and `mixing_beta`, checked here and changed by the
  remedies (see [Values changed by the relaunches](#values-changed-by-the-relaunches)),
  and `wfcdir`, which the relaunches remove (see below).
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
| `calculation` other than `scf`, `relax` | CONTROL | only `scf` and `relax` for now |
| `calculation`, `title`, `prefix`, `restart_mode`, `outdir` | CONTROL, with `[interface]` | the workflow sets them for every calculation (own `prefix` and `outdir`, `calculation` from the calculation type) |
| `wfcdir` absolute, `.`, through `..` or with `\` | CONTROL | the files that `pw.x` writes per process belong to the calculation: the worker removes them with `outdir` before a relaunch from scratch, and two calculations never share them; a relative directory (e.g. `"WFC"`) is inside the directory of each calculation |

`pseudo_dir` and `outdir` are paths on the machine that runs `pw.x`; they
are never resolved locally.

Hubbard corrections are not implemented: the generator writes no `HUBBARD`
card, which `pw.x` 7.2 needs to apply them. Their keys (`Hubbard_*`,
`starting_ns_eigenvalue(…)`) are not rejected and go to `pw.x` as given, like
`space_group` and the other keys that the project does not use.

### Values changed by the relaunches

A calculation that ends without convergence runs again with `nstep` and
`mixing_beta` changed together (levels 2 and 3 of the remedies, see
`execution.md`), starting from the values `pw.x` uses: those given here or
its defaults. So these two keys are checked:

| Key | Where | Error | Warning (recommended value) |
|---|---|---|---|
| `nstep` | CONTROL | not an integer ≥ 1 | above 250 (150) |
| `mixing_beta` | ELECTRONS | not a number in (0, 1] | below 0.1 (0.4) |

Booleans and strings are errors; a warning is logged when the file is read.
The relaunches never lower `nstep` nor raise `mixing_beta`. `nstep` matters
for `relax` only.

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
- `kdens` computes the grid when the cell is known, with the vendored
  `MeshFromDensity(structure, kdens, material_type, force_gamma=True)`:
  `round(|b_i| · kdens)` subdivisions along each reciprocal vector `b_i`,
  whose length includes 2π (e.g. a cubic cell of 3.6 Å with `kdens = 6`
  gives 10 10 10); mesh centred on Γ (offset 0 0 0). In the workflows
  `material_type` comes from the calculation type: k3 = 1 for slabs and
  interfaces. For a single calculation it is `auto`: the structure is a
  slab, with k3 = 1, when the cell is at least 6 Å longer along z than the
  atoms, and a bulk otherwise.
- None of the modes chooses the sampling for convergence: the grid or the
  density is the user's choice.
- The steps `pes` and `pes_scf` of the interface workflow require `kdens`:
  the cell of the interface is known only after the matching. With another
  mode they stop with an error on `k_points.mode`.

## `[qe]`

The `pw.x` expected on the machine that runs the calculations.

| Key | Type | | Notes |
|---|---|---|---|
| `version` | str | R with `[interface]` | as `pw.x` prints it: `major.minor` (e.g. `7.2`, every `7.2.x`) or `major.minor.patch` (e.g. `7.3.1`, that one only); required when a step that runs calculations uses `qe` |
| `upf_md5` | str | — | manifest of the expected MD5 checksums of the UPF files, relative to the folder of this file |

- The manifest has the format of `md5sum`: one line `<checksum>  <file>`
  per file, the checksum in 32 lower-case hex digits. Make it on the HPC,
  in `pseudo_dir`, with `md5sum *.UPF > pseudo.md5`; `pw.x` computes the
  same checksum of each UPF file and prints it in the output. Every UPF file
  of `[[atomic_species]]` must be in it, once; other files may be listed.
  It is read when an input is generated.
- The workflow is built to work with QE 7.1.x, 7.2.x and 7.3.x. Another
  version (e.g. `8.0`) is accepted with a warning: the results are not
  guaranteed and may hold errors, so check them. The outputs of any version
  are read, with the same warning in the log.
- In the calculations that give the charge density the version also chooses
  the values of `disk_io` refused at the preparation, those with which
  `pw.x` writes no charge density; without a version, or with another
  release, the values of every release are refused
  ([qe_charge.md](qe_charge.md)).
- Version and checksums enter the energy model of every calculation (see
  [Fingerprint and energy model](#fingerprint-and-energy-model)); the path
  of the manifest does not. Each output is compared with them: the version
  printed by `pw.x`, on the components given (`7.3` matches `7.3.1`, `7.3.1`
  does not match `7.3.2`), and the label, UPF file and checksum of every
  species. An output that differs keeps its results, but they are never
  reused nor combined.
- **Without `version` or without `upf_md5`** the energy model is not fully
  known. The inputs are generated and the calculations can run, but their
  results are never reused and their energies are never combined
  automatically: an analysis that needs them (e.g. an adhesion energy)
  stops with an error that names the calculations and the missing parts.

## Fingerprint and energy model

Each calculation has a fingerprint, which tells whether an existing result
can stand for it, and an energy model, which tells whether its energy can be
combined with others.

- **Energy model:** `[qe] version`; for every species, in order, label, UPF
  file, expected checksum and `starting_magnetization`; the `[k_points]`
  settings (not the grid: with the same `kdens` a bulk and a slab have
  different grids); the namelist keys that change the energy (cutoffs,
  occupations and smearing, spin and magnetization, including `angle1` and
  `angle2`, functional, vdW, fields, charge, symmetry, …) and every key
  that is not in the lists of `triboworkflow/engines/qe/keys.py`, e.g. a key
  that `pw.x` reads but does not document, such as `B_field`.
- **Fingerprint:** the energy model, the calculation type, the structure
  and the `if_pos` flags, the masses, the `K_POINTS` card written and every
  namelist key written, except those that change how a run gets to the
  result (`nstep`, `electron_maxstep`, mixing and diagonalization keys,
  which the relaunches may change) and the resources (`max_seconds`,
  `prefix`, `outdir`, `pseudo_dir`, `wfcdir`, `title`, …).
- Keys are compared in lower case and without blanks; values keep their
  TOML type, so `ecutwfc = 40` and `ecutwfc = 40.0` count as different (at
  worst a calculation is run again). A `starting_magnetization` left out
  and one equal to 0 also count as different.

## `[generation]` (one calculation)

Read only with `[structure]`, by `triboworkflow gen` (`cli.md`); in the
workflow the file names are fixed. With `[structure]` and no `[generation]`,
every key takes its default.

| Key | Type | Default | Notes |
|---|---|---|---|
| `file_prefix` | str | `"input.pwi"` | name of the generated input, relative to the folder of the settings file; the extxyz of its structure has the same name with the suffix `.extxyz` |
| `overwrite` | bool | `false` | if false, an existing input or extxyz is an error and nothing is written |
| `atomic_positions_units` | str | `"angstrom"` | `angstrom` |
| `cell_parameters_units` | str | `"angstrom"` | `angstrom` |

## `[execution]`

Needed only to run on HPC. No value is tied to a specific machine.

| Key | Type | | Notes |
|---|---|---|---|
| `scheduler` | str | R | `slurm`, `pbs`, `torque`, `lsf`, `sge`, `htcondor`, `condor` (the names of Xsorb) |
| `jobscript` | str | R | user jobscript, relative to the folder of the settings file; copied for every group of jobs with the launch line of the worker added, never changed |
| `run_command` | str | R | command up to the input file, e.g. `mpirun -np 32 pw.x -nk 4` |
| `pp_command` | str | none | command of pp.x up to the input file, e.g. `mpirun -np 32 pp.x`: run after pw.x, in the same job, in the calculations that also give the charge density (the charge analysis, [qe_charge.md](qe_charge.md)); required only when such calculations are prepared or submitted, not empty |
| `walltime` | int | R | > 0 s, walltime of each job; keep it equal to the one of the jobscript |
| `polling_interval` | int | R | > 0 s, time between two checks of the jobs |
| `groups` | int | none | > 0, jobs (workers) over which the calculations of a group are spread: 1 runs them one after the other, one per calculation runs them all at once; never more jobs than calculations are submitted. When left out, give it on the command line (`--groups`); in a terminal the command asks for it, elsewhere it stops with an error |
| `max_seconds_margin` | int | `600` | > 0 s and < `walltime`, time kept free at the end of the walltime |
| `job_report_wait` | int | `3600` | > 0 s, time after its submission during which a job that the scheduler does not report is still waited for; after that it needs attention, and it is never submitted again automatically |

Before each run the worker writes `max_seconds` in the input it runs:
`walltime` minus the time since the worker started minus
`max_seconds_margin`, capped by the
`max_seconds` of `[namelists.CONTROL]` when given. pw.x then stops in time to
write its restart files. When nothing is left, the worker takes no more
calculations. The margin must cover the time between two checks of pw.x, the
writing of the restart files and the commands of the jobscript that run
before the worker; 600 s is a starting value. Both values are recorded with
each run.

### `[execution.analysis]`

Optional: the job of the analyses that need a compute node, i.e. the
interpolations of the PES (`plotpes`, `plotpes_scf`), whose memory grows with
the square of the points ([interface.md](interface.md#cost)), and the shear
strength (`shear`), which interpolates the PES again and then runs the string
method for a long time. The orchestrator never runs them on the login node:
with this table it submits each in a job of its own and follows it; without it
those steps wait to be analysed by hand (`triboworkflow analyse plotpes`). One
jobscript serves all of them: give it the memory and the walltime of the
heaviest.

| Key | Type | | Notes |
|---|---|---|---|
| `jobscript` | str | R | jobscript of these jobs, relative to the folder of the settings file; copied for each job with the line that runs the analysis added, never changed. Its lines give memory, walltime and partition (e.g. `#SBATCH --mem=200G`, `#SBATCH --time=04:00:00`) and the threads of the linear algebra (`export OMP_NUM_THREADS=...`); the scheduler is the one of `[execution]` |

A job that ends without completing its step is never submitted again
automatically: the step fails with the state of the job, and after the
jobscript is given what it needs `triboworkflow retry <step>` submits a new
one ([execution.md](execution.md), *Analyses in a job of their own*).

## `[interface]` (interface workflow)

| Key | Type | Default | Notes |
|---|---|---|---|
| `slabs` | list | R | two `[[interface.slabs]]` entries, slab 1 then slab 2: their order gives the interface id (`mid1_mid2`) and the strain weights (`strain_weight_1` goes with slab 1); the slab on top is decided by the matching (see below) |
| `steps` | list of str | R | steps to run, each at most once: `match`, `pes`, `pes_scf`, `plotpes`, `plotpes_scf`, `shear`, `adhesion`, `ppes`, `charge`; a step runs once the steps it needs are done, and among the steps that are ready the order of the list decides. `pes` is the potential energy surface relaxed along z, `pes_scf` the static one: two distinct steps, each run only when listed, also both; `plotpes` interpolates `pes` and `plotpes_scf` interpolates `pes_scf`, each run only when listed, also both ([interface.md](interface.md)); `adhesion` reads the PES chosen in `[interface.adhesion]`; `ppes` reads the interface that `adhesion` relaxed, of the kind of `[interface.adhesion]` and at the site of `[interface.ppes]`; `charge` reads the same interface and the density of the slabs; `shear` reads the PES relaxed along z that `plotpes` interpolated (so it needs `pes` and `plotpes`, not `pes_scf`). `plotpes`, `plotpes_scf` and `shear` are analysed in a job of their own with `[execution.analysis]`, or by hand (`analyse plotpes`) |
| `set_bm` | bool | `true` | strain weights from the bulk moduli of the two materials |
| `engine` | str | `"qe"` | engine of the steps that run calculations |
| `engines` | table | `{}` | per-step engine for `pes`, `pes_scf`, `adhesion`, `ppes`, `charge`; only `qe` for now |

`[[interface.slabs]]`: a slab and its identifiers.

| Key | Type | | Notes |
|---|---|---|---|
| `file` | str | R | slab structure, relative to the folder of the settings file; `""` is a placeholder. A pymatgen JSON slab is used as stored, any other file is turned into a slab ([structures.md](structures.md)) |
| `mid` | str | R | material id; the interface id is `mid1_mid2` |
| `miller` | 3 int | R | |
| `formula` | str | — | given for both slabs together with `name`, or for neither |
| `name` | str | — | tells apart homonyms (e.g. C diamond, C graphene); given for both slabs together with `formula`, or for neither |
| `bulk_modulus` | float | R with `set_bm = true` | bulk modulus of the material in GPa; read only with `set_bm = true` |

The ids of the interface are `mid1_mid2`, `formula1_formula2` and
`name1_name2`. Without `formula` and `name` the interface takes the reduced
formulas of the slabs and a name made of formula and Miller indices: Cu(111)
and Al(111) give `Cu_Al` and `Cu111-Al111`. `formula` without `name`, or the
reverse, or the ids of one slab only, are errors.

With `set_bm = true` the workflow sets `strain_weight_1` and
`strain_weight_2` to the two bulk moduli, replacing any value given in
`inter_params`. With `set_bm = false` the weights come from `inter_params`
(default 1:1).

`[interface.inter_params]`: the parameters of the interface matching; other
keys are errors. Missing keys take the defaults below; the values are
checked by the vendored `InterfaceMatcher`, except `best_match`, checked
here.

| Key | Type | Default | Notes |
|---|---|---|---|
| `max_area` | float | 160.0 | largest area of the interface cell, in angstrom^2 |
| `interface_distance` | float | 2.0 | distance between the two slabs, in angstrom; also in the calculations of the two PES |
| `max_angle_tol` | float | 0.4 | tolerance on the angles of the lattice match, the same at every attempt |
| `max_length_tol` | float | 0.02 | tolerance on the lengths of the lattice vectors |
| `max_area_ratio_tol` | float | 0.05 | tolerance on the ratio of the areas of the two cells |
| `strain_weight_1`, `strain_weight_2` | float | 1, 1 | weights of the strain of slab 1 and slab 2 |
| `best_match` | str | `"area"` | the only value: the matching returns the match of smallest area |

The matching (step `match`, [interface.md](interface.md)) puts on top the
slab whose formula and Miller indices come first in alphabetical order (in
the example, Al(111) on Cu(111)), whatever the order of `[[interface.slabs]]`; with the same
formula and indices, slab 1 goes on top. When no match is found, `max_area`,
`max_length_tol` and `max_area_ratio_tol` are multiplied by 1.01, then by
1.02, and so on, until a match is found; when the factor reaches 1.5 the step
stops with `Matching not converged`. `max_angle_tol` is never enlarged. The
values used are written in `inter_params` of the document and in the message
of the step.

`max_angle_tol` has two meanings in the matching code, which uses it twice:
in the search of the supercells it is relative (0.4 allows the angles of the
two cells to differ by 40%), in the mapping of each slab to its supercell it
is absolute, in degrees (0.4°).

To remove: the bulk moduli replace the weights as in TribChem's
`FT_StartInterfaceMatch.set_bulk_modulus`; the keys left out take first
160.0 for `max_area` and 2.0 for `interface_distance`, the defaults of
TribChem's command line, which passes them as floats (Q23, A76), and 0.4 for
`max_angle_tol`, then the defaults of `highput/firetasks/defaults.json`
(`max_area` 150 and `interface_distance` the integer 2 there). TribChem's Firetasks forced `max_angle_tol` to 0.4 at
every attempt, so the 0.01 of its command line had no effect; here a value
given is used at every attempt (DEV-01). `best_match = "mismatch"` had no
effect in TribChem (the match of smallest area is always taken), and is an
error here (DEV-34). The two meanings of `max_angle_tol` are those of
pymatgen 2024.6.10 (`ZSLGenerator` and `Lattice.find_mapping`, A74).
`formula` without `name` is refused by TribChem's own check of the inputs
(A75).

`[interface.pes_scf]`: the slabs of the static PES (step `pes_scf`,
[interface.md](interface.md)); other keys are errors. There is no
`[interface.pes]` table: the relaxed PES always uses the complete slabs.

| Key | Type | Default | Notes |
|---|---|---|---|
| `short` | bool | `false` | `false`: the complete slabs; `true`: each slab reduced to its layers next to the interface |
| `layers` | 2 int | 3 for a (111) slab, 2 for the others | layers kept in the bottom and in the top slab, each at least 1; only with `short = true`, otherwise an error |

Bottom and top are those of the interface, decided by the matching, not
slab 1 and slab 2: in the example, `short = true` keeps 3 layers of Cu(111),
below, and 3 of Al(111), above; `layers = [2, 1]` keeps 2 of Cu and 1 of Al.

To remove: TribChem's command line always passed the layers, so its static
PES was always on reduced slabs (here the complete slabs are the default,
D07), and it applied the 3/2 rule to the slabs in the order given, while the
values went to the bottom and the top slab (DEV-38, A81).

`[interface.adhesion]`: the PES that gives the two sites of the adhesion and
the kind of adhesion (step `adhesion`, [interface.md](interface.md)); other
keys are errors. One choice for the whole run.

| Key | Type | Default | Notes |
|---|---|---|---|
| `pes` | str | `"pes"` | `"pes"` (relaxed along z) or `"pes_scf"` (static): the step whose minimum and maximum the adhesion is computed at; the step `adhesion` waits for it |
| `type` | str | `"regular"` | `"regular"` or `"short"`. With `pes`, `regular` starts from the interfaces that the PES relaxed. With `pes_scf`, `regular` stacks the complete slabs at the sites of the static PES, and `short` starts from the reduced slabs of its points |

The allowed combinations are (`pes`, `regular`), (`pes_scf`, `regular`) and
(`pes_scf`, `short`). (`pes`, `short`) is an error, and `short` needs
`short = true` in `[interface.pes_scf]`, whose reduced slabs it uses. The
calculation types are not settings: the two slabs and the interface at the
lowest site relax all the atoms, the interface at the highest site only along
z.

To remove: TribChem's `FT_StartAdhesion` read the PES (`pes_type`, default
`pes`) and the kind (`adhesion_type`, default `regular`) as two options; its
command line allowed (`pes`, `regular`) and (`pes_scf`, `short`), and the
code refused (`pes`, `short`) (D56).

`[interface.ppes]`: the site, the shifts and the fit of the perpendicular
PES (step `ppes`, [interface.md](interface.md)); other keys are errors. One
site for the whole run. There is no `type`: the kind of interface, `regular`
or `short`, is the `type` of `[interface.adhesion]`, since the step starts from
the interface that `adhesion` of that kind relaxed.

| Key | Type | Default | Notes |
|---|---|---|---|
| `site` | str | `"min_site"` | `"min_site"` or `"max_site"`: the interface relaxed at the site of lowest or of highest energy of the surface chosen in `[interface.adhesion]`. To compute the other site, change `site` and run again: the results of the two sites stay side by side |
| `shifts` | list of float | the 15 values below | shifts of the slab on top from the relaxed interface, in angstrom, a negative one bringing the slabs closer; at least one, finite, without repetitions; one pw.x calculation (`scf`, no movement of the atoms) for each |
| `fit` | str | `"none"` | `"none"`: only the energies are written; `"uber"`: also the fit of the universal binding energy relation. Any other name, `"uber0"` included, is an error |

The default shifts are -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5,
1.75, 2, 2.25, 2.75 and 3 Å: **there is no 2.5**, the list jumps from 2.25 to
2.75 and each site needs 15 calculations. To cover the whole range, or to get
more points at large distance, list the shifts in the table.

The fit uses only the shifts that are zero or positive, and needs at least
three of them: with fewer the fit fails and the analysis stops without
writing anything; with exactly three it runs but its standard deviations are
infinite, because the covariance cannot be estimated (the step warns); from
four on they are finite. It gives `g`, the depth of the curve, in eV per
cell (the fit receives the energies in eV, not in J/m^2), `l`, its length, in
Å, and the standard deviations `sigma_g` and `sigma_l` in the same units.
Changing only `fit` computes nothing again: the calculations are the same, and
the step only analyses them anew.

To remove: TribChem's command line passed a list of shifts, `-sh`, whose
default is this one (the jump over 2.5 is in the list, and the documentation
that says a step of 0.25 Å from -0.75 to 3 Å does not match it, A90), and never
passed `fit`, so `FT_StartPPES` ran no fit (A88): here too `"none"` is the
default, and `"uber"` has to be asked for. In `fit_ppes` only `"uber"` works:
`"uber0"` raises an error (A06) and any other name too (A89), so they are
refused when the settings are read, not after the calculations.

`[interface.charge]`: the site and the operations of the charge displacement
(step `charge`, [interface.md](interface.md)); other keys are errors. One site
for the whole run. There is no `type`: the kind of interface, `regular` or
`short`, is the `type` of `[interface.adhesion]`, since the step starts from
the interface that `adhesion` of that kind relaxed. The step needs
`[execution] pp_command` and `[k_points]` with `mode = "kdens"`
([qe_charge.md](qe_charge.md)).

| Key | Type | Default | Notes |
|---|---|---|---|
| `site` | str | `"min_site"` | `"min_site"` or `"max_site"`: the interface relaxed at the site of lowest or of highest energy of the surface chosen in `[interface.adhesion]`. To compute the other site, change `site` and run again: the calculations are named after the site and the results of the two sites stay side by side |
| `operations` | 2 str | `["-", "-"]` | `"+"` or `"-"` each: the density of the interface combined with that of the bottom slab and then with that of the top slab, so the default is interface − bottom slab − top slab. Any other text, and any number of operations but two, is an error. Changing them computes nothing again: the calculations are the same, and the step only analyses them anew |

The integration range, the axis and the number of points are not settings.
The range is the region between the two slabs, estimated from the interface;
the axis is z; the integral is computed on 1000 points. The step runs three
`scf` calculations in the same cell, each with its charge density: the
interface and, with the atoms where they are in the interface, its bottom and
its top slab.

To remove: TribChem's command line passed the operations as the text
`'[-, -]'` (`-op`) and the site (`-s`, default `min_site`), and nothing more:
the integration range is always estimated (a range given by the user makes the
workflow fail, A32), the axis is 2 and the number of points is 1000, which
`defaults.json` says and the signatures of the functions contradict with 10000
(A93). The original did not run the other site once one was done (A59): here
the site is part of the calculations and of what is done (D59).

`[interface.shear]` and `[interface.shear.mep]`: the shear strength (step
`shear`, [interface.md](interface.md)); other keys are errors. The step reads
the PES relaxed along z that `plotpes` interpolated (the surface is not a
setting) and finds three paths of minimum energy on it, one from each starting
string: the best straight line, the axis x and the axis y of the cell. The
shear strength along each path is the largest force along it, in GPa. The
defaults are the recommended values: the step logs which parameters of a run
are not the recommended ones. Changing any key
computes the shear strength again. The step has no calculation of its own: it
runs in a job of its own with `[execution.analysis]`, or by hand with `analyse
shear`, where there is enough memory.

| Key | Type | Default | Notes |
|---|---|---|---|
| `delta` | float | `0.01` | step of the finite-difference derivative of the surface along each path, in Å, that gives the force and so the shear strength; finite and > 0. Not the `delta` of `[interface.shear.mep]` |
| `seed` | int | `0` | seed of the random numbers; from 0 to 4294967295, the range of the seed of NumPy. The best straight line starts from a string perturbed by a random noise of about 1e-4 of the cell, and the seed is set before each string: the same seed gives the same result, another seed is another sample of the same noise (its effect on the shear strength is small: [interface.md](interface.md)). The axes x and y do not use random numbers |
| `mep` | table | below | the parameters of the string method |

`[interface.shear.mep]` has the parameters of the string method, the same for
the three paths:

| Key | Type | Default | Notes |
|---|---|---|---|
| `npts` | int | `101` | points of the string; ≥ 6, because the entries of the shear strength hold the values at the first six points of the path ([interface.md](interface.md), *Entries and files*) and a shorter string would fail when they are written, after it is computed |
| `extent` | 2 float | `[1.5, 1.5]` | extent of the starting string along x and along y, in lengths of the cell along those axes: it goes from −`extent` to +`extent`; finite and > 0 |
| `h` | float | `0.01` | time step of the evolution of the string; finite and > 0 |
| `nstepmax` | int | `99999` | maximum number of steps; ≥ 1 |
| `tol` | float | `1e-8` | the string stops when the mean displacement of its points in a step is at most this; finite and > 0 |
| `delta` | float | `1e-4` | step of the finite-difference derivative of the surface inside the method, in Å; finite and > 0. Not the `delta` of `[interface.shear]` |

The method (`zerotemp`), the three starting strings, the interpolation of the
surface (thin-plate spline, smoothing 1e-3, points replicated 5 × 5 around the
cell, grid of density 20 that the shear does not use) and the entry
(`shearstr`) are not settings. The cell must be a square with its sides along x
and y: the starting strings are built on the two diagonal elements of the cell
only, and the step warns when the cell is not a square.

To remove: the options of TribChem's command line `-npts`, `-fac`, `-hs`,
`-nstepmax`, `-tol` and `-opt` never reached the string method (A94): its
values were those of `defaults_tribo.json`, which are the defaults here, and
not the ones that the command line and `defaults.json` announced (`h` 0.001,
`tol` 1e-7, `delta` 0.01); the three paths were always computed. Only `-delta`
(0.01) acted, on the shear strength. Here every key acts (DEV-47), and the
default of each is the value that TribChem used. The noise of the starting
string was not seeded in TribChem (Q02, DEV-46).
