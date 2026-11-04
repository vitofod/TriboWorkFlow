# Charge density of a pw.x calculation

The charge analysis of an interface subtracts, point by point, the electron
density of the interface and those of its two slabs, computed in the same
cell. With QE the density of a calculation comes from pp.x, run after pw.x
on the files that pw.x leaves in the `outdir`.
`triboworkflow.engines.qe.density` reads the file that pp.x writes and gives
it the form the analysis works on; `engines/qe/ppi.py` writes the input of
pp.x and reads its output. The file of the density of a calculation is the
one of the post-processing of its last attempt (`density.density.path` of
the attempt in the record: `density.filplot`, or `attempts/<n>/...` once
archived).

```python
from triboworkflow.core.structures import read_structure
from triboworkflow.engines.qe import density

f = density.read_filplot('systems/Cu111_Al111/charge/interface/density.filplot')
f.grid, f.electrons                       # (nr1, nr2, nr3), electrons in the cell
chg = density.to_chgcar(f, read_structure('.../structure_out.extxyz'))
chg.data['total'].mean()                  # electrons: data is rho * V, as in a CHGCAR
```

## Preparation

The step decides which calculations also give the charge density (the charge
analysis: the interface and its two slabs); there is no key of the settings
for it. Such a calculation (`preparation.workflow_calculation(...,
density=True)`) gets, next to `input.pwi`, the pp.x input `input.ppi`
(`engines/qe/ppi.py`):

```
&INPUTPP
   prefix           = '<step>_<label>'
   outdir           = 'OUT'
   filplot          = 'density.filplot'
   plot_num         = 0
   spin_component   = 0
/
```

with the `prefix` and `outdir` of its pw.x input and no `&PLOT`, so that pp.x
only writes `density.filplot`. pp.x reads the XML file, the pseudopotentials
and the charge density that pw.x leaves in `OUT/`, not the wavefunctions. The
pw.x input and the energy model are those of the same calculation without
density; the record has `density` true and `files.pp_input`, and the
fingerprint takes the post-processing too (`pp.x`, `plot_num`,
`spin_component`), so the same calculation with and without density are two
calculations ([store.md](store.md)).

Such a calculation requires:

- `[execution] pp_command` ([configuration.md](configuration.md));
- a `disk_io` with which pw.x writes the charge density, otherwise the
  preparation stops with an error. The values that leave no charge density
  depend on the expected release of pw.x (`[qe] version`):

  | `[qe] version` | refused `disk_io` |
  |---|---|
  | 7.1.x, 7.2.x | `'nowf'`, `'none'` |
  | 7.3.x | `'minimal'`, `'none'` (`'nowf'` writes the density there) |
  | not given, or another release | `'nowf'`, `'minimal'`, `'none'` |

  In QE 7.1 and 7.2 `'minimal'` is not a value of its own and pw.x takes the
  default, which writes the density. The value is compared as pw.x compares
  it: as written, trailing blanks dropped, upper and lower case apart
  (`'NOWF'` and `' nowf'` are not those values for pw.x, which then writes
  the density). In any other calculation `disk_io` is written as given,
  whatever its value.

## Execution

pp.x runs in the same job as pw.x, right after it, in the calculation
directory, through the worker ([execution.md](execution.md)):

```
<pp_command> -in input.ppi >> output.ppo
```

only after an attempt whose output was parsed (pw.x over with the SCF and
the geometry converged, results read). The attempt keeps the
post-processing (`density` of the attempt, [store.md](store.md)): program,
command, `input.ppi`, `output.ppo` and `density.filplot` with their
digests, exit code, and the reason when the density cannot be used. Its
outcome is the state `density` of the record:

- `ok`: the last run of `output.ppo` closes with `JOB DONE.` and has no
  `Error in routine`, and `density.filplot` is read completely, is the
  charge density (`plot_num = 0`) and agrees with `structure_out.extxyz`;
- `failed` otherwise, with the reason (an error of pp.x with its routine
  and message, a run not closed, the file missing, incomplete or of another
  structure).

The exit code of pp.x is recorded; the state comes from the output and the
file. A record that gives the charge density is complete only with the
state `ok`. A density that failed is never produced again automatically:
the calculation goes to the user, and a manual relaunch runs pw.x from
scratch, then pp.x. pp.x runs in the time `max_seconds_margin` keeps free at
the end of the walltime, with no limit of its own.

## The file of pp.x

pp.x run with `plot_num = 0` and `spin_component = 0` writes the total
electron (pseudo-)charge density on the real-space grid of the density, in
the file named by `filplot`: also with `nspin = 2` or noncollinear spin, it
is the total density. The values are in e/bohr³ and integrate to the number
of electrons. The layout, with the Fortran formats of QE 7.1–7.3
(`Modules/plot_io.f90`):

| Line | Content |
|---|---|
| `(a)` | title, 75 characters; blank for the density |
| `(8i8)` | `nr1x nr2x nr3x nr1 nr2 nr3 nat ntyp` |
| `(i6,2x,6f16.8)` | `ibrav`, `celldm(1:6)` (`alat` in bohr) |
| list-directed, three lines | cell vectors in units of `alat`, only with `ibrav = 0` |
| `(3f20.10,i6)` | `gcutm` (units of (2π/`alat`)²), `dual`, `ecutwfc` (Ry), `plot_num` |
| `(i4,3x,a2,3x,f5.2)`, one per species | index, name (the first two characters of the label), valence |
| `(i4,3x,3f15.9,3x,i2)`, one per atom | index, position in units of `alat`, species |
| `5(1pe17.9)` | `nr1x * nr2x * nr3` values, x fastest |

`nr1x`, `nr2x` are the leading dimensions of the FFT arrays: equal to `nr1`,
`nr2` with FFTW, one more with ESSL (sizes that are powers of 2) or on NEC SX
(even sizes). The padding is dropped on reading. The grid points are at
`(i/nr1, j/nr2, k/nr3)` in crystal coordinates, the first at the origin of
the cell.

Reading (`read_filplot`, `parse_filplot`) is strict: every line in its
place, species and atoms numbered in order, exactly `nr1x * nr2x * nr3`
values, every number finite. A field that did not fit its format (`****`)
is an error; an exponent of three digits, which Fortran writes without the
`E` (`1.234567890-100`), is read.

## The CHGCAR form

`to_chgcar(filplot, structure)` gives a pymatgen `Chgcar` with the meaning
and the units of a VASP CHGCAR read by pymatgen:

| | |
|---|---|
| `data['total'][i, j, k]` | density (e/Å³) × volume of the cell (Å³), along the three cell vectors |
| `structure` | the structure of the calculation, in Å |

So `data['total'].mean()` is the number of electrons and
`get_average_along_axis(axis) / volume` the average density in e/Å³, as the
charge analysis computes it. The conversions use the Bohr radius of QE 7.1–7.3
(CODATA 2018).

The structure is that of the calculation (`structure_out.extxyz`: the input
of a `scf`, the final structure of a `relax`) and must agree with the file
(`structure_differences`): the same number of atoms, the same labels (their
first two characters, which the file keeps), the cell and the positions to
the digits printed (eight decimals of `celldm(1)` in bohr, nine of the
positions in units of `alat`, eight of the extxyz in Å). Only `ibrav = 0`,
the cells of this package, gives the cell: files with another `ibrav` are
read but not converted. A file that is not the charge density
(`plot_num ≠ 0`) is not converted either.

## Densities that can be combined

The densities of the interface and of its slabs are combined point by point.
`comparison_differences({name: filplot})` lists where they differ from the
first one; `check_comparable(filplots, records)` raises an error with every
difference:

- the charge density (`plot_num = 0`) in every file;
- the same grid `nr1, nr2, nr3`;
- the same cell, to the digits printed (`ibrav = 0` in every file); the grid
  starts at the origin of the cell in every file, so the same cell is the
  same grid in space;
- the same `ecutwfc`, `dual` and `gcutm`;
- with the records of the calculations, one energy model shared by all, as
  for any difference of energies (`core.store.check_energy_model`,
  [store.md](store.md)).

## Tests

`tests/test_qe_density.py` uses synthetic files written in the formats of
pp.x (`tests/charge_files.py`); `tests/test_qe_ppi.py` the pp.x input, the
check of `disk_io` and the fingerprint; `tests/test_preparation.py`,
`tests/test_worker.py`, `tests/test_reconcile.py` and `tests/test_jobs.py`
the preparation and the execution, with a fake pp.x that prints a synthetic
output and writes the file given by the test (pp.x never runs). With `TRIBOWORKFLOW_QE_PP_EXAMPLE` set to the
folder of the pp.x example of QE (`PP/examples/example01`, never stored in the
repository), it also reads `reference/sicharge` (silicon, `ibrav = 2`, 20³
points, 8 electrons). The golden cases `chgcar_*` and `qecharge_*` of
`tests/golden_cases.py` characterise the charge analysis on files of the
same kind.
