# Results of a pw.x output

`triboworkflow.engines.qe.results` reads, from the output of a `scf` or
`relax` calculation, the results of its record and the state `parsing`.
Whether the calculation is over is read by `status.py`
([qe_status.md](qe_status.md)); `execution/parsing.py` runs the reading on
the files of a calculation directory.

```python
from triboworkflow.execution.parsing import parse_calculation

parsed = parse_calculation('systems/Cu111_Al111/pes/aa')  # input.pwi, record.json,
parsed.parsing, parsed.results, parsed.observed           # structure_in.extxyz, output.pwo
```

## When the output is read

Only when the states of its last run say that the calculation is over with
the SCF converged and, for a `relax`, the geometry converged; otherwise
`parsing` and the results stay null. Then `parsing` is:

- `ok`, with the results and `structure_out.extxyz`, when the results
  required are there and agree with the input;
- `failed`, with the reason in the attempt (`parse_error`) and no results,
  otherwise.

| Required | `scf` | `relax` |
|---|---|---|
| total energy of the last SCF cycle | yes | yes |
| total forces of the last SCF cycle | no (only with `tprnfor`) | yes |
| final coordinates, closed (`Begin` … `End final coordinates`) | — | yes |
| number of atoms, labels and cell of the header equal to the input | yes | yes |
| positions of the header equal to the input | yes | no: a restart prints the positions of the restart file |
| labels and `if_pos` of the final coordinates equal to the input | — | yes |
| `Final energy` equal to the last `!` line, when printed | — | yes |

"Equal" allows the rounding of the printed values (QE 7.1 and 7.2): six
decimals for `celldm(1)` and the cell in units of `alat`, seven for the
positions in units of `alat`, eight for the `!` line, ten for `Final
energy`, and the eight decimals of the positions of `structure_in.extxyz`.
A value that is not a finite number (`NaN`, or `****` where it does not fit
its format) fails the reading, whatever the value.

## Results

The keys and units of the task documents that the workflow code reads, as
atomate wrote them for VASP:

| Key | Value |
|---|---|
| `nsites` | number of atoms |
| `input.structure` | structure of `structure_in.extxyz`, as a pymatgen `Structure` dict |
| `output.structure` | final structure, as a pymatgen `Structure` dict, in angstrom |
| `output.density` | density of the final structure, g/cm³ |
| `output.energy` | total energy, eV |
| `output.energy_per_atom` | `output.energy` / `nsites`, eV |
| `output.forces` | total force on every atom, eV/Å, as printed (fixed components included); only when printed |
| `output.stress` | stress tensor, kbar, with the sign pw.x prints (the one of VASP); only when printed |

- **Energy.** `Final energy` in a `relax`, the last `!    total energy` in
  a `scf`: the total energy of pw.x, smearing term included (`F = E − TS`),
  nothing subtracted or extrapolated. VASP gave the energy extrapolated to
  zero smearing: only differences between energies of one energy model are
  used. A `relax` without `Final energy` takes the last `!` line.
- **Final structure.** Cell, labels, `if_pos` and the other site properties
  of the input, with the final positions of a `relax` or those of the input
  of a `scf`, not moved into the cell. `magmom` (an initial moment of the
  input) is left out: atomate put final moments there, which are not read.
- **Keys of VASP only** (`bandgap`, `cbm`, `vbm`, `is_gap_direct`,
  `input.pseudo_potential`) are not written; the workflow code reads them as
  null, as any missing key.
- **Constants** of QE 7.1–7.3 (`Modules/constants.f90`, CODATA 2018):
  `RY_TO_EV`, `BOHR_TO_ANGSTROM`, `RY_KBAR`.

## Identity of the output

`observed` of the record holds what the output tells of the energy model,
in its form, and the store compares it with the energy model expected
([store.md](store.md)):

| Key | Value |
|---|---|
| `version` | version printed, cut to the components of `[qe] version`: `7.3` expected stands for every `7.3.x`, `7.3.1` for that one only |
| `species[i]` | `label`, `pseudopotential` (name of the UPF file), `md5` (null when not printed) and `path` (as printed) of species `i` of `ATOMIC_SPECIES` |

An output that differs keeps its results and `parsing` `ok`, and is never
reused nor combined. The whole version printed is `engine.version`. Outputs
of a version outside 7.1.x–7.3.x are read with a warning in the log: the
results are not guaranteed, and the user checks them.

## Lines read

Only the last run of the output, and of it the last SCF cycle (after its
last `Self-consistent Calculation` line), as in Xsorb
(`xsorb/ase_custom/espresso.py`, `read_espresso_out_custom`, after ASE's
`read_espresso_out`), with the lines recognized at their beginning:

| Line | Value |
|---|---|
| `number of atoms/cell`, `number of atomic types`, `celldm(1)` | summary of the run |
| `crystal axes:` and `a(i) = ( … )` | cell, in units of `alat` |
| `PseudoPot. # i for X read from file:`, next line, `MD5 check sum:` | UPF file and checksum of species `i` |
| `atomic species   valence    mass     pseudopotential` | label of each species |
| `positions (alat units)` | label and position of each atom (ASE's `parse_position_line`) |
| `!    total energy = … Ry` | total energy |
| `Forces acting on atoms (cartesian axes, Ry/au):` | total forces, the first block (not the contributions of high verbosity, nor `Dispersion contribution to forces`) |
| `total   stress  (Ry/bohr**3)` | first three columns, Ry/bohr³ |
| `Final energy = … Ry` | `relax` |
| `Begin final coordinates` … `ATOMIC_POSITIONS (angstrom)` … `End final coordinates` | `relax`, angstrom only (ASE's `get_atomic_positions`) |

Unlike Xsorb: one run and one SCF cycle instead of every configuration; the
values in the units of pw.x, converted with the constants of QE instead of
those of `ase.io.espresso` (CODATA 2006); the stress with the sign of pw.x
(Xsorb changes it to that of ASE); the forces as printed (an `Atoms` of
Xsorb applies its constraints in `get_forces()`); `Final energy`, the
species table and the checksums read; strict blocks, one line per atom,
numbered.

## Tests

`tests/test_qe_results.py` and `tests/test_parsing.py` use synthetic outputs
(`tests/pwo_helpers.py`). `tests/test_qe_results_corpus.py` reads the real
outputs of `TRIBOWORKFLOW_PWO_CORPUS`, never stored in the repository, and
compares the values with those of ASE's `read_espresso_out`.
