# Report on a run of the interface workflow

`tests/hpc/interface_check.py report` reads the folder of a run of the
interface workflow, made with the `triboworkflow` commands on an HPC machine,
and says for each topic whether the files agree with what the package
promises. It is the tool to look at a run that has just finished, or a part
of it: whether the steps hand each other the right data, whether the jobs
that interpolate the PES and compute the shear strength did their work, and
what they cost.

It is read only. It writes nothing, in the run or elsewhere; it runs no
program, no scheduler and no calculation. It needs the virtual environment of
the package (it reads the records and the settings with the package), not the
HPC machine: copy the run and read it where you like.

```bash
rsync -a --exclude OUT/ <run>/ <local folder>/<run>/       # on your computer
python tests/hpc/interface_check.py report <local folder>/<run> [--sacct <file>]
```

The output directories of pw.x (`OUT/`) are not needed. Copy the rest: the
records, the outputs (`output.pwo`), the structures, the group folders with
the logs of the jobs (`jobs/*/job.*.err`), the figures, the slab files, and
the charge densities (`density.filplot`, large) if you want the charge checked.
The path of the run in the report is where you copied it.

## The file of `sacct`

Memory, walltime and the state of the jobs are known to the scheduler only.
On the machine, once the jobs are over, save its accounting outside the run
(the run is not changed) and copy it next to the run:

```bash
sacct -P -j <ids> -o JobID,JobName,State,ExitCode,Elapsed,Timelimit,ReqMem,MaxRSS,NodeList > sacct.txt
```

`<ids>` are the job ids of the groups, comma separated: they are in
`jobs/*/submitted.json` (`job_ids`). Give the file with `--sacct`. Without it
the memory and the walltime of the jobs are `not observed`, and so is the
`accounting` topic. `MaxRSS` is the largest of the job and its steps.

## Verdicts

One verdict per topic, as `qe_check.py report` gives them: `ok`; `note`, a
difference to look at; `not observed`, what the run did not show (a step not
run yet, a file not copied, no `sacct` file); `PROBLEM`, an inconsistency of
the files or a behaviour other than the package's. The exit status is 1 when
a verdict is `PROBLEM`.

| Topic | What it reads | `PROBLEM` when |
|---|---|---|
| `steps` | `orchestrator.json`; the entries of `system.json` of each step, done and current | a step is `failed` or `blocked`. `not observed` while a step is waiting, running or `manual` |
| `<step> calculations` | the record of each calculation of `pes`, `pes_scf`, `adhesion`, `ppes`, `charge` | parsing is not `ok`; the version or the pseudopotentials (checksums) printed differ from the expected; `structure_out.extxyz` is missing; the charge density is not `ok`; the calculations of the step do not share one energy model |
| `spin` | `nspin` and `starting_magnetization(i)` of every `input.pwi`, against `[[atomic_species]]` | `nspin` differs among the inputs, or a magnetization differs from the settings. `note` when none is spin-polarized |
| `slab 1`, `slab 2` (`slabs` when a file cannot be read) | the slab files as the workflow reads them ([structures.md](structures.md)) | a file cannot be read (e.g. its c is not perpendicular to a and b). `note` when a slab was rotated or lost data: it says which |
| `pes`, `pes_scf` | the entries `<pes>.data`, `energy.grid`, `min_site`, `max_site`, `structure.opt` | a key of the results is missing (the verdict names the point and the key; the other comparisons go on); an energy in eV differs from `Final energy` or the last `!` line of its output times the Ry in eV; `hs_unique_ev` is not sorted, or `min_site` and `max_site` are not its ends; a row of `hs_unique_ev` does not have the energy of its point or its first lateral shift; a row of `hs_unique_jm2` is not the same row with the energy converted to J/m^2 on the area of the cell of the lowest point |
| `plotpes`, `plotpes_scf` | `energy.pes_ev`, `pes_jm2`, `grid.hs_all_*`; the figure | the minimum of `pes_ev` is not 0, or the figure is not a PNG |
| `plotpes job`, `plotpes_scf job`, `shear job` | the group folder of the analysis (`outcome.json`, `submitted.json`, the logs), the line of `sacct` | the outcome is not `analysed` or `done`, or the group folder is missing. The text gives the job, the time of the outcome, the size of the log and the warnings; with `--sacct` also the state, the walltime used and the memory |
| `adhesion` | `structure.opt.<type>.*`, `data.*`, `adhesion.<type>` | a slab is not a `Slab` with Miller indices; an interface carries no `interface_region` on both slabs; the adhesion energies or the corrugation differ from those recomputed from the energies of the four calculations. The text gives the k-point grids |
| `ppes` | the input of each shift against the interface of the adhesion | an atom of the bottom slab moved, or an atom of the top slab moved other than by the shift along z, by more than 1e-5 Å |
| `charge fragments` | the inputs of the three calculations against the interface of the adhesion and its `interface_region` | the interface of the first calculation is not the interface of the adhesion (same atoms, moved together); a fragment does not have the species, labels, order and coordinates of the atoms of its slab in that interface; the three are not in one cell |
| `charge densities` | the three `density.filplot` | one that is there cannot be read (the verdict gives the file and the error); they are not on one grid, cell, cutoffs and energy model. `not observed` only when a file is missing. The text gives the time to read them and, on Linux, the peak memory of the report |
| `charge contact plane` | `charge.<type>.displacement.<site>.z` | the region between the slabs has no width or no finite integral |
| `shear time`, `shear warnings` | the log of the job of the shear | `note` when the log has `DeprecationWarning` lines; the text gives their number, the warnings and the log size |
| `shear convergence` | `shearstr.*` against `[interface.shear.mep]` | a shear strength is not a finite number. `note` when a string stopped at `nstepmax` without reaching `tol`: it is written like one that converged ([interface.md](interface.md), *Step `shear`*) |
| `shear cell` | `a`, `b` of `structure.init.interface` against the cell of the lowest point of the PES | the difference is above 1e-3 Å. `note` above 1e-6 Å |
| `accounting` | the file of `sacct` | a job ended `OUT_OF_MEMORY` or `TIMEOUT`. `note` when some jobs are not in the file |

A check that meets an entry or a file of a shape it does not expect does not stop
the report: its verdict is a `PROBLEM` that says the check stopped, with the error,
and the other topics go on.

## What it does not say

The report checks the files against one another and against the output of
pw.x. It does not check that the numbers are right. A `PROBLEM` is a file
that disagrees with another; an `ok` is that none does.

The run that `tests/test_interface_chain.py` makes to test the report has a
stand-in for pw.x (ASE's EMT, a classical potential) and for pp.x (a uniform
density): the surfaces vary from point to point, but the charge displacement
is zero and the energies are not DFT. It shows that the steps pass the right
data along, and that the report catches the breaks that
`tests/test_interface_check.py` makes on a copy.
