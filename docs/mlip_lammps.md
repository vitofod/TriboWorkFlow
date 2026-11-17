# MLIP calculations with LAMMPS

A calculation with a machine-learned interatomic potential (MLIP) through LAMMPS
is a single point or a relaxation of one structure. LAMMPS is a program that the
workflow starts, as it starts `pw.x`, so there is no executor script: a single
module, `triboworkflow/engines/mlip/lammps_job.py`, writes what LAMMPS reads and
reads what it wrote. It imports no MLIP library and runs wherever the workflow
does. The command that starts LAMMPS (the executable with the plugin of the
model, `-in input.lammps -log log.lammps`, and the options of its build) belongs
to the configuration of the machine and is not interpreted here.

```python
from triboworkflow.engines.mlip import lammps_job

lammps_job.remove_outputs(calc_dir)                    # files of an earlier attempt
lammps_job.write_input(calc_dir, 'relax', settings.mlip, structure, max_seconds=3600)
# in calc_dir, on a GPU node:
#   <lmp> -in input.lammps -log log.lammps
status = lammps_job.lammps_status(calc_dir, 'relax')
parsed = lammps_job.parse_result(calc_dir, 'relax', structure)   # structure of the input
start_again_from = lammps_job.last_structure(calc_dir, structure)
```

Calculations run on a GPU only. The model is never loaded by the workflow.

## Files of a calculation

All in the directory of the calculation; LAMMPS is started there.

| File | | Content |
|---|---|---|
| `input.lammps` | input | the script |
| `structure.data` | input | cell and positions of the atoms, in the frame of LAMMPS |
| `log.lammps` | output | the log of LAMMPS |
| `trajectory.dump` | output | positions every 10 iterations (relaxation only) |
| `final.dump` | output | positions and forces of the final structure, constraints off |
| `result.txt` | output | energy and pressure tensor of the final structure |

`remove_outputs` deletes the four output files; call it before an attempt, so
that each file belongs to the last execution.

## The script

`lammps_input(calculation, mlip, structure, max_seconds)` gives the text. For a
relaxation, in this order:

1. `units metal`, `atom_style atomic`, `newton on`, `boundary p p p`; `atom_modify
   map yes` for MACE; `box tilt large` if the cell is tilted by more than half of
   its length (a hexagonal cell, tilted by exactly half, does not need it).
2. `read_data structure.data`, then `pair_style` and `pair_coeff * *` followed by
   the elements, in the order of the atom types of the data file (the order in
   which the elements first appear in the structure).
3. The constraints: atoms that share the same locked components form a group
   (`group blocked_xy id 5:8`) with `fix lock_xy blocked_xy setforce 0.0 0.0
   NULL`. A flag of 0 in `if_pos` locks a component, which gets `0.0`; a free one
   gets `NULL`. A calculation of type "z only" (`0 0 1`) is then `0.0 0.0 NULL`.
   Atoms with all components free are in no group.
4. The trajectory: positions every 10 iterations, flushed at each snapshot, so
   that a killed run leaves its last whole snapshot.
5. `timer timeout` (if `max_seconds` is given), `min_style cg`, `min_modify norm
   max`, `minimize 0.0 <fmax> <maxsteps> 1000000`.
6. The locks and the timer are removed, and `run 0` evaluates the model on the
   final structure: energy, forces and stress **without constraints**. The forces
   go to `final.dump`; the energy and the six components of the pressure tensor go
   to `result.txt` with a `print`.

A single point is the `run 0` alone, on the structure as it is.

| Type | `pair_style` |
|---|---|
| `DP` | `deepmd <path>`: the path of the model, as written in `[mlip]` |
| `MACE` | `mliap unified <path> 0`: `path` is the file already converted for LAMMPS (`create_lammps_model.py` of MACE; the name ends with `.model-mliap_lammps.pt`). A `.model` is an error. |

`MACE_MP` is not available with LAMMPS. The `kwargs` of `[mlip]` are the
arguments of the loaders of ASE and LAMMPS has none: a model with `kwargs` is an
error rather than silently ignored. The precision and the head of the model are
those of the file, which is why the name of a derived file carries them.

### The relaxation criterion

The relaxation stops when the largest force on **one atom**, with the constraints
applied, is below `fmax`. LAMMPS by default compares the norm of the forces of
all atoms with the tolerance, which depends on the number of atoms and is
stricter than ASE's `fmax`; `min_modify norm max` makes the two the same
criterion. `etol` is 0, so the energy never stops the run. `maxsteps` is the
number of iterations; the force evaluations are limited to 1000000, the bound
SCS uses. There is a single minimizer, the conjugate
gradient `cg`: nothing falls back on another, unlike the relaxation with ASE.

### Cell and frame

LAMMPS requires the first cell vector along x and the second in the xy plane.
The cell of a structure is in general not so, so ASE's `Prism` rotates it: the
data file holds the rotated cell and positions, and everything that comes back
(positions, forces, the pressure tensor) is rotated back with the same `Prism`.
The results are in the frame of the input structure. Positions are not wrapped
into the cell when the data file is written.

## States

`lammps_status(calc_dir, calculation)` gives the states of the record from the
log:

| Log | `process` | `scf` | `geometry` | `stop_reason` |
|---|---|---|---|---|
| none, or not closed (no `Total wall time`) | `None` | `None` | `None` | `truncated` (killed, or still running: the scheduler tells which) |
| a line `ERROR` | `crashed` | `None` | `None` | `error`, with `error` |
| single point that ended | `ended` | `n/a` | `n/a` | `completed` |
| `force tolerance`, `forces are zero` | `ended` | `n/a` | `converged` | `completed` |
| `max iterations`, `max force evaluations` | `ended` | `n/a` | `max_steps` | `max_steps` |
| `walltime limit reached` | `ended` | `n/a` | `not_converged` | `max_seconds` |
| any other criterion | `ended` | `n/a` | `not_converged` | `optimizer_failed` |

The criterion is the line `Stopping criterion` of the summary of `minimize`, and
is kept in `criterion`; `steps` is `{'cg': n, 'total': n}` with the iterations of
the summary. A closed log of a relaxation without the summary is a
`LammpsOutputError`. `scf` is `n/a` because a model has no SCF cycle.

## Results

`parse_result(calc_dir, calculation, structure)` reads the results only when
LAMMPS ended and the geometry converged (a single point has none); otherwise
`parsing` and the results are `None`. It gives `parsing` `ok` or `failed`, with
the reason and no exception, for output that is missing, damaged or does not
agree with the input. The results have the keys and the units of a pw.x output
([qe_results.md](qe_results.md)):

| Key | Value |
|---|---|
| `nsites` | number of atoms |
| `input.structure` | structure of the input, as a pymatgen `Structure` dict |
| `output.structure` | final structure, in angstrom |
| `output.density` | density of the final structure, g/cm³ |
| `output.energy` | potential energy of the model, eV |
| `output.energy_per_atom` | `output.energy` / `nsites`, eV |
| `output.forces` | force on every atom, eV/Å, without constraints |
| `output.stress` | pressure tensor, kbar, with the sign of pw.x |

- **Stress.** LAMMPS prints the pressure tensor in bar, positive for a compressed
  cell, the sign of pw.x; `output.stress` is that tensor divided by 1000, rotated
  into the frame of the input. The mean of its diagonal is the pressure, −dE/dV.
- **Final structure.** Cell, labels, `if_pos` and the other site properties of the
  input (`magmom` left out), with the final positions of a relaxation or the
  positions of the input for a single point. The final positions are the images
  nearest to the input ones: an atom that LAMMPS counted in another periodic
  image does not jump by a cell vector.
- **Agreement with the input.** `final.dump` must have the number of atoms, the
  ids `1..N` and the atom types of the input structure; for a single point also
  the positions (to 10⁻⁶ Å). Otherwise `parsing` is `failed`.
- **Observed.** The version of LAMMPS (first line of the log) and the
  `pair_style` of the script. The device is not recorded: it is not known where
  LAMMPS prints it.
- **Energy model.** The energy of an MLIP is the energy of that model: it must not
  be combined with energies of another model, engine or backend. This module does
  not build the energy model of the record, and no hash of the model file is
  computed.

`last_structure(calc_dir, structure)` gives the input structure with the
positions of the last whole snapshot of `final.dump` or, if LAMMPS did not reach
the end of the script, of `trajectory.dump`; `None` if there is none. A snapshot
is whole when its header, all its atoms and all their columns are there and the
file ends with a newline: a file cut while LAMMPS wrote it ends inside a number
that would still read as one. A new attempt starts from that structure.

## What is not done

- Other minimizers, a relaxation of the cell, molecular dynamics, the model
  deviation of SCS, a run on CPU: not available.
- The model file is not looked at; LAMMPS fails if it is wrong.
- This module is not yet joined to the steps of the workflow: how a step chooses
  this engine, writes the input in its calculations, builds the command, removes
  the outputs of an earlier attempt and keeps the states in the record is not
  done here.

## What only a run of LAMMPS can confirm

The tests do not run LAMMPS, which is not installed where they run. What is known
of its behaviour comes from its documentation, and these points are checked on the
cluster with a real build:

- the text of the summary of `minimize` (`Stopping criterion = ...`, `Iterations,
  force evaluations = ...`), the strings of the criteria, the last line of the log
  (`Total wall time`) and the line `ERROR`;
- that `timer timeout` counts from the start of `minimize` (the loading of the
  model and `read_data` excluded) and that `timer timeout off` before the `run 0`
  keeps the budget from stopping it;
- that `fix setforce` counts in the forces that `min_modify norm max` compares
  with `fmax`, and that `run 0` after `unfix` gives the forces without
  constraints;
- that `dump` writes a snapshot in the first step of the `run 0` and that `xu yu
  zu` keep the image of the data file; that `print ... file` and the thermo
  keywords `pe` and `pxx`..`pyz` can be read between runs;
- the sign and the unit of the pressure tensor (bar, positive for a compressed
  cell), and the energy of the model as `pe`;
- the device the model runs on, and the precision of the file of the model.

## Tests

`tests/test_mlip_lammps_job.py` checks the text of the script (one script in full,
and the lines that depend on the type of model, the constraints, the time budget
and the criterion), the data file, the states against logs of every stopping
criterion, and the results against files written in the format the module reads.
Those files (`tests/mlip_lammps_helpers.py`) carry the energy, forces and stress
of ASE's EMT, put in the frame of LAMMPS with the same rotation ASE uses to write
the data file, for cells in any orientation; the stress is compared with −dE/dV.
EMT is used in the tests only. No test loads `torch`, `mace` or `deepmd`, and an
isolated interpreter checks that the module imports none of them.
