# MLIP calculations with ASE

A calculation with a machine-learned interatomic potential (MLIP) through the
calculators of ASE is a single point or a relaxation of one structure. Two
modules of `triboworkflow/engines/mlip/` divide the work:

- `ase_run.py`, the **executor**: a script that runs in the Python
  environment of the MLIP (`torch` and the library of the model). It imports
  only the standard library, `ase` and the calculator named in its job, never
  `triboworkflow` nor `pymatgen`, so the workflow does not have to be
  installed there.
- `ase_job.py`, the **workflow's half**: it writes the job from the `[mlip]`
  settings ([configuration.md](configuration.md)), tells from the result how
  the executor ended, and turns the result into the states and results of the
  record. It imports no MLIP library and runs wherever the workflow does.

```python
from triboworkflow.engines.mlip import ase_job

job = ase_job.job_of('relax', settings.mlip, max_seconds=3600)   # settings.mlip: [mlip]
ase_job.write_job(calc_dir, job)        # job.json, next to structure_in.extxyz
# in calc_dir, on a GPU node, in the environment of the MLIP:
#   <python of the MLIP> <ase_job.EXECUTOR_PATH> job.json
status = ase_job.ase_status(calc_dir)
parsed = ase_job.parse_result(calc_dir, 'relax', structure)   # structure of the input
```

Calculations run on a GPU only. The model is never loaded by the workflow.

## Files of a calculation

All in the directory of the calculation; the executor is started there.

| File | | Content |
|---|---|---|
| `job.json` | input | what to run: see below |
| `structure_in.extxyz` | input | cell, positions and constraints (`move_mask`) of the atoms |
| `structure_last.extxyz` | output | the structure after every step, replaced atomically |
| `optimizer.log` | output | log of the optimizers of ASE (relaxation only) |
| `result.json` | output | written last: its presence says that the executor ended |

`result.json` of an earlier execution is deleted when the executor starts, so a
`result.json` always belongs to the last execution. `structure_last.extxyz` is
written after every step with a temporary file and `os.replace`: a process
killed while writing leaves the previous structure, never half of one. It is
where a new attempt restarts from, however the execution ended. A binary ASE
trajectory is not used for this: one cut in the middle cannot be read at all.

## `job.json`

| Key | Value |
|---|---|
| `calculation` | `scf` (a single point; there is no SCF cycle, the name is that of the record) or `relax` |
| `model` | the call that creates the calculator, `{"factory": "module:function", "args": [...], "kwargs": {...}}` |
| `package` | the distribution of the library of the model, e.g. `mace-torch`: its installed version is recorded |
| `model_path` | path of the model, recorded as resolved |
| `fmax`, `maxsteps` | threshold (eV/Å) and steps of **each** optimizer; relaxation only |
| `max_seconds` | time budget of the whole execution, model loading included, or `null` |

`job_of` takes `model` from the registry (`registry.ase_calculator_call`), so
`kwargs` reach the loader as written in `[mlip]`, never completed. For `MACE`
and `MACE_MP` the registry requires `default_dtype` with ASE, because it changes
the energies: `job_of` raises `ModelError` without it, before any job exists.

## What the executor does

Before it loads the model it checks that `torch` sees a CUDA device. Without
one it fails with a clear error: DeePMD would otherwise fall back to the CPU
without a warning. The device found is recorded in `observed`.

- **Single point**: energy, forces and stress of the structure.
- **Relaxation**: `BFGSLineSearch(c1=0.1)`, as Xsorb does, and `FIRE2` if it
  does not converge, each for `maxsteps` steps and the same `fmax`. The cell is
  fixed. `FIRE2` runs when `BFGSLineSearch` uses its steps up and also when its
  line search fails (`RuntimeError`): that is a lack of convergence. Any other
  error stops the execution. Neither optimizer writes or reads a restart file.
- **Constraints** come from `structure_in.extxyz`: `ase.io.read` turns
  `move_mask` into `FixCartesian` for each atom, and a fixed atom does not
  move. A relaxation converges when the largest force on an atom **with the
  constraints applied** (the forces the optimizer sees) is below `fmax`; the
  forces recorded are those of the model, **without** constraints, on every
  atom, as pw.x prints them.
- **Time**: the budget is checked between two steps, so a step in progress ends.
  The last structure is always on disk.

Unlike Xsorb: no `restart.json` (the second optimizer would read the one of
the first, and a file left by an earlier run would be taken for a restart); a
failed line search goes on with `FIRE2`; the steps of each optimizer are
recorded, where the log of Xsorb reports those of `FIRE2` only; the time and
the GPU are checked.

## `result.json`

`{"ok": false, "error": "Type: message"}` if the execution failed, else `ok`
true and:

| Key | Value |
|---|---|
| `calculation` | as in the job |
| `geometry` | `converged`; `max_steps` (the steps of every optimizer are used up); `not_converged` (stopped for time); `n/a` (single point) |
| `stop_reason` | `completed` (converged, or a single point), `max_steps` or `max_seconds` |
| `steps` | `{"BFGSLineSearch": n, "FIRE2": m, "total": n + m}`; `null` for a single point |
| `fallback` | why `FIRE2` ran, `max_steps` or `line_search_failed`; `null` if it did not |
| `energy`, `forces`, `stress`, `positions` | of the last structure, in the units of ASE: eV, eV/Å (no constraints applied), eV/Å³ with the sign of ASE (`null` if the calculator has no stress), Å |
| `symbols`, `cell` | of the atoms, so that the workflow can check them against its input |
| `observed` | `versions` (python, ase, the package of the model, torch), `device` (index and name of the GPU) and `model_path`, as found |
| `elapsed` | seconds |

The exit code is 0 when `result.json` was written with `ok` true and 1
otherwise. A relaxation that did not converge is not an error. A value that is
not finite (`NaN`, infinity) is an error.

## States

`ase_status(calc_dir)` gives the states of the record from `result.json`:

| `result.json` | `process` | `scf` | `geometry` | `stop_reason` |
|---|---|---|---|---|
| none | `None` | `None` | `None` | `truncated` (the executor was killed, or is running: the scheduler tells which) |
| `ok` false | `crashed` | `None` | `None` | `error`, with `error` |
| single point | `ended` | `n/a` | `n/a` | `completed` |
| relaxation that converged | `ended` | `n/a` | `converged` | `completed` |
| steps used up | `ended` | `n/a` | `max_steps` | `max_steps` |
| time used up | `ended` | `n/a` | `not_converged` | `max_seconds` |

`scf` is `n/a` because a model has no SCF cycle. `steps` and `fallback` of the
result go with the states (`AseStatus`). A `result.json` that cannot be read
is an `AseResultError`.

## Results

`parse_result(calc_dir, calculation, structure)` reads the results only when the
executor ended and the geometry converged (a single point has none); otherwise
`parsing` and the results are `None`. It gives `parsing` `ok` or `failed`, and
no exception for a result that is damaged or does not agree with the input,
with the reason. The results have the keys and the units of a pw.x output
([qe_results.md](qe_results.md)):

| Key | Value |
|---|---|
| `nsites` | number of atoms |
| `input.structure` | structure of `structure_in.extxyz`, as a pymatgen `Structure` dict |
| `output.structure` | final structure, as a `Structure` dict, in angstrom |
| `output.density` | density of the final structure, g/cm³ |
| `output.energy` | energy, eV |
| `output.energy_per_atom` | `output.energy` / `nsites`, eV |
| `output.forces` | force on every atom, eV/Å, without constraints |
| `output.stress` | stress tensor, kbar, with the sign of pw.x; absent if the model has none |

- **Stress.** ASE's stress is positive for a stretched cell; pw.x prints a
  positive stress for a compressed one. `output.stress` is minus ASE's, in kbar
  (1 eV/Å³ = 1602.176634 kbar), so that the mean of its diagonal is the pressure,
  −dE/dV, as in the outputs of pw.x. The conversion is made by the workflow
  (`stress_to_kbar`), not by the executor, and is tested against −dE/dV.
- **Final structure.** Cell, labels, `if_pos` and the other site properties of
  the input, with the final positions of a relaxation or the positions of the
  input for a single point. `magmom` is left out.
- **Agreement with the input.** The result must be of the calculation asked for,
  and have the number of atoms, the elements and the cell of the input
  structure (to 10⁻⁶ Å); for a single point also the positions. Otherwise
  `parsing` is `failed`.
- **Energy model.** The energy of an MLIP is the energy of that model: it must not
  be combined with energies of another model or engine. These modules do not build
  the energy model of the record, and no hash of the model file is computed.

## What is not done

- Other optimizers, a relaxation of the cell, molecular dynamics, a run on CPU:
  not available.
- The model file is not looked at; the loader fails if it is wrong.
- These modules are not yet joined to the steps of the workflow: how a step
  chooses this engine, writes the job in its calculations, starts the executor
  and keeps the states in the record is not done here.
- The names and the default arguments of the real calculators (`DP`,
  `MACECalculator`, `mace_mp`), the device that DeePMD uses and the behavior of
  the optimizers with a `float32` model have not been checked: no MLIP library
  is installed where the tests run. They need a GPU.

## Tests

`tests/test_mlip_ase_run.py` runs the executor in the process of the tests, with
ASE's EMT in place of the model (described, like a real one, as
`ase.calculators.emt:EMT`; EMT is used in the tests only) and with the check of
CUDA replaced, and runs it as a script in an isolated interpreter to check that
it imports nothing of the workflow or of the MLIP. `tests/test_mlip_ase_job.py`
tests the job, the states and the results on what the executor writes, with
the stress compared with −dE/dV of EMT. No test loads `torch`, `mace` or
`deepmd`.
