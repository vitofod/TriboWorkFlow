# Command line

`triboworkflow` prepares, submits, follows and analyses the steps of the
interface workflow of a run. The commands are modular, one per module of
`triboworkflow/cli/`, as in Xsorb.

```
triboworkflow [-v] [-T] <command> [-r RUN] ...
```

`-v` prints the version, `-T` the traceback of an error. Exit codes: 0 done,
1 error (`Error: <message>` on the standard error), 2 error of use. Without
a command the help is printed and the code is 2. `triboworkflow <command>
--help` gives the help of a command, with examples.

The steps themselves (`match`, `pes`, `pes_scf`, `plotpes`, `plotpes_scf`,
`shear`, `adhesion`, `ppes`, `charge`) come from a registry that each step
joins once it is implemented. In this version they are all implemented
([interface.md](interface.md)), `plotpes`, `plotpes_scf` and `shear` analysed
in a job of their own or by hand, never by the orchestrator (*The
interpolation of the PES*); a step that the registry lacked would make a
command answer `step '<name>' not available in this version`, with nothing
prepared or submitted. `check`, `status`, `stop` and the actions on jobs and
groups work on any run.

## The run

A run is a folder with its settings in `config.toml`. Every command works on
the current folder, or on the one given with `-r/--run DIR` from anywhere.
Everything the code writes stays inside the run:

```
CuAl/                       the run
├── config.toml             settings, written by the user (configuration.md)
├── jobscript.sh            resources and environment of the jobs, written by the user
├── slabs/...               the two slabs named in config.toml
├── systems/mp-30_mp-134/   system.json and one directory per calculation
├── jobs/                   groups of jobs
├── jobs.json               registry of the jobs
├── orchestrator.json       state of the steps
├── orchestrator.pid, .log  orchestrator in the background
├── run.lock                present while a command or the orchestrator changes the run
└── cli.log                 log of the commands that changed the run
```

- The system is the interface `<mid1>_<mid2>` of the two slabs: one run, one
  interface.
- Relative paths of the settings (jobscript, slabs) hold from the folder of
  the run, so a run prepared on a laptop is copied to the cluster as it is
  (e.g. with `rsync -a`).
- `[interface].steps` lists the steps to run. The graph of the steps decides
  what comes first (`match` → {`pes` → {`plotpes` → `shear`; `adhesion` →
  {`ppes`; `charge`}}; `pes_scf` → `plotpes_scf`}); `adhesion` follows the PES
  chosen in `[interface.adhesion]`, `pes` or `pes_scf`; the order of the list decides only
  between steps ready at the same time. A step already done and current is not run again;
  an entry it reads that is valid is reused even if its step is not listed.
- `[execution].groups` (jobs, i.e. workers, per group) is optional: it comes
  from `--groups M`, then from the settings; when both are missing the
  command asks for it if it runs in a terminal, and stops with an error
  otherwise, so that a command started by a script never waits.

## Commands

**Anywhere** (no command of the scheduler runs):

| Command | What it does |
|---|---|
| `check` | reads and validates `config.toml` and prints a summary: system, steps, scheduler, groups, placeholders to fill in; code 1 if the settings are not valid, with the TOML path of the field |
| `gen SETTINGS.toml` | one calculation, not a run: writes the pw.x input of `[structure]` and the extxyz of its structure next to the settings file (see *One calculation*); nothing is run |
| `status [--calcs]` | the orchestrator (running, stopped, on another host), who holds the lock of the run, each step with its state, reason and time of its last update, the calculations that need the user and why, the jobs. **It reads only the files: it is the state last saved, not a query to the scheduler.** `--calcs` lists every calculation with its state (`pending`, `running`, `completed`, `not_converged`, `crashed`, `unknown`) |
| `prepare <step> [--replace]` | phase 1: checks the entries the step reads and writes its calculations (directories, inputs, extxyz of the structures, records). A missing or stale entry blocks the step with what is missing; nothing is computed in its place. A calculation already prepared is never overwritten: one that differs from the settings stops the preparation with the list, one that differs only in keys of the attempts (e.g. `mixing_beta`, `max_seconds`) is kept with the values logged. `--replace` moves every calculation that differs, with its attempts, to `replaced/` and prepares it again (`execution.md`, *Preparation and replacement*) |
| `analyse <step>` | phase 3, from the files only: every calculation of the step must be done according to its claim, its record and the groups of jobs, with the fingerprint the settings give now; otherwise nothing is written and the error says what is missing for each calculation. For `match`, `plotpes`, `plotpes_scf` and `shear` it is the whole step; `plotpes`, `plotpes_scf` and `shear` are analysed by this command where there is enough memory, or in a job of their own with `[execution.analysis]`, and the command is refused while that job may run (*The interpolation of the PES*) |
| `retry <step>` | a step failed for an error of its own (while it was prepared or analysed, because the settings changed after its preparation, or because the job of its analysis ended without completing it) starts again; for a job of an analysis, the next `submit` or pass submits a new one |
| `clear-refusal <group>` | a group whose submission the scheduler refused is submitted again by the next `submit` or pass, once its copy of the jobscript (`jobs/<group>/jobscript.sh`) is fixed |
| `resolve-submission <group> --job-id ID` or `--no-job` | a submission whose answer was lost (e.g. `sbatch` timed out) created that job, or none |
| `release-claim <calc_id> --worker-over [--group G]` | releases an empty claim once the user has checked that its worker no longer runs |

**On the HPC machine only** (the submission command of the scheduler must be
found; elsewhere the command stops before writing anything):

| Command | What it does |
|---|---|
| `submit <step> [--groups M]` | phase 2: asks the scheduler for the jobs, reconciles the calculations of the step, writes a group with what has to run and submits it, then exits. **Running it again is how a step is followed by hand, and it may start relaunches** (a crash, `max_seconds`), never the same attempt twice. For `plotpes` and `plotpes_scf`, with `[execution.analysis]`, it submits the job of the analysis (one job, no `--groups`) and, run again, follows it |
| `start [--groups M] [--dry-run]` | starts the orchestrator of the steps of `[interface].steps` in the background, on the login node, and returns |
| `relaunch <calc_id> [--groups M] [--group G] [--level L] [--from-scratch]` | relaunches a calculation by hand, at the level of its last attempt or at level `L` (1–3), once its worker is surely over; it continues the last attempt from `OUT/` when the workflow would, or after a stop by request of the user, unless `--from-scratch` |

**Where the orchestrator runs:**

| Command | What it does |
|---|---|
| `stop [--wait S]` | **stops the automation, not the calculations**: the orchestrator ends its pass, saves its state and exits; the jobs already submitted go on, and a later `start` finds them again |

### One change at a time

Every command that changes the run (`prepare`, `submit`, `analyse`,
`relaunch`, `retry`, `clear-refusal`, `resolve-submission`,
`release-claim`), and the orchestrator in the background for its whole
life, creates `run.lock` before reading the state of the run. A second one,
in another terminal or while the orchestrator runs, stops at once and names
the holder, e.g. `the run is in use by triboworkflow submit pes (pid 5120 on
login03 since …): wait for it to end; the orchestrator in the background
ends with "triboworkflow stop"`. `check`, `status` and `stop` are always
allowed. A lock left by a process gone from the same host is replaced, with
a warning, by one process only: the one that creates `run.lock.takeover`
for the time of the replacement. A lock written on another host is never
touched, and the message says to remove it by hand after checking that no
command of the run runs there.

A command refused for its use, by the guard of the HPC machine or by the
lock writes nothing in the run, not even `cli.log`.

## Examples

### One calculation

A settings file with `[structure]` (template: `examples/single_input.toml`,
fields in `configuration.md`) gives one pw.x input, generated on any machine:

```bash
cd ~/calcs/graphene
triboworkflow gen single_input.toml
# Input: /home/.../graphene/input.pwi
# Structure: /home/.../graphene/input.extxyz
# calculation = relax; 6 atoms, 1 species; K_POINTS automatic 8 8 1  0 0 0
# Structure read from /home/.../graphene/C6.pwi (pwi); constraints: 5 default, 1 file
```

- The input is named by `[generation].file_prefix` and written in the folder
  of the settings file; the extxyz has the same name with the suffix
  `.extxyz`. It holds the cell, positions and labels of the input and its
  `if_pos` as `move_mask`, to check the input in OVITO before running it.
- The namelists are written as given, `calculation`, `prefix` and `outdir`
  included; the keys left out take the pw.x defaults. An scf makes one step
  and has no `&IONS`: when the settings say otherwise the input is changed
  and a warning names the key and its value.
- With `overwrite = false` an existing input or extxyz stops the command and
  nothing is written. A placeholder left in the settings, or a file with
  `[interface]`, stops it too.
- No run, lock or record is made: the input is run by hand, or in a run of
  the workflow.

### The match of the interface

`match` runs no DFT calculation, so it can be done on a laptop:

```bash
cd ~/runs/CuAl
triboworkflow check
triboworkflow analyse match
triboworkflow status
```

`analyse match` checks the entries it reads, matches the two slabs with the
parameters of `[interface.inter_params]` and writes its entries in
`systems/mp-30_mp-134/system.json`. Run again with the same inputs it does
nothing more. When a parameter of the match (or a slab) changes, its entries
are no longer current: the match is done again, by hand or by `start`, also
if `orchestrator.json` had it completed. The steps after it are then no
longer current either, and are done again when requested; a step that reads
an entry not done again is blocked with what is missing.

### The two PES

```toml
[interface]
steps = ["match", "pes", "pes_scf"]
```

`pes` (relaxed along z) and `pes_scf` (static) are two steps with their own
calculations and entries; each needs only `match`, so either can be
prepared, submitted and analysed first, by hand as below or by `start`.
`prepare pes_scf` writes one calculation per high-symmetry point, in
`systems/mp-30_mp-134/pes_scf/<point>/`; `analyse pes_scf` writes the entry
`pes_scf` and the structures of its minimum and maximum
([interface.md](interface.md)).

### The adhesion

```toml
[interface]
steps = ["match", "pes_scf", "adhesion"]

[interface.adhesion]
pes = "pes_scf"       # the PES whose minimum and maximum it is computed at: "pes" or "pes_scf"
type = "regular"      # "regular" or "short" (short: with short = true in [interface.pes_scf])
```

`adhesion` has four calculations, `adhesion/bot_slab`, `top_slab`,
`interface_min` and `interface_max`, and reads the PES named by `pes` and
`match`; the other PES is not needed. By hand, once that PES is analysed:

```bash
triboworkflow prepare adhesion
triboworkflow submit adhesion
triboworkflow analyse adhesion    # writes data, structure.opt.<type> and adhesion.<type>
```

`prepare adhesion` is blocked with what is missing while the PES is not done
and current. When the PES, `match` or `[interface.adhesion]` change after the
calculations were prepared, those that differ are not overwritten:
`prepare adhesion --replace` prepares them again ([interface.md](interface.md)).

### The interpolation of the PES

```toml
[interface]
steps = ["match", "pes", "pes_scf", "plotpes", "plotpes_scf"]
```

`plotpes` interpolates `pes` and `plotpes_scf` interpolates `pes_scf`: no
calculation, only an analysis whose memory grows with the square of the
points of the PES, from a few MiB for a small interface to tens of GiB for a
large one ([interface.md](interface.md#cost)). It is never run by the
orchestrator on the login node.

**In a job of its own.** With a jobscript for it in the settings:

```toml
[execution.analysis]
jobscript = "analysis.sh"     # e.g. #SBATCH --mem=200G, --time=04:00:00, the partition
```

once `pes` is completed the orchestrator writes the group
`jobs/mp-30_mp-134.plotpes.1/`, submits its job and follows it; the job
interpolates on the compute node and writes the entries and the figure, and
the next pass finds `plotpes` completed and goes on with what depends on it.
`start` does it all; by hand:

```bash
triboworkflow submit plotpes      # writes and submits the job
triboworkflow status              # the step running, with the group of its job
triboworkflow submit plotpes      # follows the job: completed once its entries are written
```

```text
mp-30_mp-134/plotpes: running (job 4171882 RUNNING)
  job of the analysis: group mp-30_mp-134.plotpes.1, folder jobs/mp-30_mp-134.plotpes.1
```

- The steps of a system run one at a time: `plotpes_scf` waits while the job
  of `plotpes` runs, and the job waits for a step that runs.
- A job that ends without the entries (out of memory, walltime, an error)
  makes the step **failed**, with the state of the job, the outcome it wrote
  in `jobs/<group>/outcome.json` and the folder of its log; the steps after
  it are blocked. **It is never submitted again automatically**: give the
  jobscript more memory or time, then `triboworkflow retry plotpes`, and the
  next `submit` or pass submits a new job (group `.2`).
- `analyse plotpes` by hand is refused while the job may still run according
  to the last query of the scheduler (`jobs.json`). The job does not take
  `run.lock`: a job killed on a compute node leaves no lock of the run.
- Set the threads of the linear algebra in the jobscript (e.g. `export
  OMP_NUM_THREADS=8`) and the memory from the cost ([interface.md](interface.md#cost)).

**By hand.** Without `[execution.analysis]`, once `pes` is completed the
orchestrator leaves `plotpes` **manual**, goes on with the other steps and
exits when nothing else waits or runs; the steps that depend on `plotpes`
are blocked until it is done:

```text
mp-30_mp-134/plotpes: manual (to analyse by hand: run "triboworkflow analyse plotpes" where there is enough memory, or give [execution.analysis] to analyse it in a job of its own)
```

Run it on a node with enough memory, e.g. in an interactive session or a job
of the scheduler on a compute node, with the orchestrator ended or stopped:

```bash
cd /scratch/.../CuAl
triboworkflow status                  # the orchestrator must not be running
triboworkflow analyse plotpes         # holds run.lock until it ends
triboworkflow analyse plotpes_scf
triboworkflow start                   # goes on with what depends on them
```

- The two steps are independent: each is analysed with its own command,
  writes its own entries and its own figure
  (`systems/<system>/plotpes/pes_<name>.png`,
  `systems/<system>/plotpes_scf/pes_<name>.png`).
- While the orchestrator runs, `analyse plotpes` is refused (`the run is in
  use by triboworkflow start …`), and nothing is written: wait for it to
  end, or `stop` it; the jobs already submitted go on.
- While `analyse plotpes` runs, it holds the run: `start`, `submit` and the
  other commands that change the run are refused until it ends.
- A step already analysed and current is recognised from its entries and
  its figure, without interpolating again: `analyse plotpes` and `start`
  report it completed. Analysing `pes` again makes it no longer current.

**The shear strength.** `shear` is analysed in the same two ways, with the
same command (`analyse shear`, `submit shear`), the same job and the same
jobscript of `[execution.analysis]`, once `plotpes` is completed: it reads the
interpolation of `plotpes` and never the one of `plotpes_scf`. It interpolates
the PES again and then runs the string method, which takes long even on a
small interface: a job for it needs time more than memory
([interface.md](interface.md), *Step `shear`*). Without `plotpes` done and
current the step is blocked with `entries missing or not current:
pes.energy.pes_jm2`.
- **Stopped before its end** (Ctrl-C, a signal, the walltime of a job):
  nothing is written, neither the entries nor the figure; the step stays
  running with no calculations, `start` leaves it manual again, and the
  next `analyse plotpes` starts it from the beginning. The other
  interpolation, if already done, is kept. A process killed on the same
  host leaves `run.lock`, which the next command replaces with a warning; a
  job killed on a compute node leaves a `run.lock` of another host, which
  no command touches: check with the scheduler that the job is over (e.g.
  `sacct -j <id>`), then remove `run.lock` by hand.

### One step by hand, phase by phase

```bash
cd /scratch/.../CuAl
triboworkflow prepare ppes
triboworkflow submit ppes
# hours later
triboworkflow status
triboworkflow submit ppes          # follows the step: it may relaunch, never twice
triboworkflow analyse ppes
```

- `prepare ppes` stops with what is missing when the entries of `adhesion`
  are not done and current; it can be run on a laptop before copying the
  run.
- `submit ppes`, with `groups = 10` and 9 calculations, writes the group
  `mp-30_mp-134.ppes.1` and submits 9 jobs, not 10, each with a worker that
  takes the free calculations one at a time. Away from the HPC machine it
  exits with code 1 and writes nothing:
  `Command 'sbatch' not found: calculations run only on an HPC machine with
  the scheduler 'slurm' of [execution]; inputs can be prepared and outputs
  analysed here`.
- Running `submit ppes` again reconciles first: calculations queued or
  running are left alone; one that crashed or stopped at `max_seconds` goes
  in a new group with its input unchanged, at most twice in a row; one that
  pw.x closed without convergence (SCF, or the step limit of a relaxation)
  goes at the next level of the remedies, with its values written in
  `run.pwi`; an attempt that pw.x closed at `max_seconds`, at the step limit
  or, in a relaxation, without SCF convergence continues from `OUT/` when its
  files are there and, but after the step limit, the command, the jobscript
  and the parallelization printed allow it
  ([execution.md](execution.md#restart)), the others start from scratch
  with `OUT/` removed; a completed calculation whose output cannot be parsed
  or used, or that continued with another parallelization, is listed among
  those that need the user; an error of pw.x goes to the user with its
  routine, code and message.
- **Settings changed after `prepare`** (e.g. `ecutwfc`): `submit`, `analyse`
  and the orchestrator compare the fingerprints the settings give now with
  those of the calculations prepared and stop before submitting or using a
  result: the step fails with the list (`settings changed since the
  preparation: …`), and the jobs still missing from its groups are not
  submitted. Either restore the settings and `retry ppes`, or replace the
  calculations: `prepare ppes --replace` moves each calculation that
  differs, with its attempts and `OUT/`, to `replaced/<system>/ppes/<k>/`,
  withdraws the groups listing them that never had a job submitted, and
  prepares the step again. It is refused, with the reason of each
  calculation and nothing changed, while a worker may still hold a
  calculation (an empty claim, or one whose job is not over) or a job of a
  group listing it may still run: cancel the job, let
  the orchestrator query the jobs again on the HPC machine (`start`), then
  repeat. A group submitted in part (some jobs over, the others refused, or
  held back since the settings changed) also stops it: it is not complete
  and cannot be withdrawn, and this version does not replace its
  calculations. The message says so and how to go on: restore the settings,
  `retry ppes`, `prepare ppes` (and `clear-refusal` for a refusal), let the
  group end with `submit ppes` or `start`, then change the settings and
  replace.
  `prepare` alone never overwrites a calculation prepared, and for a step
  already prepared only lists in the log the calculations that differ.
- **A replacement interrupted** (a crash, Ctrl-C): every command that changes
  the run stops and names `prepare ppes --replace`, which completes it
  without moving anything twice; `status` shows it too.
- `analyse ppes` writes the entries once every calculation is completed and
  current; otherwise it says, for each calculation, what is missing, e.g.
  `ppes/d3: attempt 2 recorded as running: its job may still run`, `empty
  claim: …`, `attempt 1 crashed`, `attempt 1 listed in group …, not taken
  yet`. The end of a job is read from the scheduler only by `submit`,
  `relaunch` and the orchestrator, on the HPC machine.

A step not listed in `[interface].steps` can be run by hand all the same.
In the background, write `steps = ["ppes"]` and run `start`: the valid
entries of `adhesion` are reused, and only `ppes` runs.

### The whole chain

```toml
[interface]
steps = ["match", "pes", "pes_scf", "plotpes", "plotpes_scf", "shear", "adhesion", "ppes", "charge"]
```

```bash
cd /scratch/.../CuAl
triboworkflow check
triboworkflow start
```

`start` prints `Orchestrator: running: pid 48211 on login03 since …` and the
shell comes back; the terminal can be closed. The orchestrator runs on the
login node, holds `run.lock` while it lives and, every `polling_interval`:

1. asks the scheduler for the jobs;
2. checks again the steps completed: those whose entries are no longer
   current are done again;
3. for each step ready: done and current, completed without running;
   entries it reads missing, blocked with the reason; otherwise it prepares
   the calculations, checks that they match the settings, submits them in a
   group, follows them, relaunches them as the policy says and, once they are
   all completed, analyses them; `plotpes` and `plotpes_scf` are left
   **manual**, with the command that analyses them (*The interpolation of
   the PES*);
4. a failure blocks only what depends on it: if `ppes` fails, `charge` goes
   on; a manual step blocks what depends on it until it is analysed by hand;
5. the steps of a system run one at a time, the order of `steps` deciding
   which one starts first when two are ready;
6. it ends when no step waits or runs: a manual step does not keep it
   alive.

Meanwhile:

```bash
triboworkflow status            # at any time, from any terminal
tail -f orchestrator.log        # one line per pass, e.g.
# Pass 12: steps 2 waiting, 1 running, 4 completed; jobs 4 waiting, 0 held, 0 to check; groups written 0
triboworkflow stop              # stops the automation at the end of the pass; the jobs go on
triboworkflow start             # goes on: recovers jobs and attempts, then decides, without duplicates
```

When a calculation needs the user (an error of pw.x, `CANCELLED`,
`OUT_OF_MEMORY`, too many unchanged relaunches, a remedy past level 3):

```bash
triboworkflow stop --wait 60
# look at the output in systems/mp-30_mp-134/<step>/<label>/ (or attempts/<n>/)
triboworkflow relaunch adhesion/interface_max
triboworkflow start
```

`relaunch` checks that the worker of the last attempt is surely over (no
claim, or the run of the job of its claim over); otherwise it refuses with
the state found. It writes a new group with a manual attempt at the same
level, or at the one of `--level` with the values of that level, and
submits it; the step goes back to running and the count of the unchanged
relaunches starts again. The attempt continues the last one from `OUT/`
under the same rule as the relaunches of the workflow, also after a stop by
request of the user (an exit file), which the workflow never relaunches;
`--from-scratch` starts it from scratch, `OUT/` removed. It prints the
attempt, its level and
how it starts:

```bash
triboworkflow relaunch adhesion/interface_max --level 2 --from-scratch
# mp-30_mp-134/adhesion/interface_max: relaunched by hand, attempt 4 at level 2, from scratch
```

When the launch of the last attempt is not recorded and the groups that list
it give different launches, it lists them: choose one with `--group`.

### Rare cases

`status` shows them with the command to use:

- `sbatch` timed out without an answer: the submission is never repeated by
  itself; look for the job (`squeue -u $USER`), then
  `resolve-submission <group> --job-id 123456` or `--no-job`;
- a submission refused (e.g. a wrong partition in the jobscript): each group
  submits its own copy of the jobscript, so fix `jobs/<group>/jobscript.sh`,
  keeping its last line (the worker), and the jobscript of the run for the
  next groups; then `clear-refusal <group>`. `submit` and `relaunch` exit
  with code 1 when a submission is refused or its outcome is not known;
- an empty claim (a worker that stopped right after taking a calculation):
  once its job no longer runs, `release-claim <calc_id> --worker-over`, then
  `relaunch <calc_id>`. The attempt takes its launch (kind, level, values)
  from the group that listed it; with several groups and different launches
  the command lists them and `--group` chooses; with none (only if `jobs/`
  was removed) it refuses: the level is never assigned by default. A
  `release-claim` interrupted is reported by `status` as an empty claim
  released in part: run the same `release-claim` again;
- a step failed for an error of the code while it was prepared or analysed,
  or for settings changed after its preparation: fix the cause, then
  `retry <step>`;
- `run.lock` left by a process gone on another host: check that no command
  of the run runs there, then remove it by hand. The same for
  `run.lock.takeover`, left by a command stopped while it replaced a lock:
  until it is removed, a lock left is not replaced.

## Dry run

`start --dry-run` is a **limited check of the first pass**. It copies the
run folder to a temporary directory, allowed only while the run has no
`jobs/` yet, runs one pass there with the scheduler in a dry run and prints
the steps and the commands of the scheduler that would run
(`sbatch …/jobscript.sh`). Nothing is submitted and nothing is left in the
run; later passes, jobs and relaunches are not simulated. It needs no
scheduler, so it also works on a laptop. Only the run folder is copied:
relative paths of the settings that leave it (`../`) do not hold in the
copy.
