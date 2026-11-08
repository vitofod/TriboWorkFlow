# Execution on HPC

Calculations run only on an HPC machine, inside jobs of its batch scheduler.
This page describes how a group of calculations is laid out, submitted, run
and followed; how the calculations are reconciled with their jobs and
relaunched; how the orchestrator takes the steps of a workflow forward, in
the background on the login node; and how to check the execution on a new
machine with a fake engine.

## Layout of a run

```
<run>/
├── config.toml                  settings of the run (configuration.md, cli.md)
├── systems/<system>/
│   ├── system.json
│   └── <step>/<label>/          one directory per calculation
│       ├── input.pwi            input prepared for the calculation, never changed
│       ├── structure_in.extxyz  its structure, with the constraints of the input
│       ├── input.ppi            pp.x input, if the calculation gives the charge density
│       ├── run.pwi              input of the current attempt
│       ├── output.pwo           output of the current attempt
│       ├── structure_out.extxyz final structure, once the output is parsed
│       ├── output.ppo           pp.x output of the current attempt
│       ├── density.filplot      charge density written by pp.x
│       ├── claim                taken by the worker running the attempt
│       ├── record.json
│       ├── OUT/                 outdir of pw.x
│       └── attempts/<n>/        the files of attempt n: run.pwi, output.pwo,
│                                structure_out.extxyz, output.ppo, density.filplot
├── jobs/<name>/                 one group of jobs
│   ├── manifest.json            or analysis.json, for the job of an analysis
│   ├── outcome.json             written by the job of an analysis
│   ├── jobscript.sh
│   ├── submitted.json
│   └── withdrawn.json           only for a group withdrawn before any submission
├── replaced/<system>/<step>/<k>/  one replacement of calculations
│   ├── replacement.json         journal: what, why, groups withdrawn or retired
│   └── <step>/<label>/          a calculation replaced, as it was
├── jobs.json                    registry of the job manager
├── orchestrator.json            steps of each system (schema 2)
├── orchestrator.pid             orchestrator running in the background
├── orchestrator.log             its output
├── run.lock                     held by the process that changes the run
└── cli.log                      log of the commands that changed the run
```

pw.x runs inside the calculation directory, so the relative `outdir = 'OUT'`
of each calculation is its own; the `prefix` is `<step>_<label>`
(`execution/layout.py`).

## Groups of jobs

The calculations of a group are spread over several jobs, submitted from one
jobscript (`execution/jobs.py`):

- `manifest.json`: the calculations, in the order the workers take them, and
  how to run them: scheduler, `run_command`, `pp_command` (null when not
  given), `walltime`, `max_seconds_margin` (from `[execution]`, see
  `configuration.md`) and the digest of the jobscript of the user
  (`jobscript_digest`). A group with a calculation that gives the charge
  density is not written without `pp_command`, nor if that calculation has
  no `input.ppi`. Each calculation names the
  attempt the group runs (`Launch`): its number `n`, its kind (`initial`,
  `unchanged`, `remedy`, `manual`), the level of the remedies and, above
  level 1, the values of the level and the effective values at level 1, and
  whether it continues the previous attempt from `OUT/` (`restart`). A new
  attempt of a calculation always goes in a new group. `schema_version` is
  4; this version reads no other.
- `jobscript.sh`: the jobscript of `[execution]`, copied byte for byte, with
  one line added at the end:

  ```
  <python> -m triboworkflow.execution.worker <run>/jobs/<name>/manifest.json
  ```

  The jobscript of the user is never changed. It sets the resources and the
  environment (modules, virtual environment) and must not launch pw.x
  itself.
- `submitted.json`: the ids of the jobs, each added right after its
  submission. Before each submission it gets the time it starts
  (`pending`), replaced by the id as soon as the scheduler gives it: an
  interruption in between leaves the trace of a job that may exist, never a
  job of which nothing is known. A failure that surely created no job
  removes it.
- `withdrawn.json`: only for a group withdrawn before any of its jobs was
  submitted (see *Preparation and replacement*).

`submit_jobs` submits the jobscript once per worker, `groups` times but never
more than the calculations of the group (the reduction is logged), from the
directory of the group, where the scheduler writes the logs. With 16
calculations, `groups = 1` runs them one after the other, `groups = 4` up to
four at a time, `groups = 16` all at once, as the scheduler starts the jobs.
Submitting where the command of the scheduler does not exist stops with a
clear error (`core/guard.py`); a dry run (`JobScheduler(..., dry_run=True)`)
records the commands without running them, and works anywhere.

**Job listings** (`execution/scheduler.py`). Slurm (`squeue`), LSF (`bjobs`)
and SGE (`qstat -u $USER`, the default states `psr`) are read by column. PBS
and Torque read the named attributes of `qstat -f` (`-F json` for PBS, `-1`
for Torque) and keep the jobs whose `Job_Owner` is the user; HTCondor reads
`condor_q $USER -af:j JobStatus`. An output that cannot be read raises an
error rather than return lists that look empty. The same holds for telling
running and queued jobs apart with a custom table.

**State of a job** (`JobScheduler.get_job_state`). The state is kept as the
scheduler prints it, with whether the job is active (queued or running),
ended, or in a state that is not known, and whether it is held:

| Scheduler | Command | After the job left the queue |
|---|---|---|
| Slurm | `sacct -X -n -o State%30 -j <id>` | kept by `sacct` |
| PBS | `qstat -f -F json -x <id>` | finished jobs with `-x` |
| Torque | `qstat -f -1 <id>` | completed jobs while the server keeps them |
| HTCondor | `condor_q <id> -af:j JobStatus`, then `condor_history` | kept by `condor_history` |
| LSF, SGE | the job listing | not known |

A job that the scheduler does not report has no state: its absence does not
tell whether or how it ended. A Slurm state cut by the column (`+`) is read
only if the part before `+` is a whole name, so `CANCELLED+` is cancelled and
`OUT_OF_ME+` is not known. Slurm `PREEMPTED` is ended but may be requeued, so
it is asked again. Held jobs (HTCondor `5`, PBS and Torque `H`, SGE states
with `h`) stay in the queue and are never released by the package; for SGE
the reason comes from the listing of each hold type (`hu`, `ho`, `hs`: needs
an intervention; `hd`, `hj`: waiting for other jobs; `ha`: waiting for its
start time), for PBS and Torque `Hold_Types` is shown as printed.

**Restarts.** A scheduler may requeue a job, which keeps its id and runs its
jobscript again from the start. Slurm tells how many times
(`get_restart_count`: `sacct -X -n -o Restarts -j <id>`, a query of its own),
and a run of the job sees the same number in `SLURM_RESTART_COUNT`, set only
after a restart. The other schedulers do not tell it. A job waiting in the
queue to run (`is_queued`: Slurm `PENDING`, `REQUEUED`; PBS and Torque `Q`,
`W`; LSF `PEND`; SGE `qw`; HTCondor `1`), not held, is one that was requeued
if it already ran.

A submission that fails says whether the job may exist
(`JobSubmissionError.accepted`): False when the command exited with an error
and printed no id, None when it timed out, failed in another way or printed
no id after succeeding.

None of these commands has been tried on a cluster yet.

## Job manager

`execution/manager.py` submits the groups and follows their jobs, with its
registry in `<run>/jobs.json` (written atomically). `add_group(name, count)`
plans `min(count, calculations)` jobs; each `cycle`:

1. recovers the ids of `submitted.json` missing from the registry, so that a
   job already submitted is never submitted again; a `pending` left by an
   interruption becomes a submission of outcome not known;
2. asks the state of each job that may still change;
3. submits the jobs still missing from each group. Every registered job
   counts, and so does every submission of outcome not known.

A failed submission is never repeated automatically. A refused one (no job
created) waits for `clear_refusal`; one of outcome not known takes the place
of a job, stops the group and survives a restart until `resolve_submission`
registers the job found, or frees its place.

Each job is:

- **waiting**: queued or running, also when the last query failed; or not
  reported yet, during the first `job_report_wait` seconds of `[execution]`;
- **held**: held in the queue, with its state and the reason of the hold;
- **ended**: ended at the last query;
- **attention**: not reported after `job_report_wait`, no longer reported, or
  in a state that is not known.

A job keeps its id, its last known state, the time that state was first
reported (to confirm it at a later query) and the claims of its
calculations, and is asked again at every cycle until its state is final; it
is never replaced. A group is complete when all its jobs are in a final state
confirmed by the last query, with no refused submission or submission of
outcome not known. `run(polling_interval)` repeats the cycle while a job is
waiting or jobs are to submit; held jobs, jobs that need attention and
unresolved submissions stay in the summary for the next cycles.

A group none of whose jobs was ever submitted (no job in the registry, no
submission of outcome not known, no id and no `pending` in `submitted.json`;
a refusal is allowed) can be withdrawn (`withdraw`): `withdrawn.json` is
written, the group is never submitted again, `clear_refusal` refuses it, and
its launches count for no calculation. It stays on disk and is listed apart
in the summary and by `status`.

## Worker

Each job runs one worker (`execution/worker.py`). The worker goes once
through the calculations of the manifest. It runs attempt `n` of a
calculation only while the record has `n - 1` attempts, so a late job of a
group that was replaced finds the attempt made and skips it, and only if no
other worker has taken it:

1. creates `claim` with `O_CREAT | O_EXCL`: exactly one worker succeeds, the
   others skip the calculation. It prepares and checks the input of the
   attempt in memory, then writes in the claim the attempt, the job id, the
   restart count of the job (`restart_count`, from `SLURM_RESTART_COUNT`,
   null when not set), the host, the pid and the time; nothing else is
   changed on disk before;
2. adds attempt `n` to the record, with the process `running`, the
   `run_command` and the `jobscript_digest` of the manifest and its launch in
   `remedies`, and clears the results, `observed` and the engine version of
   the run before. An attempt from scratch after the first gets
   `outdir_removal = 'started'`;
3. for such an attempt, removes the directories that pw.x writes, `outdir`
   and `wfcdir` when given (`clear_scratch`): pw.x from scratch deletes only
   some files of its output directory, and a later restart could read the
   others although they come from an older attempt. Each must lie inside
   the calculation directory, with no symbolic link on the way, checked
   again right before. The record then gets `removed` or `absent`. A removal
   that fails gives `failed`: the attempt ends without running (process
   `crashed`) and the user decides. A worker stopped during the removal
   leaves `started`, and the next attempt from scratch removes the
   directories again;
4. writes `run.pwi`, a copy of `input.pwi` with `max_seconds` set and,
   above level 1, the values of the level (`nstep` in `&CONTROL`,
   `mixing_beta` in `&ELECTRONS`), each one replacing its line or added at
   the start of its namelist; a launch that continues the previous attempt
   also gets `restart_mode = 'restart'`;
5. creates `output.pwo` with `O_EXCL`;
6. runs `<run_command> -in run.pwi >> output.pwo` with bash, in the
   calculation directory;
7. records the end time, the exit code, the digests of `run.pwi` and
   `output.pwo`, the states read from the output, the reason of the stop,
   whether pw.x closed the run and the version it printed; when the
   calculation is over with the SCF and the geometry converged, it reads the
   results (`execution/parsing.py`, [qe_results.md](qe_results.md)): the
   record gets `parsing`, the results and `observed`, the directory
   `structure_out.extxyz`, and an output that cannot be parsed leaves its
   reason in the attempt (`parse_error`) and in the log. A version of pw.x
   outside 7.1.x–7.3.x gives a warning in the log. For an attempt that
   continued the previous one, the record gets what pw.x read from its files
   (`restart`, see [Restart](#restart));
8. for a calculation that gives the charge density, once its output was
   parsed: adds the post-processing to the attempt (`density`: pp.x,
   `pp_command`, `input.ppi` and its digest), creates `output.ppo` with
   `O_EXCL`, runs `<pp_command> -in input.ppi >> output.ppo` with bash in the
   calculation directory, and records the exit code, the digests of
   `output.ppo` and `density.filplot` and the state `density`: `ok` when pp.x
   closed its run without errors and `density.filplot` is complete, is the
   charge density and agrees with `structure_out.extxyz`
   (`parsing.density_problem`, [qe_charge.md](qe_charge.md)), `failed` with
   the reason otherwise. The exit code is recorded, the state does not come
   from it. pp.x does not run after an attempt that was not parsed.

If something is wrong before step 2 (no record, a calculation other than
`scf` or `relax`, files of an earlier attempt still in place, a value of the
level whose line is not alone or whose namelist is missing, an `input.pwi`
that asks for a restart itself, a `wfcdir` outside the calculation
directory, an `outdir` missing or outside it before an attempt from
scratch, for a calculation that gives the charge density no `pp_command` in
the manifest or no `input.ppi`), the worker removes its claim, logs the
error and goes on, with nothing removed. A worker killed with its job
leaves the attempt with the process `running`, or, during pp.x, with the
post-processing begun and the state `density` null; what happened is then
known from the scheduler and the files.

A charge density that failed is never produced again automatically: pw.x
completed, so the calculation is `completed`, its results cannot be used
(`reuse_problem`: `charge density failed: <reason>`) and it goes to the
user. A manual relaunch runs pw.x again from scratch, then pp.x.

**Time limit.** `max_seconds` of each attempt is `walltime` minus the time
since the worker started minus `max_seconds_margin`, capped by the
`max_seconds` of `input.pwi` when given. pw.x then stops in time to write its
restart files. When nothing is left, the worker takes no more calculations.
pp.x runs in the time `max_seconds_margin` keeps free, with no limit of its
own: the margin must cover it too.
The `walltime` of `[execution]` must be the one of the jobscript. In
`input.pwi`, `max_seconds`, `nstep` and `restart_mode` must be alone on
their line of `&CONTROL`, `mixing_beta` on its line of `&ELECTRONS`, when
given.

**Job id.** Read from `SLURM_JOB_ID`, `PBS_JOBID` (PBS, Torque), `LSB_JOBID`
(LSF) or `JOB_ID` (SGE); null for HTCondor and outside a job.

## Archiving an attempt

Before a calculation runs again, its last attempt is archived
(`execution/archive.py`): `run.pwi` and `output.pwo` move to `attempts/<n>/`
after a check of their digests, with `structure_out.extxyz` when the output
was parsed and, after pp.x, `output.ppo` and `density.filplot` (checked
against the digests of the post-processing; `input.ppi` stays in place),
the record gets the new paths, and the claim is removed last. An interrupted archiving is completed by running it again.
An attempt still recorded as `running` is not archived; the archiving does
not touch `OUT/`, which the next attempt continues from or the worker
removes before an attempt from scratch. An attempt is archived only when it
is relaunched.

An empty claim (a worker that stopped right after creating it) gets the
number the worker would have used, the attempts in the record plus one, and
the attempt is added to the record with its end unknown. It also gets its
launch when one is given (`archive_attempt(..., launch=)`), so that it can
be relaunched at its level: `Orchestrator.release_claim`, asked by the user
once the worker is surely over, takes it from the groups that list the
attempt (`reconcile.launch_for`). A release interrupted leaves the claim
with the number of the attempt alone (`archive.released_in_part`), with
the attempt not yet in the record or already there: the reconciliation
reports it as an empty claim released in part, and releasing the claim
again completes it. With the attempt already recorded, that attempt is
archived and no other one is added; its launch is the one recorded, and a
`--group` giving another launch is refused.

## Reconciliation

Before anything new is submitted, each calculation is brought up to date
with its record, its claim, the scheduler and its output
(`execution/reconcile.py`), and classified:

| State | Meaning |
|---|---|
| `new` | no attempt, and no group for the first one |
| `pending` | a group for the next attempt is not over yet |
| `running` | an attempt runs, or the end of its job is awaited |
| `held` | the job of the claim is held: kept as it is, nothing relaunched |
| `completed` | the last attempt ended with the SCF and the geometry converged |
| `unchanged` | the last attempt crashed, was killed or stopped at `max_seconds` |
| `remedy` | pw.x closed the run without convergence (`scf_not_converged`, `max_steps`) |
| `manual` | no automatic relaunch: the user decides, with the reason |
| `uncertain` | what happened cannot be established now; the claim is kept |
| `not_taken` | every group for the next attempt is over and none took it |

An attempt recorded as `running` is fixed only when the run of the job that
wrote its claim is over: the job is in a final state confirmed by the last
query; or it is in the queue again; or a Slurm job runs with more restarts
than the `restart_count` of the claim; or a Slurm job is `PREEMPTED` at a
query at least `polling_interval` seconds after the first one that reported
it, with the restarts of the claim. Without the restart count the attempt
waits for the end of the job, and a preempted job is `uncertain`. A claim
without a job id, an empty claim and a job that the scheduler no longer
reports are `uncertain`.

A calculation that gives the charge density, whose last attempt was parsed
and whose state `density` is not known, is `running` while pp.x may run:
until the run of the job that wrote the claim is over, as above. Then the
state is read from `output.ppo` and `density.filplot` as the worker reads
it, with the exit code null; when the worker stopped before pp.x ran, or
the end of pw.x is fixed here from its output, the state is `failed` with
the reason.

The fixed attempt gets the state of the job as printed, the states read from
the output (the process `killed` when the scheduler stopped or requeued the
job, `crashed` otherwise, unless pw.x closed the run), the reason of the
stop, what a restart read and the digests of its files. The output decides
when pw.x closed the
run: a calculation whose worker was killed after pw.x ended is completed,
and its results are read as the worker reads them.

How the job ended decides when the output does not:

| Scheduler | Relaunched unchanged | Left to the user |
|---|---|---|
| Slurm | `TIMEOUT`, `NODE_FAIL`, `BOOT_FAIL`, `PREEMPTED`, `COMPLETED`, `FAILED`, requeued | `CANCELLED`, `DEADLINE`, `OUT_OF_MEMORY`, others |
| PBS, Torque | `F`, `C` | — |
| HTCondor | `4` (completed) | `3` (removed) |

So a `qdel` on PBS or Torque cannot be told from another end of the job and
is relaunched. An error of pw.x (`Error in routine`) is left to the user,
with its routine, code and message read from the output, which is kept. So
is an attempt that did not run because the output directories of the
previous attempts could not be removed (`outdir_removal = 'failed'`; the
error is in the log of the worker). A
group that ends without any of its attempts taken makes its calculations
`manual`, with its jobs and their states ("no progress"); the cause is not
deduced.

## Relaunches

The policy (`execution/policy.py`) decides what follows each reconciled
calculation:

- `new` gets its first attempt; `not_taken` the same launch again, which
  uses no relaunch;
- `unchanged` runs again with the input of its level, in a new job, at most
  twice in a row at a level; then the user decides. The count comes from the
  launches recorded in the attempts, so a restart does not reset it; a new
  level or a manual relaunch starts it again;
- `remedy` runs at the next level, up to level 3; then the user decides.

| Level | `mixing_beta` | `nstep` (relax only) |
|---|---|---|
| 1 | of the input (recommended 0.4) | of the input (recommended 150) |
| 2 | `min(β₀, 0.2)` | 200 if `n₀ ≤ 200`, 250 if `n₀ ≤ 250`, `n₀` above |
| 3 | `min(β₀, 0.1)` | `max(n₀, 250)` |

`β₀` and `n₀` are the values pw.x uses at level 1, given or default, read
from the summary at the beginning of its output; when `input.pwi` writes
them they must agree with the summary. They are kept with the launches above
level 1, whose values the worker writes in `run.pwi`. `nstep` counts the
steps of one execution of pw.x: after a restart it counts again from the
first, while the optimizer keeps its history.

`policy.manual_launch` gives the launch of a relaunch asked for by the user:
allowed only when the worker of the last attempt is surely over (no claim,
or the run of the job of the claim over), at the level of the last attempt
and with its input or at the level asked for (`--level`, with the values of
that level), continuing the last attempt on the conditions of a relaunch of
the workflow, also after a stop by request of the user, unless asked to
start from scratch (`--from-scratch`); the count of
the unchanged relaunches starts again. An attempt that groups listed and no
worker took runs again as it was listed, at no other level.

### Restart

pw.x continues a run from the files it left in its output directory
(`restart_mode = 'restart'`) only after it closed that run; it reads each
part on its own and computes again, with a message and without an error,
what it cannot read. What it writes depends on how the run stopped
(QE 7.2: `PW/src/run_pwscf.f90`, `punch.f90`, `electrons.f90`,
`Modules/bfgs_module.f90`):

| Last attempt | What the next one reads | Continued |
|---|---|---|
| `max_seconds` | positions, charge density, wavefunctions, the iteration of an interrupted SCF and its mixing, the history of the optimizer; the wavefunctions and the SCF state are written once per MPI process | yes, all of it, with the same parallelization |
| `max_steps` | the new positions, the charge density and the wavefunctions at those positions, readable with any parallelization, the history of the optimizer; the SCF starts from its first iteration | yes, all of it |
| `scf_not_converged`, relax | the positions, the last charge density (not converged), the wavefunctions written per process, the history of the optimizer; the SCF starts from its first iteration without the history of the mixing | in part, with the same parallelization |
| `scf_not_converged`, scf | — | no: from scratch |
| `user_request` (exit file) | as after `max_seconds` | only in a relaunch asked for by the user (`relaunch`), with the same parallelization; never by the workflow |
| crash, kill, any other end | — | no: from scratch |

A restart is decided (`policy.can_restart`) only when, besides, `prefix` and
`outdir` are given in `input.pwi` and `OUT/<prefix>.save/` holds
`data-file-schema.xml` and the charge density. pw.x reads the files written
per process correctly only with the same parallelization, which it does not
check (`INPUT_PW`, `restart_mode`). That parallelization is known only once
the new attempt runs: before, a restart from such files (after
`max_seconds`, `user_request`, `scf_not_converged`) needs the new attempt to
run with the same `run_command` and the same jobscript, which sets the
resources of the job (`jobscript_digest` of the attempt and of the
`[execution]` jobscript), and the output of the last attempt to print its
parallelization; otherwise it starts from scratch, with the reason.

The output of the attempt then tells what was read (`engines/qe/status.py`,
`PwRestart`) and the record keeps it per part in `restart`, with whether the
parallelization printed in the header (MPI processes, pools, band groups, R &
G space, FFT divisions) is that of the previous attempt. Positions, density
or wavefunctions not read, or another parallelization, give a warning in the
log. The result stands as any other, since the thresholds are those of the
input, except after a restart from files written per process with another
parallelization, or one not known: it is not reused
(`core.store.restart_problem`), and the user decides. An `input.pwi` that
sets `startingwfc` other than `'file'` makes pw.x skip the wavefunctions,
recorded as not read. `wfcdir`, where pw.x writes the wavefunctions of each
process, must be a directory inside the calculation directory (checked in
the settings and by the worker): it is removed with `outdir` before an
attempt from scratch and never shared with another calculation.

**Launch of an attempt not recorded.** An attempt added to the record by the
code (an empty claim released, or a claim whose worker stopped before writing
the attempt) takes its launch from the groups that list it
(`reconcile.launch_for`): the launch they all give, or, when they give
different ones, the one of the group the user chooses (`--group`); when no
group lists it the launch is not known and nothing is relaunched. The level
is never assigned by default.

## Orchestrator

`execution/orchestrator.py` takes the steps of each system forward along the
graph of the entries they read: `match` → {`pes` → {`plotpes` → `shear`;
`adhesion` → {`ppes`; `charge`}}; `pes_scf` → `plotpes_scf`}, `plotpes` and
`plotpes_scf` reading also `match`, `shear` reading also `pes` and `match`.
The two PES, `pes` and `pes_scf`, need only `match`; `plotpes` interpolates
`pes` and `plotpes_scf` interpolates `pes_scf`. `adhesion` reads the PES
chosen in `[interface.adhesion]` and `match`: the graph of a run is built
from the settings (`interface_graph`, used by `build_orchestrator`), so with
`pes = "pes_scf"` it depends on `pes_scf` and not on `pes`, and `ppes` and
`charge` stay on `adhesion`. Which of the two PES `shear` reads is decided
with that step, and until then it depends on `pes`.
Each pass:

1. runs a cycle of the job manager;
2. registers the groups written and not registered;
3. for each system, goes through the requested steps in the order of the
   graph, the order given deciding only between steps that are ready. A
   **completed** step is asked again whether its entries are done and
   current (`Step.done`): one no longer current is **waiting** again, and is
   evaluated again in the same pass. A step whose dependency failed or is
   blocked is **blocked**; one whose dependency is not completed is
   **waiting**; the steps of a system run one at a time. A step starts by
   checking its entries (already done and current: **completed** without
   running) and the entries it reads (`check_prerequisites`: missing or
   stale, **blocked** with what is missing, nothing computed again), then
   prepares its calculations. A step whose analysis needs a compute node
   (`Step.manual_analysis`: `plotpes`, `plotpes_scf`) is checked the same
   way once ready, and never analysed by the orchestrator: with
   `[execution.analysis]` it is analysed in a job of its own (below,
   *Analyses in a job of their own*); without it, when it is not completed
   or blocked it is **manual**, with the command that analyses it (`to
   analyse by hand: run "triboworkflow analyse plotpes" where there is
   enough memory, or give [execution.analysis] …`), and a step that depends
   on it is **blocked** (`depends on plotpes manual`) until it is completed. Right after the preparation, and before a
   **running** step is followed, the calculations prepared are compared with
   those the settings give now (`Step.plan`: `calc_id` and fingerprint): one
   missing, one no longer given or a fingerprint that differs makes the step
   **failed** until `retry`, and none of its calculations is used. This is
   checked before the cycle of the job manager, which submits the jobs still
   missing from a group only for a step with calculations prepared and still
   current, no error of its own and not completed: the groups of any other
   step are held back, among them those of a step whose plan cannot be read
   and those of a preparation before `retry`. A running
   step is reconciled and decided; the launches go, after the archiving of
   the last attempt, in a new group `<system>.<step>.<k>`. When every
   calculation is completed the step analyses them and is **completed**; a
   calculation completed whose output was not parsed, or whose results
   cannot be used (an energy model not fully known, an output of another
   model), needs the user and is never relaunched, as for `analyse_step`;
   when what is left needs the user it has **failed**, with the
   calculations and why;
4. saves the steps in `orchestrator.json`: for each step its status,
   reason, calculations prepared, error of its own, calculations by outcome,
   those that need the user, the group of the job of its analysis (`job`,
   null for the other steps) and the time of its update. `schema_version` is
   2; this version reads no other.

A failure blocks only the descendants: the other branches go on, and the
workflow succeeds only when every requested step is completed. A step failed
for its calculations goes back to running when they are relaunched by hand
(`Orchestrator.relaunch`); one failed for an error of its own stays failed
until `retry`. After a restart, or a pass interrupted at any point, the next
pass recovers the jobs, reconciles the attempts and only then submits,
without duplicates.

A **manual** step ends the passes as a failure does: the orchestrator exits
when every step is completed, failed, blocked or manual, so it never waits
for the user while holding the run, and the workflow has not succeeded. The
user then runs `analyse <step>` where there is enough memory, with the
orchestrator ended or stopped (it holds `run.lock`); the command holds the
lock for the whole analysis, saves the step as **running** with no
calculations before it starts, and **completed** at its end. The next
`start` finds it completed from its entries, without analysing it again, and
goes on with the steps that depend on it. An analysis stopped before its end
(a signal, the walltime of a job) writes nothing: the step stays running with
no calculations, a pass leaves it manual again without analysing it, and the
next `analyse` starts it from the beginning, or the next pass gives it to a
job with `[execution.analysis]`. The interpolation of the PES never runs in
the orchestrator because its memory grows with the square of its points
([interface.md](interface.md)).

### Analyses in a job of their own

With `[execution.analysis]` ([configuration.md](configuration.md)) a ready
step with `Step.manual_analysis` gets a group of jobs
`jobs/<system>.<step>.<k>/`, as the calculations do, holding:

- `analysis.json`: system, step, scheduler and digest of the jobscript, in
  place of `manifest.json`;
- `jobscript.sh`: the jobscript of `[execution.analysis]`, copied unchanged,
  with one line added that runs the analysis: `<python> -m
  triboworkflow.cli.analysis_job <run>/jobs/<group>/analysis.json`;
- `submitted.json`, as for every group, and `outcome.json`, written by the
  job at its end.

The group asks for one job, submitted with the scheduler of `[execution]`;
the logs of the job go in its folder. The step is then **running**, with the
group in the key `job` of its state in `orchestrator.json`, and is followed
at every pass:

- once its entries are done and current (`Step.done`) it is **completed**,
  whatever the job;
- while the job is in the queue or runs, held, not reported, or its
  submission was refused or of outcome not known, the step stays running,
  with the state of the job and the command that resolves it;
- once the job is over (a final state confirmed by the last query; for
  Slurm `PREEMPTED`, reported again `polling_interval` seconds later, as not
  requeued) without the step done, the step is **blocked** if its
  prerequisites are no longer current, and **failed** otherwise, with the
  state of the job, its outcome (`no outcome written: the job stopped before
  its end (walltime, memory, node)` when it wrote none) and the folder of its
  logs. It is never submitted again automatically, whatever the state: the
  analysis has no checkpoint, and a job killed for its memory or walltime
  would be killed again. The user gives the jobscript what it needs and runs
  `retry <step>`; the next pass writes a new group, `<k>` + 1.

The job runs `Step.analyse`, as `analyse` does by hand, after checking that
the step is not done yet and that its prerequisites are current. It never
takes `run.lock`, which the orchestrator holds, and writes only the entries
of its step (under the lock of `system.json`, after checking again the
entries it read) and its outcome; it exits without analysing if a later
group of the step exists (a `retry` replaced it) or the group was withdrawn.
The steps of a system still run one at a time: a step waits while the job of
another runs, and the job waits for the steps that run. After an
interruption between the writing of the group and the saving of the state,
the next pass takes the group again instead of writing another, so two jobs
never run the same analysis. `analyse <step>` by hand is refused while the
job of the step may run according to the last query (`jobs.json`); a group
whose job was never submitted is withdrawn, so that it is never submitted
later. `prepare <step>` has nothing to prepare, and `submit <step>` writes
and submits the job, or follows it.

The three phases of a step can also be run one at a time, with the same
state: `prepare_step` (as the start in a pass, without the scheduler),
`submit_step` (a cycle of the jobs, then the step followed and its launches
submitted, without the analysis; for a step analysed in a job of its own,
that job) and `analyse_step` (the analysis, decided
from the files alone: each calculation must have a claim, if any, of its last
attempt, no group listing a next attempt, its last attempt ended completed,
and results that can be used: its output parsed, its energy model fully
known and its output agreeing with it (`can_reuse`); never the scheduler). Phases run by hand and passes take turns on
a run. A relaunch by hand (`Orchestrator.relaunch`) is refused, with nothing
written, when the settings changed since its step was prepared.

## Preparation and replacement

A step gives its calculations as the settings make them now, nothing written
(`Step.calculations`; for pw.x, `preparation.workflow_calculation`): input,
extxyz of the structure with the constraints of the input as `move_mask`
(for a look in OVITO), and the record with energy model and fingerprint.
`preparation.prepare_calculations` writes `input.pwi`, `structure_in.extxyz`
and, last, `record.json`; a directory with the first two and no record is a
preparation interrupted, written again. The changes made to the settings in
an input (`nstep = 1` and no `&IONS` in an scf) are logged, one line per
calculation and change, with the values, in `cli.log` or `orchestrator.log`.

What is on disk is compared first, and nothing is overwritten:

| Found | What happens |
|---|---|
| same fingerprint, same input and extxyz | kept as it is |
| same fingerprint, input different only in keys of the attempts or of the resources (`mixing_beta`, `max_seconds`, …), or extxyz different only in data the input does not take (e.g. `magmom`) | kept, and each difference logged with the value requested and the one kept, e.g. `namelists.CONTROL.max_seconds: requested 3600, kept 300` |
| another fingerprint, an input different elsewhere (changed by hand), an extxyz whose cell, positions, labels or constraints differ from the input, a file missing, a record that cannot be read, a directory without record with other files | the preparation stops with the list of these calculations and why; nothing is written for the step |

For a step already prepared, `prepare` only logs the calculations that
differ now. To apply the settings, `prepare <step> --replace`
(`Orchestrator.replace_step`) replaces every calculation that differs, the
kept ones included, and leaves those already current. A completed
calculation that differs only in keys of the attempts is then computed
again: to avoid it, restore those values before replacing.

Every check comes before any change. The replacement is refused, with the
reason of each calculation and nothing changed, when a calculation has a
claim whose worker may still run (an empty claim or one that cannot be read,
a claim without a job, or one whose job is not in a final state confirmed by
the last query; the claim a worker leaves on a calculation over does not
stop it), or a group of jobs listing it may still run or submit a
job: a job not in a final state confirmed by the last query, a submission of
outcome not known or interrupted (`pending`), jobs still to submit. A group
submitted in part, with some jobs over and the others refused or held back,
is neither complete nor can it be withdrawn: this version does not replace
its calculations, and the message says so, with how to go on (settings
restored, the group let end, then the replacement). Then:

1. the journal `replaced/<system>/<step>/<k>/replacement.json` is written,
   not done: the calculations, why, the fingerprints before and after, the
   groups to withdraw and, for each complete group listing a calculation,
   whether one of its workers took its attempt;
2. the groups listing a calculation and never submitted are withdrawn;
3. the directory of each calculation, with its attempts and `OUT/`, moves to
   `<k>/<step>/<label>/`;
4. the step goes back to waiting, as with `retry`, and the journal is marked
   done;
5. the step is prepared as usual.

From step 1 the launches of the old groups no longer count for the
calculations replaced (`replacement.retired_launches`): prepared again, they
start from attempt 1, and an attempt 1 of an old group is not theirs. The
other calculations of a group keep its launches, and whether the group made
progress stays the one recorded, so a calculation not taken stays not taken.
While a journal is not done, every phase, pass and action on the run stops
and names `prepare <step> --replace`, which completes it: a group already
withdrawn and a directory already moved are left as they are, the claims of
the calculations not moved yet are checked again, and nothing is deleted. An
interruption in step 5 is a preparation interrupted, completed by `prepare`.

## Lock of the run

The state of a run is in files read once and written back whole
(`orchestrator.json`, `jobs.json`, the groups, the attempts archived and the
calculations prepared). Every command of the command line that changes the
run, and the orchestrator in the background for its whole life, holds
`<run>/run.lock` (`execution/lock.py`) before reading anything: a second
holder is refused and told who holds the run. The lock is created with
`O_EXCL` and holds the pid, the host, the time, the command and a token; a
holder removes it only while it still holds its token. A lock left by a
process gone from this host is replaced with a warning, one of another host
or one that cannot be read is never touched. A lock is replaced, and adopted
by the orchestrator in the background, only by the process holding
`<run>/run.lock.takeover`, also created with `O_EXCL`: under it the lock is
read again and removed only if it is still the one left, so two processes
that find the same lock left never both hold the run. A takeover file left
by a process stopped while replacing the lock is never removed by the code:
until it is removed by hand, after checking that no command of the run runs,
a lock left is not replaced. Whether `O_EXCL` excludes
processes on different nodes of the shared filesystem is checked on each new
machine (`locks` of `tests/hpc/hpc_check.py`).

## In the background

`execution/daemon.py` runs the orchestrator on the login node, as SCS does:

- `start(run_dir, factory, scheduler, factory_args)` checks that the
  submission command of the scheduler is found, creates `orchestrator.pid`
  with `O_EXCL` (host, time, command and a token of the start, no pid yet),
  takes the lock of the run with the same token (a run held by a command is
  refused, and the pid file removed) and starts the orchestrator in a new
  session, its output in `orchestrator.log`. The process handles SIGTERM,
  adopts the lock before reading anything of the run, builds the
  orchestrator with the factory (`"module:function"`, called with the
  directory and `factory_args`, written in its command) and then writes its
  own pid in the file, if the file still holds its token. `start` returns
  once that pid is written; a process that exits first, or does not write it
  within `timeout` seconds (120 by default, then it is killed), leaves
  neither the pid file nor the lock, and `start` raises an error naming the
  log, unless it exited with code 0 (nothing left to do);
- `status(run_dir)` tells whether it runs; a pid file of another host is
  never touched, one of a process gone is replaced at the next start;
- `stop(run_dir)` sends SIGTERM: the orchestrator ends its pass, saves its
  state and exits. The jobs submitted go on. The handler of the signal only
  records the request, so it cannot block the process whatever it is doing;
  between two passes the orchestrator checks it every 0.2 s. The lock of the
  run is removed when the process exits.

The commands of the command line (`triboworkflow start`, `stop`, `status`,
the phases of a step and the actions of the user) are described in
`cli.md`.

## Checking a new machine

`tests/hpc/hpc_check.py` checks the claims, the exit codes and the locks on
the shared filesystem of a machine, with a fake engine in place of pw.x: no
calculation runs. Run it from a login node, in the virtual environment where
the package is installed, on the filesystem where the runs will live.

**Jobscript.** Resources and environment only, e.g. for Slurm:

```bash
#!/bin/bash
#SBATCH --job-name=check
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:30:00
#SBATCH --account=<account>
#SBATCH --partition=<partition>
source <venv>/bin/activate
```

**Claims and exit codes.** Prepare the run and look at the commands (dry
run), then submit:

```bash
python tests/hpc/hpc_check.py prepare <dir>/check_run --jobscript job.sh --launcher "srun" --calcs 40 --groups 8 --walltime 1800 --margin 600 --sleep 20
```

Run it again on a new directory with `--submit` to submit the jobs. Each fake
calculation runs through the launcher (`--launcher`, as in `run_command`)
and exits on every rank with 0, 2, 3 or 255 in turn, the codes of pw.x 7.2.
Only rank 0 writes the output and the log of the run: the rank is the one
the launcher gives, from `OMPI_COMM_WORLD_RANK`, `PMIX_RANK`, `PMI_RANK` or
`SLURM_PROCID`, the first one set and not empty (under `mpirun` in a Slurm
job, `SLURM_PROCID` is 0 in every process). Before an exit code other than
0 the other ranks wait a few seconds, since the launcher stops the
remaining processes once one exits so. Once the jobs are over:

```bash
python tests/hpc/hpc_check.py report <dir>/check_run
```

The report says whether each calculation ran at most once, compares the
exit code of the fake engine with the one the worker saw through the
launcher, lists the `max_seconds` written for each job, how the
calculations were spread over jobs and hosts, and the logs that cannot be
read. With a short walltime (e.g.
`--walltime 900 --margin 600 --sleep 120` and `--time=00:15:00`) each
worker stops after three calculations for lack of time.

**Locks.** Many processes on several nodes try to take the same locks, with
`O_EXCL` and with `mkdir`; within one allocation (e.g. `salloc -N 2`):

```bash
srun -N 2 -n 64 python tests/hpc/hpc_check.py locks <dir>/locks --method excl --start-at $(( $(date +%s) + 30 ))
srun -N 2 -n 64 python tests/hpc/hpc_check.py locks <dir>/locks --method mkdir --start-at $(( $(date +%s) + 30 ))
python tests/hpc/hpc_check.py locks-report <dir>/locks
```

`report` and `locks-report` exit with status 1 when a calculation ran twice,
a log cannot be read, or a lock had more than one owner.

The check with real pw.x and pp.x calculations through the whole chain,
`tests/hpc/qe_check.py`, is described in `hpc_check.md`.
