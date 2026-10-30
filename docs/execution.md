# Execution on HPC

Calculations run only on an HPC machine, inside jobs of its batch scheduler.
This page describes how a group of calculations is laid out, submitted, run
and followed; how the calculations are reconciled with their jobs and
relaunched; how the orchestrator takes the steps of a workflow forward, in
the background on the login node; and how to check the execution on a new
machine with a fake engine. The values of the remedies are decided but not
yet written in the inputs, so a relaunch above level 1 is not submitted.

## Layout of a run

```
<run>/
├── config.toml                  settings of the run (configuration.md, cli.md)
├── systems/<system>/
│   ├── system.json
│   └── <step>/<label>/          one directory per calculation
│       ├── input.pwi            input prepared for the calculation, never changed
│       ├── run.pwi              input of the current attempt
│       ├── output.pwo           output of the current attempt
│       ├── claim                taken by the worker running the attempt
│       ├── record.json
│       ├── OUT/                 outdir of pw.x
│       └── attempts/<n>/        run.pwi and output.pwo of attempt n
├── jobs/<name>/                 one group of jobs
│   ├── manifest.json
│   ├── jobscript.sh
│   └── submitted.json
├── jobs.json                    registry of the job manager
├── orchestrator.json            steps of each system
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
  how to run them: scheduler, `run_command`, `walltime`, `max_seconds_margin`
  (from `[execution]`, see `configuration.md`). Each calculation names the
  attempt the group runs (`Launch`): its number `n`, its kind (`initial`,
  `unchanged`, `remedy`, `manual`), the level of the remedies and, above
  level 1, the values of the level and the effective values at level 1. A
  new attempt of a calculation always goes in a new group.
- `jobscript.sh`: the jobscript of `[execution]`, copied byte for byte, with
  one line added at the end:

  ```
  <python> -m triboworkflow.execution.worker <run>/jobs/<name>/manifest.json
  ```

  The jobscript of the user is never changed. It sets the resources and the
  environment (modules, virtual environment) and must not launch pw.x
  itself.
- `submitted.json`: the ids of the jobs, each added right after its
  submission.

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
   job already submitted is never submitted again;
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

## Worker

Each job runs one worker (`execution/worker.py`). The worker goes once
through the calculations of the manifest. It runs attempt `n` of a
calculation only while the record has `n - 1` attempts, so a late job of a
group that was replaced finds the attempt made and skips it, and only if no
other worker has taken it:

1. creates `claim` with `O_CREAT | O_EXCL`: exactly one worker succeeds, the
   others skip the calculation; the claim holds the attempt, the job id, the
   restart count of the job (`restart_count`, from `SLURM_RESTART_COUNT`,
   null when not set), the host, the pid and the time;
2. adds attempt `n` to the record, with the process `running` and the launch
   of the manifest in `remedies`;
3. writes `run.pwi`, a copy of `input.pwi` with `max_seconds` set;
4. creates `output.pwo` with `O_EXCL`;
5. runs `<run_command> -in run.pwi >> output.pwo` with bash, in the
   calculation directory;
6. records the end time, the exit code, the digests of `run.pwi` and
   `output.pwo`, the states read from the output, the reason of the stop and
   whether pw.x closed the run.

If something is wrong before step 2 (no record, a calculation other than
`scf` or `relax`, files of an earlier attempt still in place, a launch above
level 1 whose values this version does not write in the input), the worker
removes its claim, logs the error and goes on. A worker killed with its job
leaves the attempt with the process `running`; what happened is then known
from the scheduler.

**Time limit.** `max_seconds` of each attempt is `walltime` minus the time
since the worker started minus `max_seconds_margin`, capped by the
`max_seconds` of `input.pwi` when given. pw.x then stops in time to write its
restart files. When nothing is left, the worker takes no more calculations.
The `walltime` of `[execution]` must be the one of the jobscript. In
`input.pwi`, `max_seconds` must be alone on its line of `&CONTROL`.

**Job id.** Read from `SLURM_JOB_ID`, `PBS_JOBID` (PBS, Torque), `LSB_JOBID`
(LSF) or `JOB_ID` (SGE); null for HTCondor and outside a job.

## Archiving an attempt

Before a calculation runs again, its last attempt is archived
(`execution/archive.py`): `run.pwi` and `output.pwo` move to `attempts/<n>/`
after a check of their digests, the record gets the new paths, and the claim
is removed last. An interrupted archiving is completed by running it again.
An attempt still recorded as `running` is not archived; `OUT/` is not
touched. An attempt is archived only when it is relaunched.

An empty claim (a worker that stopped right after creating it) gets the
number the worker would have used, the attempts in the record plus one, and
the attempt is added to the record with its end unknown. It also gets its
launch when one is given (`archive_attempt(..., launch=)`), so that it can
be relaunched at its level: `Orchestrator.release_claim`, asked by the user
once the worker is surely over, takes it from the groups that list the
attempt (`reconcile.launch_for`). A release interrupted leaves the claim
with the number of the attempt alone, not yet in the record: the
reconciliation reports it as an empty claim released in part, and
releasing the claim again completes it.

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

The fixed attempt gets the state of the job as printed, the states read from
the output (the process `killed` when the scheduler stopped or requeued the
job, `crashed` otherwise, unless pw.x closed the run), the reason of the
stop and the digests of its files. The output decides when pw.x closed the
run: a calculation whose worker was killed after pw.x ended is completed.

How the job ended decides when the output does not:

| Scheduler | Relaunched unchanged | Left to the user |
|---|---|---|
| Slurm | `TIMEOUT`, `NODE_FAIL`, `BOOT_FAIL`, `PREEMPTED`, `COMPLETED`, `FAILED`, requeued | `CANCELLED`, `DEADLINE`, `OUT_OF_MEMORY`, others |
| PBS, Torque | `F`, `C` | — |
| HTCondor | `4` (completed) | `3` (removed) |

So a `qdel` on PBS or Torque cannot be told from another end of the job and
is relaunched. An error of pw.x (`Error in routine`) is left to the user,
with its routine, code and message read from the output, which is kept. A
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
level 1. **This version decides the launches above level 1 but does not
submit them**: their values are not written in the input yet, and the
calculation is reported as needing the user.

`policy.manual_launch` gives the launch of a relaunch asked for by the user:
allowed only when the worker of the last attempt is surely over (no claim,
or the run of the job of the claim over), at the level of the last attempt
and with its input; the count of the unchanged relaunches starts again.

**Launch of an attempt not recorded.** An attempt added to the record by the
code (an empty claim released, or a claim whose worker stopped before writing
the attempt) takes its launch from the groups that list it
(`reconcile.launch_for`): the launch they all give, or, when they give
different ones, the one of the group the user chooses (`--group`); when no
group lists it the launch is not known and nothing is relaunched. The level
is never assigned by default.

## Orchestrator

`execution/orchestrator.py` takes the steps of each system forward along the
graph of the entries they read: `match` → `pes` → {`plotpes` → `shear`;
`adhesion` → {`ppes`; `charge`}}, `shear` reading also `pes` and `match`.
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
   prepares its calculations. Right after the preparation, and before a
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
   calculation is completed the step analyses them and is **completed**;
   when what is left needs the user it has **failed**, with the
   calculations and why;
4. saves the steps in `orchestrator.json`.

A failure blocks only the descendants: the other branches go on, and the
workflow succeeds only when every requested step is completed. A step failed
for its calculations goes back to running when they are relaunched by hand
(`Orchestrator.relaunch`); one failed for an error of its own stays failed
until `retry`. After a restart, or a pass interrupted at any point, the next
pass recovers the jobs, reconciles the attempts and only then submits,
without duplicates.

The three phases of a step can also be run one at a time, with the same
state: `prepare_step` (as the start in a pass, without the scheduler),
`submit_step` (a cycle of the jobs, then the step followed and its launches
submitted, without the analysis) and `analyse_step` (the analysis, decided
from the files alone: each calculation must have a claim, if any, of its last
attempt, no group listing a next attempt, and its last attempt ended
completed; never the scheduler). Phases run by hand and passes take turns on
a run. A relaunch by hand (`Orchestrator.relaunch`) is refused, with nothing
written, when the settings changed since its step was prepared.

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
Once the jobs are over:

```bash
python tests/hpc/hpc_check.py report <dir>/check_run
```

The report says whether each calculation ran at most once, compares the
exit code of the fake engine with the one the worker saw through the
launcher, lists the `max_seconds` written for each job, and how the
calculations were spread over jobs and hosts. With a short walltime (e.g.
`--walltime 900 --margin 600 --sleep 120` and `--time=00:15:00`) each
worker stops after three calculations for lack of time.

**Locks.** Many processes on several nodes try to take the same locks, with
`O_EXCL` and with `mkdir`; within one allocation (e.g. `salloc -N 2`):

```bash
srun -N 2 -n 64 python tests/hpc/hpc_check.py locks <dir>/locks --method excl --start-at $(( $(date +%s) + 30 ))
srun -N 2 -n 64 python tests/hpc/hpc_check.py locks <dir>/locks --method mkdir --start-at $(( $(date +%s) + 30 ))
python tests/hpc/hpc_check.py locks-report <dir>/locks
```

`report` and `locks-report` exit with status 1 when a calculation ran twice
or a lock had more than one owner.
