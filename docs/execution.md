# Execution on HPC

Calculations run only on an HPC machine, inside jobs of its batch scheduler.
This page describes how a group of calculations is laid out, submitted and
run, and how to check the execution on a new machine with a fake engine.
Polling the jobs and relaunching calculations are not part of the package
yet.

## Layout of a run

```
<run>/
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
└── jobs/<name>/                 one group of jobs
    ├── manifest.json
    ├── jobscript.sh
    └── submitted.json
```

pw.x runs inside the calculation directory, so the relative `outdir = 'OUT'`
of each calculation is its own; the `prefix` is `<step>_<label>`
(`execution/layout.py`).

## Groups of jobs

The calculations of a group are spread over several jobs, submitted from one
jobscript (`execution/jobs.py`):

- `manifest.json`: the calculations, in the order the workers take them, and
  how to run them: scheduler, `run_command`, `walltime`, `max_seconds_margin`
  (from `[execution]`, see `configuration.md`).
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

**Job listings** (`execution/scheduler.py`) are read for Slurm, LSF and SGE.
For PBS, Torque and HTCondor the columns the parser reads do not match the
output of `qstat -u` and `condor_q`, so a running job would be missing:
their listings raise an error rather than return lists that look empty. The
same holds for telling running and queued jobs apart with a custom table.

## Worker

Each job runs one worker (`execution/worker.py`). The worker goes once
through the calculations of the manifest and runs each one that no other
worker has taken:

1. creates `claim` with `O_CREAT | O_EXCL`: exactly one worker succeeds, the
   others skip the calculation;
2. adds attempt `n` to the record, with the process `running`;
3. writes `run.pwi`, a copy of `input.pwi` with `max_seconds` set;
4. creates `output.pwo` with `O_EXCL`;
5. runs `<run_command> -in run.pwi >> output.pwo` with bash, in the
   calculation directory;
6. records the end time, the exit code, the digests of `run.pwi` and
   `output.pwo`, the states read from the output, the reason of the stop and
   whether pw.x closed the run.

If something is wrong before step 2 (no record, a calculation other than
`scf` or `relax`, files of an earlier attempt still in place), the worker
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
touched.

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
