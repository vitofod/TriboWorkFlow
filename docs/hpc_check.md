# Integrated check on an HPC machine

`tests/hpc/qe_check.py` runs small pw.x and pp.x calculations through the
whole chain of the package on an HPC machine: preparation, groups of jobs,
the orchestrator in the background, polling, relaunches and restarts,
parsing, the charge density, the actions of the user. The steps of the
interface workflow are not in this version: the script gives the command
line test steps whose calculations are those of a case of the check (module
docstring of the script). It complements `tests/hpc/hpc_check.py` (fake
engine: claims, exit codes, locks; `execution.md`, "Checking a new
machine").

Everything runs by hand, from a login node, in the virtual environment of
the package. Nothing of the runs goes into the repository: the run folders
are copied to a local folder for the analysis, without the output
directories of pw.x (`OUT/`).

In the commands below:

```bash
REPO=<clone of the repository>
QC="python $REPO/tests/hpc/qe_check.py"
M=<folder of your machine file>/machine.toml
R=<folder of the runs, on the parallel filesystem>
```

## 0. Installation and machine file

1. Clone the repository; Python 3.12 (e.g. `uv python install 3.12`); the
   environment of the lock, with the package installed without index and
   without build isolation (`envs/README.md`):

   ```bash
   PYTHON=$(uv python find 3.12) VENV=<venv> bash $REPO/envs/dev/create_env.sh
   source <venv>/bin/activate && triboworkflow --help
   ```

2. Load the module of QE and check `which pw.x pp.x`.
3. Put the UPF files of Al and Si in a `pseudo_dir` and write their
   checksums there: `md5sum *.UPF > pseudo.md5`.
4. Copy `tests/hpc/machine_example.toml` and `tests/hpc/jobscript_example.sh`
   to a folder of your own and fill them in: account, venv, partition, QoS,
   module, `pseudo_dir`, `upf_md5`, the UPF file of each species; values of
   the namelists other than the defaults of the script if needed. The
   jobscript keeps `{time}` in its `--time` line: each case writes its own
   walltime there. Use `%j` in the names of the log files.
5. The runs live on the parallel filesystem of the jobs (`$R`); note which
   one, for Q11.

`write` builds every calculation of the run at once: an error of the
machine file or of the UPF files shows there, before anything is submitted.
Save the output of every command of the rounds, e.g. with
`2>&1 | tee -a $R/<run>/qe_check/console.log`.

## 1. Fake engine (`hpc_check.py`)

As in `execution.md`, with the launcher of the machine (`mpirun` here):

```bash
python $REPO/tests/hpc/hpc_check.py prepare $R/fake --jobscript <job.sh> --launcher "mpirun -np 4" --calcs 40 --groups 8 --walltime 1800 --margin 600 --sleep 20 --submit
python $REPO/tests/hpc/hpc_check.py report $R/fake
python $REPO/tests/hpc/hpc_check.py prepare $R/fake_short --jobscript <job15.sh> --launcher "mpirun -np 4" --walltime 900 --margin 600 --sleep 120 --submit
```

`<job.sh>` and `<job15.sh>` are jobscripts with a fixed `--time`
(`00:30:00` and `00:15:00`), not the template with `{time}`. Locks on two nodes, within one allocation (`salloc -N 2 ...`), the
contenders started by `mpirun` (or `srun` where it works):

```bash
mpirun -np 64 python $REPO/tests/hpc/hpc_check.py locks $R/locks --method excl --start-at $(( $(date +%s) + 30 ))
mpirun -np 64 python $REPO/tests/hpc/hpc_check.py locks $R/locks --method mkdir --start-at $(( $(date +%s) + 30 ))
python $REPO/tests/hpc/hpc_check.py locks-report $R/locks
```

The same locks on a second machine, if there is one (Q11).

## 2. One input of `gen` run by hand (case `single`)

```bash
$QC write $M $R/single --case single --system Al
cd $R/single && triboworkflow gen config.toml && sbatch jobscript.sh
$QC report $R/single        # once the job is over
```

## 3. The chain (case `chain`, Al and Si)

For each system (`--system Al`, then `--system Si`):

```bash
$QC write $M $R/chain_Al --case chain --system Al
$QC cli check -r $R/chain_Al
$QC cli start --dry-run -r $R/chain_Al
$QC cli start -r $R/chain_Al
$QC cli status --calcs -r $R/chain_Al        # now and then: jobs waiting, running
$QC snapshot $R/chain_Al                     # while jobs are queued
```

- The orchestrator in the background: `cat $R/chain_Al/orchestrator.pid`,
  then `ps -o pid,sid,cmd -p <pid>` (a session of its own: sid = pid).
- While jobs run: `$QC cli stop -r $R/chain_Al` (SIGTERM: the jobs go on),
  then `$QC cli start -r $R/chain_Al` again; the orchestrator takes the
  jobs back.
- In the run of Si, while the jobs of `adhesion` are queued or running:
  `cli stop`, change `ecutwfc` in `config.toml`, `$QC cli prepare adhesion
  --replace -r $R/chain_Si` (refused, nothing changed), restore `ecutwfc`,
  `cli start`.
- Once the run of Al is over: change `ecutwfc` in `config.toml`, then
  `$QC cli status -r $R/chain_Al`, `$QC cli prepare pes --replace -r
  $R/chain_Al` (the calculations of `pes` move to `replaced/`); restore
  `ecutwfc` and leave the run as it is.

At the end: `$QC snapshot $R/chain_Al` and `$QC report $R/chain_Al`.

## 4. Relaunches and restarts (case `restart`)

The walltime of the case is the margin (600 s) plus about two minutes:
`long_scf` and `long_relax` must run longer than that, so that pw.x stops at
the `max_seconds` set by the walltime. If they end earlier, make them
larger in `[structures]` (`big_size`, `big_layers`) of the machine file.

```bash
$QC write $M $R/restart --case restart --system Al
$QC cli start -r $R/restart
$QC snapshot $R/restart                      # once the run is over
$QC report $R/restart
```

Another `run_command`, a second run: `write` it to `$R/restart_b`, `cli
start`; once its first jobs run, `cli stop`, change `run_command` in its
`config.toml` (e.g. `-nk 2` into `-nk 1`), wait for those jobs to end, `cli
start`; at the end `snapshot` and `report`.

## 5. By hand: scancel, exit file, hold, requeue (case `manual`)

```bash
$QC write $M $R/manual --case manual --system Al
$QC cli prepare adhesion -r $R/manual
$QC cli submit adhesion -r $R/manual
squeue -u $USER                    # at once: scontrol hold <a job still PENDING>
$QC snapshot $R/manual             # while it is held
scontrol release <that job>
$QC cli start -r $R/manual
```

The claim of a calculation names the job that runs it
(`systems/<system>/adhesion/<label>/claim`, `job_id`):

- `to_cancel`: `scancel <its job>` while pw.x runs; the orchestrator
  relaunches it from scratch.
- `to_exit`: while pw.x runs, `touch
  systems/<system>/adhesion/to_exit/adhesion_to_exit.EXIT`; pw.x stops and
  the calculation waits for the user: `cli stop`, `$QC cli relaunch
  adhesion/to_exit -r $R/manual`, `cli start`.
- `to_requeue`: `scontrol requeue <its job>` while pw.x runs; `$QC snapshot`
  right after, and again once the job runs again.

At the end `snapshot` and `report`.

## 6. Empty claim (case `claim`), phases by hand

The jobscript of this case stops the worker right after it creates the
claim of `victim` (once).

```bash
$QC write $M $R/claim --case claim --system Al
$QC cli prepare adhesion -r $R/claim
$QC cli submit adhesion -r $R/claim
$QC cli release-claim adhesion/alive --worker-over -r $R/claim     # while alive runs: refused
$QC snapshot $R/claim
```

Once the job of `victim` is over (its worker killed, the claim empty):

```bash
$QC cli submit adhesion -r $R/claim                                 # victim: uncertain
$QC cli release-claim adhesion/victim --worker-over -r $R/claim
$QC cli relaunch adhesion/victim -r $R/claim
$QC cli submit adhesion -r $R/claim                                 # once its job is over
$QC cli analyse adhesion -r $R/claim
$QC snapshot $R/claim && $QC report $R/claim
```

## 7. Charge density (case `density`)

```bash
$QC write $M $R/density --case density --system Al
```

First, in its `config.toml`, `disk_io = "nowf"` and `$QC cli prepare charge
-r $R/density`: the step fails while checking, naming `disk_io`, and nothing
is written. Restore `disk_io`, then:

```bash
$QC cli retry charge -r $R/density
$QC cli start -r $R/density
$QC snapshot $R/density && $QC report $R/density        # once the run is over
```

The pp.x example of QE 7.2: run `PP/examples/example01` (`run_example`, with
the `environment_variables` of the machine), then pp.x on its scf with
`-in` and an input with `&INPUTPP` only (no `&PLOT`, `plot_num = 0`), and
keep its output. On a copy of the results:

```bash
mkdir -p <dir>/reference && cp results/sicharge results/si.scf.out <dir>/reference/
TRIBOWORKFLOW_QE_PP_EXAMPLE=<dir> python -m pytest $REPO/tests/test_qe_density.py -k pp_example
```

## What to copy back

For each run: the run folder without `OUT/`, e.g. `rsync -a --exclude OUT/
<run>/ <local folder>/<run>/`; it holds the records, the outputs, the
claims, the groups with the logs of the workers (`jobs/*/job.*.err`), the
snapshots and `qe_check/console.log`. Also the outputs of `hpc_check.py`,
of `locks-report` and of the pp.x example. `qe_check.py report` reads a
copied run as it reads one on the cluster.

## What the report says

`report` prints each calculation with its attempts, then one verdict per
expectation: `ok`; `not observed` (the run did not show it, e.g. no
snapshot yet, or a calculation that ended before `max_seconds`); `note` (to
look at: the symmetries of pw.x differ from those of spglib, Q17);
`PROBLEM` (an inconsistency of the files, or a behaviour other than the
package's), with exit status 1.

| Round | Case | Verdicts |
|---|---|---|
| 2 | `single` | `single`, `single symmetry (Q17)` |
| 3 | `chain` | `<calc> completed`, `stress`, `z only`, `symmetry (Q17)`; `steps` |
| 4 | `restart` | `restart after max_seconds / max_steps / scf_not_converged`, `SCF restarted from its iteration`, `level 2`, `margin at max_seconds (Q09)`; `other run_command` (`restart_b`) |
| 5 | `manual` | `from scratch after killed`, `relaunch by hand after user_request`, `held job`, `requeued job` |
| 6 | `claim` | `empty claim released`, `completed` |
| 7 | `density` | `charge density`, `densities comparable`, `disk_io nowf` |
