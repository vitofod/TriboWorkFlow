# States of a pw.x output

`triboworkflow.engines.qe.status` tells, from the output of `pw.x`, whether a
`scf` or `relax` calculation is over and, if it is not, why it stopped. It
reads no energy, force or structure: those belong to the parser.

```python
from triboworkflow.engines.qe.status import read_pw_status, record_status

pw = read_pw_status('output.pwo', 'relax')   # the caller knows the calculation
status = record_status(pw, job='killed')     # states for the record
```

## Last run

An output may hold several runs, appended one after the other when a
calculation is restarted; each begins with `Program PWSCF v.<version> starts
on ...`. Only the last run is read, and `runs` counts them all. A text
without that line is an error.

pw.x prints `nstep` in its summary only for an ionic relaxation: if the
summary is complete and disagrees with the calculation given by the caller,
the output is an error.

## States

| State | Values | Read from |
|---|---|---|
| `process` | `ended`, `crashed`, null | `JOB DONE.` (any outcome); `Error in routine` without `JOB DONE.`; neither: truncated |
| `scf` | `converged`, `not_converged`, null | the last `Self-consistent Calculation` cycle: `convergence has been achieved in`, `convergence NOT achieved after`, or no outcome |
| `geometry` | `converged`, `max_steps`, `not_converged`, `n/a`, null | `n/a` for `scf`. For `relax`: `bfgs converged in`; `The maximum number of steps has been reached.`; a run that ended or crashed without either; null if truncated |
| `final_coordinates` | `complete`, `incomplete`, `absent`, `n/a` | `Begin final coordinates` … `End final coordinates`; `n/a` for `scf` |
| `stop_reason` | see below, or null | |

Null means that the output does not tell. A truncated output (no `JOB
DONE.`, no error) may belong to a job still running or killed from outside:
it is never complete, never taken as not converged, and is not restarted
automatically while its process is null. An unknown SCF outcome is null,
never `not_converged`.

Geometric convergence is only what the optimizer declares. The final
coordinates are kept apart: pw.x also writes them when the optimizer gives
up, and an output may end inside them, so a declared convergence with
incomplete coordinates is `geometry` `converged` with `final_coordinates`
`incomplete`.

Stop reasons:

| `stop_reason` | Meaning |
|---|---|
| `completed` | SCF converged (`scf`) or the optimizer declared convergence (`relax`) |
| `scf_not_converged` | the last SCF cycle did not converge |
| `max_steps` | the ionic steps reached `nstep` |
| `optimizer_failed` | `bfgs failed after ... convergence not achieved` |
| `max_seconds` | `Maximum CPU time exceeded` |
| `user_request` | `Program stopped by user request` (an exit file or a signal) |
| `error` | `Error in routine` without `JOB DONE.` |
| `truncated` | the output ends without a closing line |
| null | the run ended for a reason not recognized |

`clean_stop` is true when pw.x closed the run (`JOB DONE.`). An exact restart
needs it, but it is not enough: whether the restart files are on disk is not
written in the output.

## States of the record

`record_status(pw, job)` gives the `Status` of the record (`parsing` stays
null for the parser). The output decides whenever it can; what the scheduler
reports about the job is passed explicitly and gives only the process of a
truncated output, while the SCF and the geometry stay as read:

| `job` | `process` | `Status.summary` |
|---|---|---|
| null | null | `unknown` |
| `running` | `running` | `running` |
| `killed` (e.g. walltime) | `killed` | `crashed` |
| `finished` (over, nothing else known) | `crashed` | `crashed` |

Of a closed output, `Status.summary` is `completed` when the SCF and the
geometry converged, `not_converged` for any other stop with `JOB DONE.`
(whatever the stop reason), and `crashed` after an error.

## Versions

Versions 7.1, 7.2 and 7.3 are read; any other is an error.
`version_validated` is true for 7.1 and 7.2, whose lines were checked on real
outputs. The lines of the step limit, of `max_seconds`, of a stop by request
and of a failed optimizer come from the source code of 7.2 and were not
checked on real outputs. `scf_must_converge = .false.` and `vc-relax` are
not handled.

## Tests

`tests/test_qe_status.py` uses synthetic outputs only. Real outputs are never
stored in the repository: `tests/test_qe_status_corpus.py` reads them from the
directory named by `TRIBOWORKFLOW_PWO_CORPUS` and is skipped without it.
Comment lines beginning with `###` in those files are removed before reading.
