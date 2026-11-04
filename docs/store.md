# File store

`triboworkflow.core.store` keeps the data of the workflows on files, without
a database: one record per calculation and one aggregated document per
system. Every file is written atomically. Nothing is reused unless the record
is complete and its fingerprint is the one expected now.

## Layout

```
<run>/systems/<system>/
├── system.json             aggregated document and provenance of its entries
├── system.json.lock        present only while a writer holds system.json
└── <step>/<label>/         one directory per calculation
    ├── record.json
    └── ...                 input, output and structures of the calculation
```

A store covers one system and never reads the directories of other systems.
Workflows that follow one another (bulk, surfaces, interfaces) exchange files
of structures and parameters; none of them is a prerequisite of another.

## Three phases, invoked separately

| Phase | Reads | Writes | Where it runs |
|---|---|---|---|
| preparation | structures, settings, documents of earlier steps | inputs, records with the expected fingerprint | anywhere, including a local machine |
| execution | records | records (states, attempts, results) | HPC only: a local attempt is an error |
| analysis | records, `system.json`, files | `system.json` (entries and their provenance) | anywhere, including a local machine |

Each phase can be started on its own on the files left by the previous one;
an orchestrator may chain them but is never required. Calculations (DFT, MD,
MLIP) never run locally.

## Records

`record.json` (`record.py`):

| Key | Content |
|---|---|
| `schema_version` | 4 |
| `calc_id` | `<step>/<label>`: the directory of the calculation within the system |
| `system`, `step`, `label` | names of the three directories; no `/`, `\`, NUL, `.` or `..` |
| `calc_type` | kind of calculation, which fixes its constraints |
| `engine` | `name`, `program`, `version` (as printed in the output; null until read) |
| `energy_model` | written by the engine, not empty: pseudopotentials and their digests or the MLIP model, cutoffs, k-points, smearing, spin, vdW; a null marks a part not known |
| `fingerprint` | `sha256:` and 64 hex digits: digest of the normalized input and of the energy model |
| `status` | `process`, `scf`, `geometry`, `parsing`, each null until known |
| `attempts` | one entry per run, numbered from 1 (below) |
| `results` | results read from the output, or null; never null once `parsing` is `ok` |
| `observed` | what the output tells of the energy model, with the keys and the form of `energy_model` (written by the parser), or null; never null once `parsing` is `ok`, empty if the engine observes nothing |
| `files` | `{role: file name}` relative to the calculation directory |

Unknown or missing keys, another schema version and invalid values are
errors, with the file in the message.

States:

| State | Values |
|---|---|
| `process` | `pending`, `running`, `ended`, `crashed`, `killed` |
| `scf` | `converged`, `not_converged` |
| `geometry` | `converged`, `not_converged` (the run is over without convergence), `max_steps`, `n/a` (no relaxation) |
| `parsing` | `ok`, `failed` |

For whoever looks at a calculation, `Status.summary` sums the states up in
one word:

| `summary` | States |
|---|---|
| `pending` | `process` `pending` (queued) |
| `running` | `process` `running` |
| `completed` | `process` `ended`, `scf` `converged`, `geometry` `converged` or `n/a` |
| `not_converged` | `process` `ended` otherwise, whichever part did not converge |
| `crashed` | `process` `crashed` or `killed` |
| `unknown` | `process` null |

It describes the calculation, not the reading of its output (`parsing` does
not enter it); the separate states tell the details.

A record is complete when `process` is `ended`, `scf` is `converged`,
`geometry` is `converged` or `n/a`, and `parsing` is `ok`. `can_reuse(record,
fingerprint)` is true only for a complete record with results, with that
fingerprint, with an energy model fully known and with an output that agrees
with it, so deciding reuse and reading the results always agree. A record
whose last attempt continued one stopped by `max_seconds`, `user_request` or
`scf_not_converged`, which leave files written once per MPI process, is
reused only if `restart.same_parallelization` is true (`restart_problem`).
`reuse_problem` tells why a record cannot be reused. Which results each kind
of calculation must have (an energy, finite forces, ...) is checked by the
parser of its engine.

A null value in `energy_model` marks a part of the model that is not known,
e.g. the expected checksum of a pseudopotential not given in the settings
(`unknown_parts` lists their paths). Such a calculation can be prepared and
run, but its results are never reused and its energy is never combined with
others.

`identity_differences(record)` compares each value of `observed` with the
value at the same path of `energy_model` (keys the energy model does not
have are not compared; two lists of different length differ as a whole),
e.g. `species[0].md5: expected '...', observed '...'`. An output that
differs keeps its results and `parsing` `ok`, but does not stand for its
energy model: it is never reused, never combined, and the `tasks` view
refuses it. Absence (a null in `energy_model`) and difference are told
apart. For pw.x (`engines/qe/results.py`) `observed` holds the version, with
as many components as the version expected (`7.3` stands for every `7.3.x`,
`7.3.1` for that one only), and for each species its label, the name of its
UPF file, its checksum and the path printed.

`record.results_digest` identifies the results themselves: a calculation run
again with the same input, or an output read again, that gives different
results has a different digest. Writing the same results again keeps it.

A rerun replaces the states and the results of the record; its attempts list
keeps every run. Nothing is deleted when an analysis becomes invalid.

Attempts (`Attempt`). Each field has one source, and stays null until that
source establishes it:

| Key | Content | Source |
|---|---|---|
| `n` | number of the attempt, 1, 2, ... in the order of the list | worker |
| `job_id` | id of the job, as given by the scheduler; null outside a job | worker |
| `host` | machine the attempt ran on | worker |
| `started`, `ended` | ISO 8601 times of the start and of the end of the run | worker |
| `max_seconds` | time limit given to the engine for this attempt, s | worker |
| `max_seconds_cap` | upper limit of `max_seconds` in the prepared input; null if none | worker |
| `run_command` | command that ran the engine, up to the input file (e.g. `mpirun -np 32 pw.x -nk 4`); null when not recorded | worker |
| `jobscript_digest` | `file_digest` of the jobscript of the user that the job ran, which sets its resources; null when not recorded | worker |
| `input`, `output` | `{"path", "digest"}`: file relative to the calculation directory and its `file_digest` | worker, archiving |
| `exit_code` | exit status of the command that ran the engine; null if the worker did not see it end | worker |
| `job_state` | state of the job as printed by the scheduler, written when the reconciliation fixes an attempt whose worker did not end it (for a requeued job, the state of its new run) | reconciliation |
| `stop_reason` | why the run stopped, read from the output | output |
| `clean_stop` | whether the engine closed the run, read from the output | output |
| `parse_error` | why the output could not be parsed; given if and only if `status.parsing` is `failed` | parser |
| `status` | `process`, `scf`, `geometry`, `parsing` at the end of the attempt | output, job |
| `remedies` | launch of the attempt: `kind` (`initial`, `unchanged`, `remedy`, `manual`), `level` of the remedies, `values` of the level and `initial_values` (effective values at level 1), both empty at level 1, and `restart` (whether it continued the previous attempt from its output directory); empty when not known | worker, reconciliation |
| `restart` | for an attempt launched to continue the previous one, what its output says it read from the files of that one: `positions`, `density`, `wavefunctions`, `scf_iteration`, `bfgs_history`, each true, false or null (not told), and `same_parallelization` (the parallelization printed equals that of the previous attempt; null if either is not known); null for any other attempt | output |
| `outdir_removal` | for an attempt from scratch after the first, the removal of the directories the engine writes (`outdir`, and `wfcdir` when given) before the run: `started` (begun, its end not recorded), `removed`, `absent` (nothing to remove) or `failed` (the attempt did not run); null for a first attempt or a restart | worker |

The exit code, the state of the job and the reason of the stop are kept
apart and never derived from one another; the states of the calculation
come from the output, and from the job only where the output says nothing.
The relaunches count their attempts from `remedies`, so the record alone
tells how many unchanged relaunches were made at the current level.

`check_energy_model(records)` raises an error when the records do not share
one energy model (engine name and program, and `energy_model`), when the
energy model of one of them is not fully known, even if identical to the
others, or when the output of one of them differs from its energy model
(`identity_differences`): energies of different or unknown models never
enter one difference.

## Fingerprints

`fingerprint(value)` is the SHA-256 of the canonical JSON of a value
(`fingerprint.py`): sorted keys, no spaces, tuples and numpy arrays as lists,
numpy scalars as Python values, floats written with the shortest text that
reads back to the same float. `1` and `1.0` differ, and so do `True` and `1`.
NaN, infinities, non-string keys and other types are errors. What enters the
fingerprint of a calculation is decided by each engine. `content_digest(value)`
identifies data (results, entries read by an analysis) with the same rules,
but accepts non-finite floats in their tagged form, since results may hold
them (a fit whose uncertainty cannot be estimated). `file_digest(path)`
digests the bytes of a file in the same form.

## Aggregated document

`system.json` (`system.py`):

```json
{"schema_version": 1,
 "document": {"mid": "...", "structure": {}, "pes_scf": {}, "...": "..."},
 "provenance": [{"path": ["pes_scf", "energy"], "step": "pes_scf",
                 "parameters": {}, "calculations": {"pes_scf/aa": "sha256:..."},
                 "inputs": {"highsym": "sha256:..."}, "files": {},
                 "results": {"pes_scf/aa": "sha256:..."},
                 "fingerprint": "sha256:..."}]}
```

`document` holds the data of the system with the historic keys of the
workflows (`structure`, `comp_params`, `inter_params`, `highsym`, `pes`,
`pes_scf`, `adhesion`, `ppes`, `charge`, `shearstr`, ...); it is null until
created.

`SystemDocument.set(path, value)` merges leaves: a dict sets each of its
leaves (values that are not dicts; lists are leaves) below `path` and keeps
the other keys met on the way, so an empty dict changes nothing; any other
value replaces what is at `path`. A key on the way that holds a value other
than a dict (null included) is an error, and nothing is changed.
`replace(path, value)` puts a value as a whole. `get` returns copies.

### Provenance and completed steps

`Provenance.build(step, parameters, calculations, inputs, files)` describes
what an entry was computed from:

- `parameters`: processing parameters of the analysis (interpolation and MEP
  settings, seeds, ...), stored in full;
- `calculations`: `{calc_id: fingerprint}` of the calculations used;
- `inputs`: data read from other entries, stored as digests of their content;
- `files`: files read, stored as digests of their bytes under names chosen by
  the step, so a moved file with the same content counts as the same;
- `results`: `{calc_id: results_digest}` of the results actually consumed.

`Provenance.build` gives the provenance expected now, which is known before
any result (preparation computes it). After the analysis,
`expected.with_results(records)` adds the digests of the records it read,
exactly one per calculation, each reusable with its fingerprint; only that
provenance can be recorded with `set_provenance`.

`is_done(document, path, expected, system_dir)` is true when the entry exists
and is not empty, its recorded provenance equals `expected` (results apart)
and every calculation in it still has a reusable record with the same
fingerprint and the same results that were consumed. Changing a calculation
or its results, an input, a file or a processing parameter therefore makes
the analysis that used it not done; nothing is deleted. Since inputs count by
content, an entry computed from another one is invalidated as soon as that
one is computed again with a different result. Writing an entry drops the
provenance of every entry it overlaps (the same path, a parent or a child):
the step records the new provenance after writing.

Content alone cannot tell whether an entry read by an analysis is up to date
with the current settings of the step that computes it: after the settings of
an interpolation change, the old interpolated PES is still in the document,
unchanged, until the interpolation runs again. Each analysis therefore states
the entries it reads as prerequisites, with the provenance their steps expect
now, and calls `check_prerequisites(document, prerequisites, system_dir)`
before computing: `PrerequisiteError` lists the entries missing or not
current. Nothing is recomputed in cascade; the user, or the orchestrator,
runs the steps that compute them.

### Single writer

```python
with SystemWriter(system_dir) as writer:
    writer.document.set(['pes_scf', 'energy'], summary)
    writer.document.set_provenance(['pes_scf', 'energy'], provenance)
    writer.save()
```

Entering creates `system.json.lock` with `O_EXCL` (pid, host and time inside)
and loads the document; leaving removes the lock. Changes reach the disk only
with `save`, as one atomic write. A second writer gets `LockError`, whichever
phase it runs. A lock left by a writer that stopped is never removed
automatically: the error names the file and its owner, and the file is
removed by hand once no writer is running. Readers call `read_system` at any
time.

## Atomic writes

`write_json` writes to `.<name>.<random>.tmp` in the same directory, flushes
it to disk and renames it over the target with `os.replace`; readers see the
old file or the new one, never a partial one. A crash leaves the old file and
at most an orphan temporary file. The directory is then synchronized, so that
the rename survives a power loss; on a filesystem that does not support it,
a warning is logged once per directory (atomicity is unaffected) and any
other error is raised.

Values must be JSON as they are: string keys, lists, tuples (read back as
lists), strings, numbers, booleans and null; numpy arrays and integers are
errors. Files are standard JSON: a non-finite float is written as a tagged
object and read back as the same float.

| Value | In the file |
|---|---|
| `nan` | `{"@float": "NaN"}` |
| `inf` | `{"@float": "Infinity"}` |
| `-inf` | `{"@float": "-Infinity"}` |
| missing value | missing key |
| `None` | `null` |

A dict whose only key is `"@float"` cannot be stored, and `read_json` refuses
bare `NaN`/`Infinity` tokens and unknown tags. Whether the rename and the
lock keep their guarantees on the parallel filesystems of each HPC machine is
checked there.

## Historic database calls

`legacy.py` gives the ported workflow code `query_db`, `update_db`,
`insert_db` and `is_done` with the meaning of the historic document-database
API, on top of `SystemDocument` and the records. It is the only module that
knows those calls, and it is not imported by the package.

```python
query_db(collection, *, entry=None, fltr=None)
update_db(data, collection, *, entry=None, fltr=None, update_op='$set', to_mongodb=True)
insert_db(field, collection, *, duplicates=False)
is_done(collection, *, entry=None, fltr=None)
```

The historic arguments without a counterpart in the store (`db_file`,
`database`, `upsert`, `message`) are not accepted, and every argument after
the collection is keyword-only: a call that passes them by position, as the
historic code does, raises `TypeError` instead of giving its values to other
parameters.

| Call | Accepted | Error |
|---|---|---|
| any | `fltr` matching the document of the system by equality of top-level fields (a subset of them, each with a value) | no `fltr`, a filter that does not match, a None value in a filter, keys with dots or starting with `$` |
| `query_db` | the system collection, or `tasks` with `{'task_label': calc_id}`; `entry` a key, a list of keys or a list of such lists | an incomplete record in `tasks` |
| `update_db` | `$set`; `to_mongodb=True` merges leaves, `False` replaces the top-level key of the entry; a list of entries with a list of as many values | no document yet, other operators, `tasks` |
| `insert_db` | one dict, creating the document; inserting again a dict whose fields all equal those of the existing document does nothing, so a repeated step does not fail | any other insertion, `duplicates=True` on an existing document |
| `is_done` | the historic condition (entry present and not empty) and the expected provenance given for that entry | no expected provenance for the entry, a list of entries |

These calls write the system document only: records, and the attempts they
keep, are never touched by them.

The `tasks` view returns, for a complete record whose output agrees with its
energy model, its `results` with
`task_label`, `task_id` and `dir_name`, all equal to the `calc_id`;
`dir_name` is the directory of the calculation relative to the system, true
also after the run is copied elsewhere. Files of a calculation (e.g. a charge
density) are found from the record, not from a host and an absolute path.
