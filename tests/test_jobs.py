# Author: Vito Foderà
# Created: 2026-10-27

"""Tests of the layout of a run and of the groups of jobs
(``triboworkflow.execution.layout``, ``triboworkflow.execution.jobs``).

Submissions run in dry run or with ``subprocess.run`` replaced: no scheduler
is called.
"""

import json
import shlex
import subprocess
import tomllib
from pathlib import Path

import pytest

from triboworkflow.core.guard import NotOnHPCError
from triboworkflow.core.settings import settings_from_dict
from triboworkflow.core import store
from triboworkflow.core.store import StoreError
from triboworkflow.execution import layout
from triboworkflow.execution import scheduler as scheduler_module
from triboworkflow.execution.jobs import (ANALYSIS_FILE, JOBSCRIPT_FILE, MANIFEST_FILE,
                                          PENDING_KEY, SUBMITTED_FILE, Analysis, JobError, Launch,
                                          Manifest, analysis_jobscript_path, analysis_line,
                                          is_analysis, jobscript_path, most_jobs, read_analysis,
                                          read_manifest, submit_jobs, surely_not_created,
                                          worker_line, write_analysis, write_job)
from triboworkflow.execution.scheduler import JobScheduler

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


def example_settings(**execution):
    """Settings of examples/interface.toml, with keys of [execution] changed."""
    with open(EXAMPLES / 'interface.toml', 'rb') as f:
        data = tomllib.load(f)
    data['execution'].update(execution)
    return settings_from_dict(data)


TEMPLATE = b'#!/bin/bash\n#SBATCH --job-name=pes\n#SBATCH --time=24:00:00\nmodule load qe\n'
CALCS = (('Cu111_Al111', 'pes_scf/aa'), ('Cu111_Al111', 'pes_scf/ab'), ('Cu111_Ni111', 'pes_scf/aa'))
LAUNCHES = tuple(Launch.initial(*calc) for calc in CALCS)


def manifest(**changes):
    fields = dict(name='pes_scf', scheduler='slurm', run_command='mpirun -np 32 pw.x -nk 4',
                  walltime=86400, max_seconds_margin=600, calculations=LAUNCHES)
    fields.update(changes)
    return Manifest(**fields)


@pytest.fixture
def run_dir(tmp_path):
    run = tmp_path / 'run'
    for system, calc_id in CALCS:
        path = layout.calc_dir(run, system, calc_id)
        path.mkdir(parents=True)
        (path / layout.INPUT_FILE).write_text('&CONTROL\n/\n')
    return run


@pytest.fixture
def jobscript(tmp_path):
    path = tmp_path / 'user_job.sh'
    path.write_bytes(TEMPLATE)
    return path


# --- layout ------------------------------------------------------------------------

def test_calc_dir_prefix_and_attempts(tmp_path):
    path = layout.calc_dir(tmp_path, 'Cu111_Al111', 'pes_scf/aa')
    assert path == tmp_path / 'systems' / 'Cu111_Al111' / 'pes_scf' / 'aa'
    assert layout.calc_prefix('pes_scf/aa') == 'pes_scf_aa'
    assert layout.attempt_dir(path, 2) == path / 'attempts' / '2'
    assert layout.job_dir(tmp_path, 'pes_scf') == tmp_path / 'jobs' / 'pes_scf'


@pytest.mark.parametrize('call', [lambda p: layout.calc_dir(p, 'a/b', 'pes_scf/aa'),
                                  lambda p: layout.calc_dir(p, 'sys', 'pes_scf'),
                                  lambda p: layout.job_dir(p, '..')])
def test_layout_rejects_bad_names(tmp_path, call):
    with pytest.raises(StoreError):
        call(tmp_path)


@pytest.mark.parametrize('n', [0, -1, 1.0, True])
def test_attempt_numbers_from_one(tmp_path, n):
    with pytest.raises(ValueError, match='attempt number'):
        layout.attempt_dir(tmp_path, n)


# --- manifest ----------------------------------------------------------------------

LEVEL_2 = {'mixing_beta': 0.2, 'nstep': 200}
INITIAL = {'mixing_beta': 0.4, 'nstep': 150}


def test_manifest_round_trip(tmp_path):
    remedy = Launch('Cu111_Ni111', 'pes_scf/ab', 3, 'remedy', 2, {'mixing_beta': 0.2},
                    {'mixing_beta': 0.4}, True)
    digest = 'sha256:' + 'c' * 64
    data = manifest(calculations=LAUNCHES + (remedy,), jobscript_digest=digest).to_dict()
    assert data['schema_version'] == 4 and data['jobscript_digest'] == digest
    assert data['pp_command'] is None
    assert list(data) == ['schema_version', 'name', 'scheduler', 'run_command', 'pp_command',
                          'jobscript_digest', 'walltime', 'max_seconds_margin', 'calculations']
    assert data['calculations'][0] == {'system': 'Cu111_Al111', 'calc_id': 'pes_scf/aa', 'n': 1,
                                       'kind': 'initial', 'level': 1, 'values': {},
                                       'initial_values': {}, 'restart': False}
    assert data['calculations'][3] == {'system': 'Cu111_Ni111', 'calc_id': 'pes_scf/ab', 'n': 3,
                                       'kind': 'remedy', 'level': 2,
                                       'values': {'mixing_beta': 0.2},
                                       'initial_values': {'mixing_beta': 0.4}, 'restart': True}
    assert Manifest.from_dict(json.loads(json.dumps(data))) == \
        manifest(calculations=LAUNCHES + (remedy,), jobscript_digest=digest)
    with_pp = manifest(pp_command='srun pp.x')
    assert Manifest.from_dict(json.loads(json.dumps(with_pp.to_dict()))) == with_pp


def test_manifest_from_a_generator():
    assert manifest(calculations=(c for c in LAUNCHES)) == manifest()


def test_manifest_from_the_settings():
    settings = example_settings(max_seconds_margin=900)
    built = Manifest.from_execution('pes_scf', LAUNCHES, settings.execution)
    assert built == manifest(max_seconds_margin=900)
    settings = example_settings(pp_command='mpirun -np 32 pp.x')
    built = Manifest.from_execution('pes_scf', LAUNCHES, settings.execution)
    assert built == manifest(pp_command='mpirun -np 32 pp.x')


def give_density(run_dir, system, calc_id, pp_input=True):
    """Record of a calculation that gives the charge density, and its input.ppi."""
    step, label = calc_id.split('/')
    store.write_record(layout.system_dir(run_dir, system), store.Record(
        system=system, step=step, label=label, calc_type='slab_from_scratch',
        engine=store.Engine('qe', 'pw.x'), energy_model={'ecutwfc': 40.0},
        fingerprint='sha256:' + 'a' * 64, density=True))
    if pp_input:
        path = layout.calc_dir(run_dir, system, calc_id)
        (path / layout.PP_INPUT_FILE).write_text('&INPUTPP\n/\n')


def test_group_with_charge_density_needs_pp_command(run_dir, jobscript):
    give_density(run_dir, *CALCS[1])
    with pytest.raises(JobError, match='calculations that give the charge density, and no '
                                       r'pp_command in \[execution\] to run pp.x: '
                                       'Cu111_Al111/pes_scf/ab$'):
        write_job(run_dir, manifest(), jobscript)
    assert not layout.job_dir(run_dir, 'pes_scf').exists()
    path = write_job(run_dir, manifest(pp_command='srun pp.x'), jobscript)
    assert read_manifest(path / MANIFEST_FILE)[0].pp_command == 'srun pp.x'


def test_group_with_charge_density_needs_pp_input(run_dir, jobscript):
    give_density(run_dir, *CALCS[2], pp_input=False)
    with pytest.raises(JobError, match='calculations that give the charge density without '
                                       'input.ppi in .*: Cu111_Ni111/pes_scf/aa$'):
        write_job(run_dir, manifest(pp_command='srun pp.x'), jobscript)


@pytest.mark.parametrize('changes, message', [
    ({'name': 'a/b'}, 'not a valid name'),
    ({'run_command': ' '}, 'run_command: a non-empty string is required'),
    ({'walltime': 0}, 'walltime: 0 is not an integer > 0'),
    ({'walltime': 3600.0}, 'walltime: 3600.0 is not an integer > 0'),
    ({'max_seconds_margin': 86400}, 'must be < walltime (86400)'),
    ({'jobscript_digest': 'abc'}, 'jobscript_digest: \'abc\' is not "sha256:"'),
    ({'calculations': ()}, 'the group has no calculations'),
    ({'calculations': ('ab',)}, 'is not a sequence of Launch'),
    ({'calculations': CALCS}, 'is not a sequence of Launch'),
    ({'calculations': LAUNCHES + (LAUNCHES[0],)}, 'Cu111_Al111/pes_scf/aa is listed twice'),
    ({'calculations': LAUNCHES + (Launch('Cu111_Al111', 'pes_scf/aa', 2, 'unchanged', 1),)},
     'Cu111_Al111/pes_scf/aa is listed twice'),
    ({'pp_command': ' '}, 'pp_command: a non-empty string or null is required'),
    ({'pp_command': 1}, 'pp_command: a non-empty string or null is required'),
])
def test_manifest_errors(changes, message):
    with pytest.raises(JobError, match=message.replace('(', r'\(').replace(')', r'\)')):
        manifest(**changes)


def test_launch():
    launch = Launch('Cu111_Al111', 'pes_scf/aa', 2, 'unchanged', 1)
    assert launch.calc == ('Cu111_Al111', 'pes_scf/aa')
    assert launch.remedies() == {'kind': 'unchanged', 'level': 1, 'values': {},
                                 'initial_values': {}, 'restart': False}
    assert Launch.initial('Cu111_Al111', 'pes_scf/aa') == \
        Launch('Cu111_Al111', 'pes_scf/aa', 1, 'initial', 1, {}, {}, False)
    values = dict(LEVEL_2)
    remedy = Launch('Cu111_Al111', 'pes_scf/aa', 3, 'remedy', 2, values, INITIAL, True)
    values['nstep'] = 1
    assert remedy.values == LEVEL_2
    assert remedy.remedies() == {'kind': 'remedy', 'level': 2, 'values': LEVEL_2,
                                 'initial_values': INITIAL, 'restart': True}
    assert Launch.from_dict(remedy.to_dict()) == remedy
    assert remedy.to_dict()['restart'] is True
    # Unchanged and manual launches above level 1 repeat the values of the level.
    for kind in ('unchanged', 'manual'):
        assert Launch('Cu111_Al111', 'pes_scf/aa', 4, kind, 2, LEVEL_2, INITIAL).values == LEVEL_2


@pytest.mark.parametrize('fields, message', [
    (('a/b', 'pes_scf/aa', 1, 'initial', 1), 'not a valid name'),
    (('sys', 'pes_scf', 1, 'initial', 1), 'is not "<step>/<label>"'),
    (('sys', 'pes_scf/aa', 0, 'initial', 1), 'n = 0 is not an integer >= 1'),
    (('sys', 'pes_scf/aa', 2.0, 'unchanged', 1), 'n = 2.0 is not an integer >= 1'),
    (('sys', 'pes_scf/aa', 2, 'unchanged', True), 'level = True is not an integer >= 1'),
    (('sys', 'pes_scf/aa', 2, 'restart', 1), "kind = 'restart' is not one of initial, "
                                              "unchanged, remedy, manual"),
    (('sys', 'pes_scf/aa', 2, 'initial', 1), 'an initial launch is attempt 1 at level 1, not '
                                             'attempt 2 at level 1'),
    (('sys', 'pes_scf/aa', 1, 'initial', 2), 'an initial launch is attempt 1 at level 1'),
    (('sys', 'pes_scf/aa', 1, 'manual', 1), "attempt 1 is an initial launch, not 'manual'"),
    (('sys', 'pes_scf/aa', 2, 'remedy', 1, {'nstep': 200}), 'a remedy runs at level 2 or more'),
    (('sys', 'pes_scf/aa', 2, 'remedy', 2), 'values is empty at level 2'),
    (('sys', 'pes_scf/aa', 2, 'remedy', 2, LEVEL_2), 'initial_values is empty at level 2'),
    (('sys', 'pes_scf/aa', 2, 'unchanged', 3, {}, INITIAL), 'values is empty at level 3'),
    (('sys', 'pes_scf/aa', 2, 'manual', 1, {'nstep': 200}), "at level 1 the input runs as "
                                                            "prepared, values = {'nstep': 200}"),
    (('sys', 'pes_scf/aa', 2, 'unchanged', 1, {}, INITIAL), 'at level 1 the input runs as '
                                                            'prepared, initial_values'),
    (('sys', 'pes_scf/aa', 2, 'remedy', 2, [('nstep', 200)]), 'values must be a JSON object'),
    (('sys', 'pes_scf/aa', 2, 'remedy', 2, {1: 200}), 'values: key 1 is not a string'),
    (('sys', 'pes_scf/aa', 2, 'remedy', 2, LEVEL_2, None), 'initial_values must be a JSON '
                                                           'object'),
    (('sys', 'pes_scf/aa', 2, 'unchanged', 1, {}, {}, 1), 'restart = 1 is not true or false'),
    (('sys', 'pes_scf/aa', 1, 'initial', 1, {}, {}, True), 'attempt 1 has no previous attempt '
                                                           'to continue'),
])
def test_launch_errors(fields, message):
    with pytest.raises(JobError, match=message.replace('(', r'\(').replace(')', r'\)')):
        Launch(*fields)


@pytest.mark.parametrize('edit, message', [
    ({'schema_version': 3}, 'schema_version: 3; this version reads 4'),
    ({'pp_command': ''}, 'pp_command: a non-empty string or null is required'),
    ({'extra': 1}, "unknown 'extra'"),
    ({'calculations': {}}, 'calculations: a JSON array is required'),
    ({'calculations': [{'system': 'a'}]}, 'calculations: missing calc_id, n, kind, level, values, '
                                          'initial_values, restart'),
    ({'calculations': [{'system': 'a', 'calc_id': 's/l'}]}, 'calculations: missing n'),
    ({'calculations': [{**LAUNCHES[0].to_dict(), 'level': 0}]}, 'level = 0'),
])
def test_manifest_file_errors(edit, message):
    data = {**manifest().to_dict(), **edit}
    with pytest.raises(JobError, match=f'^m.json: .*{message}'):
        Manifest.from_dict(data, source='m.json')


# --- jobscript and group directory -----------------------------------------------------

def test_write_job(run_dir, jobscript):
    path = write_job(run_dir, manifest(), jobscript, python='/opt/venv/bin/python')
    assert path == run_dir / 'jobs' / 'pes_scf'
    assert jobscript.read_bytes() == TEMPLATE
    text = (path / JOBSCRIPT_FILE).read_bytes()
    assert text.startswith(TEMPLATE)
    added = text[len(TEMPLATE):].decode()
    assert added == f'\n{worker_line(path / MANIFEST_FILE, "/opt/venv/bin/python")}\n'
    words = shlex.split(added)
    assert words[:3] == ['/opt/venv/bin/python', '-m', 'triboworkflow.execution.worker']
    assert Path(words[3]) == (path / MANIFEST_FILE).resolve()
    read, run = read_manifest(path / MANIFEST_FILE)
    # The manifest holds the digest of the jobscript of the user, as copied.
    assert read == manifest(jobscript_digest=store.bytes_digest(TEMPLATE))
    assert run == run_dir.resolve()


def test_worker_line_quotes_the_paths(tmp_path):
    line = worker_line(tmp_path / 'my jobs' / MANIFEST_FILE, python='/opt/my venv/python')
    assert shlex.split(line)[0] == '/opt/my venv/python'
    assert shlex.split(line)[3].endswith('my jobs/manifest.json')


def test_jobscript_without_final_newline(run_dir, tmp_path):
    template = tmp_path / 'job.sh'
    template.write_bytes(b'#!/bin/bash\nmodule load qe')
    path = write_job(run_dir, manifest(), template, python='python3')
    lines = (path / JOBSCRIPT_FILE).read_text().splitlines()
    assert lines[:2] == ['#!/bin/bash', 'module load qe']
    assert lines[2].startswith('python3 -m triboworkflow.execution.worker ')


def test_write_job_errors(run_dir, jobscript, tmp_path):
    with pytest.raises(JobError, match='jobscript .* not found'):
        write_job(run_dir, manifest(), tmp_path / 'missing.sh')
    (layout.calc_dir(run_dir, *CALCS[1]) / layout.INPUT_FILE).unlink()
    with pytest.raises(JobError, match='calculations without input.pwi .*Cu111_Al111/pes_scf/ab'):
        write_job(run_dir, manifest(), jobscript)
    assert not (run_dir / 'jobs').exists()
    write_job(run_dir, manifest(calculations=LAUNCHES[:1]), jobscript)
    with pytest.raises(JobError, match='already exists'):
        write_job(run_dir, manifest(calculations=LAUNCHES[:1]), jobscript)


def test_read_manifest_checks_the_directory(run_dir, jobscript):
    path = write_job(run_dir, manifest(), jobscript)
    moved = path.parent / 'other'
    path.rename(moved)
    with pytest.raises(JobError, match="manifest of the group 'pes_scf' found in"):
        read_manifest(moved / MANIFEST_FILE)


def test_jobscript_relative_to_the_settings_file():
    settings = example_settings()
    assert jobscript_path(settings) == Path('jobscript.sh')
    settings.source = EXAMPLES / 'interface.toml'
    assert jobscript_path(settings) == EXAMPLES / 'jobscript.sh'
    settings.execution.jobscript = '/abs/job.sh'
    assert jobscript_path(settings) == Path('/abs/job.sh')


# --- submission ----------------------------------------------------------------------

def test_dry_run_submits_nothing(run_dir, jobscript, monkeypatch):
    monkeypatch.setattr(scheduler_module.subprocess, 'run',
                        lambda *a, **k: pytest.fail('no command may run in a dry run'))
    path = write_job(run_dir, manifest(), jobscript)
    scheduler = JobScheduler('slurm', dry_run=True)
    assert submit_jobs(path, scheduler, 3) == [None, None, None]
    assert scheduler.commands == [(['sbatch', str(path / JOBSCRIPT_FILE)], None, path)] * 3
    assert not (path / SUBMITTED_FILE).exists()


def test_submitted_ids_are_written(run_dir, jobscript, monkeypatch):
    ids = iter(['101', '102'])

    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=f'Submitted batch job {next(ids)}\n')

    monkeypatch.setattr(scheduler_module.subprocess, 'run', run)
    path = write_job(run_dir, manifest(), jobscript)
    assert submit_jobs(path, JobScheduler('slurm'), 2) == ['101', '102']
    assert json.loads((path / SUBMITTED_FILE).read_text()) == {'job_ids': ['101', '102']}


def test_failed_submission_keeps_the_jobs_already_submitted(run_dir, jobscript, monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(1, cmd, stderr='QOSMaxSubmitJobPerUserLimit')
        return subprocess.CompletedProcess(cmd, 0, stdout='Submitted batch job 201\n')

    monkeypatch.setattr(scheduler_module.subprocess, 'run', run)
    path = write_job(run_dir, manifest(), jobscript)
    with pytest.raises(scheduler_module.JobSchedulerError, match='QOSMaxSubmitJobPerUserLimit'):
        submit_jobs(path, JobScheduler('slurm'), 3)
    assert json.loads((path / SUBMITTED_FILE).read_text()) == {'job_ids': ['201']}


def test_pending_is_written_before_each_submission(run_dir, jobscript, monkeypatch):
    path = write_job(run_dir, manifest(), jobscript)
    seen, ids = [], iter(['301', '302'])

    def run(cmd, **kwargs):
        seen.append(json.loads((path / SUBMITTED_FILE).read_text()))
        return subprocess.CompletedProcess(cmd, 0, stdout=f'Submitted batch job {next(ids)}\n')

    monkeypatch.setattr(scheduler_module.subprocess, 'run', run)
    stamps = iter(['2026-11-02T10:00:00+01:00', '2026-11-02T10:00:05+01:00'])
    assert submit_jobs(path, JobScheduler('slurm'), 2, now=lambda: next(stamps)) == ['301', '302']
    assert seen == [{'job_ids': [], PENDING_KEY: '2026-11-02T10:00:00+01:00'},
                    {'job_ids': ['301'], PENDING_KEY: '2026-11-02T10:00:05+01:00'}]
    assert json.loads((path / SUBMITTED_FILE).read_text()) == {'job_ids': ['301', '302']}


def test_interrupted_submission_leaves_pending(run_dir, jobscript, monkeypatch):
    path = write_job(run_dir, manifest(), jobscript)

    def run(cmd, **kwargs):
        raise KeyboardInterrupt  # the process stops while sbatch runs

    monkeypatch.setattr(scheduler_module.subprocess, 'run', run)
    with pytest.raises(KeyboardInterrupt):
        submit_jobs(path, JobScheduler('slurm'), 1, now=lambda: '2026-11-02T10:00:00+01:00')
    assert json.loads((path / SUBMITTED_FILE).read_text()) == \
        {'job_ids': [], PENDING_KEY: '2026-11-02T10:00:00+01:00'}
    with pytest.raises(JobError, match='has no outcome recorded'):
        submit_jobs(path, JobScheduler('slurm'), 1)


def test_submission_of_outcome_not_known_leaves_pending(run_dir, jobscript, monkeypatch):
    def run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 30)

    monkeypatch.setattr(scheduler_module.subprocess, 'run', run)
    path = write_job(run_dir, manifest(), jobscript)
    with pytest.raises(scheduler_module.JobSchedulerError) as error:
        submit_jobs(path, JobScheduler('slurm'), 1, now=lambda: '2026-11-02T10:00:00+01:00')
    assert error.value.accepted is None and not surely_not_created(error.value)
    assert json.loads((path / SUBMITTED_FILE).read_text())[PENDING_KEY] == \
        '2026-11-02T10:00:00+01:00'


def test_refused_submission_clears_pending(run_dir, jobscript, monkeypatch):
    def run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr='invalid account')

    monkeypatch.setattr(scheduler_module.subprocess, 'run', run)
    path = write_job(run_dir, manifest(), jobscript)
    with pytest.raises(scheduler_module.JobSchedulerError) as error:
        submit_jobs(path, JobScheduler('slurm'), 1)
    assert surely_not_created(error.value)
    assert not (path / SUBMITTED_FILE).exists()  # no job was ever submitted
    (path / SUBMITTED_FILE).write_text('{"job_ids": []}')
    with pytest.raises(scheduler_module.JobSchedulerError):
        submit_jobs(path, JobScheduler('slurm'), 1)
    assert json.loads((path / SUBMITTED_FILE).read_text()) == {'job_ids': []}


def test_submission_away_from_hpc(run_dir, jobscript, monkeypatch):
    def run(cmd, **kwargs):
        raise FileNotFoundError(2, 'No such file or directory', cmd[0])

    monkeypatch.setattr(scheduler_module.subprocess, 'run', run)
    path = write_job(run_dir, manifest(), jobscript)
    with pytest.raises(NotOnHPCError, match="Command 'sbatch' not found: calculations run only"):
        submit_jobs(path, JobScheduler('slurm'), 1)
    assert not (path / SUBMITTED_FILE).exists()


@pytest.mark.parametrize('count, submitted', [(1, 1), (3, 3), (8, 3)])
def test_never_more_jobs_than_calculations(run_dir, jobscript, count, submitted, caplog):
    path = write_job(run_dir, manifest(), jobscript)
    scheduler = JobScheduler('slurm', dry_run=True)
    assert submit_jobs(path, scheduler, count) == [None] * submitted
    assert len(scheduler.commands) == submitted
    reduced = 'Group pes_scf: 3 jobs submitted instead of 8, one per calculation'
    assert (reduced in caplog.text) == (count > submitted)


@pytest.mark.parametrize('count', [0, 2.0])
def test_number_of_jobs(run_dir, jobscript, count):
    path = write_job(run_dir, manifest(), jobscript)
    with pytest.raises(JobError, match='number of jobs'):
        submit_jobs(path, JobScheduler('slurm', dry_run=True), count)


# --- groups of an analysis -------------------------------------------------------------

ANALYSIS_TEMPLATE = b'#!/bin/bash\n#SBATCH --mem=200G\n#SBATCH --time=04:00:00\n'


def analysis(**changes):
    fields = dict(name='Cu111_Al111.plotpes.1', system='Cu111_Al111', step='plotpes',
                  scheduler='slurm')
    fields.update(changes)
    return Analysis(**fields)


@pytest.fixture
def analysis_jobscript(tmp_path):
    path = tmp_path / 'analysis.sh'
    path.write_bytes(ANALYSIS_TEMPLATE)
    return path


def test_write_analysis(tmp_path, analysis_jobscript):
    run = tmp_path / 'run'
    path = write_analysis(run, analysis(), analysis_jobscript, python='/opt/venv/bin/python')
    assert path == run / 'jobs' / 'Cu111_Al111.plotpes.1' and is_analysis(path)
    assert not (path / MANIFEST_FILE).exists()
    assert analysis_jobscript.read_bytes() == ANALYSIS_TEMPLATE
    text = (path / JOBSCRIPT_FILE).read_bytes()
    assert text.startswith(ANALYSIS_TEMPLATE)
    words = shlex.split(text[len(ANALYSIS_TEMPLATE):].decode())
    assert words[:3] == ['/opt/venv/bin/python', '-m', 'triboworkflow.cli.analysis_job']
    assert words[:3] == shlex.split(analysis_line(path / ANALYSIS_FILE, '/opt/venv/bin/python'))[:3]
    assert Path(words[3]) == (path / ANALYSIS_FILE).resolve() and len(words) == 4
    read, run_found = read_analysis(path / ANALYSIS_FILE)
    # The digest is the one of the jobscript of [execution.analysis], as copied.
    assert read == analysis(jobscript_digest=store.bytes_digest(ANALYSIS_TEMPLATE))
    assert read.number == 1 and run_found == run.resolve()
    assert json.loads((path / ANALYSIS_FILE).read_text())['schema_version'] == 1


def test_write_analysis_errors(tmp_path, analysis_jobscript):
    run = tmp_path / 'run'
    with pytest.raises(JobError, match=r'jobscript .*missing.sh of \[execution.analysis\] not '
                                       'found'):
        write_analysis(run, analysis(), tmp_path / 'missing.sh')
    assert not (run / 'jobs').exists()
    write_analysis(run, analysis(), analysis_jobscript)
    with pytest.raises(JobError, match='already exists'):
        write_analysis(run, analysis(), analysis_jobscript)


@pytest.mark.parametrize('changes, message', [
    ({'name': 'Cu111_Al111.pes.1'}, "name: 'Cu111_Al111.pes.1' is not Cu111_Al111.plotpes.<k>"),
    ({'name': 'Cu111_Al111.plotpes.0'}, 'k >= 1'),
    ({'name': 'Cu111_Al111.plotpes.a'}, 'k >= 1'),
    ({'name': 'Cu111_Al111.plotpes.'}, 'k >= 1'),
    ({'system': 'a/b'}, 'system'),
    ({'scheduler': ' '}, 'scheduler: a non-empty string is required'),
    ({'jobscript_digest': 'md5:1'}, 'jobscript_digest'),
])
def test_analysis_errors(changes, message):
    with pytest.raises(JobError, match=message):
        analysis(**changes)


@pytest.mark.parametrize('edit, message', [
    (lambda d: d.update(schema_version=2), 'schema_version: 2; this version reads 1'),
    (lambda d: d.update(memory='200G'), "unknown 'memory'"),
    (lambda d: d.pop('step'), 'missing step'),
])
def test_analysis_file_errors(tmp_path, analysis_jobscript, edit, message):
    path = write_analysis(tmp_path / 'run', analysis(), analysis_jobscript) / ANALYSIS_FILE
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))
    with pytest.raises(JobError, match=f'{path.name}: .*{message}'):
        read_analysis(path)


def test_read_analysis_checks_the_directory(tmp_path, analysis_jobscript):
    path = write_analysis(tmp_path / 'run', analysis(), analysis_jobscript)
    moved = path.parent / 'Cu111_Al111.plotpes.2'
    path.rename(moved)
    with pytest.raises(JobError, match="analysis of the group 'Cu111_Al111.plotpes.1' found in"):
        read_analysis(moved / ANALYSIS_FILE)


def test_analysis_jobscript_relative_to_the_settings_file():
    settings = example_settings()
    assert analysis_jobscript_path(settings) is None  # no [execution.analysis]
    settings = example_settings(analysis={'jobscript': 'analysis.sh'})
    assert analysis_jobscript_path(settings) == Path('analysis.sh')
    settings.source = EXAMPLES / 'interface.toml'
    assert analysis_jobscript_path(settings) == EXAMPLES / 'analysis.sh'
    settings.execution.analysis.jobscript = '/abs/analysis.sh'
    assert analysis_jobscript_path(settings) == Path('/abs/analysis.sh')


@pytest.mark.parametrize('count', [1, 4])
def test_an_analysis_runs_one_job(tmp_path, analysis_jobscript, count, caplog):
    path = write_analysis(tmp_path / 'run', analysis(), analysis_jobscript)
    assert most_jobs(path) == 1
    scheduler = JobScheduler('slurm', dry_run=True)
    assert submit_jobs(path, scheduler, count) == [None]
    assert scheduler.commands == [(['sbatch', str(path / JOBSCRIPT_FILE)], None, path)]
    reduced = 'Group Cu111_Al111.plotpes.1: 1 jobs submitted instead of 4, one for an analysis'
    assert (reduced in caplog.text) == (count > 1)


def test_most_jobs(run_dir, jobscript, tmp_path):
    assert most_jobs(write_job(run_dir, manifest(), jobscript)) == 3
    with pytest.raises(JobError, match='manifest.json not found: write the group first'):
        most_jobs(tmp_path / 'empty')
