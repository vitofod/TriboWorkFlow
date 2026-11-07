# Author: Vito Foderà
# Created: 2026-10-30

"""Tests of the command line (``triboworkflow.cli``).

The commands run in the process of the test, through ``main``. No scheduler
command runs: the steps are fictitious (``cli_helpers.fake_steps``), the
scheduler is the simulated Slurm of ``run_helpers`` (``fake_slurm``), and
the commands that need the scheduler find fictitious executables first in
the ``PATH`` (``fake_scheduler_bin``), which only record their calls.
"""

import importlib
import json
import os
import tomllib

import pytest

import triboworkflow
from cli_helpers import (ROOT, SYSTEM, complete, fake_scheduler_bin,  # noqa: F401 (fixtures)
                         fake_slurm, fake_steps, write_run)
from triboworkflow.cli.main import main
from triboworkflow.cli.parser import COMMANDS
from triboworkflow.cli.run import build_orchestrator
from triboworkflow.execution import layout
from triboworkflow.execution.lock import RunLock


def files(run_dir):
    """Every file of a run, with its content."""
    return {str(p.relative_to(run_dir)): p.read_bytes() for p in sorted(run_dir.rglob('*'))
            if p.is_file()}


def first_pass(run_dir, groups=2):
    """One pass of the orchestrator of the run (fictitious steps, simulated Slurm)."""
    orch = build_orchestrator(run_dir, groups=groups)
    orch.run_pass()
    return orch


# --- structure -----------------------------------------------------------------------------

@pytest.mark.parametrize('name', [name for name, _ in COMMANDS])
def test_help_of_every_command(name, capsys):
    module = dict(COMMANDS)[name]
    first = importlib.import_module(module).CLICommand.__doc__.split('\n')[0]
    with pytest.raises(SystemExit) as stop:
        main([name, '--help'])
    out = capsys.readouterr().out
    assert stop.value.code == 0 and first in out and 'Example' in out and '--run' in out


def test_no_command_prints_the_help_and_exits_2(capsys):
    assert main([]) == 2
    assert 'usage: triboworkflow' in capsys.readouterr().err
    assert main(['-T']) == 2


def test_unknown_command_exits_2(capsys):
    with pytest.raises(SystemExit) as stop:
        main(['nonsense'])
    assert stop.value.code == 2 and 'invalid choice' in capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit) as stop:
        main(['-v'])
    assert stop.value.code == 0
    assert capsys.readouterr().out == f'triboworkflow-{triboworkflow.__version__}\n'


def test_error_exits_1_and_traceback_with_T(tmp_path, capsys):
    assert main(['status', '-r', str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert err.startswith('Error: ') and 'config.toml not found' in err
    with pytest.raises(RuntimeError, match='config.toml not found'):
        main(['-T', 'status', '-r', str(tmp_path)])


def test_entry_point_declared():
    with open(ROOT / 'pyproject.toml', 'rb') as f:
        scripts = tomllib.load(f)['project']['scripts']
    assert scripts == {'triboworkflow': 'triboworkflow.cli.main:main'}


# --- check ---------------------------------------------------------------------------------

def test_check_example(tmp_path, capsys):
    steps = ('match', 'pes', 'plotpes', 'shear', 'adhesion', 'ppes', 'charge')
    run_dir = write_run(tmp_path, steps=steps)
    assert main(['check', '-r', str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert SYSTEM in out and 'Steps: ' + ', '.join(steps) in out
    assert 'groups: 10' in out and 'Placeholders: none' in out


def test_check_without_groups(tmp_path, capsys):
    assert main(['check', '-r', str(write_run(tmp_path, groups=None))]) == 0
    assert 'groups: not set' in capsys.readouterr().out


def test_check_invalid_settings(tmp_path, capsys):
    run_dir = write_run(tmp_path)
    config = run_dir / 'config.toml'
    config.write_text(config.read_text().replace('steps = ["match", "pes"]', ''))
    assert main(['check', '-r', str(run_dir)]) == 1
    assert 'interface.steps' in capsys.readouterr().err


def test_run_option_from_another_directory(tmp_path, monkeypatch, capsys):
    run_dir = write_run(tmp_path)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert main(['check']) == 1  # no config.toml here
    assert main(['check', '--run', str(run_dir)]) == 0
    monkeypatch.chdir(run_dir)
    assert main(['check']) == 0


# --- status --------------------------------------------------------------------------------

def test_status_of_a_new_run(tmp_path, capsys):
    run_dir = write_run(tmp_path)
    before = files(run_dir)
    assert main(['status', '-r', str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert f'system {SYSTEM}' in out and 'Orchestrator: not running: no ' in out
    assert 'Lock: free' in out and 'state saved never' in out
    assert 'match' in out and 'waiting' in out
    assert files(run_dir) == before


def test_status_runs_no_scheduler_command(tmp_path, fake_steps, fake_slurm, fake_scheduler_bin,
                                          capsys):
    run_dir = write_run(tmp_path)
    first_pass(run_dir)
    before = files(run_dir)
    assert main(['status', '--calcs', '-r', str(run_dir)]) == 0
    assert not fake_scheduler_bin.exists() and files(run_dir) == before
    out = capsys.readouterr().out
    assert 'pes/c0' in out and 'pes/c1' in out  # with --calcs


def test_status_shows_when_the_state_was_saved(tmp_path, fake_steps, fake_slurm, capsys):
    run_dir = write_run(tmp_path)
    first_pass(run_dir)
    orch = first_pass(run_dir)  # the jobs submitted by the first pass are asked for
    saved = max(entry['updated'] for entry in orch.state['systems'][SYSTEM].values())
    checked = max(job['checked'] for job in orch.manager.jobs.values())
    assert main(['status', '-r', str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert f'state saved {saved}' in out and f'last query {checked}' in out
    assert f'{orch.entry(SYSTEM, "pes")["updated"]}' in out


def test_status_lists_the_calculations_that_need_the_user(tmp_path, fake_steps, fake_slurm,
                                                          capsys):
    run_dir = write_run(tmp_path)
    first_pass(run_dir)
    (layout.calc_dir(run_dir, SYSTEM, 'pes/c0') / layout.CLAIM_FILE).write_text('')
    first_pass(run_dir)
    assert main(['status', '-r', str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert 'Needs the user' in out and 'pes/c0' in out and 'uncertain' in out
    assert 'empty claim' in out and 'release-claim pes/c0 --worker-over' in out


def test_status_names_the_holder_of_the_lock(tmp_path, capsys):
    run_dir = write_run(tmp_path)
    with RunLock(run_dir, 'triboworkflow submit pes'):
        assert main(['status', '-r', str(run_dir)]) == 0
    assert 'Lock: held by triboworkflow submit pes (pid ' in capsys.readouterr().out


# --- phases and chain ------------------------------------------------------------------------

def cli(run_dir, *argv):
    return main([*argv, '-r', str(run_dir)])


def test_phases_from_the_command_line(tmp_path, fake_steps, fake_slurm, fake_scheduler_bin,
                                      capsys):
    run_dir = write_run(tmp_path)
    assert cli(run_dir, 'analyse', 'match') == 0
    assert cli(run_dir, 'prepare', 'pes') == 0
    out = capsys.readouterr().out
    assert f'{SYSTEM}/pes: running' in out and '2 calculations prepared' in out
    assert fake_slurm.submitted == 0
    assert cli(run_dir, 'submit', 'pes') == 0
    out = capsys.readouterr().out
    assert fake_slurm.submitted == 2 and f'{SYSTEM}.pes.1: 2 jobs' in out
    assert 'again' in out  # repeating submit follows the step
    for calc_id in ('pes/c0', 'pes/c1'):
        complete(run_dir, calc_id)
    assert cli(run_dir, 'analyse', 'pes') == 0
    assert f'{SYSTEM}/pes: completed' in capsys.readouterr().out
    assert not fake_scheduler_bin.exists()
    log = (run_dir / 'cli.log').read_text()
    assert 'triboworkflow prepare pes' in log and 'triboworkflow analyse pes' in log


def test_prepare_blocked_step_exits_1(tmp_path, fake_steps, capsys):
    run_dir = write_run(tmp_path)
    assert cli(run_dir, 'prepare', 'pes') == 1
    out = capsys.readouterr().out
    assert f'{SYSTEM}/pes: blocked' in out and 'entries missing or not current: match' in out


def test_analyse_refusal_lists_the_missing_evidence(tmp_path, fake_steps, fake_slurm,
                                                    fake_scheduler_bin, capsys):
    run_dir = write_run(tmp_path)
    for argv in (('analyse', 'match'), ('prepare', 'pes'), ('submit', 'pes')):
        assert cli(run_dir, *argv) == 0
    complete(run_dir, 'pes/c1')
    capsys.readouterr()
    assert cli(run_dir, 'analyse', 'pes') == 1
    err = capsys.readouterr().err
    assert f'pes/c0: attempt 1 listed in group {SYSTEM}.pes.1, not taken yet' in err
    assert 'pes/c1' not in err


@pytest.mark.parametrize('argv', [['submit', 'pes'], ['start'], ['relaunch', 'pes/aa']])
def test_hpc_commands_write_nothing_away_from_hpc(tmp_path, monkeypatch, capsys, argv):
    run_dir = write_run(tmp_path)
    before = files(run_dir)
    monkeypatch.setenv('PATH', str(tmp_path / 'nothing'))
    assert cli(run_dir, *argv) == 1
    assert "Command 'sbatch' not found" in capsys.readouterr().err
    assert files(run_dir) == before


@pytest.mark.parametrize('case', ['usage', 'guard', 'lock'])
def test_refused_command_writes_nothing(tmp_path, monkeypatch, case):
    run_dir = write_run(tmp_path)
    if case == 'usage':
        before = files(run_dir)
        with pytest.raises(SystemExit) as stop:
            cli(run_dir, 'prepare')
        assert stop.value.code == 2
    elif case == 'guard':
        monkeypatch.setenv('PATH', str(tmp_path / 'nothing'))
        before = files(run_dir)
        assert cli(run_dir, 'submit', 'pes') == 1
    else:
        with RunLock(run_dir, 'triboworkflow submit pes'):
            before = files(run_dir)
            assert cli(run_dir, 'prepare', 'pes') == 1
            assert files(run_dir) == before
        before.pop('run.lock')
    assert files(run_dir) == before and not (run_dir / 'cli.log').exists()


def test_submission_refused_exits_1(tmp_path, fake_steps, fake_slurm, fake_scheduler_bin,
                                    capsys):
    run_dir = write_run(tmp_path)
    assert cli(run_dir, 'analyse', 'match') == 0 and cli(run_dir, 'prepare', 'pes') == 0
    fake_slurm.refuse = 1
    capsys.readouterr()
    assert cli(run_dir, 'submit', 'pes') == 1
    out = capsys.readouterr().out
    assert f'group {GROUP}: submission refused by the scheduler' in out
    assert f'jobs/{GROUP}/jobscript.sh' in out and f'clear-refusal {GROUP}' in out


def test_submit_with_groups_from_the_command_line(tmp_path, fake_steps, fake_slurm,
                                                  fake_scheduler_bin, capsys):
    run_dir = write_run(tmp_path, groups=None)
    assert cli(run_dir, 'analyse', 'match') == 0 and cli(run_dir, 'prepare', 'pes') == 0
    assert cli(run_dir, 'submit', 'pes') == 1  # no terminal to ask
    assert 'groups not given' in capsys.readouterr().err and fake_slurm.submitted == 0
    assert cli(run_dir, 'submit', 'pes', '--groups', '1') == 0
    assert fake_slurm.submitted == 1
    assert 'groups: 1, from --groups' in (run_dir / 'cli.log').read_text()


def test_start_passes_the_factory_and_groups(tmp_path, monkeypatch, fake_scheduler_bin, capsys):
    from triboworkflow.execution import daemon
    run_dir = write_run(tmp_path, groups=None)
    calls = []

    def fake_start(run, factory, scheduler, factory_args=()):
        calls.append((run, factory, scheduler, factory_args))
        return daemon.DaemonStatus(True, False, 4242, 'login01', 'now', 'cmd',
                                   'running: pid 4242 on login01 since now')

    monkeypatch.setattr(daemon, 'start', fake_start)
    assert cli(run_dir, 'start', '--groups', '3') == 0
    assert calls == [(run_dir, 'triboworkflow.cli.run:daemon_factory', 'slurm', ('3',))]
    assert 'running: pid 4242 on login01 since now' in capsys.readouterr().out
    assert 'running: pid 4242' in (run_dir / 'cli.log').read_text()


@pytest.mark.skipif(os.name != 'posix', reason='the orchestrator in the background is POSIX only')
def test_start_reports_a_step_not_available(tmp_path, fake_scheduler_bin, capsys):
    run_dir = write_run(tmp_path, steps=('match', 'pes', 'plotpes'))
    assert cli(run_dir, 'start') == 1
    assert 'orchestrator.log' in capsys.readouterr().err
    assert "step 'plotpes' not available" in (run_dir / 'orchestrator.log').read_text()
    assert not (run_dir / 'orchestrator.pid').exists() and not (run_dir / 'run.lock').exists()
    assert not fake_scheduler_bin.exists()


def test_dry_run_leaves_the_run_untouched(tmp_path, fake_steps, fake_scheduler_bin, capsys):
    run_dir = write_run(tmp_path)
    before = files(run_dir)
    assert cli(run_dir, 'start', '--dry-run') == 0
    out = capsys.readouterr().out
    assert 'first pass only' in out and 'sbatch' in out
    assert files(run_dir) == before and not fake_scheduler_bin.exists()


def test_dry_run_works_away_from_hpc(tmp_path, monkeypatch, fake_steps, capsys):
    run_dir = write_run(tmp_path)
    monkeypatch.setenv('PATH', str(tmp_path / 'nothing'))
    assert cli(run_dir, 'start', '--dry-run') == 0


def test_dry_run_refused_with_jobs(tmp_path, fake_steps, capsys):
    run_dir = write_run(tmp_path)
    (run_dir / 'jobs').mkdir()
    before = files(run_dir)
    assert cli(run_dir, 'start', '--dry-run') == 1
    assert 'jobs' in capsys.readouterr().err
    assert files(run_dir) == before and not (run_dir / 'run.lock').exists()


def test_stop_without_daemon_exits_1(tmp_path, capsys):
    run_dir = write_run(tmp_path)
    assert cli(run_dir, 'stop') == 1
    assert 'nothing to stop' in capsys.readouterr().err


# --- actions of the user -------------------------------------------------------------------

GROUP = f'{SYSTEM}.pes.1'


def submitted_pes(run_dir):
    """match analysed, pes prepared and submitted from the command line."""
    for argv in (('analyse', 'match'), ('prepare', 'pes'), ('submit', 'pes')):
        assert cli(run_dir, *argv) == 0


def edit_registry(run_dir, change):
    path = run_dir / 'jobs.json'
    registry = json.loads(path.read_text())
    change(registry)
    path.write_text(json.dumps(registry))


def launches(run_dir, group):
    from triboworkflow.execution.jobs import MANIFEST_FILE, read_manifest
    manifest, _ = read_manifest(layout.job_dir(run_dir, group) / MANIFEST_FILE)
    return [(l.calc_id, l.n, l.kind, l.level) for l in manifest.calculations]


def second_launch_of_c0(run_dir, fake_slurm):
    """Attempt 2 of pes/c0 listed by two groups with different launches."""
    from triboworkflow.execution.jobs import Launch, Manifest, write_job
    complete(run_dir, 'pes/c0', status=store_status('crashed'), stop_reason='truncated',
             clean_stop=False)
    assert cli(run_dir, 'submit', 'pes') == 0  # attempt 2 unchanged, in pes.2
    write_job(run_dir, Manifest(name=f'{SYSTEM}.pes.3', scheduler='slurm', run_command='pw.x',
                                walltime=3600, max_seconds_margin=60,
                                calculations=(Launch(SYSTEM, 'pes/c0', 2, 'manual', 1),)),
              run_dir / 'jobscript.sh')


def store_status(process, scf=None):
    from triboworkflow.core.store import Status
    return Status(process, scf, 'n/a')


@pytest.mark.parametrize('holder', ['orchestrator', 'triboworkflow submit pes'])
@pytest.mark.parametrize('argv', [
    ['prepare', 'pes'], ['submit', 'pes'], ['analyse', 'pes'], ['relaunch', 'pes/aa'],
    ['retry', 'pes'], ['resolve-submission', 'g', '--no-job'], ['clear-refusal', 'g'],
    ['release-claim', 'pes/aa', '--worker-over'],
])
def test_commands_refused_while_the_run_is_locked(tmp_path, fake_scheduler_bin, capsys, argv,
                                                  holder):
    run_dir = write_run(tmp_path)
    with RunLock(run_dir, holder):
        before = files(run_dir)
        assert cli(run_dir, *argv) == 1
        assert files(run_dir) == before
    err = capsys.readouterr().err
    assert f'the run is in use by {holder} (pid ' in err


def test_relaunch_by_hand(tmp_path, fake_steps, fake_slurm, fake_scheduler_bin, capsys):
    run_dir = write_run(tmp_path)
    submitted_pes(run_dir)
    complete(run_dir, 'pes/c0', status=store_status('ended'), stop_reason='user_request')
    fake_slurm.states['900'] = 'COMPLETED'  # the job of its claim is over
    capsys.readouterr()
    assert cli(run_dir, 'relaunch', 'pes/c0') == 0
    assert f'group {SYSTEM}.pes.2: 1 jobs submitted' in capsys.readouterr().out
    assert launches(run_dir, f'{SYSTEM}.pes.2') == [('pes/c0', 2, 'manual', 1)]
    assert fake_slurm.submitted == 3


def test_relaunch_at_another_level_from_scratch(tmp_path, fake_steps, fake_slurm,
                                                fake_scheduler_bin, capsys):
    from run_helpers import JOB_DONE, header, text
    run_dir = write_run(tmp_path)
    submitted_pes(run_dir)
    complete(run_dir, 'pes/c0', status=store_status('ended'), stop_reason='user_request')
    output = layout.calc_dir(run_dir, SYSTEM, 'pes/c0') / layout.OUTPUT_FILE
    output.write_text(text(header(mixing_beta=0.4), JOB_DONE))  # the summary of the attempt
    fake_slurm.states['900'] = 'COMPLETED'
    capsys.readouterr()
    assert cli(run_dir, 'relaunch', 'pes/c0', '--level', '4') == 1
    assert 'level 4 is not an integer from 1 to 3' in capsys.readouterr().err
    assert cli(run_dir, 'relaunch', 'pes/c0', '--level', '2', '--from-scratch') == 0
    assert (f'{SYSTEM}/pes/c0: relaunched by hand, attempt 2 at level 2, from scratch'
            in capsys.readouterr().out)
    assert launches(run_dir, f'{SYSTEM}.pes.2') == [('pes/c0', 2, 'manual', 2)]


def test_relaunch_refused_while_the_worker_may_run(tmp_path, fake_steps, fake_slurm,
                                                   fake_scheduler_bin, capsys):
    run_dir = write_run(tmp_path)
    submitted_pes(run_dir)
    complete(run_dir, 'pes/c0', status=store_status('ended'), stop_reason='user_request')
    fake_slurm.states['900'] = 'RUNNING'
    assert cli(run_dir, 'relaunch', 'pes/c0') == 1
    assert 'job 900 of the claim is running' in capsys.readouterr().err
    assert fake_slurm.submitted == 2


def test_relaunch_with_group(tmp_path, fake_steps, fake_slurm, fake_scheduler_bin, capsys):
    run_dir = write_run(tmp_path)
    submitted_pes(run_dir)
    second_launch_of_c0(run_dir, fake_slurm)
    complete(run_dir, 'pes/c0', job_id='901', status=store_status('crashed'),
             stop_reason='truncated', clean_stop=False)  # attempt 2, its launch not recorded
    fake_slurm.states['901'] = 'FAILED'
    capsys.readouterr()
    assert cli(run_dir, 'relaunch', 'pes/c0') == 1
    err = capsys.readouterr().err
    assert f'{SYSTEM}.pes.2 (unchanged, level 1), {SYSTEM}.pes.3 (manual, level 1)' in err
    assert '--group' in err
    assert cli(run_dir, 'relaunch', 'pes/c0', '--group', f'{SYSTEM}.pes.2') == 0
    assert launches(run_dir, f'{SYSTEM}.pes.4') == [('pes/c0', 3, 'manual', 1)]


def test_retry_of_a_failed_step(tmp_path, fake_steps, fake_slurm, fake_scheduler_bin, capsys):
    run_dir = write_run(tmp_path)
    assert cli(run_dir, 'analyse', 'match') == 0 and cli(run_dir, 'prepare', 'pes') == 0
    assert cli(run_dir, 'retry', 'pes') == 1  # running: nothing to retry
    assert 'pes is running' in capsys.readouterr().err
    fake_steps['pes'].fingerprint = 'sha256:' + 'b' * 64  # settings changed
    assert cli(run_dir, 'submit', 'pes') == 1
    assert 'settings changed since the preparation' in capsys.readouterr().out
    del fake_steps['pes'].fingerprint  # settings as they were
    assert cli(run_dir, 'retry', 'pes') == 0
    assert f'{SYSTEM}/pes: waiting' in capsys.readouterr().out
    assert cli(run_dir, 'prepare', 'pes') == 0


def test_resolve_submission_with_and_without_job(tmp_path, fake_steps, fake_slurm,
                                                 fake_scheduler_bin, capsys):
    run_dir = write_run(tmp_path)
    submitted_pes(run_dir)
    lost = {'error': 'Command timed out after 30 seconds', 'at': '2026-10-30T10:00:00+01:00'}
    edit_registry(run_dir, lambda r: r['groups'][GROUP].update(unknown=[lost, lost]))
    assert cli(run_dir, 'resolve-submission', GROUP, '--job-id', '4242') == 0
    registry = json.loads((run_dir / 'jobs.json').read_text())
    assert '4242' in registry['jobs'] and len(registry['groups'][GROUP]['unknown']) == 1
    assert cli(run_dir, 'resolve-submission', GROUP, '--no-job') == 0
    registry = json.loads((run_dir / 'jobs.json').read_text())
    assert registry['groups'][GROUP]['unknown'] == []
    assert cli(run_dir, 'resolve-submission', GROUP, '--no-job') == 1
    assert 'no submission of outcome not known' in capsys.readouterr().err


@pytest.mark.parametrize('argv', [['resolve-submission', 'g'],
                                  ['resolve-submission', 'g', '--job-id', '1', '--no-job']])
def test_resolve_submission_needs_job_or_no_job(argv, capsys):
    with pytest.raises(SystemExit) as stop:
        main(argv)
    assert stop.value.code == 2


def test_clear_refusal(tmp_path, fake_steps, fake_slurm, fake_scheduler_bin, capsys):
    run_dir = write_run(tmp_path)
    submitted_pes(run_dir)
    edit_registry(run_dir, lambda r: r['groups'][GROUP].update(refused='invalid account'))
    assert cli(run_dir, 'clear-refusal', GROUP) == 0
    assert json.loads((run_dir / 'jobs.json').read_text())['groups'][GROUP]['refused'] is None
    assert cli(run_dir, 'clear-refusal', GROUP) == 1
    assert 'no submission refused' in capsys.readouterr().err


def test_release_claim_then_relaunch(tmp_path, fake_steps, fake_slurm, fake_scheduler_bin,
                                     capsys):
    run_dir = write_run(tmp_path)
    submitted_pes(run_dir)
    (layout.calc_dir(run_dir, SYSTEM, 'pes/c0') / layout.CLAIM_FILE).write_text('')
    capsys.readouterr()
    assert cli(run_dir, 'release-claim', 'pes/c0', '--worker-over') == 0
    assert 'attempt 1' in capsys.readouterr().out
    assert cli(run_dir, 'relaunch', 'pes/c0') == 0
    assert launches(run_dir, f'{SYSTEM}.pes.2') == [('pes/c0', 2, 'manual', 1)]
    assert fake_slurm.submitted == 3


def test_release_claim_with_different_launches_needs_group(tmp_path, fake_steps, fake_slurm,
                                                           fake_scheduler_bin, capsys):
    run_dir = write_run(tmp_path)
    submitted_pes(run_dir)
    second_launch_of_c0(run_dir, fake_slurm)
    (layout.calc_dir(run_dir, SYSTEM, 'pes/c0') / layout.CLAIM_FILE).write_text('')
    capsys.readouterr()
    assert cli(run_dir, 'release-claim', 'pes/c0', '--worker-over') == 1
    err = capsys.readouterr().err
    assert f'{SYSTEM}.pes.2 (unchanged, level 1), {SYSTEM}.pes.3 (manual, level 1)' in err
    assert cli(run_dir, 'release-claim', 'pes/c0', '--worker-over', '--group',
               f'{SYSTEM}.pes.3') == 0
    from triboworkflow.core.store import read_record
    attempt = read_record(layout.system_dir(run_dir, SYSTEM), 'pes/c0').attempts[-1]
    assert (attempt.n, attempt.remedies['kind']) == (2, 'manual')


def test_release_claim_needs_the_confirmation(capsys):
    with pytest.raises(SystemExit) as stop:
        main(['release-claim', 'pes/c0'])
    assert stop.value.code == 2 and '--worker-over' in capsys.readouterr().err


def test_status_shows_the_steps_run_by_hand(tmp_path, fake_steps, fake_slurm, capsys):
    run_dir = write_run(tmp_path, steps=('match',))
    build_orchestrator(run_dir, steps=['match', 'pes']).prepare_step(SYSTEM, 'match')
    assert main(['status', '-r', str(run_dir)]) == 0
    state = json.loads((run_dir / 'orchestrator.json').read_text())
    assert 'match' in capsys.readouterr().out and 'pes' in state['systems'][SYSTEM]
