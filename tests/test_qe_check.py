# Author: Vito Foderà
# Created: 2026-11-04

"""Local runs of ``tests/hpc/qe_check.py``, so that the integrated check works when it is used.

The runs of each case are written with a machine file whose ``pw.x`` and
``pp.x`` are the fakes of ``qe_check_fake.py``: nothing is computed. The
orchestrator runs here, pass by pass, with the test steps of the harness and
a simulated Slurm (``run_helpers.Slurm``); the workers run here too, on the
manifests of the groups, as the jobs would. Then the report reads the run.
POSIX only, since the worker launches pw.x with bash.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cli_helpers import fake_scheduler_bin  # noqa: F401
from run_helpers import Slurm
from triboworkflow.cli import run as cli_run
from triboworkflow.core.store import read_record
from triboworkflow.execution import layout
from triboworkflow.workflows.interface import registry

HPC = Path(__file__).resolve().parent / 'hpc'
SCRIPT = HPC / 'qe_check.py'
FAKE = Path(__file__).resolve().parent / 'qe_check_fake.py'
sys.path.insert(0, str(HPC))

import qe_check  # noqa: E402

pytestmark = pytest.mark.skipif(os.name != 'posix' or not Path('/bin/bash').exists(),
                                reason='the worker launches pw.x with /bin/bash')

JOBSCRIPT = '#!/bin/bash\n#SBATCH --time={time}\n#SBATCH --output=job.%j.out\n'
SYSTEMS = {'Al': 'Al111_Al111', 'Si': 'Si111_Si111'}


@pytest.fixture
def machine(tmp_path):
    """Machine file with the fake pw.x and pp.x, two UPF files and their manifest."""
    pseudo = tmp_path / 'pseudo'
    pseudo.mkdir()
    lines = []
    for name in ('Al.UPF', 'Si.UPF'):
        (pseudo / name).write_text(f'fake {name}\n')
        lines.append(f'{hashlib.md5((pseudo / name).read_bytes()).hexdigest()}  {name}')
    (pseudo / 'pseudo.md5').write_text('\n'.join(lines) + '\n')
    (tmp_path / 'job.sh').write_text(JOBSCRIPT)
    path = tmp_path / 'machine.toml'
    path.write_text(f'''scheduler = "slurm"
jobscript = "job.sh"
run_command = "{sys.executable} {FAKE} pw"
pp_command = "{sys.executable} {FAKE} pp"
[qe]
version = "7.2"
pseudo_dir = "{pseudo}"
upf_md5 = "{pseudo}/pseudo.md5"
[species.Al]
mass = 26.9815
pseudopotential = "Al.UPF"
[species.Si]
mass = 28.085
pseudopotential = "Si.UPF"
[k_points]
kdens = 2.0
[structures]
layers = 2
big_size = [2, 1]
big_layers = 2
''')
    return path


@pytest.fixture
def steps(monkeypatch):
    """The registry of the steps restored after the test; the test steps put in it."""
    monkeypatch.setattr(registry, 'STEPS', dict(registry.STEPS))
    qe_check.install()


def write(machine, run, case, system='Al', *extra):
    assert qe_check.main(['write', str(machine), str(run), '--case', case, '--system', system,
                          *extra]) == 0
    return run


def report(run):
    return subprocess.run([sys.executable, str(SCRIPT), 'report', str(run)], capture_output=True,
                          text=True, timeout=300)


def verdicts(out):
    """``{subject: (verdict, text)}`` of the verdicts of a report."""
    found, inside = {}, False
    for line in out.splitlines():
        if line == 'Verdicts:':
            inside = True
        elif inside and line.startswith('  '):
            for verdict in sorted(qe_check.VERDICTS, key=len, reverse=True):
                if line[2:].startswith(verdict):
                    subject, _, text = line[2 + 13:].partition('  ')
                    found[subject.strip()] = (verdict, text.strip())
                    break
    return found


class Cluster:
    """A run driven here: orchestrator passes, a simulated Slurm, workers on the manifests."""

    def __init__(self, run, monkeypatch, ends=None, env=None):
        self.run = run
        self.slurm = Slurm()
        monkeypatch.setattr(cli_run, 'make_scheduler', lambda settings, dry_run=False: self.slurm)
        fake = run.parent / f'{run.name}.fake.json'
        fake.write_text(json.dumps(ends or {}))
        self.env = {**os.environ, 'QE_CHECK_FAKE': str(fake), **(env or {})}
        self.workers = []

    def orchestrator(self):
        return cli_run.build_orchestrator(self.run)

    def jobs(self):
        """``{job_id: manifest}`` of the groups submitted."""
        found = {}
        for path in (self.run / layout.JOBS_DIR).glob('*/submitted.json'):
            for job_id in json.loads(path.read_text())['job_ids']:
                found[job_id] = path.parent / 'manifest.json'
        return found

    def run_jobs(self):
        """Run a worker for each job pending, as its job would; set the job over."""
        for job_id, manifest in sorted(self.jobs().items()):
            if self.slurm.states.get(job_id) != 'PENDING':
                continue
            done = subprocess.run([sys.executable, '-m', 'triboworkflow.execution.worker',
                                   str(manifest)], env={**self.env, 'SLURM_JOB_ID': job_id},
                                  capture_output=True, text=True, timeout=300)
            (manifest.parent / f'job.{job_id}.err').write_text(done.stderr)
            self.workers.append((job_id, done.returncode))
            self.slurm.states[job_id] = 'COMPLETED' if done.returncode == 0 else 'FAILED'

    def drive(self, passes=10):
        """Passes and jobs until no step waits or runs; the last report."""
        for _ in range(passes):
            report = self.orchestrator().run_pass()
            if report.finished:
                return report
            self.run_jobs()
        return report


# --- writing ---------------------------------------------------------------------------------

@pytest.mark.parametrize('system', ['Al', 'Si'])
@pytest.mark.parametrize('case, count', [('chain', 6), ('restart', 4), ('manual', 4),
                                         ('claim', 2), ('density', 4), ('single', 1)])
def test_write_every_case(machine, tmp_path, capsys, case, count, system):
    run = write(machine, tmp_path / 'run', case, system)
    out = capsys.readouterr().out
    word = 'calculation' if count == 1 else 'calculations'
    assert f'case {case}, system {system}, {count} {word}' in out
    plan = json.loads((run / 'qe_check' / 'plan.json').read_text())
    assert plan['case'] == case and plan['system'] == system
    walltime = qe_check.WALLTIMES[case]
    assert f'#SBATCH --time={qe_check.hms(walltime)}\n' in (run / 'jobscript.sh').read_text()
    config = (run / 'config.toml').read_text()
    other = qe_check.OTHER_SPECIES[system]
    assert config.index(f'label = "{system}"') < config.index(f'label = "{other}"')
    if case == 'single':
        assert '[generation]' in config and (run / 'jobscript.sh').read_text().endswith(
            'pw -in input.pwi >> output.pwo\n')
    else:
        assert f'walltime = {walltime}' in config
    assert ((run / 'qe_check' / 'inject' / 'sitecustomize.py').is_file()) == (case == 'claim')


def test_write_refuses_an_existing_directory_and_a_jobscript_without_time(machine, tmp_path,
                                                                          capsys):
    run = write(machine, tmp_path / 'run', 'chain')
    assert qe_check.main(['write', str(machine), str(run), '--case', 'chain', '--system',
                          'Al']) == 1
    assert 'exists: give a new directory' in capsys.readouterr().err
    (tmp_path / 'job.sh').write_text('#!/bin/bash\n#SBATCH --time=00:10:00\n')
    assert qe_check.main(['write', str(machine), str(tmp_path / 'other'), '--case', 'chain',
                          '--system', 'Al']) == 1
    assert 'no {time}' in capsys.readouterr().err
    machine.write_text(machine.read_text() + 'unknown = 1\n')
    assert qe_check.main(['write', str(machine), str(tmp_path / 'other'), '--case', 'chain',
                          '--system', 'Al']) == 1
    assert 'keys not known: unknown' in capsys.readouterr().err


@pytest.mark.parametrize('layers', [2, 4])
def test_si_slabs_are_whole_bilayers(layers):
    """Every surface of the Si slabs and of the halves of the interface ends a bilayer."""
    import numpy as np

    options = {**qe_check.STRUCTURE_DEFAULTS, 'layers': layers}
    atoms = qe_check.build_structures('Si', options)
    for name in ('bottom', 'top', 'interface_bottom', 'interface_top'):
        z = np.sort(atoms[name].positions[:, 2])
        gaps = np.diff(z)
        assert len(z) == layers
        assert np.allclose(gaps[0::2], gaps[0]) and gaps[0] < 1.0, name   # within a bilayer
    z = np.sort(atoms['interface'].positions[:, 2])
    assert np.diff(z)[layers - 1] == pytest.approx(np.diff(z)[1] + options['gap']) \
        if layers > 2 else np.diff(z)[1] > 2.5


def test_dry_run_and_factory_of_the_orchestrator_in_the_background(machine, tmp_path):
    run = write(machine, tmp_path / 'run', 'chain', 'Si')
    done = subprocess.run([sys.executable, str(SCRIPT), 'cli', 'start', '--dry-run', '-r',
                           str(run)], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    assert 'Si111_Si111/match: completed' in done.stdout
    assert 'Si111_Si111/pes: running\n  calculations: 3 launched' in done.stdout
    assert done.stdout.count('sbatch ') == 2
    assert not (run / 'jobs').exists()
    # The background process imports the factory by name, with this PYTHONPATH.
    code = ('import sys; from qe_check import daemon_factory; '
            'o = daemon_factory(sys.argv[1], "2"); print(sorted(o.steps), o.execution.groups)')
    done = subprocess.run([sys.executable, '-c', code, str(run)], capture_output=True, text=True,
                          env={**os.environ, 'PYTHONPATH': str(HPC)}, timeout=300, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "['adhesion', 'match', 'pes'] 2"


def test_a_step_out_of_the_case_is_an_error(machine, tmp_path, steps):
    run = write(machine, tmp_path / 'run', 'density')
    with pytest.raises(qe_check.CheckError, match="step 'pes' is not in the case"):
        qe_check.STEP_CLASSES['pes']().calculations(run, SYSTEMS['Al'])


# --- runs driven here ---------------------------------------------------------------------------

def test_chain_to_the_end_and_the_report(machine, tmp_path, monkeypatch, steps):
    run = write(machine, tmp_path / 'run', 'chain')
    cluster = Cluster(run, monkeypatch)
    final = cluster.drive()
    assert final.success, final.steps
    assert [code for _, code in cluster.workers] == [0] * len(cluster.workers)
    done = report(run)
    assert done.returncode == 0, done.stdout + done.stderr
    found = verdicts(done.stdout)
    for calc in ('pes/bottom', 'pes/top', 'pes/interface', 'adhesion/bottom', 'adhesion/top',
                 'adhesion/interface'):
        assert found[f'{calc} completed'][0] == 'ok'
        assert found[f'{calc} symmetry (Q17)'][0] == 'ok'
    for calc in ('adhesion/bottom', 'adhesion/top', 'adhesion/interface'):
        assert found[f'{calc} stress'][0] == 'ok'
    assert found['adhesion/interface z only'][0] == 'ok'
    assert found['steps'][0] == 'ok'
    assert 'observed: version 7.2; Al Al.UPF md5' in done.stdout and ', Si Si.UPF md5 ' in \
        done.stdout
    # An output changed after its attempt is an inconsistency.
    output = layout.calc_dir(run, SYSTEMS['Al'], 'pes/bottom') / 'output.pwo'
    output.write_text(output.read_text() + '\n')
    done = report(run)
    assert done.returncode == 1
    assert 'differs from the digest recorded' in verdicts(done.stdout)[
        'pes/bottom attempt 1'][1]


def test_restarts_levels_and_margin(machine, tmp_path, monkeypatch, steps):
    run = write(machine, tmp_path / 'run', 'restart')
    ends = {'long_scf': ['max_seconds'], 'long_relax': ['max_seconds'],
            'few_steps': ['max_steps'], 'scf_fail': ['scf_not_converged']}
    cluster = Cluster(run, monkeypatch, ends)
    assert cluster.drive().success
    system = SYSTEMS['Al']
    records = {label: read_record(layout.system_dir(run, system), f'adhesion/{label}')
               for label in ends}
    assert {label: len(r.attempts) for label, r in records.items()} == dict.fromkeys(ends, 2)
    found = verdicts(report(run).stdout)
    for label, (end,) in ends.items():
        assert found[f'adhesion/{label} restart after {end}'][0] == 'ok', found
    assert found['adhesion/long_scf SCF restarted from its iteration'][0] == 'ok'
    assert found['adhesion/few_steps level 2'][0] == 'ok'
    assert found['adhesion/few_steps completed'][0] == 'ok'
    assert found['adhesion/long_scf margin at max_seconds (Q09)'][0] == 'not observed'
    assert found['other run_command'][0] == 'not observed'
    # The jobs as sacct gives them: the stop at max_seconds leaves time before the limit.
    first = records['long_scf'].attempts[0]
    started = qe_check.local_time(first.started)
    ended = qe_check.local_time(first.ended)

    def snapshot(state, end):
        row = '|'.join([first.job_id, state, '0:0', started.isoformat(), started.isoformat(),
                        end.isoformat(), '00:00:30', '00:12:00', '0', 'node01'])
        folder = run / 'qe_check' / 'snapshots'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f'{state}.json').write_text(json.dumps({
            'time': end.isoformat(), 'host': 'login', 'job_ids': [first.job_id],
            'sacct': {'command': [], 'returncode': 0, 'stdout': row + '\n', 'stderr': ''},
            'squeue': None, 'claims': {}}))

    snapshot('COMPLETED', ended)
    margin = verdicts(report(run).stdout)['adhesion/long_scf margin at max_seconds (Q09)']
    assert margin[0] == 'ok' and 'pw.x stopped 0 s before the end of the job' in margin[1]
    snapshot('TIMEOUT', ended)   # the later snapshot is read
    done = report(run)
    assert done.returncode == 1
    assert verdicts(done.stdout)['adhesion/long_scf margin at max_seconds (Q09)'][0] == 'PROBLEM'


def test_empty_claim_released_and_relaunched(machine, tmp_path, monkeypatch, steps):
    run = write(machine, tmp_path / 'run', 'claim')
    inject = {'QE_CHECK_EMPTY_CLAIM': str(run / 'qe_check' / 'empty_claim.json'),
              'PYTHONPATH': str(run / 'qe_check' / 'inject')}
    assert f'export QE_CHECK_EMPTY_CLAIM={inject["QE_CHECK_EMPTY_CLAIM"]}' in \
        (run / 'jobscript.sh').read_text()
    cluster = Cluster(run, monkeypatch, env=inject)
    system, victim = SYSTEMS['Al'], 'adhesion/victim'
    cluster.orchestrator().run_pass()
    cluster.run_jobs()
    assert sorted(code for _, code in cluster.workers) == [-9, 0]   # the first one killed
    claim = layout.calc_dir(run, system, victim) / layout.CLAIM_FILE
    assert claim.read_text() == '' and read_record(layout.system_dir(run, system),
                                                   victim).attempts == ()
    assert (run / 'qe_check' / 'killed' / 'victim').is_file()
    step = cluster.orchestrator().run_pass().steps[0]
    assert step.status == 'running' and step.calcs == {'completed': 1, 'uncertain': 1}
    orch = cluster.orchestrator()
    assert orch.release_claim(system, victim) == 1
    orch.relaunch(system, victim)
    cluster.run_jobs()
    assert cluster.workers[-1][1] == 0   # killed once only
    assert cluster.drive().success
    found = verdicts(report(run).stdout)
    assert found['adhesion/victim empty claim released'] == (
        'ok', 'attempt 1 archived with nothing known, then relaunched by hand')
    assert found['adhesion/victim completed'][0] == 'ok'
    assert found['adhesion/alive completed'][0] == 'ok'


def test_density_of_the_interface_and_of_its_fragments(machine, tmp_path, monkeypatch, steps):
    run = write(machine, tmp_path / 'run', 'density')
    cluster = Cluster(run, monkeypatch)
    assert cluster.drive().success
    done = report(run)
    assert done.returncode == 0, done.stdout + done.stderr
    found = verdicts(done.stdout)
    for label in ('interface', 'bottom', 'top'):
        verdict, text = found[f'charge/{label} charge density']
        assert verdict == 'ok' and 'grid (4, 4, 24)' in text and '(QEChargeAnalyzer)' in text
    assert found['densities comparable'][0] == 'ok'
    assert found['charge/nowf disk_io nowf'][0] == 'ok'
    assert 'pp.x: exit 0, error None; pp.x and reading of the filplot' in done.stdout


def test_held_and_requeued_jobs_from_the_snapshots(machine, tmp_path):
    run = write(machine, tmp_path / 'run', 'manual')
    folder = run / 'qe_check' / 'snapshots'
    folder.mkdir()
    rows = ['1001|COMPLETED|0:0|2026-11-04T10:00:00|2026-11-04T10:01:00|2026-11-04T10:20:00|'
            '00:19:00|00:30:00|1|node01']

    def command(stdout):
        return {'command': [], 'returncode': 0, 'stdout': stdout, 'stderr': ''}

    (folder / '1.json').write_text(json.dumps({
        'time': '2026-11-04T10:00:30', 'host': 'login', 'job_ids': ['1000', '1001'],
        'sacct': command(''), 'squeue': command('1000|PENDING|JobHeldUser\n'),
        'claims': {'systems/x/adhesion/to_requeue': {'n': 1, 'job_id': '1001',
                                                    'restart_count': None}}}))
    (folder / '2.json').write_text(json.dumps({
        'time': '2026-11-04T10:30:00', 'host': 'login', 'job_ids': ['1000', '1001'],
        'sacct': command('\n'.join(rows) + '\n'), 'squeue': command(''), 'claims': {}}))
    found = verdicts(report(run).stdout)
    assert found['held job 1000'][0] == 'ok'
    assert found['requeued job 1001'][0] == 'ok'
    assert 'restart_count in the claims of the snapshots: [\'null\']' in \
        found['requeued job 1001'][1]
    assert found['adhesion/to_cancel'] == ('not observed', 'not prepared')


def test_snapshot_records_sacct_squeue_and_claims(machine, tmp_path, fake_scheduler_bin):
    run = write(machine, tmp_path / 'run', 'claim')
    group = run / 'jobs' / 'g'
    group.mkdir(parents=True)
    (group / 'submitted.json').write_text(json.dumps({'job_ids': ['1000', '1001']}))
    claim = layout.calc_dir(run, SYSTEMS['Al'], 'adhesion/victim')
    claim.mkdir(parents=True)
    (claim / layout.CLAIM_FILE).write_text('')
    assert qe_check.main(['snapshot', str(run)]) == 0
    snap = qe_check.read_snapshots(run)[-1]
    assert snap['job_ids'] == ['1000', '1001']
    assert snap['claims'] == {'systems/Al111_Al111/adhesion/victim': None}
    calls = fake_scheduler_bin.read_text()
    assert 'sacct -X -D -n -P -o JobID,State,ExitCode,Submit,Start,End,Elapsed,Timelimit,' \
           'Restarts,NodeList -j 1000,1001' in calls
    assert 'squeue -h -o %i|%T|%r -j 1000,1001' in calls


# --- reading --------------------------------------------------------------------------------

@pytest.mark.parametrize('lines, expected', [
    (['', '     No symmetry found', '', ''], {'nsym': 1, 'inversion': False, 'fractional': 0,
                                               'ignored': 0}),
    (['', '     12 Sym. Ops., with inversion, found', '', ''],
     {'nsym': 12, 'inversion': True, 'fractional': 0, 'ignored': 0}),
    (['', '     48 Sym. Ops., with inversion, found (24 have fractional translation)'],
     {'nsym': 48, 'inversion': True, 'fractional': 24, 'ignored': 0}),
    (['', '      6 Sym. Ops. (no inversion) found ( 3 have fractional translation)',
      '          (note:  2 additional sym.ops. were found but ignored',
      '           their fractional translations are incommensurate with FFT grid)'],
     {'nsym': 6, 'inversion': False, 'fractional': 3, 'ignored': 2}),
])
def test_symmetry_lines_of_pw(lines, expected):
    """The lines of PW/src/summary.f90 (QE 7.2), in the last run of the output."""
    text = '\n'.join(['     Program PWSCF v.7.2 starts', '      2 Sym. Ops. (no inversion) found',
                      '     Program PWSCF v.7.2 starts', *lines])
    assert qe_check.qe_symmetry(text) == expected
    assert qe_check.qe_symmetry('     Program PWSCF v.7.2 starts\n') is None


def test_times_and_durations_of_slurm():
    assert qe_check.duration('00:12:00') == 720 and qe_check.duration('1-02:00:05') == 93605
    assert qe_check.duration('05:30') == 330 and qe_check.duration('UNLIMITED') is None
    assert qe_check.local_time('2026-11-04T10:00:00+01:00').isoformat() == '2026-11-04T10:00:00'
    assert qe_check.local_time('Unknown') is None
