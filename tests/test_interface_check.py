# Author: Vito Foderà
# Created: 2026-11-09

"""Tests of the report on a run of the interface workflow (``tests/hpc/interface_check.py``).

The run is the one of ``test_interface_chain.py``: the nine steps, finished, with the
stand-in pw.x. The report reads copies of it in another folder, as it reads a run copied from
the HPC machine; each test that breaks something breaks its own copy and asks for the verdict
that the break must give.
"""

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

import charge_files
from chain_helpers import CUCU, posix_only
from triboworkflow.core.structures import read_structure

HPC = Path(__file__).resolve().parent / 'hpc'
sys.path.insert(0, str(HPC))

import interface_check as ic  # noqa: E402

pytestmark = posix_only
JOB_HEADER = 'JobID|JobName|State|ExitCode|Elapsed|Timelimit|ReqMem|MaxRSS|NodeList'


@pytest.fixture
def run(chain, tmp_path):
    """A copy of the finished run, in another folder."""
    copied = tmp_path / 'copy'
    shutil.copytree(chain.run_dir, copied)
    return copied


def verdicts(run_dir, sacct=None):
    """``{subject: (verdict, text)}`` of the report of a run."""
    rep = ic.make_report(run_dir, sacct)
    found = {subject: (verdict, text) for verdict, subject, text in rep.findings.items}
    assert len(found) == len(rep.findings.items), 'two verdicts with one subject'
    return found


def sdir(run_dir):
    return run_dir / 'systems' / CUCU


def edit_json(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data, indent=1) + '\n')


def edit_document(run_dir, change):
    edit_json(sdir(run_dir) / 'system.json', lambda data: change(data['document']))


def tree(run_dir):
    """Digest of every file of a run, by path."""
    return {p.relative_to(run_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(run_dir.rglob('*')) if p.is_file()}


def job_ids(run_dir, step):
    group = run_dir / 'jobs' / f'{CUCU}.{step}.1'
    return json.loads((group / 'submitted.json').read_text())['job_ids']


def sacct_file(path, jobs):
    """File of ``sacct -P``: ``jobs`` is ``{job id: (state, MaxRSS of its step)}``."""
    lines = [JOB_HEADER]
    for job_id, (state, rss) in jobs.items():
        lines.append(f'{job_id}|job|{state}|0:0|00:01:30|01:00:00|8G||node01')
        lines.append(f'{job_id}.batch|batch|{state}|0:0|00:01:30||8G|{rss}|node01')
    path.write_text('\n'.join(lines) + '\n')
    return path


# --- a finished run ------------------------------------------------------------------------

def test_a_finished_run_has_no_problem(run):
    found = verdicts(run)
    assert {v for v, _ in found.values()} == {'ok', 'note', 'not observed'}
    assert {s for s, (v, _) in found.items() if v == 'not observed'} == {'accounting'}
    # The strings of the shear stopped at nstepmax = 2: the entries say so.
    assert {s for s, (v, _) in found.items() if v == 'note'} == {'shear convergence'}
    assert 'stopped at nstepmax = 2 without converging' in found['shear convergence'][1]


def test_every_topic_is_reported(run):
    found = verdicts(run)
    for subject in ('steps', 'pes calculations', 'pes_scf calculations', 'adhesion calculations',
                    'ppes calculations', 'charge calculations', 'spin', 'slab 1', 'slab 2', 'pes',
                    'pes_scf', 'plotpes', 'plotpes job', 'plotpes_scf', 'plotpes_scf job',
                    'adhesion', 'ppes', 'charge fragments', 'charge densities',
                    'charge contact plane', 'shear job', 'shear time', 'shear warnings',
                    'shear convergence', 'shear cell', 'accounting'):
        assert subject in found, subject


def test_the_report_changes_nothing(run, capsys):
    before = tree(run)
    assert ic.main(['report', str(run)]) == 0
    out = capsys.readouterr().out
    assert tree(run) == before
    # The run read is the copy, in another folder than the one the jobs ran in.
    assert out.startswith(f'Run {run.resolve()}: system {CUCU}')
    assert out.count('\n  ok ') >= 20 and 'RESULT: ' in out and ' 0 PROBLEM' in out


# --- what each break gives -----------------------------------------------------------------

def test_an_energy_that_is_not_the_one_of_the_output(run):
    point = next(iter(json.loads((sdir(run) / 'system.json').read_text())['document']['pes']
                      ['data']))

    def change(document):
        document['pes']['data'][point]['output']['energy'] += 0.5

    edit_document(run, change)
    verdict, text = verdicts(run)['pes']
    assert verdict == 'PROBLEM' and f'pes/{point}' in text and 'in the output' in text


def test_an_analysis_job_that_failed(run, capsys):
    group = run / 'jobs' / f'{CUCU}.plotpes.1'

    def change(outcome):
        outcome.update(result='failed', error='MemoryError: out of memory')

    edit_json(group / 'outcome.json', change)
    verdict, text = verdicts(run)['plotpes job']
    assert verdict == 'PROBLEM' and 'MemoryError' in text
    assert ic.main(['report', str(run)]) == 1
    assert ' PROBLEM ' in capsys.readouterr().out


def test_an_interface_that_lost_the_slab_of_its_atoms(run):
    def change(document):
        for site in document['structure']['opt']['regular']['interface_min']['sites']:
            site['properties'].pop('interface_region', None)

    edit_document(run, change)
    found = verdicts(run)
    assert found['adhesion'][0] == 'PROBLEM' and 'interface_region' in found['adhesion'][1]
    assert found['ppes'][0] == 'PROBLEM' and 'no interface_region' in found['ppes'][1]


def test_a_bottom_atom_moved_in_the_input_of_a_shift(run):
    folder = next(p for p in sorted((sdir(run) / 'ppes').iterdir()) if p.is_dir())
    path = folder / 'structure_in.extxyz'
    lines = path.read_text().splitlines()
    for i in range(2, len(lines)):
        parts = lines[i].split()
        parts[1] = f'{float(parts[1]) + 0.1:.8f}'
        lines[i] = ' '.join(parts)
    path.write_text('\n'.join(lines) + '\n')
    verdict, text = verdicts(run)['ppes']
    assert verdict == 'PROBLEM' and 'bottom' in text


def test_densities_on_another_grid(run):
    folder = sdir(run) / 'charge' / 'min_site_top_slab'
    structure = read_structure(folder / 'structure_out.extxyz')
    (folder / 'density.filplot').write_text(charge_files.structure_filplot(
        structure, grid=(4, 4, 20), electrons=float(len(structure))))
    verdict, text = verdicts(run)['charge densities']
    assert verdict == 'PROBLEM' and 'grid' in text


def test_a_density_that_is_not_in_the_copy(run):
    (sdir(run) / 'charge' / 'min_site_bot_slab' / 'density.filplot').unlink()
    verdict, text = verdicts(run)['charge densities']
    assert verdict == 'not observed' and '2 of the 3 filplots' in text


def test_a_cell_of_the_interface_that_is_not_the_cell_of_the_pes(run):
    def change(document):
        document['structure']['init']['interface']['lattice']['matrix'][0][0] += 0.01

    edit_document(run, change)
    verdict, text = verdicts(run)['shear cell']
    assert verdict == 'PROBLEM' and 'largest difference 0.01' in text


def move_atom(path, index, dx):
    """Move an atom of an extxyz file along x, in place."""
    lines = path.read_text().splitlines()
    parts = lines[index + 2].split()
    parts[1] = f'{float(parts[1]) + dx:.8f}'
    lines[index + 2] = ' '.join(parts)
    path.write_text('\n'.join(lines) + '\n')


def fragment(run, part):
    return sdir(run) / 'charge' / f'min_site_{part}'


def test_fragments_swapped_with_the_same_number_of_atoms(run):
    # Four Cu atoms and one cell in both slabs: only their positions tell them apart.
    bot, top = (fragment(run, part) / 'structure_in.extxyz' for part in ('bot_slab', 'top_slab'))
    first, second = bot.read_text(), top.read_text()
    bot.write_text(second)
    top.write_text(first)
    verdict, text = verdicts(run)['charge fragments']
    assert verdict == 'PROBLEM' and 'positions differ' in text


def test_an_atom_of_a_fragment_that_is_not_where_it_is_in_the_interface(run):
    move_atom(fragment(run, 'top_slab') / 'structure_in.extxyz', 0, 0.1)
    verdict, text = verdicts(run)['charge fragments']
    assert verdict == 'PROBLEM' and 'charge/min_site_top_slab' in text and 'up to 0.1' in text


def test_an_interface_of_the_charge_that_is_not_the_one_of_the_adhesion(run):
    move_atom(fragment(run, 'interface') / 'structure_in.extxyz', 2, 0.3)
    verdict, text = verdicts(run)['charge fragments']
    assert verdict == 'PROBLEM' and 'interface of the adhesion moved together' in text


def test_a_density_that_is_there_and_cannot_be_read(run):
    (fragment(run, 'bot_slab') / 'density.filplot').write_text('corrupted filplot\n')
    verdict, text = verdicts(run)['charge densities']
    assert verdict == 'PROBLEM'
    assert 'charge/min_site_bot_slab: density.filplot not read' in text


def test_a_density_cut_short(run):
    path = fragment(run, 'top_slab') / 'density.filplot'
    path.write_text(''.join(path.read_text().splitlines(keepends=True)[:6]))
    verdict, text = verdicts(run)['charge densities']
    assert verdict == 'PROBLEM' and 'charge/min_site_top_slab: density.filplot not read' in text


@pytest.mark.parametrize('name', ['pes', 'pes_scf'])
@pytest.mark.parametrize('grid, change, word', [
    ('hs_unique_ev', lambda row: row.__setitem__(3, row[3] + 100.0), 'hs_unique_ev'),
    ('hs_unique_ev', lambda row: row.__setitem__(1, row[1] + 0.5), 'x, y'),
    ('hs_unique_jm2', lambda row: row.__setitem__(3, row[3] * 1.01), 'hs_unique_jm2'),
], ids=['energies of the grid in eV', 'shift of a point', 'energies in J/m^2'])
def test_a_grid_that_is_not_made_of_its_points(run, name, grid, change, word):
    # The order, the number of rows and the sites stay as they were: only the numbers change.
    def edit(document):
        for row in document[name]['energy']['grid'][grid]:
            change(row)

    edit_document(run, edit)
    verdict, text = verdicts(run)[name]
    assert verdict == 'PROBLEM' and word in text


def test_a_key_missing_from_the_results_gives_a_verdict_and_no_traceback(run, capsys):
    point = next(iter(json.loads((sdir(run) / 'system.json').read_text())['document']['pes']
                      ['data']))
    edit_document(run, lambda document: document['pes']['data'][point]['output']
                  .pop('energy_per_atom'))
    verdict, text = verdicts(run)['pes']
    assert verdict == 'PROBLEM' and f'pes/{point}: keys missing' in text and 'energy_per_atom' in text
    assert ic.main(['report', str(run)]) == 1
    out = capsys.readouterr()
    assert 'Traceback' not in out.err and 'energy_per_atom' in out.out
    assert ' adhesion ' in out.out and ' shear cell ' in out.out  # the other verdicts are there


def test_an_entry_of_a_shape_no_check_expected_stops_that_check_only(run):
    edit_document(run, lambda document: document['data'].__setitem__('bot_slab', 5))
    found = verdicts(run)
    assert found['adhesion'][0] == 'PROBLEM'
    assert 'the check stopped on a malformed entry or file: TypeError' in found['adhesion'][1]
    assert found['shear cell'][0] == 'ok' and found['charge densities'][0] == 'ok'


def test_a_cell_off_by_rounding_is_a_note(run):
    def change(document):
        document['structure']['init']['interface']['lattice']['matrix'][0][0] += 1e-5

    edit_document(run, change)
    assert verdicts(run)['shear cell'][0] == 'note'


def test_spin_that_differs_among_the_inputs(run):
    path = next((sdir(run) / 'pes').glob('*/input.pwi'))
    path.write_text(re.sub(r'nspin(\s+)= 2', r'nspin\1= 1', path.read_text(), count=1))
    verdict, text = verdicts(run)['spin']
    assert verdict == 'PROBLEM' and 'nspin differs' in text


def test_no_spin_is_a_note(run):
    for path in sdir(run).glob('*/*/input.pwi'):
        text = re.sub(r'nspin(\s+)= 2', r'nspin\1= 1', path.read_text(), count=1)
        path.write_text(re.sub(r'\s+starting_magnetization\(\d\) = [\d.]+', '', text))
    assert verdicts(run)['spin'][0] == 'note'


def test_a_run_that_never_started(run):
    (run / 'orchestrator.json').unlink()
    found = verdicts(run)
    assert found['steps'][0] == 'not observed'
    assert found['plotpes job'][0] == 'not observed'
    assert found['shear job'][0] == 'not observed'


@pytest.mark.parametrize('status, verdict', [('running', 'not observed'), ('manual', 'not observed'),
                                             ('failed', 'PROBLEM'), ('blocked', 'PROBLEM')])
def test_a_step_that_is_not_completed(run, status, verdict):
    edit_json(run / 'orchestrator.json',
              lambda data: data['systems'][CUCU]['charge'].update(status=status))
    assert verdicts(run)['steps'][0] == verdict


def test_a_slab_file_that_is_not_in_the_copy(run):
    (run / 'slabs' / 'Cu111.json').unlink()
    found = verdicts(run)
    assert found['slabs'][0] == 'PROBLEM' and 'not read' in found['slabs'][1]


# --- the accounting of the jobs ------------------------------------------------------------

@pytest.mark.parametrize('text, nbytes', [
    ('4821964K', 4821964 * 1024), ('12G', 12 * 1024 ** 3), ('8000Mn', 8000 * 1024 ** 2),
    ('1.5G', int(1.5 * 1024 ** 3)), ('', 0), ('none', 0), ('512', 512)])
def test_memory_of_sacct(text, nbytes):
    assert ic.memory_bytes(text) == nbytes


def test_sacct_by_job_with_the_memory_of_its_steps(tmp_path):
    path = sacct_file(tmp_path / 'sacct.txt', {'1004': ('COMPLETED', '2097152K')})
    path.write_text(path.read_text() + '1004.0|python||0:0|00:01:00||8G|4194304K|node01\n')
    jobs = ic.read_sacct(path)
    assert set(jobs) == {'1004'}
    assert jobs['1004']['State'] == 'COMPLETED' and jobs['1004']['Elapsed'] == '00:01:30'
    assert jobs['1004']['MaxRSS'] == 4194304 * 1024 and jobs['1004']['ReqMem'] == '8G'


def test_the_memory_and_the_walltime_of_the_jobs_when_sacct_is_given(run, tmp_path):
    ids = {j: ('COMPLETED', '2097152K') for step in ('pes', 'pes_scf', 'plotpes', 'plotpes_scf',
                                                     'shear', 'adhesion', 'ppes', 'charge')
           for j in job_ids(run, step)}
    found = verdicts(run, sacct_file(tmp_path / 'sacct.txt', ids))
    assert found['accounting'][0] == 'ok' and 'largest MaxRSS 2.00 GiB' in found['accounting'][1]
    for subject in ('plotpes job', 'shear job'):
        assert found[subject][0] == 'ok'
        assert 'MaxRSS 2.00 GiB of 8G' in found[subject][1]
        assert 'elapsed 00:01:30 of 01:00:00' in found[subject][1]


def test_an_analysis_job_that_ran_out_of_memory(run, tmp_path):
    (job,) = job_ids(run, 'shear')
    found = verdicts(run, sacct_file(tmp_path / 'sacct.txt', {job: ('OUT_OF_MEMORY', '8388608K')}))
    assert found['accounting'][0] == 'PROBLEM' and 'OUT_OF_MEMORY' in found['accounting'][1]


def test_jobs_missing_from_sacct_are_a_note(run, tmp_path):
    (job,) = job_ids(run, 'shear')
    found = verdicts(run, sacct_file(tmp_path / 'sacct.txt', {job: ('COMPLETED', '1024K')}))
    assert found['accounting'][0] == 'note' and '1 of ' in found['accounting'][1]


# --- the command line ----------------------------------------------------------------------

def test_a_folder_that_is_not_a_run(tmp_path, capsys):
    assert ic.main(['report', str(tmp_path)]) == 1
    assert 'Error:' in capsys.readouterr().err
