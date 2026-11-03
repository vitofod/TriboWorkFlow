# Author: Vito Foderà
# Created: 2026-11-02

"""Tests of the preparation of the calculations and of their replacement.

``triboworkflow.engines.qe.prepare``, ``execution/preparation.py`` and
``execution/replacement.py``, and the replacement through the orchestrator.
The calculations are pw.x calculations of the workflows, made from the
settings of ``tests/data/qe/example_relax.toml`` (illustrative values, an
expected checksum given) and a small carbon slab. No scheduler is called and
pw.x never runs: jobs and workers are simulated as in ``run_helpers``.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Kpoints

from cli_helpers import SYSTEM as CLI_SYSTEM
from cli_helpers import write_run
from run_helpers import SCF_DONE, SYSTEM, Run
from triboworkflow.cli.main import main
from triboworkflow.core import store
from triboworkflow.core import structures as st
from triboworkflow.core.settings import load_settings
from triboworkflow.core.settings.schema import ExecutionParams, QEParams
from triboworkflow.engines.qe.prepare import input_differences, single_input
from triboworkflow.execution import layout
from triboworkflow.execution import orchestrator as orchestrator_module
from triboworkflow.execution import preparation
from triboworkflow.execution.archive import archive_attempt
from triboworkflow.execution.jobs import WITHDRAWN_FILE, Launch, Manifest, write_job
from triboworkflow.execution.orchestrator import Orchestrator, OrchestratorError, Step
from triboworkflow.execution.preparation import (PreparationError, ReplacementError,
                                                 check_calculations, prepare_calculations,
                                                 workflow_calculation)
from triboworkflow.execution.reconcile import LaunchError, launch_for, read_groups
from triboworkflow.execution.replacement import retired_launches, unfinished
from triboworkflow.workflows.interface import registry

DATA = Path(__file__).resolve().parent / 'data' / 'qe'
UPF = 'C.pbe-n-kjpaw_psl.1.0.0.UPF'
MD5 = '0123456789abcdef0123456789abcdef'
EXECUTION = ExecutionParams(scheduler='slurm', jobscript='job.sh', run_command='pw.x', groups=2,
                            walltime=3600, polling_interval=300, max_seconds_margin=60)
SLAB = Structure(Lattice.hexagonal(2.46, 20.0), ['C', 'C'],
                 [[0, 0, 0.5], [1 / 3, 2 / 3, 0.5]])
KPOINTS = Kpoints.gamma_automatic((4, 4, 1))
LABELS = ('aa', 'ab')
AA, AB = 'pes_scf/aa', 'pes_scf/ab'
GROUP = f'{SYSTEM}.pes.1'


def example_settings(tmp_path):
    """Settings of the example with the expected version and checksum, as in a workflow.

    The keys that the workflows set for each calculation are left out.
    """
    settings = load_settings(DATA / 'example_relax.toml')
    for key in ('calculation', 'title', 'prefix', 'restart_mode', 'outdir'):
        del settings.namelists.CONTROL[key]
    manifest = tmp_path / 'pseudo.md5'
    manifest.write_text(f'{MD5}  {UPF}\n')
    settings.qe = QEParams('7.2', str(manifest))
    return settings


def calcs(settings, labels=LABELS, calc_type='slab_from_scratch', structure=SLAB, system=SYSTEM):
    return [workflow_calculation(system, f'pes_scf/{label}', calc_type, structure, KPOINTS,
                                 settings) for label in labels]


class PesStep(Step):
    """The calculations of LABELS, made from ``settings`` when asked."""

    name = 'pes'

    def __init__(self, settings, labels=LABELS):
        self.settings, self.labels = settings, labels

    def calculations(self, run_dir, system):
        return calcs(self.settings, self.labels, system=system)


@pytest.fixture
def run(tmp_path):
    return Run(tmp_path)


@pytest.fixture
def settings(tmp_path):
    return example_settings(tmp_path)


def orchestrator(run, step):
    return Orchestrator(run.dir, [SYSTEM], {'pes': step}, run.manager, EXECUTION, run.jobscript)


def kinds(run, settings, labels=LABELS):
    return {c.calc_id: c.kind for c in check_calculations(run.dir, SYSTEM, calcs(settings, labels))}


def first_attempt(run, taken=(AA,)):
    """A group listing aa and ab, submitted; its workers run ``taken``; its jobs end."""
    jobs = run.group(GROUP, Launch.initial(SYSTEM, AA), Launch.initial(SYSTEM, AB))
    for calc_id in taken:
        run.start(calc_id, jobs[0], output=SCF_DONE)
        run.finish(calc_id, status=store.Status('ended', 'converged', 'n/a'),
                   stop_reason='completed', clean_stop=True)
        archive_attempt(run.dir, SYSTEM, calc_id)
    run.poll(**{job_id: 'COMPLETED' for job_id in jobs})
    assert run.manager.group_complete(GROUP)
    return jobs


def change_ecutwfc(settings):
    settings.namelists.SYSTEM['ecutwfc'] = 50.0


# --- the calculation prepared -----------------------------------------------------------

def test_input_differences():
    kept = "&CONTROL\n   max_seconds      = 300\n   tprnfor          = .true.\n/\n" \
           "&ELECTRONS\n   mixing_beta      = 0.3\n/\nK_POINTS gamma\n"
    assert input_differences(kept, kept).kind == 'same'
    new = kept.replace('= 300', '= 3600').replace('0.3', '0.4')
    difference = input_differences(kept, new)
    assert difference.kind == 'attempt'
    assert difference.messages() == ('namelists.CONTROL.max_seconds: requested 3600, kept 300',
                                     'namelists.ELECTRONS.mixing_beta: requested 0.4, kept 0.3')
    added = kept.replace('/\nK_POINTS', '   electron_maxstep = 200\n/\nK_POINTS')
    assert input_differences(kept, added).messages() == \
        ('namelists.ELECTRONS.electron_maxstep: requested 200, kept absent',)
    assert input_differences(kept, kept.replace('.true.', '.false.')).kind == 'other'
    assert input_differences(kept, kept.replace('gamma', 'automatic')).kind == 'other'


def test_new_calculations_written(run, settings):
    calculations = calcs(settings)
    assert prepare_calculations(run.dir, SYSTEM, calculations, step='pes') == [AA, AB]
    for calc in calculations:
        path = run.path(calc.calc_id)
        assert (path / layout.INPUT_FILE).read_text(encoding='utf-8') == calc.input_text
        assert (path / layout.STRUCTURE_IN_FILE).read_text(encoding='utf-8') == \
            calc.structure_text
        record = run.record(calc.calc_id)
        assert record.attempts == () and record.status == store.Status()
        assert record.fingerprint == calc.record.fingerprint
        assert record.files == {'input': 'input.pwi', 'structure': 'structure_in.extxyz'}
        assert store.unknown_parts(record.energy_model) == []
    assert "prefix           = 'pes_scf_aa'" in calculations[0].input_text
    assert "outdir           = 'OUT'" in calculations[0].input_text
    assert kinds(run, settings) == {AA: 'current', AB: 'current'}


@pytest.mark.parametrize('calc_type, flags', [
    ('slab_from_scratch', [[1, 1, 1]] * 2),
    ('interface_z_relax', [[0, 0, 1]] * 2),
])
def test_extxyz_is_the_picture_of_the_input(run, settings, calc_type, flags):
    """Cell, positions, labels and constraints of the extxyz are those of input.pwi."""
    structure = SLAB.copy()
    if calc_type == 'interface_z_relax':  # as the vendored VaspInputSet sets them
        structure.add_site_property('selective_dynamics', [[False, False, True]] * 2)
    calc = calcs(settings, ('aa',), calc_type, structure)[0]
    prepare_calculations(run.dir, SYSTEM, [calc])
    path = run.path(AA)
    pwi = st.read_structure(path / layout.INPUT_FILE, 'pwi')
    extxyz = st.read_structure(path / layout.STRUCTURE_IN_FILE)
    np.testing.assert_allclose(extxyz.lattice.matrix, pwi.lattice.matrix, rtol=0, atol=1e-12)
    np.testing.assert_allclose(extxyz.cart_coords, pwi.cart_coords, rtol=0, atol=1e-8)
    assert extxyz.labels == pwi.labels == ['C', 'C']
    assert extxyz.site_properties['if_pos'] == flags
    assert [f or [1, 1, 1] for f in pwi.site_properties.get('if_pos', [None] * 2)] == flags


def test_changes_of_the_settings_logged(run, settings, caplog):
    with caplog.at_level(logging.WARNING, logger='triboworkflow'):
        prepare_calculations(run.dir, SYSTEM, calcs(settings, ('aa',)))
    lines = [r.getMessage() for r in caplog.records]
    assert any(line.startswith(f'{SYSTEM}/{AA}: namelists.CONTROL.nstep = 150 written as 1')
               for line in lines)
    assert any(line.startswith(f'{SYSTEM}/{AA}: namelists.IONS (') for line in lines)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='triboworkflow'):
        prepare_calculations(run.dir, SYSTEM, calcs(settings, ('aa',)))
    assert caplog.records == []  # logged once, when the input is written


def test_current_calculation_left_as_it_is(run, settings):
    prepare_calculations(run.dir, SYSTEM, calcs(settings))
    run.start(AA, '1000', output=SCF_DONE)
    before = run.files(AA), run.record(AA)
    prepare_calculations(run.dir, SYSTEM, calcs(settings))
    assert (run.files(AA), run.record(AA)) == before


def test_interrupted_preparation_written_again(run, settings):
    path = run.path(AA)
    path.mkdir(parents=True)
    (path / layout.INPUT_FILE).write_text('half written')
    (path / '.record.json.0123456789abcdef.tmp').write_text('{')
    assert kinds(run, settings, ('aa',)) == {AA: 'interrupted'}
    prepare_calculations(run.dir, SYSTEM, calcs(settings, ('aa',)))
    assert kinds(run, settings, ('aa',)) == {AA: 'current'}


def test_attempt_and_resource_keys_kept_with_their_values(run, settings, caplog):
    prepare_calculations(run.dir, SYSTEM, calcs(settings))
    before = run.files(AA)
    settings.namelists.CONTROL['max_seconds'] = 3600
    settings.namelists.ELECTRONS['mixing_beta'] = 0.5
    checks = check_calculations(run.dir, SYSTEM, calcs(settings))
    assert [c.kind for c in checks] == ['kept', 'kept']
    assert checks[0].warnings == (
        'input.pwi: namelists.CONTROL.max_seconds: requested 3600, kept 84600',
        'input.pwi: namelists.ELECTRONS.mixing_beta: requested 0.5, kept 0.3')
    with caplog.at_level(logging.WARNING, logger='triboworkflow'):
        prepare_calculations(run.dir, SYSTEM, calcs(settings), step='pes')
    assert run.files(AA) == before
    text = caplog.text
    assert 'max_seconds: requested 3600, kept 84600' in text
    assert '"triboworkflow prepare pes --replace"' in text


def test_other_differences_stop_and_write_nothing(run, settings):
    prepare_calculations(run.dir, SYSTEM, calcs(settings))
    change_ecutwfc(settings)
    with pytest.raises(PreparationError) as error:
        prepare_calculations(run.dir, SYSTEM, calcs(settings, LABELS + ('ac',)), step='pes')
    message = str(error.value)
    assert message.startswith(f'2 calculations of {SYSTEM} differ from the settings')
    assert f'{AA}: fingerprint differs from the settings' in message
    assert '"triboworkflow prepare pes --replace"' in message
    assert [c.kind for c in error.value.checks] == ['changed', 'changed', 'new']
    assert not run.path('pes_scf/ac').exists()


@pytest.mark.parametrize('edit, reason', [
    (lambda p: (p / 'input.pwi').write_text((p / 'input.pwi').read_text().replace(
        'ecutwfc          = 45', 'ecutwfc          = 50')), 'beyond the keys of the attempts'),
    (lambda p: (p / 'input.pwi').unlink(), 'input.pwi missing'),
    (lambda p: (p / 'structure_in.extxyz').unlink(), 'structure_in.extxyz missing'),
    (lambda p: (p / 'structure_in.extxyz').write_text('broken'), 'structure_in.extxyz not read'),
    (lambda p: (p / 'record.json').write_text('{}'), 'record.json not read'),
    (lambda p: (p / 'record.json').unlink() or (p / 'claim').write_text(''),
     'no record.json, and the directory holds claim'),
])
def test_calculation_changed_on_disk(run, settings, edit, reason):
    prepare_calculations(run.dir, SYSTEM, calcs(settings, ('aa',)))
    edit(run.path(AA))
    check = check_calculations(run.dir, SYSTEM, calcs(settings, ('aa',)))[0]
    assert check.kind == 'changed' and reason in '; '.join(check.reasons)


def test_extxyz_that_no_longer_matches_the_input(run, settings):
    prepare_calculations(run.dir, SYSTEM, calcs(settings, ('aa',)))
    path = run.path(AA) / layout.STRUCTURE_IN_FILE
    moved = st.read_structure(path)
    moved.translate_sites([0], [0, 0, 0.5], frac_coords=False)
    path.write_text(st.extxyz_text(moved))
    check = check_calculations(run.dir, SYSTEM, calcs(settings, ('aa',)))[0]
    assert check.kind == 'changed'
    assert check.reasons == ('structure_in.extxyz does not match the input: positions differ',)


def test_extxyz_differing_only_in_data_not_in_the_input_is_kept(run, settings):
    with_moments = SLAB.copy()
    with_moments.add_site_property('magmom', [0.5, 0.5])
    prepare_calculations(run.dir, SYSTEM, calcs(settings, ('aa',), structure=with_moments))
    check = check_calculations(run.dir, SYSTEM, calcs(settings, ('aa',)))[0]
    assert check.kind == 'kept'
    assert check.warnings == ('structure_in.extxyz: magmom differ from the structure of the '
                              'settings, not written in the input',)


def test_calculations_given_twice_or_of_another_system(run, settings):
    with pytest.raises(PreparationError, match='given twice'):
        prepare_calculations(run.dir, SYSTEM, calcs(settings, ('aa', 'aa')))
    other = workflow_calculation('Ag111_Au111', AA, 'slab_from_scratch', SLAB, KPOINTS, settings)
    with pytest.raises(PreparationError, match='its record is of Ag111_Au111'):
        prepare_calculations(run.dir, SYSTEM, [other])


# --- replacement --------------------------------------------------------------------------

def test_replace_moves_the_calculations_and_prepares_them_again(run, settings, caplog):
    step = PesStep(settings)
    orch = orchestrator(run, step)
    orch.prepare_step(SYSTEM, 'pes')
    first_attempt(run)
    old = {calc_id: (run.files(calc_id), run.record(calc_id)) for calc_id in (AA, AB)}
    change_ecutwfc(settings)
    with caplog.at_level(logging.WARNING, logger='triboworkflow'):
        orch.prepare_step(SYSTEM, 'pes')  # already prepared: the differences are logged
    assert f'{SYSTEM}/{AA}: prepared, differs from the settings' in caplog.text
    report, replacement = orch.replace_step(SYSTEM, 'pes', 'triboworkflow prepare pes --replace')
    assert report.status == 'running' and replacement.done
    assert replacement.path == run.dir / 'replaced' / SYSTEM / 'pes' / '1'
    journal = json.loads((replacement.path / 'replacement.json').read_text())
    assert journal['command'] == 'triboworkflow prepare pes --replace'
    assert journal['withdrawn'] == []
    assert journal['calculations'][AA]['reasons'] == ['fingerprint differs from the settings']
    assert {calc_id: entry['retired'] for calc_id, entry in journal['calculations'].items()} == \
        {AA: {GROUP: True}, AB: {GROUP: False}}
    for calc_id in (AA, AB):  # nothing lost: the old directories as they were
        kept = replacement.old_dir(calc_id)
        assert {p.name: p.read_bytes() for p in sorted(kept.iterdir())
                if p.is_file() and p.name != 'record.json'} == old[calc_id][0]
        assert json.loads((kept / 'record.json').read_text()) == old[calc_id][1].to_dict()
        record = run.record(calc_id)
        assert record.attempts == () and record.fingerprint != old[calc_id][1].fingerprint
    assert (replacement.old_dir(AA) / 'attempts' / '1' / 'output.pwo').read_text() == SCF_DONE
    assert len(json.loads((replacement.old_dir(AA) / 'record.json').read_text())['attempts']) == 1
    # The old launches no longer count for the calculations prepared again.
    assert retired_launches(run.dir) == {(SYSTEM, AA): {GROUP: True}, (SYSTEM, AB): {GROUP: False}}
    assert run.reconcile(AA).state == run.reconcile(AB).state == 'new'
    with pytest.raises(LaunchError, match='no group lists attempt 1'):
        launch_for(run.dir, SYSTEM, AA, 1)
    assert orch.entry(SYSTEM, 'pes')['calculations'] == [AA, AB]


def test_prepare_and_replace_from_the_command_line(tmp_path, settings, monkeypatch, capsys):
    run_dir = write_run(tmp_path, steps=('pes',))
    step = type('Pes', (PesStep,), {'__init__': lambda self: PesStep.__init__(self, settings)})
    monkeypatch.setattr(registry, 'STEPS', {'pes': step})
    assert main(['prepare', 'pes', '-r', str(run_dir)]) == 0
    assert '2 calculations prepared' in capsys.readouterr().out
    change_ecutwfc(settings)
    assert main(['prepare', 'pes', '-r', str(run_dir)]) == 0  # prepared: only reported
    err = capsys.readouterr().err
    assert f'{CLI_SYSTEM}/{AA}: prepared, differs from the settings' in err
    assert '"triboworkflow prepare pes --replace"' in err
    assert main(['prepare', 'pes', '--replace', '-r', str(run_dir)]) == 0
    out = capsys.readouterr().out
    folder = run_dir / 'replaced' / CLI_SYSTEM / 'pes' / '1'
    assert f'2 calculations replaced: the old ones are in {folder}' in out
    assert '2 calculations prepared' in out
    journal = json.loads((folder / 'replacement.json').read_text())
    assert journal['command'] == f'triboworkflow prepare pes --replace -r {run_dir}'
    log = (run_dir / 'cli.log').read_text()
    assert f'{CLI_SYSTEM}/{AA}: directory moved to {folder / "pes_scf" / "aa"}' in log
    assert 'namelists.CONTROL.nstep = 150 written as 1' in log


def test_replace_also_the_calculations_kept(run, settings):
    orch = orchestrator(run, PesStep(settings))
    orch.prepare_step(SYSTEM, 'pes')
    settings.namelists.CONTROL['max_seconds'] = 3600
    _, replacement = orch.replace_step(SYSTEM, 'pes')
    assert set(replacement.calculations) == {AA, AB}
    assert replacement.calculations[AA]['reasons'] == \
        ['input.pwi: namelists.CONTROL.max_seconds: requested 3600, kept 84600']
    assert 'max_seconds      = 3600' in (run.path(AA) / layout.INPUT_FILE).read_text()


def test_replace_keeps_the_calculations_already_current(run, settings):
    orch = orchestrator(run, PesStep(settings))
    orch.prepare_step(SYSTEM, 'pes')
    path = run.path(AA) / layout.INPUT_FILE
    path.write_text(path.read_text().replace('mixing_beta      = 0.3', 'mixing_beta      = 0.4'))
    before = run.files(AB), run.record(AB)
    _, replacement = orch.replace_step(SYSTEM, 'pes')
    assert list(replacement.calculations) == [AA]
    assert (run.files(AB), run.record(AB)) == before
    assert 'mixing_beta      = 0.3' in path.read_text()


def test_nothing_to_replace(run, settings):
    orch = orchestrator(run, PesStep(settings))
    orch.prepare_step(SYSTEM, 'pes')
    report, replacement = orch.replace_step(SYSTEM, 'pes')
    assert replacement is None and report.status == 'running'
    assert not (run.dir / 'replaced').exists()


def test_shared_group_keeps_its_progress_for_the_other_calculations(run, settings):
    """aa was taken by the group, ab not: after aa is replaced, ab is still not
    taken (and relaunched), not a group without progress."""
    orch = orchestrator(run, PesStep(settings))
    orch.prepare_step(SYSTEM, 'pes')
    first_attempt(run)
    assert run.reconcile(AB).state == 'not_taken'
    path = run.path(AA) / layout.INPUT_FILE
    path.write_text(path.read_text().replace('mixing_beta      = 0.3', 'mixing_beta      = 0.4'))
    _, replacement = orch.replace_step(SYSTEM, 'pes')
    assert list(replacement.calculations) == [AA]
    assert run.reconcile(AB).state == 'not_taken'
    assert run.reconcile(AA).state == 'new'


@pytest.mark.parametrize('how', ['held back', 'refused', 'not registered'])
def test_group_never_submitted_is_withdrawn(run, settings, how):
    orch = orchestrator(run, PesStep(settings))
    orch.prepare_step(SYSTEM, 'pes')
    write_job(run.dir, Manifest(name=GROUP, scheduler='slurm', run_command='pw.x', walltime=3600,
                                max_seconds_margin=60,
                                calculations=(Launch.initial(SYSTEM, AA),
                                              Launch.initial(SYSTEM, AB))), run.jobscript)
    if how != 'not registered':
        run.manager.add_group(GROUP, 2)
    if how == 'refused':
        run.slurm.refuse = 1
        run.manager.cycle()
        assert run.manager.groups[GROUP]['refused'] is not None
    change_ecutwfc(settings)
    _, replacement = orch.replace_step(SYSTEM, 'pes')
    assert replacement.withdrawn == [GROUP]
    assert (layout.job_dir(run.dir, GROUP) / WITHDRAWN_FILE).is_file()
    assert read_groups(run.dir) == {}
    assert all(not entry['retired'] for entry in replacement.calculations.values())
    summary = run.manager.cycle()
    assert run.slurm.submitted == 0 and summary.withdrawn == [GROUP]


def test_replacement_refused_while_a_job_or_a_worker_may_use_the_calculations(run, settings):
    orch = orchestrator(run, PesStep(settings))
    orch.prepare_step(SYSTEM, 'pes')
    jobs = run.group(GROUP, Launch.initial(SYSTEM, AA), Launch.initial(SYSTEM, AB))
    run.poll()
    run.start(AA, jobs[0])
    change_ecutwfc(settings)
    before = {calc_id: run.files(calc_id) for calc_id in (AA, AB)}
    with pytest.raises(ReplacementError) as error:
        orch.replace_step(SYSTEM, 'pes')
    message = str(error.value)
    assert message.startswith(f'{SYSTEM}/pes: nothing replaced')
    assert f'{AA}: it has a claim' in message
    assert f'group {GROUP}: jobs not in a final state confirmed by the last query: ' \
           f'{jobs[0]} PENDING, {jobs[1]} PENDING' in message
    assert f'{AB}: group {GROUP}' in message
    assert not (run.dir / 'replaced').exists()
    assert {calc_id: run.files(calc_id) for calc_id in (AA, AB)} == before
    assert unfinished(run.dir) == []


@pytest.mark.parametrize('refused', [True, False])
def test_replacement_refused_with_a_group_submitted_in_part(run, settings, refused):
    """One job of the group over, the other never submitted: the user is told why
    nothing is replaced and how to go on."""
    orch = orchestrator(run, PesStep(settings))
    orch.prepare_step(SYSTEM, 'pes')
    jobs = run.group(GROUP, Launch.initial(SYSTEM, AA))  # one job submitted
    run.poll(**{jobs[0]: 'COMPLETED'})
    # As if a second job had been planned and refused, or held back by the orchestrator.
    run.manager.groups[GROUP]['planned'] = 2
    if refused:
        run.manager.groups[GROUP]['refused'] = 'invalid account'
    run.manager._save()
    assert run.manager.partly_submitted(GROUP) == (1, 1, refused)
    change_ecutwfc(settings)
    with pytest.raises(ReplacementError) as error:
        orch.replace_step(SYSTEM, 'pes')
    message = str(error.value)
    why = 'its last submission was refused' if refused else 'held back since the settings changed'
    assert (f'group {GROUP}: 1 of its jobs were submitted and are over, 1 were never submitted '
            f'({why}). A group submitted in part is neither complete nor can it be '
            'withdrawn') in message
    assert 'this version does not replace a calculation it lists' in message
    assert '"triboworkflow retry pes" and "triboworkflow prepare pes"' in message
    assert (f'"triboworkflow clear-refusal {GROUP}"' in message) == refused
    assert not (run.dir / 'replaced').exists()
    assert not (layout.job_dir(run.dir, GROUP) / WITHDRAWN_FILE).exists()


@pytest.mark.parametrize('case, message', [
    ('unknown', 'submissions of outcome not known'),
    ('pending', 'has no outcome recorded'),
])
def test_replacement_refused_with_a_submission_of_outcome_not_known(run, settings, case, message):
    orch = orchestrator(run, PesStep(settings))
    orch.prepare_step(SYSTEM, 'pes')
    write_job(run.dir, Manifest(name=GROUP, scheduler='slurm', run_command='pw.x', walltime=3600,
                                max_seconds_margin=60,
                                calculations=(Launch.initial(SYSTEM, AA),)), run.jobscript)
    run.manager.add_group(GROUP, 1)
    receipt = layout.job_dir(run.dir, GROUP) / 'submitted.json'
    if case == 'unknown':
        run.manager.groups[GROUP]['unknown'].append({'error': 'timed out', 'at': 'x'})
        run.manager._save()
    else:
        receipt.write_text(json.dumps({'job_ids': [], 'pending': '2026-11-02T10:00:00+01:00'}))
    change_ecutwfc(settings)
    with pytest.raises(ReplacementError, match=message):
        orch.replace_step(SYSTEM, 'pes')
    assert not (layout.job_dir(run.dir, GROUP) / WITHDRAWN_FILE).exists()


# --- a replacement interrupted ----------------------------------------------------------

def _stop(*args, **kwargs):
    raise KeyboardInterrupt


def interrupted_rename(after):
    """os.rename that stops the process after ``after`` directories moved."""
    calls = []

    def rename(source, target):
        if len(calls) == after:
            raise KeyboardInterrupt
        calls.append(source)
        os.rename(source, target)

    return rename


@pytest.mark.parametrize('where', ['after the journal', 'after the withdrawal',
                                   'after one move', 'before done'])
def test_replacement_interrupted_is_completed_without_losses(run, settings, monkeypatch, where):
    step = PesStep(settings)
    orch = orchestrator(run, step)
    orch.prepare_step(SYSTEM, 'pes')
    first_attempt(run)
    write_job(run.dir, Manifest(name=f'{SYSTEM}.pes.2', scheduler='slurm', run_command='pw.x',
                                walltime=3600, max_seconds_margin=60,
                                calculations=(Launch.initial(SYSTEM, AB),)),
              run.jobscript)
    old = {calc_id: (run.files(calc_id), run.record(calc_id)) for calc_id in (AA, AB)}
    change_ecutwfc(settings)
    with monkeypatch.context() as patch:
        if where == 'after the journal':
            patch.setattr(orchestrator_module, 'complete_replacement', _stop)
        elif where == 'after the withdrawal':
            patch.setattr(preparation.os, 'rename', interrupted_rename(0))
        elif where == 'after one move':
            patch.setattr(preparation.os, 'rename', interrupted_rename(1))
        else:
            patch.setattr(orchestrator_module, 'mark_done', _stop)
        with pytest.raises(KeyboardInterrupt):
            orch.replace_step(SYSTEM, 'pes')
    [journal] = unfinished(run.dir)
    # The old launches no longer count from the moment the journal is written.
    assert retired_launches(run.dir) == {(SYSTEM, AA): {GROUP: True}, (SYSTEM, AB): {GROUP: False}}
    # Nothing else may change the run meanwhile.
    for action in (orch.run_pass, lambda: orch.prepare_step(SYSTEM, 'pes'),
                   lambda: orch.submit_step(SYSTEM, 'pes'), lambda: orch.retry(SYSTEM, 'pes'),
                   lambda: orch.analyse_step(SYSTEM, 'pes'),
                   lambda: orch.relaunch(SYSTEM, AA), lambda: orch.release_claim(SYSTEM, AA)):
        with pytest.raises(OrchestratorError, match='"triboworkflow prepare pes --replace"'):
            action()
    with pytest.raises(PreparationError, match='not completed'):
        prepare_calculations(run.dir, SYSTEM, calcs(settings))
    restarted = orchestrator(run, PesStep(settings))
    report, done = restarted.replace_step(SYSTEM, 'pes')
    assert done.path == journal.path and done.done and unfinished(run.dir) == []
    assert report.status == 'running'
    assert done.withdrawn == [f'{SYSTEM}.pes.2']
    for calc_id in (AA, AB):
        kept = done.old_dir(calc_id)
        assert {p.name: p.read_bytes() for p in sorted(kept.iterdir())
                if p.is_file() and p.name != 'record.json'} == old[calc_id][0]
        assert json.loads((kept / 'record.json').read_text()) == old[calc_id][1].to_dict()
        assert run.record(calc_id).attempts == ()
        assert run.reconcile(calc_id).state == 'new'
    assert len(list((run.dir / 'replaced' / SYSTEM / 'pes').iterdir())) == 1


def test_preparation_interrupted_after_the_replacement(run, settings, monkeypatch):
    orch = orchestrator(run, PesStep(settings))
    orch.prepare_step(SYSTEM, 'pes')
    first_attempt(run)
    change_ecutwfc(settings)
    with monkeypatch.context() as patch:
        patch.setattr(preparation, 'write_record', _stop)
        with pytest.raises(KeyboardInterrupt):
            orch.replace_step(SYSTEM, 'pes')
    assert unfinished(run.dir) == []  # the replacement is done, the preparation is not
    assert not (run.path(AA) / 'record.json').exists()
    assert kinds(run, settings) == {AA: 'interrupted', AB: 'new'}
    report = orch.prepare_step(SYSTEM, 'pes')
    assert report.status == 'running' and kinds(run, settings) == {AA: 'current', AB: 'current'}
    assert run.reconcile(AA).state == 'new'


def copy_example(tmp_path, **changes):
    """The settings of the example and its structure file, in ``tmp_path``; ``changes``
    replace lines of the TOML (``old=new``)."""
    text = (DATA / 'example_relax.toml').read_text(encoding='utf-8')
    for old, new in changes.values():
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / 'single.toml'
    path.write_text(text, encoding='utf-8')
    (tmp_path / 'example_relax.pwi').write_bytes((DATA / 'example_relax.pwi').read_bytes())
    return path


# --- the single calculation ---------------------------------------------------------------

def test_gen_writes_the_input_and_the_extxyz_that_matches_it(tmp_path, capsys):
    path = copy_example(tmp_path)
    assert main(['gen', str(path)]) == 0
    out = capsys.readouterr().out
    pwi, extxyz = tmp_path / 'input.pwi', tmp_path / 'input.extxyz'
    assert f'Input: {pwi.resolve()}' in out and f'Structure: {extxyz.resolve()}' in out
    assert 'calculation = relax; 6 atoms, 1 species; K_POINTS automatic 8 8 1  0 0 0' in out
    assert 'constraints: 5 default, 1 file' in out
    settings = load_settings(path)
    assert pwi.read_text(encoding='utf-8') == single_input(settings).pw.text
    text = pwi.read_text(encoding='utf-8')
    assert "calculation      = 'relax'" in text and "prefix           = 'A_prefix'" in text
    a, b = st.read_structure(pwi, 'pwi'), st.read_structure(extxyz)
    np.testing.assert_allclose(b.lattice.matrix, a.lattice.matrix, rtol=0, atol=1e-12)
    np.testing.assert_allclose(b.cart_coords, a.cart_coords, rtol=0, atol=1e-8)
    assert b.labels == a.labels == ['C'] * 6
    assert b.site_properties['if_pos'] == [[1, 1, 1]] * 5 + [[0, 0, 0]]
    assert [f or [1, 1, 1] for f in a.site_properties['if_pos']] == b.site_properties['if_pos']


def test_gen_prints_the_changes_of_an_scf(tmp_path, capsys):
    path = copy_example(tmp_path, calculation=('calculation = "relax"', 'calculation = "scf"'))
    assert main(['gen', str(path)]) == 0
    err = capsys.readouterr().err
    assert 'Warning: namelists.CONTROL.nstep = 150 written as 1' in err
    assert 'Warning: namelists.IONS (ion_dynamics) not written' in err
    assert '&IONS' not in (tmp_path / 'input.pwi').read_text()


def test_gen_never_overwrites_unless_asked(tmp_path, capsys):
    path = copy_example(tmp_path)
    assert main(['gen', str(path)]) == 0
    (tmp_path / 'input.extxyz').write_text('mine')
    pwi = (tmp_path / 'input.pwi').read_bytes()
    assert main(['gen', str(path)]) == 1
    assert 'already exist and [generation].overwrite is false' in capsys.readouterr().err
    assert (tmp_path / 'input.extxyz').read_text() == 'mine'
    assert (tmp_path / 'input.pwi').read_bytes() == pwi
    path = copy_example(tmp_path, overwrite=('overwrite = false', 'overwrite = true'),
                        name=('file_prefix = "input.pwi"', 'file_prefix = "C6.in"'))
    assert main(['gen', str(path)]) == 0
    assert (tmp_path / 'C6.in').read_bytes() == pwi and (tmp_path / 'C6.extxyz').is_file()
    assert main(['gen', str(path)]) == 0  # overwrite = true


def test_gen_refuses_a_workflow_and_placeholders(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1] / 'examples'
    assert main(['gen', str(root / 'interface.toml')]) == 1
    assert 'gen needs [structure]' in capsys.readouterr().err
    template = tmp_path / 'single_input.toml'
    template.write_bytes((root / 'single_input.toml').read_bytes())
    assert main(['gen', str(template)]) == 1
    assert 'structure.file' in capsys.readouterr().err
    assert list(tmp_path.glob('input.*')) == []


def test_replacement_of_another_step_must_be_completed_first(run, settings, monkeypatch):
    orch = orchestrator(run, PesStep(settings))
    orch.prepare_step(SYSTEM, 'pes')
    change_ecutwfc(settings)
    with monkeypatch.context() as patch:
        patch.setattr(orchestrator_module, 'complete_replacement', _stop)
        with pytest.raises(KeyboardInterrupt):
            orch.replace_step(SYSTEM, 'pes')
    other = PesStep(settings)
    other.name = 'adhesion'
    orch = Orchestrator(run.dir, [SYSTEM], {'adhesion': other}, run.manager, EXECUTION,
                        run.jobscript)
    with pytest.raises(OrchestratorError, match='replacement of the calculations of '
                                                f'{SYSTEM}/pes not completed'):
        orch.replace_step(SYSTEM, 'adhesion')
