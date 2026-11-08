# Author: Vito Foderà
# Created: 2026-11-07

"""Tests of the steps plotpes and plotpes_scf (``triboworkflow.workflows.interface.plotpes``).

The interpolation of the example interface Cu(111)/Al(111) has 108 864 points
and does not fit in memory here: the tests use the example run with two
Cu(111) slabs, whose PES has 8 points in the cell (1 152 replicated), matched
once and with the two PES analysed on synthetic energies.

Notes
-----
To remove: the golden cases ``plotpes_step_*`` (``tests/golden_cases.py``)
compare the steps with TribChem's ``FT_PESInterpolation``.
"""

import contextlib
import json
import logging
import os
import shutil
import socket
import subprocess
import sys

import numpy as np
import pytest

from cli_helpers import fake_scheduler_bin, fake_slurm  # noqa: F401 (fixtures)
from test_pes import edit_config, finish, pes_run
from triboworkflow.cli.analysis_job import run_analysis
from triboworkflow.cli.main import main
from triboworkflow.cli.run import build_orchestrator, load_run
from triboworkflow.core.store import PrerequisiteError, StoreError, SystemWriter, read_system
from triboworkflow.execution import layout
from triboworkflow.execution.jobs import ANALYSIS_FILE, Analysis, write_analysis
from triboworkflow.execution.lock import RunLock
from triboworkflow.workflows.interface.registry import get_step
from triboworkflow.vendored.physics.base.solidstate import get_cell_info
from triboworkflow.vendored.physics.base.units import Converter
from triboworkflow.vendored.physics.tribology import potensurf
from triboworkflow.workflows.interface import pes, plotpes
from triboworkflow.workflows.interface.match import MatchStep

CUCU = 'mp-30_mp-30'
STEPS = {'plotpes': plotpes.RelaxedPlotPesStep, 'plotpes_scf': plotpes.StaticPlotPesStep}
PES_STEPS = {'pes': pes.RelaxedPesStep, 'pes_scf': pes.StaticPesStep}
PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def cucu_run(tmp_path, name='run'):
    """The example run with Cu(111) as both slabs, and every step up to the interpolations."""
    run_dir = pes_run(tmp_path, name=name,
                      steps=('match', 'pes', 'pes_scf', 'plotpes', 'plotpes_scf'))
    edit_config(run_dir, 'file = "slabs/Al111.json"\nmid = "mp-134"',
                'file = "slabs/Cu111.json"\nmid = "mp-30"')
    return run_dir


@pytest.fixture(scope='module')
def pes_done(tmp_path_factory):
    """The Cu/Cu run matched, with the two PES analysed: run folder."""
    run_dir = cucu_run(tmp_path_factory.mktemp('plotpes'))
    MatchStep().analyse(run_dir, CUCU, [])
    for name, step in PES_STEPS.items():
        # Energies of the static PES other than those of the relaxed one.
        inputs = pes.pes_inputs(run_dir, CUCU, load_run(run_dir), name)
        energies = {point: -1000.0 - (0.01 if name == 'pes' else 0.02) * ((i * 3) % 5)
                    for i, (point, _, _) in enumerate(inputs.points)}
        step().analyse(run_dir, CUCU, finish(run_dir, name, energies, system=CUCU))
    return run_dir


@pytest.fixture
def pes_copy(pes_done, tmp_path):
    copied = tmp_path / 'copy'
    shutil.copytree(pes_done, copied)
    return copied


@pytest.fixture(scope='module')
def interpolated(pes_done, tmp_path_factory):
    """The run with the two PES interpolated."""
    run_dir = tmp_path_factory.mktemp('interpolated') / 'run'
    shutil.copytree(pes_done, run_dir)
    for step in STEPS.values():
        step().analyse(run_dir, CUCU, [])
    return run_dir


@pytest.fixture
def interpolated_copy(interpolated, tmp_path):
    copied = tmp_path / 'interpolated'
    shutil.copytree(interpolated, copied)
    return copied


def sdir(run_dir):
    return layout.system_dir(run_dir, CUCU)


def document(run_dir):
    return read_system(sdir(run_dir)).document


def system_bytes(run_dir):
    return (sdir(run_dir) / 'system.json').read_bytes()


def figures(run_dir):
    return sorted(str(path.relative_to(sdir(run_dir))) for path in sdir(run_dir).rglob('*.png'))


def png_size(path):
    """Width and height of a PNG, from its header."""
    data = path.read_bytes()
    assert data[:8] == PNG_SIGNATURE
    return int.from_bytes(data[16:20], 'big'), int.from_bytes(data[20:24], 'big')


def count_get_pes(monkeypatch):
    """Record the calls to get_pes of the steps, which still run."""
    calls = []

    def spy(hs_all, E, cell, **kwargs):
        calls.append({'E': [list(row) for row in E], **kwargs})
        return potensurf.get_pes(hs_all, E, cell, **kwargs)

    monkeypatch.setattr(plotpes, 'get_pes', spy)
    return calls


# --- the two steps ------------------------------------------------------------------------

def test_two_steps_each_on_its_pes():
    assert plotpes.PES_OF == {'plotpes': 'pes', 'plotpes_scf': 'pes_scf'}
    assert {name: step.name for name, step in STEPS.items()} == {name: name for name in STEPS}
    assert plotpes.interpolation_leaves('pes_scf') == [
        ('pes_scf', 'energy', 'grid', 'hs_all_ev'), ('pes_scf', 'energy', 'grid', 'hs_all_jm2'),
        ('pes_scf', 'energy', 'pes_ev'), ('pes_scf', 'energy', 'pes_jm2')]


@pytest.mark.parametrize('name', ['plotpes', 'plotpes_scf'])
def test_expected_entries(pes_done, name):
    pes_name = plotpes.PES_OF[name]
    expected = STEPS[name]().expected(pes_done, CUCU)
    assert set(expected) == {('highsym',), (pes_name, 'data')}
    assert expected[(pes_name, 'data')] == PES_STEPS[pes_name]().provenance(pes_done, CUCU)


@pytest.mark.parametrize('name', ['plotpes', 'plotpes_scf'])
def test_inputs_read_from_the_document(pes_done, name):
    doc = document(pes_done)
    inputs = plotpes.plotpes_inputs(sdir(pes_done), read_system(sdir(pes_done)),
                                    load_run(pes_done), name)
    pes_name = plotpes.PES_OF[name]
    assert (inputs.pes, inputs.calc_type) == (pes_name, pes.CALC_TYPES[pes_name])
    assert inputs.all_shifts == doc['highsym']['inter_all']
    assert inputs.ids == {key: doc[key] for key in ('mid', 'formula', 'name', 'miller')}
    assert inputs.title == doc['name'] == 'Cu111-Cu111'
    ev, jm2 = inputs.values
    # The points sorted by the energy of the PES, ties in the order of the points.
    energies = {s: doc[pes_name]['data'][s]['output']['energy']
                for s in doc['highsym']['inter_unique']}
    assert [row[0] for row in ev] == sorted(energies, key=energies.get)
    assert [row[3] for row in ev] == sorted(energies.values())
    assert [row[1:3] for row in ev] == [doc['highsym']['inter_unique'][row[0]][0] for row in ev]
    # J/m^2 on the area of the cell of the final structure of the minimum.
    minimum = doc[pes_name]['data'][ev[0][0]]['output']['structure']['lattice']['matrix']
    area, _ = get_cell_info(minimum)
    assert [row[3] for row in jm2] == [Converter.eV_to_Jm2(row[3], area * 1e-20) for row in ev]
    np.testing.assert_array_equal(inputs.cell, np.array(minimum))


@pytest.mark.parametrize('ids, title', [
    ({'name': 'Cu111-Al111', 'formula': 'Cu_Al', 'miller': [[1, 1, 1], [1, 1, 1]]}, 'Cu111-Al111'),
    ({'formula': 'Cu_Al', 'miller': [[1, 1, 1], [1, 0, 0]]}, 'Cu111Al100'),
])
def test_name_of_the_interface(ids, title):
    assert plotpes.interface_name(ids) == title


@pytest.mark.parametrize('ids, message', [
    ({'mid': 'mp-30_mp-134'}, 'neither a name nor a formula'),
    ({'name': 'Cu/Al'}, 'cannot name a file'),
])
def test_name_of_the_interface_errors(ids, message):
    with pytest.raises(StoreError, match=message):
        plotpes.interface_name(ids)


# --- analysis -----------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['plotpes', 'plotpes_scf'])
def test_entries_written(interpolated, name):
    doc = document(interpolated)
    pes_name = plotpes.PES_OF[name]
    energy = doc[pes_name]['energy']
    inter_all = doc['highsym']['inter_all']
    ev_points = {row[0]: row[3] for row in energy['grid']['hs_unique_ev']}
    jm2_points = {row[0]: row[3] for row in energy['grid']['hs_unique_jm2']}
    # Every replica of every point, with the energy of its point.
    for key, points in (('hs_all_ev', ev_points), ('hs_all_jm2', jm2_points)):
        rows = energy['grid'][key]
        assert len(rows) == sum(len(v) for v in inter_all.values())
        assert all(row[3] == points[row[0]] for row in rows)
        assert sorted(tuple(row[1:3]) for row in rows) == sorted(
            tuple(xy) for v in inter_all.values() for xy in v)
    # [x, y, energy] with the minimum at zero.
    for key in ('pes_ev', 'pes_jm2'):
        data = np.array(energy[key])
        assert data.shape[1] == 3 and data[:, 2].min() == 0.0
    # The figure is a file, not a key of the document (DEV-42).
    assert 'image' not in energy
    # The entries of the PES step are as it wrote them, and still current.
    assert {'hs_unique_ev', 'hs_unique_jm2', 'hs_all_ev', 'hs_all_jm2'} == set(energy['grid'])
    assert PES_STEPS[pes_name]().done(interpolated, CUCU)
    assert STEPS[name]().done(interpolated, CUCU)
    assert MatchStep().done(interpolated, CUCU)


@pytest.mark.parametrize('name', ['plotpes', 'plotpes_scf'])
def test_same_values_as_get_pes(interpolated, name):
    """The entries are what get_pes gives on the points of arrange_data."""
    inputs = plotpes.plotpes_inputs(sdir(interpolated), read_system(sdir(interpolated)),
                                    load_run(interpolated), name)
    energy = document(interpolated)[inputs.pes]['energy']
    for values, unit in zip(inputs.values, ('ev', 'jm2')):
        _, v_list, data, _ = potensurf.get_pes(inputs.all_shifts, values, inputs.cell,
                                               title=inputs.title, to_fig=None)
        assert energy['grid'][f'hs_all_{unit}'] == json.loads(json.dumps(v_list))
        assert energy[f'pes_{unit}'] == data.tolist()


@pytest.mark.parametrize('name', ['plotpes', 'plotpes_scf'])
def test_provenance_of_the_entries(interpolated, name):
    document_ = read_system(sdir(interpolated))
    leaves = plotpes.interpolation_leaves(plotpes.PES_OF[name])
    recorded = [document_.provenance(leaf) for leaf in leaves]
    assert recorded[0] == STEPS[name]().provenance(interpolated, CUCU)
    assert all(provenance == recorded[0] for provenance in recorded)
    assert recorded[0].step == name
    assert recorded[0].parameters == {'calc_type': pes.CALC_TYPES[plotpes.PES_OF[name]],
                                      'pes_type': 'energy'}


def test_two_interpolations_eV_then_J_m2_with_the_figure(pes_copy, monkeypatch):
    calls = count_get_pes(monkeypatch)
    plotpes.RelaxedPlotPesStep().analyse(pes_copy, CUCU, [])
    inputs = plotpes.plotpes_inputs(sdir(pes_copy), read_system(sdir(pes_copy)),
                                    load_run(pes_copy), 'plotpes')
    assert [call['E'] for call in calls] == [list(map(list, v)) for v in inputs.values]
    assert [call['title'] for call in calls] == ['Cu111-Cu111'] * 2
    assert calls[0]['to_fig'] is None
    assert calls[1]['to_fig'].endswith('.partial.png')


def test_figures_of_the_two_steps(interpolated):
    assert figures(interpolated) == ['plotpes/pes_Cu111-Cu111.png',
                                     'plotpes_scf/pes_Cu111-Cu111.png']
    first, second = (sdir(interpolated) / name / 'pes_Cu111-Cu111.png' for name in STEPS)
    # 7 x 7 inches at 300 dpi; the two PES differ, so do the figures.
    assert png_size(first) == png_size(second) == (2100, 2100)
    assert first.read_bytes() != second.read_bytes()


def test_figure_closed_once_saved(pes_copy):
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    plotpes.StaticPlotPesStep().analyse(pes_copy, CUCU, [])
    assert plt.get_fignums() == before


def test_two_interpolations_in_one_document(interpolated):
    doc = document(interpolated)
    assert doc['pes']['energy']['pes_ev'] != doc['pes_scf']['energy']['pes_ev']
    assert all(step().done(interpolated, CUCU) for step in STEPS.values())


def test_message_of_the_analysis(pes_copy, caplog):
    with caplog.at_level(logging.INFO, logger=plotpes.__name__):
        plotpes.StaticPlotPesStep().analyse(pes_copy, CUCU, [])
    lines = [record.getMessage() for record in caplog.records if record.name == plotpes.__name__]
    assert lines[0].startswith('plotpes_scf: interpolation in eV: ')
    assert lines[1].startswith('plotpes_scf: interpolation in J/m^2 and figure: ')
    assert lines[2] == (f'{CUCU}: plotpes_scf: PES pes_scf (slab_from_scratch) interpolated: '
                        '4 high-symmetry points, 8 in the cell, 8 after removing the duplicates')
    assert lines[3].startswith(f'{CUCU}: interpolation in eV ')
    assert lines[4] == f'{CUCU}: figure written in plotpes_scf/pes_Cu111-Cu111.png'


def test_analysis_again_gives_the_same_document(interpolated_copy):
    before = document(interpolated_copy)
    plotpes.RelaxedPlotPesStep().analyse(interpolated_copy, CUCU, [])
    assert document(interpolated_copy) == before
    assert all(step().done(interpolated_copy, CUCU) for step in STEPS.values())


# --- done ---------------------------------------------------------------------------------

def test_done_only_after_the_analysis(pes_copy):
    assert not any(step().done(pes_copy, CUCU) for step in STEPS.values())
    plotpes.RelaxedPlotPesStep().analyse(pes_copy, CUCU, [])
    assert plotpes.RelaxedPlotPesStep().done(pes_copy, CUCU)
    assert not plotpes.StaticPlotPesStep().done(pes_copy, CUCU)


def test_done_needs_the_figure(interpolated_copy):
    (sdir(interpolated_copy) / 'plotpes' / 'pes_Cu111-Cu111.png').unlink()
    assert not plotpes.RelaxedPlotPesStep().done(interpolated_copy, CUCU)
    assert plotpes.StaticPlotPesStep().done(interpolated_copy, CUCU)


def test_done_without_calling_get_pes(interpolated_copy, monkeypatch):
    calls = count_get_pes(monkeypatch)
    assert all(step().done(interpolated_copy, CUCU) for step in STEPS.values())
    assert calls == []


def test_done_follows_the_entries_read(interpolated_copy):
    with SystemWriter(sdir(interpolated_copy)) as writer:
        inter_all = writer.document.get(['highsym', 'inter_all'])
        point = next(iter(inter_all))
        provenance = writer.document.provenance(['highsym'])
        writer.document.set(['highsym', 'inter_all', point],
                            [[x + 0.1, y] for x, y in inter_all[point]])
        writer.document.set_provenance(['highsym'], provenance)
        writer.save()
    # highsym keeps a provenance of match, but the points read changed.
    assert not plotpes.RelaxedPlotPesStep().done(interpolated_copy, CUCU)
    assert not plotpes.StaticPlotPesStep().done(interpolated_copy, CUCU)


def test_pes_analysed_again_removes_its_interpolation(interpolated_copy):
    calc_ids = [f'pes/{point}' for point in document(interpolated_copy)['highsym']['inter_unique']]
    pes.RelaxedPesStep().analyse(interpolated_copy, CUCU, calc_ids)
    doc = document(interpolated_copy)
    assert not {'hs_all_ev', 'hs_all_jm2'} & set(doc['pes']['energy']['grid'])
    assert 'pes_ev' not in doc['pes']['energy']
    assert not plotpes.RelaxedPlotPesStep().done(interpolated_copy, CUCU)
    assert plotpes.StaticPlotPesStep().done(interpolated_copy, CUCU)
    assert pes.RelaxedPesStep().done(interpolated_copy, CUCU)


# --- errors -------------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['plotpes', 'plotpes_scf'])
def test_without_the_pes_the_step_stops(tmp_path, name):
    run_dir = cucu_run(tmp_path)
    MatchStep().analyse(run_dir, CUCU, [])
    before = system_bytes(run_dir)
    with pytest.raises(PrerequisiteError, match=rf'{plotpes.PES_OF[name]}\.data'):
        STEPS[name]().analyse(run_dir, CUCU, [])
    assert system_bytes(run_dir) == before
    assert figures(run_dir) == []
    assert not STEPS[name]().done(run_dir, CUCU)


def test_failure_of_the_interpolation_writes_nothing(pes_copy, monkeypatch):
    def fail(hs_all, E, cell, to_fig=None, **kwargs):
        if to_fig is not None:
            potensurf.get_pes(hs_all, E, cell, to_fig=to_fig, **kwargs)
            raise MemoryError('no memory left')
        return potensurf.get_pes(hs_all, E, cell, to_fig=to_fig, **kwargs)

    monkeypatch.setattr(plotpes, 'get_pes', fail)
    before = system_bytes(pes_copy)
    with pytest.raises(MemoryError):
        plotpes.RelaxedPlotPesStep().analyse(pes_copy, CUCU, [])
    # The document is as it was, the temporary figure removed, nothing locked.
    assert system_bytes(pes_copy) == before
    assert figures(pes_copy) == []
    assert not list((sdir(pes_copy) / 'plotpes').iterdir())
    assert not SystemWriter(sdir(pes_copy)).lock_path.exists()
    assert not plotpes.RelaxedPlotPesStep().done(pes_copy, CUCU)


def test_entries_changed_during_the_interpolation(pes_copy, monkeypatch):
    interpolate = plotpes.interpolate

    def change(inputs, figure):
        result = interpolate(inputs, figure)
        # Another analysis of the PES in the meantime, with other energies.
        points = document(pes_copy)['highsym']['inter_unique']
        calc_ids = finish(pes_copy, 'pes', {p: -1000.0 - 0.03 * i for i, p in enumerate(points)},
                          prepare=False, system=CUCU)
        pes.RelaxedPesStep().analyse(pes_copy, CUCU, calc_ids)
        return result

    monkeypatch.setattr(plotpes, 'interpolate', change)
    with pytest.raises(StoreError, match='the entries read changed during the interpolation'):
        plotpes.RelaxedPlotPesStep().analyse(pes_copy, CUCU, [])
    assert 'pes_ev' not in document(pes_copy)['pes']['energy']
    assert figures(pes_copy) == []
    assert not plotpes.RelaxedPlotPesStep().done(pes_copy, CUCU)


# --- the analysis by hand: orchestrator and command line ------------------------------------

ALL_STEPS = ('match', 'pes', 'pes_scf', 'plotpes', 'plotpes_scf')


def statuses(orch):
    return {name: orch.entry(CUCU, name)['status'] for name in orch.order}


def cli(run_dir, *argv):
    return main([*argv, '-r', str(run_dir)])


def run_files(run_dir):
    """Every file of a run, with its content."""
    return {str(p.relative_to(run_dir)): p.read_bytes() for p in sorted(run_dir.rglob('*'))
            if p.is_file()}


def no_pes_analysis(monkeypatch):
    """Record the preparations and analyses of the PES steps, which no longer run."""
    calls = []
    monkeypatch.setattr(pes.PesStep, 'prepare', lambda self, *args: calls.append(self.name) or [])
    monkeypatch.setattr(pes.PesStep, 'analyse', lambda self, *args: calls.append(self.name))
    return calls


def test_the_passes_never_interpolate(pes_copy, fake_slurm, monkeypatch):
    calls = count_get_pes(monkeypatch)
    orch = build_orchestrator(pes_copy)
    report = orch.run_pass()
    assert statuses(orch) == {'match': 'completed', 'pes': 'completed', 'pes_scf': 'completed',
                              'plotpes': 'manual', 'plotpes_scf': 'manual'}
    for name in STEPS:
        assert orch.entry(CUCU, name)['reason'] == (
            f'to analyse by hand: run "triboworkflow analyse {name}" where there is enough '
            'memory, or give [execution.analysis] to analyse it in a job of its own')
    assert report.finished and not report.success
    assert calls == [] and figures(pes_copy) == []


def test_analysis_by_hand_leaves_the_pes_completed(pes_copy, fake_slurm, monkeypatch):
    # The interpolations write into the entries of the PES: the PES stay
    # completed and current, and the passes neither run them again nor end
    # with a step completed whose entries are not current.
    calls = count_get_pes(monkeypatch)
    pes_runs = no_pes_analysis(monkeypatch)
    build_orchestrator(pes_copy).run_pass()
    for name in STEPS:  # each as the command analyse builds it
        assert build_orchestrator(pes_copy).analyse_step(CUCU, name).status == 'completed'
    assert len(calls) == 4
    orch = build_orchestrator(pes_copy)
    for _ in range(2):
        report = orch.run_pass()
        assert report.success and statuses(orch) == dict.fromkeys(ALL_STEPS, 'completed')
    assert all(get_step(name).done(pes_copy, CUCU) for name in ALL_STEPS)
    assert len(calls) == 4 and pes_runs == []


@pytest.mark.parametrize('name', ['plotpes', 'plotpes_scf'])
def test_analyse_by_hand_from_the_command_line(pes_copy, fake_slurm, fake_scheduler_bin,
                                               monkeypatch, capsys, name):
    calls = count_get_pes(monkeypatch)
    assert cli(pes_copy, 'prepare', name) == 0  # left to the user, not an error
    out = capsys.readouterr().out
    assert f'{CUCU}/{name}: manual (to analyse by hand: run "triboworkflow analyse {name}"' in out
    assert cli(pes_copy, 'submit', name) == 1
    assert f'it is analysed by hand, with "analyse {name}"' in capsys.readouterr().err
    assert calls == [] and figures(pes_copy) == []
    assert cli(pes_copy, 'analyse', name) == 0
    assert f'{CUCU}/{name}: completed' in capsys.readouterr().out
    assert len(calls) == 2 and figures(pes_copy) == [f'{name}/pes_Cu111-Cu111.png']
    # Again, and its state: recognised from the files, nothing interpolated.
    before = run_files(pes_copy)
    assert cli(pes_copy, 'analyse', name) == 0
    assert cli(pes_copy, 'status') == 0
    out = capsys.readouterr().out
    assert f'{CUCU}/{name}: completed' in out and f'  {name:<11} completed ' in out
    assert len(calls) == 2
    after = run_files(pes_copy)
    assert {path: data for path, data in after.items() if path != 'cli.log'} == \
        {path: data for path, data in before.items() if path != 'cli.log'}


def dead_pid():
    """A pid of this host whose process has ended."""
    process = subprocess.Popen([sys.executable, '-c', 'pass'])
    process.wait()
    return process.pid


@pytest.mark.parametrize('holder', ['orchestrator', 'another host'])
def test_analyse_refused_while_the_run_is_held(pes_copy, fake_slurm, monkeypatch, capsys,
                                              holder):
    calls = count_get_pes(monkeypatch)
    lock = pes_copy / 'run.lock'
    if holder == 'another host':  # a job of the analysis killed on a node
        lock.write_text(json.dumps({'pid': dead_pid(), 'host': 'node07', 'token': 'aa',
                                    'since': '2026-11-07T10:00:00+01:00',
                                    'command': 'triboworkflow analyse plotpes'}) + '\n')
        context = contextlib.nullcontext()
    else:
        context = RunLock(pes_copy, 'triboworkflow start')
    with context:
        before = run_files(pes_copy)
        assert cli(pes_copy, 'analyse', 'plotpes') == 1
        after = run_files(pes_copy)
    err = capsys.readouterr().err
    if holder == 'another host':
        assert ('the run is in use by triboworkflow analyse plotpes (pid ' in err
                and 'on node07' in err and f'remove {lock} by hand only after checking that no '
                'command of the run runs there' in err)
    else:
        assert 'the run is in use by triboworkflow start (pid ' in err
        assert 'the orchestrator in the background ends with "triboworkflow stop"' in err
    assert after == before and calls == []


class Stopped(BaseException):
    """The process stopped during the interpolation (a signal, the walltime)."""


def test_an_analysis_stopped_is_done_again_alone(pes_copy, fake_slurm, monkeypatch, capsys):
    calls = count_get_pes(monkeypatch)
    assert cli(pes_copy, 'analyse', 'plotpes') == 0
    done = system_bytes(pes_copy)
    interpolate = plotpes.interpolate

    def stop(inputs, figure):
        figure.write_bytes(PNG_SIGNATURE)  # the figure begun
        raise Stopped

    monkeypatch.setattr(plotpes, 'interpolate', stop)
    with pytest.raises(Stopped):
        cli(pes_copy, 'analyse', 'plotpes_scf')
    monkeypatch.setattr(plotpes, 'interpolate', interpolate)
    state = json.loads((pes_copy / 'orchestrator.json').read_text())['systems'][CUCU]
    assert (state['plotpes_scf']['status'], state['plotpes_scf']['calculations']) == \
        ('running', [])
    assert system_bytes(pes_copy) == done and len(calls) == 2
    assert figures(pes_copy) == ['plotpes/pes_Cu111-Cu111.png']
    assert not plotpes.StaticPlotPesStep().done(pes_copy, CUCU)
    # A pass: the interpolation done is kept, the one stopped left to the user.
    orch = build_orchestrator(pes_copy)
    orch.run_pass()
    assert statuses(orch)['plotpes'] == 'completed' and statuses(orch)['plotpes_scf'] == 'manual'
    assert len(calls) == 2
    # What a process killed leaves: its lock, on this host, and the figure begun.
    (pes_copy / 'run.lock').write_text(json.dumps(
        {'pid': dead_pid(), 'host': socket.gethostname(), 'token': 'bb',
         'since': '2026-11-07T10:00:00+01:00',
         'command': 'triboworkflow analyse plotpes_scf'}) + '\n')
    partial = plotpes.partial_path(plotpes.figure_path(sdir(pes_copy), 'plotpes_scf',
                                                       'Cu111-Cu111'))
    partial.write_bytes(PNG_SIGNATURE)
    assert not plotpes.StaticPlotPesStep().done(pes_copy, CUCU)
    # The next analysis replaces that lock and starts from the beginning.
    assert cli(pes_copy, 'analyse', 'plotpes_scf') == 0
    assert 'no longer runs on' in capsys.readouterr().err
    assert len(calls) == 4 and not partial.exists() and not (pes_copy / 'run.lock').exists()
    assert figures(pes_copy) == ['plotpes/pes_Cu111-Cu111.png', 'plotpes_scf/pes_Cu111-Cu111.png']
    assert all(step().done(pes_copy, CUCU) for step in STEPS.values())


# --- the analysis in a job of its own -------------------------------------------------------

def analysis_group(run_dir, name, k=1):
    """The group of the analysis of ``name``, as the orchestrator writes it: its analysis.json."""
    jobscript = run_dir / 'analysis.sh'
    jobscript.write_text('#!/bin/bash\n#SBATCH --mem=8G\n')
    group = Analysis(name=f'{CUCU}.{name}.{k}', system=CUCU, step=name, scheduler='slurm')
    return write_analysis(run_dir, group, jobscript) / ANALYSIS_FILE


@pytest.mark.parametrize('name', ['plotpes', 'plotpes_scf'])
def test_the_job_interpolates_while_the_orchestrator_holds_the_run(pes_copy, interpolated,
                                                                   monkeypatch, name):
    calls = count_get_pes(monkeypatch)
    path = analysis_group(pes_copy, name)
    with RunLock(pes_copy, 'triboworkflow start'):  # the orchestrator in the background
        outcome = run_analysis(path)
    assert (outcome['result'], outcome['error']) == ('analysed', None)
    assert len(calls) == 2 and figures(pes_copy) == [f'{name}/pes_Cu111-Cu111.png']
    assert get_step(name).done(pes_copy, CUCU)
    # The entries of the analysis by hand, with their provenance.
    for leaf in plotpes.interpolation_leaves(plotpes.PES_OF[name]):
        found, expected = read_system(sdir(pes_copy)), read_system(sdir(interpolated))
        assert found.get(leaf) == expected.get(leaf)
        assert found.provenance(leaf) == expected.provenance(leaf)
    assert not (pes_copy / 'run.lock').exists()
    # A job again (e.g. a late one): recognised from the files, nothing interpolated.
    before = system_bytes(pes_copy)
    assert run_analysis(path)['result'] == 'done'
    assert len(calls) == 2 and system_bytes(pes_copy) == before


def test_the_jobscript_of_the_group_runs_the_analysis(pes_copy):
    # The jobscript as the scheduler runs it on the node, in a process of its own.
    path = analysis_group(pes_copy, 'plotpes')
    env = {**os.environ, 'SLURM_JOB_ID': '5151'}
    done = subprocess.run(['bash', str(path.parent / 'jobscript.sh')], cwd=path.parent, env=env,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr
    outcome = json.loads((path.parent / 'outcome.json').read_text())
    assert (outcome['result'], outcome['job_id']) == ('analysed', '5151')
    assert plotpes.RelaxedPlotPesStep().done(pes_copy, CUCU)
    assert 'interpolation in eV' in done.stderr  # the log of the job


def test_the_job_writes_nothing_when_the_pes_changes(pes_copy, monkeypatch):
    interpolate = plotpes.interpolate

    def change(inputs, figure):
        result = interpolate(inputs, figure)
        points = document(pes_copy)['highsym']['inter_unique']
        calc_ids = finish(pes_copy, 'pes', {p: -1000.0 - 0.03 * i for i, p in enumerate(points)},
                          prepare=False, system=CUCU)
        pes.RelaxedPesStep().analyse(pes_copy, CUCU, calc_ids)
        return result

    monkeypatch.setattr(plotpes, 'interpolate', change)
    outcome = run_analysis(analysis_group(pes_copy, 'plotpes'))
    assert outcome['result'] == 'failed'
    assert outcome['error'].startswith('StoreError: ') and \
        'the entries read changed during the interpolation' in outcome['error']
    assert figures(pes_copy) == [] and not plotpes.RelaxedPlotPesStep().done(pes_copy, CUCU)


def with_analysis_job(run_dir):
    """[execution.analysis] in the settings of the run, with its jobscript."""
    (run_dir / 'analysis.sh').write_text('#!/bin/bash\n#SBATCH --mem=8G\n')
    with open(run_dir / 'config.toml', 'a', encoding='utf-8') as f:
        f.write('\n[execution.analysis]\njobscript = "analysis.sh"\n')


def the_node_runs(run_dir, slurm, group, state='COMPLETED'):
    """The job of a group on a compute node: the line of its jobscript, then its end."""
    path = layout.job_dir(run_dir, group)
    outcome = run_analysis(path / ANALYSIS_FILE)
    for job_id in json.loads((path / 'submitted.json').read_text())['job_ids']:
        slurm.states[job_id] = state
    return outcome


def test_a_run_interpolates_in_jobs_of_their_own(pes_copy, fake_slurm, monkeypatch):
    calls = count_get_pes(monkeypatch)
    with_analysis_job(pes_copy)
    orch = build_orchestrator(pes_copy)
    report = orch.run_pass()
    first, second = f'{CUCU}.plotpes.1', f'{CUCU}.plotpes_scf.1'
    assert statuses(orch) == {'match': 'completed', 'pes': 'completed', 'pes_scf': 'completed',
                              'plotpes': 'running', 'plotpes_scf': 'waiting'}
    assert orch.entry(CUCU, 'plotpes_scf')['reason'] == 'ready; plotpes runs first'
    assert report.submitted == [first] and fake_slurm.submitted == 1 and not report.finished
    assert calls == []  # the orchestrator never interpolates
    assert the_node_runs(pes_copy, fake_slurm, first)['result'] == 'analysed'
    report = orch.run_pass()
    assert statuses(orch)['plotpes'] == 'completed' and statuses(orch)['plotpes_scf'] == 'running'
    assert report.submitted == [second] and len(calls) == 2
    assert the_node_runs(pes_copy, fake_slurm, second)['result'] == 'analysed'
    for _ in range(2):
        report = orch.run_pass()
        assert report.success and statuses(orch) == dict.fromkeys(ALL_STEPS, 'completed')
    assert len(calls) == 4 and fake_slurm.submitted == 2
    assert figures(pes_copy) == ['plotpes/pes_Cu111-Cu111.png', 'plotpes_scf/pes_Cu111-Cu111.png']


def test_the_job_of_the_analysis_from_the_command_line(pes_copy, fake_slurm, fake_scheduler_bin,
                                                       monkeypatch, capsys):
    calls = count_get_pes(monkeypatch)
    with_analysis_job(pes_copy)
    group = f'{CUCU}.plotpes.1'
    assert cli(pes_copy, 'prepare', 'plotpes') == 0
    assert (f'{CUCU}/plotpes: waiting (ready: nothing to prepare; its analysis runs in a job of '
            'its own') in capsys.readouterr().out
    assert cli(pes_copy, 'submit', 'plotpes') == 0
    out = capsys.readouterr().out
    assert f'{CUCU}/plotpes: running (analysis in a job of its own, group {group})' in out
    assert f'  job of the analysis: group {group}, folder jobs/{group}' in out
    assert f'group {group}: 1 jobs submitted' in out and 'again to follow its job' in out
    # While its job may run, the analysis by hand is refused and changes nothing.
    before = run_files(pes_copy)
    assert cli(pes_copy, 'analyse', 'plotpes') == 1
    assert (f'{CUCU}/plotpes not analysed by hand: its analysis runs in a job of its own (group '
            f'{group}, job ') in capsys.readouterr().err
    assert {p: d for p, d in run_files(pes_copy).items() if p != 'cli.log'} == \
        {p: d for p, d in before.items() if p != 'cli.log'}
    assert cli(pes_copy, 'status') == 0
    assert f'job of the analysis: jobs/{group}' in capsys.readouterr().out
    assert calls == []
    the_node_runs(pes_copy, fake_slurm, group)
    assert cli(pes_copy, 'submit', 'plotpes') == 0
    out = capsys.readouterr().out
    assert f'{CUCU}/plotpes: completed (analysed in the job of group {group})' in out
    assert 'no group written' in out and len(calls) == 2 and fake_slurm.submitted == 1


def test_the_job_without_the_pes(tmp_path):
    run_dir = cucu_run(tmp_path)
    MatchStep().analyse(run_dir, CUCU, [])
    before = system_bytes(run_dir)
    outcome = run_analysis(analysis_group(run_dir, 'plotpes_scf'))
    assert outcome['result'] == 'blocked' and 'pes_scf.data' in outcome['error']
    assert system_bytes(run_dir) == before and figures(run_dir) == []
