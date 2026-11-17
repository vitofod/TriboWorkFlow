# Author: Vito Foderà
# Created: 2026-10-30

"""Tests of the opening of a run by the command line (``triboworkflow.cli.run``).

A run is the folder of its ``config.toml``; the system is the interface of
its two slabs; the number of jobs per group comes from the command line,
then from the settings, then from a question in a terminal, else an error;
the steps come from the registry, which has every step of the graph.
"""

import io
import re
import shutil

import pytest

from cli_helpers import EXAMPLE, ROOT, SYSTEM, fake_steps, write_run  # noqa: F401 (fixture)
from triboworkflow.cli.run import (CONFIG_FILE, GROUPS_DEFAULT, GROUPS_OVERRIDE, RunError,
                                   build_orchestrator, daemon_factory, load_run, resolve_groups,
                                   system_name, terminal_ask)
from triboworkflow.core.settings.schema import INTERFACE_STEPS
from triboworkflow.execution.orchestrator import INTERFACE_GRAPH
from triboworkflow.workflows.interface.registry import STEPS, StepNotAvailable, get_step


class Terminal(io.StringIO):
    """A stream that says it is a terminal."""

    def isatty(self):
        return True


def test_missing_config_is_an_error(tmp_path):
    with pytest.raises(RunError, match=f'{CONFIG_FILE} not found.*-r/--run'):
        load_run(tmp_path)


def test_structure_settings_have_no_steps(tmp_path):
    shutil.copy(ROOT / 'examples' / 'single_input.toml', tmp_path / CONFIG_FILE)
    with pytest.raises(RunError, match=r'\[structure\].*\[interface\]'):
        load_run(tmp_path)


def test_system_is_the_interface_of_the_two_slabs(tmp_path):
    assert system_name(load_run(write_run(tmp_path))) == SYSTEM == 'Cu111_Al111'


def test_groups_resolution(tmp_path, caplog):
    never = lambda: pytest.fail('nothing to ask')  # noqa: E731
    steps = ('match', 'pes', 'ppes')
    settings = load_run(write_run(tmp_path, steps=steps, groups=10))
    with caplog.at_level('INFO'):
        # --groups holds for every step, over their sections.
        assert resolve_groups(settings, 4, never) == (4, True)
        assert 'groups: 4, from --groups' in caplog.text
        caplog.clear()
        # The settings give every step: pes its section, ppes [execution].
        assert resolve_groups(settings, None, never) == (None, False)
        assert 'groups: pes 6 (from [execution.pes]), ppes 10 (from [execution])' in caplog.text
        caplog.clear()
        # ppes has none: the answer takes the place of [execution].groups.
        settings = load_run(write_run(tmp_path, steps=steps, groups=None, name='none'))
        assert resolve_groups(settings, None, lambda: 3) == (3, False)
        assert 'groups: 3, from the answer in the terminal, for pes, ppes' in caplog.text
        # Only the steps the command writes groups for.
        assert resolve_groups(settings, None, None, steps=['match']) == (None, False)


def test_groups_missing_without_terminal_is_an_error(tmp_path):
    settings = load_run(write_run(tmp_path, groups=None))
    ask = terminal_ask(io.StringIO('5\n'), io.StringIO())
    assert ask is None
    with pytest.raises(RunError, match=r'groups not given for pes: set groups in their sections '
                                       r'\[execution.<step>\] or in \[execution\], or pass '
                                       '--groups'):
        resolve_groups(settings, None, ask)


def test_terminal_question_until_a_positive_integer():
    stdout = Terminal()
    ask = terminal_ask(Terminal('ten\n0\n5\n'), stdout)
    assert ask() == 5
    assert stdout.getvalue().count('Number of jobs (workers) per group:') == 3


def test_terminal_without_answer_is_an_error():
    with pytest.raises(RunError, match='no answer'):
        terminal_ask(Terminal(''), Terminal())()


def test_step_not_in_the_registry(tmp_path):
    # Every step of the graph is available: a name that the registry lacks is refused before
    # anything is built.
    with pytest.raises(StepNotAvailable, match="step 'bulk' not available in this version"):
        get_step('bulk')
    run_dir = write_run(tmp_path, steps=('match',))
    with pytest.raises(StepNotAvailable, match="step 'bulk' not available"):
        build_orchestrator(run_dir, steps=('match', 'bulk'))


def test_every_step_of_the_graph_is_in_the_registry():
    assert set(STEPS) == set(INTERFACE_STEPS) == set(INTERFACE_GRAPH)
    assert all(get_step(name).name == name for name in INTERFACE_STEPS)


def test_steps_in_the_registry(tmp_path):
    from triboworkflow.workflows.interface.adhesion import AdhesionStep
    from triboworkflow.workflows.interface.charge import ChargeStep
    from triboworkflow.workflows.interface.match import MatchStep
    from triboworkflow.workflows.interface.pes import RelaxedPesStep, StaticPesStep
    from triboworkflow.workflows.interface.plotpes import RelaxedPlotPesStep, StaticPlotPesStep
    from triboworkflow.workflows.interface.ppes import PpesStep
    from triboworkflow.workflows.interface.shear import ShearStep
    names = ('match', 'pes', 'pes_scf', 'plotpes', 'plotpes_scf', 'shear', 'adhesion', 'ppes',
             'charge')
    assert {name: type(get_step(name)) for name in names} == {
        'match': MatchStep, 'pes': RelaxedPesStep, 'pes_scf': StaticPesStep,
        'plotpes': RelaxedPlotPesStep, 'plotpes_scf': StaticPlotPesStep, 'shear': ShearStep,
        'adhesion': AdhesionStep, 'ppes': PpesStep, 'charge': ChargeStep}
    # Only the interpolations and the shear strength are left to the user.
    assert [name for name in names if get_step(name).manual_analysis] == [
        'plotpes', 'plotpes_scf', 'shear']
    assert build_orchestrator(write_run(tmp_path, steps=('match',))).order == ['match']
    run_dir = write_run(tmp_path, steps=('pes_scf', 'pes', 'match'), name='pes')
    assert build_orchestrator(run_dir).order == ['match', 'pes_scf', 'pes']
    run_dir = write_run(tmp_path, steps=('plotpes_scf', 'plotpes', 'pes', 'pes_scf', 'match'),
                        name='plotpes')
    assert build_orchestrator(run_dir).order == ['match', 'pes', 'plotpes', 'pes_scf',
                                                 'plotpes_scf']


def test_graph_of_the_run_follows_the_surface_of_the_adhesion(tmp_path, fake_steps):
    run_dir = write_run(tmp_path, steps=('adhesion', 'pes_scf', 'pes', 'match'))
    orch = build_orchestrator(run_dir)
    assert orch.graph['adhesion'] == ('pes', 'match')
    assert orch.order == ['match', 'pes_scf', 'pes', 'adhesion']
    config = run_dir / CONFIG_FILE
    config.write_text(re.sub(r'^pes = "pes"', 'pes = "pes_scf"', config.read_text(encoding='utf-8'),
                             count=1, flags=re.M), encoding='utf-8')
    orch = build_orchestrator(run_dir)
    assert load_run(run_dir).interface.adhesion.pes == 'pes_scf'
    assert orch.graph['adhesion'] == ('pes_scf', 'match')
    assert orch.order == ['match', 'pes_scf', 'adhesion', 'pes']


def test_daemon_factory_takes_the_groups_of_start(tmp_path, fake_steps):
    run_dir = write_run(tmp_path, groups=None)
    assert daemon_factory(run_dir, '7').execution.groups == 7
    with pytest.raises(RunError, match='groups not given'):  # no terminal in the background
        daemon_factory(run_dir)
    # --groups holds for every step; an answer only for those without groups of their own.
    run_dir = write_run(tmp_path, steps=('match', 'pes', 'ppes'), groups=None, name='kinds')
    config = run_dir / CONFIG_FILE
    config.write_text(config.read_text().replace('[execution.pes]\n',
                                                 '[execution.pes]\ngroups = 6\n'))
    override = daemon_factory(run_dir, '7', GROUPS_OVERRIDE).execution
    assert (override.groups_of('pes'), override.groups_of('ppes')) == (7, 7)
    answer = daemon_factory(run_dir, '7', GROUPS_DEFAULT).execution
    assert (answer.groups_of('pes'), answer.groups_of('ppes')) == (6, 7)
    assert daemon_factory(write_run(tmp_path, name='given')).execution.groups_of('pes') == 6


def test_orchestrator_of_the_run(tmp_path, fake_steps):
    run_dir = write_run(tmp_path, steps=('pes', 'match'), groups=None)
    orch = build_orchestrator(run_dir, groups=3)
    assert orch.systems == [SYSTEM] and orch.order == ['match', 'pes']
    assert orch.execution.groups == 3 and orch.jobscript == run_dir / 'jobscript.sh'
    assert orch.manager.scheduler.dry_run is False
    assert build_orchestrator(run_dir).execution.groups is None
    extra = build_orchestrator(run_dir, steps=['match', 'pes', 'ppes'], dry_run=True)
    assert extra.order == ['match', 'pes', 'ppes'] and extra.manager.scheduler.dry_run is True
