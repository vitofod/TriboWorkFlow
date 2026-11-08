# Author: Vito Foderà
# Created: 2026-10-30

"""Tests of the opening of a run by the command line (``triboworkflow.cli.run``).

A run is the folder of its ``config.toml``; the system is the interface of
its two slabs; the number of jobs per group comes from the command line,
then from the settings, then from a question in a terminal, else an error;
the steps come from the registry, with only ``match``, ``pes`` and ``pes_scf``
in this version.
"""

import io
import shutil

import pytest

from cli_helpers import EXAMPLE, ROOT, SYSTEM, fake_steps, write_run  # noqa: F401 (fixture)
from triboworkflow.cli.run import (CONFIG_FILE, RunError, build_orchestrator, daemon_factory,
                                   load_run, resolve_groups, system_name, terminal_ask)
from triboworkflow.workflows.interface.registry import StepNotAvailable, get_step


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
    assert system_name(load_run(write_run(tmp_path))) == SYSTEM == 'mp-30_mp-134'


@pytest.mark.parametrize('given, toml, answer, expected', [
    (4, 10, None, 4), (None, 10, None, 10), (None, None, 3, 3),
])
def test_groups_resolution(tmp_path, caplog, given, toml, answer, expected):
    settings = load_run(write_run(tmp_path, groups=toml))
    ask = (lambda: answer) if answer is not None else None
    with caplog.at_level('INFO'):
        assert resolve_groups(settings, given, ask) == expected
    origin = {4: '--groups', 10: '[execution].groups', 3: 'terminal'}[expected]
    assert f'groups: {expected}' in caplog.text and origin in caplog.text


def test_groups_missing_without_terminal_is_an_error(tmp_path):
    settings = load_run(write_run(tmp_path, groups=None))
    ask = terminal_ask(io.StringIO('5\n'), io.StringIO())
    assert ask is None
    with pytest.raises(RunError, match=r'groups not given: set \[execution\]\.groups or pass '
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
    with pytest.raises(StepNotAvailable, match="step 'shear' not available in this version"):
        get_step('shear')
    with pytest.raises(StepNotAvailable, match="step 'shear' not available"):
        build_orchestrator(write_run(tmp_path, steps=('match', 'pes', 'plotpes', 'shear')))


def test_steps_in_the_registry(tmp_path):
    from triboworkflow.workflows.interface.match import MatchStep
    from triboworkflow.workflows.interface.pes import RelaxedPesStep, StaticPesStep
    from triboworkflow.workflows.interface.plotpes import RelaxedPlotPesStep, StaticPlotPesStep
    names = ('match', 'pes', 'pes_scf', 'plotpes', 'plotpes_scf')
    assert {name: type(get_step(name)) for name in names} == {
        'match': MatchStep, 'pes': RelaxedPesStep, 'pes_scf': StaticPesStep,
        'plotpes': RelaxedPlotPesStep, 'plotpes_scf': StaticPlotPesStep}
    # Only the interpolations are left to the user.
    assert [name for name in names if get_step(name).manual_analysis] == ['plotpes',
                                                                          'plotpes_scf']
    assert build_orchestrator(write_run(tmp_path, steps=('match',))).order == ['match']
    run_dir = write_run(tmp_path, steps=('pes_scf', 'pes', 'match'), name='pes')
    assert build_orchestrator(run_dir).order == ['match', 'pes_scf', 'pes']
    run_dir = write_run(tmp_path, steps=('plotpes_scf', 'plotpes', 'pes', 'pes_scf', 'match'),
                        name='plotpes')
    assert build_orchestrator(run_dir).order == ['match', 'pes', 'plotpes', 'pes_scf',
                                                 'plotpes_scf']


def test_daemon_factory_takes_the_groups_of_start(tmp_path, fake_steps):
    run_dir = write_run(tmp_path, groups=None)
    assert daemon_factory(run_dir, '7').execution.groups == 7
    with pytest.raises(RunError, match='groups not given'):  # no terminal in the background
        daemon_factory(run_dir)


def test_orchestrator_of_the_run(tmp_path, fake_steps):
    run_dir = write_run(tmp_path, steps=('pes', 'match'), groups=None)
    orch = build_orchestrator(run_dir, groups=3)
    assert orch.systems == [SYSTEM] and orch.order == ['match', 'pes']
    assert orch.execution.groups == 3 and orch.jobscript == run_dir / 'jobscript.sh'
    assert orch.manager.scheduler.dry_run is False
    assert build_orchestrator(run_dir).execution.groups is None
    extra = build_orchestrator(run_dir, steps=['match', 'pes', 'ppes'], dry_run=True)
    assert extra.order == ['match', 'pes', 'ppes'] and extra.manager.scheduler.dry_run is True
