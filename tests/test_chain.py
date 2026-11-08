# Author: Vito Foderà
# Created: 2026-11-08

"""Tests of the chain of steps that a step reads from (``workflows.interface.chain``).

The steps of the interface workflow write their entries with the content of
the entries they read; they are done only if the steps they read from are done
too. The run is the example one, matched, with its two PES analysed on
synthetic energies.
"""

import shutil

import pytest

from cli_helpers import SYSTEM
from test_adhesion import matched, surfaces  # noqa: F401 (fixtures)
from test_pes import edit_config
from triboworkflow.core.store import PrerequisiteError
from triboworkflow.workflows.interface import adhesion, chain, pes, plotpes
from triboworkflow.workflows.interface.match import MatchStep


@pytest.fixture
def run_dir(surfaces, tmp_path):  # noqa: F811
    copied = tmp_path / 'run'
    shutil.copytree(surfaces, copied)
    return copied


def names(steps):
    return [step.name for step in steps]


@pytest.mark.parametrize('step, upstream', [
    (MatchStep, []),
    (pes.RelaxedPesStep, ['match']),
    (pes.StaticPesStep, ['match']),
    (plotpes.RelaxedPlotPesStep, ['pes', 'match']),
    (plotpes.StaticPlotPesStep, ['pes_scf', 'match']),
    (adhesion.AdhesionStep, ['pes', 'match']),
])
def test_upstream_steps_follow_the_graph(run_dir, step, upstream):
    assert names(chain.upstream_steps(step(), run_dir)) == upstream


def test_upstream_of_the_adhesion_is_the_surface_of_the_settings(run_dir):
    edit_config(run_dir, 'pes = "pes"', 'pes = "pes_scf"')
    assert names(chain.upstream_steps(adhesion.AdhesionStep(), run_dir)) == ['pes_scf', 'match']


def test_everything_upstream_done(run_dir):
    for step in (pes.RelaxedPesStep(), adhesion.AdhesionStep(), plotpes.StaticPlotPesStep()):
        assert chain.upstream_done(step, run_dir, SYSTEM)
        assert chain.not_done_upstream(step, run_dir, SYSTEM) == []
        chain.check_upstream(step, run_dir, SYSTEM)


def test_match_not_current_is_reported_first(run_dir):
    edit_config(run_dir, 'max_area = 160.0', 'max_area = 150.0')
    assert not chain.upstream_done(adhesion.AdhesionStep(), run_dir, SYSTEM)
    # The surface is not done either, since match is not; the farthest comes first, once.
    assert chain.not_done_upstream(adhesion.AdhesionStep(), run_dir, SYSTEM) == ['match', 'pes']
    assert chain.not_done_upstream(pes.StaticPesStep(), run_dir, SYSTEM) == ['match']
    with pytest.raises(PrerequisiteError) as error:
        chain.check_upstream(adhesion.AdhesionStep(), run_dir, SYSTEM)
    assert str(error.value) == (f'{SYSTEM}: steps upstream of adhesion not done or not current: '
                                'match, pes; run them first')


def test_only_the_surface_not_current(run_dir):
    edit_config(run_dir, 'short = false', 'short = true')
    assert chain.not_done_upstream(plotpes.StaticPlotPesStep(), run_dir, SYSTEM) == ['pes_scf']
    # The relaxed surface has no table of its own: it stays done.
    assert chain.not_done_upstream(plotpes.RelaxedPlotPesStep(), run_dir, SYSTEM) == []
    assert chain.upstream_done(adhesion.AdhesionStep(), run_dir, SYSTEM)


def test_the_step_itself_is_not_asked(run_dir):
    edit_config(run_dir, 'short = false', 'short = true')
    # The static surface is not done, but the steps upstream of it are.
    assert not pes.StaticPesStep().done(run_dir, SYSTEM)
    assert chain.not_done_upstream(pes.StaticPesStep(), run_dir, SYSTEM) == []
    chain.check_upstream(pes.StaticPesStep(), run_dir, SYSTEM)
