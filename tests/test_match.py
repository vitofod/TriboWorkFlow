# Author: Vito Foderà
# Created: 2026-11-05

"""Tests of the step match (``triboworkflow.workflows.interface.match``).

The golden cases ``match_step_*`` compare the database calls of the step with
those of the historic code. These tests check the step in a run: entries,
provenance, files and message; when it is done again; the deviations of the
port (the angle tolerance used unchanged, the entries emptied before a new
write, no write when the matching fails); formula and name.
"""

import json
import logging
import shutil
import tomllib

import numpy as np
import pytest

import golden_cases
from cli_helpers import EXAMPLE, SYSTEM, write_run
from triboworkflow.cli.main import main
from triboworkflow.cli.run import build_orchestrator, load_run
from triboworkflow.core.settings.loader import settings_from_dict
from triboworkflow.core.store import StoreError, SystemDocument, SystemWriter, read_system
from triboworkflow.core.store.legacy import LegacyStore
from triboworkflow.core.structures import read_structure
from triboworkflow.execution import layout
from triboworkflow.vendored.highput.utils.errors import InterfaceMatchError
from triboworkflow.workflows.interface import match as step
from triboworkflow.workflows.interface.registry import get_step

# Slab files of the example, written from the slabs of TribChem's matching script.
SLABS = {'Cu111': 'cu111', 'Al111': 'al111'}
ENTRIES = [('structure', 'init', 'interface'), ('structure', 'init', 'bot_slab'),
           ('structure', 'init', 'top_slab'), ('inter_params',), ('highsym',)]


def match_run(tmp_path, replace=None, name='run'):
    """A run of the example with steps = ["match"] and its two slabs as JSON files.

    ``replace`` maps texts of ``config.toml`` to their replacements.
    """
    run_dir = write_run(tmp_path, steps=('match',), name=name)
    config = run_dir / 'config.toml'
    text = config.read_text(encoding='utf-8')
    (run_dir / 'slabs').mkdir()
    dicts = golden_cases.slab_dicts()
    for file, slab in SLABS.items():
        (run_dir / 'slabs' / f'{file}.json').write_text(json.dumps(dicts[slab]))
        text = text.replace(f'slabs/{file}.extxyz', f'slabs/{file}.json')
    for old, new in (replace or {}).items():
        assert old in text
        text = text.replace(old, new)
    config.write_text(text, encoding='utf-8')
    return run_dir


def edit_config(run_dir, old, new):
    config = run_dir / 'config.toml'
    text = config.read_text(encoding='utf-8')
    assert old in text
    config.write_text(text.replace(old, new), encoding='utf-8')


def system_dir(run_dir):
    return layout.system_dir(run_dir, SYSTEM)


@pytest.fixture(scope='module')
def analysed(tmp_path_factory):
    """The example run analysed once by ``triboworkflow analyse match``: run folder and exit code."""
    run_dir = match_run(tmp_path_factory.mktemp('match'))
    return run_dir, main(['analyse', 'match', '-r', str(run_dir)])


@pytest.fixture
def run_copy(analysed, tmp_path):
    """A copy of the analysed run, to change."""
    run_dir, _ = analysed
    copied = tmp_path / 'copy'
    shutil.copytree(run_dir, copied)
    return copied


# --- registry and entries ------------------------------------------------------------

def test_match_is_registered():
    assert type(get_step('match')) is step.MatchStep and step.MatchStep.name == 'match'


def test_entries_of_the_step():
    assert step.match_entries() == ENTRIES


def test_analyse_writes_the_document(analysed):
    run_dir, code = analysed
    assert code == 0
    document = read_system(system_dir(run_dir)).document
    assert list(document) == ['mid', 'formula', 'miller', 'structure', 'name', 'inter_params',
                              'highsym']
    assert {k: document[k] for k in ('mid', 'formula', 'name', 'miller')} == {
        'mid': SYSTEM, 'formula': 'Cu_Al', 'name': 'Cu111-Al111',
        'miller': [[1, 1, 1], [1, 1, 1]]}
    assert 'comp_params' not in document  # DEV-35
    assert list(document['structure']['init']) == ['interface', 'bot_slab', 'top_slab']
    assert document['inter_params'] == {
        'max_area': 160.0, 'interface_distance': 2.0, 'max_angle_tol': 0.4,
        'max_length_tol': 0.02, 'max_area_ratio_tol': 0.05, 'strain_weight_1': 140.0,
        'strain_weight_2': 76.0, 'best_match': 'area'}
    assert list(document['highsym']) == ['bot_unique', 'bot_all', 'top_unique', 'top_all',
                                         'inter_unique', 'inter_all']
    assert document['highsym']['inter_unique']


def test_analyse_records_the_provenance_of_each_entry(analysed):
    run_dir, _ = analysed
    document = read_system(system_dir(run_dir))
    expected = step.match_provenance(load_run(run_dir))
    for entry in ENTRIES:
        assert document.provenance(entry) == expected
    assert sorted(expected.files) == ['slab_1', 'slab_2']
    assert expected.parameters['inter_params'] == document.get('inter_params')
    assert get_step('match').done(run_dir, SYSTEM)


def test_analyse_writes_the_interface_files(analysed):
    run_dir, _ = analysed
    out = system_dir(run_dir) / 'match'
    document = read_system(system_dir(run_dir)).document
    stored = document['structure']['init']['interface']
    data = json.loads((out / 'interface.json').read_text())
    assert data['lattice']['matrix'] == stored['lattice']['matrix']
    assert len(data['sites']) == len(stored['sites'])
    structure = read_structure(out / 'interface.extxyz')
    assert len(structure) == len(stored['sites'])
    np.testing.assert_allclose(structure.lattice.matrix, stored['lattice']['matrix'],
                               rtol=0, atol=1e-12)


def test_analyse_tells_the_slab_on_top_the_attempts_and_the_strain(analysed):
    run_dir, _ = analysed
    log = (run_dir / 'cli.log').read_text(encoding='utf-8')
    assert 'Al(111) (slab 2, mp-134) on top of Cu(111) (slab 1, mp-30)' in log
    assert 'match found at attempt 1' in log
    assert 'max_angle_tol = 0.4 (never enlarged)' in log
    assert 'strain of the top slab: a ' in log and 'of the bottom slab: a ' in log
    assert 'lateral shifts (inter_unique)' in log
    assert 'questo è' not in log  # the prints of the matching code are only at debug level


def test_orchestrator_keeps_the_step_completed(run_copy):
    report = build_orchestrator(run_copy, steps=['match']).analyse_step(SYSTEM, 'match')
    assert report.status == 'completed'


def test_orchestrator_completes_a_step_already_done(run_copy):
    (run_copy / 'orchestrator.json').unlink()
    report = build_orchestrator(run_copy, steps=['match']).analyse_step(SYSTEM, 'match')
    assert report.status == 'completed' and 'already done' in report.reason


# --- done and current ----------------------------------------------------------------

@pytest.mark.parametrize('old, new', [
    ('max_length_tol = 0.02', 'max_length_tol = 0.03'),
    ('# max_angle_tol = 0.4 ', 'max_angle_tol = 0.3 '),
    ('bulk_modulus = 76.0', 'bulk_modulus = 75.0'),
    ('set_bm = true', 'set_bm = false'),
])
def test_done_no_more_when_the_settings_change(run_copy, old, new):
    if old == 'set_bm = true':
        edit_config(run_copy, 'bulk_modulus = 140.0', '')
        edit_config(run_copy, 'bulk_modulus = 76.0', '')
    edit_config(run_copy, old, new)
    assert not get_step('match').done(run_copy, SYSTEM)


def test_done_no_more_when_a_slab_file_changes(run_copy):
    path = run_copy / 'slabs' / 'Al111.json'
    data = json.loads(path.read_text())
    data['sites'][0]['abc'][0] += 1e-6
    path.write_text(json.dumps(data))
    assert not get_step('match').done(run_copy, SYSTEM)


def test_done_needs_all_five_entries(run_copy):
    with SystemWriter(system_dir(run_copy)) as writer:
        writer.document.set(['highsym', 'extra'], 1)  # drops the provenance of highsym
        writer.save()
    assert not get_step('match').done(run_copy, SYSTEM)


# --- writing again -------------------------------------------------------------------

def test_new_write_leaves_nothing_of_the_old_entries(run_copy, caplog):
    # DEV-36: with the historic merge, the old labels of highsym would stay.
    with SystemWriter(system_dir(run_copy)) as writer:
        for entry in ENTRIES:
            writer.document.set(entry + ('stale_key',), [1.0, 2.0, 3.0])
        writer.save()
    before = read_system(system_dir(run_copy)).document
    with caplog.at_level(logging.DEBUG, logger='triboworkflow'):
        get_step('match').analyse(run_copy, SYSTEM, [])
    # What the matching code prints is kept at debug level.
    assert 'output of the matching code' in caplog.text
    after = read_system(system_dir(run_copy))
    for entry in ENTRIES:
        assert 'stale_key' not in after.get(entry)
    for key in ('mid', 'formula', 'miller', 'name'):
        assert after.document[key] == before[key]
    assert get_step('match').done(run_copy, SYSTEM)


def test_document_with_other_ids_is_an_error(run_copy):
    with SystemWriter(system_dir(run_copy)) as writer:
        writer.document.replace(['name'], 'other-name')
        writer.save()
    before = (system_dir(run_copy) / 'system.json').read_text()
    with pytest.raises(StoreError, match="does not match the document of system"):
        get_step('match').analyse(run_copy, SYSTEM, [])
    assert (system_dir(run_copy) / 'system.json').read_text() == before


def test_nothing_written_when_the_matching_does_not_converge(tmp_path):
    run_dir = match_run(tmp_path, {'max_area = 160.0': 'max_area = 0.0'})
    with pytest.raises(InterfaceMatchError, match='Matching not converged'):
        get_step('match').analyse(run_dir, SYSTEM, [])
    assert not (system_dir(run_dir) / 'system.json').exists()
    assert not (system_dir(run_dir) / 'match').exists()
    report = build_orchestrator(run_dir, steps=['match']).analyse_step(SYSTEM, 'match')
    assert report.status == 'failed' and 'Matching not converged' in report.reason


def test_slabs_of_the_wrong_system_are_an_error(run_copy):
    with pytest.raises(ValueError, match="make the interface 'mp-30_mp-134'"):
        get_step('match').analyse(run_copy, 'mp-1_mp-2', [])


# --- matching ------------------------------------------------------------------------

def matching_parameters(**given):
    """Complete matching parameters of the example with ``[interface.inter_params]`` = ``given``."""
    data = tomllib.loads(EXAMPLE.read_text(encoding='utf-8'))
    data['interface']['inter_params'] = given
    return step.match_parameters(settings_from_dict(data))[1]


def test_defaults_of_the_matching_parameters():
    assert matching_parameters() == {
        'max_area': 160.0, 'interface_distance': 2, 'max_angle_tol': 0.4,
        'max_length_tol': 0.02, 'max_area_ratio_tol': 0.05, 'strain_weight_1': 140.0,
        'strain_weight_2': 76.0, 'best_match': 'area'}


def test_max_angle_tol_is_used_unchanged_at_every_attempt(monkeypatch):
    # DEV-01: the other tolerances are enlarged, max_angle_tol never.
    seen = []
    matcher = step.InterfaceMatcher

    def recording(**kwargs):
        seen.append({k: kwargs[k] for k in ('max_area', 'max_length_tol', 'max_area_ratio_tol',
                                            'max_angle_tol')})
        return matcher(**kwargs)

    monkeypatch.setattr(step, 'InterfaceMatcher', recording)
    inter_params = matching_parameters(max_area=20.0, max_angle_tol=0.3)
    slabs = [golden_cases.load_slab('cu111'), golden_cases.load_slab('al111')]
    match = step.generate_interface(slabs, inter_params)
    assert match.attempts == len(seen) > 1
    assert [s['max_angle_tol'] for s in seen] == [0.3] * len(seen)
    factor, area, length = 1, 20.0, 0.02
    for s in seen:
        area, length = area * factor, length * factor
        assert (s['max_area'], s['max_length_tol']) == (area, length)
        factor += 0.01
    assert inter_params['max_area'] == seen[-1]['max_area']
    assert inter_params['max_angle_tol'] == 0.3


def test_the_two_matchings_give_the_same_slabs(tmp_path):
    inputs =step.match_inputs(load_run(match_run(tmp_path)))
    result = step.run_match(inputs, LegacyStore(tmp_path, SystemDocument(), step.COLLECTION))
    assert result.match.bot_slab.as_dict() == result.hs_match.bot_slab.as_dict()
    assert result.match.top_slab.as_dict() == result.hs_match.top_slab.as_dict()
    assert result.inter_params == inputs.inter_params  # matched at the first attempt


# --- formula and name ----------------------------------------------------------------

def test_formula_and_name_of_the_settings(tmp_path):
    run_dir = match_run(tmp_path, {
        '# formula = "Cu"\n# name = "Cu-fcc"': 'formula = "Cu"\nname = "Cu-fcc"',
        'mid = "mp-134"': 'mid = "mp-134"\nformula = "Al"\nname = "Al-fcc"'})
    inputs = step.match_inputs(load_run(run_dir))
    assert inputs.ids == {'mid': SYSTEM, 'formula': 'Cu_Al', 'name': 'Cu-fcc_Al-fcc',
                          'miller': [[1, 1, 1], [1, 1, 1]]}
    assert step.set_filter(inputs.ids) == inputs.ids


def test_formula_without_name_is_refused_by_the_matching_code():
    # A75: the settings refuse it first; the check of the matching code stays.
    with pytest.raises(InterfaceMatchError, match='formula or name is wrong'):
        InterfaceMatchError.check_inputs({'mid': ['a', 'b'], 'miller': [[1, 1, 1]] * 2,
                                          'formula': ['Cu', 'Al'], 'name': None})
