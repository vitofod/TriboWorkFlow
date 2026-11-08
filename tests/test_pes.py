# Author: Vito Foderà
# Created: 2026-11-07

"""Tests of the steps pes and pes_scf (``triboworkflow.workflows.interface.pes``).

The example run is matched once (step ``match``); the tests check what the
two PES steps read from its document and the calculations they make: one per
high-symmetry point, relaxed along z for ``pes`` and static for ``pes_scf``,
on complete or reduced slabs, with the ids of the document.

Notes
-----
To remove: the golden cases ``pes_step_*`` (``tests/golden_cases.py``) compare
the steps with TribChem's ``FT_StartPES``, ``InterfaceWF.calculate_pes``,
``FT_MoveTagResults`` and ``FT_PotentialEnergySurface``: points, structures,
k-points and database writes.
"""

import dataclasses
import json
import logging
import re
import shutil

import numpy as np
import pytest
from monty.json import jsanitize

import golden_cases
from cli_helpers import SYSTEM, complete, write_run
from triboworkflow.cli.main import main
from triboworkflow.cli.run import load_run
from triboworkflow.core import store
from triboworkflow.core.settings import SettingsError
from triboworkflow.core.store import PrerequisiteError, StoreError, SystemWriter, read_system
from triboworkflow.execution import layout
from triboworkflow.vendored.physics.base.shaper import Shaper
from triboworkflow.vendored.physics.base.units import Converter
from triboworkflow.workflows.interface import pes
from triboworkflow.workflows.interface.match import MatchStep, match_provenance

SLABS = {'Cu111': 'cu111', 'Al111': 'al111'}
# Expected checksums of the UPF files of the example (the files are not read).
MD5 = {'Cu.UPF': '0' * 32, 'Al.UPF': '1' * 32}
READ = [('structure', 'init', 'interface'), ('structure', 'init', 'bot_slab'),
        ('structure', 'init', 'top_slab'), ('highsym',)]


def pes_run(tmp_path, name='run', steps=('match', 'pes', 'pes_scf')):
    """A run of the example with the two PES steps, its slabs as JSON files and the checksums."""
    run_dir = write_run(tmp_path, steps=steps, name=name)
    (run_dir / 'pseudo.md5').write_text(''.join(f'{md5}  {upf}\n' for upf, md5 in MD5.items()))
    config = run_dir / 'config.toml'
    text = config.read_text(encoding='utf-8')
    (run_dir / 'slabs').mkdir()
    dicts = golden_cases.slab_dicts()
    for file, slab in SLABS.items():
        (run_dir / 'slabs' / f'{file}.json').write_text(json.dumps(dicts[slab]))
        text = text.replace(f'slabs/{file}.extxyz', f'slabs/{file}.json')
    config.write_text(text, encoding='utf-8')
    return run_dir


def edit_config(run_dir, old, new):
    config = run_dir / 'config.toml'
    text = config.read_text(encoding='utf-8')
    assert old in text
    config.write_text(text.replace(old, new), encoding='utf-8')


@pytest.fixture(scope='module')
def matched(tmp_path_factory):
    """The example run after the analysis of match."""
    run_dir = pes_run(tmp_path_factory.mktemp('pes'))
    MatchStep().analyse(run_dir, SYSTEM, [])
    return run_dir


@pytest.fixture
def run_copy(matched, tmp_path):
    copied = tmp_path / 'copy'
    shutil.copytree(matched, copied)
    return copied


@pytest.fixture(scope='module')
def analysed(matched, tmp_path_factory):
    """The matched run with the two PES analysed: run folder and ``calc_id`` of each step."""
    run_dir = tmp_path_factory.mktemp('analysed') / 'run'
    shutil.copytree(matched, run_dir)
    calc_ids = {}
    for name, step in STEPS.items():
        calc_ids[name] = finish(run_dir, name)
        step().analyse(run_dir, SYSTEM, calc_ids[name])
    return run_dir, calc_ids


@pytest.fixture
def analysed_copy(analysed, tmp_path):
    copied = tmp_path / 'analysed'
    shutil.copytree(analysed[0], copied)
    return copied, analysed[1]


def document(run_dir):
    return read_system(layout.system_dir(run_dir, SYSTEM)).document


def layers(structure):
    return len(Shaper._get_layers(structure, 0.1))


STEPS = {'pes': pes.RelaxedPesStep, 'pes_scf': pes.StaticPesStep}
# Keys of the results of each point in the document.
OUTPUT_KEYS = {'nsites', 'density', 'energy', 'energy_per_atom', 'bandgap', 'cbm', 'vbm',
               'is_gap_direct', 'forces', 'stress', 'pseudo_potential', 'dir_name', 'task_id',
               'task_label', 'structure'}


def status_of(name):
    """States of a completed calculation of the step: a relax has a geometry."""
    return store.Status('ended', 'converged', 'converged' if name == 'pes' else 'n/a', 'ok')


def finish(run_dir, name, energies=None, prepare=True, system=SYSTEM, **changes):
    """Prepare the calculations of a PES step and complete them with synthetic results.

    The energies are ``energies[point]``; by default they decrease along the
    points, so the minimum is the last point. The final structure of a relax
    is the input moved by 0.01 angstrom along z; that of a scf is the input.
    With ``prepare`` False the calculations are already prepared.

    Returns
    -------
    list of str
        ``calc_id`` of the calculations, in the order of the points.
    """
    inputs = pes.pes_inputs(run_dir, system, load_run(run_dir), name)
    if prepare:
        STEPS[name]().prepare(run_dir, system)
    sdir = layout.system_dir(run_dir, system)
    points = [point for point, _, _ in inputs.points]
    if energies is None:
        energies = {point: -1000.0 - 0.01 * i for i, point in enumerate(points)}
    for point, structure, _ in inputs.points:
        final = structure.copy()
        if name == 'pes':
            final.translate_sites(list(range(len(final))), [0, 0, 0.01], frac_coords=False)
        energy = energies[point]
        results = {'nsites': len(structure), 'input': {'structure': structure.as_dict()},
                   'output': {'structure': final.as_dict(), 'density': float(final.density),
                              'energy': energy, 'energy_per_atom': energy / len(structure)}}
        record = store.read_record(sdir, f'{name}/{point}')
        store.write_record(sdir, dataclasses.replace(
            record, status=status_of(name), results=jsanitize(results, strict=False),
            observed={}, **changes))
    return [f'{name}/{point}' for point in points]


def stored(value):
    """A value as the historic writes leave it: the empty dicts are not written (lists as given)."""
    if not isinstance(value, dict):
        return value
    kept = {key: stored(item) for key, item in value.items()}
    return {key: item for key, item in kept.items() if not (isinstance(item, dict) and not item)}


def system_bytes(run_dir):
    return (layout.system_dir(run_dir, SYSTEM) / 'system.json').read_bytes()


def position_flags(input_text):
    """The if_pos of each line of ATOMIC_POSITIONS (None where not written)."""
    lines = input_text.split('ATOMIC_POSITIONS')[1].splitlines()[1:]
    flags = []
    for line in lines:
        parts = line.split()
        if len(parts) not in (4, 7):
            break
        flags.append(parts[4:] if len(parts) == 7 else None)
    return flags


# --- the two steps --------------------------------------------------------------------

def test_two_steps_with_their_calc_type():
    assert pes.RelaxedPesStep.name == 'pes' and pes.StaticPesStep.name == 'pes_scf'
    assert pes.CALC_TYPES == {'pes': 'interface_z_relax', 'pes_scf': 'slab_from_scratch'}
    assert pes.PES_TYPE == 'energy'


@pytest.mark.parametrize('step, calculation, flags', [
    (pes.RelaxedPesStep, 'relax', ['0', '0', '1']),
    (pes.StaticPesStep, 'scf', None),
])
def test_one_calculation_per_high_symmetry_point(matched, step, calculation, flags):
    points = list(document(matched)['highsym']['inter_unique'])
    calcs = step().calculations(matched, SYSTEM)
    assert [calc.calc_id for calc in calcs] == [f'{step.name}/{point}' for point in points]
    for calc in calcs:
        assert calc.record.calc_type == pes.CALC_TYPES[step.name]
        assert calc.record.step == step.name and calc.record.system == SYSTEM
        assert re.search(rf"calculation\s*=\s*'{calculation}'", calc.input_text)
        assert re.search(rf"prefix\s*=\s*'{step.name}_", calc.input_text)
        # Relaxed along z: only z moves (no hydrogen in Cu/Al); static: nothing to fix.
        assert set(map(lambda f: None if f is None else tuple(f),
                       position_flags(calc.input_text))) == {None if flags is None
                                                             else tuple(flags)}


def test_same_structures_for_the_two_steps(matched):
    settings = load_run(matched)
    relaxed = pes.pes_inputs(matched, SYSTEM, settings, 'pes')
    static = pes.pes_inputs(matched, SYSTEM, settings, 'pes_scf')
    assert [p[0] for p in relaxed.points] == [p[0] for p in static.points]
    for (_, a, ka), (_, b, kb) in zip(relaxed.points, static.points):
        assert a.lattice == b.lattice and a.cart_coords.tolist() == b.cart_coords.tolist()
        assert ka.kpts == kb.kpts
    assert relaxed.points[0][1].site_properties['selective_dynamics'][0] == [False, False, True]


def test_inputs_read_from_the_document_of_match(matched):
    settings = load_run(matched)
    inputs = pes.pes_inputs(matched, SYSTEM, settings, 'pes')
    doc = document(matched)
    assert inputs.read == {'interface': doc['structure']['init']['interface'],
                           'bot_slab': doc['structure']['init']['bot_slab'],
                           'top_slab': doc['structure']['init']['top_slab'],
                           'shifts': doc['highsym']['inter_unique']}
    # The ids of the document written by match select it in the writes.
    assert inputs.ids == {'mid': SYSTEM, 'formula': 'Cu_Al', 'name': 'Cu111-Al111',
                          'miller': [[1, 1, 1], [1, 1, 1]]}
    assert inputs.parameters == {'calc_type': 'interface_z_relax', 'pes_type': 'energy',
                                 'comp_params': {'kdens': 7.5}, 'interface_distance': 2.0,
                                 'layers': None}


def test_expected_entries_of_match(matched):
    expected = pes.RelaxedPesStep().expected(matched, SYSTEM)
    assert list(expected) == READ == pes.match_entries_read()
    provenance = match_provenance(load_run(matched))
    assert all(value == provenance for value in expected.values())


# --- reduced slabs of the static PES --------------------------------------------------

def test_static_pes_on_complete_slabs_by_default(matched):
    inputs = pes.pes_inputs(matched, SYSTEM, load_run(matched), 'pes_scf')
    assert inputs.parameters['layers'] is None
    assert (layers(inputs.bot_slab), layers(inputs.top_slab)) == (4, 4)


def test_short_with_the_layers_of_the_rule(run_copy):
    edit_config(run_copy, 'short = false', 'short = true')
    full = pes.pes_inputs(run_copy, SYSTEM, load_run(run_copy), 'pes')
    inputs = pes.pes_inputs(run_copy, SYSTEM, load_run(run_copy), 'pes_scf')
    # Both slabs are (111): 3 layers each.
    assert inputs.parameters['layers'] == [3, 3]
    assert (layers(inputs.bot_slab), layers(inputs.top_slab)) == (3, 3)
    assert len(inputs.points[0][1]) < len(full.points[0][1])
    # The relaxed PES always uses the complete slabs.
    assert full.parameters['layers'] is None
    assert (layers(full.bot_slab), layers(full.top_slab)) == (4, 4)


def test_short_with_the_layers_given(run_copy):
    edit_config(run_copy, 'short = false', 'short = true\nlayers = [2, 1]')
    inputs = pes.pes_inputs(run_copy, SYSTEM, load_run(run_copy), 'pes_scf')
    assert inputs.parameters['layers'] == [2, 1]
    # The first value goes to the bottom slab (Cu), the second to the top one (Al).
    assert inputs.bot_slab.composition.reduced_formula == 'Cu'
    assert (layers(inputs.bot_slab), layers(inputs.top_slab)) == (2, 1)


def test_rule_of_the_layers_on_the_physical_slabs():
    class Slab:
        def __init__(self, miller):
            self.miller_index = miller

    assert pes.short_layers(Slab((1, 1, 1)), Slab((1, 0, 0))) == [3, 2]
    assert pes.short_layers(Slab((1, 0, 0)), Slab((1, 1, 1))) == [2, 3]
    assert pes.short_layers(Slab((1, 0, 0)), Slab((1, 1, 1)), [4, 4]) == [4, 4]


def test_provenance_follows_the_reduced_slabs(run_copy):
    complete = pes.StaticPesStep().provenance(run_copy, SYSTEM)
    edit_config(run_copy, 'short = false', 'short = true')
    reduced = pes.StaticPesStep().provenance(run_copy, SYSTEM)
    assert reduced != complete
    assert reduced.parameters['layers'] == [3, 3] and complete.parameters['layers'] is None
    assert set(reduced.calculations) == set(complete.calculations)
    assert reduced.calculations != complete.calculations


# --- errors -----------------------------------------------------------------------------

def test_without_match_the_step_stops(tmp_path):
    run_dir = pes_run(tmp_path)
    with pytest.raises(PrerequisiteError, match='run match first'):
        pes.RelaxedPesStep().calculations(run_dir, SYSTEM)


def test_k_points_from_their_density_only(matched):
    settings = load_run(matched)
    settings = dataclasses.replace(settings, k_points=dataclasses.replace(
        settings.k_points, mode='automatic', grid=[4, 4, 1], offset=[0, 0, 0]))
    with pytest.raises(SettingsError, match=r'k_points\.mode.*kdens'):
        pes.pes_inputs(matched, SYSTEM, settings, 'pes_scf')


# --- analysis ---------------------------------------------------------------------------

def test_entry_of_each_calc_type():
    for name, calc_type in pes.CALC_TYPES.items():
        assert pes.pes_entry(calc_type) == name
        entry_from, entry_to = pes.transfer_entries(calc_type, 'aa_1')
        assert len(entry_from) == len(entry_to) == len(pes.TRANSFER_FROM) + 2
        assert all(entry[:3] == [name, 'data', 'aa_1'] for entry in entry_to)
        # Each key goes to the last part of its source, below output; the
        # input structure below input.
        assert [entry[-1] for entry in entry_to] == [entry[-1] for entry in entry_from]
        assert [entry[3] for entry in entry_to] == ['output'] * (len(entry_to) - 1) + ['input']


@pytest.mark.parametrize('name', ['pes', 'pes_scf'])
def test_results_of_every_point(analysed, name):
    run_dir, calc_ids = analysed
    doc = document(run_dir)
    sdir = layout.system_dir(run_dir, SYSTEM)
    points = list(doc['highsym']['inter_unique'])
    assert list(doc[name]['data']) == points
    for point, calc_id in zip(points, calc_ids[name]):
        data = doc[name]['data'][point]
        results = store.read_record(sdir, calc_id).results
        assert set(data) == {'input', 'output'}
        assert set(data['output']) == OUTPUT_KEYS
        # The structures without their empty dicts ('properties': {}), as merged
        # leaf by leaf; they are read again as they were.
        assert data['input'] == {'structure': stored(results['input']['structure'])}
        assert data['output']['structure'] == stored(results['output']['structure'])
        assert data['input']['structure'] != results['input']['structure']
        for key in ('input', 'output'):
            read, given = (pes._structure(d) for d in (data[key]['structure'],
                                                       results[key]['structure']))
            assert type(read) is type(given)
            assert read.lattice.matrix.tolist() == given.lattice.matrix.tolist()
            assert read.cart_coords.tolist() == given.cart_coords.tolist()
            assert read.species == given.species
            assert read.site_properties == given.site_properties
        assert data['output']['nsites'] == results['nsites']
        for key in ('density', 'energy', 'energy_per_atom'):
            assert data['output'][key] == results['output'][key]
        # Keys that pw.x does not give are None, as missing keys.
        for key in ('bandgap', 'cbm', 'vbm', 'is_gap_direct', 'forces', 'stress',
                    'pseudo_potential'):
            assert data['output'][key] is None
        assert data['output']['task_label'] == data['output']['task_id'] == calc_id
        assert data['output']['dir_name'] == calc_id


@pytest.mark.parametrize('name', ['pes', 'pes_scf'])
def test_grid_minimum_and_maximum(analysed, name):
    run_dir, _ = analysed
    doc = document(run_dir)
    shifts = doc['highsym']['inter_unique']
    data = doc[name]['data']
    energy = doc[name]['energy']
    # Sorted by energy: the energies decrease along the points.
    expected = [[point, *shifts[point][0], data[point]['output']['energy']]
                for point in reversed(list(shifts))]
    assert energy['grid']['hs_unique_ev'] == expected
    assert energy['min_site'] == expected[0][0] and energy['max_site'] == expected[-1][0]
    # J/m^2 on the area of the cell of the final structure of the minimum.
    a, b, _ = np.array(data[energy['min_site']]['output']['structure']['lattice']['matrix'])
    area = np.linalg.norm(np.cross(a, b))
    assert energy['grid']['hs_unique_jm2'] == [
        [p, x, y, Converter.eV_to_Jm2(v, area * 1e-20)] for p, x, y, v in expected]
    assert energy['grid']['hs_unique_jm2'][0][3] == pytest.approx(
        expected[0][3] * 16.021766 / area, rel=1e-6)
    assert set(doc[name]) == {'data', 'energy'}
    assert set(energy) == {'grid', 'min_site', 'max_site'}
    opt = doc['structure']['opt'][name]
    assert opt == {'min': data[energy['min_site']]['output']['structure'],
                   'max': data[energy['max_site']]['output']['structure']}


def test_the_two_pes_in_one_document(analysed):
    run_dir, _ = analysed
    doc = document(run_dir)
    assert set(doc['structure']['opt']) == {'pes', 'pes_scf'}
    assert doc['pes']['data'] != doc['pes_scf']['data']
    for name, step in STEPS.items():
        assert step().done(run_dir, SYSTEM)
        provenance = read_system(layout.system_dir(run_dir, SYSTEM)).provenance([name, 'data'])
        assert provenance.step == name
        assert set(provenance.calculations) == set(analysed[1][name])
    # The entries of match keep their provenance.
    assert MatchStep().done(run_dir, SYSTEM)


def test_ties_keep_the_order_of_the_points(run_copy):
    calc_ids = finish(run_copy, 'pes_scf', energies={
        point: -1000.0 for point in document(run_copy)['highsym']['inter_unique']})
    pes.StaticPesStep().analyse(run_copy, SYSTEM, calc_ids)
    points = list(document(run_copy)['highsym']['inter_unique'])
    energy = document(run_copy)['pes_scf']['energy']
    assert [row[0] for row in energy['grid']['hs_unique_ev']] == points
    assert energy['min_site'] == points[0] and energy['max_site'] == points[-1]


def test_done_only_after_the_analysis(run_copy):
    assert not pes.RelaxedPesStep().done(run_copy, SYSTEM)
    calc_ids = finish(run_copy, 'pes')
    assert not pes.RelaxedPesStep().done(run_copy, SYSTEM)
    pes.RelaxedPesStep().analyse(run_copy, SYSTEM, calc_ids)
    assert pes.RelaxedPesStep().done(run_copy, SYSTEM)
    assert not pes.StaticPesStep().done(run_copy, SYSTEM)


def test_done_without_match(tmp_path):
    assert not pes.StaticPesStep().done(pes_run(tmp_path), SYSTEM)


@pytest.mark.parametrize('old, new, changed', [
    ('kdens = 7.5', 'kdens = 8.0', {'pes', 'pes_scf'}),
    ('short = false', 'short = true', {'pes_scf'}),
])
def test_done_follows_the_settings(analysed_copy, old, new, changed):
    run_dir, _ = analysed_copy
    edit_config(run_dir, old, new)
    assert {name for name, step in STEPS.items() if not step().done(run_dir, SYSTEM)} == changed


def test_done_follows_the_results(analysed_copy):
    run_dir, calc_ids = analysed_copy
    sdir = layout.system_dir(run_dir, SYSTEM)
    record = store.read_record(sdir, calc_ids['pes'][0])
    results = {**record.results, 'output': {**record.results['output'], 'energy': -999.0}}
    store.write_record(sdir, dataclasses.replace(record, results=results))
    assert not pes.RelaxedPesStep().done(run_dir, SYSTEM)
    assert pes.StaticPesStep().done(run_dir, SYSTEM)


def test_analysis_again_gives_the_same_document(analysed_copy):
    run_dir, calc_ids = analysed_copy
    before = document(run_dir)
    pes.StaticPesStep().analyse(run_dir, SYSTEM, calc_ids['pes_scf'])
    assert document(run_dir) == before
    assert all(step().done(run_dir, SYSTEM) for step in STEPS.values())


OLD_POINT = 'old_1-old_1'


def other_pes(name):
    return 'pes_scf' if name == 'pes' else 'pes'


def leave_old_data(run_dir, name):
    """Leave in the three entries of a PES what an analysis of an earlier interface left."""
    with SystemWriter(layout.system_dir(run_dir, SYSTEM)) as writer:
        writer.document.set([name, 'data', OLD_POINT], {'output': {'energy': -1.0}})
        writer.document.set([name, 'energy', 'old'], 1.0)
        for end in ('min', 'max'):
            writer.document.set(['structure', 'opt', name, end, 'old'], 1.0)
        writer.save()


def provenances(run_dir, name):
    document_ = read_system(layout.system_dir(run_dir, SYSTEM))
    return [document_.provenance(entry) for entry in pes.pes_leaves(name)]


@pytest.mark.parametrize('name', ['pes', 'pes_scf'])
def test_analysis_again_leaves_nothing_of_the_old_entries(analysed_copy, name):
    run_dir, calc_ids = analysed_copy
    clean = document(run_dir)
    other = provenances(run_dir, other_pes(name))
    leave_old_data(run_dir, name)
    doc = document(run_dir)
    assert OLD_POINT in doc[name]['data']
    assert doc['structure']['opt'][name]['min']['old'] == 1.0
    STEPS[name]().analyse(run_dir, SYSTEM, calc_ids[name])
    # The three entries hold only the new analysis; the other PES and the
    # entries of match are as they were, with their provenance.
    assert document(run_dir) == clean
    assert provenances(run_dir, other_pes(name)) == other
    assert all(step().done(run_dir, SYSTEM) for step in STEPS.values())
    assert MatchStep().done(run_dir, SYSTEM)


@pytest.mark.parametrize('name', ['pes', 'pes_scf'])
def test_failure_after_the_emptying_saves_nothing(analysed_copy, name, monkeypatch):
    run_dir, calc_ids = analysed_copy
    leave_old_data(run_dir, name)
    before = system_bytes(run_dir)
    seen = []

    def fail(store, ids, calc_type, pes_type=pes.PES_TYPE):
        seen.append(store.document.get([name]))
        raise RuntimeError('arrange_data failed')

    monkeypatch.setattr(pes, 'arrange_data', fail)
    with pytest.raises(RuntimeError, match='arrange_data failed'):
        STEPS[name]().analyse(run_dir, SYSTEM, calc_ids[name])
    # The entry had been emptied and filled with the new points only ...
    points = list(document(run_dir)['highsym']['inter_unique'])
    assert set(seen[0]) == {'data'} and list(seen[0]['data']) == points
    # ... but nothing reached the disk: the old data, the other PES and the
    # provenance are those of before; the lock is released.
    assert system_bytes(run_dir) == before
    assert OLD_POINT in document(run_dir)[name]['data']
    assert not SystemWriter(layout.system_dir(run_dir, SYSTEM)).lock_path.exists()
    assert STEPS[other_pes(name)]().done(run_dir, SYSTEM)
    assert not STEPS[name]().done(run_dir, SYSTEM)


# Entries that the interpolation of a PES writes into its entry ``energy``.
INTERPOLATION = (('grid', 'hs_all_ev'), ('grid', 'hs_all_jm2'), ('pes_ev',), ('pes_jm2',))


def write_interpolation(run_dir, name):
    """Write the entries of an interpolation of the PES ``name``, with a provenance."""
    provenance = store.Provenance.build(f'plot{name}', parameters={'density': 50})
    with SystemWriter(layout.system_dir(run_dir, SYSTEM)) as writer:
        writer.document.set([name, pes.PES_TYPE], {'grid': {'hs_all_ev': [[0.0, 0.0, 0.0]],
                                                            'hs_all_jm2': [[0.0, 0.0, 0.0]]},
                                                   'pes_ev': [[0.0]], 'pes_jm2': [[0.0]]})
        for leaf in INTERPOLATION:
            writer.document.set_provenance([name, pes.PES_TYPE, *leaf], provenance)
        writer.save()
    return provenance


@pytest.mark.parametrize('name', ['pes', 'pes_scf'])
def test_provenance_on_the_entries_written(analysed, name):
    run_dir, _ = analysed
    document_ = read_system(layout.system_dir(run_dir, SYSTEM))
    leaves = pes.pes_leaves(name)
    assert all(document_.get(leaf) is not None for leaf in leaves)
    first, *others = provenances(run_dir, name)
    assert first is not None and all(other == first for other in others)
    # Not on the whole entry, which the interpolation writes into.
    assert document_.provenance([name]) is None
    assert document_.provenance([name, pes.PES_TYPE]) is None


@pytest.mark.parametrize('name', ['pes', 'pes_scf'])
def test_interpolation_leaves_the_pes_current(analysed_copy, name):
    """The entries of an interpolation written into the entry of the PES
    leave the PES done: with the provenance on the whole entry, this write
    dropped it and the PES was no longer current."""
    run_dir, _ = analysed_copy
    before = provenances(run_dir, name)
    other = provenances(run_dir, other_pes(name))
    write_interpolation(run_dir, name)
    assert provenances(run_dir, name) == before
    assert provenances(run_dir, other_pes(name)) == other
    assert all(step().done(run_dir, SYSTEM) for step in STEPS.values())


@pytest.mark.parametrize('name', ['pes', 'pes_scf'])
def test_analysis_again_removes_the_interpolation(analysed_copy, name):
    run_dir, calc_ids = analysed_copy
    clean = document(run_dir)
    write_interpolation(run_dir, name)
    write_interpolation(run_dir, other_pes(name))
    STEPS[name]().analyse(run_dir, SYSTEM, calc_ids[name])
    # The entry of the PES analysed again holds nothing of the interpolation,
    # nor its provenance; the interpolation of the other PES stays.
    doc = document(run_dir)
    assert doc[name] == clean[name]
    sdir = layout.system_dir(run_dir, SYSTEM)
    for leaf in INTERPOLATION:
        assert read_system(sdir).provenance([name, pes.PES_TYPE, *leaf]) is None
        assert read_system(sdir).provenance([other_pes(name), pes.PES_TYPE, *leaf]) is not None
    assert 'pes_ev' in doc[other_pes(name)][pes.PES_TYPE]
    assert all(step().done(run_dir, SYSTEM) for step in STEPS.values())


def test_message_of_the_analysis(run_copy, caplog):
    run_dir = run_copy
    edit_config(run_dir, 'short = false', 'short = true\nlayers = [2, 1]')
    calc_ids = finish(run_dir, 'pes_scf')
    with caplog.at_level(logging.INFO, logger=pes.__name__):
        pes.StaticPesStep().analyse(run_dir, SYSTEM, calc_ids)
    energy = document(run_dir)['pes_scf']['energy']
    ev, jm2 = energy['grid']['hs_unique_ev'], energy['grid']['hs_unique_jm2']
    lines = [record.getMessage() for record in caplog.records if record.name == pes.__name__]
    assert lines == [
        f'{SYSTEM}: pes_scf: {len(ev)} high-symmetry points, static, on the slabs reduced to 2 '
        '(bottom) and 1 (top) layers (slab_from_scratch)',
        f'{SYSTEM}: minimum at {ev[0][0]}: {ev[0][3]:.6f} eV; maximum at {ev[-1][0]}: '
        f'{ev[-1][3]:.6f} eV',
        f'{SYSTEM}: corrugation (maximum - minimum): {ev[-1][3] - ev[0][3]:.6f} eV, '
        f'{jm2[-1][3] - jm2[0][3]:.6f} J/m^2',
    ]


@pytest.mark.parametrize('names', [('pes',), ('pes_scf',), ('pes_scf', 'pes')])
def test_the_steps_from_the_command_line(tmp_path, names, capsys):
    """match, then each PES prepared, completed by a worker and analysed: the
    static PES needs nothing of the relaxed one."""
    run_dir = pes_run(tmp_path, steps=('match', *sorted(names)))
    assert main(['analyse', 'match', '-r', str(run_dir)]) == 0
    for name in names:
        assert main(['prepare', name, '-r', str(run_dir)]) == 0
        calc_ids = [calc.calc_id for calc in STEPS[name]().calculations(run_dir, SYSTEM)]
        assert f'{len(calc_ids)} calculations prepared' in capsys.readouterr().out
        for calc_id in calc_ids:
            complete(run_dir, calc_id, status=status_of(name))
        finish(run_dir, name, prepare=False)
        assert main(['analyse', name, '-r', str(run_dir)]) == 0
        assert f'{SYSTEM}/{name}: completed' in capsys.readouterr().out
        assert STEPS[name]().done(run_dir, SYSTEM)
    doc = document(run_dir)
    assert {name for name in STEPS if name in doc} == set(names)
    assert set(doc['structure']['opt']) == set(names)


# --- errors of the analysis -------------------------------------------------------------

def test_incomplete_record_stops_the_analysis(run_copy):
    calc_ids = finish(run_copy, 'pes')
    sdir = layout.system_dir(run_copy, SYSTEM)
    record = store.read_record(sdir, calc_ids[1])
    store.write_record(sdir, dataclasses.replace(
        record, status=store.Status('ended', 'converged', 'not_converged', None), results=None,
        observed=None))
    before = system_bytes(run_copy)
    with pytest.raises(StoreError, match=rf'{re.escape(calc_ids[1])}: not complete'):
        pes.RelaxedPesStep().analyse(run_copy, SYSTEM, calc_ids)
    assert system_bytes(run_copy) == before


def test_one_energy_model_for_all_points(run_copy):
    calc_ids = finish(run_copy, 'pes_scf')
    sdir = layout.system_dir(run_copy, SYSTEM)
    record = store.read_record(sdir, calc_ids[0])
    store.write_record(sdir, dataclasses.replace(
        record, energy_model={**record.energy_model, 'ecutwfc': 1.0}))
    before = system_bytes(run_copy)
    with pytest.raises(StoreError, match='different energy models'):
        pes.StaticPesStep().analyse(run_copy, SYSTEM, calc_ids)
    assert system_bytes(run_copy) == before


def test_calculations_of_the_settings_only(run_copy):
    calc_ids = finish(run_copy, 'pes_scf')
    with pytest.raises(ValueError, match='not those of the settings'):
        pes.StaticPesStep().analyse(run_copy, SYSTEM, calc_ids[:-1])


def test_missing_record(run_copy):
    calc_ids = finish(run_copy, 'pes_scf')
    store.record_path(layout.system_dir(run_copy, SYSTEM), calc_ids[0]).unlink()
    with pytest.raises(StoreError, match=rf'{re.escape(calc_ids[0])}: no record'):
        pes.StaticPesStep().analyse(run_copy, SYSTEM, calc_ids)


def test_match_not_current_stops_the_analysis(run_copy):
    calc_ids = finish(run_copy, 'pes_scf')
    before = system_bytes(run_copy)
    edit_config(run_copy, 'max_area = 160.0', 'max_area = 150.0')
    with pytest.raises(PrerequisiteError, match='structure.init.interface'):
        pes.StaticPesStep().analyse(run_copy, SYSTEM, calc_ids)
    assert system_bytes(run_copy) == before
