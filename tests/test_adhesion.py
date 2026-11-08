# Author: Vito Foderà
# Created: 2026-11-08

"""Tests of the step adhesion (``triboworkflow.workflows.interface.adhesion``).

The example run is matched once and its two PES are analysed on synthetic
energies (the static one in the opposite order to the relaxed one, so that the
sites of the two differ); the tests check what the step reads from that
document for each of the three variants of ``[interface.adhesion]``, and the
four calculations it makes: the two slabs and the interface at the lowest and
at the highest site.

Notes
-----
To remove: the golden cases ``adhesion_step_*`` (``tests/golden_cases.py``)
compare the steps with TribChem's ``FT_StartAdhesion``,
``InterfaceWF.adhesion_energy``, ``FT_MoveTagResults`` and
``FT_AdhesionEnergy``: structures, k-points, database writes and energies.
"""

import copy
import dataclasses
import json
import logging
import re
import shutil

import numpy as np
import pytest
from monty.json import jsanitize
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from cli_helpers import SYSTEM, complete, fake_slurm  # noqa: F401 (fixture)
from test_pes import (document, edit_config, finish, layers, pes_run, position_flags, status_of,
                      stored)
from triboworkflow.cli.main import main
from triboworkflow.cli.run import build_orchestrator, load_run
from triboworkflow.core import store
from triboworkflow.core.settings import SettingsError
from triboworkflow.core.settings.schema import AdhesionParams
from triboworkflow.core.store import PrerequisiteError, StoreError, SystemDocument, SystemWriter
from triboworkflow.core.store.legacy import LegacyStore
from triboworkflow.execution.layout import system_dir
from triboworkflow.vendored.highput.utils.manipulate_struct import apply_interface_shifts
from triboworkflow.vendored.highput.utils.tasktools import select_struct_func
from triboworkflow.workflows.interface import adhesion, pes
from triboworkflow.workflows.interface.match import COLLECTION, MatchStep, match_provenance

PES_STEPS = {'pes': pes.RelaxedPesStep, 'pes_scf': pes.StaticPesStep}
VARIANTS = [('pes', 'regular'), ('pes_scf', 'regular'), ('pes_scf', 'short')]
LABELS = ['bot_slab', 'top_slab', 'interface_min', 'interface_max']


def adhesion_run(tmp_path, name='run'):
    """The example run with the steps up to adhesion."""
    return pes_run(tmp_path, name=name, steps=('match', 'pes', 'pes_scf', 'adhesion'))


def with_variant(settings, pes_name, kind):
    """The settings with another ``[interface.adhesion]``, as the TOML would give them."""
    interface = dataclasses.replace(settings.interface,
                                    adhesion=AdhesionParams(pes=pes_name, type=kind))
    changed = dataclasses.replace(settings, interface=interface)
    changed.source = settings.source
    return changed


def analyse_surfaces(run_dir, names=('pes', 'pes_scf')):
    """Analyse the PES steps on synthetic energies: the static one in the reverse order."""
    for name in names:
        inputs = pes.pes_inputs(run_dir, SYSTEM, load_run(run_dir), name)
        points = [point for point, _, _ in inputs.points]
        order = points if name == 'pes' else points[::-1]
        energies = {point: -1000.0 - 0.01 * i for i, point in enumerate(order)}
        PES_STEPS[name]().analyse(run_dir, SYSTEM, finish(run_dir, name, energies))


@pytest.fixture(scope='module')
def matched(tmp_path_factory):
    """The example run after the analysis of match."""
    run_dir = adhesion_run(tmp_path_factory.mktemp('adhesion'))
    MatchStep().analyse(run_dir, SYSTEM, [])
    return run_dir


@pytest.fixture(scope='module')
def surfaces(matched, tmp_path_factory):
    """The matched run with the two PES analysed (complete slabs)."""
    run_dir = tmp_path_factory.mktemp('surfaces') / 'run'
    shutil.copytree(matched, run_dir)
    analyse_surfaces(run_dir)
    return run_dir


@pytest.fixture(scope='module')
def short_surface(matched, tmp_path_factory):
    """The matched run with the static PES on reduced slabs, analysed, and no relaxed PES."""
    run_dir = tmp_path_factory.mktemp('short') / 'run'
    shutil.copytree(matched, run_dir)
    edit_config(run_dir, 'short = false', 'short = true')
    analyse_surfaces(run_dir, names=('pes_scf',))
    return run_dir


@pytest.fixture
def surfaces_copy(surfaces, tmp_path):
    copied = tmp_path / 'surfaces'
    shutil.copytree(surfaces, copied)
    return copied


def structure_of(data):
    return select_struct_func(data['@class']).from_dict(data)


def sites(run_dir):
    energy = document(run_dir)['pes_scf']['energy']
    return energy['min_site'], energy['max_site']


# --- the step and its variants ------------------------------------------------------------

def test_step_and_calculations():
    assert adhesion.AdhesionStep.name == adhesion.STEP == 'adhesion'
    assert adhesion.LABELS == tuple(LABELS)
    assert adhesion.CALC_TYPES == {'bot_slab': 'slab_pos_relax', 'top_slab': 'slab_pos_relax',
                                   'interface_min': 'slab_pos_relax',
                                   'interface_max': 'interface_z_relax'}


def inputs_of(run_dir, pes_name, kind):
    return adhesion.adhesion_inputs(run_dir, SYSTEM, with_variant(load_run(run_dir), pes_name, kind))


def test_the_sites_of_the_two_surfaces_differ(surfaces):
    # The synthetic energies of the static PES are in the reverse order.
    relaxed = document(surfaces)['pes']['energy']
    static = document(surfaces)['pes_scf']['energy']
    assert relaxed['min_site'] == static['max_site'] and relaxed['max_site'] == static['min_site']
    assert relaxed['min_site'] != relaxed['max_site']


@pytest.mark.parametrize('pes_name, kind', VARIANTS)
def test_four_calculations_with_their_calc_type(surfaces, short_surface, pes_name, kind):
    run_dir = short_surface if kind == 'short' else surfaces
    inputs = inputs_of(run_dir, pes_name, kind)
    calcs = adhesion.adhesion_calculations(SYSTEM, inputs)
    assert [calc.calc_id for calc in calcs] == [f'adhesion/{label}' for label in LABELS]
    for calc, label in zip(calcs, LABELS):
        assert calc.record.calc_type == adhesion.CALC_TYPES[label]
        assert calc.record.step == 'adhesion' and calc.record.system == SYSTEM
        # Every adhesion calculation is a relax: the slabs and the interface at the
        # minimum with all the atoms free, the interface at the maximum along z only
        # (there is no hydrogen in Cu/Al).
        assert re.search(r"calculation\s*=\s*'relax'", calc.input_text)
        assert re.search(rf"prefix\s*=\s*'adhesion_{label}'", calc.input_text)
        # pw.x writes no flag for a free atom.
        flags = {None if f is None else tuple(f) for f in position_flags(calc.input_text)}
        assert flags == ({('0', '0', '1')} if label == 'interface_max' else {None})
    for label, calc_type, structure, _ in inputs.points:
        moves = {tuple(move) for move in structure.site_properties['selective_dynamics']}
        assert moves == ({(False, False, True)} if label == 'interface_max'
                         else {(True, True, True)})


def structures_of(inputs):
    """``{label: structure}`` of the calculations of the inputs."""
    return {label: structure for label, _, structure, _ in inputs.points}


def same_positions(a, b):
    return a.lattice == b.lattice and len(a) == len(b) and \
        np.allclose(a.cart_coords, b.cart_coords)


# --- relaxed surface, regular slabs ---------------------------------------------------------

def test_structures_of_the_relaxed_surface(surfaces):
    doc = document(surfaces)
    inputs = inputs_of(surfaces, 'pes', 'regular')
    found = structures_of(inputs)
    opt, init = doc['structure']['opt']['pes'], doc['structure']['init']
    # The interfaces are the final structures of the minimum and of the maximum.
    assert same_positions(found['interface_min'], structure_of(opt['min']))
    assert same_positions(found['interface_max'], structure_of(opt['max']))
    assert same_positions(found['bot_slab'], structure_of(init['bot_slab']))
    assert same_positions(found['top_slab'], structure_of(init['top_slab']))
    assert inputs.sites is None
    assert inputs.init == {'bot_slab': init['bot_slab'], 'top_slab': init['top_slab']}


# --- static surface, regular slabs ----------------------------------------------------------

def test_structures_of_the_static_surface(surfaces):
    doc = document(surfaces)
    inputs = inputs_of(surfaces, 'pes_scf', 'regular')
    low, high = sites(surfaces)
    assert inputs.sites == (low, high)
    found = structures_of(inputs)
    bot, top = structure_of(doc['structure']['init']['bot_slab']), \
        structure_of(doc['structure']['init']['top_slab'])
    shifts = doc['highsym']['inter_unique']
    for label, site in (('interface_min', low), ('interface_max', high)):
        assert same_positions(found[label], apply_interface_shifts(bot, top, shifts[site]))
    assert same_positions(found['bot_slab'], bot) and same_positions(found['top_slab'], top)
    # The sites of the static PES are not those of the relaxed one.
    assert found['interface_min'].cart_coords.tolist() != \
        structures_of(inputs_of(surfaces, 'pes', 'regular'))['interface_min'].cart_coords.tolist()


def test_distance_of_the_interface_is_that_of_the_function(surfaces):
    # A86: the interface is stacked without the distance of [interface.inter_params].
    settings = with_variant(load_run(surfaces), 'pes_scf', 'regular')
    inter_params = dataclasses.replace(settings.interface.inter_params, interface_distance=3.0)
    other = dataclasses.replace(settings, interface=dataclasses.replace(
        settings.interface, inter_params=inter_params))
    other.source = settings.source
    first = structures_of(adhesion.adhesion_inputs(surfaces, SYSTEM, settings))
    second = structures_of(adhesion.adhesion_inputs(surfaces, SYSTEM, other))
    assert same_positions(first['interface_min'], second['interface_min'])
    doc = document(surfaces)
    bot, top = structure_of(doc['structure']['init']['bot_slab']), \
        structure_of(doc['structure']['init']['top_slab'])
    shift = doc['highsym']['inter_unique'][sites(surfaces)[0]]
    wider = apply_interface_shifts(bot, top, shift, 3.0)
    assert not same_positions(first['interface_min'], wider)


# --- static surface, reduced slabs ----------------------------------------------------------

def test_reduced_slabs_are_the_parts_of_the_interface(short_surface):
    doc = document(short_surface)
    inputs = inputs_of(short_surface, 'pes_scf', 'short')
    low, high = sites(short_surface)
    data = doc['pes_scf']['data']
    found = structures_of(inputs)
    # The interfaces are the input structures of the static PES at the two sites.
    assert same_positions(found['interface_min'], structure_of(data[low]['input']['structure']))
    assert same_positions(found['interface_max'], structure_of(data[high]['input']['structure']))
    # The slabs are its atoms below and above the plane z = 0, as slabs again.
    bot, top = found['bot_slab'], found['top_slab']
    assert len(bot) + len(top) == len(found['interface_min'])
    assert (bot.cart_coords[:, 2] < 0).all() and (top.cart_coords[:, 2] > 0).all()
    assert (layers(bot), layers(top)) == (3, 3)
    regular = doc['structure']['init']
    for key, reference in (('bot_slab', regular['bot_slab']), ('top_slab', regular['top_slab'])):
        assert inputs.init[key]['@class'] == 'Slab'
        assert inputs.init[key]['miller_index'] == reference['miller_index']
    # With the interface of the minimum they are what the step writes in structure.init.short.
    assert set(inputs.init) == {'bot_slab', 'top_slab', 'interface'}
    assert inputs.init['interface'] == data[low]['input']['structure']


# --- what is read ---------------------------------------------------------------------------

def test_inputs_read_from_the_document(surfaces, short_surface):
    doc = document(surfaces)
    base = {'bot_slab': doc['structure']['init']['bot_slab'],
            'top_slab': doc['structure']['init']['top_slab']}
    ids = {'mid': SYSTEM, 'formula': 'Cu_Al', 'name': 'Cu111-Al111',
           'miller': [[1, 1, 1], [1, 1, 1]]}
    low, high = sites(surfaces)
    opt = doc['structure']['opt']['pes']
    inputs = inputs_of(surfaces, 'pes', 'regular')
    assert inputs.read == base | {'min': opt['min'], 'max': opt['max']}
    assert inputs.ids == ids
    assert inputs.parameters == {'pes': 'pes', 'type': 'regular',
                                 'calc_types': adhesion.CALC_TYPES, 'comp_params': {'kdens': 7.5}}
    inputs = inputs_of(surfaces, 'pes_scf', 'regular')
    assert inputs.read == base | {'min_site': low, 'max_site': high,
                                  'shifts': doc['highsym']['inter_unique']}
    assert inputs.parameters['pes'] == 'pes_scf' and inputs.parameters['type'] == 'regular'
    # The vendored code changes the structures it is given: the entries read stay as stored.
    short = document(short_surface)
    low, high = sites(short_surface)
    inputs = inputs_of(short_surface, 'pes_scf', 'short')
    data = short['pes_scf']['data']
    assert inputs.read == base | {'min_site': low, 'max_site': high,
                                  'min_input': data[low]['input']['structure'],
                                  'max_input': data[high]['input']['structure']}
    assert inputs.ids == ids


def test_expected_entries(surfaces, short_surface):
    settings = load_run(surfaces)
    match = match_provenance(settings)
    relaxed = pes.RelaxedPesStep().provenance(surfaces, SYSTEM)
    static = pes.StaticPesStep().provenance(surfaces, SYSTEM)
    bot, top = ('structure', 'init', 'bot_slab'), ('structure', 'init', 'top_slab')
    expected = adhesion.adhesion_expected(surfaces, SYSTEM, with_variant(settings, 'pes', 'regular'))
    assert expected == {bot: match, top: match, ('structure', 'opt', 'pes', 'min'): relaxed,
                        ('structure', 'opt', 'pes', 'max'): relaxed}
    expected = adhesion.adhesion_expected(surfaces, SYSTEM,
                                          with_variant(settings, 'pes_scf', 'regular'))
    assert expected == {bot: match, top: match, ('highsym',): match,
                        ('pes_scf', 'energy', 'min_site'): static,
                        ('pes_scf', 'energy', 'max_site'): static}
    settings = load_run(short_surface)
    static = pes.StaticPesStep().provenance(short_surface, SYSTEM)
    expected = adhesion.adhesion_expected(short_surface, SYSTEM,
                                          with_variant(settings, 'pes_scf', 'short'))
    assert expected == {bot: match, top: match, ('pes_scf', 'energy', 'min_site'): static,
                        ('pes_scf', 'energy', 'max_site'): static, ('pes_scf', 'data'): static}


def test_the_step_follows_the_settings_of_the_run(surfaces_copy):
    step = adhesion.AdhesionStep()
    relaxed = step.provenance(surfaces_copy, SYSTEM)
    assert relaxed.parameters['pes'] == 'pes'
    assert list(step.expected(surfaces_copy, SYSTEM))[2] == ('structure', 'opt', 'pes', 'min')
    assert [calc.calc_id for calc in step.calculations(surfaces_copy, SYSTEM)] == \
        [f'adhesion/{label}' for label in LABELS]
    edit_config(surfaces_copy, 'pes = "pes"', 'pes = "pes_scf"')
    static = step.provenance(surfaces_copy, SYSTEM)
    assert static.parameters['pes'] == 'pes_scf' and static != relaxed
    assert list(step.expected(surfaces_copy, SYSTEM))[2] == ('highsym',)


def test_provenance_follows_the_surface_and_the_kind(surfaces, short_surface):
    found = {}
    for run_dir, variant in ((surfaces, ('pes', 'regular')), (surfaces, ('pes_scf', 'regular')),
                             (short_surface, ('pes_scf', 'short'))):
        inputs = inputs_of(run_dir, *variant)
        plan = {calc.calc_id: calc.record.fingerprint
                for calc in adhesion.adhesion_calculations(SYSTEM, inputs)}
        found[variant] = (plan, adhesion.adhesion_provenance(inputs, plan))
    assert len({provenance.fingerprint for _, provenance in found.values()}) == 3
    relaxed, static = found[('pes', 'regular')][0], found[('pes_scf', 'regular')][0]
    # The same slabs, other interfaces: a change of surface changes the interfaces only.
    assert relaxed['adhesion/bot_slab'] == static['adhesion/bot_slab']
    assert relaxed['adhesion/top_slab'] == static['adhesion/top_slab']
    assert relaxed['adhesion/interface_min'] != static['adhesion/interface_min']
    assert relaxed['adhesion/interface_max'] != static['adhesion/interface_max']
    provenance = found[('pes', 'regular')][1]
    assert provenance.step == 'adhesion' and provenance.calculations == relaxed
    assert set(provenance.inputs) == {'bot_slab', 'top_slab', 'min', 'max'}


# --- errors ---------------------------------------------------------------------------------

def test_without_match_the_step_stops(tmp_path):
    run_dir = adhesion_run(tmp_path)
    with pytest.raises(PrerequisiteError, match='run match first'):
        adhesion.AdhesionStep().calculations(run_dir, SYSTEM)


@pytest.mark.parametrize('pes_name, kind, missing', [
    ('pes', 'regular', 'structure.opt.pes.min, structure.opt.pes.max; run pes first'),
    ('pes_scf', 'regular', 'pes_scf.energy.min_site, pes_scf.energy.max_site; run pes_scf first'),
])
def test_without_the_surface_the_step_stops(matched, pes_name, kind, missing):
    with pytest.raises(PrerequisiteError, match=re.escape(f'entries missing: {missing}')):
        inputs_of(matched, pes_name, kind)


def test_without_the_input_structures_of_the_sites_the_step_stops(short_surface, tmp_path):
    run_dir = tmp_path / 'run'
    shutil.copytree(short_surface, run_dir)
    low, _ = sites(run_dir)
    with SystemWriter(system_dir(run_dir, SYSTEM)) as writer:
        writer.document.replace(('pes_scf', 'data', low, 'input'), {})
        writer.save()
    with pytest.raises(PrerequisiteError, match=rf'pes_scf\.data\.{low}\.input\.structure.*run '
                                                 'pes_scf first'):
        inputs_of(run_dir, 'pes_scf', 'short')


def test_a_site_without_shifts_stops_the_step(surfaces_copy):
    with SystemWriter(system_dir(surfaces_copy, SYSTEM)) as writer:
        writer.document.set(('pes_scf', 'energy', 'min_site'), 'nowhere')
        writer.save()
    with pytest.raises(PrerequisiteError, match='site nowhere.*highsym.inter_unique'):
        inputs_of(surfaces_copy, 'pes_scf', 'regular')


def test_short_needs_the_static_surface(surfaces):
    with pytest.raises(SettingsError, match="interface.adhesion.type.*'short' needs pes = 'pes_scf'"):
        inputs_of(surfaces, 'pes', 'short')


def test_k_points_from_their_density_only(surfaces):
    settings = load_run(surfaces)
    settings = dataclasses.replace(settings, k_points=dataclasses.replace(
        settings.k_points, mode='automatic', grid=[4, 4, 1], offset=[0, 0, 0]))
    with pytest.raises(SettingsError, match=r'k_points\.mode.*kdens'):
        adhesion.adhesion_inputs(surfaces, SYSTEM, settings)


def test_not_done_before_the_analysis(surfaces_copy, matched):
    assert adhesion.AdhesionStep().done(surfaces_copy, SYSTEM) is False
    # Nor without the PES: the inputs are not even read.
    assert adhesion.AdhesionStep().done(matched, SYSTEM) is False


def test_entries_of_each_kind():
    assert adhesion.data_entry('regular') == ('data',)
    assert adhesion.data_entry('short') == ('data', 'short')
    regular = adhesion.adhesion_leaves('regular')
    assert regular == [('adhesion', 'regular')] + [('data', label) for label in LABELS] + \
        [('structure', 'opt', 'regular', label) for label in LABELS]
    short = adhesion.adhesion_leaves('short')
    assert short == [('adhesion', 'short')] + [('data', 'short', label) for label in LABELS] + \
        [('structure', 'opt', 'short', label) for label in LABELS] + \
        [('structure', 'init', 'short', key) for key in ('bot_slab', 'top_slab', 'interface')]


# --- analysis -------------------------------------------------------------------------------

# Energies of the four calculations (eV): the interface at the lowest site binds by 2 eV, at
# the highest site by 1.5 eV, so the corrugation is 0.5 eV.
ENERGIES = {'bot_slab': -1000.0, 'top_slab': -1008.0, 'interface_min': -2010.0,
            'interface_max': -2009.5}
JOULE = 1.602176634e-19  # one eV in joule
# Keys of the results of each calculation in the document, as TRANSFER_FROM names them.
RESULT_KEYS = {'nsites', 'density', 'energy', 'energy_per_atom', 'bandgap', 'cbm', 'vbm',
               'is_gap_direct', 'forces', 'stress', 'pseudo_potential', 'dir_name', 'task_id',
               'task_label'}
NO_RESULT = ('bandgap', 'cbm', 'vbm', 'is_gap_direct', 'forces', 'stress', 'pseudo_potential')


def configure(run_dir, pes_name, kind):
    """Write ``[interface.adhesion]`` of the variant in the settings of the run."""
    if pes_name != 'pes':
        edit_config(run_dir, 'pes = "pes"', f'pes = "{pes_name}"')
    if kind != 'regular':
        edit_config(run_dir, 'type = "regular"', f'type = "{kind}"')


def finish_adhesion(run_dir, energies=None, prepare=True, **changes):
    """Prepare the four calculations and complete them with synthetic results.

    The final structure of each is its input, as a plain structure, moved by
    0.01 angstrom along z. With ``prepare`` False the calculations are already
    prepared.

    Returns
    -------
    list of str
        ``calc_id`` of the calculations, in the order of the labels.
    """
    step = adhesion.AdhesionStep()
    inputs = step.inputs(run_dir, SYSTEM)
    if prepare:
        step.prepare(run_dir, SYSTEM)
    sdir = system_dir(run_dir, SYSTEM)
    energies = ENERGIES if energies is None else energies
    for label, _, structure, _ in inputs.points:
        final = structure.copy()
        final.translate_sites(list(range(len(final))), [0, 0, 0.01], frac_coords=False,
                              to_unit_cell=False)
        final = Structure(final.lattice, final.species, final.frac_coords)
        energy = energies[label]
        results = {'nsites': len(structure), 'input': {'structure': structure.as_dict()},
                   'output': {'structure': final.as_dict(), 'density': float(final.density),
                              'energy': energy, 'energy_per_atom': energy / len(structure)}}
        record = store.read_record(sdir, f'adhesion/{label}')
        store.write_record(sdir, dataclasses.replace(
            record, status=store.Status('ended', 'converged', 'converged', 'ok'),
            results=jsanitize(results, strict=False), observed={}, **changes))
    return [f'adhesion/{label}' for label in LABELS]


@pytest.fixture(scope='module', params=VARIANTS, ids=lambda variant: '-'.join(variant))
def analysed(request, surfaces, short_surface, tmp_path_factory):
    """The run of each variant with the four calculations completed and analysed.

    Returns
    -------
    run_dir, calc_ids, pes_name, kind
    """
    pes_name, kind = request.param
    run_dir = tmp_path_factory.mktemp('analysed') / 'run'
    shutil.copytree(short_surface if kind == 'short' else surfaces, run_dir)
    configure(run_dir, pes_name, kind)
    calc_ids = finish_adhesion(run_dir)
    adhesion.AdhesionStep().analyse(run_dir, SYSTEM, calc_ids)
    return run_dir, calc_ids, pes_name, kind


@pytest.fixture
def analysed_copy(analysed, tmp_path):
    copied = tmp_path / 'analysed'
    shutil.copytree(analysed[0], copied)
    return (copied, *analysed[1:])


def data_of(doc, kind):
    return doc['data'] if kind == 'regular' else doc['data']['short']


def system_bytes(run_dir):
    return (system_dir(run_dir, SYSTEM) / 'system.json').read_bytes()


def test_results_of_every_calculation(analysed):
    run_dir, calc_ids, _, kind = analysed
    doc = document(run_dir)
    data = data_of(doc, kind)
    assert set(data) == set(LABELS)
    sizes = {label: len(structure) for label, _, structure, _ in
             adhesion.AdhesionStep().inputs(run_dir, SYSTEM).points}
    for label, calc_id in zip(LABELS, calc_ids):
        results = data[label]
        assert set(results) == RESULT_KEYS
        assert results['energy'] == ENERGIES[label]
        assert results['nsites'] == sizes[label]
        assert results['energy_per_atom'] == pytest.approx(ENERGIES[label] / sizes[label])
        assert results['task_label'] == results['task_id'] == results['dir_name'] == calc_id
        # What a calculation does not give is written as None.
        assert all(results[key] is None for key in NO_RESULT)
    # Regular and short results are in different entries.
    assert ('short' in doc['data']) == (kind == 'short')


def test_final_structures(analysed):
    run_dir, _, _, kind = analysed
    doc = document(run_dir)
    found = structures_of(adhesion.AdhesionStep().inputs(run_dir, SYSTEM))
    opt = doc['structure']['opt'][kind]
    assert set(opt) == set(LABELS)
    for label in LABELS:
        final = structure_of(opt[label])
        assert final.lattice == found[label].lattice and len(final) == len(found[label])
        assert np.allclose(final.cart_coords, found[label].cart_coords + [0, 0, 0.01])
    # The slabs are slabs again, with the Miller indices of those they started from.
    init = doc['structure']['init'] if kind == 'regular' else doc['structure']['init']['short']
    for label in ('bot_slab', 'top_slab'):
        assert opt[label]['@class'] == 'Slab'
        assert opt[label]['miller_index'] == init[label]['miller_index']
        assert opt[label]['oriented_unit_cell'] == init[label]['oriented_unit_cell']
    assert opt['interface_min']['@class'] == opt['interface_max']['@class'] == 'Structure'


def test_structures_init_short_only_with_short(analysed):
    run_dir, _, _, kind = analysed
    init = document(run_dir)['structure']['init']
    assert ('short' in init) == (kind == 'short')
    if kind == 'short':
        inputs = adhesion.AdhesionStep().inputs(run_dir, SYSTEM)
        # As the document holds them: tuples are read back as lists, and the
        # empty dicts are not written (A82).
        assert init['short'] == stored(json.loads(json.dumps(inputs.init)))


def test_adhesion_energy_and_corrugation(analysed):
    run_dir, _, _, kind = analysed
    doc = document(run_dir)
    assert set(doc['adhesion']) == {kind}
    values = doc['adhesion'][kind]
    assert set(values) == {'min_site', 'max_site', 'corrugation_ev', 'corrugation_jm2'}
    assert set(values['min_site']) == set(values['max_site']) == \
        {'adhesion_energy_ev', 'adhesion_energy_jm2'}
    # E(interface) - E(top) - E(bottom): negative, the interface binds.
    assert values['min_site']['adhesion_energy_ev'] == pytest.approx(-2.0)
    assert values['max_site']['adhesion_energy_ev'] == pytest.approx(-1.5)
    assert values['corrugation_ev'] == pytest.approx(0.5)
    # J/m^2 on the area of the cell of the relaxed interface at the lowest site.
    cell = structure_of(doc['structure']['opt'][kind]['interface_min']).lattice.matrix
    area = np.linalg.norm(np.cross(cell[0], cell[1])) * 1e-20
    assert values['min_site']['adhesion_energy_jm2'] == pytest.approx(-2.0 * JOULE / area)
    assert values['max_site']['adhesion_energy_jm2'] == pytest.approx(-1.5 * JOULE / area)
    assert values['corrugation_jm2'] == pytest.approx(0.5 * JOULE / area)
    assert values['min_site']['adhesion_energy_jm2'] < 0 < values['corrugation_jm2']


# --- sign and units, on handmade energies -----------------------------------------------

def memory_store(tmp_path, kind, energies, cell=(4.0, 5.0, 20.0)):
    """A document with the energies of the four calculations and a cell of known area."""
    interface = Structure(Lattice.orthorhombic(*cell), ['Cu'], [[0, 0, 0]])
    ids = {'mid': 'a_b', 'miller': [[1, 1, 1], [1, 1, 1]]}
    data = {label: {'energy': energy} for label, energy in energies.items()}
    document_ = {**ids, 'data': data if kind == 'regular' else {'short': data},
                 'structure': {'opt': {kind: {'interface_min': interface.as_dict()}}}}
    return LegacyStore(tmp_path, SystemDocument(document_), COLLECTION), ids


@pytest.mark.parametrize('kind', ['regular', 'short'])
@pytest.mark.parametrize('energies, adhesion_min, adhesion_max, corrugation', [
    # Bound at both sites, more at the lowest.
    ({'bot_slab': -4.0, 'top_slab': -3.0, 'interface_min': -10.0, 'interface_max': -9.0},
     -3.0, -2.0, 1.0),
    # Not bound at the highest site: positive.
    ({'bot_slab': -4.0, 'top_slab': -3.0, 'interface_min': -7.5, 'interface_max': -6.0},
     -0.5, 1.0, 1.5),
])
def test_sign_and_units_of_the_adhesion_energy(tmp_path, kind, energies, adhesion_min,
                                               adhesion_max, corrugation):
    memory, ids = memory_store(tmp_path, kind, energies)
    written = adhesion.calculate_adhesion(memory, ids, kind)
    assert list(written) == [kind]
    values = written[kind]
    # The area of the cell is 4 x 5 = 20 angstrom^2.
    per_ev = JOULE / 20e-20
    for site, expected in (('min_site', adhesion_min), ('max_site', adhesion_max)):
        assert values[site]['adhesion_energy_ev'] == pytest.approx(expected)
        assert values[site]['adhesion_energy_jm2'] == pytest.approx(expected * per_ev)
    assert values['corrugation_ev'] == pytest.approx(corrugation)
    assert values['corrugation_jm2'] == pytest.approx(corrugation * per_ev)
    # Computing writes nothing; the entry is written apart.
    assert memory.document.get(('adhesion',)) is None
    adhesion.update_adhesion(memory, ids, written)
    assert memory.document.get(('adhesion',)) == written


def test_corrugation_is_the_highest_minus_the_lowest(tmp_path):
    # The maximum below the minimum gives a negative corrugation: nothing reorders the sites.
    energies = {'bot_slab': 0.0, 'top_slab': 0.0, 'interface_min': -1.0, 'interface_max': -3.0}
    memory, ids = memory_store(tmp_path, 'regular', energies)
    values = adhesion.calculate_adhesion(memory, ids, 'regular')['regular']
    assert values['corrugation_ev'] == pytest.approx(-2.0)
    assert values['min_site']['adhesion_energy_ev'] == pytest.approx(-1.0)
    assert values['max_site']['adhesion_energy_ev'] == pytest.approx(-3.0)


def test_a_calculation_without_energy(tmp_path):
    energies = {'bot_slab': -4.0, 'top_slab': -3.0, 'interface_min': -10.0, 'interface_max': None}
    memory, ids = memory_store(tmp_path, 'regular', energies)
    with pytest.raises(StoreError, match='no energy in the results of interface_max'):
        adhesion.calculate_adhesion(memory, ids, 'regular')
    assert memory.document.get(('adhesion',)) is None


# --- provenance and the state of the step -----------------------------------------------------

def provenances(run_dir, kind):
    document_ = store.read_system(system_dir(run_dir, SYSTEM))
    return {entry: document_.provenance(entry) for entry in adhesion.adhesion_leaves(kind)}


def test_provenance_on_the_entries_written(analysed):
    run_dir, calc_ids, _, kind = analysed
    expected = adhesion.AdhesionStep().provenance(run_dir, SYSTEM)
    found = provenances(run_dir, kind)
    assert set(found) == set(adhesion.adhesion_leaves(kind))
    for entry, provenance in found.items():
        assert provenance is not None, entry
        # The same provenance on every entry, with the results consumed.
        assert provenance == dataclasses.replace(expected, results=provenance.results)
        assert set(provenance.results) == set(calc_ids) == set(provenance.calculations)
        assert provenance.step == 'adhesion' and provenance.parameters['type'] == kind
    # The entries of match and of the PES that were read carry theirs, unchanged.
    assert MatchStep().done(run_dir, SYSTEM)
    assert all(step().done(run_dir, SYSTEM) for name, step in PES_STEPS.items()
               if name == 'pes_scf' or kind == 'regular' and name == 'pes')


def test_done_only_after_the_analysis(surfaces_copy):
    step = adhesion.AdhesionStep()
    assert not step.done(surfaces_copy, SYSTEM)
    calc_ids = finish_adhesion(surfaces_copy)
    assert not step.done(surfaces_copy, SYSTEM)
    step.analyse(surfaces_copy, SYSTEM, calc_ids)
    assert step.done(surfaces_copy, SYSTEM)


def test_done_after_the_analysis(analysed):
    run_dir, _, _, _ = analysed
    assert adhesion.AdhesionStep().done(run_dir, SYSTEM)


def test_done_follows_the_settings(analysed_copy):
    run_dir, _, _, _ = analysed_copy
    edit_config(run_dir, 'kdens = 7.5', 'kdens = 8.0')
    assert not adhesion.AdhesionStep().done(run_dir, SYSTEM)


def test_done_follows_the_results(analysed_copy):
    run_dir, calc_ids, _, _ = analysed_copy
    sdir = system_dir(run_dir, SYSTEM)
    record = store.read_record(sdir, calc_ids[0])
    results = {**record.results, 'output': {**record.results['output'], 'energy': -999.0}}
    store.write_record(sdir, dataclasses.replace(record, results=results))
    assert not adhesion.AdhesionStep().done(run_dir, SYSTEM)


def test_done_follows_the_surface(analysed_copy):
    run_dir, _, pes_name, kind = analysed_copy
    # The surface is analysed again with other sites: the structures that the
    # adhesion started from are no longer those of the surface.
    with SystemWriter(system_dir(run_dir, SYSTEM)) as writer:
        if pes_name == 'pes':
            writer.document.set(('structure', 'opt', 'pes', 'min', 'charge'), 1)
        else:
            writer.document.set(('pes_scf', 'energy', 'min_site'), 'nowhere')
        writer.save()
    assert not adhesion.AdhesionStep().done(run_dir, SYSTEM)


def test_done_without_the_surface(analysed_copy):
    run_dir, _, pes_name, _ = analysed_copy
    # What the surface wrote and the adhesion reads: the structures of the relaxed PES, the
    # entry of the static one.
    entry = ('structure', 'opt', 'pes') if pes_name == 'pes' else ('pes_scf',)
    with SystemWriter(system_dir(run_dir, SYSTEM)) as writer:
        writer.document.replace(entry, {})
        writer.save()
    assert not adhesion.AdhesionStep().done(run_dir, SYSTEM)


def test_analysis_again_gives_the_same_document(analysed_copy):
    run_dir, calc_ids, _, _ = analysed_copy
    before = document(run_dir)
    adhesion.AdhesionStep().analyse(run_dir, SYSTEM, calc_ids)
    assert document(run_dir) == before
    assert adhesion.AdhesionStep().done(run_dir, SYSTEM)


def test_analysis_leaves_the_other_kind_alone(analysed_copy):
    run_dir, calc_ids, _, kind = analysed_copy
    other = 'short' if kind == 'regular' else 'regular'
    with SystemWriter(system_dir(run_dir, SYSTEM)) as writer:
        writer.document.set(('adhesion', other), {'min_site': {'adhesion_energy_ev': -9.0}})
        writer.document.set(adhesion.data_entry(other) + ('bot_slab',), {'energy': -9.0})
        writer.document.set(('structure', 'opt', other, 'bot_slab'), {'marker': 1})
        writer.save()
    before = document(run_dir)
    adhesion.AdhesionStep().analyse(run_dir, SYSTEM, calc_ids)
    assert document(run_dir) == before
    assert document(run_dir)['adhesion'][other] == {'min_site': {'adhesion_energy_ev': -9.0}}


def test_message_of_the_analysis(surfaces_copy, caplog):
    calc_ids = finish_adhesion(surfaces_copy)
    with caplog.at_level(logging.INFO, logger=adhesion.__name__):
        adhesion.AdhesionStep().analyse(surfaces_copy, SYSTEM, calc_ids)
    doc = document(surfaces_copy)
    values, low, high = doc['adhesion']['regular'], doc['pes']['energy']['min_site'], \
        doc['pes']['energy']['max_site']
    lines = [record.getMessage() for record in caplog.records if record.name == adhesion.__name__]
    assert lines == [
        f'{SYSTEM}: adhesion: regular adhesion on the complete slabs, from the PES pes (relaxed '
        f'along z), at {low} (lowest energy) and {high} (highest energy)',
        f'{SYSTEM}: adhesion energy at {low}: '
        f'{values["min_site"]["adhesion_energy_ev"]:.6f} eV, '
        f'{values["min_site"]["adhesion_energy_jm2"]:.6f} J/m^2',
        f'{SYSTEM}: adhesion energy at {high}: '
        f'{values["max_site"]["adhesion_energy_ev"]:.6f} eV, '
        f'{values["max_site"]["adhesion_energy_jm2"]:.6f} J/m^2',
        f'{SYSTEM}: corrugation (maximum - minimum): {values["corrugation_ev"]:.6f} eV, '
        f'{values["corrugation_jm2"]:.6f} J/m^2',
    ]


# --- energies of different models are not combined --------------------------------------------

def change_model(model, change):
    """A copy of an energy model with one part changed."""
    model = copy.deepcopy(model)
    if change == 'version':
        model['version'] = '7.3'
    elif change == 'pseudopotential':
        model['species'][0]['md5'] = 'f' * 32
    elif change == 'k-points':
        model['k_points']['kdens'] = 4.0
    elif change == 'spin':
        model['namelists']['SYSTEM'] = [[key, 1 if key == 'nspin' else value]
                                        for key, value in model['namelists']['SYSTEM']]
    elif change == 'cutoff':
        model['namelists']['SYSTEM'] = [[key, 30.0 if key == 'ecutwfc' else value]
                                        for key, value in model['namelists']['SYSTEM']]
    elif change == 'smearing':
        model['namelists']['SYSTEM'] = [[key, 'gauss' if key == 'smearing' else value]
                                        for key, value in model['namelists']['SYSTEM']]
    return model


@pytest.mark.parametrize('change', ['version', 'pseudopotential', 'k-points', 'spin', 'cutoff',
                                    'smearing'])
def test_energies_of_different_models_are_not_combined(surfaces_copy, change):
    calc_ids = finish_adhesion(surfaces_copy)
    sdir = system_dir(surfaces_copy, SYSTEM)
    record = store.read_record(sdir, calc_ids[1])
    changed = change_model(record.energy_model, change)
    assert changed != record.energy_model
    store.write_record(sdir, dataclasses.replace(record, energy_model=changed))
    before = system_bytes(surfaces_copy)
    with pytest.raises(StoreError, match='different energy models'):
        adhesion.AdhesionStep().analyse(surfaces_copy, SYSTEM, calc_ids)
    assert system_bytes(surfaces_copy) == before


def test_energies_of_another_engine_are_not_combined(surfaces_copy):
    calc_ids = finish_adhesion(surfaces_copy)
    sdir = system_dir(surfaces_copy, SYSTEM)
    record = store.read_record(sdir, calc_ids[2])
    store.write_record(sdir, dataclasses.replace(record, engine=store.Engine('qe', 'other.x')))
    before = system_bytes(surfaces_copy)
    with pytest.raises(StoreError, match='different energy models'):
        adhesion.AdhesionStep().analyse(surfaces_copy, SYSTEM, calc_ids)
    assert system_bytes(surfaces_copy) == before


def test_an_energy_model_not_known_is_not_combined(surfaces_copy):
    # A record whose pseudopotentials were not checked has an energy model not fully known.
    calc_ids = finish_adhesion(surfaces_copy)
    sdir = system_dir(surfaces_copy, SYSTEM)
    record = store.read_record(sdir, calc_ids[0])
    model = copy.deepcopy(record.energy_model)
    model['species'][0]['md5'] = None
    store.write_record(sdir, dataclasses.replace(record, energy_model=model))
    before = system_bytes(surfaces_copy)
    with pytest.raises(StoreError, match='energy model not known'):
        adhesion.AdhesionStep().analyse(surfaces_copy, SYSTEM, calc_ids)
    assert system_bytes(surfaces_copy) == before


# --- the command line and the orchestrator -----------------------------------------------------

RELAX_DONE = store.Status('ended', 'converged', 'converged', 'ok')


def cli(run_dir, *argv):
    return main([*argv, '-r', str(run_dir)])


def run_surface(run_dir, name, capsys):
    """match analysed, then the PES step ``name`` prepared, completed and analysed."""
    assert cli(run_dir, 'analyse', 'match') == 0
    assert cli(run_dir, 'prepare', name) == 0
    calc_ids = [calc.calc_id for calc in PES_STEPS[name]().calculations(run_dir, SYSTEM)]
    for calc_id in calc_ids:
        complete(run_dir, calc_id, status=status_of(name))
    finish(run_dir, name, prepare=False)
    assert cli(run_dir, 'analyse', name) == 0
    capsys.readouterr()


@pytest.mark.parametrize('pes_name, kind', VARIANTS)
def test_the_step_from_the_command_line(tmp_path, capsys, pes_name, kind):
    """The static surface alone is enough: the relaxed one is never computed."""
    run_dir = pes_run(tmp_path, steps=('match', pes_name, 'adhesion'))
    if kind == 'short':
        edit_config(run_dir, 'short = false', 'short = true')
    configure(run_dir, pes_name, kind)
    run_surface(run_dir, pes_name, capsys)
    assert cli(run_dir, 'prepare', 'adhesion') == 0
    assert '4 calculations prepared' in capsys.readouterr().out
    for label in LABELS:
        complete(run_dir, f'adhesion/{label}', status=RELAX_DONE)
    finish_adhesion(run_dir, prepare=False)
    assert cli(run_dir, 'analyse', 'adhesion') == 0
    assert f'{SYSTEM}/adhesion: completed' in capsys.readouterr().out
    assert adhesion.AdhesionStep().done(run_dir, SYSTEM)
    doc = document(run_dir)
    assert set(doc['adhesion']) == {kind}
    assert ('pes' in doc) == (pes_name == 'pes')


def test_the_step_waits_for_the_surface(tmp_path, capsys):
    run_dir = pes_run(tmp_path, steps=('match', 'pes', 'adhesion'))
    assert cli(run_dir, 'analyse', 'match') == 0
    capsys.readouterr()
    assert cli(run_dir, 'prepare', 'adhesion') == 1
    out = capsys.readouterr().out
    assert f'{SYSTEM}/adhesion: blocked' in out
    assert 'structure.opt.pes.min, structure.opt.pes.max' in out
    # The static surface is not the one that this run reads.
    run_surface(run_dir, 'pes_scf', capsys)
    assert cli(run_dir, 'prepare', 'adhesion') == 1
    assert 'structure.opt.pes.min' in capsys.readouterr().out


def test_the_orchestrator_runs_the_chain(tmp_path, fake_slurm):
    """match, the static surface, then the adhesion that reads it: a pass at a time."""
    run_dir = pes_run(tmp_path, steps=('adhesion', 'pes_scf', 'match'))
    configure(run_dir, 'pes_scf', 'regular')
    orch = build_orchestrator(run_dir, groups=2)
    assert orch.order == ['match', 'pes_scf', 'adhesion']

    def statuses():
        return {name: orch.entry(SYSTEM, name)['status'] for name in orch.order}

    def settle(step, calc_ids, write):
        """The workers have run the calculations of ``step``: results and finished jobs."""
        for calc_id in calc_ids:
            complete(run_dir, calc_id, status=write)
        fake_slurm.states.update({job_id: 'COMPLETED' for job_id in orch.manager.jobs})

    orch.run_pass()
    assert statuses() == {'match': 'completed', 'pes_scf': 'running', 'adhesion': 'waiting'}
    surface = [calc.calc_id for calc in pes.StaticPesStep().calculations(run_dir, SYSTEM)]
    settle('pes_scf', surface, status_of('pes_scf'))
    finish(run_dir, 'pes_scf', prepare=False)
    orch.run_pass()
    assert statuses() == {'match': 'completed', 'pes_scf': 'completed', 'adhesion': 'running'}
    four = [f'adhesion/{label}' for label in LABELS]
    assert [calc.calc_id for calc in adhesion.AdhesionStep().calculations(run_dir, SYSTEM)] == four
    settle('adhesion', four, RELAX_DONE)
    finish_adhesion(run_dir, prepare=False)
    report = orch.run_pass()
    assert report.success and set(statuses().values()) == {'completed'}
    assert adhesion.AdhesionStep().done(run_dir, SYSTEM)
    assert set(document(run_dir)['adhesion']) == {'regular'}


# --- errors of the analysis -------------------------------------------------------------------

def test_incomplete_record_stops_the_analysis(surfaces_copy):
    calc_ids = finish_adhesion(surfaces_copy)
    sdir = system_dir(surfaces_copy, SYSTEM)
    record = store.read_record(sdir, calc_ids[3])
    store.write_record(sdir, dataclasses.replace(
        record, status=store.Status('ended', 'converged', 'not_converged', None), results=None,
        observed=None))
    before = system_bytes(surfaces_copy)
    with pytest.raises(StoreError, match=rf'{re.escape(calc_ids[3])}: not complete'):
        adhesion.AdhesionStep().analyse(surfaces_copy, SYSTEM, calc_ids)
    assert system_bytes(surfaces_copy) == before


def test_calculations_of_the_settings_only(surfaces_copy):
    calc_ids = finish_adhesion(surfaces_copy)
    with pytest.raises(ValueError, match='not those of the settings'):
        adhesion.AdhesionStep().analyse(surfaces_copy, SYSTEM, calc_ids[:-1])


def test_missing_record(surfaces_copy):
    calc_ids = finish_adhesion(surfaces_copy)
    store.record_path(system_dir(surfaces_copy, SYSTEM), calc_ids[0]).unlink()
    with pytest.raises(StoreError, match=rf'{re.escape(calc_ids[0])}: no record'):
        adhesion.AdhesionStep().analyse(surfaces_copy, SYSTEM, calc_ids)


def test_match_not_current_stops_the_analysis(surfaces_copy):
    calc_ids = finish_adhesion(surfaces_copy)
    before = system_bytes(surfaces_copy)
    edit_config(surfaces_copy, 'max_area = 160.0', 'max_area = 150.0')
    with pytest.raises(PrerequisiteError, match='structure.init.bot_slab'):
        adhesion.AdhesionStep().analyse(surfaces_copy, SYSTEM, calc_ids)
    assert system_bytes(surfaces_copy) == before


def test_surface_not_current_stops_the_analysis(surfaces_copy):
    calc_ids = finish_adhesion(surfaces_copy)
    before = system_bytes(surfaces_copy)
    # The relaxed PES read by the adhesion no longer has the settings of its analysis.
    edit_config(surfaces_copy, 'kdens = 7.5', 'kdens = 8.0')
    with pytest.raises(PrerequisiteError, match=r'structure\.opt\.pes\.min'):
        adhesion.AdhesionStep().analyse(surfaces_copy, SYSTEM, calc_ids)
    assert system_bytes(surfaces_copy) == before


def test_failure_after_the_writes_saves_nothing(surfaces_copy, monkeypatch):
    calc_ids = finish_adhesion(surfaces_copy)
    before = system_bytes(surfaces_copy)
    seen = []

    def fail(store_, ids, kind):
        seen.append(store_.document.get(('data',)))
        raise RuntimeError('change_slabs failed')

    monkeypatch.setattr(adhesion, 'change_slabs', fail)
    with pytest.raises(RuntimeError, match='change_slabs failed'):
        adhesion.AdhesionStep().analyse(surfaces_copy, SYSTEM, calc_ids)
    # The results had been written in memory ... but nothing reached the disk,
    # and the lock is released.
    assert set(seen[0]) == set(LABELS)
    assert system_bytes(surfaces_copy) == before
    assert 'data' not in document(surfaces_copy) and 'adhesion' not in document(surfaces_copy)
    assert not SystemWriter(system_dir(surfaces_copy, SYSTEM)).lock_path.exists()
    assert not adhesion.AdhesionStep().done(surfaces_copy, SYSTEM)
