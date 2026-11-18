# Author: Vito Foderà
# Created: 2026-11-02

"""Tests of ``triboworkflow.engines.qe.keys`` and ``model``.

The classes of the pw.x keys, the energy model and the fingerprint of a
calculation, and the records whose energy model is not fully known: never
reused, never combined. The settings are those of
``tests/data/qe/example_relax.toml``.
"""

import dataclasses
from pathlib import Path

import pytest
from ase.io.espresso_namelist.keys import ALL_KEYS

from triboworkflow.core import store
from triboworkflow.core import structures as st
from triboworkflow.core.settings import load_settings
from triboworkflow.core.settings.schema import QE_VERSIONS, QEParams, Species
from triboworkflow.engines.qe import keys, status
from triboworkflow.engines.qe.model import energy_model, fingerprint_data, input_fingerprint
from triboworkflow.engines.qe.pwi import KPointsCard, pw_input

DATA = Path(__file__).resolve().parent / 'data' / 'qe'
UPF = 'C.pbe-n-kjpaw_psl.1.0.0.UPF'


def example(version='7.2'):
    """Settings of the example, with [qe]."""
    settings = load_settings(DATA / 'example_relax.toml')
    settings.qe = QEParams(version)
    return settings


def written(settings, if_pos=None):
    loaded = st.load_structure(settings)
    k = settings.k_points
    return pw_input(loaded.structure, settings.atomic_species, settings.namelists,
                    KPointsCard(k.mode, tuple(k.grid), tuple(k.offset)),
                    loaded.constraints if if_pos is None else if_pos)


def fingerprint(settings, **kwargs):
    return input_fingerprint(written(settings), energy_model(settings), **kwargs)


def canonical_model(settings):
    """The energy model as check_energy_model compares it (40 and 40.0 differ)."""
    return store.canonical_json(energy_model(settings))


# --- classes of the keys ------------------------------------------------------------

def test_every_documented_key_is_in_one_class():
    documented = {k.lower() for name in ('control', 'system', 'electrons', 'ions')
                  for k in ALL_KEYS['pw'][name]}
    classes = list(keys.CLASSES.values())
    assert sum(map(len, classes)) == len(set().union(*classes)) == len(documented) == 194
    assert set().union(*classes) == documented
    assert {name: len(names) for name, names in keys.CLASSES.items()} == {
        'model': 96, 'input': 63, 'attempt': 17, 'resource': 7, 'generated': 11}


@pytest.mark.parametrize('key, cls', [
    ('ecutwfc', 'model'), ('London_C6( 2 )', 'model'), ('angle1(1)', 'model'),
    ('angle2(1)', 'model'), ('Hubbard_alpha(1)', 'model'), ('nosym', 'model'),
    ('conv_thr', 'input'), ('calculation', 'input'), ('starting_spin_angle', 'input'),
    ('space_group', 'input'), ('ion_dynamics', 'input'),
    ('nstep', 'attempt'), ('mixing_beta', 'attempt'), ('diagonalization', 'attempt'),
    ('max_seconds', 'resource'), ('pseudo_dir', 'resource'), ('prefix', 'resource'),
    ('nat', 'generated'), ('starting_magnetization(1)', 'generated'),
    # Read by pw.x but not documented, or unknown: in the energy model.
    ('B_field(3)', 'model'), ('ts_vdw', 'model'), ('mbd_vdw', 'model'), ('nr1b', 'model'),
    ('ecutwfcc', 'model'),
])
def test_class_of_a_key(key, cls):
    assert keys.key_class(key) == cls


def test_normal_key():
    assert keys.normal_key('London_C6( 1 )') == 'london_c6(1)'
    assert keys.normal_key('starting_ns_eigenvalue(1, 2, 1)') == 'starting_ns_eigenvalue(1,2,1)'


def test_versions_of_the_settings_are_those_read_from_the_output():
    assert QE_VERSIONS == status.SUPPORTED_VERSIONS


# --- energy model ------------------------------------------------------------------

def test_energy_model_of_the_example():
    settings = example()
    assert energy_model(settings) == {
        'version': '7.2',
        'species': [{'label': 'C', 'pseudopotential': UPF, 'starting_magnetization': 0.5}],
        'k_points': {'mode': 'automatic', 'grid': [8, 8, 1], 'offset': [0, 0, 0],
                     'gamma_for_unshifted_1x1x1': True},
        'namelists': {'CONTROL': [],
                      'SYSTEM': [['ecutwfc', 45.0], ['ecutrho', 360.0], ['occupations', 'smearing'],
                                 ['degauss', 0.01], ['smearing', 'mv'], ['nspin', 2],
                                 ['vdw_corr', 'none']],
                      'ELECTRONS': [], 'IONS': []},
    }
    assert store.unknown_parts(energy_model(settings)) == []


@pytest.mark.parametrize('version, unknown', [('7.2', []), (None, ['version'])])
def test_energy_model_not_known(version, unknown):
    assert store.unknown_parts(energy_model(example(version))) == unknown


def test_energy_model_without_starting_magnetization():
    settings = example()
    settings.atomic_species[0].starting_magnetization = None
    assert energy_model(settings)['species'] == [{'label': 'C', 'pseudopotential': UPF}]


def test_energy_model_of_the_k_points_settings():
    settings = example()
    settings.k_points = dataclasses.replace(settings.k_points, mode='kdens', grid=None, offset=None,
                                            kdens=7.5)
    assert energy_model(settings)['k_points'] == {'mode': 'kdens', 'kdens': 7.5,
                                                  'gamma_for_unshifted_1x1x1': True}


def test_energy_model_is_the_same_for_scf_and_relax():
    relax = example()
    scf = example()
    scf.namelists.CONTROL['calculation'] = 'scf'
    scf.namelists.CONTROL['etot_conv_thr'] = 1e-5
    assert energy_model(scf) == energy_model(relax)
    assert fingerprint(scf) != fingerprint(relax)


# --- fingerprint -------------------------------------------------------------------

@pytest.mark.parametrize('namelist, key, value', [
    ('CONTROL', 'nstep', 50), ('CONTROL', 'max_seconds', 3600), ('CONTROL', 'prefix', 'other'),
    ('CONTROL', 'outdir', 'elsewhere'), ('CONTROL', 'pseudo_dir', '/other/pseudo'),
    ('CONTROL', 'title', 'other'), ('ELECTRONS', 'mixing_beta', 0.1),
    ('ELECTRONS', 'electron_maxstep', 80), ('ELECTRONS', 'diagonalization', 'cg'),
])
def test_attempt_and_resource_keys_out_of_the_fingerprint(namelist, key, value):
    settings = example()
    before = fingerprint(settings)
    getattr(settings.namelists, namelist)[key] = value
    assert fingerprint(settings) == before


@pytest.mark.parametrize('namelist, key, value, model', [
    ('SYSTEM', 'ecutwfc', 50.0, True), ('SYSTEM', 'ecutwfc', 45, True),   # 45 is not 45.0
    ('SYSTEM', 'nspin', 1, True), ('SYSTEM', 'angle1(1)', 90.0, True),
    ('SYSTEM', 'B_field(3)', 0.01, True), ('CONTROL', 'tefield', True, True),
    ('ELECTRONS', 'conv_thr', 1e-8, False), ('CONTROL', 'tprnfor', True, False),
    ('CONTROL', 'forc_conv_thr', 1e-4, False), ('IONS', 'bfgs_ndim', 3, False),
    ('SYSTEM', 'nbnd', 40, False),
])
def test_model_and_input_keys_in_the_fingerprint(namelist, key, value, model):
    settings = example()
    before, model_before = fingerprint(settings), canonical_model(settings)
    getattr(settings.namelists, namelist)[key] = value
    assert fingerprint(settings) != before
    assert (canonical_model(settings) != model_before) == model


def test_keys_compared_without_case_and_blanks():
    settings = example()
    settings.namelists.SYSTEM['london_c6(1)'] = 1.0
    before = fingerprint(settings)
    settings.namelists.SYSTEM = {{'ecutwfc': 'ECUTWFC', 'london_c6(1)': 'London_C6( 1 )'}.get(k, k): v
                                 for k, v in settings.namelists.SYSTEM.items()}
    assert fingerprint(settings) == before


def test_ions_of_an_scf_not_in_the_fingerprint():
    settings = example()
    settings.namelists.CONTROL['calculation'] = 'scf'
    before = fingerprint(settings)
    settings.namelists.IONS['bfgs_ndim'] = 3
    assert fingerprint(settings) == before


@pytest.mark.parametrize('change, model', [
    (lambda s: setattr(s.qe, 'version', '7.3'), True),
    (lambda s: setattr(s.atomic_species[0], 'starting_magnetization', 0.2), True),
    (lambda s: setattr(s.atomic_species[0], 'pseudopotential', 'C.UPF'), True),
    (lambda s: setattr(s.k_points, 'grid', [6, 6, 1]), True),
    (lambda s: setattr(s.atomic_species[0], 'mass', 12.0), False),
    (lambda s: s.atomic_species.append(Species('H', 1.008, 'H.UPF')), True),
])
def test_identity_in_the_fingerprint(change, model):
    settings = example()
    before, model_before = fingerprint(settings), canonical_model(settings)
    change(settings)
    assert fingerprint(settings) != before
    assert (canonical_model(settings) != model_before) == model


def test_structure_and_constraints_in_the_fingerprint():
    settings = example()
    pw, model = written(settings), energy_model(settings)
    free = written(settings, if_pos=[(1, 1, 1)] * 6)
    assert input_fingerprint(free, model) != input_fingerprint(pw, model)
    loaded = st.load_structure(settings)
    moved = loaded.structure.copy()
    moved.translate_sites([0], [0.0, 0.0, 0.01], frac_coords=False)
    shifted = pw_input(moved, settings.atomic_species, settings.namelists,
                       KPointsCard('automatic', (8, 8, 1), (0, 0, 0)), loaded.constraints)
    assert input_fingerprint(shifted, model) != input_fingerprint(pw, model)


def test_calc_type_in_the_fingerprint():
    settings = example()
    pw, model = written(settings), energy_model(settings)
    assert 'calc_type' not in fingerprint_data(pw, model)
    assert fingerprint_data(pw, model, 'slab_pos_relax')['calc_type'] == 'slab_pos_relax'
    assert input_fingerprint(pw, model, 'slab_pos_relax') != input_fingerprint(pw, model)


def test_fingerprint_data_of_the_example():
    settings = example()
    data = fingerprint_data(written(settings), energy_model(settings))
    assert data['namelists']['CONTROL'] == [
        ['calculation', 'relax'], ['restart_mode', 'from_scratch'], ['tstress', False],
        ['disk_io', 'low'], ['etot_conv_thr', 0.0001], ['forc_conv_thr', 0.001]]
    assert data['namelists']['ELECTRONS'] == [['conv_thr', 1e-06]]
    assert data['namelists']['IONS'] == [['ion_dynamics', 'bfgs']]
    assert data['cards']['ATOMIC_SPECIES'] == ['ATOMIC_SPECIES', f'C 12.011 {UPF}']
    assert data['cards']['K_POINTS'] == ['K_POINTS automatic', '8 8 1  0 0 0']
    assert data['cards']['ATOMIC_POSITIONS'][-1].endswith(' 0 0 0')


# --- records with an energy model not known -----------------------------------------

def record(settings, label='aa'):
    pw, model = written(settings), energy_model(settings)
    return store.Record(system='C', step='single', label=label, calc_type='relax',
                        engine=store.Engine('qe', 'pw.x', '7.2'), energy_model=model,
                        fingerprint=input_fingerprint(pw, model),
                        status=store.Status('ended', 'converged', 'converged', 'ok'),
                        results={'energy': -10.0}, observed={})


@pytest.mark.parametrize('version, problem', [
    ('7.2', None),
    (None, 'energy model not known (version)'),
])
def test_record_reused_only_with_its_identity_known(version, problem):
    rec = record(example(version))
    assert rec.is_complete
    assert store.reuse_problem(rec, rec.fingerprint) == problem
    assert store.can_reuse(rec, rec.fingerprint) == (problem is None)


def test_energies_combined_only_with_their_identity_known():
    known = [record(example(), label) for label in ('aa', 'ab')]
    store.check_energy_model(known)
    unknown = [record(example(None), label) for label in ('aa', 'ab')]
    assert unknown[0].energy_model == unknown[1].energy_model
    with pytest.raises(store.StoreError, match='energy model not known'):
        store.check_energy_model(unknown)
    with pytest.raises(store.StoreError, match='single/ab'):
        store.check_energy_model([known[0], unknown[1]])
