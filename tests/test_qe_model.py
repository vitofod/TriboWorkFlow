# Author: Vito Foderà
# Created: 2026-11-02

"""Tests of ``triboworkflow.engines.qe.keys`` and ``model``.

The classes of the pw.x keys, the manifest of the expected UPF checksums, the
energy model and the fingerprint of a calculation, and the records whose
energy model is not fully known: never reused, never combined. The settings
are those of ``tests/data/qe/example_relax.toml``; the checksums are
illustrative.
"""

import dataclasses
from pathlib import Path

import pytest
from ase.io.espresso_namelist.keys import ALL_KEYS

from triboworkflow.core import store
from triboworkflow.core import structures as st
from triboworkflow.core.settings import load_settings
from triboworkflow.core.settings.schema import QE_VERSIONS, QEParams, SettingsError, Species
from triboworkflow.engines.qe import keys, status
from triboworkflow.engines.qe.model import (energy_model, expected_md5, fingerprint_data,
                                            input_fingerprint, read_upf_md5)
from triboworkflow.engines.qe.pwi import KPointsCard, pw_input

DATA = Path(__file__).resolve().parent / 'data' / 'qe'
UPF = 'C.pbe-n-kjpaw_psl.1.0.0.UPF'
MD5 = '0123456789abcdef0123456789abcdef'
OTHER_MD5 = 'fedcba9876543210fedcba9876543210'


def example(tmp_path, version='7.2', md5=MD5):
    """Settings of the example, with [qe] and, if md5 is given, a manifest."""
    settings = load_settings(DATA / 'example_relax.toml')
    manifest = None
    if md5 is not None:
        manifest = tmp_path / 'pseudo.md5'
        manifest.write_text(f'{md5}  {UPF}\n')
    settings.qe = QEParams(version, None if manifest is None else str(manifest))
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


# --- manifest of the checksums -------------------------------------------------------

def test_manifest_in_the_format_of_md5sum(tmp_path):
    path = tmp_path / 'pseudo.md5'
    path.write_text(f'{MD5}  Cu.UPF\n\n{OTHER_MD5} *Al pbe.UPF\n')
    assert read_upf_md5(path) == {'Cu.UPF': MD5, 'Al pbe.UPF': OTHER_MD5}


@pytest.mark.parametrize('line', [
    f'{MD5.upper()}  Cu.UPF', f'{MD5[:-1]}  Cu.UPF', f'{MD5} Cu.UPF', f'{MD5}  ',
    f' {MD5}  Cu.UPF', 'Cu.UPF', f'{MD5}\tCu.UPF',
])
def test_manifest_line_in_another_format(tmp_path, line):
    path = tmp_path / 'pseudo.md5'
    path.write_text(f'{OTHER_MD5}  Al.UPF\n{line}\n')
    with pytest.raises(ValueError, match='line 2: .*format of md5sum'):
        read_upf_md5(path)


def test_manifest_with_a_file_given_twice(tmp_path):
    path = tmp_path / 'pseudo.md5'
    path.write_text(f'{MD5}  Cu.UPF\n{MD5}  Al.UPF\n{OTHER_MD5}  Cu.UPF\n')
    with pytest.raises(ValueError, match='line 3: Cu.UPF is already at line 1'):
        read_upf_md5(path)


def test_manifest_relative_to_the_settings(tmp_path):
    settings = example(tmp_path)
    (tmp_path / 'pseudo').mkdir()
    (tmp_path / 'pseudo' / 'all.md5').write_text(f'{OTHER_MD5}  Other.UPF\n{MD5}  {UPF}\n')
    settings.source = tmp_path / 'settings.toml'
    settings.qe.upf_md5 = 'pseudo/all.md5'
    assert expected_md5(settings) == (MD5,)


def test_manifest_without_a_species(tmp_path):
    settings = example(tmp_path)
    settings.atomic_species += [Species('Fe1', 55.845, 'Fe.UPF'), Species('Fe2', 55.845, 'Fe.UPF')]
    with pytest.raises(SettingsError, match=r'qe.upf_md5: .*no checksum for Fe.UPF$'):
        expected_md5(settings)


def test_manifest_that_cannot_be_read(tmp_path):
    settings = example(tmp_path)
    settings.qe.upf_md5 = str(tmp_path / 'missing.md5')
    with pytest.raises(SettingsError, match='qe.upf_md5: cannot read the manifest'):
        expected_md5(settings)
    (tmp_path / 'bad.md5').write_text('not a checksum\n')
    settings.qe.upf_md5 = str(tmp_path / 'bad.md5')
    with pytest.raises(SettingsError, match='qe.upf_md5: .*line 1'):
        energy_model(settings)


def test_no_manifest_no_checksums(tmp_path):
    settings = example(tmp_path, md5=None)
    settings.atomic_species.append(Species('H', 1.008, 'H.UPF'))
    assert expected_md5(settings) == (None, None)


# --- energy model ------------------------------------------------------------------

def test_energy_model_of_the_example(tmp_path):
    settings = example(tmp_path)
    assert energy_model(settings) == {
        'version': '7.2',
        'species': [{'label': 'C', 'pseudopotential': UPF, 'md5': MD5,
                     'starting_magnetization': 0.5}],
        'k_points': {'mode': 'automatic', 'grid': [8, 8, 1], 'offset': [0, 0, 0],
                     'gamma_for_unshifted_1x1x1': True},
        'namelists': {'CONTROL': [],
                      'SYSTEM': [['ecutwfc', 45.0], ['ecutrho', 360.0], ['occupations', 'smearing'],
                                 ['degauss', 0.01], ['smearing', 'mv'], ['nspin', 2],
                                 ['vdw_corr', 'none']],
                      'ELECTRONS': [], 'IONS': []},
    }
    assert store.unknown_parts(energy_model(settings)) == []


@pytest.mark.parametrize('version, md5, unknown', [
    ('7.2', None, ['species[0].md5']),
    (None, MD5, ['version']),          # a single calculation with a manifest, without version
    (None, None, ['version', 'species[0].md5']),
])
def test_energy_model_not_known(tmp_path, version, md5, unknown):
    assert store.unknown_parts(energy_model(example(tmp_path, version, md5))) == unknown


def test_energy_model_without_starting_magnetization(tmp_path):
    settings = example(tmp_path)
    settings.atomic_species[0].starting_magnetization = None
    assert energy_model(settings)['species'] == [{'label': 'C', 'pseudopotential': UPF, 'md5': MD5}]


def test_energy_model_of_the_k_points_settings(tmp_path):
    settings = example(tmp_path)
    settings.k_points = dataclasses.replace(settings.k_points, mode='kdens', grid=None, offset=None,
                                            kdens=7.5)
    assert energy_model(settings)['k_points'] == {'mode': 'kdens', 'kdens': 7.5,
                                                  'gamma_for_unshifted_1x1x1': True}


def test_energy_model_is_the_same_for_scf_and_relax(tmp_path):
    relax = example(tmp_path)
    scf = example(tmp_path)
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
def test_attempt_and_resource_keys_out_of_the_fingerprint(tmp_path, namelist, key, value):
    settings = example(tmp_path)
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
def test_model_and_input_keys_in_the_fingerprint(tmp_path, namelist, key, value, model):
    settings = example(tmp_path)
    before, model_before = fingerprint(settings), canonical_model(settings)
    getattr(settings.namelists, namelist)[key] = value
    assert fingerprint(settings) != before
    assert (canonical_model(settings) != model_before) == model


def test_keys_compared_without_case_and_blanks(tmp_path):
    settings = example(tmp_path)
    settings.namelists.SYSTEM['london_c6(1)'] = 1.0
    before = fingerprint(settings)
    settings.namelists.SYSTEM = {{'ecutwfc': 'ECUTWFC', 'london_c6(1)': 'London_C6( 1 )'}.get(k, k): v
                                 for k, v in settings.namelists.SYSTEM.items()}
    assert fingerprint(settings) == before


def test_ions_of_an_scf_not_in_the_fingerprint(tmp_path):
    settings = example(tmp_path)
    settings.namelists.CONTROL['calculation'] = 'scf'
    before = fingerprint(settings)
    settings.namelists.IONS['bfgs_ndim'] = 3
    assert fingerprint(settings) == before


@pytest.mark.parametrize('change, model', [
    (lambda s: setattr(s.qe, 'version', '7.3'), True),
    (lambda s: _manifest(s, OTHER_MD5), True),
    (lambda s: setattr(s.atomic_species[0], 'starting_magnetization', 0.2), True),
    (lambda s: setattr(s.atomic_species[0], 'pseudopotential', 'C.UPF') or _manifest(s, MD5, 'C.UPF'),
     True),
    (lambda s: setattr(s.k_points, 'grid', [6, 6, 1]), True),
    (lambda s: setattr(s.atomic_species[0], 'mass', 12.0), False),
    (lambda s: s.atomic_species.append(Species('H', 1.008, 'H.UPF')) or _manifest(s, MD5, 'H.UPF'),
     True),
])
def test_identity_in_the_fingerprint(tmp_path, change, model):
    settings = example(tmp_path)
    before, model_before = fingerprint(settings), canonical_model(settings)
    change(settings)
    assert fingerprint(settings) != before
    assert (canonical_model(settings) != model_before) == model


def _manifest(settings, md5, extra=None):
    """Rewrite the manifest of the settings: md5 for the UPF of the example (and extra)."""
    path = settings.qe.upf_md5
    lines = f'{md5}  {UPF}\n' + (f'{MD5}  {extra}\n' if extra else '')
    with open(path, 'w') as f:
        f.write(lines)
    return path


def test_path_of_the_manifest_not_in_the_identity(tmp_path):
    settings = example(tmp_path)
    before, model = fingerprint(settings), energy_model(settings)
    (tmp_path / 'elsewhere').mkdir()
    moved = tmp_path / 'elsewhere' / 'checksums.txt'
    moved.write_text(f'{OTHER_MD5}  Other.UPF\n{MD5}  {UPF}\n')
    settings.qe.upf_md5 = str(moved)
    assert (fingerprint(settings), energy_model(settings)) == (before, model)


def test_structure_and_constraints_in_the_fingerprint(tmp_path):
    settings = example(tmp_path)
    pw, model = written(settings), energy_model(settings)
    free = written(settings, if_pos=[(1, 1, 1)] * 6)
    assert input_fingerprint(free, model) != input_fingerprint(pw, model)
    loaded = st.load_structure(settings)
    moved = loaded.structure.copy()
    moved.translate_sites([0], [0.0, 0.0, 0.01], frac_coords=False)
    shifted = pw_input(moved, settings.atomic_species, settings.namelists,
                       KPointsCard('automatic', (8, 8, 1), (0, 0, 0)), loaded.constraints)
    assert input_fingerprint(shifted, model) != input_fingerprint(pw, model)


def test_calc_type_in_the_fingerprint(tmp_path):
    settings = example(tmp_path)
    pw, model = written(settings), energy_model(settings)
    assert 'calc_type' not in fingerprint_data(pw, model)
    assert fingerprint_data(pw, model, 'slab_pos_relax')['calc_type'] == 'slab_pos_relax'
    assert input_fingerprint(pw, model, 'slab_pos_relax') != input_fingerprint(pw, model)


def test_fingerprint_data_of_the_example(tmp_path):
    settings = example(tmp_path)
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


@pytest.mark.parametrize('version, md5, problem', [
    ('7.2', MD5, None),
    ('7.2', None, 'energy model not known (species[0].md5)'),
    (None, MD5, 'energy model not known (version)'),
    (None, None, 'energy model not known (version, species[0].md5)'),
])
def test_record_reused_only_with_its_identity_known(tmp_path, version, md5, problem):
    rec = record(example(tmp_path, version, md5))
    assert rec.is_complete
    assert store.reuse_problem(rec, rec.fingerprint) == problem
    assert store.can_reuse(rec, rec.fingerprint) == (problem is None)


@pytest.mark.parametrize('version, md5', [('7.2', None), (None, MD5)])
def test_energies_combined_only_with_their_identity_known(tmp_path, version, md5):
    known = [record(example(tmp_path), label) for label in ('aa', 'ab')]
    store.check_energy_model(known)
    unknown = [record(example(tmp_path, version, md5), label) for label in ('aa', 'ab')]
    assert unknown[0].energy_model == unknown[1].energy_model
    with pytest.raises(store.StoreError, match='energy model not known'):
        store.check_energy_model(unknown)
    with pytest.raises(store.StoreError, match='single/ab'):
        store.check_energy_model([known[0], unknown[1]])
