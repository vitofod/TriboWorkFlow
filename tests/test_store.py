# Author: Vito Foderà
# Created: 2026-10-23

"""Tests of ``triboworkflow.core.store``.

Atomic writes, fingerprints, records and their reuse, the aggregated document
with the provenance of its entries and its single writer, writers and readers
in separate processes, and the historic database calls of ``legacy.py``.

Notes
-----
To remove: the expected behaviour of the historic calls follows TribChem's
``FireTaskTribChem`` methods over ``highput/database/query.py`` and
``dbtools.py`` (``$set`` with the dotted keys of ``convert_dict_to_mongodb``),
as read in the code: there is no MongoDB to compare with. The paths of the
document are those written by the interface Firetasks (``pes.py:108``,
``pes.py:137``, ``wfcore.py:89``, ``run.py:1165-1176``).
"""

import copy
import dataclasses
import errno
import hashlib
import json
import logging
import math
import multiprocessing
import os
import stat
import time
from pathlib import Path

import numpy as np
import pytest

from triboworkflow.core import store
from triboworkflow.core.store import legacy
from triboworkflow.core.store.atomic import read_json, write_json

StoreError = store.StoreError
Provenance = store.Provenance

FP_A = 'sha256:' + 'a' * 64
FP_B = 'sha256:' + 'b' * 64
INTERFACES = 'PBE.interface_elements'
SYSTEM = 'Cu111_Ni111'
FLTR = {'mid': 'mp-30_mp-23', 'formula': 'Cu_Ni', 'name': SYSTEM, 'miller': [[1, 1, 1], [1, 1, 1]]}

# Fork where available: the workers are functions of this module.
MP = multiprocessing.get_context('fork' if 'fork' in multiprocessing.get_all_start_methods()
                                 else 'spawn')


def make_record(step='pes_scf', label='aa', fingerprint=FP_A, status=None, results=None,
                **kwargs):
    if status is None:
        status = store.Status('ended', 'converged', 'n/a', 'ok')
    if results is None:
        results = {'nsites': 4,
                   'output': {'energy': -10.5, 'structure': {'@class': 'Structure', 'id': label}},
                   'input': {'structure': {'@class': 'Structure', 'id': 'in'}}}
    return store.Record(system=SYSTEM, step=step, label=label, calc_type='slab_from_scratch',
                        engine=store.Engine('qe', 'pw.x', '7.2'),
                        energy_model={'pseudopotentials': {'Cu': FP_B, 'Ni': FP_B},
                                      'ecutwfc': 40.0, 'ecutrho': 320.0, 'kpoints': [6, 6, 1],
                                      'smearing': 'mv', 'nspin': 1, 'vdw_corr': 'none'},
                        fingerprint=fingerprint, status=status, results=results,
                        files={'input': 'input.pwi', 'output': 'output.pwo'}, **kwargs)


def interface_document():
    # The document created by the match step: identity, empty structure, comp_params.
    return {'mid': 'mp-30_mp-23', 'formula': 'Cu_Ni', 'miller': [[1, 1, 1], [1, 1, 1]],
            'structure': {}, 'comp_params': {'use_vdw': False}, 'name': SYSTEM}


@pytest.fixture
def system_dir(tmp_path):
    path = tmp_path / SYSTEM
    path.mkdir()
    return path


# -- atomic writes -------------------------------------------------------------

def _strict_json(text):
    # A reader that accepts standard JSON only.
    def refuse(name):
        raise ValueError(name)

    return json.loads(text, parse_constant=refuse)


def test_write_json_round_trip(tmp_path):
    path = tmp_path / 'a.json'
    write_json(path, {'tuple': (1, 2), 'nested': {'x': [1.5, None, True]}, 'text': 'Å',
                      'sigma': [math.inf, -math.inf, math.nan, np.float64('inf')]})
    data = read_json(path)
    assert data['tuple'] == [1, 2] and data['nested'] == {'x': [1.5, None, True]}
    assert data['text'] == 'Å'
    inf, minus_inf, nan, inf64 = data['sigma']
    assert inf == math.inf and minus_inf == -math.inf and math.isnan(nan) and inf64 == math.inf
    stored = _strict_json(path.read_text(encoding='utf-8'))
    assert stored['sigma'][:3] == [{'@float': 'Infinity'}, {'@float': '-Infinity'},
                                   {'@float': 'NaN'}]
    assert b'\r\n' not in path.read_bytes()
    assert list(tmp_path.glob('.*.tmp')) == []


@pytest.mark.parametrize('text', ['{"a": NaN}', '{"a": Infinity}', '{"a": {"@float": "inf"}}'])
def test_read_json_accepts_standard_json_and_known_tags_only(tmp_path, text):
    path = tmp_path / 'a.json'
    path.write_text(text)
    with pytest.raises(StoreError, match='a.json'):
        read_json(path)


def _fsync_refused_on_directories(code):
    real = os.fsync

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(code, os.strerror(code))
        real(fd)

    return fsync


@pytest.mark.skipif(os.name != 'posix', reason='directories are synchronized on POSIX only')
def test_unsupported_directory_sync_is_reported_once(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(os, 'fsync', _fsync_refused_on_directories(errno.EINVAL))
    with caplog.at_level(logging.WARNING, logger='triboworkflow.core.store.atomic'):
        write_json(tmp_path / 'a.json', {'a': 1})
        write_json(tmp_path / 'b.json', {'b': 2})
    warnings = [r for r in caplog.records if 'synchronize directories' in r.getMessage()]
    assert len(warnings) == 1 and str(tmp_path) in warnings[0].getMessage()
    assert read_json(tmp_path / 'b.json') == {'b': 2}
    monkeypatch.setattr(os, 'fsync', _fsync_refused_on_directories(errno.EIO))
    with pytest.raises(OSError):
        write_json(tmp_path / 'c.json', {'c': 3})


@pytest.mark.skipif(os.name != 'posix', reason='POSIX permissions')
def test_write_json_permissions_follow_the_umask(tmp_path):
    umask = os.umask(0o022)
    os.umask(umask)
    path = tmp_path / 'a.json'
    write_json(path, {})
    assert stat.S_IMODE(path.stat().st_mode) == 0o666 & ~umask


@pytest.mark.parametrize('value', [{1: 'a'}, {'a': np.array([1.0])}, {'a': np.int64(1)},
                                   {'a': np.bool_(True)}, {'a': {1, 2}}, {'a': Path('x')},
                                   {'a': {'@float': 'NaN'}}])
def test_write_json_rejects_values_that_would_change(tmp_path, value):
    path = tmp_path / 'a.json'
    write_json(path, {'old': 1})
    with pytest.raises(StoreError):
        write_json(path, value)
    assert read_json(path) == {'old': 1}
    assert list(tmp_path.glob('.*.tmp')) == []


def test_failed_rename_keeps_the_original(tmp_path, monkeypatch):
    path = tmp_path / 'a.json'
    write_json(path, {'old': 1})

    def fail(*args):
        raise OSError('no space left')

    monkeypatch.setattr(os, 'replace', fail)
    with pytest.raises(OSError):
        write_json(path, {'new': 2})
    monkeypatch.undo()
    assert read_json(path) == {'old': 1}
    assert list(tmp_path.glob('.*.tmp')) == []


def _crash_before_rename(path):
    os.replace = lambda *args: os._exit(3)
    write_json(path, {'new': list(range(1000))})


def test_crash_during_a_write_leaves_the_original(tmp_path):
    path = tmp_path / 'a.json'
    write_json(path, {'old': 1})
    process = MP.Process(target=_crash_before_rename, args=(path,))
    process.start()
    process.join(60)
    assert process.exitcode == 3
    assert read_json(path) == {'old': 1}
    orphans = list(tmp_path.glob('.a.json.*.tmp'))
    assert len(orphans) == 1 and json.loads(orphans[0].read_text())['new'][-1] == 999


def test_read_json_errors(tmp_path):
    path = tmp_path / 'a.json'
    path.write_text('{"a": ')
    with pytest.raises(StoreError, match='a.json'):
        read_json(path)
    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / 'missing.json')


# -- fingerprints --------------------------------------------------------------

def test_fingerprint_reference_value():
    value = {'b': [1, 2.5, True, None], 'a': 'x'}
    text = b'{"a":"x","b":[1,2.5,true,null]}'
    assert store.canonical_json(value).encode() == text
    assert store.fingerprint(value) == 'sha256:' + hashlib.sha256(text).hexdigest()
    assert store.fingerprint(value) == \
        'sha256:eefd48e492206a5b6adb10ad876ec7cc056518581617631bafb28ef496c571f9'


def test_fingerprint_does_not_depend_on_the_representation():
    base = {'a': [1, 2], 'b': {'x': 1.5, 'y': 'z'}, 'c': [[0.0, 1.0], [2.0, 3.0]]}
    variants = [
        {'c': [[0.0, 1.0], [2.0, 3.0]], 'b': {'y': 'z', 'x': 1.5}, 'a': [1, 2]},
        {'a': (1, 2), 'b': {'x': 1.5, 'y': 'z'}, 'c': ((0.0, 1.0), (2.0, 3.0))},
        {'a': np.array([1, 2]), 'b': {'x': np.float64(1.5), 'y': np.str_('z')},
         'c': np.array([[0.0, 1.0], [2.0, 3.0]])},
        {'a': [np.int64(1), np.int32(2)], 'b': {'x': 1.5, 'y': 'z'},
         'c': [np.array([0.0, 1.0]), (2.0, np.float32(3.0))]},
    ]
    for variant in variants:
        assert store.fingerprint(variant) == store.fingerprint(base)


def test_fingerprint_changes_with_every_value():
    base = {'ecutwfc': 40.0, 'kpoints': [6, 6, 1], 'smearing': 'mv', 'spin': True,
            'vdw': None, 'nested': {'x': 0.1}}
    changes = [('ecutwfc', np.nextafter(40.0, 41.0)), ('ecutwfc', 40), ('kpoints', [6, 1, 6]),
               ('kpoints', [6, 6, 1, 1]), ('smearing', 'gaussian'), ('spin', 1),
               ('vdw', 'none'), ('nested', {'x': 0.1, 'y': 0}),
               ('nested', {'x': np.nextafter(0.1, 1.0)})]
    prints = {store.fingerprint(base)}
    for key, value in changes:
        changed = copy.deepcopy(base)
        changed[key] = value
        prints.add(store.fingerprint(changed))
    missing = copy.deepcopy(base)
    del missing['vdw']
    prints.add(store.fingerprint(missing))
    assert len(prints) == len(changes) + 2


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf, np.float64('nan'),
                                   {1: 'a'}, {2, 3}, Path('x'), b'bytes', object(),
                                   {'@float': 'NaN'}])
def test_fingerprint_rejects(value):
    with pytest.raises(StoreError):
        store.fingerprint({'a': value})


def test_content_digest_accepts_non_finite_floats():
    finite = {'energy': -10.5, 'forces': [[0.0, 0.1, 0.2]]}
    assert store.content_digest(finite) == store.fingerprint(finite)
    digests = {store.content_digest({'sigma': value})
               for value in (math.nan, math.inf, -math.inf, np.float64('nan'), 1e308)}
    assert len(digests) == 4
    assert store.content_digest({'sigma': math.nan}) == store.content_digest({'sigma': np.nan})
    for value in ({'sigma': {'@float': 'NaN'}}, {1: 'a'}, {'a': object()}):
        with pytest.raises(StoreError):
            store.content_digest(value)


def test_file_digest(tmp_path):
    path = tmp_path / 'f.bin'
    path.write_bytes(b'abc' * 100_000)
    assert store.file_digest(path) == 'sha256:' + hashlib.sha256(b'abc' * 100_000).hexdigest()


@pytest.mark.parametrize('value', ['sha256:' + 'A' * 64, 'sha256:' + 'a' * 63,
                                   'md5:' + 'a' * 64, None])
def test_check_fingerprint_rejects(value):
    with pytest.raises(StoreError):
        store.check_fingerprint(value)


# -- records -------------------------------------------------------------------

def test_record_round_trip(system_dir):
    record = make_record(attempts=(store.Attempt('4521', {'mixing_beta': 0.3}),))
    path = store.write_record(system_dir, record)
    assert path == system_dir / 'pes_scf' / 'aa' / 'record.json'
    data = read_json(path)
    assert list(data) == ['schema_version', 'calc_id', 'system', 'step', 'label', 'calc_type',
                          'engine', 'energy_model', 'fingerprint', 'status', 'attempts',
                          'results', 'files']
    assert data['calc_id'] == 'pes_scf/aa' and data['schema_version'] == 1
    assert store.read_record(system_dir, 'pes_scf/aa') == record
    store.write_record(system_dir, make_record(step='adhesion', label='top_slab'))
    assert [r.calc_id for r in store.iter_records(system_dir)] == ['adhesion/top_slab',
                                                                    'pes_scf/aa']


def test_record_unknown_states_are_null():
    data = make_record(status=store.Status(), results=None).to_dict()
    data['results'] = None
    record = store.Record.from_dict(data)
    assert record.status == store.Status(None, None, None, None)
    assert record.results is None and not record.is_complete
    assert record.results_digest is None


def test_parsed_record_without_results_is_an_error():
    # A complete record always has results, so reuse and reading them agree.
    for states in COMPLETE + [('crashed', None, None, 'ok')]:
        with pytest.raises(StoreError, match='results is null'):
            dataclasses.replace(make_record(), status=store.Status(*states), results=None)
    data = make_record().to_dict()
    data['results'] = None
    with pytest.raises(StoreError, match='results is null'):
        store.Record.from_dict(data)


def test_record_keeps_non_finite_results(system_dir):
    record = make_record(results={'output': {'energy': -10.5}, 'fit': {'sigma': [math.inf, 0.1]}})
    store.write_record(system_dir, record)
    assert store.read_record(system_dir, 'pes_scf/aa') == record
    assert record.results_digest == store.content_digest(record.results)


def test_record_holds_copies():
    record = make_record()
    data = record.to_dict()
    data['energy_model']['ecutwfc'] = 1.0
    data['results']['output']['energy'] = 0.0
    assert record.energy_model['ecutwfc'] == 40.0
    assert record.results['output']['energy'] == -10.5


INVALID_RECORDS = [
    ('unknown key', lambda d: d.update(extra=1)),
    ('missing key', lambda d: d.pop('files')),
    ('schema version', lambda d: d.update(schema_version=2)),
    ('schema version bool', lambda d: d.update(schema_version=True)),
    ('calc_id', lambda d: d.update(calc_id='pes_scf/ab')),
    ('step with slash', lambda d: d.update(step='pes/scf', calc_id='pes/scf/aa')),
    ('label dot dot', lambda d: d.update(label='..', calc_id='pes_scf/..')),
    ('empty system', lambda d: d.update(system='')),
    ('process state', lambda d: d['status'].update(process='done')),
    ('scf state', lambda d: d['status'].update(scf=True)),
    ('geometry state', lambda d: d['status'].update(geometry='relaxed')),
    ('parsing state', lambda d: d['status'].update(parsing='OK')),
    ('status key', lambda d: d['status'].pop('parsing')),
    ('fingerprint', lambda d: d.update(fingerprint='abc')),
    ('empty energy model', lambda d: d.update(energy_model={})),
    ('energy model with nan', lambda d: d['energy_model'].update(ecutwfc=math.nan)),
    ('engine version', lambda d: d['engine'].update(version=7.2)),
    ('engine name', lambda d: d['engine'].update(name='')),
    ('attempt keys', lambda d: d.update(attempts=[{'job_id': '1'}])),
    ('attempts type', lambda d: d.update(attempts={})),
    ('results type', lambda d: d.update(results=[1])),
    ('absolute file', lambda d: d['files'].update(output='/tmp/output.pwo')),
    ('file outside', lambda d: d['files'].update(output='../output.pwo')),
    ('empty calc_type', lambda d: d.update(calc_type='')),
]


@pytest.mark.parametrize('name, change', INVALID_RECORDS, ids=[n for n, _ in INVALID_RECORDS])
def test_record_rejects(name, change):
    data = make_record().to_dict()
    change(data)
    with pytest.raises(StoreError):
        store.Record.from_dict(data)


def test_record_errors_name_the_file(system_dir):
    path = store.write_record(system_dir, make_record())
    data = read_json(path)
    data['status']['process'] = 'done'
    write_json(path, data)
    with pytest.raises(StoreError, match='record.json'):
        store.read_record(system_dir, 'pes_scf/aa')


def test_record_belongs_to_its_system_and_place(tmp_path, system_dir):
    other = tmp_path / 'Other'
    other.mkdir()
    with pytest.raises(StoreError):
        store.write_record(other, make_record())
    path = store.write_record(system_dir, make_record())
    target = system_dir / 'pes_scf' / 'ab' / 'record.json'
    target.parent.mkdir()
    target.write_bytes(path.read_bytes())
    with pytest.raises(StoreError, match='in place of'):
        store.read_record(system_dir, 'pes_scf/ab')
    with pytest.raises(StoreError):
        store.read_record(system_dir, 'pes_scf/aa/x')


COMPLETE = [('ended', 'converged', 'converged', 'ok'), ('ended', 'converged', 'n/a', 'ok')]
INCOMPLETE = ([(p, 'converged', 'n/a', 'ok') for p in (None, 'pending', 'running', 'crashed',
                                                       'killed')]
              + [('ended', s, 'n/a', 'ok') for s in (None, 'not_converged')]
              + [('ended', 'converged', g, 'ok') for g in (None, 'not_converged', 'max_steps')]
              + [('ended', 'converged', 'n/a', p) for p in (None, 'failed')])


@pytest.mark.parametrize('states', COMPLETE)
def test_complete_record_is_reused_only_with_its_fingerprint(states):
    record = make_record(status=store.Status(*states))
    assert record.is_complete
    assert store.can_reuse(record, FP_A)
    assert not store.can_reuse(record, FP_B)


@pytest.mark.parametrize('states', INCOMPLETE)
def test_incomplete_record_is_never_reused(states):
    record = make_record(status=store.Status(*states))
    assert not record.is_complete
    assert not store.can_reuse(record, FP_A)


SUMMARIES = [((None, None, None, None), 'unknown'),
             ((None, 'converged', 'converged', None), 'unknown'),
             (('pending', None, None, None), 'pending'),
             (('running', 'converged', None, None), 'running'),
             (('crashed', None, 'not_converged', None), 'crashed'),
             (('killed', 'converged', None, None), 'crashed'),
             (('ended', 'converged', 'converged', None), 'completed'),
             (('ended', 'converged', 'n/a', 'failed'), 'completed'),
             (('ended', 'not_converged', 'n/a', None), 'not_converged'),
             (('ended', None, 'n/a', None), 'not_converged'),
             (('ended', 'converged', 'not_converged', None), 'not_converged'),
             (('ended', 'converged', 'max_steps', None), 'not_converged'),
             (('ended', 'converged', None, None), 'not_converged')]


@pytest.mark.parametrize('states, summary', SUMMARIES)
def test_status_summary(states, summary):
    assert store.Status(*states).summary == summary


def test_status_summary_covers_its_values():
    assert {summary for _, summary in SUMMARIES} == set(store.record.SUMMARY_STATES)
    assert all(store.Status(*states).summary == 'completed' for states in COMPLETE)


def test_missing_record_is_not_reused():
    assert not store.can_reuse(None, FP_A)
    with pytest.raises(StoreError):
        store.can_reuse(make_record(), 'sha256:x')


def test_energy_models_are_not_mixed():
    first, second = make_record(label='aa'), make_record(label='ab')
    store.check_energy_model([first, second])
    cutoff = dataclasses.replace(second, energy_model={**second.energy_model, 'ecutwfc': 45.0})
    with pytest.raises(StoreError, match='pes_scf/ab'):
        store.check_energy_model([first, cutoff])
    engine = dataclasses.replace(second, engine=store.Engine('cp2k', 'cp2k.psmp'))
    with pytest.raises(StoreError):
        store.check_energy_model([first, engine])


# -- aggregated document -------------------------------------------------------

def test_leaf_merge_on_the_paths_of_the_interface_steps():
    doc = store.SystemDocument(interface_document())
    for shift, energy in (('aa', -10.0), ('ab', -9.5)):
        doc.set(['pes_scf', 'data', shift, 'output', 'energy'], energy)
        doc.set(['pes_scf', 'data', shift, 'output', 'structure'], {'@class': 'Structure'})
    grid = {'hs_unique_ev': [['aa', 0.0, 0.0, -10.0], ['ab', 0.5, 0.0, -9.5]]}
    # The summary of the PES goes next to the data, which stay.
    doc.set('pes_scf', {'energy': {'grid': grid, 'max_site': 'ab', 'min_site': 'aa'}})
    assert doc.get(['pes_scf', 'data', 'ab', 'output', 'energy']) == -9.5
    assert doc.get(['pes_scf', 'energy', 'min_site']) == 'aa'
    # The optimal structure goes below the empty structure entry.
    doc.set(['structure', 'opt', 'pes_scf', 'min'], {'@class': 'Structure'})
    assert doc.get('structure') == {'opt': {'pes_scf': {'min': {'@class': 'Structure'}}}}
    # A second summary changes its leaves only.
    doc.set('pes_scf', {'energy': {'max_site': 'aa'}})
    assert doc.get(['pes_scf', 'energy']) == {'grid': grid, 'max_site': 'aa', 'min_site': 'aa'}
    assert doc.get(['missing', 'key']) is None


def test_leaves_lists_and_empty_dicts():
    doc = store.SystemDocument({'a': {'list': [1, 2, 3], 'x': {'y': 1}}})
    doc.set('a', {'list': [4]})
    assert doc.get(['a', 'list']) == [4]
    doc.set(['a', 'x'], 5)
    assert doc.get(['a', 'x']) == 5
    before = doc.document
    doc.set('b', {})
    doc.set('a', {'z': {}})
    doc.set((), {'c': {'d': {}}})
    assert doc.document == before


def test_dict_over_a_leaf_is_an_error_and_changes_nothing():
    doc = store.SystemDocument({'a': {'x': 5, 'n': None}})
    for path, value in ((['a', 'x'], {'y': 1}), (['a', 'x', 'y'], 1), (['a', 'n', 'y'], 1),
                        ('a', {'new': 1, 'x': {'y': 1}})):
        with pytest.raises(StoreError, match='not a dict'):
            doc.set(path, value)
    assert doc.document == {'a': {'x': 5, 'n': None}}
    with pytest.raises(StoreError):
        doc.get(['a', 'x', 'y'])
    with pytest.raises(StoreError):
        doc.set((), 1)


def test_document_holds_copies():
    data = {'a': {'b': [1]}}
    doc = store.SystemDocument(data)
    data['a']['b'].append(2)
    doc.get('a')['b'].append(3)
    value = {'c': [1]}
    doc.set('x', value)
    value['c'].append(2)
    assert doc.document == {'a': {'b': [1]}, 'x': {'c': [1]}}


def test_document_must_exist_once():
    doc = store.SystemDocument()
    assert not doc.exists and doc.get('a') is None
    with pytest.raises(StoreError, match='no document'):
        doc.set('a', 1)
    doc.create({'a': 1})
    with pytest.raises(StoreError, match='already'):
        doc.create({'a': 1})


def _with_pes(system_dir, labels=('aa', 'ab')):
    for label in labels:
        store.write_record(system_dir, make_record(label=label))
    doc = store.SystemDocument(interface_document())
    doc.set(['pes_scf', 'energy'], {'min_site': 'aa', 'max_site': 'ab'})
    return doc


def _pes_provenance(**changes):
    args = {'step': 'pes_scf', 'parameters': {'pes_type': 'energy'},
            'calculations': {'pes_scf/aa': FP_A, 'pes_scf/ab': FP_A},
            'inputs': {'highsym': {'aa': [[0.0, 0.0]], 'ab': [[0.5, 0.0]]}}}
    args.update(changes)
    return Provenance.build(**args)


def _consumed(system_dir, provenance):
    # The provenance recorded by an analysis that read the current records.
    return provenance.with_results(store.read_record(system_dir, calc_id)
                                   for calc_id in provenance.calculations)


def test_is_done_requires_the_expected_provenance(system_dir):
    doc = _with_pes(system_dir)
    path = ['pes_scf', 'energy']
    expected = _pes_provenance()
    assert not store.is_done(doc, path, expected, system_dir)
    with pytest.raises(StoreError, match='with_results'):
        doc.set_provenance(path, expected)
    doc.set_provenance(path, _consumed(system_dir, expected))
    assert store.is_done(doc, path, expected, system_dir)
    for changed in (_pes_provenance(step='pes'),
                    _pes_provenance(parameters={'pes_type': 'forces'}),
                    _pes_provenance(calculations={'pes_scf/aa': FP_A, 'pes_scf/ab': FP_B}),
                    _pes_provenance(calculations={'pes_scf/aa': FP_A}),
                    _pes_provenance(inputs={'highsym': {'aa': [[0.0, 0.0]], 'ab': [[0.5, 0.1]]}})):
        assert not store.is_done(doc, path, changed, system_dir)


def test_is_done_follows_the_records(system_dir):
    doc = _with_pes(system_dir)
    path, expected = ['pes_scf', 'energy'], _pes_provenance()
    doc.set_provenance(path, _consumed(system_dir, expected))
    store.write_record(system_dir, make_record(label='ab', status=store.Status(
        'ended', 'not_converged', 'n/a', 'ok')))
    assert not store.is_done(doc, path, expected, system_dir)
    store.write_record(system_dir, make_record(label='ab', fingerprint=FP_B))
    assert not store.is_done(doc, path, expected, system_dir)
    (system_dir / 'pes_scf' / 'ab' / 'record.json').unlink()
    assert not store.is_done(doc, path, expected, system_dir)
    store.write_record(system_dir, make_record(label='ab'))
    assert store.is_done(doc, path, expected, system_dir)


def test_new_results_with_the_same_input_invalidate_the_analysis(system_dir):
    results = {'nsites': 4, 'output': {'energy': -10.0}}
    store.write_record(system_dir, make_record(label='aa', results=results))
    store.write_record(system_dir, make_record(label='ab'))
    doc = store.SystemDocument(interface_document())
    doc.set(['pes_scf', 'energy'], {'min_site': 'aa', 'min': -10.0})
    path, expected = ['pes_scf', 'energy'], _pes_provenance()
    doc.set_provenance(path, _consumed(system_dir, expected))
    assert store.is_done(doc, path, expected, system_dir)
    # The calculation is run again, or its output read again: same input, new energy.
    store.write_record(system_dir, make_record(label='aa', results={'nsites': 4,
                                                                    'output': {'energy': -11.0}}))
    assert not store.is_done(doc, path, expected, system_dir)
    assert doc.get(['pes_scf', 'energy', 'min']) == -10.0
    # The same results again, written anew, keep the analysis valid.
    store.write_record(system_dir, make_record(label='aa', results=results))
    assert store.is_done(doc, path, expected, system_dir)


def test_with_results_takes_exactly_the_reusable_records(system_dir):
    expected = _pes_provenance()
    for label in ('aa', 'ab', 'ba'):
        store.write_record(system_dir, make_record(label=label))
    aa, ab, ba = (store.read_record(system_dir, f'pes_scf/{x}') for x in ('aa', 'ab', 'ba'))
    assert set(expected.with_results([aa, ab]).results) == {'pes_scf/aa', 'pes_scf/ab'}
    for records in ([aa], [aa, ab, ba], [aa, aa, ab],
                    [aa, dataclasses.replace(ab, fingerprint=FP_B)],
                    [aa, dataclasses.replace(ab, status=store.Status('running'))]):
        with pytest.raises(StoreError):
            expected.with_results(records)
    with pytest.raises(StoreError, match='not in calculations'):
        dataclasses.replace(expected, results={'pes_scf/zz': FP_A})


def test_is_done_needs_a_non_empty_entry(system_dir):
    doc = store.SystemDocument({'mid': 'm', 'shearstr': {}, 'ppes': []})
    expected = Provenance.build('shear')
    for path in ('shearstr', 'ppes'):
        doc.set_provenance(path, expected)
        assert not store.is_done(doc, path, expected, system_dir)


def _shear_provenance(doc, **settings):
    mep = {'method': 'zts', 'optimization': 'bs_line', 'npts': 1000, 'nstepmax': 99999,
           'tol': 1e-7, 'delta': 0.01, 'seed': 0}
    mep.update(settings)
    return Provenance.build('shear', parameters={'mep': mep, 'replicate_of': [5, 5],
                                                 'density': 20, 'tol': 1e-4},
                            inputs={'pes_jm2': doc.get(['pes_scf', 'energy', 'grid']),
                                    'cell': [[2.5, 0.0], [1.25, 2.165]]})


def test_analysis_settings_and_inputs_invalidate_the_analysis(system_dir):
    doc = store.SystemDocument(interface_document())
    doc.set(['pes_scf', 'energy', 'grid', 'hs_unique_jm2'], [['aa', 0.0, 0.0, -1.2]])
    doc.set('shearstr', {'bs_line': {'shear_strength': 1.1}})
    doc.set_provenance('shearstr', _shear_provenance(doc))
    assert store.is_done(doc, 'shearstr', _shear_provenance(doc), system_dir)
    assert not store.is_done(doc, 'shearstr', _shear_provenance(doc, nstepmax=1000), system_dir)
    assert not store.is_done(doc, 'shearstr', _shear_provenance(doc, seed=1), system_dir)
    # A PES computed again with other values invalidates the analysis downstream.
    doc.set(['pes_scf', 'energy', 'grid', 'hs_unique_jm2'], [['aa', 0.0, 0.0, -1.3]])
    assert doc.provenance('shearstr') is not None
    assert not store.is_done(doc, 'shearstr', _shear_provenance(doc), system_dir)


def _plotpes_provenance(doc, density=20):
    return Provenance.build('plotpes', parameters={'density': density, 'replicate_of': [5, 5]},
                            inputs={'grid': doc.get(['pes_scf', 'energy', 'grid'])})


def test_stale_prerequisites_stop_the_analysis(system_dir):
    doc = store.SystemDocument(interface_document())
    doc.set(['pes_scf', 'energy', 'grid', 'hs_unique_jm2'], [['aa', 0.0, 0.0, -1.2]])
    doc.set(['pes', 'energy', 'pes_jm2'], [[0.0, 0.0, -1.2]])
    doc.set_provenance(['pes', 'energy', 'pes_jm2'], _plotpes_provenance(doc))
    prerequisites = {('pes', 'energy', 'pes_jm2'): _plotpes_provenance(doc)}
    store.check_prerequisites(doc, prerequisites, system_dir)
    # The interpolation is asked with another density: the PES read by the shear
    # strength is not current, though its content did not change.
    prerequisites = {('pes', 'energy', 'pes_jm2'): _plotpes_provenance(doc, density=30)}
    with pytest.raises(store.PrerequisiteError, match='pes.energy.pes_jm2'):
        store.check_prerequisites(doc, prerequisites, system_dir)
    with pytest.raises(store.PrerequisiteError, match='ppes'):
        store.check_prerequisites(doc, {'ppes': Provenance.build('ppes')}, system_dir)
    assert doc.get(['pes', 'energy', 'pes_jm2']) == [[0.0, 0.0, -1.2]]


def test_inputs_may_hold_non_finite_values(system_dir):
    fit = {'uber': {'params': [1.2, 0.8], 'sigma': [math.inf, math.inf]}}
    doc = store.SystemDocument({'mid': 'm', 'ppes': fit, 'charge': {'a': 1}})
    with store.SystemWriter(system_dir) as writer:
        writer.document.create(doc.document)
        provenance = Provenance.build('charge', inputs={'ppes': writer.document.get('ppes')})
        writer.document.set_provenance('charge', provenance)
        writer.save()
    saved = store.read_system(system_dir)
    assert saved.get(['ppes', 'uber', 'sigma']) == [math.inf, math.inf]
    reread = Provenance.build('charge', inputs={'ppes': saved.get('ppes')})
    assert store.is_done(saved, 'charge', reread, system_dir)


def test_files_count_by_content(tmp_path):
    first = tmp_path / 'density.cube'
    first.write_text('1 2 3')
    moved = tmp_path / 'moved' / 'density.cube'
    moved.parent.mkdir()
    moved.write_text('1 2 3')
    same = Provenance.build('charge', files={'density': first})
    assert same == Provenance.build('charge', files={'density': moved})
    first.write_text('1 2 4')
    assert Provenance.build('charge', files={'density': first}).fingerprint != same.fingerprint


def test_writing_an_entry_drops_the_provenance_it_overlaps():
    doc = store.SystemDocument(interface_document())
    doc.set(['pes_scf', 'data', 'aa', 'output', 'energy'], -10.0)
    doc.set(['pes_scf', 'energy', 'min_site'], 'aa')
    mark = Provenance.build('pes_scf')
    for path in (['pes_scf', 'data', 'aa', 'output'], ['pes_scf', 'energy'], ['structure']):
        doc.set_provenance(path, mark)
    doc.set('pes_scf', {'energy': {'grid': {'hs_unique_ev': []}}})
    assert doc.provenance(['pes_scf', 'energy']) is None
    assert doc.provenance(['pes_scf', 'data', 'aa', 'output']) == mark
    doc.set('pes_scf', {})
    assert doc.provenance(['pes_scf', 'data', 'aa', 'output']) == mark
    doc.replace('pes_scf', {'data': {}})
    assert doc.provenance(['pes_scf', 'data', 'aa', 'output']) is None
    assert doc.provenance('structure') == mark
    doc.set(['structure', 'opt', 'pes_scf', 'min'], {'@class': 'Structure'})
    assert doc.provenance('structure') is None


def test_set_provenance_needs_the_entry():
    doc = store.SystemDocument(interface_document())
    with pytest.raises(StoreError, match='no such entry'):
        doc.set_provenance(['pes', 'energy'], Provenance.build('pes'))
    with pytest.raises(StoreError):
        Provenance.build('pes', parameters={'tol': math.nan})


def test_writer_saves_the_document_and_its_provenance(system_dir):
    assert not store.read_system(system_dir).exists
    mark = Provenance.build('match', parameters={'max_area': 150})
    with store.SystemWriter(system_dir) as writer:
        assert writer.lock_path.exists()
        writer.document.create(interface_document())
        writer.document.set('highsym', {'inter_unique': {'aa': [[0.0, 0.0]]}})
        writer.document.set_provenance('highsym', mark)
        writer.save()
    assert not (system_dir / 'system.json.lock').exists()
    saved = store.read_system(system_dir)
    assert saved.document == {**interface_document(),
                              'highsym': {'inter_unique': {'aa': [[0.0, 0.0]]}}}
    assert saved.provenance('highsym') == mark
    data = read_json(system_dir / 'system.json')
    assert list(data) == ['schema_version', 'document', 'provenance']
    assert data['provenance'][0]['path'] == ['highsym']
    assert data['provenance'][0]['fingerprint'] == mark.fingerprint


def test_tampered_provenance_is_an_error(system_dir):
    with store.SystemWriter(system_dir) as writer:
        writer.document.create({'highsym': {'a': 1}})
        writer.document.set_provenance('highsym', Provenance.build('match'))
        writer.save()
    path = system_dir / 'system.json'
    data = read_json(path)
    data['provenance'][0]['parameters'] = {'max_area': 100}
    write_json(path, data)
    with pytest.raises(StoreError, match='does not match its content'):
        store.read_system(system_dir)


def test_writer_saves_only_on_request(system_dir):
    with store.SystemWriter(system_dir) as writer:
        writer.document.create({'a': 1})
    assert not store.read_system(system_dir).exists
    with pytest.raises(RuntimeError):
        with store.SystemWriter(system_dir) as writer:
            writer.document.create({'a': 1})
            raise RuntimeError('stop')
    assert not store.read_system(system_dir).exists
    assert not (system_dir / 'system.json.lock').exists()
    with pytest.raises(StoreError, match='writer context'):
        store.SystemWriter(system_dir).save()


def test_second_writer_is_refused(system_dir):
    with store.SystemWriter(system_dir):
        with pytest.raises(store.LockError, match='system.json.lock'):
            with store.SystemWriter(system_dir):
                pass
        assert (system_dir / 'system.json.lock').exists()
    with store.SystemWriter(system_dir):
        pass


def test_stale_lock_is_never_removed(system_dir):
    lock = system_dir / 'system.json.lock'
    lock.write_text('{"pid": 1, "host": "node1"}\n')
    with pytest.raises(store.LockError, match='node1'):
        with store.SystemWriter(system_dir):
            pass
    assert lock.exists()


# -- separate processes --------------------------------------------------------

def _hold_the_lock(system_dir, locked, release):
    with store.SystemWriter(system_dir):
        locked.set()
        release.wait(60)


def test_writer_of_another_process_is_exclusive(system_dir):
    locked, release = MP.Event(), MP.Event()
    holder = MP.Process(target=_hold_the_lock, args=(system_dir, locked, release))
    holder.start()
    try:
        assert locked.wait(60)
        with pytest.raises(store.LockError):
            with store.SystemWriter(system_dir):
                pass
    finally:
        release.set()
        holder.join(60)
    assert holder.exitcode == 0
    with store.SystemWriter(system_dir):
        pass


def _increment(system_dir, times):
    done = 0
    while done < times:
        try:
            with store.SystemWriter(system_dir) as writer:
                writer.document.set('counter', writer.document.get('counter') + 1)
                writer.save()
        except store.LockError:
            time.sleep(0.001)
        else:
            done += 1


def test_concurrent_writers_lose_no_update(system_dir):
    with store.SystemWriter(system_dir) as writer:
        writer.document.create({'counter': 0})
        writer.save()
    workers = [MP.Process(target=_increment, args=(system_dir, 25)) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(300)
    assert [worker.exitcode for worker in workers] == [0] * 4
    assert store.read_system(system_dir).get('counter') == 100
    assert not (system_dir / 'system.json.lock').exists()


def _rewrite_record(system_dir, label, times):
    for i in range(times):
        forces = [[0.1, 0.2, 0.3]] * (50 + (37 * i) % 400)
        store.write_record(system_dir, make_record(label=label, results={
            'output': {'energy': float(i), 'forces': forces}}))


def _read_records(system_dir, stop, results):
    reads = failures = 0
    while not stop.is_set():
        for path in Path(system_dir).glob('*/*/record.json'):
            try:
                store.Record.from_dict(read_json(path))
            except StoreError:
                failures += 1
            reads += 1
    results.put((reads, failures))


def test_records_written_in_parallel_are_never_read_half_written(system_dir):
    labels = ['aa', 'ab', 'ba', 'bb']
    for label in labels:
        store.write_record(system_dir, make_record(label=label))
    stop, results = MP.Event(), MP.Queue()
    readers = [MP.Process(target=_read_records, args=(system_dir, stop, results))
               for _ in range(2)]
    writers = [MP.Process(target=_rewrite_record, args=(system_dir, label, 200))
               for label in labels]
    for process in readers + writers:
        process.start()
    for process in writers:
        process.join(300)
    stop.set()
    counts = [results.get(timeout=60) for _ in readers]
    for process in readers:
        process.join(60)
    assert [p.exitcode for p in writers + readers] == [0] * 6
    assert all(reads > 0 and failures == 0 for reads, failures in counts)
    for label in labels:
        assert store.read_record(system_dir, f'pes_scf/{label}').results['output']['energy'] \
            == 199.0
    assert list(system_dir.glob('*/*/.*.tmp')) == []


# -- historic database calls ---------------------------------------------------

def legacy_store(system_dir, expected=None, document=None):
    for label in ('aa', 'ab'):
        store.write_record(system_dir, make_record(label=label))
    return legacy.LegacyStore(system_dir, document or store.SystemDocument(), INTERFACES,
                              expected)


def test_document_is_created_as_by_the_match_step(system_dir):
    db = legacy_store(system_dir)
    assert db.query_db(INTERFACES, fltr=FLTR) == (None, None)
    assert not db.is_done(INTERFACES, entry='highsym', fltr=FLTR)
    db.insert_db(interface_document(), INTERFACES)
    field, entry = db.query_db(INTERFACES, entry='comp_params', fltr=FLTR)
    assert field == interface_document() and entry == {'use_vdw': False}
    field['mid'] = 'changed'
    assert db.document.get('mid') == 'mp-30_mp-23'
    # The same document again does nothing; with duplicates it is an error.
    db.insert_db(interface_document(), INTERFACES)
    with pytest.raises(StoreError, match='already has a document'):
        db.insert_db(interface_document(), INTERFACES, duplicates=True)
    # Once the structure entry is filled, the template is a different document.
    db.update_db({'@class': 'Slab'}, INTERFACES, entry=['structure', 'init', 'top_slab'],
                 fltr=FLTR)
    with pytest.raises(StoreError, match='already has a document'):
        db.insert_db(interface_document(), INTERFACES)
    with pytest.raises(StoreError):
        db.insert_db([interface_document()], INTERFACES)


def test_filters_select_by_equality_of_top_level_fields(system_dir):
    db = legacy_store(system_dir)
    db.insert_db(interface_document(), INTERFACES)

    def query(fltr):
        return db.query_db(INTERFACES, fltr=fltr)[0]

    for fltr in ({'mid': 'mp-30_mp-23', 'miller': [[1, 1, 1], [1, 1, 1]]},
                 {'mid': 'mp-30_mp-23', 'miller': ((1, 1, 1), (1, 1, 1))},
                 {'mid': 'mp-30_mp-23', 'miller': [[1.0, 1, 1], [1, 1, 1]]},
                 {'comp_params': {'use_vdw': False}}, {}):
        assert query(fltr) == interface_document()
    for fltr in ({'mid': 'mp-30'}, {'miller': [[1, 1, 1], [0, 0, 1]]},
                 {'formula': 'Cu_Ni', 'extra': 1}, {'miller': [[1, 1, 1], [1, 1, True]]}):
        with pytest.raises(StoreError, match='does not match'):
            query(fltr)
    for fltr in ({'mid': 'mp-30_mp-23', 'adsorbate': None}, {'name': None}):
        with pytest.raises(StoreError, match='is None'):
            query(fltr)
    with pytest.raises(StoreError, match='array'):
        query({'miller': 1})
    for fltr in ({'mid.x': 1}, {'mid': np.int64(1)}, None):
        with pytest.raises(StoreError):
            query(fltr)
    with pytest.raises(StoreError, match='not in this store'):
        db.query_db('PBE.slab_elements', fltr=FLTR)


def test_historic_arguments_without_effect_are_refused(system_dir):
    db = legacy_store(system_dir)
    db.insert_db(interface_document(), INTERFACES)
    # Positional calls as in the historic code would shift their arguments.
    calls = [lambda: db.query_db(INTERFACES, None, 'tribchem', fltr=FLTR),
             lambda: db.update_db(1, INTERFACES, 'x', None, 'tribchem', FLTR),
             lambda: db.insert_db(interface_document(), INTERFACES, None, 'tribchem'),
             lambda: db.is_done(INTERFACES, None, 'tribchem', entry='x', fltr=FLTR),
             lambda: db.query_db(INTERFACES, db_file=None, fltr=FLTR),
             lambda: db.query_db(INTERFACES, database='tribchem', fltr=FLTR),
             lambda: db.update_db(1, INTERFACES, entry='x', fltr=FLTR, upsert=False),
             lambda: db.insert_db(interface_document(), INTERFACES, message='inserted'),
             lambda: db.query_db(INTERFACES, mid='mp-30_mp-23')]
    for call in calls:
        with pytest.raises(TypeError):
            call()
    assert db.document.document == interface_document()


def test_query_entries(system_dir):
    db = legacy_store(system_dir)
    db.insert_db(interface_document(), INTERFACES)

    def entry(value):
        return db.query_db(INTERFACES, entry=value, fltr=FLTR)[1]

    assert entry(None) is None
    assert entry('formula') == 'Cu_Ni'
    assert entry(['comp_params', 'use_vdw']) is False
    assert entry([['mid'], ['comp_params', 'use_vdw'], ['missing']]) == ['mp-30_mp-23', False,
                                                                        None]
    assert entry(['missing', 'key']) is None
    for bad in (1, [], ['a.b'], [['a'], 'b'], ['$set']):
        with pytest.raises(StoreError):
            entry(bad)


ENTRY_FROM = [['nsites'], ['output', 'energy'], ['dir_name'], ['task_label'],
              ['output', 'structure'], ['input', 'structure']]


def entry_to(shift):
    out = ['pes_scf', 'data', shift, 'output']
    return [out + ['nsites'], out + ['energy'], out + ['dir_name'], out + ['task_label'],
            out + ['structure'], ['pes_scf', 'data', shift, 'input', 'structure']]


def test_results_are_moved_from_the_tasks_view(system_dir):
    db = legacy_store(system_dir)
    db.insert_db(interface_document(), INTERFACES)
    for shift in ('aa', 'ab'):
        _, values = db.query_db(legacy.TASKS, entry=ENTRY_FROM,
                                fltr={'task_label': f'pes_scf/{shift}'})
        db.update_db(values, INTERFACES, entry=entry_to(shift), fltr=FLTR)
    out = db.document.get(['pes_scf', 'data', 'ab', 'output'])
    assert out['energy'] == -10.5 and out['nsites'] == 4 and out['task_label'] == 'pes_scf/ab'
    assert out['structure'] == {'@class': 'Structure', 'id': 'ab'}
    # The directory is relative to the system: it stays true when the run is
    # copied elsewhere, and the historic 'host:path' split fails instead of
    # giving a wrong path.
    assert out['dir_name'] == 'pes_scf/ab' and (system_dir / out['dir_name']).is_dir()
    with pytest.raises(IndexError):
        out['dir_name'].split(':')[1]
    assert db.document.get(['pes_scf', 'data', 'aa', 'input', 'structure'])['id'] == 'in'
    assert db.query_db(legacy.TASKS, fltr={'task_label': 'pes_scf/zz'}) == (None, None)


def test_tasks_view_is_read_only_and_complete(system_dir):
    db = legacy_store(system_dir)
    db.insert_db(interface_document(), INTERFACES)
    task = {'task_label': 'pes_scf/aa'}
    with pytest.raises(StoreError, match='task_label only'):
        db.query_db(legacy.TASKS, fltr={**task, 'state': 'successful'})
    with pytest.raises(StoreError, match='read-only'):
        db.update_db(1, legacy.TASKS, entry='x', fltr=task)
    with pytest.raises(StoreError, match='read-only'):
        db.insert_db({'a': 1}, legacy.TASKS)
    with pytest.raises(StoreError):
        db.is_done(legacy.TASKS, entry='x', fltr=task)
    store.write_record(system_dir, make_record(label='aa', status=store.Status('running')))
    with pytest.raises(StoreError, match='not complete'):
        db.query_db(legacy.TASKS, fltr=task)
    store.write_record(system_dir, make_record(label='aa', results={'task_label': 'x'}))
    with pytest.raises(StoreError, match='cannot hold'):
        db.query_db(legacy.TASKS, fltr=task)


def test_updates_merge_leaves_as_dotted_set_keys(system_dir):
    db = legacy_store(system_dir)
    db.insert_db(interface_document(), INTERFACES)
    db.update_db(-10.0, INTERFACES, entry=['pes_scf', 'data', 'aa', 'output', 'energy'],
                 fltr=FLTR)
    summary = {'energy': {'grid': {'hs_unique_ev': [['aa', 0.0, 0.0, -10.0]]},
                          'max_site': 'aa', 'min_site': 'aa'}}
    db.update_db(summary, collection=INTERFACES, entry='pes_scf', fltr=FLTR)
    db.update_db({'@class': 'Structure'}, collection=INTERFACES,
                 entry=['structure', 'opt', 'pes_scf', 'min'], fltr=FLTR)
    doc = db.document
    assert doc.get(['pes_scf', 'data', 'aa', 'output', 'energy']) == -10.0
    assert doc.get(['pes_scf', 'energy', 'min_site']) == 'aa'
    assert doc.get('structure') == {'opt': {'pes_scf': {'min': {'@class': 'Structure'}}}}
    before = doc.document
    db.update_db({}, INTERFACES, entry='charge', fltr=FLTR)
    db.update_db({'regular': {'data': {}}}, INTERFACES, entry='charge', fltr=FLTR)
    assert doc.document == before
    db.update_db({'inter_params': {'max_area': 150}}, INTERFACES, fltr=FLTR)
    assert doc.get(['inter_params', 'max_area']) == 150


def test_update_without_to_mongodb_replaces_the_top_level_key(system_dir):
    db = legacy_store(system_dir)
    db.insert_db(interface_document(), INTERFACES)
    db.update_db(-10.0, INTERFACES, entry=['pes_scf', 'data', 'aa', 'output', 'energy'],
                 fltr=FLTR)
    db.update_db(1.0, INTERFACES, entry=['pes_scf', 'energy', 'min'], fltr=FLTR,
                 to_mongodb=False)
    assert db.document.get('pes_scf') == {'energy': {'min': 1.0}}
    db.update_db({'comp_params': {'x': 1}}, INTERFACES, fltr=FLTR, to_mongodb=False)
    assert db.document.get('comp_params') == {'x': 1}


def test_update_errors(system_dir):
    db = legacy_store(system_dir)
    with pytest.raises(StoreError, match='no document'):
        db.update_db(1, INTERFACES, entry='x', fltr=FLTR)
    db.insert_db(interface_document(), INTERFACES)
    bad_calls = [
        dict(data=1, entry='x'),                                   # no filter
        dict(data=1, entry='x', fltr=FLTR, update_op='$inc'),
        dict(data=1, entry='x', fltr={'mid': 'mp-30'}),
        dict(data=[1], entry=[['a'], ['b']], fltr=FLTR),
        dict(data=(1, 2), entry=[['a'], ['b']], fltr=FLTR),
        dict(data={'a.b': 1}, entry='x', fltr=FLTR),
        dict(data=1, entry=['x', '$y'], fltr=FLTR),
        dict(data=np.array([1.0]), entry='x', fltr=FLTR),
        dict(data=1, entry=None, fltr=FLTR),
        dict(data=1, entry=['mid', 'x'], fltr=FLTR),
        dict(data={'a': {'@float': 'NaN'}}, entry='x', fltr=FLTR),
    ]
    before = db.document.document
    for call in bad_calls:
        with pytest.raises(StoreError):
            db.update_db(collection=INTERFACES, **call)
    assert db.document.document == before


def test_is_done_adds_the_provenance_to_the_historic_condition(system_dir):
    path = ('pes_scf', 'energy')
    expected = _pes_provenance()
    db = legacy_store(system_dir, expected={path: expected})
    assert not db.is_done(INTERFACES, entry=list(path), fltr=FLTR)
    db.insert_db(interface_document(), INTERFACES)
    assert not db.is_done(INTERFACES, entry=None, fltr=FLTR)
    assert not db.is_done(INTERFACES, entry=list(path), fltr=FLTR)
    db.update_db({'min_site': 'aa'}, INTERFACES, entry=list(path), fltr=FLTR)
    assert not db.is_done(INTERFACES, entry=list(path), fltr=FLTR)
    db.document.set_provenance(path, _consumed(system_dir, expected))
    assert db.is_done(INTERFACES, entry=['pes_scf', 'energy'], fltr=FLTR)
    with pytest.raises(StoreError, match='no expected provenance'):
        db.is_done(INTERFACES, entry='formula', fltr=FLTR)
    with pytest.raises(StoreError, match='list of entries'):
        db.is_done(INTERFACES, entry=[['formula'], ['missing']], fltr=FLTR)
    with pytest.raises(StoreError, match='fltr is required'):
        db.is_done(INTERFACES, entry='formula')
    other = legacy.LegacyStore(system_dir, db.document, INTERFACES,
                               {path: _pes_provenance(parameters={'pes_type': 'forces'})})
    assert not other.is_done(INTERFACES, entry=list(path), fltr=FLTR)
    db.update_db({'max_site': 'ab'}, INTERFACES, entry=list(path), fltr=FLTR)
    assert not db.is_done(INTERFACES, entry=list(path), fltr=FLTR)


def test_legacy_store_arguments(system_dir):
    with pytest.raises(StoreError):
        legacy.LegacyStore(system_dir, {}, INTERFACES)
    with pytest.raises(StoreError):
        legacy.LegacyStore(system_dir, store.SystemDocument(), legacy.TASKS)
    with pytest.raises(StoreError):
        legacy.LegacyStore(system_dir, store.SystemDocument(), INTERFACES, {'pes': 'x'})
