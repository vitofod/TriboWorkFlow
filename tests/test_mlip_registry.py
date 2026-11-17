# Author: Vito Foderà
# Created: 2026-11-14

"""Tests of the registry of MLIP model types (``triboworkflow.engines.mlip.registry``).

No model is loaded and no MLIP library is needed: the registry only describes the call of the
ASE calculator, and it does not look at the model file.
"""

import builtins
import json
import os
import pathlib
import subprocess
import sys

import pytest

from triboworkflow.engines.mlip.registry import (MODEL_TYPES, ModelError, ModelType,
                                                 ase_calculator_call, check_model,
                                                 get_model_type, register_model_type)

PATH = '/leonardo/models/mace-omat-0-medium.model'
TYPES = ('DP', 'MACE', 'MACE_MP')


def test_registered_types_and_backends():
    assert tuple(MODEL_TYPES) == TYPES
    assert get_model_type('DP').backends == ('ase', 'lammps')
    assert get_model_type('MACE').backends == ('ase', 'lammps')
    assert get_model_type('MACE_MP').backends == ('ase',)
    assert get_model_type('DP').package == 'deepmd-kit'
    assert get_model_type('MACE').package == get_model_type('MACE_MP').package == 'mace-torch'


# --- the type is compared whole ------------------------------------------------------

@pytest.mark.parametrize('name', ['ML/MACE', 'ML/DP', 'mace', 'Mace', 'MACE ', 'GRACE', 'Eqv2', ''])
def test_type_is_compared_whole(name):
    with pytest.raises(ModelError) as error:
        check_model(name, PATH, {})
    assert error.value.key == 'type'
    assert "allowed values: 'DP', 'MACE', 'MACE_MP'" in error.value.message


# --- the call of the calculator ------------------------------------------------------

def test_call_dp():
    assert ase_calculator_call('DP', PATH, {}) == {
        'factory': 'deepmd.calculator:DP', 'args': [PATH], 'kwargs': {}}


def test_call_mace():
    assert ase_calculator_call('MACE', PATH, {'default_dtype': 'float64'}) == {
        'factory': 'mace.calculators:MACECalculator', 'args': [],
        'kwargs': {'model_path': PATH, 'device': 'cuda', 'default_dtype': 'float64'}}


def test_call_mace_mp():
    assert ase_calculator_call('MACE_MP', PATH, {'default_dtype': 'float32'}) == {
        'factory': 'mace.calculators:mace_mp', 'args': [],
        'kwargs': {'model': PATH, 'device': 'cuda', 'default_dtype': 'float32'}}


def test_kwargs_pass_as_written():
    kwargs = {'default_dtype': 'float64', 'enable_cueq': True, 'cutoff': 5, 'list': [1, 2.5],
              'head': 'Omat24'}
    call = ase_calculator_call('MACE', PATH, kwargs)
    assert call['kwargs'] == {'model_path': PATH, 'device': 'cuda', **kwargs}
    assert [type(v) for v in call['kwargs'].values()] == [str, str, str, bool, int, list, str]
    assert kwargs == {'default_dtype': 'float64', 'enable_cueq': True, 'cutoff': 5, 'list': [1, 2.5],
                      'head': 'Omat24'}
    # the dictionary of the call is a copy: changing it leaves the settings alone
    call['kwargs']['extra'] = 1
    assert 'extra' not in kwargs


def test_dp_takes_any_kwargs():
    assert ase_calculator_call('DP', PATH, {'head': 'Omat24', 'type_dict': {'Fe': 0}})['kwargs'] == {
        'head': 'Omat24', 'type_dict': {'Fe': 0}}


def test_call_is_json():
    for name in TYPES:
        call = ase_calculator_call(name, PATH, {'default_dtype': 'float64', 'head': 'Omat24'})
        assert json.loads(json.dumps(call)) == call


# --- head (DEV-51) ---------------------------------------------------------------------

@pytest.mark.parametrize('name', ['MACE', 'MACE_MP'])
def test_head_only_from_kwargs(name):
    assert 'head' not in ase_calculator_call(name, PATH, {'default_dtype': 'float64'})['kwargs']
    kwargs = ase_calculator_call(name, PATH, {'default_dtype': 'float64', 'head': 'Omat24'})['kwargs']
    assert kwargs['head'] == 'Omat24'


# --- default_dtype (DEV-57) --------------------------------------------------------------

@pytest.mark.parametrize('name', ['MACE', 'MACE_MP'])
def test_default_dtype_required_with_ase(name):
    for call in (lambda: ase_calculator_call(name, PATH, {}),
                 lambda: check_model(name, PATH, {'head': 'Omat24'}, backend='ase')):
        with pytest.raises(ModelError) as error:
            call()
        assert error.value.key == 'kwargs'
        assert 'default_dtype' in error.value.message
    assert check_model(name, PATH, {'default_dtype': 'float64'}, backend='ase').name == name


def test_default_dtype_is_not_always_required():
    assert check_model('MACE', PATH, {}).name == 'MACE'  # before a step has chosen the backend
    assert check_model('MACE', PATH, {}, backend='lammps').name == 'MACE'  # precision in the file name
    assert check_model('DP', PATH, {}, backend='ase').name == 'DP'
    assert check_model('DP', PATH, {}, backend='lammps').name == 'DP'


def test_default_dtype_is_never_filled_in():
    call = ase_calculator_call('MACE', PATH, {'default_dtype': 'float32'})
    assert call['kwargs']['default_dtype'] == 'float32'
    assert 'default_dtype' not in ase_calculator_call('DP', PATH, {})['kwargs']


# --- backends --------------------------------------------------------------------------

def test_mace_mp_is_not_declared_for_lammps():
    with pytest.raises(ModelError, match="MACE_MP is not available with the backend 'lammps'") as error:
        check_model('MACE_MP', PATH, {'default_dtype': 'float64'}, backend='lammps')
    assert error.value.key == 'type'


def test_unknown_backend():
    with pytest.raises(ModelError, match="not available with the backend 'cp2k'"):
        check_model('DP', PATH, {}, backend='cp2k')


# --- the path ---------------------------------------------------------------------------

@pytest.mark.parametrize('path', ['', 'model.pth', './model.pth', '../model.pth', 'models/m.pth',
                                  'C:\\models\\m.pth', '~/m.pth', None, 5])
def test_path_must_be_absolute(path):
    with pytest.raises(ModelError) as error:
        check_model('MACE', path, {})
    assert error.value.key == 'path'
    assert 'absolute path' in error.value.message
    with pytest.raises(ModelError):
        ase_calculator_call('DP', path, {})


def test_model_file_is_not_looked_at(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('the registry looked at the file of the model')

    with monkeypatch.context() as patch:
        for module, name in [(os.path, 'exists'), (os.path, 'isfile'), (os, 'stat'),
                             (os, 'listdir'), (pathlib.Path, 'exists'), (pathlib.Path, 'stat'),
                             (pathlib.Path, 'is_file'), (pathlib.Path, 'resolve'),
                             (builtins, 'open')]:
            patch.setattr(module, name, forbidden)
        for name in TYPES:
            check_model(name, '/does/not/exist.model', {'default_dtype': 'float64'}, backend='ase')
            ase_calculator_call(name, '/does/not/exist.model', {'default_dtype': 'float64'})


# --- kwargs set by the workflow -----------------------------------------------------------

@pytest.mark.parametrize('name, key, reason', [
    ('MACE', 'device', "set by the workflow to 'cuda'"),
    ('MACE_MP', 'device', "set by the workflow to 'cuda'"),
    ('MACE', 'model_path', 'the path of the model, which is set by "path"'),
    ('MACE_MP', 'model', 'the path of the model, which is set by "path"'),
])
def test_reserved_kwargs(name, key, reason):
    with pytest.raises(ModelError) as error:
        check_model(name, PATH, {'default_dtype': 'float64', key: 'x'})
    assert error.value.key == 'kwargs'
    assert error.value.message == f'{key!r} is {reason}: remove it from kwargs'


def test_keys_of_another_type_are_not_reserved():
    # MACE_MP takes no model_path, and DP takes no device, so the registry says nothing of them.
    assert check_model('MACE_MP', PATH, {'model_path': 'x'}).name == 'MACE_MP'
    assert check_model('DP', PATH, {'device': 'cuda'}).name == 'DP'


# --- extension --------------------------------------------------------------------------------

NEW = ModelType('GRACE', 'grace-tensorpotential', 'tensorpotential.calculator:TPCalculator',
                path_argument='model', fixed_kwargs={'device': 'cuda'})


def test_register_a_type(mlip_types_restored):
    register_model_type(NEW)
    assert tuple(MODEL_TYPES) == TYPES + ('GRACE',)
    assert get_model_type('GRACE') is NEW
    assert ase_calculator_call('GRACE', PATH, {'pad': 4}) == {
        'factory': 'tensorpotential.calculator:TPCalculator', 'args': [],
        'kwargs': {'model': PATH, 'device': 'cuda', 'pad': 4}}
    assert check_model('GRACE', PATH, {}, backend='lammps').name == 'GRACE'  # both backends by default


def test_register_twice(mlip_types_restored):
    register_model_type(NEW)
    with pytest.raises(ValueError, match="'GRACE' is already registered"):
        register_model_type(NEW)
    with pytest.raises(ValueError, match="'MACE' is already registered"):
        register_model_type(ModelType('MACE', 'other', 'other:calculator'))
    assert get_model_type('MACE').package == 'mace-torch'


def test_types_restored_between_tests():
    assert tuple(MODEL_TYPES) == TYPES


# --- no MLIP library ---------------------------------------------------------------------------

def test_import_needs_no_mlip_library():
    code = ('import sys\n'
            'import triboworkflow.engines.mlip.registry as registry\n'
            'registry.ase_calculator_call("MACE", "/m.model", {"default_dtype": "float64"})\n'
            'loaded = [name for name in ("torch", "mace", "deepmd") if name in sys.modules]\n'
            'assert not loaded, loaded\n')
    done = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
