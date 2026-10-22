"""Historic TribChem environment, as built by envs/historic/create_env.sh:
locked versions, recorded sources, atomate change, and the physics modules
of TribChem importing and running.

Skipped when TribChem is not importable, that is outside the historic venv.
"""

import hashlib
import importlib
import inspect
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

tribchem = pytest.importorskip('tribchem')

import matplotlib  # noqa: E402

matplotlib.use('Agg')

ENVS = Path(__file__).resolve().parents[2] / 'envs'
SOURCE_ROOT = Path(tribchem.__file__).resolve().parents[1]
PHYSICS = SOURCE_ROOT / 'tribchem' / 'physics'

# Modules of tribchem.physics that do not import here, with the reason.
# Folders named tests or scripts hold scripts and are not imported.
IMPORT_FAILURES = {
    'tribchem.physics.tribology.simple_pes': TabError,  # mixed tabs and spaces
    'tribchem.physics.ml.bayes': ModuleNotFoundError,  # imports tribchem.physics.base.qeio, which does not exist
    'tribchem.physics.ml.deepmd.deepmd_io': ModuleNotFoundError,  # needs dpdata, not in requirements.txt
    'tribchem.physics.dft.lammps': FileNotFoundError,  # reads tests/IO_files/md_ph_iron.pwi on import
    'tribchem.physics.electronics.charge.vasp_pymatgen': FileNotFoundError,  # reads tests/Charges/CHGCAR on import
    'tribchem.physics.electronics.dos.plotdos': FileNotFoundError,  # reads DOS.pdos_tot on import
}


def physics_modules():
    names = []
    for path in sorted(PHYSICS.rglob('*.py')):
        parts = path.relative_to(SOURCE_ROOT).with_suffix('').parts
        if {'tests', 'scripts'} & set(parts):
            continue
        if parts[-1] == '__init__':
            parts = parts[:-1]
        names.append('.'.join(parts))
    return names


def test_python_version():
    assert sys.version_info[:3] == (3, 12, 4)


def test_installed_versions_match_lock(read_lock, installed_versions):
    expected = read_lock('historic') | {'tribchem': '1.0.0'}
    assert installed_versions == expected


def test_sources_match_manifest():
    changed = []
    for line in (ENVS / 'historic' / 'tribchem-source.sha256').read_text().splitlines():
        digest, path = line.split('  ', 1)
        if hashlib.sha256((SOURCE_ROOT / path).read_bytes()).hexdigest() != digest:
            changed.append(path)
    assert not changed


def test_atomate_vdw_kernel_change():
    from atomate.vasp.fireworks import core

    copy_kernel = ('if vasp_input_set.vdw is not None:\n'
                   '            t.append(CopyFiles(from_dir=vdw_kernel_dir))')
    for firework in (core.OptimizeFW, core.StaticFW):
        assert 'vdw_kernel_dir' in inspect.signature(firework.__init__).parameters
        assert copy_kernel in inspect.getsource(firework.__init__)


def test_scipy_simps_with_even_avg():
    import numpy as np
    from scipy import integrate

    x = np.linspace(0, 1, 6)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        value = integrate.simps(x**2, x, even='avg')
    assert value == pytest.approx(1 / 3, abs=2e-3)


def test_distutils_available():
    # tribchem/highput/database/filetools.py; distutils left the standard
    # library in Python 3.12 and comes from setuptools here.
    from distutils.dir_util import copy_tree, remove_tree  # noqa: F401


def test_pymatgen_tools():
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder  # noqa: F401
    from pymatgen.analysis.interfaces.zsl import ZSLGenerator  # noqa: F401


def test_import_failures_are_physics_modules():
    assert set(IMPORT_FAILURES) <= set(physics_modules())


@pytest.mark.parametrize('name', physics_modules())
def test_physics_module_import(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    if name in IMPORT_FAILURES:
        with pytest.raises(IMPORT_FAILURES[name]):
            importlib.import_module(name)
    else:
        importlib.import_module(name)


def run_script(folder, script):
    return subprocess.run([sys.executable, script], cwd=folder, capture_output=True,
                          text=True, timeout=600, env=os.environ | {'MPLBACKEND': 'Agg'})


@pytest.mark.parametrize('script', ['test_matching_interfaces.py.py', 'test_pesmodel.py'])
def test_original_tribology_script_runs(script, tmp_path):
    # test_pesmodel.py also reads ../../ml/tests/graphene/energy.txt
    physics = tmp_path / 'physics'
    shutil.copytree(PHYSICS / 'tribology' / 'tests', physics / 'tribology' / 'tests')
    shutil.copytree(PHYSICS / 'ml' / 'tests' / 'graphene', physics / 'ml' / 'tests' / 'graphene')
    run = run_script(physics / 'tribology' / 'tests', script)
    assert run.returncode == 0, run.stderr[-3000:]


def test_original_replicate_script_reproduces_stored_outputs(tmp_path):
    outputs = ['fe_new', 'o_new', 'cell_new', 'fe_orthorombic', 'o_orthorombic']
    folder = tmp_path / 'tests'
    shutil.copytree(PHYSICS / 'base' / 'tests', folder)
    for name in outputs:
        (folder / name).unlink()
    run = run_script(folder, 'test_replicate.py')
    assert run.returncode == 0, run.stderr[-3000:]
    for name in outputs:
        assert (folder / name).read_bytes() == (PHYSICS / 'base' / 'tests' / name).read_bytes(), name
