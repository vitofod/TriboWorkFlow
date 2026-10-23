"""Development environment: it matches envs/dev/requirements.lock, which keeps
the versions of the historic environment except for ASE, and adds dacite."""

import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

import triboworkflow

ROOT = Path(__file__).resolve().parents[1]


def test_python_version():
    assert sys.version_info[:2] == (3, 12)


def test_installed_versions_match_lock(read_lock, installed_versions):
    expected = read_lock('dev') | {'triboworkflow': triboworkflow.__version__}
    assert installed_versions == expected


def test_dev_lock_aligned_with_historic_lock(read_lock):
    dev, historic = read_lock('dev'), read_lock('historic')
    different = {name: (historic.get(name), version)
                 for name, version in dev.items() if historic.get(name) != version}
    assert different == {'ase': ('3.23.0', '3.26.0'), 'dacite': (None, '1.8.1')}


def test_pyproject_dependencies_satisfied_by_lock(read_lock):
    pins = read_lock('dev')
    with open(ROOT / 'pyproject.toml', 'rb') as f:
        dependencies = tomllib.load(f)['project']['dependencies']
    for requirement in map(Requirement, dependencies):
        assert requirement.specifier.contains(pins[canonicalize_name(requirement.name)]), requirement
