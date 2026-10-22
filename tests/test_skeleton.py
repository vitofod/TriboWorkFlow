"""Package skeleton: every subpackage of the reference layout is importable."""

import importlib
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SUBPACKAGES = [
    'vendored',
    'core',
    'core.settings',
    'core.structures',
    'core.store',
    'engines',
    'engines.qe',
    'engines.mlip',
    'engines.cp2k',
    'execution',
    'workflows',
    'workflows.interface',
    'workflows.slab',
    'workflows.bulk',
    'analysis',
    'cli',
]


@pytest.mark.parametrize('name', SUBPACKAGES)
def test_subpackage_importable(name):
    module = importlib.import_module(f'triboworkflow.{name}')
    assert module.__doc__


def test_version_defined():
    import triboworkflow

    assert isinstance(triboworkflow.__version__, str)
    assert triboworkflow.__version__


def test_optional_extras_declared():
    with open(ROOT / 'pyproject.toml', 'rb') as f:
        project = tomllib.load(f)['project']
    assert project['requires-python'] == '>=3.12'
    assert {'mlip', 'cp2k', 'dev'} <= set(project['optional-dependencies'])
