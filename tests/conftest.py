"""Shared fixtures: lock files of envs/ and installed distributions."""

import re
from importlib import metadata
from pathlib import Path

import pytest
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
ENVS = ROOT / 'envs'


@pytest.fixture
def read_lock():
    """Return a function giving the {name: version} pins of envs/<env>/requirements.lock."""

    def read(env):
        text = (ENVS / env / 'requirements.lock').read_text()
        pins = re.findall(r'^([A-Za-z0-9_.-]+)==(\S+)', text, re.M)
        return {canonicalize_name(name): version for name, version in pins}

    return read


@pytest.fixture
def installed_versions():
    """{name: version} of the distributions installed in the running interpreter.

    Build metadata left in the repository root (found when the root is on
    sys.path, as with 'python -m pytest') is not an installed distribution.
    """
    return {canonicalize_name(d.metadata['Name']): d.version
            for d in metadata.distributions()
            if Path(d.locate_file('')).resolve() != ROOT}
