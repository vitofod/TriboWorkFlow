"""Vendored TribChem code: closure of the imports and comparison with the
originals (through the SHA-256 manifest of the sources, see vendoring.py)."""

import ast
import json
import os
import subprocess
import sys
import tomllib
from fnmatch import fnmatch

import pytest

import vendoring

# Top-level packages the vendored code must not import, directly or not.
FORBIDDEN = {'fireworks', 'atomate', 'pymongo', 'tribchem'}

PY_FILES = sorted(p.relative_to(vendoring.VENDORED).as_posix()
                  for p in vendoring.VENDORED.rglob('*.py'))


def module_name(rel):
    parts = ('triboworkflow', 'vendored', *rel.removesuffix('.py').split('/'))
    return '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)


def imports(rel):
    """(line, module) of every absolute import in triboworkflow/vendored/<rel>."""
    tree = ast.parse((vendoring.VENDORED / rel).read_text())
    for node in ast.walk(tree):
        for name in vendoring.imported_modules(node):
            yield node.lineno, name


@pytest.mark.parametrize('rel', PY_FILES)
def test_no_forbidden_imports(rel):
    found = [f'line {n}: {m}' for n, m in imports(rel) if m.split('.')[0] in FORBIDDEN]
    assert not found


def test_vendored_imports_point_to_vendored_modules():
    modules = {module_name(rel) for rel in PY_FILES}
    missing = [f'{rel}:{n}: {m}' for rel in PY_FILES for n, m in imports(rel)
               if m.startswith('triboworkflow.') and m not in modules]
    assert not missing


def test_import_closure():
    """Importing every vendored module loads none of the forbidden packages."""
    code = ('import importlib, json, sys\n'
            f'for name in {[module_name(rel) for rel in PY_FILES]!r}:\n'
            '    importlib.import_module(name)\n'
            'print(json.dumps(sorted({m.split(".")[0] for m in sys.modules})))\n')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            cwd=vendoring.ROOT, env=dict(os.environ, MPLBACKEND='Agg'))
    assert result.returncode == 0, result.stderr
    loaded = set(json.loads(result.stdout.splitlines()[-1]))
    assert not loaded & FORBIDDEN


@pytest.mark.parametrize('rel', vendoring.vendored_files())
def test_matches_tribchem(rel):
    assert vendoring.check(rel, vendoring.read_manifest()) == []


@pytest.mark.parametrize('rel, old, new', [
    # a line of code of a whole module
    ('physics/tribology/mep.py', "'/defaults_tribo.json'", "'/defaults.json'"),
    # an import pointed to another module without a redirect in the header
    ('highput/utils/manipulate_struct.py', 'vendored.physics.base.shaper import',
     'vendored.physics.base.solidstate import'),
    # the source hash in the header
    ('physics/base/units.py', '# Source SHA-256: ', '# Source SHA-256: 0'),
    # a statement of a partial module
    ('highput/utils/tasktools.py', "''.join(str(s) for s in miller)",
     "''.join(map(str, miller))"),
    # a line added between the statements of a partial module
    ('highput/utils/io_files.py', '\n\ndef remove_matching_files',
     '\n# note\ndef remove_matching_files'),
    # a byte copy
    ('physics/tribology/defaults_tribo.json', '\n', ' \n'),
])
def test_check_detects_changes(tmp_path, rel, old, new):
    data = (vendoring.VENDORED / rel).read_bytes()
    assert old.encode() in data
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_bytes(data.replace(old.encode(), new.encode(), 1))
    assert vendoring.check(rel, vendoring.read_manifest(), vendored=tmp_path)


def test_package_data_includes_vendored_data():
    with open(vendoring.ROOT / 'pyproject.toml', 'rb') as f:
        package_data = tomllib.load(f)['tool']['setuptools']['package-data']
    patterns = package_data['triboworkflow.vendored']
    data = [rel for rel in vendoring.vendored_files() if not rel.endswith('.py')]
    assert data
    assert [rel for rel in data if not any(fnmatch(rel, p) for p in patterns)] == []
