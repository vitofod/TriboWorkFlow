"""Comparison of triboworkflow/vendored/ with the TribChem originals.

Every file of triboworkflow/vendored/, apart from OWN_FILES, mirrors
tribchem/<same relative path> of the TribChem sources and is one of:

- a byte copy (data files, empty __init__.py), identical to the original;
- a whole module: a provenance header, then the original file in which only
  the import statements of TribChem modules are rewritten ('tribchem.' ->
  'triboworkflow.vendored.', or the redirects listed in the header as
  'Import line N');
- a partial module: a provenance header listing the original line ranges of
  the statements kept, then those statements, unchanged except for the same
  import rewriting and separated by blank lines only.

Without the sources, byte copies and rebuilt whole modules are compared with
the SHA-256 manifest of the sources (envs/historic/tribchem-source.sha256),
and the statements of partial modules with the hashes in their header. Given
the sources, the original files are checked against the manifest and the
recorded hashes against the original lines.

Usage: python tests/vendoring.py [<TribChem folder with setup.py>]
"""

import ast
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / 'triboworkflow' / 'vendored'
MANIFEST = ROOT / 'envs' / 'historic' / 'tribchem-source.sha256'

# Files of the vendored tree that are not TribChem copies.
OWN_FILES = {
    '__init__.py',  # docstring of the vendored package
    'highput/database/__init__.py',  # TribChem has '__init__py' there (no dot)
}

START = '# --- TribChem code vendored by TriboWorkFlow '
END = '# --- end of the TriboWorkFlow header '
OLD, NEW = 'tribchem.', 'triboworkflow.vendored.'
REQUIRED = ('Source', 'Version', 'Source SHA-256', 'Copied', 'Licence', 'Kind')

FIELD = re.compile(r'^# (Source|Version|Source SHA-256|Copied|Licence|Kind): (.+)$')
REDIRECT = re.compile(r'^# Import line (\d+): (\S+) -> (\S+)$')
STATEMENT = re.compile(r'^# Statement lines (\d+)-(\d+): ([0-9a-f]{64})$')


@dataclass
class Header:
    fields: dict
    redirects: dict  # {line: (original module, vendored module)}
    statements: list  # [(first line, last line, SHA-256 of the original lines)]


def sha256(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def read_manifest():
    """{path from the TribChem folder: SHA-256} of the TribChem sources."""
    entries = {}
    for line in MANIFEST.read_text().splitlines():
        digest, path = line.split('  ', 1)
        entries[path.removeprefix('./')] = digest
    return entries


def vendored_files():
    """Paths, relative to VENDORED, of the files copied from TribChem."""
    files = []
    for path in sorted(VENDORED.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            rel = path.relative_to(VENDORED).as_posix()
            if rel not in OWN_FILES:
                files.append(rel)
    return files


def split_header(text):
    """(Header, body) of a vendored module; (None, text) if it has no header."""
    if not text.startswith(START):
        return None, text
    lines = text.split('\n')
    end = next((i for i, line in enumerate(lines) if line.startswith(END)), None)
    if end is None:
        raise ValueError('header without end line')
    fields, redirects, statements = {}, {}, []
    for line in lines[1:end]:
        if m := FIELD.match(line):
            fields[m[1]] = m[2]
        elif m := REDIRECT.match(line):
            redirects[int(m[1])] = (m[2], m[3])
        elif m := STATEMENT.match(line):
            statements.append((int(m[1]), int(m[2]), m[3]))
    return Header(fields, redirects, statements), '\n'.join(lines[end + 1:])


def imported_modules(node):
    """Absolute module names of an import statement, [] for other nodes."""
    if isinstance(node, ast.ImportFrom):
        return [node.module] if node.level == 0 else []
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    return []


def restore_imports(text, redirects=None, first_line=1):
    """Undo the import rewriting in text, whose first line is first_line."""
    redirects = redirects or {}
    lines = text.split('\n')
    for node in ast.walk(ast.parse(text)):
        if not any(name.startswith(NEW) for name in imported_modules(node)):
            continue
        old, new = redirects.get(node.lineno + first_line - 1, (OLD, NEW))
        lines[node.lineno - 1] = lines[node.lineno - 1].replace(new, old, 1)
    return '\n'.join(lines)


def check_partial(body, header, original=None):
    """Problems of the statements of a partial module."""
    problems = []
    lines = body.split('\n')
    nodes = ast.parse(body).body
    if len(nodes) != len(header.statements):
        return [f'{len(nodes)} statements, {len(header.statements)} listed in the header']
    original_lines = original.decode().split('\n') if original is not None else None
    covered = set()
    for node, (first, last, digest) in zip(nodes, header.statements):
        a = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
        b = node.end_lineno
        covered.update(range(a, b + 1))
        text = restore_imports('\n'.join(lines[a - 1:b]) + '\n', first_line=first)
        if b - a != last - first or sha256(text) != digest:
            problems.append(f'statement of original lines {first}-{last} differs')
        if original_lines is not None and \
                sha256('\n'.join(original_lines[first - 1:last]) + '\n') != digest:
            problems.append(f'original lines {first}-{last} differ from the header hash')
    extra = [i + 1 for i, line in enumerate(lines) if line.strip() and i + 1 not in covered]
    if extra:
        problems.append(f'lines outside the listed statements: {extra}')
    return problems


def kind(rel):
    """'byte copy', 'whole' or 'partial'."""
    header, _ = split_header((VENDORED / rel).read_bytes().decode(errors='replace'))
    return 'byte copy' if header is None else header.fields.get('Kind', '?')


def check(rel, manifest, source_root=None, vendored=VENDORED):
    """Problems of <vendored>/<rel>; an empty list if it matches the original."""
    original_rel = 'tribchem/' + rel
    expected = manifest.get(original_rel)
    if expected is None:
        return [f'{original_rel} is not among the TribChem sources']
    problems = []
    original = None
    if source_root is not None:
        original = (Path(source_root) / original_rel).read_bytes()
        if sha256(original) != expected:
            problems.append(f'{original_rel} in {source_root} differs from the manifest')
    data = (Path(vendored) / rel).read_bytes()
    if not (rel.endswith('.py') and data.startswith(START.encode())):
        if sha256(data) != expected:
            problems.append('differs from the original')
        return problems

    header, body = split_header(data.decode())
    missing = [key for key in REQUIRED if key not in header.fields]
    if missing:
        return problems + [f'header without {missing}']
    if header.fields['Source'] != original_rel:
        problems.append(f"header names {header.fields['Source']} as source")
    if header.fields['Source SHA-256'] != expected:
        problems.append('header hash differs from the manifest')
    if header.fields['Kind'] == 'whole':
        if header.statements:
            problems.append('whole module with statement lines in the header')
        if sha256(restore_imports(body, header.redirects)) != expected:
            problems.append('differs from the original outside the import statements')
    elif header.fields['Kind'] == 'partial':
        if header.redirects:
            problems.append('partial module with import redirects')
        problems += check_partial(body, header, original)
    else:
        problems.append(f"unknown kind {header.fields['Kind']!r}")
    return problems


def main(argv):
    source_root = Path(argv[1]) if len(argv) > 1 else None
    manifest = read_manifest()
    files = vendored_files()
    failed = 0
    for rel in files:
        problems = check(rel, manifest, source_root)
        failed += bool(problems)
        print(f"{'FAIL' if problems else 'ok  '}  {kind(rel):9s}  {rel}")
        for problem in problems:
            print(f'      {problem}')
    against = f'the sources in {source_root}' if source_root else 'the manifest only'
    print(f'{len(files)} files checked against {against}: {failed} failed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
