# Author: Vito Foderà
# Created: 2026-11-03

"""Tests of ``triboworkflow.execution.parsing``: the results read from the files of a calculation.

The outputs are synthetic (``pwo_helpers.py``); pw.x never runs.
"""

import numpy as np
import pytest

import pwo_helpers as pwo
from triboworkflow.core import store
from triboworkflow.core.structures import extxyz_text, read_structure
from triboworkflow.engines.qe.results import Parsed
from triboworkflow.execution import layout, parsing
from triboworkflow.execution.parsing import parse_calculation, with_parsed

SYSTEM = 'Cu111_Al111'
FORCES = ((0.001, -0.002, 0.003), (-0.001, 0.002, -0.003))
MODEL = {'version': '7.3', 'species': [{'label': label, 'pseudopotential': upf, 'md5': md5}
                                       for label, _, upf, md5 in pwo.SPECIES]}


def calculation(tmp_path, output, calculation='relax', structure=True, s=None):
    """Directory of a calculation with its input, record, input structure and output."""
    s = pwo.structure() if s is None else s
    sdir = tmp_path / SYSTEM
    path = sdir / 'pes' / 'aa'
    path.mkdir(parents=True)
    (path / layout.INPUT_FILE).write_text(f"&CONTROL\n  calculation = '{calculation}'\n/\n")
    store.write_record(sdir, store.Record(
        system=SYSTEM, step='pes', label='aa', calc_type='slab_pos_relax',
        engine=store.Engine('qe', 'pw.x'), energy_model=MODEL, fingerprint='sha256:' + 'a' * 64,
        attempts=(store.Attempt(1, status=store.Status('ended', 'converged', 'converged')),),
        status=store.Status('ended', 'converged', 'converged')))
    if structure:
        (path / layout.STRUCTURE_IN_FILE).write_text(extxyz_text(s))
    (path / layout.OUTPUT_FILE).write_text(output)
    return sdir, path


def test_calculation_parsed_from_its_files_alone(tmp_path):
    s = pwo.structure()
    output = pwo.pw_output(s, 'relax', version='7.3.1', forces=FORCES)
    sdir, path = calculation(tmp_path, output, s=s)
    parsed = parse_calculation(path)
    assert parsed.parsing == 'ok'
    # The version observed has the components of the expected one.
    assert (parsed.values.version, parsed.observed['version']) == ('7.3.1', '7.3')
    final = read_structure(path / layout.STRUCTURE_OUT_FILE)
    assert np.allclose(final.cart_coords, pwo.relaxed(s), atol=1e-8)
    record = store.read_record(sdir, 'pes/aa')
    done = with_parsed(record, record.attempts[-1], parsed, parsed.values.version)
    store.write_record(sdir, done)
    again = store.read_record(sdir, 'pes/aa')
    assert again.status == again.attempts[-1].status == \
        store.Status('ended', 'converged', 'converged', 'ok')
    assert (again.engine.version, again.results, again.observed) == \
        ('7.3.1', parsed.results, parsed.observed)
    assert store.can_reuse(again, again.fingerprint)


def test_output_not_over_is_not_read_and_needs_no_input_structure(tmp_path):
    _, path = calculation(tmp_path, pwo.pw_output(pwo.structure(), 'relax', job_done=False),
                          structure=False)
    assert parse_calculation(path) == Parsed(None)
    assert not (path / layout.STRUCTURE_OUT_FILE).exists()


def test_input_structure_not_read_fails(tmp_path):
    _, path = calculation(tmp_path, pwo.pw_output(pwo.structure(), 'relax', forces=FORCES),
                          structure=False)
    parsed = parse_calculation(path)
    assert parsed.parsing == 'failed'
    assert parsed.reason.startswith('input structure not read: ')
    assert 'structure_in.extxyz' in parsed.reason


def test_error_of_the_parser_fails_without_raising(tmp_path, monkeypatch, caplog):
    _, path = calculation(tmp_path, pwo.pw_output(pwo.structure(), 'relax', forces=FORCES))

    def broken(*args):
        raise RuntimeError('boom')
    monkeypatch.setattr(parsing, 'parse_output', broken)
    parsed = parse_calculation(path)
    assert (parsed.parsing, parsed.reason) == ('failed', 'error of the parser: RuntimeError: boom')
    assert 'error of the parser' in caplog.text


def test_final_structure_not_written_fails(tmp_path, monkeypatch):
    _, path = calculation(tmp_path, pwo.pw_output(pwo.structure(), 'relax', forces=FORCES))

    def refuse(target, text):
        raise OSError('disk full')
    monkeypatch.setattr(parsing, 'write_text', refuse)
    parsed = parse_calculation(path)
    assert (parsed.parsing, parsed.results, parsed.structure) == ('failed', None, None)
    assert parsed.reason == 'structure_out.extxyz not written: disk full'
    assert parsed.observed is not None


def test_with_parsed_gives_the_reason_to_the_attempt(tmp_path):
    sdir, _ = calculation(tmp_path, '')
    record = store.read_record(sdir, 'pes/aa')
    done = with_parsed(record, record.attempts[-1], Parsed('failed', reason='no forces'), '7.2')
    assert done.attempts[-1].parse_error == 'no forces'
    assert done.status.parsing == 'failed' and done.results is None
    with pytest.raises(store.StoreError, match='observed is null'):
        with_parsed(record, record.attempts[-1], Parsed('ok', results={'nsites': 2}), '7.2')
