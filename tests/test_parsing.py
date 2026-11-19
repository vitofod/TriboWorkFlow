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
from triboworkflow.engines.qe.results import BOHR_TO_ANGSTROM, RY_TO_EV, Parsed
from triboworkflow.execution import layout, parsing
from triboworkflow.execution.parsing import parse_calculation, with_parsed

SYSTEM = 'Cu111_Al111'
FORCES = ((0.001, -0.002, 0.003), (-0.001, 0.002, -0.003))
MODEL = {'version': '7.3', 'species': [{'label': label, 'pseudopotential': upf}
                                       for label, _, upf, _ in pwo.SPECIES]}


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


# --- trajectory of a relaxation ---------------------------------------------------------------

def read_frames(path):
    import ase.io
    return ase.io.read(path / layout.TRAJECTORY_FILE, index=':', format='extxyz')


def test_trajectory_of_a_relaxation_has_every_ionic_step(tmp_path):
    s = pwo.structure(if_pos=[[0, 0, 1], [1, 1, 1]])
    output = pwo.pw_output(s, 'relax', energy=-31.1, forces=FORCES, steps=3,
                           final_if_pos=[[0, 0, 1], [1, 1, 1]])
    _, path = calculation(tmp_path, output, s=s)
    parsed = parse_calculation(path)
    assert parsed.parsing == 'ok'
    frames = read_frames(path)
    # One frame per SCF cycle: the header, then the positions after each ionic step.
    assert len(frames) == 3
    energies = [frame.get_potential_energy() for frame in frames]
    # With the constants of QE, not those of ASE's reader (3.7e-5 eV apart here): the last
    # frame has the energy and the forces of the record.
    assert np.allclose(energies, np.array([-30.9, -31.0, -31.1]) * RY_TO_EV, rtol=0, atol=1e-7)
    assert energies[-1] == pytest.approx(parsed.results['output']['energy'], abs=1e-7)
    # The forces as printed: the constraints of the frames are not applied.
    forces = [frame.get_forces(apply_constraint=False) for frame in frames]
    assert np.allclose(forces[-1], parsed.results['output']['forces'], atol=1e-8)
    assert np.allclose(forces[0], np.asarray(FORCES) * RY_TO_EV / BOHR_TO_ANGSTROM, atol=1e-8)
    # Positions of the output (alat units in the header, angstrom after); cell, labels and
    # constraints of the input.
    expected = pwo.structure_in(s)
    assert np.allclose(frames[0].positions, expected.cart_coords, atol=1e-5)
    assert np.allclose(frames[1].positions, s.cart_coords, atol=1e-8)
    for k in range(3):
        frame = read_structure(path / layout.TRAJECTORY_FILE, format='extxyz', frame=k)
        assert np.allclose(frame.lattice.matrix, expected.lattice.matrix, rtol=0, atol=1e-12)
        assert frame.labels == ['Fe1', 'H']
        assert frame.site_properties['if_pos'] == [[0, 0, 1], [1, 1, 1]]


def test_trajectory_of_a_relaxation_cut_short_or_not_converged(tmp_path):
    s = pwo.structure()
    # Stopped before its end (e.g. the walltime): nothing is parsed, the frames are kept.
    output = pwo.pw_output(s, 'relax', forces=FORCES, steps=3, job_done=False)
    cut = output[:output.index('     bfgs converged')]
    _, path = calculation(tmp_path, cut)
    assert parse_calculation(path) == Parsed(None)
    assert len(read_frames(path)) == 3
    assert not (path / layout.STRUCTURE_OUT_FILE).exists()


def test_no_trajectory_for_a_scf_or_without_ionic_step(tmp_path, caplog):
    _, path = calculation(tmp_path, pwo.pw_output(pwo.structure(), 'scf', forces=FORCES),
                          calculation='scf')
    assert parse_calculation(path).parsing == 'ok'
    assert not (path / layout.TRAJECTORY_FILE).exists()
    # A relaxation stopped in its first SCF cycle: no frame, no file, no warning.
    text = pwo.pw_output(pwo.structure(), 'relax', forces=FORCES, job_done=False)
    _, path = calculation(tmp_path / 'first', text[:text.index('     Self-consistent')])
    caplog.set_level('INFO')
    assert parse_calculation(path) == Parsed(None)
    assert not (path / layout.TRAJECTORY_FILE).exists()
    assert 'no ionic step with results' in caplog.text and 'WARNING' not in caplog.text


def test_trajectory_not_written_is_only_a_warning(tmp_path, monkeypatch, caplog):
    _, path = calculation(tmp_path, pwo.pw_output(pwo.structure(), 'relax', forces=FORCES))

    def broken(*args):
        raise RuntimeError('unreadable')
    monkeypatch.setattr(parsing, 'trajectory_text', broken)
    parsed = parse_calculation(path)
    assert parsed.parsing == 'ok' and (path / layout.STRUCTURE_OUT_FILE).is_file()
    assert not (path / layout.TRAJECTORY_FILE).exists()
    assert 'trajectory.extxyz not written: RuntimeError: unreadable' in caplog.text


def test_with_parsed_gives_the_reason_to_the_attempt(tmp_path):
    sdir, _ = calculation(tmp_path, '')
    record = store.read_record(sdir, 'pes/aa')
    done = with_parsed(record, record.attempts[-1], Parsed('failed', reason='no forces'), '7.2')
    assert done.attempts[-1].parse_error == 'no forces'
    assert done.status.parsing == 'failed' and done.results is None
    with pytest.raises(store.StoreError, match='observed is null'):
        with_parsed(record, record.attempts[-1], Parsed('ok', results={'nsites': 2}), '7.2')
