# Author: Vito Foderà
# Created: 2026-10-27

"""Tests of ``triboworkflow.engines.qe.status``.

The outputs are synthetic: each one is built from the lines pw.x prints, as
written in its format statements, with only the lines the states depend on
and a few around them. One case per branch of the reading, including those
never seen in a real output (step limit, ``max_seconds``, stop by request,
failed optimizer).
"""

import re

import pytest

from triboworkflow.core.store import Status
from triboworkflow.engines.qe import status as qe
from triboworkflow.engines.qe.status import PwOutputError, pw_status, read_pw_status, record_status

ERROR_RULE = ' ' + '%' * 78


def header(version='7.2', relax=False, summary=True):
    """Header of a run, with the end of its summary unless ``summary`` is False."""
    lines = ['', f'     Program PWSCF v.{version} starts on  1Jan2026 at 10: 0: 0 ', '',
             '     Reading input from input.pwi', '']
    if summary:
        lines += ['     bravais-lattice index     =            0',
                  '     number of atoms/cell      =            2',
                  '     Exchange-correlation= PBE']
        if relax:
            lines += ['     energy convergence thresh.=      1.0E-04',
                      '     force convergence thresh. =      1.0E-03',
                      '     nstep                     =           50']
        lines += ['', '     crystal axes: (cart. coord. in units of alat)', '']
    return lines


def scf_cycle(outcome='converged'):
    """One SCF cycle; ``outcome`` None leaves it open."""
    lines = ['     Self-consistent Calculation', '',
             '     iteration #  1     ecut=    30.00 Ry     beta= 0.30',
             '     total energy              =     -31.00000000 Ry']
    if outcome == 'converged':
        lines += ['', '     End of self-consistent calculation', '',
                  '!    total energy              =     -31.10000000 Ry', '',
                  '     convergence has been achieved in   7 iterations', '']
    elif outcome == 'not_converged':
        lines += ['', '     End of self-consistent calculation', '',
                  '     convergence NOT achieved after 100 iterations: stopping', '']
    return lines


def bfgs_step(step):
    return ['     number of scf cycles    = ' + f'{step + 1:3d}',
            '     number of bfgs steps    = ' + f'{step:3d}', '']


def final_coordinates(closed=True):
    lines = ['Begin final coordinates', '', 'ATOMIC_POSITIONS (angstrom)',
             'H             0.0000000000        0.0000000000        0.0000000000',
             'H             0.0000000000        0.0000000000        0.7400000000']
    return lines + (['End final coordinates', ''] if closed else [])


BFGS_CONVERGED = ['', '     bfgs converged in   3 scf cycles and   2 bfgs steps',
                  '     (criteria: energy <  1.0E-04 Ry, force <  1.0E-03 Ry/Bohr)', '',
                  '     End of BFGS Geometry Optimization', '',
                  '     Final energy             =     -31.1000000000 Ry', '']
BFGS_FAILED = ['', '     bfgs failed after   5 scf cycles and   4 bfgs steps, convergence '
               'not achieved', '     (criteria: energy <  1.0E-04 Ry, force <  1.0E-03 Ry/Bohr)',
               '', '     End of BFGS Geometry Optimization', '',
               '     Final energy             =     -31.1000000000 Ry', '']
MAX_STEPS = ['', '     The maximum number of steps has been reached.', '',
             '     End of BFGS Geometry Optimization', '']
MAX_SECONDS = ['', '     Maximum CPU time exceeded', '',
               '     max_seconds     =     100.00', '     elapsed seconds =     101.00']
USER_REQUEST = ['', '     Program stopped by user request']
ERROR = [ERROR_RULE, '     Error in routine  cdiaghg (309):',
         '     problems computing cholesky', ERROR_RULE, '', '     stopping ...']
JOB_DONE = ['', '     PWSCF        :      1.00s CPU      1.10s WALL', '',
            '   This run was terminated on:  10: 0: 1   1Jan2026', '',
            '=' + '-' * 78 + '=', '   JOB DONE.', '=' + '-' * 78 + '=']


def relax_steps(n):
    """``n`` converged SCF cycles of a relax, each followed by a BFGS step."""
    lines = []
    for step in range(n):
        lines += scf_cycle('converged') + bfgs_step(step)
    return lines


def text(*parts):
    return '\n'.join(line for part in parts for line in part) + '\n'


SCF = header()
RELAX = header(relax=True)

# (name, calculation, output, process, scf, geometry, final coordinates, stop reason)
CASES = [
    ('scf converged', 'scf', text(SCF, scf_cycle('converged'), JOB_DONE),
     'ended', 'converged', 'n/a', 'n/a', 'completed'),
    ('scf not converged', 'scf', text(SCF, scf_cycle('not_converged'), JOB_DONE),
     'ended', 'not_converged', 'n/a', 'n/a', 'scf_not_converged'),
    ('scf truncated in the cycle', 'scf', text(SCF, scf_cycle(None)),
     None, None, 'n/a', 'n/a', 'truncated'),
    ('scf truncated before the cycle', 'scf', text(SCF),
     None, None, 'n/a', 'n/a', 'truncated'),
    ('scf error', 'scf', text(SCF, scf_cycle(None), ERROR),
     'crashed', None, 'n/a', 'n/a', 'error'),
    ('scf max_seconds', 'scf', text(SCF, scf_cycle(None), MAX_SECONDS, JOB_DONE),
     'ended', None, 'n/a', 'n/a', 'max_seconds'),
    ('scf user request', 'scf', text(SCF, scf_cycle(None), USER_REQUEST, JOB_DONE),
     'ended', None, 'n/a', 'n/a', 'user_request'),
    ('scf ended for no known reason', 'scf', text(SCF, scf_cycle(None), JOB_DONE),
     'ended', None, 'n/a', 'n/a', None),
    ('relax converged', 'relax',
     text(RELAX, relax_steps(2), scf_cycle('converged'), BFGS_CONVERGED, final_coordinates(),
          JOB_DONE),
     'ended', 'converged', 'converged', 'complete', 'completed'),
    ('relax converged, truncated in the final coordinates', 'relax',
     text(RELAX, relax_steps(2), scf_cycle('converged'), BFGS_CONVERGED,
          final_coordinates(closed=False)),
     None, 'converged', 'converged', 'incomplete', 'truncated'),
    ('relax converged, truncated before the final coordinates', 'relax',
     text(RELAX, relax_steps(2), scf_cycle('converged'), BFGS_CONVERGED),
     None, 'converged', 'converged', 'absent', 'truncated'),
    ('relax at the step limit', 'relax',
     text(RELAX, relax_steps(2), scf_cycle('converged'), MAX_STEPS, JOB_DONE),
     'ended', 'converged', 'max_steps', 'absent', 'max_steps'),
    ('relax with a failed optimizer', 'relax',
     text(RELAX, relax_steps(4), scf_cycle('converged'), BFGS_FAILED, final_coordinates(),
          JOB_DONE),
     'ended', 'converged', 'not_converged', 'complete', 'optimizer_failed'),
    ('relax with an SCF not converged', 'relax',
     text(RELAX, relax_steps(2), scf_cycle('not_converged'), JOB_DONE),
     'ended', 'not_converged', 'not_converged', 'absent', 'scf_not_converged'),
    ('relax error', 'relax', text(RELAX, relax_steps(2), scf_cycle(None), ERROR),
     'crashed', None, 'not_converged', 'absent', 'error'),
    ('relax truncated in a cycle', 'relax', text(RELAX, relax_steps(2), scf_cycle(None)),
     None, None, None, 'absent', 'truncated'),
    ('relax truncated between steps', 'relax', text(RELAX, relax_steps(2)),
     None, 'converged', None, 'absent', 'truncated'),
    ('relax max_seconds', 'relax',
     text(RELAX, relax_steps(2), scf_cycle(None), MAX_SECONDS, JOB_DONE),
     'ended', None, 'not_converged', 'absent', 'max_seconds'),
    ('relax user request', 'relax', text(RELAX, relax_steps(2), USER_REQUEST, JOB_DONE),
     'ended', 'converged', 'not_converged', 'absent', 'user_request'),
    ('relax ended for no known reason', 'relax', text(RELAX, relax_steps(2), JOB_DONE),
     'ended', 'converged', 'not_converged', 'absent', None),
]


@pytest.mark.parametrize('name, calculation, output, process, scf, geometry, coordinates, reason',
                         CASES, ids=[case[0] for case in CASES])
def test_states_of_one_run(name, calculation, output, process, scf, geometry, coordinates,
                           reason):
    status = pw_status(output, calculation)
    assert (status.process, status.scf, status.geometry, status.final_coordinates,
            status.stop_reason) == (process, scf, geometry, coordinates, reason)
    assert (status.calculation, status.version, status.runs) == (calculation, '7.2', 1)
    assert status.clean_stop == (process == 'ended')
    assert status.stop_reason in qe.STOP_REASONS + (None,)
    assert status.final_coordinates in qe.FINAL_COORDINATES


def test_markers_need_not_be_indented():
    output = text(header(), [line.strip() for line in scf_cycle('converged') + JOB_DONE])
    status = pw_status(output, 'scf')
    assert (status.process, status.scf, status.stop_reason) == ('ended', 'converged', 'completed')


def test_end_of_bfgs_alone_is_not_convergence():
    output = text(RELAX, relax_steps(1), scf_cycle('converged'),
                  ['', '     End of BFGS Geometry Optimization', ''], JOB_DONE)
    assert pw_status(output, 'relax').geometry == 'not_converged'


def test_only_the_last_run_is_read():
    converged = text(RELAX, relax_steps(2), scf_cycle('converged'), BFGS_CONVERGED,
                     final_coordinates(), JOB_DONE)
    not_converged = text(RELAX, relax_steps(1), scf_cycle('not_converged'), JOB_DONE)
    truncated = text(RELAX, relax_steps(1), scf_cycle(None))

    status = pw_status(not_converged + converged, 'relax')
    assert (status.runs, status.process, status.scf, status.geometry,
            status.final_coordinates) == (2, 'ended', 'converged', 'converged', 'complete')

    status = pw_status(converged + truncated, 'relax')
    assert (status.runs, status.process, status.scf, status.geometry, status.final_coordinates,
            status.stop_reason) == (2, None, None, None, 'absent', 'truncated')

    status = pw_status(converged + not_converged + not_converged, 'relax')
    assert (status.runs, status.scf, status.geometry, status.stop_reason) == \
        (3, 'not_converged', 'not_converged', 'scf_not_converged')


def test_text_before_the_first_run_is_ignored():
    output = 'srun: warning\n' + text(SCF, scf_cycle('converged'), JOB_DONE)
    assert pw_status(output, 'scf').process == 'ended'


@pytest.mark.parametrize('output', ['', '\n\n', 'srun: error: node failure\n',
                                    text(scf_cycle('converged'), JOB_DONE)],
                         ids=['empty', 'blank', 'no pw.x', 'no header'])
def test_output_without_header_is_an_error(output):
    with pytest.raises(PwOutputError, match='not a pw.x output'):
        pw_status(output, 'scf')


@pytest.mark.parametrize('calculation, run', [('scf', RELAX), ('relax', SCF)])
def test_summary_must_agree_with_the_calculation(calculation, run):
    with pytest.raises(PwOutputError, match='nstep'):
        pw_status(text(run, scf_cycle('converged'), JOB_DONE), calculation)


@pytest.mark.parametrize('calculation', qe.CALCULATIONS)
def test_summary_is_not_checked_before_it_ends(calculation):
    status = pw_status(text(header(summary=False)), calculation)
    assert (status.process, status.stop_reason) == (None, 'truncated')


def test_unknown_calculation_is_an_error():
    with pytest.raises(ValueError, match='calculation') as err:
        pw_status(text(SCF, scf_cycle('converged'), JOB_DONE), 'vc-relax')
    assert not isinstance(err.value, PwOutputError)


@pytest.mark.parametrize('version, validated', [('7.1', True), ('7.2', True), ('7.3', False),
                                                ('7.3.1', False)])
def test_supported_versions(version, validated):
    status = pw_status(text(header(version), scf_cycle('converged'), JOB_DONE), 'scf')
    assert (status.version, status.version_validated) == (version, validated)


@pytest.mark.parametrize('version', ['6.8', '7.0', '7.4', '8.1', '7', 'x.y', ''])
def test_other_versions_are_an_error(version):
    with pytest.raises(PwOutputError, match='version'):
        pw_status(text(header(version), scf_cycle('converged'), JOB_DONE), 'scf')


def test_version_is_that_of_the_last_run():
    old = text(header('6.8'), scf_cycle('converged'), JOB_DONE)
    new = text(header('7.2'), scf_cycle('converged'), JOB_DONE)
    status = pw_status(old + new, 'scf')
    assert (status.version, status.runs) == ('7.2', 2)
    with pytest.raises(PwOutputError, match="'6.8'"):
        pw_status(new + old, 'scf')


def test_read_pw_status_from_a_file(tmp_path):
    path = tmp_path / 'output.pwo'
    path.write_bytes(text(SCF, scf_cycle('converged'), ['     \xff\xfe path'], JOB_DONE)
                     .encode('latin-1'))
    assert read_pw_status(path, 'scf').process == 'ended'
    path.write_text('no output\n', encoding='utf-8')
    with pytest.raises(PwOutputError, match=re.escape(str(path))):
        read_pw_status(path, 'scf')


def output_of(name):
    return next(case for case in CASES if case[0] == name)


TRUNCATED = [(None, None, 'unknown'), ('running', 'running', 'running'),
             ('killed', 'killed', 'crashed'), ('finished', 'crashed', 'crashed')]


@pytest.mark.parametrize('job, process, summary', TRUNCATED)
def test_record_status_of_a_truncated_relax(job, process, summary):
    pw = pw_status(output_of('relax truncated between steps')[2], 'relax')
    status = record_status(pw, job)
    assert status == Status(process=process, scf='converged', geometry=None)
    assert status.summary == summary


@pytest.mark.parametrize('job, process, summary', TRUNCATED)
def test_record_status_of_a_truncated_scf(job, process, summary):
    pw = pw_status(output_of('scf truncated in the cycle')[2], 'scf')
    status = record_status(pw, job)
    assert status == Status(process=process, scf=None, geometry='n/a')
    assert status.summary == summary


@pytest.mark.parametrize('job', [None] + list(qe.JOB_STATES))
def test_record_status_keeps_a_declared_geometry(job):
    pw = pw_status(output_of('relax converged, truncated in the final coordinates')[2], 'relax')
    assert record_status(pw, job).geometry == 'converged'


@pytest.mark.parametrize('name', ['relax converged', 'relax error', 'scf not converged'])
@pytest.mark.parametrize('job', [None] + list(qe.JOB_STATES))
def test_record_status_follows_a_closed_output(name, job):
    _, calculation, output, process, scf, geometry, _, _ = output_of(name)
    status = record_status(pw_status(output, calculation), job)
    assert status == Status(process=process, scf=scf, geometry=geometry, parsing=None)
    assert not status.complete


SUMMARY_OF_CASE = {'completed': 'completed', 'scf_not_converged': 'not_converged',
                   'max_steps': 'not_converged', 'optimizer_failed': 'not_converged',
                   'max_seconds': 'not_converged', 'user_request': 'not_converged',
                   None: 'not_converged', 'error': 'crashed', 'truncated': 'unknown'}


@pytest.mark.parametrize('name, calculation, output, reason',
                         [(case[0], case[1], case[2], case[7]) for case in CASES],
                         ids=[case[0] for case in CASES])
def test_summary_of_each_output(name, calculation, output, reason):
    assert record_status(pw_status(output, calculation)).summary == SUMMARY_OF_CASE[reason]


def test_record_status_rejects_an_unknown_job():
    pw = pw_status(output_of('scf truncated in the cycle')[2], 'scf')
    with pytest.raises(ValueError, match='job'):
        record_status(pw, 'ended')
