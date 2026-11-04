# Author: Vito Foderà
# Created: 2026-10-27

"""Tests of ``triboworkflow.engines.qe.status`` on real pw.x outputs.

Real outputs are never stored in the repository. They are read from the
directory named by the environment variable ``TRIBOWORKFLOW_PWO_CORPUS``;
without it these tests are skipped. With it, every file listed here must be
there.

The files carry comment lines added by hand, beginning with ``###`` (possibly
indented); they are removed before reading, so that their words cannot be
taken for lines of pw.x.
"""

import os
import re
from pathlib import Path

import pytest

from triboworkflow.engines.qe.status import PwError, PwRestart, pw_status, record_status

CORPUS_VARIABLE = 'TRIBOWORKFLOW_PWO_CORPUS'
CORPUS = os.environ.get(CORPUS_VARIABLE)
COMMENT = re.compile(r'^\s*###')

pytestmark = pytest.mark.skipif(not CORPUS, reason=f'{CORPUS_VARIABLE} is not set')

# (file, calculation, version, runs, process, scf, geometry, final coordinates, stop reason,
#  summary of the record without news of the job)
EXPECTED = [
    ('file_output_relax_convergenza_raggiunta_con_commenti.pwo', 'relax', '7.2', 1,
     'ended', 'converged', 'converged', 'complete', 'completed', 'completed'),
    ('file_output_scf_convergenza_raggiunta_con_stress_e_pressioneDiCella_e_con_commenti.pwo',
     'scf', '7.1', 1, 'ended', 'converged', 'n/a', 'n/a', 'completed', 'completed'),
    ('relax_andato_correttamente_a_convergenza.pwo', 'relax', '7.2', 1,
     'ended', 'converged', 'converged', 'complete', 'completed', 'completed'),
    ('relax_convergenza_non_raggiunta_dopo_200_iterazioni_con_commenti.pwo', 'relax', '7.2', 3,
     'ended', 'not_converged', 'not_converged', 'absent', 'scf_not_converged', 'not_converged'),
    ('relax_crash_con_errore_con_commenti.pwo', 'relax', '7.1', 1,
     'crashed', None, 'not_converged', 'absent', 'error', 'crashed'),
    ('scf_interrotto_per_crash_durante_calcolo_con_commenti.pwo', 'scf', '7.2', 1,
     None, None, 'n/a', 'n/a', 'truncated', 'unknown'),
    ('scf_interrotto_per_crash_durante_calcolo__con_messaggio_sistema_con_commenti.pwo', 'scf',
     '7.2', 1, None, None, 'n/a', 'n/a', 'truncated', 'unknown'),
]


def read_corpus_file(name):
    """Text of a corpus file without the comment lines."""
    text = (Path(CORPUS) / name).read_text(encoding='utf-8')
    return '\n'.join(line for line in text.splitlines() if not COMMENT.match(line)) + '\n'


@pytest.mark.parametrize('name, calculation, version, runs, process, scf, geometry, '
                         'coordinates, reason, summary', EXPECTED,
                         ids=[case[0] for case in EXPECTED])
def test_corpus_file(name, calculation, version, runs, process, scf, geometry, coordinates,
                     reason, summary):
    status = pw_status(read_corpus_file(name), calculation)
    assert (status.version, status.runs, status.process, status.scf, status.geometry,
            status.final_coordinates, status.stop_reason) == \
        (version, runs, process, scf, geometry, coordinates, reason)
    assert status.version_validated
    assert record_status(status).summary == summary


@pytest.mark.parametrize('name, calculation, mixing_beta, nstep', [
    ('file_output_relax_convergenza_raggiunta_con_commenti.pwo', 'relax', 0.25, 200),
    ('file_output_scf_convergenza_raggiunta_con_stress_e_pressioneDiCella_e_con_commenti.pwo',
     'scf', 0.3, None),
    ('relax_andato_correttamente_a_convergenza.pwo', 'relax', 0.3, 200),
    # Three runs, 0.3 then 0.15 twice: the last one is read.
    ('relax_convergenza_non_raggiunta_dopo_200_iterazioni_con_commenti.pwo', 'relax', 0.15, 200),
    ('scf_interrotto_per_crash_durante_calcolo_con_commenti.pwo', 'scf', 0.2, None),
])
def test_corpus_values_of_the_summary(name, calculation, mixing_beta, nstep):
    status = pw_status(read_corpus_file(name), calculation)
    assert (status.mixing_beta, status.nstep) == (mixing_beta, nstep)


def test_corpus_error_of_pw():
    # The error is printed once per process; the last one is read.
    status = pw_status(read_corpus_file('relax_crash_con_errore_con_commenti.pwo'), 'relax')
    assert status.error == PwError('cdiaghg', '309', 'problems computing cholesky')


def runs_up_to(text, k):
    """The text up to the end of run ``k`` (from 1): its last run is run ``k``."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.lstrip().startswith('Program PWSCF v.')]
    end = starts[k] if k < len(starts) else len(lines)
    return '\n'.join(lines[:end]) + '\n'


THREE_RUNS = 'relax_convergenza_non_raggiunta_dopo_200_iterazioni_con_commenti.pwo'
FROM_SCRATCH = PwRestart(False, False, False, False, False)


@pytest.mark.parametrize('name, run, restart', [
    # From scratch; the optimizer makes its first steps.
    (THREE_RUNS, 1, FROM_SCRATCH),
    # Restart after an SCF without convergence: positions, density and
    # wavefunctions read, the SCF from its first iteration; it stops again in
    # the first SCF, before any step of the optimizer.
    (THREE_RUNS, 2, PwRestart(True, True, True, False, None)),
    # Restart asked for without the data file: everything from the input.
    (THREE_RUNS, 3, FROM_SCRATCH),
    ('file_output_relax_convergenza_raggiunta_con_commenti.pwo', 1, FROM_SCRATCH),
    ('relax_andato_correttamente_a_convergenza.pwo', 1, FROM_SCRATCH),
], ids=['three runs, 1', 'three runs, 2', 'three runs, 3', 'restart without files',
        'from scratch'])
def test_corpus_restart(name, run, restart):
    status = pw_status(runs_up_to(read_corpus_file(name), run), 'relax')
    assert status.restart == restart


def test_corpus_parallelization():
    text = read_corpus_file(THREE_RUNS)
    layouts = {pw_status(runs_up_to(text, k), 'relax').parallelization for k in (1, 2, 3)}
    assert layouts == {('Number of MPI processes: 4', 'K-points division: npool = 2',
                        'R & G space division: proc/nbgrp/npool/nimage = 2')}
    other = pw_status(read_corpus_file('file_output_relax_convergenza_raggiunta_con_commenti.pwo'),
                      'relax')
    assert other.parallelization == ('Number of MPI processes: 4',
                                     'R & G space division: proc/nbgrp/npool/nimage = 4')


@pytest.mark.parametrize('name, calculation', [case[:2] for case in EXPECTED],
                         ids=[case[0] for case in EXPECTED])
def test_corpus_file_of_the_other_calculation_is_an_error(name, calculation):
    other = 'scf' if calculation == 'relax' else 'relax'
    with pytest.raises(ValueError, match='nstep'):
        pw_status(read_corpus_file(name), other)
