# Author: Vito Foderà
# Created: 2026-11-04

"""Tests of the pp.x input (``engines/qe/ppi.py``) and of its place in the fingerprint."""

import pytest

from triboworkflow.engines.qe import ppi
from triboworkflow.engines.qe.model import fingerprint_data, input_fingerprint
from triboworkflow.engines.qe.pwi import PwInput, PwInputError


def test_density_input():
    pp = ppi.density_input('charge_interface', 'OUT', 'density.filplot')
    assert pp.text == ("&INPUTPP\n"
                       "   prefix           = 'charge_interface'\n"
                       "   outdir           = 'OUT'\n"
                       "   filplot          = 'density.filplot'\n"
                       "   plot_num         = 0\n"
                       "   spin_component   = 0\n"
                       "/\n")
    assert '&PLOT' not in pp.text
    assert pp.fingerprint_data() == {
        'program': 'pp.x', 'namelists': {'INPUTPP': [['plot_num', 0], ['spin_component', 0]]}}


def test_density_input_refuses_what_a_namelist_cannot_hold():
    with pytest.raises(PwInputError, match='apostrophe'):
        ppi.density_input("it's", 'OUT', 'density.filplot')


@pytest.mark.parametrize('control, problem', [
    ({'disk_io': 'nowf'}, True),
    ({'disk_io': 'none'}, True),
    ({'disk_io': 'nowf   '}, True),       # pw.x drops the trailing blanks
    ({'Disk_IO': 'none'}, True),         # keys in any case
    ({'disk_io': ' nowf'}, False),       # not the leading ones
    ({'disk_io': 'NOWF'}, False),        # values compared in their case
    ({'disk_io': 'minimal'}, False),     # QE 7.1, 7.2: the default, the density is written
    ({'disk_io': 'low'}, False),
    ({'disk_io': 'medium'}, False),
    ({'disk_io': 'high'}, False),
    ({'disk_io': 1}, False),             # not a string: pw.x decides
    ({}, False),
    ({'tprnfor': True}, False),
])
@pytest.mark.parametrize('version', ['7.1', '7.2', '7.2.1'])
def test_disk_io_problem(control, problem, version):
    found = ppi.disk_io_problem(control, version)
    assert (found is not None) == problem
    if problem:
        key, value = next(iter(control.items()))
        release = version[:3]
        assert found == (f'{key} = {value!r}: pw.x of QE {release} writes no charge density '
                         "with disk_io = 'nowf' or 'none', and pp.x reads it")


@pytest.mark.parametrize('value, problem', [
    ('minimal', True), ('none', True), ('minimal  ', True), ('MINIMAL', False),
    ('nowf', False),                     # QE 7.3: the charge density is written
    ('low', False),
])
@pytest.mark.parametrize('version', ['7.3', '7.3.1'])
def test_disk_io_problem_of_qe_7_3(value, problem, version):
    found = ppi.disk_io_problem({'disk_io': value}, version)
    assert (found is not None) == problem
    if problem:
        assert found == (f'disk_io = {value!r}: pw.x of QE 7.3 writes no charge density with '
                         "disk_io = 'minimal' or 'none', and pp.x reads it")


@pytest.mark.parametrize('version, given', [
    (None, 'not given'), ('7.4', "'7.4', not one of 7.1, 7.2, 7.3"),
    ('6.8.1', "'6.8.1', not one of 7.1, 7.2, 7.3")])
@pytest.mark.parametrize('value', ['nowf', 'minimal', 'none'])
def test_disk_io_problem_without_a_known_release(version, given, value):
    found = ppi.disk_io_problem({'disk_io': value}, version)
    assert found == (f"disk_io = {value!r}: with [qe] version {given}, disk_io = 'nowf' or "
                     "'minimal' or 'none' are refused, as pw.x writes no charge density with "
                     "them in some release (7.1: 'nowf' and 'none'; 7.2: 'nowf' and 'none'; "
                     "7.3: 'minimal' and 'none'), and pp.x reads it; give [qe] version")
    assert ppi.disk_io_problem({'disk_io': 'low'}, version) is None


def test_no_density_disk_io():
    assert ppi.no_density_disk_io('7.2.1') == ('7.2', ('nowf', 'none'))
    assert ppi.no_density_disk_io('7.3') == ('7.3', ('minimal', 'none'))
    assert ppi.no_density_disk_io('7.30') == (None, ('nowf', 'minimal', 'none'))
    assert ppi.no_density_disk_io(None) == (None, ('nowf', 'minimal', 'none'))


START = '     Program POST-PROC v.7.2 starts on  1Jan2026 at 11: 0: 0 '
DONE = '=' + '-' * 78 + '=\n   JOB DONE.\n' + '=' + '-' * 78 + '='
ERROR = (' ' + '%' * 78 + '\n     Error in routine postproc (1):\n     reading inputpp namelist\n'
         ' ' + '%' * 78 + '\n\n     stopping ...')


@pytest.mark.parametrize('text, problem', [
    (f'{START}\n     Calling punch_plot, plot_num =   0\n{DONE}\n', None),
    # The message of chdens without &PLOT is not an error.
    (f'{START}\n     Message from routine chdens:\n     namelist plot not found or invalid, '
     f'exiting\n{DONE}\n', None),
    (f'{START}\n{ERROR}\n', 'pp.x stopped on an error in routine postproc (1): reading inputpp '
                            'namelist'),
    (f'{START}\n     Calling punch_plot, plot_num =   0\n', 'pp.x did not close its run (no '
                                                           '"JOB DONE.")'),
    ('', 'no "Program POST-PROC v." line: pp.x did not start, or this is not its output'),
    # Only the last run counts.
    (f'{START}\n{ERROR}\n{START}\n{DONE}\n', None),
])
def test_pp_output_problem(text, problem):
    assert ppi.pp_output_problem(text) == problem


def test_post_processing_enters_the_fingerprint_only_when_given():
    pw = PwInput('&CONTROL\n/\n', 'scf', namelists=(('CONTROL', (('calculation', 'scf'),)),),
                 cards=(('K_POINTS', ('K_POINTS gamma',)),))
    model = {'version': '7.2'}
    plain = fingerprint_data(pw, model, 'slab_from_scratch')
    assert 'post_processing' not in plain
    pp = ppi.density_input('p', 'OUT', 'density.filplot')
    with_pp = fingerprint_data(pw, model, 'slab_from_scratch', pp)
    assert with_pp == {**plain, 'post_processing': pp.fingerprint_data()}
    assert input_fingerprint(pw, model, 'slab_from_scratch') != \
        input_fingerprint(pw, model, 'slab_from_scratch', pp)
    # Files and folders do not enter it.
    other = ppi.density_input('q', 'OUT2', 'other.filplot')
    assert input_fingerprint(pw, model, 'slab_from_scratch', pp) == \
        input_fingerprint(pw, model, 'slab_from_scratch', other)
