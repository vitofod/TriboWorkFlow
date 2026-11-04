# Author: Vito Foderà
# Created: 2026-11-04

"""pp.x input and output, for the charge density of a pw.x calculation.

``density_input`` writes the input with which pp.x, run after pw.x in the
same directory, writes the total charge density of the calculation in the
file ``filplot``::

    &INPUTPP
       prefix           = '<prefix of pw.x>'
       outdir           = '<outdir of pw.x>'
       filplot          = '<file>'
       plot_num         = 0
       spin_component   = 0
    /

``plot_num = 0`` is the electron (pseudo-)charge density and
``spin_component = 0`` its total, also with ``nspin = 2`` or noncollinear
spin; the settings of pw.x (``nspin``, magnetization) are not changed. The
input has no ``&PLOT``: pp.x then only writes ``filplot``, saying that the
namelist ``plot`` is not found. pp.x reads the XML file, the
pseudopotentials and the charge density that pw.x leaves in ``outdir``, not
the wavefunctions.

Which values of ``disk_io`` leave no charge density depends on the release
of pw.x: ``'nowf'`` and ``'none'`` in QE 7.1 and 7.2, ``'minimal'`` and
``'none'`` in QE 7.3, where ``'nowf'`` writes it (``NO_DENSITY_DISK_IO``).
``disk_io_problem`` tells whether the ``&CONTROL`` of the settings has one of
the values of the expected release (``[qe] version``), compared as pw.x
compares it (the string as written, trailing blanks dropped, upper and lower
case apart). Without a version, or with a release not in the table, all of
them are refused, as some release in the table writes no charge density with
each. The check is made only for the calculations that give the charge
density; in any other calculation ``disk_io`` is written as given.

``pp_output_problem`` reads the output of pp.x: its last run (from the line
``Program POST-PROC v.``) must close with ``JOB DONE.`` and have no error
(``Error in routine``, read as in a pw.x output, ``status.py``).
"""

from dataclasses import dataclass

from .keys import normal_key
from .pwi import namelist_lines
from .status import _JOB_DONE, _error, _has, _starts

PROGRAM = 'pp.x'
# First line of a run of pp.x (Modules/environment.f90, environment_start).
RUN_START = 'Program POST-PROC v.'
NAMELIST = 'INPUTPP'
# The total charge density.
DENSITY_ENTRIES = (('plot_num', 0), ('spin_component', 0))
# Keys naming files and folders, left out of the fingerprint.
RESOURCE_KEYS = ('prefix', 'outdir', 'filplot')
# disk_io values with which pw.x writes no charge density, by release
# (major.minor; PW/src/input.f90 and PW/src/punch.f90 of each release).
NO_DENSITY_DISK_IO = {'7.1': ('nowf', 'none'),
                      '7.2': ('nowf', 'none'),
                      '7.3': ('minimal', 'none')}
# Without a release of the table: every value of the table.
ANY_NO_DENSITY_DISK_IO = ('nowf', 'minimal', 'none')


@dataclass(frozen=True)
class PpInput:
    """A pp.x input.

    Attributes
    ----------
    text : str
        Content of the input file.
    entries : tuple
        Keys and values of ``&INPUTPP`` as written: ``((key, value), ...)``.
    """

    text: str
    entries: tuple

    def fingerprint_data(self):
        """What the fingerprint of the calculation takes of it: program and keys, not files."""
        return {'program': PROGRAM,
                'namelists': {NAMELIST: [[key, value] for key, value in self.entries
                                         if key not in RESOURCE_KEYS]}}


def density_input(prefix, outdir, filplot):
    """The pp.x input that writes the total charge density of a calculation.

    Parameters
    ----------
    prefix, outdir : str
        ``prefix`` and ``outdir`` of the pw.x calculation.
    filplot : str
        File to write, relative to the directory pp.x runs in.

    Returns
    -------
    PpInput

    Raises
    ------
    PwInputError
        For a value that a namelist cannot hold (``pwi.fortran_value``).
    """
    entries = (('prefix', prefix), ('outdir', outdir), ('filplot', filplot), *DENSITY_ENTRIES)
    return PpInput(''.join(namelist_lines(NAMELIST, entries, NAMELIST)), entries)


def no_density_disk_io(version):
    """The ``disk_io`` values with which pw.x of a version writes no charge density.

    Parameters
    ----------
    version : str or None
        Expected version of pw.x (``[qe] version``): ``major.minor`` or
        ``major.minor.patch``.

    Returns
    -------
    release : str or None
        ``major.minor`` of ``version`` when it is in ``NO_DENSITY_DISK_IO``;
        None otherwise.
    values : tuple of str
        The values of that release, or ``ANY_NO_DENSITY_DISK_IO``.
    """
    release = None if version is None else '.'.join(version.split('.')[:2])
    if release in NO_DENSITY_DISK_IO:
        return release, NO_DENSITY_DISK_IO[release]
    return None, ANY_NO_DENSITY_DISK_IO


def disk_io_problem(control, version=None):
    """Why pw.x would write no charge density with the ``&CONTROL`` of the settings.

    Parameters
    ----------
    control : dict
        Keys and values of ``[namelists.CONTROL]``, as written.
    version : str or None
        Expected version of pw.x (``[qe] version``), which chooses the values
        (``no_density_disk_io``).

    Returns
    -------
    str or None
        The key and its value, with the reason, when ``disk_io`` (in any
        case) is a string equal to one of those values once its trailing
        blanks are dropped; None otherwise.
    """
    release, values = no_density_disk_io(version)
    for key, value in control.items():
        if normal_key(key) == 'disk_io' and isinstance(value, str) \
                and value.rstrip(' ') in values:
            listed = ' or '.join(map(repr, values))
            if release is not None:
                return (f'{key} = {value!r}: pw.x of QE {release} writes no charge density with '
                        f'disk_io = {listed}, and pp.x reads it')
            known = '; '.join(f'{r}: {" and ".join(map(repr, v))}'
                              for r, v in NO_DENSITY_DISK_IO.items())
            given = 'not given' if version is None else \
                f'{version!r}, not one of {", ".join(NO_DENSITY_DISK_IO)}'
            return (f'{key} = {value!r}: with [qe] version {given}, disk_io = {listed} are '
                    f'refused, as pw.x writes no charge density with them in some release '
                    f'({known}), and pp.x reads it; give [qe] version')
    return None


def pp_output_problem(text):
    """Why a pp.x output does not show a run that ended well.

    Parameters
    ----------
    text : str
        Content of the output.

    Returns
    -------
    str or None
        No ``Program POST-PROC v.`` line; the last error of the last run
        (``error in routine <name> (<code>): <message>``); no ``JOB DONE.`` in
        the last run. None when the last run closed without errors.
    """
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if _starts(line, RUN_START)]
    if not starts:
        return f'no "{RUN_START}" line: pp.x did not start, or this is not its output'
    run = lines[starts[-1]:]
    error = _error(run)
    if error is not None:
        return f'pp.x stopped on an {error}'
    if not _has(run, _JOB_DONE):
        return f'pp.x did not close its run (no "{_JOB_DONE}")'
    return None
