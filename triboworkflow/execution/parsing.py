# Author: Vito Foderà
# Created: 2026-11-03

"""Reading of the results of a calculation from the files of its directory.

``parse_calculation`` reads the output of the current attempt of a
calculation (``output.pwo``) with the structure of its input
(``structure_in.extxyz``), as ``engines/qe/results.py`` does, and writes the
final structure in ``structure_out.extxyz`` when the output is parsed. It
needs nothing but the files of the directory, so it can be run on its own;
the worker calls it at the end of each attempt, and the reconciliation when
it fixes an attempt whose worker did not end it.

``with_parsed`` puts what it read in the record: the state ``parsing``, the
results, ``observed``, the version printed (``Engine.version``) and, for an
output that cannot be parsed, the reason in the attempt (``parse_error``).

The output is read only when its states say that the calculation is over
with the SCF and the geometry converged; otherwise ``parsing`` stays None.
A structure of the input that cannot be read, an unexpected error of the
parser or a final structure that cannot be written give ``parsing =
'failed'`` with the reason, never an exception: the attempt is recorded in
any case.
"""

import dataclasses
import logging
from pathlib import Path

from triboworkflow.core.store import Record
from triboworkflow.core.store.atomic import read_json, write_text
from triboworkflow.core.structures import StructureError, read_structure
from triboworkflow.engines.qe.results import Parsed, parse_output
from triboworkflow.engines.qe.status import pw_status

from .layout import (INPUT_FILE, OUTPUT_FILE, RECORD_FILE, STRUCTURE_IN_FILE,
                     STRUCTURE_OUT_FILE)

logger = logging.getLogger(__name__)


def _converged(pw):
    return pw.process == 'ended' and pw.scf == 'converged' and pw.geometry in ('converged', 'n/a')


def parse_calculation(path, calculation=None, energy_model=None):
    """Results of the current attempt of a calculation, from the files of its directory.

    Parameters
    ----------
    path : path-like
        Directory of the calculation, with ``output.pwo`` and
        ``structure_in.extxyz``.
    calculation : str, optional
        ``'scf'`` or ``'relax'``; read from ``input.pwi`` when not given.
    energy_model : dict, optional
        Energy model of the record, for the form of ``observed``; read from
        ``record.json`` when not given.

    Returns
    -------
    triboworkflow.engines.qe.results.Parsed
        As ``parse_output``. With ``parsing = 'ok'``, ``structure_out.extxyz``
        is written in the directory.

    Raises
    ------
    OSError
        If ``output.pwo`` (or, when needed, ``input.pwi`` or ``record.json``)
        cannot be read.
    triboworkflow.engines.qe.status.PwOutputError
        If ``output.pwo`` is not a pw.x output whose states can be read.
    """
    path = Path(path)
    if calculation is None:
        # Imported here: the worker imports this module.
        from .worker import read_input
        calculation, _ = read_input(path / INPUT_FILE)
    if energy_model is None:
        energy_model = Record.from_dict(read_json(path / RECORD_FILE),
                                        source=path / RECORD_FILE).energy_model
    text = (path / OUTPUT_FILE).read_text(encoding='utf-8', errors='replace')
    if not _converged(pw_status(text, calculation)):
        return Parsed(None)
    try:
        structure = read_structure(path / STRUCTURE_IN_FILE, format='extxyz')
    except StructureError as error:
        return Parsed('failed', reason=f'input structure not read: {error}')
    try:
        parsed = parse_output(text, calculation, structure, energy_model)
    except Exception as error:  # the attempt is recorded whatever the parser does
        logger.exception('%s: error of the parser', path)
        return Parsed('failed', reason=f'error of the parser: {type(error).__name__}: {error}')
    if parsed.parsing == 'ok':
        try:
            write_text(path / STRUCTURE_OUT_FILE, parsed.structure_text)
        except (OSError, StructureError) as error:
            return dataclasses.replace(parsed, parsing='failed', results=None, structure=None,
                                       reason=f'{STRUCTURE_OUT_FILE} not written: {error}')
    return parsed


def with_parsed(record, attempt, parsed, version):
    """Record and last attempt with what the reading of the output gave.

    Parameters
    ----------
    record : triboworkflow.core.store.Record
    attempt : triboworkflow.core.store.Attempt
        The attempt that wrote the output, with its states (``parsing``
        None); it replaces the last attempt of the record.
    parsed : triboworkflow.engines.qe.results.Parsed
    version : str or None
        Version printed in the output, None if the output could not be read.

    Returns
    -------
    triboworkflow.core.store.Record
        With the states of the attempt and ``parsing``, the results,
        ``observed`` and ``engine.version``; the attempt has ``parse_error``
        when the parsing failed.
    """
    status = dataclasses.replace(attempt.status, parsing=parsed.parsing)
    done = dataclasses.replace(attempt, status=status, parse_error=parsed.reason)
    return dataclasses.replace(record, status=status, attempts=record.attempts[:-1] + (done,),
                               results=parsed.results, observed=parsed.observed,
                               engine=dataclasses.replace(record.engine, version=version))
