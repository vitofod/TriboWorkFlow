# Author: Vito Foderà
# Created: 2026-11-04

"""Integrated check of the workflow with pw.x on an HPC machine: small real calculations.

Run by hand on the HPC machine, from the login node, in the virtual
environment where the package is installed; the protocol is in
``docs/hpc_check.md``. The steps of the interface workflow are not in this
version, so the command line prepares nothing; this script gives it test
steps, whose small pw.x calculations go through the whole chain:
preparation, groups of jobs, polling, relaunches, parsing, charge density.

Commands:

``write MACHINE DIR --case CASE --system {Al,Si}``
    Writes the folder of a new run for one case of the check:
    ``config.toml``, ``jobscript.sh``, the structures and
    ``qe_check/plan.json``, the calculations of each test step. ``MACHINE``
    is a TOML file with the jobscript, the commands, pw.x and the
    pseudopotentials of the machine (``tests/hpc/machine_example.toml``).
``cli ARGS...``
    ``triboworkflow ARGS...`` with the test steps in the registry of the
    steps; ``start`` runs the orchestrator in the background with the
    factory of this script (``daemon_factory``), which puts them there too.
``snapshot RUN``
    On the HPC machine: ``sacct`` (every run of each job) and ``squeue``
    (state, reason) of the jobs of the run, and the claims on disk, saved in
    ``qe_check/snapshots/<time>.json``; taken while jobs are held, after a
    requeue and once the jobs are over.
``report RUN``
    Read only, anywhere the run is copied: for each calculation its attempts
    (job, times, ``max_seconds``, exit code, stop, states, launch and level,
    restart read part by part, parallelization, removal of the output
    directory, time from the stop of pw.x to the end of the worker and of the
    job, pp.x), then the verdicts on what the case looks for (the
    expectations of each calculation in the plan: completed and parsed with
    the energy model expected, stress, moves along z only, symmetries of
    pw.x against spglib, restarts, levels, margin, relaunches, empty claim,
    charge density read as a ``Chgcar`` and by ``QEChargeAnalyzer``; and of
    the run: steps completed, another ``run_command``, held and requeued
    jobs, densities that can be combined). Exit status 1 when a verdict is
    ``PROBLEM``: an inconsistency of the files, or a behaviour other than
    the one the package gives; ``not observed`` is something the run did not
    show (yet), ``note`` a difference to look at (e.g. the symmetries, Q17).

Cases (``--case``), one run each:

``chain``
    ``match`` (no calculation), ``pes`` (scf of the two slabs and of the
    interface), ``adhesion`` (relax of the two slabs, every atom free, and
    of the interface, along z only), with ``tstress``; two jobs per group.
``restart``
    Calculations that stop and are relaunched: a long scf and a long relax
    stopped at ``max_seconds`` by the short walltime, a relax with a small
    ``nstep`` (``max_steps``) and one with few SCF iterations allowed
    (``scf_not_converged``).
``manual``
    Long relaxations to act on by hand: ``scancel``, exit file of pw.x,
    hold, requeue.
``claim``
    A calculation whose worker stops right after creating its claim (see
    "Empty claim" below) and a long one, whose claim belongs to a live
    worker.
``density``
    The interface and its two fragments in the cell of the interface, scf
    with the charge density (pp.x after pw.x), and an scf without it, with
    ``disk_io = 'nowf'``.
``single``
    The single calculation of ``[structure]``, for ``triboworkflow gen``;
    ``jobscript.sh`` runs pw.x on the input generated.

Structures: slabs (111) built with ASE, 1x1 for the small calculations and
a larger supercell for the long ones, with a vacuum on both sides: Al (fcc)
or Si (diamond, whole bilayers only, no passivation). The interface is a
slab of twice the layers with an extra gap between its two halves. Every
run declares a second species without atoms (Si for Al, Al for Si). The
values of the namelists are the defaults of this script, replaced by those
of the machine file and, for some calculations of a case, by the values of
the case (e.g. ``nstep``).

Test steps
----------
Each test step takes its calculations from ``qe_check/plan.json``: label,
calc_type, structure, values of the namelists that replace those of
``config.toml`` for that calculation only, and whether it gives the charge
density. They are built with ``preparation.workflow_calculation``, as the
steps of the workflow build theirs, with the structure and the k-points
that the vendored ``VaspInputSet`` gives for the calc_type (k-point density
``kdens`` of ``[k_points]``). A step is done when its entry in
``system.json`` exists with the provenance of its calculations as the
settings give them now: a change of the settings makes it run again. Its
analysis writes the energy of each calculation.

Empty claim
-----------
In the case ``claim`` the jobscript puts ``qe_check/inject`` first in
``PYTHONPATH``: its ``sitecustomize.py``, loaded by the worker at start-up,
kills the worker (SIGKILL) when it reads the input of a listed calculation
while the claim of that calculation exists and is empty, that is right
after creating it and before writing it and the attempt; once per
calculation (a file in ``qe_check/killed/``). Nothing of the package is
changed: the worker of a real job stops at that point, and the claim stays
empty.
"""

import contextlib
import copy
import dataclasses
import datetime
import io
import json
import os
import re
import shlex
import shutil
import sys
import tomllib
import warnings
from pathlib import Path

from triboworkflow.core.settings import load_settings
from triboworkflow.core.settings.schema import INTERFACE_STEPS, NAMELISTS, key_name
from triboworkflow.core.store import (Provenance, SystemWriter, is_done, read_record,
                                      read_system)
from triboworkflow.core.structures import read_structure, write_structure
from triboworkflow.core.structures.convert import to_structure
from triboworkflow.execution.layout import CLAIM_FILE, calc_dir, system_dir
from triboworkflow.execution.orchestrator import INTERFACE_GRAPH, Step
from triboworkflow.execution.preparation import workflow_calculation

HERE = Path(__file__).resolve().parent
MODULE = 'qe_check'
FACTORY = f'{MODULE}:daemon_factory'
PLAN_FILE = Path('qe_check') / 'plan.json'
INJECT_DIR = Path('qe_check') / 'inject'
EMPTY_CLAIM_FILE = Path('qe_check') / 'empty_claim.json'
KILLED_DIR = Path('qe_check') / 'killed'
STRUCTURES_DIR = 'structures'
CONFIG_FILE = 'config.toml'
JOBSCRIPT_FILE = 'jobscript.sh'
MD5_FILE = 'pseudo.md5'
PLAN_VERSION = 1

CASES = ('chain', 'restart', 'manual', 'claim', 'density', 'single')
SYSTEMS = ('Al', 'Si')
# The species of a run: its own, then one without atoms.
OTHER_SPECIES = {'Al': 'Si', 'Si': 'Al'}

# Namelists of the runs, replaced key by key by those of the machine file.
NAMELIST_DEFAULTS = {
    'CONTROL': {'tprnfor': True, 'tstress': False, 'disk_io': 'low', 'nstep': 150,
                'etot_conv_thr': 1.0e-4, 'forc_conv_thr': 1.0e-3},
    'SYSTEM': {'ecutwfc': 30.0, 'ecutrho': 240.0, 'occupations': 'smearing', 'smearing': 'mv',
               'degauss': 0.02},
    'ELECTRONS': {'conv_thr': 1.0e-7, 'mixing_beta': 0.4, 'mixing_mode': 'local-TF',
                  'electron_maxstep': 100},
    'IONS': {},
}
# Structures: atomic layers of the small slabs, supercell and layers of the
# large ones, vacuum on each side and extra gap of the interface (angstrom),
# shift along z of the top layer of the slabs to relax (angstrom).
STRUCTURE_DEFAULTS = {'layers': 4, 'big_size': [3, 3], 'big_layers': 6, 'vacuum': 5.0,
                      'gap': 0.4, 'shift': 0.15}
# Walltime of each case, seconds: the restart case leaves about two minutes
# to each attempt beyond the margin of 600 s, so that the long calculations
# stop at max_seconds.
WALLTIMES = {'chain': 1800, 'restart': 720, 'manual': 1800, 'claim': 1200, 'density': 1800,
             'single': 1800}
# Values of the case restart: nstep of the relax stopped at max_steps,
# electron_maxstep of the one stopped with scf_not_converged.
RESTART_DEFAULTS = {'nstep': 2, 'electron_maxstep': 6}
# Tighter thresholds of the long relaxations, so that they make more steps.
LONG = {'CONTROL': {'forc_conv_thr': 1.0e-4}, 'ELECTRONS': {'conv_thr': 1.0e-9}}

_MACHINE_KEYS = ('scheduler', 'jobscript', 'run_command', 'pp_command', 'polling_interval',
                 'max_seconds_margin', 'job_report_wait', 'qe', 'species', 'k_points',
                 'namelists', 'structures', 'cases')
_CASE_KEYS = ('walltime', 'groups', 'nstep', 'electron_maxstep')


class CheckError(RuntimeError):
    """A machine file, a case or a run of the check that this script cannot use."""


# --- machine file ------------------------------------------------------------------

def _table(data, key, where):
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise CheckError(f'{where}{key}: a table is required')
    return value


def _required(data, key, where, kind=str):
    value = data.get(key)
    if value is None or value == '' or not isinstance(value, kind):
        raise CheckError(f'{where}{key}: required ({kind.__name__})')
    return value


def read_machine(path):
    """Machine file of the check, with its jobscript.

    Parameters
    ----------
    path : path-like
        TOML file (``tests/hpc/machine_example.toml``).

    Returns
    -------
    dict
        The tables of the file, and ``jobscript_text``: the text of its
        jobscript, read relative to the file.

    Raises
    ------
    CheckError
        For a key not known, a value required and missing, or a jobscript
        without ``{time}``.
    """
    path = Path(path)
    with open(path, 'rb') as f:
        data = tomllib.load(f)
    where = f'{path}: '
    unknown = sorted(set(data) - set(_MACHINE_KEYS))
    if unknown:
        raise CheckError(f'{where}keys not known: {", ".join(unknown)}; known: '
                         f'{", ".join(_MACHINE_KEYS)}')
    for key in ('scheduler', 'jobscript', 'run_command'):
        _required(data, key, where)
    qe = _table(data, 'qe', where)
    for key in ('version', 'pseudo_dir', 'upf_md5'):
        _required(qe, key, f'{where}qe.')
    species = _table(data, 'species', where)
    for label in SYSTEMS:
        entry = _table(species, label, f'{where}species.')
        _required(entry, 'pseudopotential', f'{where}species.{label}.')
        if not isinstance(entry.get('mass'), (int, float)):
            raise CheckError(f'{where}species.{label}.mass: required (number)')
    kdens = _table(data, 'k_points', where).get('kdens')
    if not isinstance(kdens, (int, float)) or kdens <= 0:
        raise CheckError(f'{where}k_points.kdens: required (number > 0)')
    for name, values in _table(data, 'namelists', where).items():
        if name not in NAMELISTS or not isinstance(values, dict):
            raise CheckError(f'{where}namelists.{name}: a table of {", ".join(NAMELISTS)} '
                             'is required')
    unknown = sorted(set(_table(data, 'structures', where)) - set(STRUCTURE_DEFAULTS))
    if unknown:
        raise CheckError(f'{where}structures: keys not known: {", ".join(unknown)}')
    for case, values in _table(data, 'cases', where).items():
        if case not in CASES or not isinstance(values, dict):
            raise CheckError(f'{where}cases.{case}: a table of a case is required '
                             f'({", ".join(CASES)})')
        unknown = sorted(set(values) - set(_CASE_KEYS))
        if unknown:
            raise CheckError(f'{where}cases.{case}: keys not known: {", ".join(unknown)}')
    jobscript = Path(data['jobscript'])
    if not jobscript.is_absolute():
        jobscript = path.parent / jobscript
    text = jobscript.read_text(encoding='utf-8')
    if '{time}' not in text:
        raise CheckError(f'{jobscript}: no {{time}}: write it where the walltime goes (e.g. '
                         '"#SBATCH --time={time}"), since each case sets its own')
    upf_md5 = Path(qe['upf_md5'])
    if not upf_md5.is_absolute():
        upf_md5 = path.parent / upf_md5
    return {**data, 'source': str(path.resolve()), 'jobscript_text': text,
            'upf_md5_path': str(upf_md5)}


def hms(seconds):
    """``HH:MM:SS`` of a number of seconds."""
    seconds = int(seconds)
    return f'{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}'


# --- structures -------------------------------------------------------------------------

def _sorted_centred(atoms, vacuum):
    """Atoms sorted by z, the lowest at ``vacuum``, the cell ``vacuum`` above the highest."""
    import numpy as np

    atoms = atoms[np.argsort(atoms.positions[:, 2], kind='stable')]
    z = atoms.positions[:, 2]
    atoms.positions[:, 2] += vacuum - z.min()
    cell = atoms.cell.array.copy()
    cell[2] = [0.0, 0.0, z.max() - z.min() + 2 * vacuum]
    atoms.set_cell(cell)
    return atoms


def slab_atoms(system, size, layers, vacuum):
    """A (111) slab of ``layers`` atomic layers, sorted by z.

    Si keeps whole bilayers only (two layers 0.78 angstrom apart, bound by
    three bonds per atom), so that each surface atom has one broken bond:
    ``layers`` must be even. ASE ends the slab with a layer bound to the rest
    by one bond; two more layers are built and the lowest ``layers`` that
    belong to whole bilayers are kept.
    """
    import numpy as np
    from ase.build import diamond111, fcc111

    if system == 'Al':
        return _sorted_centred(fcc111('Al', size=(*size, layers), vacuum=vacuum,
                                      orthogonal=False), vacuum)
    if layers % 2:
        raise CheckError(f'Si: layers = {layers}: whole bilayers only, give an even number')
    atoms = diamond111('Si', size=(*size, layers + 2), vacuum=vacuum, orthogonal=False)
    z = np.round(atoms.positions[:, 2], 3)
    levels = np.unique(z)
    paired = set()
    for low, high in zip(levels, levels[1:]):
        if high - low < 1.5 and low not in paired:
            paired.update((low, high))
    kept = sorted(paired)[:layers]
    if len(kept) < layers:
        raise CheckError(f'Si: {layers} layers in whole bilayers not found in the slab of ASE')
    return _sorted_centred(atoms[[i for i in range(len(atoms)) if z[i] in kept]], vacuum)


def shifted_top(atoms, shift):
    """Copy of a slab with its highest layer moved up by ``shift``."""
    atoms = atoms.copy()
    z = atoms.positions[:, 2]
    atoms.positions[z > z.max() - 0.1, 2] += shift
    return atoms


def build_structures(system, options):
    """The structures of every case, as ASE atoms by name.

    ``bottom`` and ``top`` are the two slabs, ``interface`` the slab of twice
    the layers with ``gap`` added between its halves, ``interface_bottom``
    and ``interface_top`` its halves in its cell; ``small_shifted`` is
    ``bottom`` with its top layer moved; ``big`` and ``big_shifted`` the
    larger supercell.
    """
    n, vacuum = options['layers'], options['vacuum']
    double = slab_atoms(system, (1, 1), 2 * n, vacuum)
    half = len(double) // 2
    interface = double.copy()
    interface.positions[half:, 2] += options['gap']
    interface = _sorted_centred(interface, vacuum)
    bottom = slab_atoms(system, (1, 1), n, vacuum)
    top = _sorted_centred(double[half:], vacuum)
    big = slab_atoms(system, tuple(options['big_size']), options['big_layers'], vacuum)
    return {'bottom': bottom, 'top': top, 'interface': interface,
            'interface_bottom': interface[:half], 'interface_top': interface[half:],
            'small_shifted': shifted_top(bottom, options['shift']), 'big': big,
            'big_shifted': shifted_top(big, options['shift'])}


# --- cases ---------------------------------------------------------------------------

def _calc(label, calc_type, structure, purpose, expect, namelists=None, density=False):
    return {'label': label, 'calc_type': calc_type,
            'structure': f'{STRUCTURES_DIR}/{structure}.extxyz', 'purpose': purpose,
            'expect': list(expect), 'namelists': namelists or {}, 'density': density}


# What the report looks for, per calculation (``expect`` of the plan; see
# ``EXPECTATIONS``) and per run (``RUN_EXPECTATIONS``).
SCF_DONE = ('completed', 'symmetry')
RELAX_DONE = ('completed', 'stress', 'symmetry')


def case_plan(case, values):
    """Steps, calculations and settings of a case.

    Parameters
    ----------
    case : str
        One of ``CASES`` but ``single``.
    values : dict
        ``[cases.<case>]`` of the machine file.

    Returns
    -------
    steps : dict
        ``{step: [calculation, ...]}`` in the order of ``[interface].steps``.
    control : dict
        ``&CONTROL`` values of the run, over those of the machine.
    groups : int
        Jobs per group.
    empty_claim : list of str
        ``calc_id`` of the calculations whose worker stops at the claim.
    run_expect : list of str
        What the report looks for in the whole run (``RUN_EXPECTATIONS``).
    """
    if case == 'chain':
        steps = {'match': [],
                 'pes': [_calc('bottom', 'slab_from_scratch', 'bottom', 'scf of a slab',
                               SCF_DONE),
                         _calc('top', 'slab_from_scratch', 'top', 'scf of a slab', SCF_DONE),
                         _calc('interface', 'slab_from_scratch', 'interface',
                               'scf of the interface', SCF_DONE)],
                 'adhesion': [_calc('bottom', 'slab_pos_relax', 'bottom', 'relax, atoms free',
                                    RELAX_DONE),
                              _calc('top', 'slab_pos_relax', 'top', 'relax, atoms free',
                                    RELAX_DONE),
                              _calc('interface', 'interface_z_relax', 'interface',
                                    'relax along z only', (*RELAX_DONE, 'z_only'))]}
        return steps, {'tstress': True}, 2, [], ['steps_completed']
    if case == 'restart':
        nstep = values.get('nstep', RESTART_DEFAULTS['nstep'])
        maxstep = values.get('electron_maxstep', RESTART_DEFAULTS['electron_maxstep'])
        steps = {'adhesion': [
            _calc('long_scf', 'slab_from_scratch', 'big', 'scf stopped at max_seconds, '
                  'restarted', ('restart_after:max_seconds', 'scf_iteration', 'margin'),
                  {'ELECTRONS': {'conv_thr': 1.0e-10, 'electron_maxstep': 500}}),
            _calc('long_relax', 'slab_pos_relax', 'big_shifted', 'relax stopped at '
                  'max_seconds, restarted', ('restart_after:max_seconds', 'margin'), LONG),
            _calc('few_steps', 'slab_pos_relax', 'small_shifted', 'relax stopped at '
                  'max_steps, level 2', ('restart_after:max_steps', 'level:2', 'completed'),
                  {'CONTROL': {'nstep': nstep}}),
            _calc('scf_fail', 'slab_pos_relax', 'small_shifted', 'relax stopped with '
                  'scf_not_converged, restarted', ('restart_after:scf_not_converged',),
                  {'ELECTRONS': {'electron_maxstep': maxstep}})]}
        return steps, {}, 4, [], ['other_command']
    if case == 'manual':
        steps = {'adhesion': [
            _calc('to_cancel', 'slab_pos_relax', 'big_shifted',
                  'scancel, then relaunch from scratch', ('scratch_after:killed',), LONG),
            _calc('to_exit', 'slab_pos_relax', 'big_shifted', 'exit file, then relaunch by '
                  'hand', ('manual_after:user_request',), LONG),
            _calc('to_hold', 'slab_pos_relax', 'big_shifted', 'long: its job may be held',
                  (), LONG),
            _calc('to_requeue', 'slab_pos_relax', 'big_shifted', 'long: its job may be '
                  'requeued', (), LONG)]}
        return steps, {}, 4, [], ['held', 'requeued']
    if case == 'claim':
        steps = {'adhesion': [
            _calc('victim', 'slab_from_scratch', 'bottom', 'worker stopped at the claim',
                  ('empty_claim_released', 'completed')),
            _calc('alive', 'slab_pos_relax', 'big_shifted', 'claim of a live worker',
                  ('completed',), LONG)]}
        return steps, {}, 2, ['adhesion/victim'], []
    if case == 'density':
        steps = {'charge': [
            _calc('interface', 'slab_from_scratch', 'interface', 'density of the interface',
                  ('completed', 'density'), density=True),
            _calc('bottom', 'slab_from_scratch', 'interface_bottom', 'density of a fragment',
                  ('completed', 'density'), density=True),
            _calc('top', 'slab_from_scratch', 'interface_top', 'density of a fragment',
                  ('completed', 'density'), density=True),
            _calc('nowf', 'slab_from_scratch', 'bottom', "no density, disk_io = 'nowf'",
                  ('completed', 'nowf'), {'CONTROL': {'disk_io': 'nowf'}})]}
        return steps, {}, 4, [], ['comparable']
    raise CheckError(f'case {case!r} has no steps')


# --- TOML ----------------------------------------------------------------------------

def _toml_key(key):
    return key if key.replace('_', '').replace('-', '').isalnum() and key.isascii() \
        else json.dumps(key)


def _toml_value(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_toml_value(v) for v in value) + ']'
    raise CheckError(f'{value!r} cannot be written in TOML here')


def _toml_table(name, values, array=False):
    lines = [f'[[{name}]]' if array else f'[{name}]']
    lines += [f'{_toml_key(k)} = {_toml_value(v)}' for k, v in values.items()]
    return '\n'.join(lines) + '\n'


def _namelists(machine, control):
    """Namelists of a run: defaults, then the machine, then ``control``, key by key."""
    result = copy.deepcopy(NAMELIST_DEFAULTS)
    result['CONTROL']['pseudo_dir'] = machine['qe']['pseudo_dir']
    for changes in (machine.get('namelists', {}), {'CONTROL': control}):
        for name, values in changes.items():
            table = result[name]
            for key, value in values.items():
                for old in [k for k in table if key_name(k) == key_name(key) and '(' not in k]:
                    del table[old]
                table[key] = value
    return result


def config_text(machine, case, system, steps, control, groups):
    """Text of ``config.toml`` of a run of the check."""
    other = OTHER_SPECIES[system]
    head = (f'# Written by tests/hpc/qe_check.py write, case {case}, system {system}, on '
            f'{datetime.date.today().isoformat()}.\n# Run of the check, not of the workflow: '
            'see docs/hpc_check.md.\n\nschema_version = 1\n\n')
    parts = [head]
    if case == 'single':
        parts.append(_toml_table('structure', {'file': f'{STRUCTURES_DIR}/bottom.extxyz'}))
        control = {'calculation': 'scf', 'prefix': 'single', 'outdir': 'OUT', **control}
    else:
        parts.append(_toml_table('interface', {'steps': list(steps), 'engine': 'qe',
                                               'set_bm': False}))
        for name in ('bottom', 'top'):
            parts.append(_toml_table('interface.slabs', {
                'file': f'{STRUCTURES_DIR}/{name}.extxyz', 'mid': f'{system}111',
                'formula': system, 'miller': [1, 1, 1]}, array=True))
    for name, values in _namelists(machine, control).items():
        if values:
            parts.append(_toml_table(f'namelists.{name}', values))
    for label in (system, other):
        entry = machine['species'][label]
        parts.append(_toml_table('atomic_species', {
            'label': label, 'mass': entry['mass'],
            'pseudopotential': entry['pseudopotential']}, array=True))
    parts.append(_toml_table('k_points', {'mode': 'kdens',
                                          'kdens': machine['k_points']['kdens']}))
    parts.append(_toml_table('qe', {'version': machine['qe']['version'], 'upf_md5': MD5_FILE}))
    if case == 'single':
        parts.append(_toml_table('generation', {'file_prefix': 'input.pwi', 'overwrite': False}))
    else:
        execution = {'scheduler': machine['scheduler'], 'jobscript': JOBSCRIPT_FILE,
                     'run_command': machine['run_command']}
        if machine.get('pp_command'):
            execution['pp_command'] = machine['pp_command']
        execution.update(groups=groups, walltime=walltime_of(machine, case),
                         polling_interval=machine.get('polling_interval', 120),
                         max_seconds_margin=machine.get('max_seconds_margin', 600),
                         job_report_wait=machine.get('job_report_wait', 3600))
        parts.append(_toml_table('execution', execution))
    return '\n'.join(parts)


def walltime_of(machine, case):
    """Walltime of a case, seconds: ``[cases.<case>].walltime``, else ``WALLTIMES``."""
    return machine.get('cases', {}).get(case, {}).get('walltime', WALLTIMES[case])


# --- empty claim --------------------------------------------------------------------

SITECUSTOMIZE = '''\
# Written by tests/hpc/qe_check.py, case claim: not part of the package.
"""Stop the worker right after it creates the claim of chosen calculations (empty claim).

Loaded at start-up by the Python of the jobs of the case claim, which have
this directory first in PYTHONPATH. When the worker reads the input of a
calculation of QE_CHECK_EMPTY_CLAIM while its claim exists and is empty
(created, not written yet), the worker is killed with SIGKILL, once per
calculation: the claim stays empty and no attempt is written.
"""
import os

_SPEC = os.environ.get('QE_CHECK_EMPTY_CLAIM')
if _SPEC:
    import json
    import signal
    import socket

    import ase.io.espresso as _espresso

    _read = _espresso.read_fortran_namelist

    def read_fortran_namelist(fileobj, *args, **kwargs):
        name = getattr(fileobj, 'name', None)
        if isinstance(name, str):
            with open(_SPEC) as f:
                spec = json.load(f)
            calc = os.path.realpath(os.path.dirname(os.path.abspath(name)))
            claim = os.path.join(calc, spec['claim_file'])
            if (calc in spec['calculations'] and os.path.isfile(claim)
                    and os.path.getsize(claim) == 0):
                marker = os.path.join(spec['killed'], spec['calculations'][calc])
                try:
                    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
                except FileExistsError:
                    pass
                else:
                    os.write(fd, json.dumps({
                        'calculation': spec['calculations'][calc], 'pid': os.getpid(),
                        'host': socket.gethostname(),
                        'job_id': os.environ.get('SLURM_JOB_ID')}).encode() + b'\\n')
                    os.fsync(fd)
                    os.close(fd)
                    os.kill(os.getpid(), signal.SIGKILL)
        return _read(fileobj, *args, **kwargs)

    _espresso.read_fortran_namelist = read_fortran_namelist
'''


def empty_claim_lines(run_dir):
    """Lines added to the jobscript of the case claim (see "Empty claim")."""
    spec, inject = run_dir / EMPTY_CLAIM_FILE, run_dir / INJECT_DIR
    return ('\n# qe_check, case claim: the worker stops right after creating the claim of the\n'
            f'# calculations of {EMPTY_CLAIM_FILE}, once each ({INJECT_DIR}/sitecustomize.py).\n'
            f'export QE_CHECK_EMPTY_CLAIM={shlex.quote(str(spec))}\n'
            f'export PYTHONPATH={shlex.quote(str(inject))}${{PYTHONPATH:+:$PYTHONPATH}}\n')


def write_empty_claim(run_dir, system, calc_ids):
    """``sitecustomize.py``, the list of the calculations and the folder of the markers."""
    (run_dir / INJECT_DIR).mkdir(parents=True)
    (run_dir / INJECT_DIR / 'sitecustomize.py').write_text(SITECUSTOMIZE)
    (run_dir / KILLED_DIR).mkdir(parents=True)
    calculations = {os.path.realpath(calc_dir(run_dir, system, calc_id)): calc_id.split('/')[1]
                    for calc_id in calc_ids}
    (run_dir / EMPTY_CLAIM_FILE).write_text(json.dumps(
        {'claim_file': CLAIM_FILE, 'killed': os.path.realpath(run_dir / KILLED_DIR),
         'calculations': calculations}, indent=2) + '\n')


# --- write -----------------------------------------------------------------------------

def write(args):
    """Command ``write``: the folder of a new run of one case."""
    machine = read_machine(args.machine)
    run_dir = Path(args.dir).resolve()
    if run_dir.exists():
        raise CheckError(f'{run_dir} exists: give a new directory')
    case, system = args.case, args.system
    values = machine.get('cases', {}).get(case, {})
    options = {**STRUCTURE_DEFAULTS, **machine.get('structures', {})}
    if case == 'single':
        steps, control, groups, empty, run_expect = {}, {}, None, [], ['single']
    else:
        steps, control, groups, empty, run_expect = case_plan(case, values)
        groups = args.groups or values.get('groups', groups)
        if case in ('density',) and not machine.get('pp_command'):
            raise CheckError(f'{machine["source"]}: pp_command required by the case {case}')
    upf_md5 = Path(machine['upf_md5_path'])
    if not upf_md5.is_file():
        raise CheckError(f'{upf_md5} not found: write it with "md5sum *.UPF > {upf_md5.name}" '
                         'in the pseudo_dir')
    run_dir.mkdir(parents=True)
    (run_dir / 'qe_check').mkdir()
    shutil.copyfile(machine['source'], run_dir / 'qe_check' / 'machine.toml')
    shutil.copyfile(upf_md5, run_dir / MD5_FILE)
    structures = run_dir / STRUCTURES_DIR
    structures.mkdir()
    for name, atoms in build_structures(system, options).items():
        write_structure(to_structure(atoms), structures / f'{name}.extxyz')
    (run_dir / CONFIG_FILE).write_text(
        config_text(machine, case, system, steps, control, groups), encoding='utf-8')
    jobscript = machine['jobscript_text'].replace('{time}', hms(walltime_of(machine, case)))
    jobscript += '' if jobscript.endswith('\n') else '\n'
    if case == 'single':
        jobscript += f'\n{machine["run_command"]} -in input.pwi >> output.pwo\n'
    elif empty:
        jobscript += empty_claim_lines(run_dir)
    (run_dir / JOBSCRIPT_FILE).write_text(jobscript, encoding='utf-8')
    system_name = f'{system}111_{system}111'
    if empty:
        write_empty_claim(run_dir, system_name, empty)
    plan = {'schema_version': PLAN_VERSION, 'case': case, 'system': system,
            'written': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
            'command': shlex.join(sys.argv), 'structures': options, 'steps': steps,
            'empty_claim': empty, 'run_expect': run_expect}
    (run_dir / PLAN_FILE).write_text(json.dumps(plan, indent=2) + '\n')
    try:
        count = check_run(run_dir)
    except (CheckError, OSError, ValueError) as error:
        raise CheckError(f'{run_dir} written, but its calculations cannot be prepared: {error}; '
                         'remove the directory once the machine file is fixed') from error
    print(f'Run written in {run_dir}: case {case}, system {system}, {count} '
          f'calculation{"" if count == 1 else "s"}'
          + (f' in {", ".join(s for s in steps if steps[s])}' if steps else ''))
    me = shlex.quote(str(Path(__file__).resolve()))
    if case == 'single':
        print(f'Next: cd {shlex.quote(str(run_dir))} && triboworkflow gen {CONFIG_FILE} && '
              f'sbatch {JOBSCRIPT_FILE}')
    else:
        print(f'Next: python {me} cli check -r {shlex.quote(str(run_dir))}')
        print(f'      python {me} cli start --dry-run -r {shlex.quote(str(run_dir))}')
    return 0


def check_run(run_dir):
    """Read the settings of a run written and build its calculations; their number.

    Raises
    ------
    SettingsError, PwInputError, StructureError
        As the settings and the preparation of the calculations.
    """
    from triboworkflow.cli.run import system_name

    settings = load_settings(Path(run_dir) / CONFIG_FILE)
    if settings.interface is None:
        from triboworkflow.engines.qe.prepare import single_input
        single_input(settings)
        return 1
    return sum(len(STEP_CLASSES[name]().calculations(run_dir, system_name(settings)))
               for name in settings.interface.steps)


# --- test steps --------------------------------------------------------------------------

def read_plan(run_dir):
    """``qe_check/plan.json`` of a run of the check.

    Raises
    ------
    CheckError
        If the run has none, or of another version.
    """
    path = Path(run_dir) / PLAN_FILE
    if not path.is_file():
        raise CheckError(f'{path} not found: the test steps run only in a run written by '
                         '"qe_check.py write"')
    plan = json.loads(path.read_text())
    if plan.get('schema_version') != PLAN_VERSION:
        raise CheckError(f'{path}: schema_version {plan.get("schema_version")!r}, this script '
                         f'reads {PLAN_VERSION}')
    return plan


def with_namelists(settings, changes):
    """Settings whose namelists have the values of ``changes``, key by key.

    Parameters
    ----------
    settings : triboworkflow.core.settings.Settings
    changes : dict
        ``{namelist: {key: value}}``; a key replaces the one of the same name
        whatever its case.

    Returns
    -------
    Settings
        A copy, with the same ``source``; ``settings`` itself without changes.
    """
    if not changes:
        return settings
    namelists = copy.deepcopy(settings.namelists)
    for name, values in changes.items():
        table = getattr(namelists, name)
        for key, value in values.items():
            old = namelists.key(name, key_name(key))
            if old is not None:
                del table[old]
            table[key] = value
    changed = dataclasses.replace(settings, namelists=namelists)
    changed.source = settings.source
    return changed


def vasp_kpoints(structure, calc_type, kdens):
    """k-points of the vendored ``VaspInputSet`` for a calc_type; it also sets the constraints.

    The structure gets the ``selective_dynamics`` of the calc_type, as the
    workflow code sets them.
    """
    from pymatgen.io.vasp.sets import BadInputSetWarning

    from triboworkflow.vendored.highput.utils.vasp_tools import VaspInputSet

    # The vendored code prints, and pymatgen warns about the POTCAR functional
    # of the VASP input set, which pw.x does not use.
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Overriding the POTCAR functional',
                                BadInputSetWarning)
        return VaspInputSet(structure, {'kdens': kdens}, calc_type).get_vasp_settings().kpoints


class CheckStep(Step):
    """A test step: the calculations of its name in ``qe_check/plan.json``."""

    name = None

    def _items(self, run_dir):
        steps = read_plan(run_dir)['steps']
        if self.name not in steps:
            raise CheckError(f'step {self.name!r} is not in the case of this run: '
                             f'{", ".join(steps)}')
        return steps[self.name]

    def calculations(self, run_dir, system):
        from triboworkflow.cli.run import load_run

        settings = load_run(run_dir)
        if settings.k_points.mode != 'kdens':
            raise CheckError('[k_points]: the test steps take the grid from kdens, as the '
                             'workflow code does')
        calcs = []
        for item in self._items(run_dir):
            structure = read_structure(Path(run_dir) / item['structure'])
            kpoints = vasp_kpoints(structure, item['calc_type'], settings.k_points.kdens)
            calcs.append(workflow_calculation(
                system, f'{self.name}/{item["label"]}', item['calc_type'], structure, kpoints,
                with_namelists(settings, item['namelists']), density=item['density']))
        return calcs

    def provenance(self, run_dir, system):
        """Provenance of the entry of the step: its calculations as the settings give them now."""
        plan = read_plan(run_dir)
        return Provenance.build(self.name, {'case': plan['case'], 'system': plan['system']},
                                calculations=self.plan(run_dir, system))

    def done(self, run_dir, system):
        sdir = system_dir(run_dir, system)
        return is_done(read_system(sdir), [self.name], self.provenance(run_dir, system), sdir)

    def expected(self, run_dir, system):
        steps = read_plan(run_dir)['steps']
        return {(dep,): STEP_CLASSES[dep]().provenance(run_dir, system)
                for dep in INTERFACE_GRAPH[self.name] if dep in steps}

    def analyse(self, run_dir, system, calculations):
        sdir = system_dir(run_dir, system)
        records = [read_record(sdir, calc_id) for calc_id in calculations]
        provenance = self.provenance(run_dir, system).with_results(records)
        data = {'case': read_plan(run_dir)['case'],
                'energies': {r.label: r.results['output']['energy'] for r in records}}
        with SystemWriter(sdir) as writer:
            if not writer.document.exists:
                writer.document.create({})
            writer.document.set([self.name], data)
            writer.document.set_provenance([self.name], provenance)
            writer.save()


STEP_CLASSES = {name: type(f'Check_{name}', (CheckStep,), {'name': name})
                for name in INTERFACE_STEPS}


def install():
    """Put the test steps in the registry of the steps of this process."""
    from triboworkflow.workflows.interface import registry

    registry.STEPS.update(STEP_CLASSES)


def daemon_factory(run_dir, groups=None):
    """Factory of the orchestrator in the background, with the test steps.

    As ``triboworkflow.cli.run.daemon_factory``, once the test steps are in
    the registry of the background process.
    """
    from triboworkflow.cli import run as cli_run

    install()
    return cli_run.daemon_factory(run_dir, groups)


def cli(argv):
    """Command ``cli``: ``triboworkflow argv`` with the test steps.

    The directory of this script goes first in ``PYTHONPATH``, so that the
    orchestrator started in the background by ``start`` imports this module
    for its factory (``FACTORY``).
    """
    from triboworkflow.cli import start
    from triboworkflow.cli.main import main as triboworkflow

    install()
    path = os.environ.get('PYTHONPATH')
    os.environ['PYTHONPATH'] = os.pathsep.join([str(HERE), path]) if path else str(HERE)
    start.FACTORY = FACTORY
    return triboworkflow(argv)


# --- snapshot ------------------------------------------------------------------------------

SNAPSHOTS_DIR = Path('qe_check') / 'snapshots'
SACCT_FIELDS = ('JobID', 'State', 'ExitCode', 'Submit', 'Start', 'End', 'Elapsed', 'Timelimit',
                'Restarts', 'NodeList')


def run_job_ids(run_dir):
    """Ids of the jobs submitted for the groups of a run (``submitted.json``), in order."""
    from triboworkflow.core.store import read_json
    from triboworkflow.execution.jobs import SUBMITTED_FILE
    from triboworkflow.execution.layout import JOBS_DIR

    ids = []
    for path in sorted((Path(run_dir) / JOBS_DIR).glob(f'*/{SUBMITTED_FILE}')):
        ids += [i for i in read_json(path).get('job_ids', []) if i not in ids]
    return ids


def current_claims(run_dir):
    """``{calculation directory, relative: claim}`` of the claims on disk; None for an empty one."""
    from triboworkflow.execution.layout import SYSTEMS_DIR

    found = {}
    for path in sorted((Path(run_dir) / SYSTEMS_DIR).glob(f'*/*/*/{CLAIM_FILE}')):
        text = path.read_text(encoding='utf-8')
        found[path.parent.relative_to(run_dir).as_posix()] = json.loads(text) if text.strip() \
            else None
    return found


def _command(argv):
    import subprocess

    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'command': argv, 'returncode': None, 'stdout': '', 'stderr': str(error)}
    return {'command': argv, 'returncode': done.returncode, 'stdout': done.stdout,
            'stderr': done.stderr}


def snapshot(args):
    """Command ``snapshot``: the jobs of the run as Slurm gives them now, and the claims.

    Written in ``qe_check/snapshots/<time>.json``: the output of ``sacct``
    (every run of each job, ``-D``) and of ``squeue`` (state and reason, e.g.
    a hold) for the jobs of the groups, and the claims on disk. Run on the
    HPC machine: while the jobs are queued or held, after a requeue, and once
    they are over.
    """
    import socket

    run_dir = Path(args.run).resolve()
    read_plan(run_dir)
    ids = run_job_ids(run_dir)
    now = datetime.datetime.now()
    data = {'time': now.isoformat(timespec='seconds'), 'host': socket.gethostname(),
            'job_ids': ids, 'sacct': None, 'squeue': None, 'claims': current_claims(run_dir)}
    if ids:
        data['sacct'] = _command(['sacct', '-X', '-D', '-n', '-P', '-o', ','.join(SACCT_FIELDS),
                                  '-j', ','.join(ids)])
        data['squeue'] = _command(['squeue', '-h', '-o', '%i|%T|%r', '-j', ','.join(ids)])
    folder = run_dir / SNAPSHOTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{now.strftime("%Y%m%dT%H%M%S")}.json'
    path.write_text(json.dumps(data, indent=2) + '\n')
    rows = len(sacct_rows(data))
    print(f'Snapshot {path}: {len(ids)} jobs, {rows} rows of sacct, '
          f'{len(squeue_rows(data))} jobs in the queue, {len(data["claims"])} claims')
    for name in ('sacct', 'squeue'):
        result = data[name]
        if result is not None and result['returncode'] not in (0, None) and result['stderr']:
            print(f'  {name}: exit {result["returncode"]}: {result["stderr"].strip()}')
    return 0


def sacct_rows(snap):
    """Rows of ``sacct`` of a snapshot: ``{field: value}``, in the order of ``SACCT_FIELDS``."""
    result = (snap or {}).get('sacct')
    if not result:
        return []
    return [dict(zip(SACCT_FIELDS, line.split('|')))
            for line in result['stdout'].splitlines() if line.count('|') == len(SACCT_FIELDS) - 1]


def squeue_rows(snap):
    """Rows of ``squeue`` of a snapshot: ``{'JobID', 'State', 'Reason'}``."""
    result = (snap or {}).get('squeue')
    if not result:
        return []
    return [dict(zip(('JobID', 'State', 'Reason'), line.split('|', 2)))
            for line in result['stdout'].splitlines() if line.count('|') >= 2]


def read_snapshots(run_dir):
    """Snapshots of a run, oldest first."""
    folder = Path(run_dir) / SNAPSHOTS_DIR
    return [json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))] \
        if folder.is_dir() else []


# --- report: times, logs, symmetries -------------------------------------------------------

_LOG_LINE = re.compile(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ (\w+) (.*)')
_GROUP_FILES = ('manifest.json', 'jobscript.sh', 'submitted.json', 'withdrawn.json')
_SYM = re.compile(r'\s*(\d+) Sym\. Ops\.(, with inversion,| \(no inversion\)) found'
                  r'(?: \(\s*(\d+) have fractional translation\))?')
_SYM_NONE = 'No symmetry found'
_SYM_IGNORED = re.compile(r'\s*\(note:\s*(\d+) additional sym\.ops\. were found')
_RUN_START = 'Program PWSCF v.'
_SCF_RESTARTED = 'Calculation restarted from scf iteration #'
SPGLIB_PRECISIONS = (1e-5, 1e-3)


def local_time(text):
    """A time of the cluster without its offset: ISO of the records, ``sacct``, the logs.

    None for ``Unknown``, ``None`` or an empty value.
    """
    if not text or text in ('Unknown', 'None'):
        return None
    try:
        return datetime.datetime.fromisoformat(text.replace(' ', 'T')).replace(tzinfo=None)
    except ValueError:
        return None


def duration(text):
    """Seconds of a duration of Slurm (``[D-]HH:MM:SS`` or ``MM:SS``); None otherwise."""
    match = re.fullmatch(r'(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)', text or '')
    if not match:
        return None
    days, hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def worker_logs(run_dir):
    """Lines of the logs of the workers in the folders of the groups.

    Returns
    -------
    list of (datetime, str, str or None, str)
        Time, level, job id taken from the name of the file (its first
        number) and message, of every line written by the logging of the
        worker, in the files of each ``jobs/<group>/`` other than those of
        the package (e.g. ``job.<id>.err``).
    """
    from triboworkflow.execution.layout import JOBS_DIR

    lines = []
    for path in sorted((Path(run_dir) / JOBS_DIR).glob('*/*')):
        if path.name in _GROUP_FILES or not path.is_file():
            continue
        found = re.search(r'\d+', path.name)
        job_id = found.group(0) if found else None
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            match = _LOG_LINE.fullmatch(line)
            if match:
                lines.append((local_time(match.group(1)), match.group(2), job_id, match.group(3)))
    return lines


def last_run(text):
    """Text of the last run of a pw.x output."""
    index = text.rfind(_RUN_START)
    return text if index < 0 else text[index:]


def qe_symmetry(text):
    """Symmetry operations that pw.x found in the last run of an output.

    Returns
    -------
    dict or None
        ``nsym``, ``inversion``, ``fractional`` (those with a fractional
        translation) and ``ignored`` (found and left out, their fractional
        translation not on the FFT grid); None if the run prints none of the
        lines of ``summary.f90``.
    """
    run = last_run(text).splitlines()
    for i, line in enumerate(run):
        match = _SYM.fullmatch(line.rstrip())
        none = line.strip() == _SYM_NONE
        if match or none:
            ignored = 0
            for after in run[i + 1:i + 3]:
                note = _SYM_IGNORED.match(after)
                if note:
                    ignored = int(note.group(1))
            if none:
                return {'nsym': 1, 'inversion': False, 'fractional': 0, 'ignored': ignored}
            return {'nsym': int(match.group(1)), 'inversion': 'with' in match.group(2),
                    'fractional': int(match.group(3) or 0), 'ignored': ignored}
    return None


def spglib_symmetry(structure, symprec):
    """Operations and distinct rotations that spglib finds for a structure."""
    import spglib

    cell = (structure.lattice.matrix, structure.frac_coords, [s.specie.Z for s in structure])
    found = spglib.get_symmetry(cell, symprec=symprec)
    if found is None:
        return None
    return len(found['rotations']), len({r.tobytes() for r in found['rotations']})


# --- report ---------------------------------------------------------------------------------

VERDICTS = ('ok', 'note', 'not observed', 'PROBLEM')


class Findings:
    """Verdicts of the report: ``(verdict, subject, text)``, in order."""

    def __init__(self):
        self.items = []

    def add(self, verdict, subject, text):
        assert verdict in VERDICTS
        self.items.append((verdict, subject, text))

    @property
    def failed(self):
        return any(verdict == 'PROBLEM' for verdict, _, _ in self.items)


def _seconds(a, b):
    return None if a is None or b is None else (b - a).total_seconds()


def _show(seconds):
    return '?' if seconds is None else f'{seconds:.0f} s'


class RunReport:
    """What the report reads of a run of the check, and its verdicts."""

    def __init__(self, run_dir):
        from triboworkflow.cli.run import load_run, system_name

        self.run_dir = Path(run_dir).resolve()
        self.plan = read_plan(self.run_dir)
        self.snapshots = read_snapshots(self.run_dir)
        self.jobs = {}
        for row in sacct_rows(self.snapshots[-1] if self.snapshots else None):
            self.jobs.setdefault(row['JobID'], []).append(row)
        self.logs = worker_logs(self.run_dir)
        self.findings = Findings()
        self.settings = load_run(self.run_dir) if self.plan['case'] != 'single' else None
        self.system = system_name(self.settings) if self.settings is not None else None
        self.records = {}
        self.filplots = {}

    def job_row(self, attempt):
        """Row of ``sacct`` of the run of the job in which an attempt started."""
        rows = self.jobs.get(attempt.job_id or '', [])
        started = local_time(attempt.started)
        for row in rows:
            start, end = local_time(row['Start']), local_time(row['End'])
            if start is not None and started is not None and start <= started \
                    and (end is None or started <= end):
                return row
        return rows[-1] if rows else None

    def log_after(self, job_id, since, text):
        """First line of the log of a job after ``since`` whose message contains ``text``."""
        for time_, _, job, message in self.logs:
            if job == job_id and (since is None or time_ >= since) and text in message:
                return time_, message
        return None


def _status(status):
    return (f'{status.process}/{status.scf}/{status.geometry} parsing {status.parsing} '
            f'density {status.density}')


def _launch(remedies):
    if not remedies:
        return 'launch not recorded'
    text = f'{remedies.get("kind")} level {remedies.get("level")}'
    if remedies.get('values'):
        text += f' {remedies["values"]}'
    return text + (', restart' if remedies.get('restart') else ', from scratch')


def report_attempt(rep, path, record, attempt, previous):
    """Print one attempt; consistency problems go to the findings."""
    from triboworkflow.core.store import file_digest
    from triboworkflow.engines.qe.status import PwOutputError, read_parallelization, read_pw_status

    subject = f'{record.calc_id} attempt {attempt.n}'
    row = rep.job_row(attempt) if attempt.job_id else None
    job = f'job {attempt.job_id}'
    if row:
        job += (f' [{row["State"]}, exit {row["ExitCode"]}, restarts {row["Restarts"]}, '
                f'{row["Start"]} -> {row["End"]}, limit {row["Timelimit"]}, {row["NodeList"]}]')
    started, ended = local_time(attempt.started), local_time(attempt.ended)
    print(f'  attempt {attempt.n}: {job} on {attempt.host}')
    print(f'    pw.x {attempt.started} -> {attempt.ended} ({_show(_seconds(started, ended))}), '
          f'max_seconds {attempt.max_seconds} (cap {attempt.max_seconds_cap}), exit '
          f'{attempt.exit_code}, stop {attempt.stop_reason} (closed {attempt.clean_stop})')
    print(f'    {_status(attempt.status)}; {_launch(attempt.remedies)}; outdir removal '
          f'{attempt.outdir_removal}; job state {attempt.job_state}')
    if previous is not None and attempt.run_command != previous.run_command:
        print(f'    run_command differs from attempt {previous.n}: {attempt.run_command!r}')
    if attempt.restart is not None:
        print('    restart: ' + ', '.join(f'{k} {v}' for k, v in attempt.restart.items()))
    if attempt.parse_error:
        print(f'    parse error: {attempt.parse_error}')
    output = None if attempt.output is None else path / attempt.output.path
    if output is not None and output.is_file():
        if attempt.output.digest is not None and file_digest(output) != attempt.output.digest:
            rep.findings.add('PROBLEM', subject, f'{output} differs from the digest recorded')
        try:
            pw = read_pw_status(output, 'relax' if record.calc_type != 'slab_from_scratch'
                                else 'scf')
            lines = read_parallelization(output)
        except PwOutputError as error:
            print(f'    output not read: {error}')
        else:
            print(f'    output: pw.x {pw.version}, stop {pw.stop_reason}; ' + '; '.join(lines))
            if attempt.stop_reason is not None and pw.stop_reason != attempt.stop_reason:
                rep.findings.add('PROBLEM', subject, f'stop {attempt.stop_reason} recorded, '
                                 f'{pw.stop_reason} in the output')
    if row is not None and ended is not None:
        job_end = local_time(row['End'])
        done = rep.log_after(attempt.job_id, ended, 'Worker done')
        limit = duration(row['Timelimit'])
        left = None
        if job_end is not None and limit is not None and local_time(row['Start']) is not None:
            left = limit - _seconds(local_time(row['Start']), job_end)
        print(f'    after pw.x: worker done +{_show(_seconds(ended, done[0] if done else None))}, '
              f'job end +{_show(_seconds(ended, job_end))}, {_show(left)} before the time limit')
    if attempt.density is not None:
        d = attempt.density
        text = f'{record.system}/{record.calc_id}: charge density of attempt {attempt.n}'
        begun = rep.log_after(attempt.job_id, ended, f'{text}:')
        over = rep.log_after(attempt.job_id, ended, f'{text} ')
        print(f'    pp.x: exit {d.exit_code}, error {d.error}; pp.x and reading of the filplot '
              f'{_show(_seconds(begun[0] if begun else None, over[0] if over else None))}')


def report_calc(rep, step, item):
    """Print one calculation and check what the plan expects of it."""
    from triboworkflow.core.store import read_record as read

    calc_id = f'{step}/{item["label"]}'
    path = calc_dir(rep.run_dir, rep.system, calc_id)
    print(f'{calc_id}: {item["purpose"]}')
    try:
        record = read(system_dir(rep.run_dir, rep.system), calc_id)
    except FileNotFoundError:
        print('  not prepared')
        rep.findings.add('not observed', calc_id, 'not prepared')
        return
    rep.records[calc_id] = record
    attempts = record.attempts
    print(f'  {len(attempts)} attempts; record {record.status.summary}: {_status(record.status)}')
    if [a.n for a in attempts] != list(range(1, len(attempts) + 1)):
        rep.findings.add('PROBLEM', calc_id, f'attempts numbered {[a.n for a in attempts]}')
    if attempts and record.status != attempts[-1].status:
        rep.findings.add('PROBLEM', calc_id, f'record {_status(record.status)}, last attempt '
                         f'{_status(attempts[-1].status)}')
    claim = path / CLAIM_FILE
    if claim.is_file():
        text = claim.read_text(encoding='utf-8')
        print(f'  claim: {text.strip() or "empty"}')
    for i, attempt in enumerate(attempts):
        report_attempt(rep, path, record, attempt, attempts[i - 1] if i else None)
    for name in item['expect']:
        kind, _, arg = name.partition(':')
        EXPECTATIONS[kind](rep, path, record, item, arg)


# --- expectations of a calculation ------------------------------------------------------------

def _completed(rep, path, record, item, arg):
    from triboworkflow.core.store import identity_differences, unknown_parts

    subject = f'{record.calc_id} completed'
    if record.status.summary != 'completed':
        rep.findings.add('not observed', subject, f'record {record.status.summary}')
        return
    problems = []
    if record.status.parsing != 'ok' or record.results is None:
        problems.append(f'parsing {record.status.parsing}')
    problems += identity_differences(record)
    problems += [f'{part} not known' for part in unknown_parts(record.energy_model)]
    if not (path / 'structure_out.extxyz').is_file():
        problems.append('structure_out.extxyz missing')
    if record.observed:
        species = ', '.join(f'{s["label"]} {s["pseudopotential"]} md5 {s["md5"]}'
                            for s in record.observed.get('species', []))
        print(f'  observed: version {record.observed.get("version")}; {species}')
    rep.findings.add('PROBLEM' if problems else 'ok', subject,
                     '; '.join(problems) or f'parsing ok, observed as expected, '
                     f'{len(record.attempts)} attempts')


def _stress(rep, path, record, item, arg):
    subject = f'{record.calc_id} stress'
    if record.results is None:
        rep.findings.add('not observed', subject, 'no results')
    elif record.results['output'].get('stress') is None:
        rep.findings.add('PROBLEM', subject, 'tstress = true, no stress in the results')
    else:
        rep.findings.add('ok', subject, f'stress {record.results["output"]["stress"]} kbar')


def _z_only(rep, path, record, item, arg):
    import numpy as np
    from pymatgen.core import Structure

    subject = f'{record.calc_id} z only'
    if record.results is None:
        rep.findings.add('not observed', subject, 'no results')
        return
    before = Structure.from_dict(record.results['input']['structure']).cart_coords
    after = Structure.from_dict(record.results['output']['structure']).cart_coords
    xy, z = np.abs(after - before)[:, :2].max(), np.abs(after - before)[:, 2].max()
    ok = xy < 1e-6 and z > 1e-4
    rep.findings.add('ok' if ok else 'PROBLEM', subject,
                     f'largest move along x, y {xy:.2e} A, along z {z:.4f} A')


def _symmetry(rep, path, record, item, arg):
    from triboworkflow.core.structures import read_structure as read

    subject = f'{record.calc_id} symmetry (Q17)'
    first = record.attempts[0] if record.attempts else None
    output = None if first is None or first.output is None else path / first.output.path
    if output is None or not output.is_file():
        rep.findings.add('not observed', subject, 'no output of attempt 1')
        return
    qe = qe_symmetry(output.read_text(encoding='utf-8', errors='replace'))
    structure = read(path / 'structure_in.extxyz')
    found = {p: spglib_symmetry(structure, p) for p in SPGLIB_PRECISIONS}
    shown = '; '.join(f'symprec {p:g} A: {n[0]} operations, {n[1]} rotations' if n else
                      f'symprec {p:g} A: none' for p, n in found.items())
    if qe is None:
        rep.findings.add('PROBLEM', subject, f'no line of the symmetries in {output}')
        return
    total = qe['nsym'] + qe['ignored']
    reference = found[SPGLIB_PRECISIONS[0]]
    text = (f'pw.x {qe["nsym"]} operations ({qe["fractional"]} with fractional translation, '
            f'{qe["ignored"]} left out, inversion {qe["inversion"]}); spglib {shown}')
    rep.findings.add('ok' if reference and total == reference[0] else 'note', subject, text)


def _attempt_after(record, reason):
    for i, attempt in enumerate(record.attempts[:-1]):
        if attempt.stop_reason == reason:
            return attempt, record.attempts[i + 1]
    return None, None


def _restart_after(rep, path, record, item, reason):
    from triboworkflow.core.store.record import PER_PROCESS_STOPS

    subject = f'{record.calc_id} restart after {reason}'
    stopped, after = _attempt_after(record, reason)
    if stopped is None:
        stops = [a.stop_reason for a in record.attempts]
        rep.findings.add('not observed', subject, f'no attempt stopped at {reason} with a '
                         f'later one; stops {stops}')
        return
    if not after.remedies.get('restart'):
        # With the same command and jobscript the policy continues the attempt,
        # but an scf stopped with scf_not_converged, which starts from scratch.
        same = (after.run_command == stopped.run_command
                and after.jobscript_digest == stopped.jobscript_digest)
        scf_from_scratch = reason == 'scf_not_converged' and record.calc_type == 'slab_from_scratch'
        verdict = 'PROBLEM' if same and not scf_from_scratch else 'not observed'
        rep.findings.add(verdict, subject, f'attempt {after.n} from scratch'
                         + (' with the same command and jobscript' if same else
                            ', with another command or jobscript'))
        return
    if after.restart is None:
        rep.findings.add('not observed', subject, f'attempt {after.n} has no restart outcome '
                         f'({after.status.summary})')
        return
    missing = [p for p in ('positions', 'density', 'wavefunctions') if after.restart[p] is not True]
    if reason in PER_PROCESS_STOPS and after.restart['same_parallelization'] is not True:
        missing.append(f'same_parallelization {after.restart["same_parallelization"]}')
    rep.findings.add('PROBLEM' if missing else 'ok', subject,
                     f'attempt {after.n} {_launch(after.remedies)}: '
                     + (f'not read: {", ".join(missing)}' if missing else
                        'positions, density and wavefunctions read, same parallelization')
                     + f'; {after.restart}')


def _scf_iteration(rep, path, record, item, arg):
    subject = f'{record.calc_id} SCF restarted from its iteration'
    stopped, after = _attempt_after(record, 'max_seconds')
    if stopped is None or after.restart is None:
        rep.findings.add('not observed', subject, 'no restart after max_seconds with an outcome')
        return
    output = None if after.output is None else path / after.output.path
    line = output is not None and output.is_file() and _SCF_RESTARTED in last_run(
        output.read_text(encoding='utf-8', errors='replace'))
    verdict = 'ok' if after.restart['scf_iteration'] is True and line else 'not observed'
    rep.findings.add(verdict, subject, f'scf_iteration {after.restart["scf_iteration"]}, line '
                     f'"{_SCF_RESTARTED}" {"found" if line else "not found"} in attempt {after.n}')


def _level(rep, path, record, item, level):
    subject = f'{record.calc_id} level {level}'
    stopped, after = _attempt_after(record, 'max_steps')
    if stopped is None:
        rep.findings.add('not observed', subject, 'no attempt stopped at max_steps')
        return
    ok = after.remedies.get('level') == int(level) and after.remedies.get('kind') == 'remedy'
    rep.findings.add('ok' if ok else 'PROBLEM', subject,
                     f'attempt {after.n}: {_launch(after.remedies)}')


def _margin(rep, path, record, item, arg):
    subject = f'{record.calc_id} margin at max_seconds (Q09)'
    stops = [a for a in record.attempts if a.stop_reason == 'max_seconds'
             and (a.max_seconds_cap is None or a.max_seconds_cap > a.max_seconds)]
    if not stops:
        rep.findings.add('not observed', subject, 'no stop at a max_seconds set by the walltime')
        return
    for attempt in stops:
        row = rep.job_row(attempt)
        if row is None:
            rep.findings.add('not observed', subject, f'attempt {attempt.n}: job not in sacct '
                             '(take a snapshot)')
            continue
        start, end = local_time(row['Start']), local_time(row['End'])
        limit = duration(row['Timelimit'])
        left = None if None in (start, end, limit) else limit - _seconds(start, end)
        after = _seconds(local_time(attempt.ended), end)
        bad = row['State'].startswith('TIMEOUT') or (left is not None and left < 0)
        rep.findings.add('PROBLEM' if bad else 'ok', subject,
                         f'attempt {attempt.n}: job {row["State"]}, pw.x stopped '
                         f'{_show(after)} before the end of the job, {_show(left)} before the '
                         f'time limit')


def _scratch_after(rep, path, record, item, state):
    subject = f'{record.calc_id} from scratch after {state}'
    for i, attempt in enumerate(record.attempts[:-1]):
        if attempt.status.process == state or attempt.job_state == state:
            after = record.attempts[i + 1]
            ok = not after.remedies.get('restart') and after.outdir_removal == 'removed'
            rep.findings.add('ok' if ok else 'PROBLEM', subject,
                             f'attempt {after.n}: {_launch(after.remedies)}, outdir removal '
                             f'{after.outdir_removal}')
            return
    rep.findings.add('not observed', subject, f'no attempt {state} with a later one')


def _manual_after(rep, path, record, item, reason):
    subject = f'{record.calc_id} relaunch by hand after {reason}'
    stopped, after = _attempt_after(record, reason)
    if stopped is None:
        rep.findings.add('not observed', subject, f'no attempt stopped by {reason} with a '
                         'later one')
        return
    ok = after.remedies.get('kind') == 'manual'
    rep.findings.add('ok' if ok else 'PROBLEM', subject,
                     f'attempt {after.n}: {_launch(after.remedies)}')


def _empty_claim_released(rep, path, record, item, arg):
    subject = f'{record.calc_id} empty claim released'
    marker = rep.run_dir / KILLED_DIR / item['label']
    if not marker.is_file():
        rep.findings.add('not observed', subject, 'the worker was not stopped at the claim')
        return
    print(f'  worker stopped at the claim: {marker.read_text().strip()}')
    if not record.attempts:
        rep.findings.add('not observed', subject, 'claim not released yet')
        return
    first = record.attempts[0]
    released = first.started is None and first.exit_code is None and first.status.process is None
    later = record.attempts[1] if len(record.attempts) > 1 else None
    problems = [] if released else [f'attempt 1 {_status(first.status)}, started {first.started}']
    if later is not None and later.remedies.get('kind') != 'manual':
        problems.append(f'attempt 2: {_launch(later.remedies)}')
    rep.findings.add('PROBLEM' if problems else 'ok', subject,
                     '; '.join(problems) or 'attempt 1 archived with nothing known'
                     + (', then relaunched by hand' if later else ', not relaunched yet'))


def _density(rep, path, record, item, arg):
    import time

    from triboworkflow.core.store import file_digest
    from triboworkflow.core.structures import read_structure as read
    from triboworkflow.engines.qe import density

    subject = f'{record.calc_id} charge density'
    last = record.attempts[-1] if record.attempts else None
    if last is None or last.density is None:
        rep.findings.add('not observed', subject, 'no post-processing recorded')
        return
    d = last.density
    problems = [] if record.status.density == 'ok' and d.exit_code == 0 and d.error is None \
        else [f'density {record.status.density}, pp.x exit {d.exit_code}, {d.error}']
    filplot = None if d.density is None else path / d.density.path
    if filplot is None or not filplot.is_file():
        rep.findings.add('PROBLEM', subject, '; '.join(problems + ['no filplot']))
        return
    if d.density.digest is not None and file_digest(filplot) != d.density.digest:
        problems.append('filplot differs from the digest recorded')
    begin = time.perf_counter()
    try:
        f = density.read_filplot(filplot)
        read_time = time.perf_counter() - begin
        chgcar = density.to_chgcar(f, read(path / 'structure_out.extxyz'))
    except (density.DensityError, OSError, ValueError) as error:
        rep.findings.add('PROBLEM', subject, '; '.join(problems + [str(error)]))
        return
    electrons = float(chgcar.data['total'].mean())
    analyzer = qe_analyzer_electrons(filplot)
    if analyzer is not None and abs(analyzer - electrons) > 1e-5 * abs(electrons):
        problems.append(f'QEChargeAnalyzer gives {analyzer} electrons, the Chgcar {electrons}')
    rep.filplots[record.calc_id] = f
    rep.findings.add('PROBLEM' if problems else 'ok', subject,
                     '; '.join(problems) or f'grid {f.grid}, {filplot.stat().st_size / 1e6:.1f} MB, '
                     f'read here in {read_time:.1f} s; {electrons:.6f} electrons (Chgcar), '
                     f'{analyzer} (QEChargeAnalyzer)')


def qe_analyzer_electrons(filplot):
    """Electrons of a filplot according to the vendored ``QEChargeAnalyzer`` (cell average)."""
    import tempfile

    from triboworkflow.vendored.physics.electronics.charge.analyzer import QEChargeAnalyzer

    with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp), \
            contextlib.redirect_stdout(io.StringIO()):
        analyzer = QEChargeAnalyzer(str(filplot), unpack=False)
        analyzer.average_charge('xyz', to_file=False, to_plot=False)
    return float(analyzer.charge['cell']['rho_tot'])


def _nowf(rep, path, record, item, arg):
    subject = f'{record.calc_id} disk_io nowf'
    text = (path / 'input.pwi').read_text(encoding='utf-8')
    ok = "disk_io          = 'nowf'" in text and not record.density
    rep.findings.add('ok' if ok else 'PROBLEM', subject,
                     "input with disk_io = 'nowf', no charge density" if ok else
                     'the input or the record is not the one of the case')


EXPECTATIONS = {'completed': _completed, 'stress': _stress, 'z_only': _z_only,
                'symmetry': _symmetry, 'restart_after': _restart_after,
                'scf_iteration': _scf_iteration, 'level': _level, 'margin': _margin,
                'scratch_after': _scratch_after, 'manual_after': _manual_after,
                'empty_claim_released': _empty_claim_released, 'density': _density,
                'nowf': _nowf}


# --- expectations of a run -----------------------------------------------------------------

def _steps_completed(rep):
    from triboworkflow.core.store import read_json

    path = rep.run_dir / 'orchestrator.json'
    if not path.is_file():
        rep.findings.add('not observed', 'steps', 'no orchestrator.json')
        return
    steps = read_json(path)['systems'].get(rep.system, {})
    states = {name: entry['status'] for name, entry in steps.items()}
    done = {name: STEP_CLASSES[name]().done(rep.run_dir, rep.system) for name in steps}
    bad = [n for n, s in states.items() if s in ('failed', 'blocked')]
    if bad:
        rep.findings.add('PROBLEM', 'steps', f'{states}')
    elif all(s == 'completed' for s in states.values()) and all(done.values()):
        rep.findings.add('ok', 'steps', f'{states}; entries of system.json done and current')
    else:
        rep.findings.add('not observed', 'steps', f'{states}; entries done: {done}')


def _other_command(rep):
    for calc_id, record in rep.records.items():
        stopped, after = _attempt_after(record, 'max_seconds')
        if stopped is not None and not after.remedies.get('restart') \
                and after.run_command != stopped.run_command:
            ok = after.outdir_removal == 'removed'
            rep.findings.add('ok' if ok else 'PROBLEM', f'{calc_id} other run_command',
                             f'attempt {after.n} from scratch after max_seconds with '
                             f'{after.run_command!r} (was {stopped.run_command!r}), outdir '
                             f'removal {after.outdir_removal}')
            return
    rep.findings.add('not observed', 'other run_command', 'no relaunch from scratch after '
                     'max_seconds with another run_command')


def _attempts_of_job(rep, job_id):
    return [(calc_id, a) for calc_id, r in rep.records.items() for a in r.attempts
            if a.job_id == job_id]


def _held(rep):
    seen = {}
    for snap in rep.snapshots:
        for row in squeue_rows(snap):
            if 'held' in row['Reason'].lower():
                seen[row['JobID']] = local_time(snap['time'])
    if not seen:
        rep.findings.add('not observed', 'held job', 'no snapshot with a held job')
        return
    for job_id, last in seen.items():
        early = [f'{c} attempt {a.n}' for c, a in _attempts_of_job(rep, job_id)
                 if local_time(a.started) is not None and local_time(a.started) <= last]
        later = [f'{c} attempt {a.n}' for c, a in _attempts_of_job(rep, job_id)]
        rep.findings.add('PROBLEM' if early else 'ok', f'held job {job_id}',
                         f'attempts started while held: {early}' if early else
                         f'held until at least {last}, no attempt while held; later: {later}')


def _requeued(rep):
    found = False
    for job_id, rows in rep.jobs.items():
        restarts = max((int(r['Restarts']) for r in rows if r['Restarts'].isdigit()), default=0)
        if restarts < 1 and len(rows) < 2:
            continue
        found = True
        claims = sorted({json.dumps(c.get('restart_count')) for s in rep.snapshots
                         for c in s['claims'].values() if c and c.get('job_id') == job_id})
        attempts = [f'{c} attempt {a.n} {a.status.summary}' for c, a in _attempts_of_job(rep,
                                                                                          job_id)]
        rep.findings.add('ok', f'requeued job {job_id}',
                         f'sacct: restarts {restarts}, {len(rows)} runs; restart_count in the '
                         f'claims of the snapshots: {claims or "none seen"}; attempts {attempts}')
    if not found:
        rep.findings.add('not observed', 'requeued job', 'no job with restarts in sacct '
                         '(take a snapshot after the requeue)')


def _comparable(rep):
    from triboworkflow.engines.qe import density

    names = [f'charge/{label}' for label in ('interface', 'bottom', 'top')]
    if not all(name in rep.filplots for name in names):
        rep.findings.add('not observed', 'densities comparable', 'filplots of interface and '
                         'fragments not all read')
        return
    try:
        density.check_comparable({n: rep.filplots[n] for n in names},
                                 [rep.records[n] for n in names])
    except density.DensityError as error:
        rep.findings.add('PROBLEM', 'densities comparable', str(error))
    else:
        rep.findings.add('ok', 'densities comparable', 'same grid, cell, cutoffs and energy model')


RUN_EXPECTATIONS = {'steps_completed': _steps_completed, 'other_command': _other_command,
                    'held': _held, 'requeued': _requeued, 'comparable': _comparable}


def report_single(rep):
    """The case single: the output of the input of ``gen`` run by hand."""
    from triboworkflow.core.settings import load_settings
    from triboworkflow.core.structures import read_structure as read
    from triboworkflow.engines.qe.model import energy_model
    from triboworkflow.engines.qe.results import read_pw_values
    from triboworkflow.engines.qe.status import PwOutputError, read_pw_status

    output = rep.run_dir / 'output.pwo'
    if not output.is_file():
        rep.findings.add('not observed', 'single', f'{output} not found')
        return
    settings = load_settings(rep.run_dir / CONFIG_FILE)
    try:
        pw = read_pw_status(output, 'scf')
        values = read_pw_values(output, 'scf')
    except PwOutputError as error:
        rep.findings.add('PROBLEM', 'single', str(error))
        return
    expected = energy_model(settings)['species']
    printed = [(s.label, s.pseudopotential.rsplit('/', 1)[-1], s.md5) for s in values.species]
    print(f'single: pw.x {pw.version}, stop {pw.stop_reason}, process {pw.process}, scf {pw.scf}')
    print('  species: ' + ', '.join(f'{label} {upf} md5 {md5}' for label, upf, md5 in printed))
    differ = [f'{e["label"]}: expected md5 {e["md5"]}, printed {p[2]}'
              for e, p in zip(expected, printed) if e['md5'] != p[2]]
    if len(printed) != len(expected):
        differ.append(f'{len(printed)} species printed, {len(expected)} in the settings')
    ok = pw.stop_reason == 'completed' and not differ
    rep.findings.add('ok' if ok else 'PROBLEM', 'single', '; '.join(differ) or
                     f'completed, {len(printed)} species (one without atoms), md5 as the manifest')
    qe = qe_symmetry(output.read_text(encoding='utf-8', errors='replace'))
    found = spglib_symmetry(read(rep.run_dir / 'input.extxyz'), SPGLIB_PRECISIONS[0])
    rep.findings.add('ok' if qe and found and qe['nsym'] + qe['ignored'] == found[0] else 'note',
                     'single symmetry (Q17)', f'pw.x {qe}; spglib {found}')


def report(args):
    """Command ``report``: what the files of a run of the check tell, with verdicts.

    Read only. ``qe_check/snapshots/`` gives the jobs as Slurm saw them; the
    logs of the workers are read in the folders of the groups. Exit status 1
    when a verdict is ``PROBLEM``.
    """
    rep = RunReport(args.run)
    plan = rep.plan
    print(f'Run {rep.run_dir}: case {plan["case"]}, system {plan["system"]}, written '
          f'{plan["written"]}; {len(rep.snapshots)} snapshots')
    if plan['case'] == 'single':
        report_single(rep)
    else:
        for step, items in plan['steps'].items():
            for item in items:
                report_calc(rep, step, item)
        for name in plan['run_expect']:
            RUN_EXPECTATIONS[name](rep)
    print('\nVerdicts:')
    width = max((len(subject) for _, subject, _ in rep.findings.items), default=0)
    for verdict, subject, text in rep.findings.items:
        print(f'  {verdict:<12} {subject:<{width}}  {text}')
    counts = {v: sum(1 for item in rep.findings.items if item[0] == v) for v in VERDICTS}
    print('RESULT: ' + ', '.join(f'{n} {v}' for v, n in counts.items()))
    return 1 if rep.findings.failed else 0


# --- command line ------------------------------------------------------------------------

def main(argv=None):
    import argparse

    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['cli']:
        return cli(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('cli', help='triboworkflow ARGS... with the test steps (all the arguments '
                   'after cli go to triboworkflow)')
    p = sub.add_parser('write', help='write the folder of a new run of one case')
    p.add_argument('machine', help='TOML file of the machine (tests/hpc/machine_example.toml)')
    p.add_argument('dir', help='new directory of the run, on the filesystem of the runs')
    p.add_argument('--case', required=True, choices=CASES)
    p.add_argument('--system', required=True, choices=SYSTEMS)
    p.add_argument('--groups', type=int, default=None,
                   help='jobs per group; else [cases.<case>].groups, else the default of the '
                   'case')
    p = sub.add_parser('snapshot', help='save sacct, squeue and the claims of a run (on the '
                       'HPC machine)')
    p.add_argument('run')
    p = sub.add_parser('report', help='what the files of a run tell, with verdicts')
    p.add_argument('run')
    args = parser.parse_args(argv)
    try:
        return {'write': write, 'snapshot': snapshot, 'report': report}[args.command](args)
    except (CheckError, OSError, ValueError) as error:  # ValueError: settings, inputs, structures
        print(f'Error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
