# Author: Vito Foderà
# Created: 2026-11-09

"""Report on a run of the interface workflow made with the command line: read only.

The run is prepared and launched with the ``triboworkflow`` commands, on the HPC
machine, and its folder is copied where this script runs (the machine is not needed):

    python interface_check.py report RUN [--sacct FILE]

``RUN`` is the folder of the run (``config.toml``, ``orchestrator.json``,
``systems/``, ``jobs/``). Nothing is written, in ``RUN`` or elsewhere, and nothing is
run: no program, no scheduler, no calculation. ``FILE`` is the output of
``sacct -P -j <ids> -o JobID,JobName,State,ExitCode,Elapsed,Timelimit,ReqMem,MaxRSS,NodeList``,
saved by whoever has the machine at hand; without it the memory and the walltime of the
jobs are ``not observed``. The protocol is in ``docs/hpc_check.md``.

For each topic the report reads the files and gives a verdict, as ``qe_check.py report``
does: ``ok``; ``note``, a difference to look at; ``not observed``, what the run did not
show (a step not run yet, a file not copied); ``PROBLEM``, an inconsistency of the files or
a behaviour other than the one the package gives. The exit status is 1 when a verdict is
``PROBLEM``.

Topics, one verdict each unless said otherwise:

``steps``
    The steps of ``[interface]``: state in ``orchestrator.json`` and entries of
    ``system.json`` done and current.
``calculations <step>``
    For each step with calculations: complete, parsed, observed (version, pseudopotentials) as
    expected, one energy model for the step, final structure written, charge density ``ok``.
``spin``
    ``nspin`` and ``starting_magnetization(i)`` of the inputs against ``[[atomic_species]]``.
``slabs``
    The slab files as the workflow reads them: path, rotation and data dropped.
``pes``, ``pes_scf``
    The keys and units of the entries; the energy of each point against the energy of its
    output in Ry; the order and the sites.
``plotpes``, ``plotpes_scf``
    The entries and the figure; the job of the interpolation: group, outcome, times, memory.
``adhesion``
    The final structures (slabs, ``interface_region`` of the interfaces) and the energies
    recomputed from the entries; the k-point grids.
``ppes``
    The slab on top moves along z by the shift, the bottom one does not.
``charge``
    The fragments take the atoms of their slab; the three densities are on one grid and one
    cell; time and memory of reading them; the region between the slabs.
``shear``
    The job (time, memory, log, warnings of ``DeprecationWarning``), the time of the
    interpolation and of each string, whether each string converged, the cell of the
    interface against the cell of the PES.
``accounting``
    The states of the jobs in the file of ``sacct``.
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

try:
    import resource
except ImportError:  # Windows: the peak memory of the process is then not reported
    resource = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from qe_check import VERDICTS, Findings, duration, local_time  # noqa: E402

PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
# Keys of the results of a calculation in the entries of the PES.
OUTPUT_KEYS = {'nsites', 'density', 'energy', 'energy_per_atom', 'bandgap', 'cbm', 'vbm',
               'is_gap_direct', 'forces', 'stress', 'pseudo_potential', 'dir_name', 'task_id',
               'task_label', 'structure'}
ANALYSES = ('plotpes', 'plotpes_scf', 'shear')
PES_OF = {'plotpes': 'pes', 'plotpes_scf': 'pes_scf'}
SUCCESS = ('analysed', 'done')
# Tolerances: the extxyz of an input has eight decimals; a cell compared with a cell of the
# entries is the same to rounding.
POSITION_TOL = 1e-5
CELL_OK, CELL_NOTE = 1e-6, 1e-3

_INTERPOLATION = re.compile(r'interpolation of the PES: ([\d.]+) s, (\d+) centres')
_STRING = re.compile(r'string (\w+): (\d+) steps, ([\d.]+) s')
_SUMMARY = re.compile(r'\S+: PES of (\d+) points, (\d+) centres')
_MEMORY = {'K': 1024, 'M': 1024 ** 2, 'G': 1024 ** 3, 'T': 1024 ** 4}


def dig(data, *path):
    """The value at a path of nested dicts, or None where a key is missing."""
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _gib(nbytes):
    return f'{nbytes / 1024 ** 3:.2f} GiB'


def memory_bytes(text):
    """Bytes of a memory of ``sacct`` (``4821964K``, ``12G``, ``8000Mn``); 0 if not given."""
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT]?)[nc]?', (text or '').strip())
    if not match:
        return 0
    return int(float(match.group(1)) * _MEMORY.get(match.group(2), 1))


def read_sacct(path):
    """Jobs of the output of ``sacct -P``, by job id.

    Returns
    -------
    dict
        ``{job id: {'State', 'Elapsed', 'Timelimit', 'ReqMem', 'NodeList', 'JobName', 'MaxRSS'}}``
        with ``MaxRSS`` the largest of the job and its steps, in bytes.
    """
    lines = Path(path).read_text(encoding='utf-8', errors='replace').splitlines()
    if not lines:
        return {}
    header = lines[0].split('|')
    jobs = {}
    for line in lines[1:]:
        if line.count('|') != len(header) - 1:
            continue
        row = dict(zip(header, line.split('|')))
        job_id = row['JobID'].split('.')[0]
        entry = jobs.setdefault(job_id, {'State': None, 'Elapsed': None, 'Timelimit': None,
                                         'ReqMem': None, 'NodeList': None, 'JobName': None,
                                         'MaxRSS': 0})
        if '.' not in row['JobID']:
            entry.update({key: row.get(key) for key in
                          ('State', 'Elapsed', 'Timelimit', 'ReqMem', 'NodeList', 'JobName')})
        entry['MaxRSS'] = max(entry['MaxRSS'], memory_bytes(row.get('MaxRSS')))
    return jobs


def peak_rss():
    """Peak resident memory of this process in bytes, or None where it is not known."""
    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak * 1024 if sys.platform.startswith('linux') else (peak or None)


class InterfaceReport:
    """What the report reads of a run, and its verdicts."""

    def __init__(self, run_dir, sacct=None):
        from triboworkflow.cli.run import load_run, system_name
        from triboworkflow.core.store import read_json, read_system
        from triboworkflow.execution.layout import INPUT_FILE, system_dir

        self.run_dir = Path(run_dir).resolve()
        self.settings = load_run(self.run_dir)
        self.interface = self.settings.interface
        self.system = system_name(self.settings)
        self.sdir = system_dir(self.run_dir, self.system)
        self.steps = list(self.interface.steps)
        self.doc = read_system(self.sdir).document
        state = self.run_dir / 'orchestrator.json'
        self.orchestrator = read_json(state)['systems'].get(self.system, {}) \
            if state.is_file() else None
        self.sacct = read_sacct(sacct) if sacct else {}
        self.findings = Findings()
        self.calc_ids = sorted(path.parent.relative_to(self.sdir).as_posix()
                               for path in self.sdir.glob(f'*/*/{INPUT_FILE}'))
        self.records = {}
        self._load_records()

    def _load_records(self):
        from triboworkflow.core.store import read_record

        for calc_id in self.calc_ids:
            try:
                self.records[calc_id] = read_record(self.sdir, calc_id)
            except (OSError, ValueError) as error:
                self.findings.add('PROBLEM', calc_id, f'record not read: {error}')

    def path(self, calc_id):
        return self.sdir / calc_id

    def records_of(self, step):
        return {calc_id: record for calc_id, record in self.records.items()
                if calc_id.startswith(f'{step}/')}

    def entry(self, step):
        """State of a step in ``orchestrator.json``, or None."""
        return None if self.orchestrator is None else self.orchestrator.get(step)

    def group_dir(self, step):
        """Folder of the group of the analysis of a step in a job of its own, or None."""
        entry = self.entry(step) or {}
        return None if not entry.get('job') else self.run_dir / 'jobs' / entry['job']

    def job_ids(self, group):
        path = group / 'submitted.json'
        return json.loads(path.read_text())['job_ids'] if path.is_file() else []

    def job_log(self, group):
        """Text of the logs ``job.<id>.err`` and ``job.<id>.out`` of a group."""
        return '\n'.join(path.read_text(encoding='utf-8', errors='replace')
                         for path in sorted(group.glob('job.*')) if path.is_file())


# --- the run -------------------------------------------------------------------------------

def check_steps(rep):
    from triboworkflow.workflows.interface.registry import get_step

    if rep.orchestrator is None:
        rep.findings.add('not observed', 'steps', 'no orchestrator.json: the run never started')
        return
    states = {step: (rep.entry(step) or {}).get('status') for step in rep.steps}
    done = {}
    for step in rep.steps:
        try:
            done[step] = get_step(step).done(rep.run_dir, rep.system)
        except Exception as error:  # reading the files of a copy that lacks some
            done[step] = f'{type(error).__name__}: {error}'
    bad = [step for step, state in states.items() if state in ('failed', 'blocked')]
    reasons = {step: (rep.entry(step) or {}).get('reason') for step in bad}
    if bad:
        rep.findings.add('PROBLEM', 'steps', f'{states}; {reasons}')
    elif all(state == 'completed' for state in states.values()) \
            and all(value is True for value in done.values()):
        rep.findings.add('ok', 'steps', f'{states}; entries of system.json done and current')
    else:
        rep.findings.add('not observed', 'steps', f'{states}; entries done: {done}')


def check_calculations(rep):
    from triboworkflow.core.store import identity_differences, unknown_parts

    by_step = {}
    for calc_id, record in rep.records.items():
        by_step.setdefault(record.step, {})[calc_id] = record
    for step, records in by_step.items():
        subject = f'{step} calculations'
        complete = {c: r for c, r in records.items() if r.is_complete}
        if len(complete) < len(records):
            waiting = sorted(set(records) - set(complete))
            rep.findings.add('not observed', subject, f'{len(complete)} of {len(records)} '
                             f'complete; not complete: {waiting}')
            continue
        problems = []
        for calc_id, record in records.items():
            if record.status.parsing != 'ok' or record.results is None:
                problems.append(f'{calc_id}: parsing {record.status.parsing}')
            problems += [f'{calc_id}: {d}' for d in identity_differences(record)]
            problems += [f'{calc_id}: {part} not known' for part in
                         unknown_parts(record.energy_model)]
            if not (rep.path(calc_id) / 'structure_out.extxyz').is_file():
                problems.append(f'{calc_id}: structure_out.extxyz missing')
            if record.density and record.status.density != 'ok':
                problems.append(f'{calc_id}: charge density {record.status.density}')
        models = {json.dumps(r.energy_model, sort_keys=True) for r in records.values()}
        if step in ('pes', 'pes_scf', 'adhesion', 'ppes', 'charge') and len(models) != 1:
            problems.append(f'{len(models)} energy models among the calculations of the step '
                            '(energies of different models are never combined)')
        versions = sorted({(r.observed or {}).get('version') or '?' for r in records.values()})
        attempts = sum(len(r.attempts) for r in records.values())
        rep.findings.add('PROBLEM' if problems else 'ok', subject,
                         '; '.join(problems) or f'{len(records)} complete, parsing ok, observed '
                         f'as expected (pw.x {", ".join(versions)}), one energy model, '
                         f'{attempts} attempts')


def check_spin(rep):
    from ase.io.espresso import read_fortran_namelist

    if not rep.records:
        rep.findings.add('not observed', 'spin', 'no calculation prepared')
        return
    expected = {i + 1: s.starting_magnetization for i, s in enumerate(rep.settings.atomic_species)}
    nspins, problems = {}, []
    for calc_id in rep.records:
        with open(rep.path(calc_id) / 'input.pwi') as f:
            system = read_fortran_namelist(f)[0].get('system', {})
        nspin = system.get('nspin', 1)
        nspins.setdefault(nspin, []).append(calc_id)
        if nspin == 2:
            for i, value in expected.items():
                if system.get(f'starting_magnetization({i})') != value:
                    problems.append(f'{calc_id}: starting_magnetization({i}) = '
                                    f'{system.get(f"starting_magnetization({i})")}, settings {value}')
    if len(nspins) > 1:
        rep.findings.add('PROBLEM', 'spin', 'nspin differs among the calculations: '
                         + '; '.join(f'nspin = {n} in {len(c)}' for n, c in nspins.items()))
    elif problems:
        rep.findings.add('PROBLEM', 'spin', '; '.join(problems[:4]))
    elif 2 in nspins:
        moments = ', '.join(f'{s.label} {s.starting_magnetization}'
                            for s in rep.settings.atomic_species)
        rep.findings.add('ok', 'spin', f'nspin = 2 in all {len(rep.records)} inputs, '
                         f'starting_magnetization as the settings: {moments}')
    else:
        rep.findings.add('note', 'spin', f'nspin = {next(iter(nspins))} in all '
                         f'{len(rep.records)} inputs: not spin-polarized')


def check_slabs(rep):
    from triboworkflow.core.structures import StructureError
    from triboworkflow.workflows.interface.match import match_inputs

    try:
        inputs = match_inputs(rep.settings)
    except (StructureError, OSError, ValueError) as error:
        rep.findings.add('PROBLEM', 'slabs', f'the slab files are not read: {error}')
        return
    for i, (slab, name) in enumerate(zip(inputs.slabs, inputs.files), 1):
        changes = []
        if slab.rotated:
            deviation = float(np.abs(np.asarray(slab.rotation) - np.eye(3)).max())
            changes.append(f'rotated to the frame of the workflow (largest element of R - I '
                           f'{deviation:.3g})')
        if slab.dropped:
            changes.append(f'dropped: {", ".join(slab.dropped)}')
        if slab.added:
            changes.append(f'added: {", ".join(slab.added)}')
        text = f'{name} read as {slab.path}; ' + ('; '.join(changes) or 'no rotation, nothing '
                                                  'dropped or added')
        rep.findings.add('note' if slab.rotated or slab.dropped else 'ok', f'slab {i}', text)


# --- the PES -------------------------------------------------------------------------------

def _calculation_of(calc_type):
    return 'scf' if calc_type == 'slab_from_scratch' else 'relax'


def _energy_of_output(rep, calc_id, record):
    """Energy in eV from the last output of a calculation, or None if it is not there."""
    from triboworkflow.engines.qe.results import RY_TO_EV, read_pw_values

    attempt = record.attempts[-1] if record.attempts else None
    name = attempt.output.path if attempt is not None and attempt.output is not None \
        else 'output.pwo'
    path = rep.path(calc_id) / name
    if not path.is_file():
        return None
    values = read_pw_values(path, _calculation_of(record.calc_type))
    ry = values.final_energy if values.final_energy is not None else values.energy
    return None if ry is None else ry * RY_TO_EV


def _is_grid(rows):
    """Whether ``rows`` is a list of ``[point, x, y, energy]`` with numbers."""
    return isinstance(rows, list) and all(
        isinstance(row, list) and len(row) == 4 and isinstance(row[0], str)
        and all(_finite(value) for value in row[1:]) for row in rows)


def _grid_problems(rep, name, entry, grid):
    """What differs between the grids of a PES and the points they come from.

    The energy of each row of ``hs_unique_ev`` is that of its point (``<pes>.data``) and its
    ``x, y`` the first lateral shift of the point (``highsym.inter_unique``); each row of
    ``hs_unique_jm2`` is the same row with the energy in J/m^2 on the area of the cell of the
    final structure of the lowest point, converted as the step converts it.
    """
    from triboworkflow.vendored.physics.base.solidstate import get_cell_info
    from triboworkflow.vendored.physics.base.units import Converter

    problems = []
    for point, x, y, energy in grid:
        data = dig(entry, 'data', point, 'output', 'energy')
        if not _finite(data) or abs(energy - data) > 1e-9 * max(1.0, abs(data)):
            problems.append(f'hs_unique_ev {point}: energy {energy!r} eV, {data!r} eV in '
                            f'{name}.data')
        shifts = dig(rep.doc, 'highsym', 'inter_unique', point)
        shift = shifts[0] if isinstance(shifts, list) and shifts else None
        if shift is None or abs(x - shift[0]) > 1e-9 or abs(y - shift[1]) > 1e-9:
            problems.append(f'hs_unique_ev {point}: x, y = {x!r}, {y!r}, shift of the point '
                            f'{shift!r}')
    jm2 = dig(entry, 'energy', 'grid', 'hs_unique_jm2')
    matrix = dig(entry, 'data', grid[0][0], 'output', 'structure', 'lattice', 'matrix')
    if not jm2:
        problems.append('no energy.grid.hs_unique_jm2')
    elif not _is_grid(jm2) or len(jm2) != len(grid):
        problems.append('hs_unique_jm2: not [point, x, y, energy] rows, or not as many as '
                        'hs_unique_ev')
    elif matrix is None:
        problems.append(f'no output.structure.lattice of {name}.data.{grid[0][0]}, the area '
                        'of the J/m^2 is not known')
    else:
        area, _ = get_cell_info(matrix)
        for (point, x, y, energy), (p2, x2, y2, value) in zip(grid, jm2):
            expected = Converter.eV_to_Jm2(energy, area * 1e-20)
            if (p2, x2, y2) != (point, x, y) or abs(value - expected) > 1e-9 * max(
                    1e-3, abs(expected)):
                problems.append(f'hs_unique_jm2 {p2}: {value!r} J/m^2 for {energy!r} eV on '
                                f'{area:.6g} angstrom^2 ({expected!r} expected)')
    return problems


def check_pes(rep, name):
    subject = name
    entry = rep.doc.get(name)
    if not entry or 'data' not in entry:
        rep.findings.add('not observed', subject, f'no entry {name}.data in system.json')
        return
    problems, checked, missing_output = [], 0, 0
    for point, result in entry['data'].items():
        calc_id = f'{name}/{point}'
        output = result.get('output') if isinstance(result, dict) else None
        if not isinstance(output, dict):
            problems.append(f'{calc_id}: no output in the entry')
            continue
        # A key that is missing is reported with its point; the comparisons that need it are
        # skipped, the others go on.
        absent = sorted(OUTPUT_KEYS - set(output))
        if absent:
            problems.append(f'{calc_id}: keys missing {absent}')
        if not _finite(output.get('energy')) or not output.get('nsites'):
            problems.append(f'{calc_id}: energy {output.get("energy")}, nsites '
                            f'{output.get("nsites")}')
            continue
        per_atom = output.get('energy_per_atom')
        if per_atom is not None and (not _finite(per_atom) or
                                     abs(per_atom - output['energy'] / output['nsites']) > 1e-9):
            problems.append(f'{calc_id}: energy_per_atom is not energy / nsites')
        if 'task_label' in output and output['task_label'] != calc_id:
            problems.append(f'{calc_id}: task_label {output.get("task_label")!r}')
        if 'structure' not in (result.get('input') or {}):
            problems.append(f'{calc_id}: no input.structure')
        record = rep.records.get(calc_id)
        read = None if record is None else _energy_of_output(rep, calc_id, record)
        if read is None:
            missing_output += 1
        else:
            checked += 1
            if abs(read - output['energy']) > 1e-8 * max(1.0, abs(read)):
                problems.append(f'{calc_id}: energy {output["energy"]!r} eV in the entry, '
                                f'{read!r} eV in the output (Ry x RY_TO_EV)')
    grid = dig(entry, 'energy', 'grid', 'hs_unique_ev')
    energies = []
    if not grid:
        problems.append('no energy.grid.hs_unique_ev')
    elif not _is_grid(grid):
        problems.append('hs_unique_ev: rows are not [point, x, y, energy]')
    else:
        energies = [row[3] for row in grid]
        if energies != sorted(energies):
            problems.append('hs_unique_ev is not sorted by energy')
        if dig(entry, 'energy', 'min_site') != grid[0][0] \
                or dig(entry, 'energy', 'max_site') != grid[-1][0]:
            problems.append('min_site and max_site are not the first and the last of hs_unique_ev')
        if len(grid) != len(entry['data']):
            problems.append(f'{len(grid)} rows in hs_unique_ev, {len(entry["data"])} points')
        problems += _grid_problems(rep, name, entry, grid)
    for which in ('min', 'max'):
        if dig(rep.doc, 'structure', 'opt', name, which) is None:
            problems.append(f'no structure.opt.{name}.{which}')
    if problems:
        rep.findings.add('PROBLEM', subject, '; '.join(problems[:5]))
        return
    spread = max(energies) - min(energies)
    note = f'; {missing_output} outputs not in the copy' if missing_output else ''
    rep.findings.add('ok' if not missing_output else 'note', subject,
                     f'{len(entry["data"])} points with the keys of the entries, energy in eV '
                     f'equal to the output in Ry on {checked}, sorted, min_site '
                     f'{entry["energy"]["min_site"]}, spread {spread:.4g} eV{note}')


def _analysis_job(rep, step):
    """The job of the analysis of a step: ``(verdict, text, facts)`` of its group."""
    entry = rep.entry(step)
    if entry is None or entry.get('status') != 'completed':
        return 'not observed', f'{step} not completed', {}
    group = rep.group_dir(step)
    if group is None:
        return 'not observed', f'{step} analysed by hand, or [execution.analysis] not given: ' \
            'no job', {}
    if not group.is_dir():
        return 'PROBLEM', f'the folder of the group {group.name} is not in the copy', {}
    outcome_path = group / 'outcome.json'
    if not outcome_path.is_file():
        return 'PROBLEM', f'{group.name}: no outcome.json', {}
    outcome = json.loads(outcome_path.read_text())
    started, ended = local_time(outcome.get('started')), local_time(outcome.get('ended'))
    seconds = None if started is None or ended is None else (ended - started).total_seconds()
    job_ids = rep.job_ids(group)
    log = rep.job_log(group)
    facts = {'group': group, 'outcome': outcome, 'seconds': seconds, 'job_ids': job_ids,
             'log': log, 'deprecation': log.count('DeprecationWarning'),
             'warnings': log.count(' WARNING '), 'log_bytes': len(log.encode())}
    sacct = [rep.sacct[j] for j in job_ids if j in rep.sacct]
    facts['sacct'] = sacct[0] if sacct else None
    text = (f'group {group.name}, job {",".join(job_ids) or "?"}, outcome {outcome.get("result")}'
            f', {"?" if seconds is None else f"{seconds:.0f}"} s from the outcome; log '
            f'{facts["log_bytes"]} bytes, {facts["warnings"]} warnings')
    if facts['sacct']:
        job = facts['sacct']
        text += (f'; sacct {job["State"]}, elapsed {job["Elapsed"]} of {job["Timelimit"]}, '
                 f'MaxRSS {_gib(job["MaxRSS"])} of {job["ReqMem"]}')
    else:
        text += '; memory and walltime not observed (no sacct line for the job)'
    if outcome.get('result') not in SUCCESS or outcome.get('error'):
        return 'PROBLEM', f'{text}; error {outcome.get("error")}', facts
    return 'ok', text, facts


def check_plotpes(rep, name):
    pes_name = PES_OF[name]
    energy = dig(rep.doc, pes_name, 'energy') or {}
    leaves = {'pes_ev': energy.get('pes_ev'), 'pes_jm2': energy.get('pes_jm2'),
              'hs_all_ev': dig(energy, 'grid', 'hs_all_ev'),
              'hs_all_jm2': dig(energy, 'grid', 'hs_all_jm2')}
    absent = [key for key, value in leaves.items() if not value]
    if absent:
        rep.findings.add('not observed', name, f'entries missing in {pes_name}.energy: {absent}')
    else:
        problems = []
        lowest = min(row[2] for row in leaves['pes_ev'])
        if abs(lowest) > 1e-9:
            problems.append(f'the minimum of pes_ev is {lowest!r}, not 0')
        figures = sorted(rep.sdir.glob(f'{name}/*.png'))
        if not figures:
            problems.append(f'no figure in {name}/')
        for figure in figures:
            if figure.read_bytes()[:8] != PNG_SIGNATURE:
                problems.append(f'{figure.name} is not a PNG')
        rep.findings.add('PROBLEM' if problems else 'ok', name, '; '.join(problems) or
                         f'{len(leaves["pes_ev"])} points interpolated, minimum at 0, '
                         f'{len(leaves["hs_all_ev"])} replicas, figure {figures[0].name} '
                         f'({figures[0].stat().st_size} bytes)')
    verdict, text, facts = _analysis_job(rep, name)
    rep.findings.add(verdict, f'{name} job', text)


# --- adhesion, ppes, charge ----------------------------------------------------------------

def _kpoints(path):
    text = (path / 'input.pwi').read_text()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('K_POINTS'):
            return ' '.join(lines[i + 1].split()[:6])
    return None


def check_adhesion(rep):
    from pymatgen.core import Structure

    typ = rep.interface.adhesion.type
    labels = ('bot_slab', 'top_slab', 'interface_min', 'interface_max')
    entry = dig(rep.doc, 'adhesion', typ)
    if not entry:
        rep.findings.add('not observed', 'adhesion', f'no entry adhesion.{typ}')
        return
    problems = []
    energies = {}
    for label in labels:
        data = dig(rep.doc, 'data', label) if typ == 'regular' else dig(rep.doc, 'data', 'short',
                                                                       label)
        structure = dig(rep.doc, 'structure', 'opt', typ, label)
        if data is None or structure is None:
            problems.append(f'{label}: no data or no final structure')
            continue
        energies[label] = data['energy']
        record = rep.records.get(f'adhesion/{label}')
        if record is not None and record.results is not None \
                and record.results['output']['energy'] != data['energy']:
            problems.append(f'{label}: energy of the entry differs from the record')
        slab = label.endswith('_slab')
        if slab and (structure.get('@class') != 'Slab' or 'miller_index' not in structure):
            problems.append(f'{label}: the final structure is not a Slab with its Miller indices')
        if not slab:
            regions = set(Structure.from_dict(structure).site_properties.get('interface_region')
                          or [])
            if regions != {'bot_slab', 'top_slab'}:
                problems.append(f'{label}: interface_region {sorted(regions)}')
        if len(structure['sites']) != data['nsites']:
            problems.append(f'{label}: {len(structure["sites"])} sites, nsites {data["nsites"]}')
    if len(energies) == 4:
        expected = {'min_site': energies['interface_min'] - energies['bot_slab'] - energies['top_slab'],
                    'max_site': energies['interface_max'] - energies['bot_slab'] - energies['top_slab']}
        for site, value in expected.items():
            found = dig(entry, site, 'adhesion_energy_ev')
            if not _finite(found) or abs(found - value) > 1e-9 * max(1.0, abs(value)):
                problems.append(f'{site}: adhesion_energy_ev {found}, recomputed {value}')
        corrugation = energies['interface_max'] - energies['interface_min']
        if abs(entry.get('corrugation_ev', math.nan) - corrugation) > 1e-9 * max(1.0, abs(corrugation)):
            problems.append(f'corrugation_ev {entry.get("corrugation_ev")}, recomputed {corrugation}')
    grids = {label: _kpoints(rep.path(f'adhesion/{label}')) for label in labels
             if (rep.path(f'adhesion/{label}') / 'input.pwi').is_file()}
    if problems:
        rep.findings.add('PROBLEM', 'adhesion', '; '.join(problems[:5]))
        return
    rep.findings.add('ok', 'adhesion', f'{typ}: slabs read as Slab, interfaces with interface_region '
                     f'on both slabs, energies recomputed from the entries: adhesion '
                     f'{entry["min_site"]["adhesion_energy_ev"]:.4f} eV at the lowest site, '
                     f'{entry["max_site"]["adhesion_energy_ev"]:.4f} eV at the highest, '
                     f'corrugation {entry["corrugation_ev"]:.4f} eV; k-points {grids}')


def _interface(rep, label_site):
    from pymatgen.core import Structure

    label = 'interface_min' if label_site == 'min_site' else 'interface_max'
    data = dig(rep.doc, 'structure', 'opt', rep.interface.adhesion.type, label)
    return None if data is None else Structure.from_dict(data)


def check_ppes(rep):
    from triboworkflow.core.structures import read_structure

    site = rep.interface.ppes.site
    interface = _interface(rep, site)
    if interface is None:
        rep.findings.add('not observed', 'ppes', 'no interface of the adhesion in the entries')
        return
    regions = interface.site_properties.get('interface_region')
    if regions is None:
        rep.findings.add('PROBLEM', 'ppes', 'the interface of the adhesion has no '
                         'interface_region')
        return
    top = np.array([r == 'top_slab' for r in regions])
    calcs = {c: r for c, r in rep.records.items() if c.startswith(f'ppes/{site}_')}
    if not calcs:
        rep.findings.add('not observed', 'ppes', f'no calculation ppes/{site}_*')
        return
    problems, worst = [], {'bottom': 0.0, 'top xy': 0.0, 'top z': 0.0}
    for calc_id in sorted(calcs):
        shift = float(calc_id.rpartition(f'{site}_')[2])
        path = rep.path(calc_id) / 'structure_in.extxyz'
        if not path.is_file():
            problems.append(f'{calc_id}: no structure_in.extxyz')
            continue
        structure = read_structure(path)
        if len(structure) != len(interface) or [str(s.specie) for s in structure] != \
                [str(s.specie) for s in interface]:
            problems.append(f'{calc_id}: atoms differ from those of the interface')
            continue
        delta = structure.cart_coords - interface.cart_coords
        worst['bottom'] = max(worst['bottom'], float(np.abs(delta[~top]).max()))
        worst['top xy'] = max(worst['top xy'], float(np.abs(delta[top][:, :2]).max()))
        worst['top z'] = max(worst['top z'], float(np.abs(delta[top][:, 2] - shift).max()))
    for part, value in worst.items():
        if value > POSITION_TOL:
            problems.append(f'largest deviation of the {part} atoms {value:.3g} angstrom')
    energies = dig(rep.doc, 'ppes', rep.interface.adhesion.type, 'energy', site, 'ppes_ev')
    if energies is None or len(energies) != len(calcs):
        problems.append(f'ppes_ev has {None if energies is None else len(energies)} rows for '
                        f'{len(calcs)} calculations')
    if problems:
        rep.findings.add('PROBLEM', 'ppes', '; '.join(problems[:5]))
        return
    rep.findings.add('ok', 'ppes', f'{len(calcs)} shifts: the {int((~top).sum())} atoms of the '
                     f'bottom slab stay (largest move {worst["bottom"]:.2g} angstrom), the '
                     f'{int(top.sum())} of the top slab move along z by the shift (largest '
                     f'error {worst["top z"]:.2g} angstrom, in x, y {worst["top xy"]:.2g})')


def _same_atoms(found, expected, where):
    """What differs between two structures that must hold the same atoms: a list of texts."""
    problems = []
    if len(found) != len(expected):
        return [f'{where}: {len(found)} atoms, {len(expected)} expected']
    if [str(s.specie) for s in found] != [str(s.specie) for s in expected] \
            or list(found.labels) != list(expected.labels):
        problems.append(f'{where}: species or labels differ from those expected')
    delta = float(np.abs(found.cart_coords - expected.cart_coords).max())
    if delta > POSITION_TOL:
        problems.append(f'{where}: positions differ from those of the atoms of its slab in the '
                        f'interface by up to {delta:.3g} angstrom')
    return problems


def _fragment_problems(rep, names, interface):
    """What differs between the three calculations of the charge and the interface they come from.

    The interface of the first calculation is that of the adhesion, brought around z = 0 and
    moved by half a cell: the same atoms, in the same order, moved together by a vector of the
    cell and then by one translation. Each fragment holds the atoms of one slab, told by
    ``interface_region`` of the interface of the adhesion, with the species, the labels, the
    order and the cartesian coordinates they have in the interface of the first calculation.

    Returns
    -------
    problems : list of str
    counts : str
        ``interface/bottom/top`` atoms.
    """
    from triboworkflow.core.structures import read_structure

    problems, structures = [], {}
    for part, name in zip(('interface', 'bot_slab', 'top_slab'), names):
        path = rep.path(name) / 'structure_in.extxyz'
        if path.is_file():
            structures[part] = read_structure(path)
        else:
            problems.append(f'{name}: no structure_in.extxyz')
    regions = interface.site_properties.get('interface_region')
    if regions is None or len(regions) != len(interface):
        return problems + ['the interface of the adhesion has no interface_region for its '
                           'atoms'], '?'
    counts = f'{len(interface)}/{regions.count("bot_slab")}/{regions.count("top_slab")}'
    if len(structures) < 3:
        return problems, counts
    union = structures['interface']
    if len(union) != len(interface) or [str(s.specie) for s in union] != \
            [str(s.specie) for s in interface] or list(union.labels) != list(interface.labels):
        problems.append(f'{names[0]}: its atoms are not those of the interface of the adhesion')
    elif float(np.abs(union.lattice.matrix - interface.lattice.matrix).max()) > POSITION_TOL:
        problems.append(f'{names[0]}: its cell is not the cell of the interface of the adhesion')
    else:
        # The same atoms moved together: the differences of their fractional coordinates are
        # whole vectors of the cell, up to one translation.
        shift = union.frac_coords - interface.frac_coords
        rel = shift - shift[0]
        rel -= np.round(rel)
        deviation = float(np.abs(rel @ np.asarray(union.lattice.matrix)).max())
        if deviation > POSITION_TOL:
            problems.append(f'{names[0]}: its atoms are not those of the interface of the '
                            f'adhesion moved together (up to {deviation:.3g} angstrom)')
    for part, name in zip(('bot_slab', 'top_slab'), names[1:]):
        mask = np.array([region == part for region in regions])
        if float(np.abs(structures[part].lattice.matrix - union.lattice.matrix).max()) \
                > POSITION_TOL:
            problems.append(f'{name}: not in the cell of the interface of the charge')
        if len(union) == len(mask):
            problems += _same_atoms(structures[part], _subset(union, mask), name)
    return problems, counts


def _subset(structure, mask):
    """The atoms of a structure where ``mask`` is true, in order, with their labels."""
    from pymatgen.core import Structure

    keep = [i for i, flag in enumerate(mask) if flag]
    return Structure(structure.lattice, [structure[i].specie for i in keep],
                     [structure[i].coords for i in keep], coords_are_cartesian=True,
                     labels=[structure[i].label for i in keep])


def check_charge(rep):
    from triboworkflow.engines.qe import density

    site = rep.interface.charge.site
    typ = rep.interface.adhesion.type
    parts = ('interface', 'bot_slab', 'top_slab')
    names = [f'charge/{site}_{part}' for part in parts]
    interface = _interface(rep, site)
    if interface is None or not all(n in rep.records for n in names):
        rep.findings.add('not observed', 'charge', 'no interface of the adhesion or calculations '
                         'not prepared')
        return
    problems, counts = _fragment_problems(rep, names, interface)
    rep.findings.add('PROBLEM' if problems else 'ok', 'charge fragments',
                     '; '.join(problems[:5]) or f'atoms of interface/bottom/top: {counts}; the '
                     'fragments are the atoms of their slab in the geometry of the interface '
                     '(species, labels, order, coordinates), in one cell')
    # The three densities: a file that is there and cannot be read is a problem of its own,
    # not a file that was not copied.
    filplots, seconds, sizes, unreadable = {}, 0.0, 0, []
    for name in names:
        record = rep.records[name]
        attempt = record.attempts[-1] if record.attempts else None
        found = None if attempt is None or attempt.density is None else attempt.density.density
        path = rep.path(name) / (found.path if found is not None else 'density.filplot')
        if not path.is_file():
            continue
        begin = time.perf_counter()
        try:
            filplots[name] = density.read_filplot(path)
        except (density.DensityError, OSError, ValueError) as error:
            unreadable.append(f'{name}: {path.name} not read: {error}')
        seconds += time.perf_counter() - begin
        sizes += path.stat().st_size
    if unreadable:
        rep.findings.add('PROBLEM', 'charge densities', '; '.join(unreadable))
    elif len(filplots) < 3:
        rep.findings.add('not observed', 'charge densities', f'{len(filplots)} of the 3 filplots '
                         'in the copy (density.filplot of each calculation)')
    else:
        try:
            density.check_comparable(filplots, [rep.records[n] for n in names])
        except density.DensityError as error:
            rep.findings.add('PROBLEM', 'charge densities', str(error))
        else:
            grid = next(iter(filplots.values())).grid
            peak = peak_rss()
            rep.findings.add('ok', 'charge densities',
                             f'same grid {grid}, cell, cutoffs and energy model; read here in '
                             f'{seconds:.1f} s, {sizes / 1e6:.1f} MB of text'
                             + ('' if peak is None else f', peak memory of this process '
                                f'{_gib(peak)}'))
    displacement = dig(rep.doc, 'charge', typ, 'displacement', site, 'z')
    width = dig(displacement, 'interface', 'width')
    if displacement is None:
        rep.findings.add('not observed', 'charge contact plane', 'no displacement entry')
    elif not _finite(width) or width <= 0 or not _finite(
            dig(displacement, 'interface', 'integral')):
        rep.findings.add('PROBLEM', 'charge contact plane', f'region between the slabs: width '
                         f'{width}')
    else:
        rep.findings.add('ok', 'charge contact plane',
                         f'region between the slabs {width:.3f} angstrom, integral '
                         f'{displacement["interface"]["integral"]:.4g} e/angstrom^3 there, '
                         f'{displacement["all"]["integral"]:.4g} over the cell')


# --- shear ---------------------------------------------------------------------------------

def check_shear(rep):
    verdict, text, facts = _analysis_job(rep, 'shear')
    rep.findings.add(verdict, 'shear job', text)
    log = facts.get('log', '')
    interpolation = _INTERPOLATION.search(log)
    strings = _STRING.findall(log)
    if facts:
        rep.findings.add('ok' if interpolation and len(strings) == 3 else 'not observed',
                         'shear time',
                         (f'interpolation {interpolation.group(1)} s on {interpolation.group(2)} '
                          'centres; ' if interpolation else 'interpolation not in the log; ')
                         + ('; '.join(f'string {n}: {s} steps, {t} s' for n, s, t in strings)
                            or 'strings not in the log'))
        rep.findings.add('note' if facts['deprecation'] else 'ok', 'shear warnings',
                         f'{facts["deprecation"]} DeprecationWarning lines and '
                         f'{facts["warnings"]} warnings in a log of {facts["log_bytes"]} bytes')
    strengths = rep.doc.get('shearstr')
    if not strengths:
        rep.findings.add('not observed', 'shear', 'no entry shearstr')
        return
    mep = rep.interface.shear.mep
    problems, stopped = [], []
    for path in ('along_bs_line', 'along_x', 'along_y'):
        entry = strengths.get(path) or {}
        if not _finite(entry.get('shear_strength_GPa')):
            problems.append(f'{path}: shear_strength_GPa {entry.get("shear_strength_GPa")}')
        if _finite(entry.get('tol')) and entry['tol'] > mep.tol:
            stopped.append(f'{path} (tol {entry["tol"]:.3g} > {mep.tol:g}, nstep '
                           f'{entry.get("nstep")})')
    if problems:
        rep.findings.add('PROBLEM', 'shear', '; '.join(problems))
    else:
        values = ', '.join(f'{p} {strengths[p]["shear_strength_GPa"]:.4f}' for p in strengths)
        if stopped:
            rep.findings.add('note', 'shear convergence', f'strength (GPa): {values}; the string '
                             f'stopped at nstepmax = {mep.nstepmax} without converging: '
                             f'{"; ".join(stopped)}')
        else:
            rep.findings.add('ok', 'shear convergence', f'strength (GPa): {values}; the three '
                             f'strings reached tol = {mep.tol:g}')
    # The cell of the interface against the cell of the PES that plotpes interpolated.
    cell = dig(rep.doc, 'structure', 'init', 'interface', 'lattice', 'matrix')
    minimum = dig(rep.doc, 'pes', 'energy', 'grid', 'hs_unique_ev')
    final = None if not minimum else dig(rep.doc, 'pes', 'data', minimum[0][0], 'output',
                                         'structure', 'lattice', 'matrix')
    if cell is None or final is None:
        rep.findings.add('not observed', 'shear cell', 'the cell of the interface or of the PES '
                         'is not in the entries')
        return
    deviation = float(np.abs(np.asarray(cell)[:2, :2] - np.asarray(final)[:2, :2]).max())
    verdict = 'ok' if deviation <= CELL_OK else 'note' if deviation <= CELL_NOTE else 'PROBLEM'
    rep.findings.add(verdict, 'shear cell', f'a, b of structure.init.interface against the cell '
                     f'of the minimum of the PES: largest difference {deviation:.3g} angstrom')


def check_accounting(rep):
    from triboworkflow.execution.jobs import SUBMITTED_FILE

    if not rep.sacct:
        rep.findings.add('not observed', 'accounting', 'no file of sacct given (--sacct): the '
                         'states, the walltime and the memory of the jobs are not known')
        return
    ids = []
    for path in sorted((rep.run_dir / 'jobs').glob(f'*/{SUBMITTED_FILE}')):
        ids += [j for j in json.loads(path.read_text()).get('job_ids', []) if j not in ids]
    known = [j for j in ids if j in rep.sacct]
    states = {}
    for job_id in known:
        states.setdefault(rep.sacct[job_id]['State'], []).append(job_id)
    bad = {s: j for s, j in states.items() if s.startswith(('OUT_OF_ME', 'TIMEOUT'))}
    longest = max((duration(rep.sacct[j]['Elapsed']) or 0 for j in known), default=0)
    peak = max((rep.sacct[j]['MaxRSS'] for j in known), default=0)
    text = (f'{len(known)} of {len(ids)} jobs in the file: '
            + ', '.join(f'{len(j)} {s}' for s, j in states.items())
            + f'; longest {longest} s, largest MaxRSS {_gib(peak)}')
    rep.findings.add('PROBLEM' if bad else 'ok' if len(known) == len(ids) else 'note',
                     'accounting', text + (f'; {bad}' if bad else ''))


# --- report --------------------------------------------------------------------------------

def make_report(run_dir, sacct=None):
    """The checks of a run, without printing: an ``InterfaceReport`` with its findings.

    Parameters
    ----------
    run_dir : path-like
        Folder of the run.
    sacct : path-like or None, optional
        Output of ``sacct -P`` (see the module notes).

    Returns
    -------
    InterfaceReport
        ``findings.items`` holds ``(verdict, subject, text)`` in order.
    """
    rep = InterfaceReport(run_dir, sacct)
    _run(rep, 'steps', check_steps)
    _run(rep, 'calculations', check_calculations)
    _run(rep, 'spin', check_spin)
    _run(rep, 'slabs', check_slabs)
    for name in ('pes', 'pes_scf'):
        if name in rep.steps:
            _run(rep, name, check_pes, name)
    for name in ('plotpes', 'plotpes_scf'):
        if name in rep.steps:
            _run(rep, name, check_plotpes, name)
    for name, check in (('adhesion', check_adhesion), ('ppes', check_ppes),
                        ('charge', check_charge), ('shear', check_shear)):
        if name in rep.steps:
            _run(rep, name, check)
    _run(rep, 'accounting', check_accounting)
    return rep


def _run(rep, subject, check, *args):
    """Run a check; one that stops on a file it cannot make sense of is a ``PROBLEM``.

    The checks look for what is wrong in the entries and the files and say it. An entry or
    a file of a shape no check expected must not take the other checks down with it: the
    check that met it gives a ``PROBLEM`` that names it and the error, and the report goes on.
    """
    try:
        check(rep, *args)
    except Exception as error:  # a malformed entry or file: reported, not hidden
        rep.findings.add('PROBLEM', subject, f'the check stopped on a malformed entry or file: '
                         f'{type(error).__name__}: {error}')


def report(args):
    """Command ``report``: what the files of a run tell, with verdicts. Read only."""
    rep = make_report(args.run, args.sacct)
    print(f'Run {rep.run_dir}: system {rep.system}, steps {rep.steps}, '
          f'{len(rep.records)} calculations, {len(list((rep.run_dir / "jobs").glob("*/")))} '
          f'groups of jobs')
    print('\nVerdicts:')
    width = max((len(subject) for _, subject, _ in rep.findings.items), default=0)
    for verdict, subject, text in rep.findings.items:
        print(f'  {verdict:<12} {subject:<{width}}  {text}')
    counts = {v: sum(1 for item in rep.findings.items if item[0] == v) for v in VERDICTS}
    print('RESULT: ' + ', '.join(f'{n} {v}' for v, n in counts.items()))
    return 1 if rep.findings.failed else 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('report', help='what the files of a run tell, with verdicts (read only)')
    p.add_argument('run', help='folder of the run (config.toml, orchestrator.json, systems/)')
    p.add_argument('--sacct', default=None, help='output of "sacct -P -j <ids> -o JobID,JobName,'
                   'State,ExitCode,Elapsed,Timelimit,ReqMem,MaxRSS,NodeList"')
    args = parser.parse_args(argv)
    from triboworkflow.cli.run import RunError

    try:
        return {'report': report}[args.command](args)
    except (OSError, ValueError, RunError) as error:  # settings, records, structures: the files
        print(f'Error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
