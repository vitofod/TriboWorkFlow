# Author: Vito Foderà
# Created: 2026-11-04

"""Fake pw.x and pp.x for the tests of ``tests/hpc/qe_check.py``: no electrons are computed.

``python qe_check_fake.py pw -in run.pwi`` prints on the standard output a
pw.x 7.2 output of the input, with the lines the package reads (built with
``pwo_helpers``): the parallelization, the summary with ``nstep`` and
``mixing beta`` of the input, the checksums of the UPF files of
``pseudo_dir``, the positions, the symmetry operations that spglib finds,
and an end chosen per calculation. ``QE_CHECK_FAKE`` names a JSON file
``{label: [end of attempt 1, end of attempt 2, ...]}``, the label being the
name of the directory of the calculation; the attempt is the number of
attempts already archived plus one, and a calculation or an attempt not
listed ends ``completed``. The ends:

- ``completed``: SCF converged; a relax converged, the free coordinates moved
  by a little, the fixed ones kept;
- ``max_seconds``, ``user_request``: stopped inside the SCF;
- ``max_steps`` (relax): steps over;
- ``scf_not_converged``: the SCF not converged.

With ``--emt`` after ``pw`` the converged run has the energy, the forces and
the stress of ASE's EMT, a classical potential, in place of constants, and a
relax is a real minimization (BFGS) of the free coordinates: the surfaces that
the steps of the workflow build from such runs vary from point to point, for
the tests of the whole chain (``tests/test_interface_chain.py``). EMT covers
Ag, Al, Au, C, Cu, H, N, Ni, O, Pd and Pt, not Fe. This is a stand-in for a
program on the HPC machine, kept in the tests: the package never computes.

Every end but ``completed`` leaves ``<outdir>/<prefix>.save/`` with the files
a restart needs. With ``restart_mode = 'restart'`` the output says that
positions, density and wavefunctions were read from them, the SCF iteration
too after ``max_seconds`` in an scf, and the history of the optimizer in a
relax.

``python qe_check_fake.py pp -in input.ppi`` writes the ``filplot`` of the
input, a uniform density in the cell of ``structure_out.extxyz``
(``charge_files.structure_filplot``), and prints a pp.x output that closes
its run.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import charge_files  # noqa: E402
import pwo_helpers  # noqa: E402
from run_helpers import JOB_DONE, MAX_SECONDS, MAX_STEPS, NOT_CONVERGED, PARALLEL  # noqa: E402

USER_REQUEST = ['', '     Program stopped by user request']
RESTART_LINES = ['     Atomic positions and unit cell read from directory:',
                 '     OUT/prefix.save/', '     Atomic positions from file used, from input '
                 'discarded', '', '     The initial density is read from file :',
                 '     OUT/prefix.save/charge-density', '', '     Starting wfcs from file']
FILPLOT_GRID = (4, 4, 24)


def _input(path):
    from ase.io.espresso import read_fortran_namelist

    with open(path) as f:
        namelists, cards = read_fortran_namelist(f)
    return namelists, cards


def _species(cards, pseudo_dir):
    """``(label, element, UPF file, md5)`` of each line of ATOMIC_SPECIES."""
    start = next(i for i, line in enumerate(cards) if line.startswith('ATOMIC_SPECIES'))
    species = []
    for line in cards[start + 1:]:
        parts = line.split()
        if len(parts) != 3:
            break
        label, _, upf = parts
        md5 = hashlib.md5((Path(pseudo_dir) / upf).read_bytes()).hexdigest()
        species.append((label, label, upf, md5))
    return species


def _symmetry_line(structure):
    import spglib

    cell = (structure.lattice.matrix, structure.frac_coords, [s.specie.Z for s in structure])
    n = len(spglib.get_symmetry(cell, symprec=1e-5)['rotations'])
    return ['', f'     {n:2d} Sym. Ops. (no inversion) found', '']


def _summary(lines, nstep, beta):
    """The summary of ``pwo_helpers.header`` with the values of the input."""
    out = []
    for line in lines:
        if line.strip().startswith('mixing beta'):
            line = f'     mixing beta               = {beta:12.4f}'
        elif line.strip().startswith('nstep '):
            line = f'     nstep                     = {nstep:12d}'
        out.append(line)
    return out


def _save_dir(control):
    folder = Path(control['outdir']) / f'{control["prefix"]}.save'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'data-file-schema.xml').write_text('<qes:espresso/>\n')
    (folder / 'charge-density.dat').write_bytes(b'\0' * 8)


def pw(argv):
    from triboworkflow.core.structures import read_structure

    run_input = argv[argv.index('-in') + 1]
    namelists, cards = _input(run_input)
    control, electrons = namelists['control'], namelists.get('electrons', {})
    calculation = control.get('calculation', 'scf')
    structure = read_structure(run_input, 'pwi')
    species = _species(cards, control['pseudo_dir'])
    label = Path.cwd().name
    ends = json.loads(Path(os.environ['QE_CHECK_FAKE']).read_text()).get(label, []) \
        if os.environ.get('QE_CHECK_FAKE') else []
    attempts = Path('attempts')
    n = len([p for p in attempts.iterdir() if p.is_dir()]) + 1 if attempts.is_dir() else 1
    end = ends[n - 1] if n <= len(ends) else 'completed'
    restart = control.get('restart_mode') == 'restart'

    head = _summary(pwo_helpers.header(structure, calculation, '7.2', species),
                    control.get('nstep', 50), electrons.get('mixing_beta', 0.7))
    head = head[:2] + PARALLEL + head[2:] + _symmetry_line(structure)
    if restart:
        head += RESTART_LINES
        if calculation == 'scf' and n > 1:
            head += ['', '     Calculation restarted from scf iteration #     4']
        if calculation == 'relax':
            head += ['', '     number of scf cycles    =   2']
    open_cycle = ['     Self-consistent Calculation', '',
                  '     iteration #  1     ecut=    30.00 Ry     beta= 0.40',
                  '     total energy              =     -31.00000000 Ry']
    if end == 'completed':
        text = completed(structure, calculation, control, species, emt='--emt' in argv)
        lines = text.splitlines()
        body_start = next(i for i, line in enumerate(lines)
                          if line.strip() == 'Self-consistent Calculation')
        sys.stdout.write('\n'.join(head + lines[body_start:]) + '\n')
        return 0
    _save_dir(control)
    if end in ('max_seconds', 'user_request'):
        body = open_cycle + (MAX_SECONDS if end == 'max_seconds' else USER_REQUEST)
        code = 255 if end == 'max_seconds' else 0
    elif end == 'max_steps':
        body = pwo_helpers.scf_cycle(len(structure), -31.1, forces=[(0, 0, 0)] * len(structure)) \
            + MAX_STEPS
        code = 3
    elif end == 'scf_not_converged':
        body = open_cycle + NOT_CONVERGED
        code = 2
    else:
        raise SystemExit(f'fake pw.x: end {end!r} not known')
    sys.stdout.write('\n'.join(head + body + JOB_DONE) + '\n')
    return code


def emt_values(structure, calculation, control):
    """Energy, forces, stress and positions of a structure by ASE's EMT, in the units of pw.x.

    A ``relax`` minimizes the energy with BFGS, the atoms moving along the free
    components (``if_pos``) only, to the force threshold ``forc_conv_thr`` of the
    input. The run stops with an error if the minimization does not converge.

    Returns
    -------
    energy : float
        In Ry.
    forces : list of tuple
        In Ry/bohr, one triple per atom (the fixed components are zero).
    stress : list of tuple
        In Ry/bohr^3, the sign pw.x prints (opposite to ASE's).
    positions : list of tuple
        Cartesian, in angstrom (those of the input for a ``scf``).
    """
    from ase.calculators.emt import EMT
    from ase.optimize import BFGS

    from triboworkflow.core.structures import to_atoms
    from triboworkflow.engines.qe.results import BOHR_TO_ANGSTROM, RY_TO_EV

    atoms = to_atoms(structure)
    atoms.calc = EMT()
    if calculation == 'relax':
        fmax = control.get('forc_conv_thr', 1.0e-3) * RY_TO_EV / BOHR_TO_ANGSTROM
        if not BFGS(atoms, logfile=None).run(fmax=fmax, steps=min(control.get('nstep', 50), 250)):
            raise SystemExit('fake pw.x with EMT: the relaxation did not converge')
    forces = atoms.get_forces() * BOHR_TO_ANGSTROM / RY_TO_EV
    stress = -atoms.get_stress(voigt=False) * BOHR_TO_ANGSTROM ** 3 / RY_TO_EV
    return (atoms.get_potential_energy() / RY_TO_EV, [tuple(f) for f in forces],
            [tuple(row) for row in stress], [tuple(p) for p in atoms.get_positions()])


def completed(structure, calculation, control, species, emt=False):
    """Text of a converged run (``pwo_helpers.pw_output``); constants unless ``emt``."""
    nat = len(structure)
    forces = [(0.0, 0.0, 0.0)] * nat if control.get('tprnfor') or calculation == 'relax' \
        else None
    stress = [(0.0001, 0, 0), (0, 0.0001, 0), (0, 0, 0.0002)] if control.get('tstress') else None
    if_pos = structure.site_properties.get('if_pos') or [None] * nat
    flags = [tuple(f) if f is not None else (1, 1, 1) for f in if_pos]
    energy = {}
    if emt:
        value, emt_forces, emt_stress, positions = emt_values(structure, calculation, control)
        energy = {'energy': value}
        forces = emt_forces if forces is not None else None
        stress = emt_stress if stress is not None else None
    if calculation == 'scf':
        return pwo_helpers.pw_output(structure, 'scf', species=species, forces=forces,
                                     stress=stress, **energy)
    if emt:
        moved = positions
    else:
        moved = [tuple(c + 0.01 * f for c, f in zip(site.coords, flag))
                 for site, flag in zip(structure, flags)]
    return pwo_helpers.pw_output(structure, 'relax', species=species, forces=forces,
                                 stress=stress, final_positions=moved,
                                 final_if_pos=[f if 0 in f else None for f in flags], **energy)


def pp(argv):
    from triboworkflow.core.structures import read_structure

    namelists, _ = _input(argv[argv.index('-in') + 1])
    filplot = namelists['inputpp']['filplot']
    structure = read_structure('structure_out.extxyz')
    Path(filplot).write_text(charge_files.structure_filplot(structure, grid=FILPLOT_GRID,
                                                            electrons=float(len(structure))))
    sys.stdout.write('\n     Program POST-PROC v.7.2 starts on  1Jan2026 at 10: 0: 0 \n\n'
                     '     Reading xml data from directory:\n\n'
                     + '\n'.join(JOB_DONE) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit({'pw': pw, 'pp': pp}[sys.argv[1]](sys.argv[2:]))
