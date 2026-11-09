# Author: Vito Foderà
# Created: 2026-11-09

"""Cost of the analysis of the step shear, and how much its seed matters.

For each interface of the golden cases of match, with the synthetic energies of
the golden cases on its high-symmetry points, the script builds the PES in J/m^2
that ``plotpes`` writes and measures what the analysis of ``shear`` does with it:

- the interpolation of the PES for the three strings (``PES.make_pes`` with the
  points replicated 5 x 5 and the grid of density 20 that the step does not use),
  and, apart, the query of that grid; the time of the interpolation of ``plotpes``
  in eV (6 x 6 replicas, density 50) on the same points, to give the ratio;
- each string (best straight line, axis x, axis y): the starting string
  (``get_bs_line`` for the first one), the steps of the string method with the time
  of one step, and the shear strength along the path;
- the size: points of the PES, centres of the radial basis functions (100 times
  the points), the memory of their dense matrix (8 bytes per element, the matrix
  alone) and the peak memory of the process so far;
- with ``--seeds N``, the shear strength along the best straight line for N seeds
  of the noise of its starting string: mean, standard deviation and spread, and the
  steps done.

The parameters of the string method are the recommended ones; ``--npts``, ``--h``,
``--nstepmax``, ``--tol`` and ``--mep-delta`` change them, to show their effect on the
cost (they are the settings ``[interface.shear.mep]``). An interface whose matrix would
exceed ``--max-gib`` is only counted. The BLAS libraries run on one thread, as in the
golden cases: on a node with more threads the solution is faster, and the numbers can
differ in the last digits.

Usage, in the development environment (not collected by pytest)::

    python tests/shear_cost.py [--pairs cu111_cu111 ...] [--seeds 8] [--nstepmax 1000]

Notes
-----
To remove: the measures of D25 for the string method; the interpolation of ``plotpes`` is
TribChem's ``get_pes`` as ``FT_PESInterpolation`` calls it (A07).
"""

import argparse
import contextlib
import io
import sys
import time
import warnings

import golden_cases as gc  # sets the BLAS threads before numpy
import plotpes_cost as pc

import numpy as np  # noqa: E402

from triboworkflow.core.settings.schema import SHEAR_DELTA, SHEAR_MEP  # noqa: E402
from triboworkflow.vendored.physics.base.solidstate import generate_uniform_grid  # noqa: E402
from triboworkflow.vendored.physics.tribology import potensurf  # noqa: E402
from triboworkflow.vendored.physics.tribology.mep import MEP, get_bs_line  # noqa: E402
from triboworkflow.vendored.physics.tribology.pes import PES  # noqa: E402

# Interfaces of the golden cases of match, from the smallest.
PAIRS = pc.PAIRS
OPTIMIZATIONS = ('bs_line', 'x', 'y')
# Arguments of the interpolation of the step.
REPLICATE_OF, DENSITY, TOL = (5, 5), 20, 1e-4
REPLICAS = (2 * REPLICATE_OF[0]) * (2 * REPLICATE_OF[1])


def quiet(function, *args, **kwargs):
    """The result of ``function`` with what it prints discarded."""
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def string_method(rbf, cell, optimization, seed, mep):
    """One string and its shear strength: time of the method, steps, strength and time of it."""
    np.random.seed(seed)
    string = MEP(rbf=rbf, cell=cell)
    start = time.perf_counter()
    quiet(string.get_mep, method='zerotemp', optimization=optimization, **mep)
    seconds = time.perf_counter() - start
    quiet(string.get_shear_strenght, delta=SHEAR_DELTA, npts=mep['npts'], optimization=optimization)
    data = string.mep_data[f'along_{optimization}']
    return {'seconds': seconds, 'nstep': data['nstep'], 'tol': data['tol'],
            'shear': data['shear_strength_GPa']}


def measure(pair, mep, seeds):
    """Sizes and times of the analysis of one pair, as a dict."""
    hs_all, values, cell = pc.inputs(pair)
    data_jm2 = quiet(potensurf.get_pes, hs_all, values[1], cell, title=pair, to_fig=None)[2]
    row = {'pair': pair, 'unique': len(values[0]), 'points': len(data_jm2)}
    row['plotpes'], _ = pc.best(lambda: quiet(potensurf.get_pes, hs_all, values[0], cell,
                                              title=pair, to_fig=None), 1)

    def build():
        solver = PES(np.array(data_jm2), cell)
        quiet(solver.make_pes, replicate_of=REPLICATE_OF, density=DENSITY, tol=TOL)
        return solver

    row['build'], solver = pc.best(build, 1)
    rbf = solver._rbf
    row['centres'] = len(rbf.y)
    grid = generate_uniform_grid(solver.cell, DENSITY, to_plot=False)[:, :2]
    row['grid_points'] = len(grid)
    row['grid'], _ = pc.best(lambda: rbf(grid), 1)
    np.random.seed(0)
    row['bs_start'], _ = pc.best(lambda: quiet(get_bs_line, cell, rbf, mep['npts'], mep['extent'],
                                                1.0, mep['delta']), 1)
    row['strings'] = {opt: string_method(rbf, cell, opt, 0, mep) for opt in OPTIMIZATIONS}
    row['peak_gib'] = pc.peak_gib()
    if seeds:
        row['seeds'] = [string_method(rbf, cell, 'bs_line', seed, mep) for seed in range(seeds)]
    return row


def report(row):
    """Print the measures of one pair."""
    print(f'{row["pair"]}: {row["unique"]} unique points, {row["points"]} points of the PES, '
          f'{row["centres"]} centres: matrix {pc.matrix_gib(row["centres"]):.3f} GiB; peak of the '
          f'process so far {row["peak_gib"]:.2f} GiB', flush=True)
    print(f'  interpolation of plotpes in eV {row["plotpes"]:.2f} s; interpolation of the shear '
          f'{row["build"]:.2f} s (ratio {row["build"] / row["plotpes"]:.2f}); query of its '
          f'unused grid ({row["grid_points"]} points) {row["grid"]:.3f} s')
    total = row['build']
    for opt, string in row['strings'].items():
        total += string['seconds']
        print(f'  string {opt}: {string["nstep"]} steps, {string["seconds"]:.1f} s '
              f'({1000 * string["seconds"] / max(string["nstep"], 1):.2f} ms per step), final '
              f'displacement {string["tol"]:.2e}, shear strength {string["shear"]:.6f} GPa')
    print(f'  starting string of bs_line {row["bs_start"]:.2f} s; analysis of the step '
          f'{total:.1f} s, {total / row["plotpes"]:.1f} times the interpolation of plotpes',
          flush=True)
    if row.get('seeds'):
        shear = np.array([s['shear'] for s in row['seeds']])
        steps = np.array([s['nstep'] for s in row['seeds']])
        spread = shear.max() - shear.min()
        print(f'  {len(shear)} seeds of bs_line: shear strength mean {shear.mean():.8f} GPa, '
              f'standard deviation {shear.std(ddof=1) if len(shear) > 1 else 0.0:.2e}, spread '
              f'{spread:.2e} (relative {spread / shear.mean():.2e}); steps {steps.min()} to '
              f'{steps.max()}', flush=True)


def main(argv=None):
    recommended = {key: list(value) if key == 'extent' else value
                   for key, value in SHEAR_MEP.items()}
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--pairs', nargs='+', default=list(PAIRS), choices=PAIRS,
                        help='interfaces to measure (default: all of them)')
    parser.add_argument('--max-gib', type=float, default=4.0,
                        help='measure only the interfaces whose matrix takes at most this (GiB)')
    parser.add_argument('--seeds', type=int, default=0,
                        help='seeds of the noise of the best straight line to compare')
    parser.add_argument('--npts', type=int, default=recommended['npts'])
    parser.add_argument('--h', type=float, default=recommended['h'])
    parser.add_argument('--nstepmax', type=int, default=recommended['nstepmax'])
    parser.add_argument('--tol', type=float, default=recommended['tol'])
    parser.add_argument('--mep-delta', type=float, default=recommended['delta'])
    args = parser.parse_args(argv)
    mep = {'npts': args.npts, 'extent': recommended['extent'], 'h': args.h,
           'nstepmax': args.nstepmax, 'tol': args.tol, 'delta': args.mep_delta}
    warnings.simplefilter('ignore')
    print(f'BLAS threads: {gc.BLAS_THREADS}; string method {mep}', flush=True)
    for pair in args.pairs:
        hs_all, values, cell = pc.inputs(pair)
        unfolded, data = pc.points(hs_all, values[1], cell)
        centres = REPLICAS * len(data)
        if pc.matrix_gib(centres) > args.max_gib:
            print(f'{pair}: {len(values[0])} unique points, {len(data)} left after the '
                  f'duplicates, {centres} centres ({REPLICATE_OF[0] * 2} x {REPLICATE_OF[1] * 2} '
                  f'replicas): matrix {pc.matrix_gib(centres):.1f} GiB, not measured', flush=True)
            continue
        report(measure(pair, mep, args.seeds))
    return 0


if __name__ == '__main__':
    sys.exit(main())
