# Author: Vito Foderà
# Created: 2026-11-07

"""Cost of the interpolation of the PES of the steps plotpes and plotpes_scf.

For each interface of the golden cases of match, with the synthetic energies
of the golden cases on its high-symmetry points, the script measures what
the analysis of the step does:

- the two calls to ``get_pes``, as the step makes them: energies in eV
  without the figure, then in J/m^2 with the figure; and their ratio;
- within the second call, separately: the construction of the interpolation
  (``PES.make_pes`` without the grid: points brought into the cell, minimum
  at zero, replicas, radial basis functions solved), the query of the grid
  of density 50 that ``make_pes`` computes, the query of the 300 x 300
  points that the figure draws, and the whole figure saved;
- the size: unique, unfolded and remaining points, centres of the radial
  basis functions and the memory of their dense matrix (8 bytes per element,
  the matrix alone, not the peak); the peak memory of the process so far.

An interface whose matrix would exceed ``--max-gib`` is only counted, its
centres being 144 times its points (12 x 12 replicas). Each part is repeated
``--repeat`` times, keeping the shortest, only for the interfaces whose
matrix takes less than ``REPEAT_GIB``; the larger ones run once. The BLAS
libraries run on one thread, as in the golden cases: on a node with more
threads the solution is faster.

Usage, in the development environment (not collected by pytest)::

    python tests/plotpes_cost.py [--max-gib 4] [--repeat 3] [--figures DIR]

Notes
-----
To remove: the measures of D25 for the interpolation of the PES; the two
calls are those of TribChem's ``FT_PESInterpolation.compute_pes`` (A07).
"""

import argparse
import contextlib
import io
import resource
import sys
import tempfile
import time
from pathlib import Path

import golden_cases as gc  # sets the BLAS threads before numpy

import matplotlib  # noqa: E402

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from triboworkflow.vendored.physics.base.solidstate import (  # noqa: E402
    generate_uniform_grid, get_cell_info, pbc_coordinates)
from triboworkflow.vendored.physics.base.units import Converter  # noqa: E402
from triboworkflow.vendored.physics.tribology import potensurf  # noqa: E402
from triboworkflow.vendored.physics.tribology.pes import PES  # noqa: E402

# Interfaces of the golden cases of match, from the smallest.
PAIRS = ('cu111_cu111', 'pt111_ag111', 'cu111_al111_enlarged', 'cu111_al111')
# Arguments of get_pes (potensurf.py) and of the figure it draws.
REPLICATE_OF, DENSITY, TOL = (6, 6), 50, 1e-4
EXTENT, MPTS = (2, 2), (300j, 300j)
REPLICAS = (2 * REPLICATE_OF[0]) * (2 * REPLICATE_OF[1])
# Interfaces with a larger matrix (GiB) are measured once.
REPEAT_GIB = 0.5


def inputs(pair):
    """``hs_all``, energies ``[eV, J/m^2]`` and cell of the interface of a pair.

    The energies are those of ``golden_cases.synthetic_energies``, sorted as
    the step sorts them, and converted to J/m^2 on the area of the cell.
    """
    document = gc.pes_document(pair)
    cell = np.array(document['structure']['init']['interface']['lattice']['matrix'])
    ev = sorted(gc.synthetic_energies(document['highsym']['inter_unique'], cell),
                key=lambda row: row[3])
    area, _ = get_cell_info(cell)
    jm2 = [[s, x, y, Converter.eV_to_Jm2(v, area * 1e-20)] for s, x, y, v in ev]
    return document['highsym']['inter_all'], [ev, jm2], cell


def points(hs_all, values, cell):
    """Points that ``get_pes`` interpolates: unfolded, without duplicates, in the cell, minimum at zero.

    Returns
    -------
    unfolded : int
    data : numpy.ndarray
    """
    v_list, data = potensurf.unfold_pes(hs_all, values)
    data = potensurf.remove_duplicates(data)
    data[:, :2] = pbc_coordinates(data[:, :2], cell, to_array=True)
    data[:, 2] = data[:, 2] - min(data[:, 2])
    return len(v_list), data


def best(function, repeat):
    """Shortest time of ``repeat`` calls of ``function``, and its last result."""
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - start)
    return min(times), result


def peak_gib():
    """Peak resident memory of this process so far, in GiB (Linux: kB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def measure(pair, repeat, figures):
    """Sizes and times of the interpolation of one pair, as a dict."""
    hs_all, values, cell = inputs(pair)
    unfolded, data = points(hs_all, values[1], cell)
    row = {'pair': pair, 'unique': len(values[0]), 'unfolded': unfolded, 'points': len(data)}
    figure = figures / f'pes_{pair}.png'

    def call(energies, to_fig):
        with contextlib.redirect_stdout(io.StringIO()):
            solver, *_ = potensurf.get_pes(hs_all, energies, cell, title=pair, to_fig=to_fig)
        plt.close('all')
        return solver

    row['call_1'], solver = best(lambda: call(values[0], None), repeat)
    row['call_2'], _ = best(lambda: call(values[1], str(figure)), repeat)
    row['centres'] = len(solver._rbf.y)

    def build():
        pes = PES(data.copy(), cell)
        with contextlib.redirect_stdout(io.StringIO()):
            pes.make_pes(replicate_of=REPLICATE_OF, density=None, tol=TOL)
        return pes

    row['build'], pes = best(build, repeat)
    grid = generate_uniform_grid(pes.cell, DENSITY, to_plot=False)[:, :2]
    row['grid_points'] = len(grid)
    row['grid'], _ = best(lambda: pes._rbf(grid), repeat)
    a, b = np.sum(np.abs(pes.cell_2d[:, 0])), np.sum(np.abs(pes.cell_2d[:, 1]))
    x, y = np.mgrid[0:EXTENT[0] * a:MPTS[0], 0:EXTENT[1] * b:MPTS[1]]
    row['plot_query'], _ = best(lambda: pes.rbf(x.reshape(1, -1), y.reshape(1, -1)), repeat)

    def plot():
        with contextlib.redirect_stdout(io.StringIO()):
            pes.plot(extent=EXTENT, mpts=MPTS, title=pair, to_fig=str(figure))
        plt.close(pes.pes_plot)

    row['plot'], _ = best(plot, repeat)
    row['peak_gib'] = peak_gib()
    return row


def matrix_gib(centres):
    """GiB of a dense square matrix of float64 on ``centres`` points."""
    return 8 * centres ** 2 / 2**30


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--max-gib', type=float, default=4.0,
                        help='measure only the interfaces whose matrix takes at most this (GiB)')
    parser.add_argument('--repeat', type=int, default=3,
                        help='calls of each part; the shortest time is kept')
    parser.add_argument('--figures', type=Path,
                        help='folder of the figures; a temporary one by default')
    args = parser.parse_args(argv)
    with contextlib.ExitStack() as stack:
        figures = args.figures or Path(stack.enter_context(tempfile.TemporaryDirectory()))
        figures.mkdir(parents=True, exist_ok=True)
        print(f'BLAS threads: {gc.BLAS_THREADS}; repeat {args.repeat}; figures in {figures}')
        for pair in PAIRS:
            hs_all, values, cell = inputs(pair)
            unfolded, data = points(hs_all, values[1], cell)
            centres = REPLICAS * len(data)
            size = (f'{pair}: {len(values[0])} unique points, {unfolded} unfolded, '
                    f'{len(data)} left after the duplicates')
            if matrix_gib(centres) > args.max_gib:
                print(f'{size}, {centres} centres (12 x 12 replicas): matrix '
                      f'{matrix_gib(centres):.1f} GiB, not measured', flush=True)
                continue
            row = measure(pair, args.repeat if matrix_gib(centres) < REPEAT_GIB else 1, figures)
            print(f'{size}, {row["centres"]} centres: matrix {matrix_gib(row["centres"]):.2f} '
                  f'GiB; peak of the process so far {row["peak_gib"]:.2f} GiB', flush=True)
            print(f'  get_pes in eV {row["call_1"]:.2f} s, in J/m^2 with the figure '
                  f'{row["call_2"]:.2f} s (ratio {row["call_2"] / row["call_1"]:.2f})')
            print(f'  construction {row["build"]:.2f} s; query of the grid of density '
                  f'{DENSITY} ({row["grid_points"]} points) {row["grid"]:.3f} s; query of the '
                  f'{int(MPTS[0].imag) * int(MPTS[1].imag)} points of the figure '
                  f'{row["plot_query"]:.3f} s; figure saved {row["plot"]:.2f} s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
