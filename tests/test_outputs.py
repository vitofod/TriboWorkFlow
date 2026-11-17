# Author: Vito Foderà
# Created: 2026-11-17

"""Tests of the files that the steps write next to their entries
(``triboworkflow.workflows.interface.outputs``).
"""

import dataclasses

import numpy as np
import pytest
from matplotlib.figure import Figure
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Chgcar

from triboworkflow.core.store import Provenance
from triboworkflow.workflows.interface import figures, outputs

STAMP = 'triboworkflow analysis sha256:' + '0' * 64
OTHER = 'triboworkflow analysis sha256:' + '1' * 64


def restamp(path):
    """Give a file of an analysis the stamp of another analysis, as if it were left by it."""
    stamp = outputs.file_stamp(path).encode()
    other = stamp[:-1] + (b'1' if stamp.endswith(b'0') else b'0')
    path.write_bytes(path.read_bytes().replace(stamp, other))


def interface(z_bottom, z_top, slabs=('bot_slab', 'top_slab')):
    """Two atoms in a cell of 20 angstrom along z, each of one slab."""
    structure = Structure(Lattice.from_parameters(3, 3, 20, 90, 90, 90), ['Fe', 'Fe'],
                          [[0, 0, z_bottom], [0.5, 0.5, z_top]], coords_are_cartesian=True)
    if slabs is not None:
        structure.add_site_property('interface_region', list(slabs))
    return structure.as_dict()


def test_distance_between_the_slabs():
    assert outputs.slab_distance(interface(-1.0, 1.0)) == pytest.approx(2.0)
    # Without the slab of the atoms, or with the slabs overlapping: not told.
    assert outputs.slab_distance(interface(-1.0, 1.0, slabs=None)) is None
    assert outputs.slab_distance(interface(1.0, -1.0)) is None
    assert outputs.slab_distance(interface(-1.0, 1.0, ('bot_slab', 'bot_slab'))) is None


def test_the_files_are_renamed_only_when_placed(tmp_path):
    path = tmp_path / 'step' / 'table.output'
    with pytest.raises(RuntimeError):
        with outputs.staged([path]) as partials:
            partials[path].write_text('half')
            raise RuntimeError
    assert not path.exists() and list((tmp_path / 'step').iterdir()) == []
    with outputs.staged([path]) as partials:
        assert partials[path].name == '.table.partial.output'
        partials[path].write_text('done')
        outputs.place(partials)
    assert path.read_text() == 'done' and list((tmp_path / 'step').iterdir()) == [path]


def test_the_stamp_follows_the_provenance_and_its_results():
    provenance = Provenance('ppes', {'fit': 'none'}, {'ppes/a': 'sha256:' + 'a' * 64}, {}, {},
                            results={'ppes/a': 'sha256:' + 'b' * 64})
    stamp = outputs.analysis_stamp(provenance)
    assert outputs.STAMP.fullmatch(stamp.encode())
    assert outputs.analysis_stamp(dataclasses.replace(provenance, parameters={'fit': 'uber'})) \
        != stamp
    assert outputs.analysis_stamp(dataclasses.replace(
        provenance, results={'ppes/a': 'sha256:' + 'c' * 64})) != stamp
    assert outputs.analysis_stamp(None) is None


def test_the_stamp_of_a_text_file_and_of_a_figure(tmp_path):
    text = tmp_path / 'table.output'
    text.write_text('\n'.join(outputs.header('Table of', 'Cu_Cu', 'ppes', stamp=STAMP)) + '\n1 2\n')
    fig = Figure()
    fig.subplots().plot([0, 1], [1, 0])
    png, pdf = tmp_path / 'figure.png', tmp_path / 'figure.pdf'
    figures.save_figure(fig, png, pdf, STAMP)
    assert [outputs.file_stamp(path) for path in (text, png, pdf)] == [STAMP] * 3
    assert outputs.written_by([text, png, pdf], STAMP)
    assert not outputs.written_by([text, png, pdf], OTHER)
    assert not outputs.written_by([text, tmp_path / 'missing.png'], STAMP)
    assert not outputs.written_by([text], None)
    # A figure saved without a stamp has none.
    figures.save_figure(fig, png, pdf)
    assert outputs.file_stamp(png) is None and outputs.file_stamp(pdf) is None


@pytest.mark.parametrize('where', ['start', 'end'])
def test_the_stamp_of_a_large_file_is_found_at_its_start_or_its_end(tmp_path, where):
    path = tmp_path / 'large.cube'
    filler = b'0.00000E+00 ' * 50_000
    path.write_bytes(STAMP.encode() + b'\n' + filler if where == 'start'
                     else filler + b'\n' + STAMP.encode() + b'\n')
    assert outputs.file_stamp(path) == STAMP
    # Not in the middle of a file larger than what is read at its start and its end.
    path.write_bytes(filler + STAMP.encode() + filler)
    assert outputs.file_stamp(path) is None


def test_cube_file(tmp_path):
    structure = Structure.from_dict(interface(-1.0, 1.0))
    grid = np.arange(2 * 3 * 7, dtype=float).reshape(2, 3, 7)
    chgcar = Chgcar(structure, {'total': grid * structure.volume})
    path = tmp_path / 'a.cube'
    outputs.write_cube(path, chgcar, 'comment')
    lines = path.read_text().splitlines()
    assert lines[1] == 'comment' and lines[2].split()[0] == '2'
    assert [int(line.split()[0]) for line in lines[3:6]] == [2, 3, 7]
    assert float(lines[5].split()[3]) == pytest.approx(20 / 7 / outputs.BOHR)
    assert lines[6].split()[0] == '26'
    # Each column of z on lines of six values, a new line for the next column.
    values = lines[8:]
    assert len(values) == 2 * 3 * 2 and len(values[0].split()) == 6 and len(values[1].split()) == 1
    numbers = np.array([float(w) for line in values for w in line.split()])
    assert np.allclose(numbers, grid.ravel() * outputs.BOHR ** 3, rtol=1e-5)
