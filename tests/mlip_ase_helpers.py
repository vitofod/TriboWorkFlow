# Author: Vito Foderà
# Created: 2026-11-14

"""Helpers for the tests of the ASE executor and of its job and results.

No MLIP is installed or needed: the "model" is ASE's EMT, a classical potential for a few
metals, created from the same ``{'factory', 'args', 'kwargs'}`` call that describes a real model
(``ase_run.load_calculator``). EMT lives in the tests only, never in the package.
"""

import json
import time

import numpy as np
from ase.build import fcc111
from ase.calculators.emt import EMT
from pymatgen.core import Lattice, Structure

from triboworkflow.core.structures import extxyz_text
from triboworkflow.engines.mlip import ase_run

EMT_CALL = {'factory': 'ase.calculators.emt:EMT', 'args': [], 'kwargs': {}}
FAKE_DEVICE = {'torch': '0.0-test', 'device': 'cuda:0 test GPU'}


def helper_call(name, **kwargs):
    """Call that creates the calculator made by the factory ``name`` of this module."""
    return {'factory': f'mlip_ase_helpers:{name}', 'args': [], 'kwargs': kwargs}


class SlowEMT(EMT):
    """EMT that takes ``delay`` seconds per calculation, to run out of time."""

    def __init__(self, delay=0.05, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay

    def calculate(self, *args, **kwargs):
        time.sleep(self.delay)
        super().calculate(*args, **kwargs)


def slow_emt(delay=0.05):
    """Factory of ``SlowEMT`` (the call ``{'factory': 'mlip_ase_helpers:slow_emt', ...}``)."""
    return SlowEMT(delay=delay)


def slab(if_pos=None, rattle=0.08, seed=1, interface_region=True):
    """Cu(111) slab of 2x2x3 atoms with the atoms displaced by a seeded random amount.

    Parameters
    ----------
    if_pos : list of list of int, optional
        ``if_pos`` flags of the atoms (``0`` fixes a component): the constraints of the input.
    rattle : float, optional
        Standard deviation of the displacement of every coordinate, in angstrom.
    interface_region : bool, optional
        Add the site property ``interface_region`` (True for the atoms of the top layer).

    Returns
    -------
    pymatgen.core.structure.Structure
        Labels ``Cu1``..: the label does not change the element the model sees.
    """
    atoms = fcc111('Cu', size=(2, 2, 3), vacuum=6.0, periodic=True)
    positions = atoms.get_positions() + np.random.default_rng(seed).normal(0.0, rattle, (len(atoms), 3))
    structure = Structure(Lattice(atoms.cell[:]), ['Cu'] * len(atoms), positions,
                          coords_are_cartesian=True, labels=['Cu1'] * len(atoms))
    if if_pos is not None:
        structure.add_site_property('if_pos', if_pos)
    if interface_region:
        structure.add_site_property('interface_region', [z > 10.0 for z in positions[:, 2]])
    return structure


def write_job(directory, structure, calculation='relax', model=None, fmax=0.01, maxsteps=200,
              max_seconds=None, package='ase', model_path='/models/test.model'):
    """Write ``structure_in.extxyz`` and ``job.json`` of a calculation; return the job path."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ase_run.STRUCTURE_FILE).write_text(extxyz_text(structure))
    job = {'calculation': calculation, 'model': model or EMT_CALL, 'package': package,
           'model_path': model_path, 'fmax': fmax, 'maxsteps': maxsteps,
           'max_seconds': max_seconds}
    path = directory / 'job.json'
    path.write_text(json.dumps(job))
    return path


def run(directory, structure, **job):
    """Write a job and run the executor in this process; return ``result.json``.

    The test module fakes ``ase_run.check_cuda`` (``fake_cuda``): there is no GPU here.
    """
    path = write_job(directory, structure, **job)
    code = ase_run.main([str(path)])
    result = json.loads((directory / ase_run.RESULT_FILE).read_text())
    assert (code == 0) == result['ok']
    return result


def nan_emt():
    """Factory of an EMT whose energy is NaN."""
    class NanEMT(EMT):
        def calculate(self, *args, **kwargs):
            super().calculate(*args, **kwargs)
            self.results['energy'] = float('nan')
    return NanEMT()
