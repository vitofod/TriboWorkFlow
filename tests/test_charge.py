# Author: Vito Foderà
# Created: 2026-11-09

"""Tests of the step charge (``triboworkflow.workflows.interface.charge``).

The example run is matched, its PES are analysed on synthetic energies and its
adhesion is analysed for each of the three variants of ``[interface.adhesion]``
(``test_adhesion``); the tests check what the step reads from that document and
the calculations it makes (the interface relaxed by the adhesion and its two
slabs, each with its charge density), and the charge displacement it writes from
densities that the tests make: a Gaussian sheet of charge for every atom, in the
format of the file that pp.x writes (``charge_files``). The two slabs of the
interface are told by the slab each atom belongs to (``interface_region``),
checked against the input of the adhesion that gave the interface, and not by
the sign of z.

Notes
-----
To remove: the golden cases ``charge_step_*`` (``tests/golden_cases.py``) compare
the step with TribChem's ``FT_StartChargeDisplacement``,
``ElectronicsWF.charge_displacement``, ``FT_MoveTagResults``,
``FT_AverageCharge`` and ``FT_ChargeDisplacement``: structures, k-points,
database writes and numbers. Here ``original_inputs`` repeats the original
sequence of the vendored functions (``fix_interface``, ``estimate_interface``,
``generate_chargedisp_inputs``) to show that where it was right the step makes
the same inputs, and ``position_decomposition`` shows where it was not (A92).
"""

import contextlib
import copy
import dataclasses
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from monty.json import jsanitize
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from scipy import integrate

import charge_files
from cli_helpers import SYSTEM, complete, fake_slurm  # noqa: F401 (fixture)
from test_adhesion import (NO_RESULT, RELAX_DONE, RESULT_KEYS, adhesion_run,  # noqa: F401
                           analysed, analysed_copy, configure, finish_adhesion, matched,
                           short_surface, structure_of, surfaces, surfaces_copy)
from test_pes import document, edit_config, finish, pes_run, position_flags, status_of
from test_ppes import (adapter_pair, different_species, in_reverse_order,  # noqa: F401 (fixture)
                       put_interface, slabs_of, thicker_slab, with_slabs,
                       with_the_slab_of_two_atoms_swapped, without_slabs, without_the_last_atom)
from triboworkflow.cli.run import build_orchestrator, load_run
from triboworkflow.core.settings import SettingsError
from triboworkflow.core.store import (Attempt, AttemptFile, DensityRun, PrerequisiteError, Status,
                                      StoreError, SystemWriter, check_prerequisites, file_digest,
                                      read_record, read_system, record_path, write_record)
from triboworkflow.core.store.legacy import LegacyStore
from triboworkflow.core.structures import read_structure
from triboworkflow.engines.qe import density
from triboworkflow.execution.layout import calc_dir, system_dir
from triboworkflow.vendored.highput.utils.manipulate_struct import (ManipulateStruct,
                                                                    generate_chargedisp_inputs)
from triboworkflow.vendored.physics.electronics.charge import chgcar_integrator as ci
from triboworkflow.workflows.interface import adhesion, charge, pes, ppes

PARTS = ['interface', 'bot_slab', 'top_slab']
LABELS = [f'min_site_{part}' for part in PARTS]
CALC_IDS = [f'charge/{label}' for label in LABELS]
# The density of the tests: n1 x n2 x n3 points, a Gaussian sheet of charge for each atom.
GRID = (4, 4, 80)
VALENCE = {'Cu': 11.0, 'Al': 3.0}
SIGMA = 0.6


def inputs_of(run_dir):
    return charge.charge_inputs(run_dir, SYSTEM, load_run(run_dir))


def interface_of(run_dir, kind, site='min_site'):
    """The interface the step starts from, as the document holds it."""
    return document(run_dir)['structure']['opt'][kind][ppes.INTERFACE_OF[site]]


def with_pp_command(run_dir):
    """The example asks for pp.x in its comments: the calculations of the charge need it."""
    edit_config(run_dir, '# pp_command = "mpirun -np 32 pp.x"', 'pp_command = "mpirun -np 32 pp.x"')
    return run_dir


def parts_of(inputs):
    """``{part: structure}`` of the calculations of the inputs."""
    return {part: structure for _, part, structure, _ in inputs.points}


def quiet(function, *args):
    """The result of a vendored function, which prints and warns about what the workflow does."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return function(*args)


def original_inputs(run_dir, kind, site='min_site'):
    """What TribChem makes of the interface: the sequence of ``ElectronicsWF.charge_displacement``.

    ``fix_interface``, the range of ``estimate_interface`` moved by half a cell and
    ``generate_chargedisp_inputs``, with the vendored code.

    Returns
    -------
    structures, kpoints, integrate_in
        Interface, bottom slab and top slab, their k-points and the range of integration.
    """
    interface = structure_of(interface_of(run_dir, kind, site))
    comp_params = {'kdens': load_run(run_dir).k_points.kdens}

    def build():
        fixed = ManipulateStruct(interface).fix_interface(interface)
        low, up = ManipulateStruct(fixed).estimate_interface('zero', 2)
        slabs, vis, _ = generate_chargedisp_inputs(fixed, comp_params, 'tag', 'zero',
                                                   'slab_from_scratch')
        return slabs, [v.kpoints for v in vis], [low + interface.lattice.c / 2,
                                                 up + interface.lattice.c / 2]
    return quiet(build)


def position_decomposition(structure):
    """The two slabs the original tells by the sign of z, once ``fix_interface`` has run.

    Returns
    -------
    bottom, top : pymatgen.core.structure.Structure
    """
    def decompose():
        fixed = ManipulateStruct(structure).fix_interface(structure)
        return ManipulateStruct(fixed).decompose_structure('zero')
    return quiet(decompose)


# --- the step and its entries ---------------------------------------------------------------

def test_step_and_constants():
    assert charge.ChargeStep.name == charge.STEP == 'charge'
    # What the command line of the original workflow fixes (A93: 1000 and not the 10000 of the
    # signatures of the functions).
    assert charge.CALC_TYPE == 'slab_from_scratch' and charge.INTER_TYPE == 'zero'
    assert charge.AXIS == 2 and charge.RESCALE == 1000
    assert charge.LABELS == tuple(PARTS)
    # The slab of each atom is the property that match gives to the atoms of an interface.
    assert (charge.SLAB_PROPERTY, charge.TOP_SLAB, charge.BOTTOM_SLAB) == (
        'interface_region', 'top_slab', 'bot_slab')
    assert charge.INTERFACE_OF == {'min_site': 'interface_min', 'max_site': 'interface_max'}


@pytest.mark.parametrize('kind', ['regular', 'short'])
@pytest.mark.parametrize('site', ['min_site', 'max_site'])
def test_entries_of_each_kind_and_site(kind, site):
    assert charge.interface_entry(kind, site) == ppes.interface_entry(kind, site)
    data = ('charge', kind, 'data', site)
    average = ('charge', kind, 'average', site)
    displacement = ('charge', kind, 'displacement', site)
    assert charge.data_entry(kind, site) == data
    assert charge.average_entry(kind, site) == average
    assert charge.displacement_entry(kind, site) == displacement
    # The three entries that the step writes, each with the provenance of the step.
    assert charge.charge_leaves(kind, site) == [data, average, displacement]


def test_label_of_a_calculation():
    assert [charge.charge_label('min_site', part) for part in PARTS] == LABELS
    assert charge.charge_label('max_site', 'bot_slab') == 'max_site_bot_slab'


def test_keys_moved_from_the_task_of_a_calculation():
    entry_from, entry_to = charge.transfer_entries('regular', 'min_site', 'bot_slab')
    assert len(entry_from) == len(entry_to) == 15
    assert entry_from[0] == ['nsites'] and entry_from[2] == ['output', 'energy']
    assert entry_from[-1] == ['output', 'structure']
    # No level 'output' in the entries of the charge: the key is the last part of the origin.
    base = ['charge', 'regular', 'data', 'min_site', 'bot_slab']
    assert entry_to[0] == base + ['nsites'] and entry_to[2] == base + ['energy']
    assert entry_to[-1] == base + ['structure']
    assert [path[-1] for path in entry_to[:-1]] == [
        'nsites', 'density', 'energy', 'energy_per_atom', 'bandgap', 'cbm', 'vbm', 'is_gap_direct',
        'forces', 'stress', 'pseudo_potential', 'dir_name', 'task_id', 'task_label']


# --- inputs -----------------------------------------------------------------------------------

def test_inputs_read_from_the_document(analysed):
    run_dir, _, pes_name, kind = analysed
    inputs = inputs_of(run_dir)
    assert (inputs.type, inputs.site, inputs.operations) == (kind, 'min_site', ['-', '-'])
    # The interface of the adhesion at the lowest site, as the document holds it.
    assert inputs.read == {'interface': interface_of(run_dir, kind)}
    doc = document(run_dir)
    assert inputs.ids == {key: doc[key] for key in ('mid', 'formula', 'name', 'miller')}
    assert inputs.parameters == {
        'type': kind, 'site': 'min_site', 'operations': ['-', '-'], 'calc_type': 'slab_from_scratch',
        'inter_type': 'zero', 'axis': 2, 'rescale': 1000,
        'comp_params': {'kdens': load_run(run_dir).k_points.kdens}}
    assert [(label, part) for label, part, *_ in inputs.points] == list(zip(LABELS, PARTS))


def test_the_site_and_the_operations_come_from_the_settings(analysed_copy):
    run_dir, _, _, kind = analysed_copy
    edit_config(run_dir, 'site = "min_site"', 'site = "max_site"')
    inputs = inputs_of(run_dir)
    assert inputs.site == 'max_site' and inputs.read == {'interface': interface_of(
        run_dir, kind, 'max_site')}
    assert [label for label, *_ in inputs.points] == [f'max_site_{part}' for part in PARTS]
    edit_config(run_dir, '[interface.charge]\n', '[interface.charge]\noperations = ["+", "-"]\n')
    assert inputs_of(run_dir).operations == ['+', '-']
    assert inputs_of(run_dir).parameters['operations'] == ['+', '-']


def test_the_entry_read_is_not_changed_by_the_preparation(analysed):
    run_dir, _, _, kind = analysed
    before = interface_of(run_dir, kind)
    inputs = inputs_of(run_dir)
    assert interface_of(run_dir, kind) == before
    assert inputs.read['interface'] == before
    # The slabs are told by the slab of each atom, by its index in the interface.
    top = slabs_of(structure_of(before))
    assert inputs.top == list(np.flatnonzero(top))
    assert inputs.bottom == list(np.flatnonzero(~top))


def test_the_interface_and_the_slabs_are_as_in_the_original_workflow(analysed):
    """Where the original told the slabs right, the structures, the range and the k-points are its."""
    run_dir, _, _, kind = analysed
    inputs = inputs_of(run_dir)
    structures, kpoints, integrate_in = original_inputs(run_dir, kind)
    assert [s.as_dict() for s in parts_of(inputs).values()] == [s.as_dict() for s in structures]
    assert [str(k) for *_, k in inputs.points] == [str(k) for k in kpoints]
    assert inputs.integrate_in == pytest.approx(integrate_in, abs=1e-12)
    # The slabs are moved by half a cell and the interface with them, none wrapped.
    interface = structure_of(interface_of(run_dir, kind))
    assert len(parts_of(inputs)['interface']) == len(interface)
    assert len(parts_of(inputs)['bot_slab']) + len(parts_of(inputs)['top_slab']) == len(interface)


def test_the_cell_is_the_same_for_the_three_calculations(analysed):
    """The densities are subtracted point by point: one cell and one grid of k-points."""
    run_dir, _, _, kind = analysed
    inputs = inputs_of(run_dir)
    cell = structure_of(interface_of(run_dir, kind)).lattice.matrix
    for _, _, structure, _ in inputs.points:
        assert np.allclose(structure.lattice.matrix, cell)
    assert len({str(kpoints) for *_, kpoints in inputs.points}) == 1


def test_the_range_of_the_interface_is_between_the_two_slabs(analysed):
    run_dir, _, _, kind = analysed
    inputs = inputs_of(run_dir)
    parts = parts_of(inputs)
    low = max(site.coords[2] for site in parts['bot_slab'])
    high = min(site.coords[2] for site in parts['top_slab'])
    assert inputs.integrate_in[0] == pytest.approx(low, abs=1e-9)
    assert inputs.integrate_in[1] == pytest.approx(high, abs=1e-9)
    assert inputs.integrate_in[0] < inputs.integrate_in[1]


def test_the_slabs_keep_the_slab_of_each_atom(analysed):
    run_dir, _, _, _ = analysed
    for part, structure in parts_of(inputs_of(run_dir)).items():
        slabs = set(structure.site_properties['interface_region'])
        assert slabs == {'bot_slab', 'top_slab'} if part == 'interface' else slabs == {part}


def test_k_points_from_their_density_only(surfaces):
    settings = load_run(surfaces)
    settings = dataclasses.replace(settings, k_points=dataclasses.replace(
        settings.k_points, mode='automatic', grid=[4, 4, 1], offset=[0, 0, 0]))
    with pytest.raises(SettingsError, match=r'k_points\.mode.*kdens'):
        charge.charge_inputs(surfaces, SYSTEM, settings)


# --- the slabs are told by the atoms each one holds, not by the sign of z (A92) ---------------

@pytest.mark.parametrize('slab', ['top_slab', 'bot_slab'])
def test_the_fragments_are_the_slabs_whatever_their_thickness(surfaces_copy, adapter_pair, slab):
    """A slab thicker than half the cell does not change which atoms are in each fragment.

    With the top slab thicker than half the cell, ``fix_interface`` brings some of its atoms to
    negative z and the sign of z puts them in the bottom fragment: the original does it without an
    error. The step takes the atoms from the slab each one carries.
    """
    run_dir = surfaces_copy
    _, final = adapter_pair
    thick = thicker_slab(final, slab)
    top = slabs_of(thick)
    put_interface(run_dir, thick)
    # The original (vendored) is wrong with the top slab thick: one fragment has atoms of the other.
    bottom_original, top_original = position_decomposition(thick)
    if slab == 'top_slab':
        assert len(top_original) != top.sum() and len(bottom_original) != (~top).sum()
    inputs = inputs_of(run_dir)
    parts = parts_of(inputs)
    assert len(parts['top_slab']) == top.sum() and len(parts['bot_slab']) == (~top).sum()
    symbols = np.array([site.specie.symbol for site in thick])
    assert {site.specie.symbol for site in parts['top_slab']} == set(symbols[top])
    assert {site.specie.symbol for site in parts['bot_slab']} == set(symbols[~top])
    assert inputs.top == list(np.flatnonzero(top)) and inputs.bottom == list(np.flatnonzero(~top))
    # The range is the original one: the atoms next to the contact plane of the two slabs.
    structures, _, integrate_in = original_inputs(run_dir, 'regular')
    assert inputs.integrate_in == pytest.approx(integrate_in, abs=1e-12)
    # Every atom of the interface is in one fragment, once, at its place in the fixed interface.
    interface = parts['interface']
    fragments = [*parts['bot_slab'], *parts['top_slab']]
    assert len(interface) == len(fragments)
    expected = [interface[i].coords for i in inputs.bottom + inputs.top]
    assert np.allclose([site.coords for site in fragments], expected)


def test_slabs_of_the_same_material_are_told_apart(surfaces_copy, adapter_pair):
    """Two slabs of one element, one of them thick: only the slab each atom carries tells them."""
    _, base = adapter_pair
    same = Structure(base.lattice, ['Cu'] * len(base), base.frac_coords,
                     site_properties={'interface_region': base.site_properties['interface_region']})
    thick = thicker_slab(same, 'top_slab')
    assert {site.specie.symbol for site in thick} == {'Cu'}
    put_interface(surfaces_copy, thick)
    top = slabs_of(thick)
    parts = parts_of(inputs_of(surfaces_copy))
    assert len(parts['top_slab']) == top.sum() and len(parts['bot_slab']) == (~top).sum()


@pytest.mark.parametrize('wrapped', [False, True], ids=['unwrapped', 'wrapped'])
def test_the_interface_with_its_slabs_at_the_usual_place_does_not_depend_on_the_wrap(
        surfaces_copy, adapter_pair, wrapped):
    """The coordinates of the interface wrapped into the cell or not give the same fragments."""
    _, final = adapter_pair
    other = final.copy()
    other.translate_sites(list(range(len(other))), [0.0, 0.0, 0.0], frac_coords=False,
                          to_unit_cell=wrapped)
    put_interface(surfaces_copy, other)
    inputs = inputs_of(surfaces_copy)
    top = slabs_of(other)
    assert inputs.top == list(np.flatnonzero(top))
    reference = parts_of(inputs)
    assert len(reference['top_slab']) == top.sum()


@pytest.mark.parametrize('wrapped', [False, True], ids=['unwrapped', 'wrapped'])
@pytest.mark.parametrize('turn', [0.5, 0.3, -0.7], ids=['half', 'third', 'below'])
def test_an_interface_with_its_contact_plane_elsewhere_stops_the_step(surfaces_copy, adapter_pair,
                                                                       turn, wrapped):
    """The range of the integration is that of an interface at z = 0: it is not guessed (A92).

    ``estimate_interface`` takes the atoms next to z = 0; with the origin of the cell moved they
    are not those of the two slabs, and the original integrates over the wrong region without an
    error. The slabs are right here and the range is not, so the step stops.
    """
    _, final = adapter_pair
    translated = final.copy()
    translated.translate_sites(list(range(len(translated))), [0.0, 0.0, turn * final.lattice.c],
                               frac_coords=False, to_unit_cell=wrapped)
    put_interface(surfaces_copy, translated, given=adapter_pair[0])
    with pytest.raises(StoreError, match=r'contact plane.*z = 0') as error:
        inputs_of(surfaces_copy)
    assert not isinstance(error.value, PrerequisiteError)


def slabs_around_the_plane(z_bottom, z_top, c=40.0):
    """A cell of two slabs of one atom per layer, with the z (angstrom) of each layer.

    Returns
    -------
    fixed, top, bottom : Structure, list of int, list of int
        The bottom slab is the first atoms. The z are the Cartesian ones, as they are once
        the interface is brought around z = 0.
    """
    z = [*z_bottom, *z_top]
    fixed = Structure(Lattice.orthorhombic(3.0, 3.0, c), ['Cu'] * len(z),
                      [[0.0, 0.0, value / c] for value in z])
    return (fixed, list(range(len(z_bottom), len(z))), list(range(len(z_bottom))))


@pytest.mark.parametrize('z_bottom, z_top, atoms', [
    ([-4.0, -2.0], [0.0, 2.0, 4.0], '2'),
    ([-4.0, 0.0], [1.0, 2.0, 4.0], '1'),
    ([-4.0, -2.0, 0.0], [0.0, 2.0], '2, 3'),
], ids=['top-slab-atom', 'bottom-slab-atom', 'both'])
def test_an_atom_at_z_zero_is_a_problem_of_the_contact_plane(z_bottom, z_top, atoms):
    """The atoms next to the plane would be those of the slabs and the atom on it is in the range.

    The slab at -4, -2 and the other at 0, 2, 4: the highest atom below z = 0 and the lowest above
    it are of the bottom and of the top slab, and the range of the integration, from -2 to 2,
    would hold an atom of the top slab; the atom at 0 is on neither side. Whichever slab it is of,
    the plane has to be told.
    """
    fixed, top, bottom = slabs_around_the_plane(z_bottom, z_top)
    assert (np.array([site.coords[2] for site in fixed]) == 0).sum() == len(atoms.split(','))
    problem = charge.contact_plane_problem(fixed, top, bottom)
    assert problem is not None
    assert f'passes through the atoms {atoms}' in problem and 'neither side' in problem
    # The original stops on it too, when it tells the slabs by the sign of z.
    with pytest.raises(ValueError, match='One atom found with z=0'):
        quiet(lambda: ManipulateStruct(fixed).decompose_structure('zero'))


def test_the_atoms_next_to_the_plane_are_told_when_none_is_on_it():
    fixed, top, bottom = slabs_around_the_plane([-4.0, -2.0], [1.0, 2.0, 4.0])
    assert charge.contact_plane_problem(fixed, top, bottom) is None


@pytest.mark.parametrize('slab', ['bot_slab', 'top_slab'])
def test_an_atom_on_the_contact_plane_stops_the_step(surfaces_copy, adapter_pair, slab):
    """An atom of the interface at z = 0 stops the step, as it stops the original decomposition.

    The atom of the slab that lies next to the plane is put on it, with the others where they
    are: the atoms next to the plane are still those of the two slabs, so the check of the slabs
    next to it passes, and the range of the integration would hold that atom.
    """
    structure_in, final = adapter_pair
    fixed = quiet(lambda: ManipulateStruct(final).fix_interface(final))
    z = np.array([site.coords[2] for site in fixed])
    in_slab = np.array(final.site_properties['interface_region']) == slab
    chosen = np.flatnonzero(in_slab)[np.argmin(np.abs(z[in_slab]))]
    # The cell of the interface has some 1e-15 angstrom in the z of a and b, which would keep the
    # atom a hair off z = 0: the same cell without them.
    matrix = final.lattice.matrix.copy()
    matrix[:2, 2] = 0.0
    frac = final.frac_coords.copy()
    frac[chosen, 2] = 0.0
    on_plane = Structure(Lattice(matrix), final.species, frac,
                         site_properties=final.site_properties)
    refixed = quiet(lambda: ManipulateStruct(on_plane).fix_interface(on_plane))
    assert [i for i, site in enumerate(refixed) if site.coords[2] == 0] == [chosen]
    with pytest.raises(ValueError, match='One atom found with z=0'):
        position_decomposition(on_plane)
    put_interface(surfaces_copy, on_plane, given=structure_in)
    with pytest.raises(StoreError, match=rf'contact plane.*passes through the atoms {chosen}, '
                                         r'on neither side of it') as error:
        inputs_of(surfaces_copy)
    assert not isinstance(error.value, PrerequisiteError)


# --- the slab of each atom must be there and be the one of the input (D58) -------------------

def test_without_the_slab_of_the_atoms_the_step_stops(surfaces_copy, adapter_pair):
    structure_in, final = adapter_pair
    put_interface(surfaces_copy, without_slabs(final), given=structure_in)
    with pytest.raises(PrerequisiteError, match=r'structure\.opt\.regular\.interface_min.*'
                                                r'interface_region.*run adhesion again'):
        inputs_of(surfaces_copy)


def test_without_the_slab_of_the_atoms_of_the_input_the_step_stops(surfaces_copy, adapter_pair):
    structure_in, final = adapter_pair
    put_interface(surfaces_copy, final, given=without_slabs(structure_in))
    with pytest.raises(PrerequisiteError, match=r'input of adhesion/interface_min.*'
                                                r'interface_region'):
        inputs_of(surfaces_copy)


def test_a_slab_that_is_neither_top_nor_bottom_stops_the_step(surfaces_copy, adapter_pair):
    structure_in, final = adapter_pair
    slabs = list(final.site_properties['interface_region'])
    slabs[2] = 'middle_slab'
    put_interface(surfaces_copy, with_slabs(final, slabs), given=structure_in)
    with pytest.raises(StoreError, match=r"atom 2.*'middle_slab'.*'bot_slab' or 'top_slab'") as error:
        inputs_of(surfaces_copy)
    assert not isinstance(error.value, PrerequisiteError)


@pytest.mark.parametrize('only', ['bot_slab', 'top_slab'])
def test_an_interface_with_one_slab_only_stops_the_step(surfaces_copy, adapter_pair, only):
    structure_in, final = adapter_pair
    slabs = [only] * len(final)
    put_interface(surfaces_copy, with_slabs(final, slabs), given=with_slabs(structure_in, slabs))
    other = 'top_slab' if only == 'bot_slab' else 'bot_slab'
    with pytest.raises(StoreError, match=f"no atom.*'{other}'"):
        inputs_of(surfaces_copy)


def test_without_the_record_of_the_adhesion_the_step_stops(surfaces_copy, adapter_pair):
    structure_in, final = adapter_pair
    put_interface(surfaces_copy, final, given=structure_in)
    record_path(system_dir(surfaces_copy, SYSTEM), 'adhesion/interface_min').unlink()
    with pytest.raises(PrerequisiteError, match=r'record adhesion/interface_min.*missing.*'
                                                r'run adhesion again'):
        inputs_of(surfaces_copy)


@pytest.mark.parametrize('change, message', [
    (without_the_last_atom, r'has \d+ atoms and the input of adhesion/interface_min has \d+'),
    (different_species, r'species.*differ.*input of adhesion/interface_min'),
    (in_reverse_order, r'species.*differ.*input of adhesion/interface_min'),
    (with_the_slab_of_two_atoms_swapped, r"atom \d+ is of the slab '\w+' in the interface and of "
                                         r"the slab '\w+' in the input of adhesion/interface_min")])
def test_an_interface_that_does_not_match_its_input_stops_the_step(surfaces_copy, adapter_pair,
                                                                   change, message):
    given, interface = change(*adapter_pair)
    put_interface(surfaces_copy, interface, given=given)
    with pytest.raises(StoreError, match=message) as error:
        inputs_of(surfaces_copy)
    assert not isinstance(error.value, PrerequisiteError)


def test_without_match_the_step_stops(tmp_path):
    run_dir = adhesion_run(tmp_path)
    with pytest.raises(PrerequisiteError, match='run match first'):
        charge.ChargeStep().calculations(run_dir, SYSTEM)


def test_without_the_adhesion_the_step_stops(surfaces):
    with pytest.raises(PrerequisiteError, match=re.escape(
            'entries missing: structure.opt.regular.interface_min; run adhesion first')):
        inputs_of(surfaces)


def test_without_the_interface_of_the_other_site_the_step_stops(analysed_copy):
    run_dir, _, _, kind = analysed_copy
    edit_config(run_dir, 'site = "min_site"', 'site = "max_site"')
    with SystemWriter(system_dir(run_dir, SYSTEM)) as writer:
        writer.document.replace(('structure', 'opt', kind, 'interface_max'), {})
        writer.save()
    with pytest.raises(PrerequisiteError, match=rf'structure\.opt\.{kind}\.interface_max'):
        inputs_of(run_dir)
    edit_config(run_dir, 'site = "max_site"', 'site = "min_site"')
    assert inputs_of(run_dir).site == 'min_site'


# --- calculations -----------------------------------------------------------------------------

def test_three_static_calculations_with_the_charge_density(analysed_copy):
    run_dir, _, _, _ = analysed_copy
    with_pp_command(run_dir)
    inputs = inputs_of(run_dir)
    calcs = charge.charge_calculations(SYSTEM, inputs)
    assert [calc.calc_id for calc in calcs] == CALC_IDS
    assert [calc.calc_id for calc in charge.ChargeStep().calculations(run_dir, SYSTEM)] == CALC_IDS
    for calc, label in zip(calcs, LABELS):
        assert calc.record.calc_type == 'slab_from_scratch' and calc.record.density is True
        assert calc.record.step == 'charge' and calc.record.system == SYSTEM
        assert calc.record.label == label
        assert re.search(r"calculation\s*=\s*'scf'", calc.input_text)
        assert re.search(rf"prefix\s*=\s*'charge_{label}'", calc.input_text)
        # pp.x runs after pw.x on the density that pw.x leaves.
        assert 'plot_num' in calc.pp_input_text and "filplot" in calc.pp_input_text
        assert f"prefix           = 'charge_{label}'" in calc.pp_input_text
        # No atom is fixed: the flags of the relaxed interface do not reach a static one.
        assert set(map(str, position_flags(calc.input_text))) == {'None'}


def test_the_density_needs_the_command_of_pp(analysed_copy):
    run_dir, _, _, _ = analysed_copy
    with pytest.raises(SettingsError, match=r'execution\.pp_command.*charge/min_site_interface'):
        charge.ChargeStep().calculations(run_dir, SYSTEM)


@pytest.mark.parametrize('disk_io', ['nowf', 'none'])
def test_the_density_needs_a_disk_io_that_writes_it(analysed_copy, disk_io):
    run_dir, _, _, _ = analysed_copy
    with_pp_command(run_dir)
    edit_config(run_dir, 'disk_io = "low"', f'disk_io = "{disk_io}"')
    with pytest.raises(SettingsError, match=r'disk_io'):
        charge.ChargeStep().calculations(run_dir, SYSTEM)


def test_the_fingerprints_follow_the_structures_and_not_the_operations(analysed_copy):
    run_dir, _, _, _ = analysed_copy
    with_pp_command(run_dir)
    plan = charge.ChargeStep().plan(run_dir, SYSTEM)
    assert list(plan) == CALC_IDS and len(set(plan.values())) == 3
    # The algebra of the densities is an analysis (P4): the same calculations.
    edit_config(run_dir, '[interface.charge]\n', '[interface.charge]\noperations = ["+", "+"]\n')
    assert charge.ChargeStep().plan(run_dir, SYSTEM) == plan


def test_the_site_changes_the_calculations(analysed_copy):
    run_dir, _, _, _ = analysed_copy
    with_pp_command(run_dir)
    plan = charge.ChargeStep().plan(run_dir, SYSTEM)
    edit_config(run_dir, 'site = "min_site"', 'site = "max_site"')
    other = charge.ChargeStep().plan(run_dir, SYSTEM)
    assert set(other) == {calc_id.replace('min_site', 'max_site') for calc_id in plan}
    # The two interfaces differ, so do the calculations of the interface and of the top slab (the
    # bottom slab may be the same atoms at both sites).
    assert other['charge/max_site_interface'] != plan['charge/min_site_interface']
    assert other['charge/max_site_top_slab'] != plan['charge/min_site_top_slab']


def test_other_slabs_give_other_calculations(analysed_copy):
    """The calculations follow the atoms of each fragment: results of other atoms are not reused."""
    run_dir, _, _, kind = analysed_copy
    with_pp_command(run_dir)
    plan = charge.ChargeStep().plan(run_dir, SYSTEM)
    inputs = inputs_of(run_dir)
    # The atom of each slab that is farthest from the contact plane changes slab: the interface
    # is the same structure, the two fragments are not.
    interface = parts_of(inputs)['interface']
    far_bottom = min(inputs.bottom, key=lambda i: interface[i].coords[2])
    far_top = max(inputs.top, key=lambda i: interface[i].coords[2])
    slabs = list(structure_of(interface_of(run_dir, kind)).site_properties['interface_region'])
    slabs[far_bottom], slabs[far_top] = 'top_slab', 'bot_slab'
    put_interface(run_dir, with_slabs(structure_of(interface_of(run_dir, kind)), slabs), kind=kind)
    other = charge.ChargeStep().plan(run_dir, SYSTEM)
    assert list(other) == list(plan)
    assert other[CALC_IDS[0]] == plan[CALC_IDS[0]]
    assert other[CALC_IDS[1]] != plan[CALC_IDS[1]] and other[CALC_IDS[2]] != plan[CALC_IDS[2]]


# --- what the step reads is current along the whole chain (P1) ---------------------------------

def test_expected_is_the_interface_with_the_provenance_of_the_adhesion(analysed):
    run_dir, _, _, kind = analysed
    assert charge.ChargeStep().expected(run_dir, SYSTEM) == {
        charge.interface_entry(kind, 'min_site'):
            adhesion.AdhesionStep().provenance(run_dir, SYSTEM)}


def test_expected_follows_the_site(analysed_copy):
    run_dir, _, _, kind = analysed_copy
    edit_config(run_dir, 'site = "min_site"', 'site = "max_site"')
    assert list(charge.ChargeStep().expected(run_dir, SYSTEM)) == [
        ('structure', 'opt', kind, 'interface_max')]


def test_match_changed_after_its_analysis_stops_the_step(analysed_copy):
    run_dir, _, pes_name, _ = analysed_copy
    edit_config(run_dir, 'max_area = 160.0', 'max_area = 150.0')
    with pytest.raises(PrerequisiteError, match=f'upstream of adhesion not done or not current: '
                                                f'match, {pes_name}; run them first'):
        charge.ChargeStep().expected(run_dir, SYSTEM)


# --- the densities of the tests ----------------------------------------------------------------

def sheets(structure, grid=GRID, transfer=None):
    """Toy density (e/A^3), shape ``(n3, n2, n1)``: a Gaussian sheet for each atom.

    The sheet of an atom is uniform in the plane, a Gaussian of ``SIGMA`` along the third vector
    of the cell, and integrates to the valence of its element; ``transfer`` is the height, width
    and electrons of one more sheet.
    """
    n1, n2, n3 = grid
    c, volume = structure.lattice.c, structure.volume
    area = volume / c
    along = np.arange(n3) / n3

    def sheet(position, sigma, electrons):
        distance = ((along - position + 0.5) % 1.0 - 0.5) * c
        g = np.exp(-0.5 * (distance / sigma) ** 2)
        return electrons * g / (g.sum() * c / n3) / area

    profile = np.zeros(n3)
    for site in structure:
        profile += sheet(site.frac_coords[2], SIGMA, VALENCE[site.specie.symbol])
    if transfer is not None:
        position, sigma, electrons = transfer
        profile += sheet(position, sigma, electrons) - electrons / volume
    return np.broadcast_to(profile[:, None, None], (n3, n2, n1)).copy()


def filplot_of(structure, grid=GRID, transfer=None, **kwargs):
    """Text of the file that pp.x writes for a calculation: the toy density of its structure."""
    rho = sheets(structure, grid, transfer) * charge_files.BOHR_TO_ANGSTROM ** 3
    return charge_files.structure_filplot(structure, grid=grid, values=rho, **kwargs)


def finish_charge(run_dir, prepare=True, grid=GRID, **changes):
    """Prepare the calculations of the step and complete them with synthetic results.

    The structure of each is its input (a ``scf``), the file of the density is the toy density of
    its atoms, with a sheet of transferred charge at the contact plane in the interface, and the
    record is that of a calculation whose pp.x ran well. ``changes`` are fields of the records.

    Returns
    -------
    list of str
        ``calc_id`` of the calculations, in the order of the labels.
    """
    step = charge.ChargeStep()
    inputs = step.inputs(run_dir, SYSTEM)
    if prepare:
        step.prepare(run_dir, SYSTEM)
    sdir = system_dir(run_dir, SYSTEM)
    for label, part, structure, _ in inputs.points:
        calc_id = f'charge/{label}'
        path = calc_dir(run_dir, SYSTEM, calc_id)
        shutil.copy(path / 'structure_in.extxyz', path / 'structure_out.extxyz')
        final = read_structure(path / 'structure_out.extxyz', format='extxyz')
        # The contact plane is in the middle of the cell, since the step moves everything by half
        # a cell: the interface has 0.4 electrons more there than the slabs.
        transfer = (0.5, 0.5, 0.4) if part == 'interface' else None
        (path / 'density.filplot').write_text(filplot_of(final, grid, transfer))
        energy = -1000.0 - 0.1 * len(structure)
        results = {'nsites': len(structure), 'input': {'structure': structure.as_dict()},
                   'output': {'structure': structure.as_dict(), 'density': float(structure.density),
                              'energy': energy, 'energy_per_atom': energy / len(structure)}}
        attempt = Attempt(
            1, input=AttemptFile('run.pwi'), output=AttemptFile('output.pwo'), exit_code=0,
            status=Status('ended', 'converged', 'n/a', 'ok', 'ok'),
            density=DensityRun('pp.x', command='mpirun -np 32 pp.x -in input.ppi',
                               density=AttemptFile('density.filplot',
                                                   file_digest(path / 'density.filplot')),
                               exit_code=0))
        record = read_record(sdir, calc_id)
        write_record(sdir, dataclasses.replace(
            record, status=attempt.status, attempts=(attempt,),
            results=jsanitize(results, strict=False), observed={}, **changes))
    return [f'charge/{label}' for label, *_ in inputs.points]


def charged(run_dir):
    """The run with the pp.x command, the three calculations done and the step analysed."""
    with_pp_command(run_dir)
    calc_ids = finish_charge(run_dir)
    charge.ChargeStep().analyse(run_dir, SYSTEM, calc_ids)
    return calc_ids


@pytest.fixture(scope='module')
def done_run(analysed, tmp_path_factory):
    """Each variant of the adhesion with the step run: run folder, calculations, surface, kind."""
    run_dir, _, pes_name, kind = analysed
    copied = tmp_path_factory.mktemp('charge') / 'run'
    shutil.copytree(run_dir, copied)
    return copied, charged(copied), pes_name, kind


@pytest.fixture
def done_copy(done_run, tmp_path):
    copied = tmp_path / 'done'
    shutil.copytree(done_run[0], copied)
    return (copied, *done_run[1:])


def charge_entry(run_dir, kind, part='data', site='min_site'):
    return document(run_dir)['charge'][kind][part][site]


def chgcars_of(run_dir, calc_ids=CALC_IDS):
    """The three densities as ``Chgcar`` objects, read from the files of the calculations."""
    chgcars = []
    for calc_id in calc_ids:
        path = calc_dir(run_dir, SYSTEM, calc_id)
        chgcars.append(density.to_chgcar(
            density.read_filplot(path / 'density.filplot'),
            read_structure(path / 'structure_out.extxyz', format='extxyz')))
    return chgcars


# --- analysis ---------------------------------------------------------------------------------

def test_the_results_of_each_calculation_are_moved_into_the_document(done_run):
    run_dir, _, _, kind = done_run
    data = charge_entry(run_dir, kind)
    assert list(data) == PARTS
    for part, label in zip(PARTS, LABELS):
        # The keys of every calculation, None where it gives nothing, then the final structure
        # (and no input structure: the original does not copy it for the charge).
        assert set(data[part]) == RESULT_KEYS | {'structure'}
        assert all(data[part][key] is None for key in NO_RESULT)
        assert data[part]['task_label'] == data[part]['task_id'] == f'charge/{label}'
        assert data[part]['dir_name'] == f'charge/{label}'
        assert data[part]['structure']['@class'] == 'Structure'
        assert data[part]['nsites'] == len(parts_of(inputs_of(run_dir))[part])
        assert data[part]['energy'] == pytest.approx(-1000.0 - 0.1 * data[part]['nsites'])


def test_the_planar_averages_of_each_density(done_run):
    """``average.<site>.<part>.<axis>``: ``[x, density]`` in angstrom and e/A^3, z from the contact plane."""
    run_dir, _, _, kind = done_run
    average = charge_entry(run_dir, kind, 'average')
    chgcars = chgcars_of(run_dir)
    c_2 = interface_of(run_dir, kind)['lattice']['c'] / 2
    assert list(average) == PARTS
    for part, chgcar in zip(PARTS, chgcars):
        assert list(average[part]) == ['x', 'y', 'z']
        for axis, name in enumerate('xyz'):
            x, y = ci.chgcar_average_along_axis(chgcar, axis)
            if axis == 2:
                x = np.asarray(x) - c_2
            assert average[part][name][0] == pytest.approx(list(x), abs=1e-12)
            assert average[part][name][1] == pytest.approx(list(y), rel=1e-12)
            assert len(average[part][name][0]) == len(average[part][name][1]) == GRID[axis]


def test_the_charge_displacement(done_run):
    """``displacement.<site>.z``: the integral of the density of interface - bottom - top."""
    run_dir, _, _, kind = done_run
    displacement = charge_entry(run_dir, kind, 'displacement')
    assert list(displacement) == ['z']
    z = displacement['z']
    assert set(z) == {'all', 'interface', 'grid', 'rescale'}
    assert z['rescale'] == 1000 and type(z['rescale']) is int
    chgcars = chgcars_of(run_dir)
    combined = ci.chgcar_algebra_from_list(chgcars, ['-', '-'], 2)
    inputs = inputs_of(run_dir)
    integral, width, curve, _ = ci.charge_displacement(combined, 2, *inputs.integrate_in, 1000)
    total, total_width, _, _ = ci.charge_displacement(combined, 2, rescale=1000)
    assert z['interface'] == {'integral': pytest.approx(integral, rel=1e-12),
                              'width': pytest.approx(width, rel=1e-12)}
    assert z['all'] == {'integral': pytest.approx(total, rel=1e-12),
                        'width': pytest.approx(total_width, rel=1e-12)}
    c_2 = interface_of(run_dir, kind)['lattice']['c'] / 2
    assert z['grid'][0] == pytest.approx(list(np.asarray(curve[0]) - c_2), abs=1e-12)
    assert z['grid'][1] == pytest.approx(list(curve[1]), rel=1e-12)
    assert all(type(z[key][name]) is float for key in ('all', 'interface')
               for name in ('integral', 'width'))
    assert z['interface']['width'] == pytest.approx(inputs.integrate_in[1] - inputs.integrate_in[0])


def test_the_displacement_is_that_of_the_densities(done_run):
    """The number from the arrays of the densities alone, without ``Chgcar`` or the vendored module.

    The planar average is the mean over x and y in e/A^3; the curve is interpolated at 1000 points
    between the sites next to the contact plane, in absolute value, and integrated with Simpson's
    rule over the width.
    """
    run_dir, _, _, kind = done_run
    inputs = inputs_of(run_dir)
    rhos = []
    for calc_id in CALC_IDS:
        path = calc_dir(run_dir, SYSTEM, calc_id)
        rhos.append(density.read_filplot(path / 'density.filplot').values / charge_files.BOHR_TO_ANGSTROM ** 3)
    difference = (rhos[0] - rhos[1] - rhos[2]).mean(axis=(1, 2))  # z index first
    c = inputs.read['interface']['lattice']['c']
    x = np.linspace(0.0, c, len(difference))
    p1, p2 = inputs.integrate_in
    grid = np.linspace(p1, p2, 1000)
    expected = integrate.simpson(np.abs(np.interp(grid, x, difference)), x=grid) / (p2 - p1)
    shown = charge_entry(run_dir, kind, 'displacement')['z']['interface']
    assert shown['integral'] == pytest.approx(expected, rel=1e-9)
    assert shown['integral'] > 0
    # The charge moved to the contact plane is the only difference between the three densities.
    assert abs(difference.mean()) < 1e-8 * abs(rhos[0]).max()
    assert difference.max() > 0 > difference.min()


def test_the_operations_of_the_settings_are_those_of_the_algebra(done_copy):
    run_dir, _, _, kind = done_copy
    before = charge_entry(run_dir, kind, 'displacement')['z']['interface']['integral']
    edit_config(run_dir, '[interface.charge]\n', '[interface.charge]\noperations = ["+", "-"]\n')
    # Another analysis of the same calculations: nothing to prepare.
    charge.ChargeStep().analyse(run_dir, SYSTEM, CALC_IDS)
    after = charge_entry(run_dir, kind, 'displacement')['z']['interface']['integral']
    combined = ci.chgcar_algebra_from_list(chgcars_of(run_dir), ['+', '-'], 2)
    expected = ci.charge_displacement(combined, 2, *inputs_of(run_dir).integrate_in, 1000)[0]
    assert after == pytest.approx(expected, rel=1e-12) and after != pytest.approx(before, rel=1e-3)


def test_the_provenance_of_the_three_entries(done_run):
    run_dir, calc_ids, _, kind = done_run
    document_ = read_system(system_dir(run_dir, SYSTEM))
    inputs = inputs_of(run_dir)
    expected = charge.ChargeStep().provenance(run_dir, SYSTEM)
    assert expected.step == 'charge' and expected.parameters == inputs.parameters
    assert set(expected.calculations) == set(calc_ids)
    # The files of the three densities, which the results of the records do not cover.
    assert set(expected.files) == {f'density_{part}' for part in PARTS}
    for part, calc_id in zip(PARTS, calc_ids):
        digest = file_digest(calc_dir(run_dir, SYSTEM, calc_id) / 'density.filplot')
        assert expected.files[f'density_{part}'] == digest
    for entry in charge.charge_leaves(kind, 'min_site'):
        stored = document_.provenance(entry)
        assert stored.fingerprint == expected.fingerprint
        assert set(stored.results) == set(calc_ids)
        assert stored.results == {c: read_record(system_dir(run_dir, SYSTEM), c).results_digest
                                  for c in calc_ids}


def test_the_step_is_done_with_its_entries_current(done_run):
    run_dir, _, _, _ = done_run
    assert charge.ChargeStep().done(run_dir, SYSTEM)


def test_the_step_is_not_done_before_the_analysis(analysed_copy):
    run_dir, _, _, _ = analysed_copy
    assert not charge.ChargeStep().done(run_dir, SYSTEM)
    with_pp_command(run_dir)
    finish_charge(run_dir)
    # Calculated and not analysed.
    assert not charge.ChargeStep().done(run_dir, SYSTEM)


def test_a_density_that_changed_after_the_analysis_is_not_current(done_copy):
    run_dir, calc_ids, _, _ = done_copy
    path = calc_dir(run_dir, SYSTEM, calc_ids[1]) / 'density.filplot'
    assert charge.ChargeStep().done(run_dir, SYSTEM)
    path.write_text(path.read_text() + ' \n')
    assert not charge.ChargeStep().done(run_dir, SYSTEM)


def test_a_density_that_is_gone_is_not_done(done_copy):
    run_dir, calc_ids, _, _ = done_copy
    (calc_dir(run_dir, SYSTEM, calc_ids[0]) / 'density.filplot').unlink()
    assert not charge.ChargeStep().done(run_dir, SYSTEM)


def test_the_operations_changed_after_the_analysis_are_not_current(done_copy):
    run_dir, _, _, _ = done_copy
    edit_config(run_dir, '[interface.charge]\n', '[interface.charge]\noperations = ["+", "-"]\n')
    assert not charge.ChargeStep().done(run_dir, SYSTEM)


def test_the_other_site_is_not_done_and_both_stay(done_copy):
    """One site per run (A59): the other site is not done, and its results live next to these."""
    run_dir, _, _, kind = done_copy
    edit_config(run_dir, 'site = "min_site"', 'site = "max_site"')
    assert not charge.ChargeStep().done(run_dir, SYSTEM)
    calc_ids = finish_charge(run_dir)
    assert calc_ids == [f'charge/max_site_{part}' for part in PARTS]
    charge.ChargeStep().analyse(run_dir, SYSTEM, calc_ids)
    assert charge.ChargeStep().done(run_dir, SYSTEM)
    for part in ('data', 'average', 'displacement'):
        assert set(document(run_dir)['charge'][kind][part]) == {'min_site', 'max_site'}
    edit_config(run_dir, 'site = "max_site"', 'site = "min_site"')
    assert charge.ChargeStep().done(run_dir, SYSTEM)


def test_the_adhesion_changed_after_the_analysis_is_not_current(done_copy):
    run_dir, _, pes_name, kind = done_copy
    sdir = system_dir(run_dir, SYSTEM)
    calc_id = adhesion.AdhesionStep().calculations(run_dir, SYSTEM)[0].calc_id
    record = read_record(sdir, calc_id)
    results = {**record.results, 'output': {**record.results['output'], 'energy': -999.0}}
    write_record(sdir, dataclasses.replace(record, results=results))
    assert not charge.ChargeStep().done(run_dir, SYSTEM)
    with pytest.raises(PrerequisiteError, match='entries missing or not current'):
        check_prerequisites(read_system(sdir), charge.ChargeStep().expected(run_dir, SYSTEM), sdir)


def test_the_other_entries_of_the_document_are_not_touched(analysed_copy):
    run_dir, _, _, kind = analysed_copy
    before = copy.deepcopy(document(run_dir))
    charged(run_dir)
    after = document(run_dir)
    assert set(after) - set(before) == {'charge'}
    assert {key: after[key] for key in before} == before
    assert set(after['charge']) == {kind} and set(after['charge'][kind]) == {
        'data', 'average', 'displacement'}


def test_the_densities_are_not_changed_by_the_analysis(done_copy):
    run_dir, calc_ids, _, _ = done_copy
    before = [(calc_dir(run_dir, SYSTEM, c) / 'density.filplot').read_bytes() for c in calc_ids]
    charge.ChargeStep().analyse(run_dir, SYSTEM, calc_ids)
    assert before == [(calc_dir(run_dir, SYSTEM, c) / 'density.filplot').read_bytes()
                      for c in calc_ids]


# --- errors of the analysis: nothing is saved ---------------------------------------------------

def system_bytes(run_dir):
    return (system_dir(run_dir, SYSTEM) / 'system.json').read_bytes()


def prepared(analysed_copy):
    run_dir, *_ = analysed_copy
    with_pp_command(run_dir)
    return run_dir, finish_charge(run_dir)


def test_the_calculations_must_be_those_of_the_settings(analysed_copy):
    run_dir, calc_ids = prepared(analysed_copy)
    before = system_bytes(run_dir)
    with pytest.raises(ValueError, match='not those of the settings'):
        charge.ChargeStep().analyse(run_dir, SYSTEM, calc_ids[:2])
    assert system_bytes(run_dir) == before


def test_a_calculation_without_a_record_stops_the_analysis(analysed_copy):
    run_dir, calc_ids = prepared(analysed_copy)
    record_path(system_dir(run_dir, SYSTEM), calc_ids[2]).unlink()
    before = system_bytes(run_dir)
    with pytest.raises(StoreError, match=r'charge/min_site_top_slab: no record'):
        charge.ChargeStep().analyse(run_dir, SYSTEM, calc_ids)
    assert system_bytes(run_dir) == before


def test_a_density_that_failed_stops_the_analysis(analysed_copy):
    run_dir, calc_ids = prepared(analysed_copy)
    sdir = system_dir(run_dir, SYSTEM)
    record = read_record(sdir, calc_ids[1])
    attempt = dataclasses.replace(
        record.attempts[0], status=Status('ended', 'converged', 'n/a', 'ok', 'failed'),
        density=dataclasses.replace(record.attempts[0].density, error='output.ppo: no JOB DONE'))
    write_record(sdir, dataclasses.replace(record, status=attempt.status, attempts=(attempt,)))
    before = system_bytes(run_dir)
    with pytest.raises(StoreError, match=r'charge/min_site_bot_slab.*charge density failed.*'
                                         r'no JOB DONE'):
        charge.ChargeStep().analyse(run_dir, SYSTEM, calc_ids)
    assert system_bytes(run_dir) == before


def test_a_calculation_that_gives_no_density_stops_the_analysis(analysed_copy):
    run_dir, calc_ids = prepared(analysed_copy)
    sdir = system_dir(run_dir, SYSTEM)
    record = read_record(sdir, calc_ids[0])
    attempt = dataclasses.replace(record.attempts[0], density=None,
                                  status=Status('ended', 'converged', 'n/a', 'ok'))
    write_record(sdir, dataclasses.replace(record, status=attempt.status, attempts=(attempt,)))
    with pytest.raises(StoreError, match=r'charge/min_site_interface.*charge density'):
        charge.ChargeStep().analyse(run_dir, SYSTEM, calc_ids)


def test_densities_on_other_grids_cannot_be_combined(analysed_copy):
    run_dir, calc_ids = prepared(analysed_copy)
    path = calc_dir(run_dir, SYSTEM, calc_ids[2])
    final = read_structure(path / 'structure_out.extxyz', format='extxyz')
    (path / 'density.filplot').write_text(filplot_of(final, (4, 4, 40)))
    before = system_bytes(run_dir)
    with pytest.raises(density.DensityError, match=r'cannot be combined.*grid \(4, 4, 40\)'):
        charge.ChargeStep().analyse(run_dir, SYSTEM, calc_ids)
    assert system_bytes(run_dir) == before


def test_densities_of_another_cutoff_cannot_be_combined(analysed_copy):
    run_dir, calc_ids = prepared(analysed_copy)
    path = calc_dir(run_dir, SYSTEM, calc_ids[1])
    final = read_structure(path / 'structure_out.extxyz', format='extxyz')
    (path / 'density.filplot').write_text(filplot_of(final, ecut=45.0))
    with pytest.raises(density.DensityError, match=r'bot_slab|ecut = 45.0'):
        charge.ChargeStep().analyse(run_dir, SYSTEM, calc_ids)


def test_a_density_of_another_structure_stops_the_analysis(analysed_copy):
    run_dir, calc_ids = prepared(analysed_copy)
    path = calc_dir(run_dir, SYSTEM, calc_ids[1])
    moved = read_structure(path / 'structure_out.extxyz', format='extxyz')
    moved.translate_sites([0], [0.0, 0.0, 0.05], frac_coords=False)
    (path / 'density.filplot').write_text(filplot_of(moved))
    with pytest.raises(density.DensityError, match='disagrees with the structure'):
        charge.ChargeStep().analyse(run_dir, SYSTEM, calc_ids)


def test_energies_of_another_model_cannot_be_combined(analysed_copy):
    run_dir, calc_ids = prepared(analysed_copy)
    sdir = system_dir(run_dir, SYSTEM)
    record = read_record(sdir, calc_ids[2])
    model = {**record.energy_model, 'ecutwfc': record.energy_model.get('ecutwfc', 0) + 5}
    observed = {**record.observed}
    write_record(sdir, dataclasses.replace(record, energy_model=model, observed=observed))
    before = system_bytes(run_dir)
    with pytest.raises(StoreError):
        charge.ChargeStep().analyse(run_dir, SYSTEM, calc_ids)
    assert system_bytes(run_dir) == before


def test_the_prerequisites_are_checked_before_the_analysis(analysed_copy):
    run_dir, calc_ids = prepared(analysed_copy)
    edit_config(run_dir, 'max_area = 160.0', 'max_area = 150.0')
    before = system_bytes(run_dir)
    with pytest.raises(PrerequisiteError, match='upstream of adhesion not done or not current'):
        charge.ChargeStep().analyse(run_dir, SYSTEM, calc_ids)
    assert system_bytes(run_dir) == before


def test_the_analysis_writes_through_the_calls_of_the_store(analysed_copy):
    """``run_charge_analysis`` moves the results, then writes the averages and the displacement.

    Each part is written with ``update_db`` under ``charge.<kind>``, as the original Firetasks do,
    and the document changes in memory only: whoever holds the writer saves it.
    """
    run_dir, calc_ids = prepared(analysed_copy)
    inputs = inputs_of(run_dir)
    calls = []

    class Recording(LegacyStore):
        def update_db(self, data, collection, *, entry=None, fltr=None, **kwargs):
            calls.append((entry, data))
            super().update_db(data, collection, entry=entry, fltr=fltr, **kwargs)

    before = system_bytes(run_dir)
    with SystemWriter(system_dir(run_dir, SYSTEM)) as writer:
        store = Recording(system_dir(run_dir, SYSTEM), writer.document, charge.COLLECTION)
        charge.run_charge_analysis(inputs, store, calc_ids, chgcars_of(run_dir))
    assert system_bytes(run_dir) == before
    kind = inputs.type
    # Three moves of results (a list of entries each), then the averages, then the displacement.
    assert len(calls) == 5
    for (entry, data), part in zip(calls[:3], PARTS):
        assert isinstance(entry, list) and all(isinstance(path, list) for path in entry)
        assert entry[0][:5] == ['charge', kind, 'data', 'min_site', part]
        assert len(entry) == len(data) == 15
    assert calls[3][0] == ['charge', kind] and list(calls[3][1]) == ['average']
    assert calls[4][0] == ['charge', kind] and list(calls[4][1]) == ['displacement']
    assert list(calls[3][1]['average']['min_site']) == PARTS
    assert list(calls[4][1]['displacement']['min_site']['z']) == ['all', 'interface', 'grid',
                                                                  'rescale']


# --- the orchestrator with the fake pw.x and pp.x (the small case of the step) -------------------

FAKE = Path(__file__).resolve().parent / 'qe_check_fake.py'
posix_only = pytest.mark.skipif(os.name != 'posix' or not Path('/bin/bash').exists(),
                                reason='the worker launches pw.x and pp.x with /bin/bash')


def run_with_the_fake_engines(tmp_path):
    """The example run up to the step, with ``fake_pw.x`` and ``fake_pp.x`` as its programs.

    pw.x and pp.x are the programs of ``qe_check_fake.py``: nothing is computed. pw.x prints the
    output of a converged run of its input, with the checksums of the UPF files of
    ``pseudo_dir``; pp.x writes a uniform density in the cell of the final structure. The UPF
    files and their checksums are made before any calculation is prepared, since the checksums
    are part of what a calculation is.
    """
    run_dir = pes_run(tmp_path, steps=('charge', 'adhesion', 'pes_scf', 'match'))
    pseudo = tmp_path / 'pseudo'
    pseudo.mkdir()
    lines = []
    for name in ('Cu.UPF', 'Al.UPF'):
        (pseudo / name).write_text(f'fake {name}\n')
        lines.append(f'{hashlib.md5((pseudo / name).read_bytes()).hexdigest()}  {name}')
    (run_dir / 'pseudo.md5').write_text('\n'.join(lines) + '\n')
    edit_config(run_dir, 'pseudo_dir = "/path/to/pseudo"', f'pseudo_dir = "{pseudo}"')
    edit_config(run_dir, 'run_command = "mpirun -np 32 pw.x -nk 4"',
                f'run_command = "{sys.executable} {FAKE} pw"')
    edit_config(run_dir, '# pp_command = "mpirun -np 32 pp.x"',
                f'pp_command = "{sys.executable} {FAKE} pp"')
    configure(run_dir, 'pes_scf', 'regular')
    return run_dir


def run_the_workers(run_dir, slurm):
    """A worker for each job submitted and not yet run, as its job would run it; the job over."""
    ran = []
    for submitted in sorted((run_dir / 'jobs').glob('*/submitted.json')):
        for job_id in json.loads(submitted.read_text())['job_ids']:
            if slurm.states.get(job_id) != 'PENDING':
                continue
            done = subprocess.run(
                [sys.executable, '-m', 'triboworkflow.execution.worker',
                 str(submitted.parent / 'manifest.json')],
                env={**os.environ, 'SLURM_JOB_ID': job_id}, capture_output=True, text=True,
                timeout=600)
            assert done.returncode == 0, done.stdout + done.stderr
            slurm.states[job_id] = 'COMPLETED'
            ran.append(job_id)
    return ran


@posix_only
def test_the_orchestrator_runs_the_step_with_the_fake_engines(tmp_path, fake_slurm):
    """Match, the static surface and the adhesion done, the step runs through the workers.

    The three calculations are launched in jobs, run by the worker with the fake pw.x, which
    prints its output, and the fake pp.x, which writes the density after it; the records are
    complete with the density ``ok`` and the step analyses them.
    """
    run_dir = run_with_the_fake_engines(tmp_path)
    orch = build_orchestrator(run_dir, groups=2)
    assert orch.order == ['match', 'pes_scf', 'adhesion', 'charge']

    def statuses():
        return {name: orch.entry(SYSTEM, name)['status'] for name in orch.order}

    orch.run_pass()
    assert statuses() == {'match': 'completed', 'pes_scf': 'running', 'adhesion': 'waiting',
                          'charge': 'waiting'}
    for calc in pes.StaticPesStep().calculations(run_dir, SYSTEM):
        complete(run_dir, calc.calc_id, status=status_of('pes_scf'))
    fake_slurm.states.update({job_id: 'COMPLETED' for job_id in orch.manager.jobs})
    finish(run_dir, 'pes_scf', prepare=False)
    orch.run_pass()
    assert statuses()['adhesion'] == 'running'
    for label in adhesion.LABELS:
        complete(run_dir, f'adhesion/{label}', status=RELAX_DONE)
    fake_slurm.states.update({job_id: 'COMPLETED' for job_id in orch.manager.jobs})
    finish_adhesion(run_dir, prepare=False)
    orch.run_pass()
    assert statuses() == {'match': 'completed', 'pes_scf': 'completed', 'adhesion': 'completed',
                          'charge': 'running'}
    # The step prepared its three calculations, with the density, and submitted them.
    sdir = system_dir(run_dir, SYSTEM)
    for calc_id in CALC_IDS:
        folder = calc_dir(run_dir, SYSTEM, calc_id)
        assert (folder / 'input.pwi').is_file() and (folder / 'input.ppi').is_file()
        assert read_record(sdir, calc_id).density is True
    assert run_the_workers(run_dir, fake_slurm)
    for calc_id in CALC_IDS:
        record = read_record(sdir, calc_id)
        assert record.is_complete and record.status.density == 'ok', calc_id
        assert (calc_dir(run_dir, SYSTEM, calc_id) / 'density.filplot').is_file()
    report = orch.run_pass()
    assert report.success and set(statuses().values()) == {'completed'}, report.steps
    assert charge.ChargeStep().done(run_dir, SYSTEM)
    # The fake pp.x writes a uniform density of one electron per atom: the interface has the
    # atoms of the two slabs, so the difference of the three is zero.
    kind = 'regular'
    displacement = charge_entry(run_dir, kind, 'displacement')['z']
    assert displacement['interface']['integral'] == pytest.approx(0.0, abs=1e-7)
    assert displacement['all']['integral'] == pytest.approx(0.0, abs=1e-7)
    assert displacement['rescale'] == 1000
    assert set(charge_entry(run_dir, kind)) == set(PARTS)
    assert set(charge_entry(run_dir, kind, 'average')) == set(PARTS)
    # Nothing to do on another pass.
    jobs = dict(orch.manager.jobs)
    assert orch.run_pass().success and orch.manager.jobs == jobs
