# Author: Vito Foderà
# Created: 2026-11-08

"""Tests of the step ppes (``triboworkflow.workflows.interface.ppes``).

The example run is matched, its PES are analysed on synthetic energies and its
adhesion is analysed for each of the three variants of ``[interface.adhesion]``
(``test_adhesion``); the tests check what the step reads from that document and
the calculations it makes: one static calculation of the interface relaxed by
the adhesion for each shift of the slab on top.

Notes
-----
To remove: the golden cases ``ppes_step_*`` (``tests/golden_cases.py``) compare
the step with TribChem's ``FT_StartPPES``, ``InterfaceWF.calculate_ppes``,
``FT_MoveTagResults`` and ``FT_FitPPES``: structures, k-points, database writes
and energies.
"""

import copy
import dataclasses
import logging
import math
import re
import shutil
import warnings

import numpy as np
import pytest
from monty.json import jsanitize
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

import golden_cases
from cli_helpers import SYSTEM, complete, fake_slurm  # noqa: F401 (fixture)
from pwo_helpers import pw_output, relaxed
from test_adhesion import (JOULE, NO_RESULT, RELAX_DONE, RESULT_KEYS,  # noqa: F401 (fixtures)
                           adhesion_run, analysed, analysed_copy, change_model, configure,
                           finish_adhesion, matched, short_surface, structure_of, surface_step,
                           surfaces, surfaces_copy)
from test_pes import document, edit_config, finish, pes_run, position_flags, status_of
from triboworkflow.cli.main import main
from triboworkflow.cli.run import build_orchestrator, load_run
from triboworkflow.core.settings import SettingsError
from triboworkflow.core.settings.schema import PPES_SHIFTS
from triboworkflow.core.store import (PrerequisiteError, Status, StoreError, SystemDocument,
                                      SystemWriter, check_prerequisites, read_record,
                                      read_system, record_path, write_record)
from triboworkflow.core.store.legacy import LegacyStore
from triboworkflow.core.structures.readers import read_extxyz_text
from triboworkflow.engines.qe.results import parse_output
from triboworkflow.execution.layout import system_dir
from triboworkflow.vendored.physics.tribology.ppes import fit_ppes
from triboworkflow.workflows.interface import adhesion, pes, ppes

SHIFT_LABELS = [f'min_site_{shift}' for shift in PPES_SHIFTS]


def inputs_of(run_dir):
    return ppes.ppes_inputs(run_dir, SYSTEM, load_run(run_dir))


def interface_of(run_dir, kind, site='min_site'):
    """The interface the step starts from, as the document holds it."""
    return document(run_dir)['structure']['opt'][kind][ppes.INTERFACE_OF[site]]


def top_species(run_dir):
    """Elements of the slab on top, from the slab of match."""
    return {specie.symbol for specie in structure_of(
        document(run_dir)['structure']['init']['top_slab']).species}


# --- the step and its entries ---------------------------------------------------------------

def test_step_and_constants():
    assert ppes.PpesStep.name == ppes.STEP == 'ppes'
    assert ppes.CALC_TYPE == 'slab_from_scratch' and ppes.PPES_TYPE == 'energy'
    assert ppes.INTER_TYPE == 'middle'
    assert ppes.INTERFACE_OF == {'min_site': 'interface_min', 'max_site': 'interface_max'}
    assert len(SHIFT_LABELS) == 15 and len(set(SHIFT_LABELS)) == 15


@pytest.mark.parametrize('kind', ['regular', 'short'])
@pytest.mark.parametrize('site', ['min_site', 'max_site'])
def test_entries_of_each_kind_and_site(kind, site):
    assert ppes.interface_entry(kind, site) == ('structure', 'opt', kind,
                                                'interface_min' if site == 'min_site'
                                                else 'interface_max')
    data = ('ppes', kind, 'data', site)
    energy = ('ppes', kind, 'energy', site)
    assert ppes.data_entry(kind, site) == data and ppes.energy_entry(kind, site) == energy
    # What the step empties before writing and what it gives a provenance: those two only.
    assert ppes.ppes_entries(kind, site) == [data, energy]
    assert ppes.ppes_leaves(kind, site) == [data, energy]


def test_label_of_a_shift_as_written():
    assert [ppes.shift_label('min_site', s) for s in (-0.75, 0.0, 3.0, 1e-05)] == [
        'min_site_-0.75', 'min_site_0.0', 'min_site_3.0', 'min_site_1e-05']
    assert ppes.shift_label('max_site', 2.25) == 'max_site_2.25'


# --- inputs -----------------------------------------------------------------------------------

def test_inputs_read_from_the_document(analysed):
    run_dir, _, pes_name, kind = analysed
    inputs = inputs_of(run_dir)
    assert (inputs.type, inputs.site, inputs.fit) == (kind, 'min_site', 'none')
    assert inputs.shifts == list(PPES_SHIFTS)
    # The interface of the adhesion at the lowest site, as the document holds it.
    assert inputs.read == {'interface': interface_of(run_dir, kind)}
    doc = document(run_dir)
    assert inputs.ids == {key: doc[key] for key in ('mid', 'formula', 'name', 'miller')}
    assert inputs.parameters == {
        'type': kind, 'site': 'min_site', 'shifts': list(PPES_SHIFTS), 'fit': 'none',
        'calc_type': 'slab_from_scratch', 'ppes_type': 'energy',
        'comp_params': {'kdens': load_run(run_dir).k_points.kdens}}
    assert [label for label, *_ in inputs.points] == SHIFT_LABELS
    assert [shift for _, shift, *_ in inputs.points] == list(PPES_SHIFTS)


def test_the_site_and_the_shifts_come_from_the_settings(analysed_copy):
    run_dir, _, _, kind = analysed_copy
    edit_config(run_dir, 'site = "min_site"', 'site = "max_site"')
    edit_config(run_dir, '\nfit = "none"', '\nshifts = [0.5, -1, 2.0]\nfit = "uber"')
    inputs = inputs_of(run_dir)
    assert (inputs.site, inputs.shifts, inputs.fit) == ('max_site', [0.5, -1.0, 2.0], 'uber')
    assert inputs.read == {'interface': interface_of(run_dir, kind, 'max_site')}
    assert [label for label, *_ in inputs.points] == ['max_site_0.5', 'max_site_-1.0',
                                                      'max_site_2.0']
    assert inputs.parameters['fit'] == 'uber' and inputs.parameters['shifts'] == [0.5, -1.0, 2.0]


def test_the_entry_read_is_not_changed_by_the_vendored_code(analysed):
    run_dir, _, _, kind = analysed
    before = interface_of(run_dir, kind)
    inputs = inputs_of(run_dir)
    assert interface_of(run_dir, kind) == before
    assert inputs.read['interface'] == before
    # The vendored code prints which sites it moves; the step keeps it for the log.
    assert 'site to shift' in inputs.printed


def test_only_the_slab_on_top_moves(analysed):
    run_dir, _, _, kind = analysed
    original = structure_of(interface_of(run_dir, kind))
    top = np.array([site.specie.symbol in top_species(run_dir) for site in original])
    assert top.any() and not top.all()
    for label, shift, moved, _ in inputs_of(run_dir).points:
        assert np.allclose(moved.lattice.matrix, original.lattice.matrix)
        assert [site.specie for site in moved] == [site.specie for site in original]
        displacement = np.asarray(moved.cart_coords) - np.asarray(original.cart_coords)
        assert np.allclose(displacement[top], [0.0, 0.0, shift], atol=1e-9), label
        assert np.allclose(displacement[~top], 0.0, atol=1e-9), label


def test_no_constraint_in_a_static_calculation(analysed):
    run_dir, _, _, _ = analysed
    for _, _, moved, _ in inputs_of(run_dir).points:
        assert 'selective_dynamics' not in moved.site_properties


# --- calculations -----------------------------------------------------------------------------

def test_one_static_calculation_per_shift(analysed):
    run_dir, _, _, _ = analysed
    inputs = inputs_of(run_dir)
    calcs = ppes.ppes_calculations(SYSTEM, inputs)
    assert [calc.calc_id for calc in calcs] == [f'ppes/{label}' for label in SHIFT_LABELS]
    assert [calc.calc_id for calc in ppes.PpesStep().calculations(run_dir, SYSTEM)] == [
        calc.calc_id for calc in calcs]
    for calc, label in zip(calcs, SHIFT_LABELS):
        assert calc.record.calc_type == 'slab_from_scratch'
        assert calc.record.step == 'ppes' and calc.record.system == SYSTEM
        assert calc.record.label == label
        assert re.search(r"calculation\s*=\s*'scf'", calc.input_text)
        assert re.search(rf"prefix\s*=\s*'ppes_{re.escape(label)}'", calc.input_text)
        # No atom is fixed: the flags of the relaxed interface do not reach a static one.
        assert set(map(str, position_flags(calc.input_text))) == {'None'}


def test_k_points_from_the_density(analysed):
    run_dir, _, _, _ = analysed
    kpoints = [kpoints for *_, kpoints in inputs_of(run_dir).points]
    # The cell does not change with the shift: one grid for all the calculations.
    assert all(str(k) == str(kpoints[0]) for k in kpoints)
    assert 'K_POINTS' in ppes.ppes_calculations(SYSTEM, inputs_of(run_dir))[0].input_text


def test_k_points_from_their_density_only(surfaces):
    settings = load_run(surfaces)
    settings = dataclasses.replace(settings, k_points=dataclasses.replace(
        settings.k_points, mode='automatic', grid=[4, 4, 1], offset=[0, 0, 0]))
    with pytest.raises(SettingsError, match=r'k_points\.mode.*kdens'):
        ppes.ppes_inputs(surfaces, SYSTEM, settings)


def test_fingerprints_follow_the_structures_and_not_the_fit(analysed_copy):
    run_dir, _, _, _ = analysed_copy
    plan = ppes.PpesStep().plan(run_dir, SYSTEM)
    assert len(plan) == 15 and len(set(plan.values())) == 15
    # The fit is not part of a calculation (P4): the same plan with the other fit.
    edit_config(run_dir, 'fit = "none"', 'fit = "uber"')
    assert ppes.PpesStep().plan(run_dir, SYSTEM) == plan
    # Another list of shifts: the calculations of the shifts in common are the same.
    edit_config(run_dir, '\nfit = "uber"', '\nshifts = [0.0, 0.25, 9.0]\nfit = "uber"')
    other = ppes.PpesStep().plan(run_dir, SYSTEM)
    assert list(other) == ['ppes/min_site_0.0', 'ppes/min_site_0.25', 'ppes/min_site_9.0']
    assert other['ppes/min_site_0.0'] == plan['ppes/min_site_0.0']
    assert other['ppes/min_site_0.25'] == plan['ppes/min_site_0.25']
    assert other['ppes/min_site_9.0'] not in plan.values()


def test_the_site_changes_the_calculations(analysed_copy):
    run_dir, _, _, _ = analysed_copy
    plan = ppes.PpesStep().plan(run_dir, SYSTEM)
    edit_config(run_dir, 'site = "min_site"', 'site = "max_site"')
    other = ppes.PpesStep().plan(run_dir, SYSTEM)
    assert set(other) == {calc_id.replace('min_site', 'max_site') for calc_id in plan}
    # The two interfaces differ, so the calculations of the same shift differ too.
    assert not set(other.values()) & set(plan.values())


# --- what the step reads is current along the whole chain (P1) ---------------------------------

def test_expected_is_the_interface_with_the_provenance_of_the_adhesion(analysed):
    run_dir, _, _, kind = analysed
    assert ppes.PpesStep().expected(run_dir, SYSTEM) == {
        ppes.interface_entry(kind, 'min_site'): adhesion.AdhesionStep().provenance(run_dir, SYSTEM)}


def test_expected_follows_the_site(analysed_copy):
    run_dir, _, _, kind = analysed_copy
    edit_config(run_dir, 'site = "min_site"', 'site = "max_site"')
    assert list(ppes.PpesStep().expected(run_dir, SYSTEM)) == [
        ('structure', 'opt', kind, 'interface_max')]


def change_energy(run_dir, calc_id):
    """Complete a calculation again with another energy, and analyse nothing."""
    sdir = system_dir(run_dir, SYSTEM)
    record = read_record(sdir, calc_id)
    results = {**record.results, 'output': {**record.results['output'], 'energy': -999.0}}
    write_record(sdir, dataclasses.replace(record, results=results))


def test_the_surface_changed_after_its_analysis_stops_the_step(analysed_copy):
    """The adhesion read the entries of the surface as they were and keeps them current alone."""
    run_dir, _, pes_name, _ = analysed_copy
    ppes.PpesStep().expected(run_dir, SYSTEM)
    change_energy(run_dir, surface_step(pes_name).calculations(run_dir, SYSTEM)[0].calc_id)
    # The provenance of the adhesion is unchanged: it holds what it read.
    assert adhesion.AdhesionStep().provenance(run_dir, SYSTEM)
    with pytest.raises(PrerequisiteError) as error:
        ppes.PpesStep().expected(run_dir, SYSTEM)
    assert str(error.value) == (f'{SYSTEM}: steps upstream of adhesion not done or not current: '
                                f'{pes_name}; run them first')


def test_the_settings_of_the_surface_changed_after_its_analysis_stop_the_step(analysed_copy):
    run_dir, _, pes_name, kind = analysed_copy
    if (pes_name, kind) != ('pes_scf', 'regular'):
        pytest.skip('a setting of the static surface only')
    edit_config(run_dir, 'short = false', 'short = true')
    with pytest.raises(PrerequisiteError, match='upstream of adhesion not done or not current: '
                                                'pes_scf; run them first'):
        ppes.PpesStep().expected(run_dir, SYSTEM)


def test_match_changed_after_its_analysis_stops_the_step(analysed_copy):
    run_dir, _, pes_name, _ = analysed_copy
    edit_config(run_dir, 'max_area = 160.0', 'max_area = 150.0')
    with pytest.raises(PrerequisiteError, match=f'upstream of adhesion not done or not current: '
                                                f'match, {pes_name}; run them first'):
        ppes.PpesStep().expected(run_dir, SYSTEM)


def test_the_adhesion_not_current_stops_the_step_with_its_entry(analysed_copy):
    run_dir, calc_ids, _, kind = analysed_copy
    change_energy(run_dir, calc_ids[0])
    expected = ppes.PpesStep().expected(run_dir, SYSTEM)
    sdir = system_dir(run_dir, SYSTEM)
    with pytest.raises(PrerequisiteError, match=rf'entries missing or not current: '
                                                rf'structure\.opt\.{kind}\.interface_min'):
        check_prerequisites(read_system(sdir), expected, sdir)


# --- errors -----------------------------------------------------------------------------------

def test_without_match_the_step_stops(tmp_path):
    run_dir = adhesion_run(tmp_path)
    with pytest.raises(PrerequisiteError, match='run match first'):
        ppes.PpesStep().calculations(run_dir, SYSTEM)


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
    # The other site is still there.
    edit_config(run_dir, 'site = "max_site"', 'site = "min_site"')
    assert inputs_of(run_dir).site == 'min_site'


# --- the slab on top, through the adapter of pw.x (R1) ------------------------------------------

def relaxed_through_the_adapter(run_dir, kind, label='interface_min'):
    """The final structure of a calculation of the adhesion as the adapter of pw.x returns it.

    The structure of the input is written by the adapter (``structure_in.extxyz``), a relaxation
    of it is printed as pw.x prints it (with the flags of the atoms that it keeps fixed), and the
    results are read from that output.
    """
    calc = {c.calc_id: c for c in adhesion.AdhesionStep().calculations(run_dir, SYSTEM)}[
        f'adhesion/{label}']
    structure_in = read_extxyz_text(calc.structure_text)
    labels = sorted(set(structure_in.labels), key=list(structure_in.labels).index)
    species = tuple((name, name, f'{name}.UPF', '0' * 32) for name in labels)
    flags = structure_in.site_properties.get('if_pos')
    text = pw_output(structure_in, 'relax', species=species,
                     forces=[(0.0, 0.0, 0.0)] * len(structure_in),
                     final_positions=relaxed(structure_in, (0.0, 0.0, 0.02)),
                     final_if_pos=None if flags is None else [tuple(f) for f in flags])
    parsed = parse_output(text, 'relax', structure_in)
    assert parsed.parsing == 'ok', parsed.reason
    return structure_in, parsed.results['output']['structure']


def put_interface(run_dir, structure, kind='regular', site='min_site'):
    """Replace the interface that the step starts from."""
    with SystemWriter(system_dir(run_dir, SYSTEM)) as writer:
        writer.document.set(ppes.interface_entry(kind, site), structure.as_dict())
        writer.save()


def moved_atoms(run_dir, final):
    """Whether each atom of ``final`` is moved by the shift (True) or stays (False), per shift."""
    moved = {}
    for label, shift, structure, _ in inputs_of(run_dir).points:
        displacement = np.asarray(structure.cart_coords) - np.asarray(final.cart_coords)
        assert np.allclose(displacement[:, :2], 0.0, atol=1e-9), label
        moved[shift] = np.isclose(displacement[:, 2], shift, atol=1e-9)
        assert np.allclose(displacement[~moved[shift], 2], 0.0, atol=1e-9), label
    return moved


def test_the_slab_on_top_moves_after_the_adapter(surfaces_copy):
    """The interface that the adapter returns for a relaxation keeps the slab on top the one moved.

    The interface has its contact plane at z = 0 and the vacuum around the middle of the cell:
    the slab on top is in the lower half of the cell, the bottom one in the upper half (at
    negative z once shifted by a cell). The relaxation moves the atoms a little and wraps none.
    """
    run_dir = surfaces_copy
    structure_in, output = relaxed_through_the_adapter(run_dir, 'regular')
    final = structure_of(output)
    top = np.array([site.specie.symbol in top_species(run_dir) for site in final])
    assert top.any() and not top.all()
    # Fractional z as the vendored function wraps it, then as the adapter returns it.
    assert (np.asarray(final.frac_coords)[top, 2] < 0.5).all()
    assert (np.asarray(final.frac_coords)[~top, 2] > 0.5).all()
    # The adapter did move the atoms from those of its input (by 0.02 angstrom along z).
    assert np.allclose(np.asarray(final.cart_coords) - np.asarray(structure_in.cart_coords),
                       [0.0, 0.0, 0.02], atol=1e-6)
    put_interface(run_dir, final)
    for shift, moved in moved_atoms(run_dir, final).items():
        # Only the atoms of the slab on top (a shift of zero moves none).
        if shift != 0.0:
            assert moved[top].all() and not moved[~top].any(), shift


def test_the_flags_of_a_relaxation_do_not_fix_a_static_calculation(surfaces_copy):
    """The interface at the highest site is relaxed along z only: its atoms come back with flags."""
    run_dir = surfaces_copy
    structure_in, output = relaxed_through_the_adapter(run_dir, 'regular', 'interface_max')
    final = structure_of(output)
    assert {tuple(f) for f in final.site_properties['if_pos']} == {(0, 0, 1)}
    put_interface(run_dir, final, site='max_site')
    edit_config(run_dir, 'site = "min_site"', 'site = "max_site"')
    inputs = inputs_of(run_dir)
    assert len(inputs.points) == 15
    for calc in ppes.ppes_calculations(SYSTEM, inputs):
        assert re.search(r"calculation\s*=\s*'scf'", calc.input_text), calc.calc_id
        # pw.x gets no flag: the static calculation moves nothing, and a flag would only
        # say that the atoms were fixed in the relaxation before.
        assert set(map(str, position_flags(calc.input_text))) == {'None'}, calc.calc_id


def test_the_slab_on_top_is_chosen_from_the_origin_of_the_cell(surfaces_copy):
    """An interface with the origin moved by half a cell has the other slab moved (A91).

    The vendored function moves the atoms whose fractional z, wrapped into the cell, is below
    one half, whatever slab they belong to: they are the slab on top only when each slab lies in
    one half of the cell, the one on top with the contact plane of the interface at the origin.
    The adapter writes and reads the positions as they are (no origin is chosen), so here the
    structure has to be translated before reaching the step to get the wrong slab; the next two
    tests show the slabs that are thicker than half the cell, which need no translation. All
    three document the selection of the original code and are to be replaced by the selection
    by the slab each atom belongs to.
    """
    run_dir = surfaces_copy
    _, output = relaxed_through_the_adapter(run_dir, 'regular')
    final = structure_of(output)
    top = np.array([site.specie.symbol in top_species(run_dir) for site in final])
    translated = final.copy()
    translated.translate_sites(list(range(len(translated))), [0.0, 0.0, final.lattice.c / 2],
                               frac_coords=False, to_unit_cell=False)
    put_interface(run_dir, translated)
    for shift, moved in moved_atoms(run_dir, translated).items():
        if shift != 0.0:
            assert moved[~top].all() and not moved[top].any(), shift


def thicker_slab(final, slab, extra=7.5, height=26.0):
    """The interface with one slab thicker, a second block of its atoms next to the first.

    Parameters
    ----------
    final : pymatgen.core.structure.Structure
        The interface as the adapter returns it, with its slabs in the site property
        ``interface_region`` (``bot_slab`` or ``top_slab``).
    slab : str
        The slab made thicker, ``top_slab`` (upwards from the contact plane at z = 0) or
        ``bot_slab`` (downwards from the top of the cell).
    extra : float
        Distance between the two blocks, in angstrom.
    height : float
        Height of the new cell: the slabs are kept inside it, none wrapped.

    Returns
    -------
    pymatgen.core.structure.Structure
        With ``interface_region`` on every site.
    """
    c = final.lattice.c
    species, positions, regions = [], [], []
    for site, region in zip(final, final.site_properties['interface_region']):
        x, y, z = site.coords
        z = z if region == 'top_slab' else height - (c - z)
        for block in (0, 1) if region == slab else (0,):
            species.append(site.specie.symbol)
            positions.append([x, y, z + (block if region == 'top_slab' else -block) * extra])
            regions.append(region)
    lattice = Lattice([final.lattice.matrix[0], final.lattice.matrix[1], [0.0, 0.0, height]])
    return Structure(lattice, species, positions, coords_are_cartesian=True,
                     site_properties={'interface_region': regions})


def test_a_top_slab_thicker_than_half_the_cell_is_deformed(surfaces_copy):
    """The atoms of the slab on top above the middle of the cell stay where they are (A91)."""
    run_dir = surfaces_copy
    final = structure_of(relaxed_through_the_adapter(run_dir, 'regular')[1])
    thick = thicker_slab(final, 'top_slab')
    top = np.array(thick.site_properties['interface_region']) == 'top_slab'
    z = np.asarray(thick.frac_coords)[:, 2]
    assert z[top].min() < 0.5 < z[top].max() and (z[~top] > 0.5).all()
    put_interface(run_dir, thick)
    for shift, moved in moved_atoms(run_dir, thick).items():
        if shift != 0.0:
            # Only the atoms below the middle move, and they are part of the slab on top.
            assert (moved == (z < 0.5)).all(), shift
            assert 0 < moved[top].sum() < top.sum() and not moved[~top].any(), shift


def test_a_bottom_slab_thicker_than_half_the_cell_moves_with_the_top_one(surfaces_copy):
    """The atoms of the bottom slab below the middle of the cell are moved (A91)."""
    run_dir = surfaces_copy
    final = structure_of(relaxed_through_the_adapter(run_dir, 'regular')[1])
    thick = thicker_slab(final, 'bot_slab')
    top = np.array(thick.site_properties['interface_region']) == 'top_slab'
    z = np.asarray(thick.frac_coords)[:, 2]
    assert (z[top] < 0.5).all() and z[~top].min() < 0.5 < z[~top].max()
    put_interface(run_dir, thick)
    for shift, moved in moved_atoms(run_dir, thick).items():
        if shift != 0.0:
            assert (moved == (z < 0.5)).all(), shift
            assert moved[top].all() and 0 < moved[~top].sum() < (~top).sum(), shift


# --- the calculations completed ------------------------------------------------------------------

def energy_of(shift):
    """Energy (eV) of the interface at a shift: an UBER curve with its minimum at zero.

    The depth is 0.4 eV and the length 0.8 angstrom, 1000.4 eV below zero at the minimum, with a
    deterministic ripple of 0.1 meV so that the fit has residuals.
    """
    return (-1000.4 + 0.4 * (1 - (1 + shift / 0.8) * math.exp(-shift / 0.8))
            + 1e-4 * math.sin(40 * shift))


def finish_ppes(run_dir, energy=energy_of, prepare=True, **changes):
    """Prepare the calculations of the step and complete them with synthetic results.

    The final structure of each is its input, as for a static calculation. With ``prepare`` False
    the calculations are already prepared.

    Returns
    -------
    list of str
        ``calc_id`` of the calculations, in the order of the shifts.
    """
    step = ppes.PpesStep()
    inputs = step.inputs(run_dir, SYSTEM)
    if prepare:
        step.prepare(run_dir, SYSTEM)
    sdir = system_dir(run_dir, SYSTEM)
    for label, shift, structure, _ in inputs.points:
        value = energy(shift)
        results = {'nsites': len(structure), 'input': {'structure': structure.as_dict()},
                   'output': {'structure': structure.as_dict(), 'density': float(structure.density),
                              'energy': value,
                              'energy_per_atom': None if value is None else value / len(structure)}}
        record = read_record(sdir, f'ppes/{label}')
        write_record(sdir, dataclasses.replace(
            record, status=Status('ended', 'converged', 'n/a', 'ok'),
            results=jsanitize(results, strict=False), observed={}, **changes))
    return [f'ppes/{label}' for label, *_ in inputs.points]


@pytest.fixture(scope='module')
def done_run(analysed, tmp_path_factory):
    """Each variant of the adhesion with the step run: run folder, calculations, surface, kind."""
    run_dir, _, pes_name, kind = analysed
    copied = tmp_path_factory.mktemp('ppes') / 'run'
    shutil.copytree(run_dir, copied)
    calc_ids = finish_ppes(copied)
    ppes.PpesStep().analyse(copied, SYSTEM, calc_ids)
    return copied, calc_ids, pes_name, kind


@pytest.fixture(scope='module')
def adhered(surfaces, tmp_path_factory):
    """The run with the adhesion (relaxed surface, complete slabs) analysed, and no step after it."""
    run_dir = tmp_path_factory.mktemp('adhered') / 'run'
    shutil.copytree(surfaces, run_dir)
    adhesion.AdhesionStep().analyse(run_dir, SYSTEM, finish_adhesion(run_dir))
    return run_dir


@pytest.fixture
def adhered_copy(adhered, tmp_path):
    copied = tmp_path / 'adhered'
    shutil.copytree(adhered, copied)
    return copied


@pytest.fixture(scope='module')
def ppes_run(adhered, tmp_path_factory):
    """The run of ``adhered`` with the step run (minimum site, 15 shifts, no fit)."""
    run_dir = tmp_path_factory.mktemp('ppes_run') / 'run'
    shutil.copytree(adhered, run_dir)
    ppes.PpesStep().analyse(run_dir, SYSTEM, finish_ppes(run_dir))
    return run_dir


@pytest.fixture
def ppes_copy(ppes_run, tmp_path):
    copied = tmp_path / 'ppes'
    shutil.copytree(ppes_run, copied)
    return copied


def ppes_doc(run_dir, kind='regular'):
    return document(run_dir)['ppes'][kind]


def key_of(shift):
    return 'shift_' + str(shift).replace('.', ',')


def area_of(structure):
    cell = structure.lattice.matrix
    return float(np.linalg.norm(np.cross(cell[0], cell[1]))) * 1e-20


def system_bytes(run_dir):
    return (system_dir(run_dir, SYSTEM) / 'system.json').read_bytes()


def test_results_of_every_shift(done_run):
    run_dir, calc_ids, _, kind = done_run
    inputs = ppes.PpesStep().inputs(run_dir, SYSTEM)
    data = ppes_doc(run_dir, kind)['data']['min_site']
    assert list(data) == [key_of(shift) for shift in PPES_SHIFTS]
    for (label, shift, structure, _), calc_id in zip(inputs.points, calc_ids):
        results = data[key_of(shift)]
        assert set(results) == {'output', 'input'}
        # The keys of every calculation, None where it gives nothing, then the structures.
        assert set(results['output']) == RESULT_KEYS | {'structure'}
        assert set(results['input']) == {'structure'}
        output = results['output']
        assert output['energy'] == energy_of(shift) and output['nsites'] == len(structure)
        assert output['energy_per_atom'] == pytest.approx(energy_of(shift) / len(structure))
        assert output['task_label'] == output['task_id'] == output['dir_name'] == calc_id
        assert all(output[key] is None for key in NO_RESULT)
        assert np.allclose(structure_of(results['input']['structure']).cart_coords,
                           structure.cart_coords)
        assert np.allclose(structure_of(output['structure']).cart_coords, structure.cart_coords)
    # The results of the two kinds are in different entries, the other site is not there.
    assert set(document(run_dir)['ppes']) == {kind}
    assert set(ppes_doc(run_dir, kind)) == {'data', 'energy'}
    assert set(ppes_doc(run_dir, kind)['energy']) == set(ppes_doc(run_dir, kind)['data']) == {
        'min_site'}


def test_energies_in_ev_and_in_joule_per_square_metre(done_run):
    run_dir, _, _, kind = done_run
    energy = ppes_doc(run_dir, kind)['energy']['min_site']
    assert energy['ppes_ev'] == [[shift, energy_of(shift)] for shift in PPES_SHIFTS]
    # The area of the cell of the interface, and the energy as it is (not referred to the minimum).
    cell = structure_of(interface_of(run_dir, kind))
    per_ev = JOULE / area_of(cell)
    assert [row[0] for row in energy['ppes_jm2']] == list(PPES_SHIFTS)
    assert [row[1] for row in energy['ppes_jm2']] == pytest.approx(
        [energy_of(shift) * per_ev for shift in PPES_SHIFTS])
    assert all(value < 0 for _, value in energy['ppes_jm2'])


def test_no_fit_by_default(done_run):
    run_dir, _, _, kind = done_run
    assert set(ppes_doc(run_dir, kind)['energy']['min_site']) == {'ppes_ev', 'ppes_jm2'}


def test_provenance_on_the_entries_written(done_run):
    run_dir, calc_ids, _, kind = done_run
    expected = ppes.PpesStep().provenance(run_dir, SYSTEM)
    found = read_system(system_dir(run_dir, SYSTEM))
    leaves = ppes.ppes_leaves(kind, 'min_site')
    assert leaves == [('ppes', kind, 'data', 'min_site'), ('ppes', kind, 'energy', 'min_site')]
    for entry in leaves:
        provenance = found.provenance(entry)
        # The same provenance on every entry, with the results consumed.
        assert provenance == dataclasses.replace(expected, results=provenance.results)
        assert set(provenance.results) == set(calc_ids) == set(provenance.calculations)
        assert provenance.step == 'ppes' and provenance.parameters['fit'] == 'none'
        assert provenance.parameters['type'] == kind
    assert ppes.PpesStep().done(run_dir, SYSTEM)


def test_units_of_the_fit(ppes_copy):
    edit_config(ppes_copy, 'fit = "none"', 'fit = "uber"')
    calc_ids = finish_ppes(ppes_copy, prepare=False)
    ppes.PpesStep().analyse(ppes_copy, SYSTEM, calc_ids)
    energy = ppes_doc(ppes_copy)['energy']['min_site']
    fit = energy['uber']
    assert set(fit) == {'g', 'l', 'sigma_g', 'sigma_l'}
    # The fit receives the energies in eV: g is a depth in eV per cell (0.4), and l a length in
    # angstrom (0.8), the values of the curve; not J/m^2, whose depth would be about 0.1.
    assert fit['g'] == pytest.approx(0.4, abs=2e-3) and fit['l'] == pytest.approx(0.8, abs=1e-2)
    assert 0 < fit['sigma_g'] < 1e-2 and 0 < fit['sigma_l'] < 1e-2
    # The very call of the original, on the shifts and the energies in eV.
    assert fit == {key: float(value) for key, value in
                   fit_ppes(energy['ppes_ev'], 'uber').items()}
    assert set(energy) == {'ppes_ev', 'ppes_jm2', 'uber'}
    assert read_system(system_dir(ppes_copy, SYSTEM)).provenance(
        ppes.energy_entry('regular', 'min_site')).parameters['fit'] == 'uber'


# --- the fit (P3) ----------------------------------------------------------------------------------

def test_none_is_no_fit_before_the_historic_check():
    # The string "none" is true in a condition: it becomes None first.
    assert ppes.historic_fit('none') is None
    assert ppes.historic_fit('uber') == 'uber'


def test_no_fit_is_asked_for_with_none(ppes_copy, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('the fit was called')

    monkeypatch.setattr(ppes, 'fit_ppes', fail)
    calc_ids = finish_ppes(ppes_copy, prepare=False)
    ppes.PpesStep().analyse(ppes_copy, SYSTEM, calc_ids)
    assert 'none' not in ppes_doc(ppes_copy)['energy']['min_site']


def run_with(run_dir, shifts, fit='uber'):
    """Settings with the shifts and the fit given, the calculations of the step completed."""
    edit_config(run_dir, '\nfit = "none"', f'\nshifts = {shifts}\nfit = "{fit}"')
    return finish_ppes(run_dir)


@pytest.mark.parametrize('shifts, used', [
    ([-0.5, -0.25, 0.0, 1.0], 2),
    ([-0.5, 1.0, -0.25], 1),
    ([-0.5, -0.25], 0),
])
def test_too_few_shifts_fail_the_fit_and_nothing_is_saved(adhered_copy, shifts, used):
    calc_ids = run_with(adhered_copy, shifts)
    before = system_bytes(adhered_copy)
    with pytest.raises(StoreError, match=rf"the fit 'uber' failed with {used} shifts that are "
                                         r'zero or positive.*needs at least three of them'):
        ppes.PpesStep().analyse(adhered_copy, SYSTEM, calc_ids)
    assert system_bytes(adhered_copy) == before


def test_the_error_of_the_fit_keeps_its_cause(adhered_copy):
    calc_ids = run_with(adhered_copy, [-0.5, 0.0, 1.0])
    with pytest.raises(StoreError) as error:
        ppes.PpesStep().analyse(adhered_copy, SYSTEM, calc_ids)
    assert isinstance(error.value.__cause__, TypeError)
    assert 'must not exceed the number of data points=2' in str(error.value)


def test_three_shifts_give_deviations_that_are_infinite(adhered_copy, caplog):
    calc_ids = run_with(adhered_copy, [-0.5, 0.0, 0.5, 1.0][1:])
    with caplog.at_level(logging.WARNING, logger=ppes.__name__):
        ppes.PpesStep().analyse(adhered_copy, SYSTEM, calc_ids)
    fit = ppes_doc(adhered_copy)['energy']['min_site']['uber']
    assert math.isinf(fit['sigma_g']) and math.isinf(fit['sigma_l'])
    assert math.isfinite(fit['g']) and math.isfinite(fit['l'])
    assert [record.getMessage() for record in caplog.records if record.name == ppes.__name__] == [
        f'{SYSTEM}/ppes: the covariance of the fit could not be estimated: the deviations of g '
        'and l are infinite']
    # What was written is read back and still current.
    assert ppes.PpesStep().done(adhered_copy, SYSTEM)


def test_four_shifts_give_finite_deviations(adhered_copy):
    calc_ids = run_with(adhered_copy, [-0.5, 0.0, 0.5, 1.0, 1.5])
    ppes.PpesStep().analyse(adhered_copy, SYSTEM, calc_ids)
    fit = ppes_doc(adhered_copy)['energy']['min_site']['uber']
    assert all(math.isfinite(fit[key]) for key in ('g', 'l', 'sigma_g', 'sigma_l'))


# --- the fit on the data of the original PPES -----------------------------------------------------

PPES_DATA = golden_cases.DATA / golden_cases.PPES_DATA


def curves_of_ppes_data():
    """The curves of the files of ``ppes_data``, as the shifts from the equilibrium distance.

    Returns
    -------
    dict
        File name (relative to the folder) to ``[shift, energy]`` for each row: the
        distance of the file minus the distance of the lowest energy, the way the
        step receives them.
    """
    curves = {}
    for path in sorted(PPES_DATA.rglob('*.txt')):
        data = np.loadtxt(path)
        data[:, 0] -= data[np.argmin(data[:, 1]), 0]
        curves[path.relative_to(PPES_DATA).as_posix()] = data.tolist()
    return curves


def fit_of_the_step(tmp_path, curve):
    """What the step fits on a curve: ``arrange_data`` on a document that holds its energies.

    Returns
    -------
    dict
        ``{'result': fit}``, or ``{'error': <name of the exception>, 'message': ...}`` for the
        exception the fit raised, as the golden file of the fit records them.
    """
    structure = Structure(Lattice.orthorhombic(4.0, 5.0, 20.0), ['Cu'], [[0, 0, 0]]).as_dict()
    ids = {'mid': 'a_b', 'miller': [[1, 1, 1], [1, 1, 1]]}
    data = {ppes.shift_key(shift): {'output': {'energy': energy, 'structure': structure}}
            for shift, energy in curve}
    document_ = {**ids, 'ppes': {'regular': {'data': {'min_site': data}}}}
    store = LegacyStore(tmp_path, SystemDocument(document_), ppes.COLLECTION)
    shifts = [shift for shift, _ in curve]
    assert len(data) == len(shifts)  # a repeated shift would share its entry
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            values = ppes.arrange_data(store, ids, 'regular', 'min_site', shifts, 'uber')
        except StoreError as error:
            return {'error': type(error.__cause__).__name__, 'message': str(error.__cause__)}
    return {'result': golden_cases.to_data(values['uber'])}


def test_the_fit_of_the_step_on_ppes_data_is_the_original_one(tmp_path):
    expected = golden_cases.read_golden('uber_ppes_data')['relative']
    curves = curves_of_ppes_data()
    assert sorted(curves) == sorted(expected) and len(curves) > 200
    differences = []
    for name, curve in curves.items():
        differences += golden_cases.compare(expected[name], fit_of_the_step(tmp_path, curve),
                                            None, name)
    assert not differences, f'{len(differences)} differences, the first ones: {differences[:10]}'
    # Not a comparison of failures: most of the curves are fitted, and the golden holds both
    # the fits that give the deviations and the failures.
    assert sum('result' in fit for fit in expected.values()) > len(expected) // 2


# --- only the entries of the kind and site run are replaced (DEV-43) --------------------------------

def test_analysis_again_gives_the_same_document(ppes_copy):
    before = document(ppes_copy)
    ppes.PpesStep().analyse(ppes_copy, SYSTEM, finish_ppes(ppes_copy, prepare=False))
    assert document(ppes_copy) == before
    assert ppes.PpesStep().done(ppes_copy, SYSTEM)


def without(doc, *entries):
    """A copy of the document without the entries (paths) given."""
    doc = copy.deepcopy(doc)
    for entry in entries:
        node = doc
        for key in entry[:-1]:
            node = node[key]
        del node[entry[-1]]
    return doc


def test_other_shifts_and_no_fit_leave_nothing_of_the_old_run(ppes_copy):
    # First with the fit, then with other shifts and no fit.
    edit_config(ppes_copy, 'fit = "none"', 'fit = "uber"')
    ppes.PpesStep().analyse(ppes_copy, SYSTEM, finish_ppes(ppes_copy, prepare=False))
    assert 'uber' in ppes_doc(ppes_copy)['energy']['min_site']
    before = document(ppes_copy)
    edit_config(ppes_copy, '\nfit = "uber"', '\nshifts = [0.0, 0.5, 1.0, 2.0, 5.0]\nfit = "none"')
    calc_ids = finish_ppes(ppes_copy)
    ppes.PpesStep().analyse(ppes_copy, SYSTEM, calc_ids)
    doc = ppes_doc(ppes_copy)
    keys = ['shift_0,0', 'shift_0,5', 'shift_1,0', 'shift_2,0', 'shift_5,0']
    assert list(doc['data']['min_site']) == keys
    assert [row[0] for row in doc['energy']['min_site']['ppes_ev']] == [0.0, 0.5, 1.0, 2.0, 5.0]
    assert set(doc['energy']['min_site']) == {'ppes_ev', 'ppes_jm2'}
    # Everything but the two entries of the site is as it was.
    assert without(document(ppes_copy), ('ppes', 'regular')) == without(before, ('ppes', 'regular'))


def test_the_other_site_the_other_kind_and_the_other_steps_stay(ppes_copy):
    marks = {('ppes', 'regular', 'data', 'max_site'): {'shift_0,0': {'marker': 1}},
             ('ppes', 'regular', 'energy', 'max_site'): {'ppes_ev': [[0.0, -1.0]]},
             ('ppes', 'short'): {'data': {'min_site': {'marker': 2}}},
             ('charge', 'regular'): {'marker': 3}}
    with SystemWriter(system_dir(ppes_copy, SYSTEM)) as writer:
        for entry, value in marks.items():
            writer.document.set(entry, value)
        writer.save()
    before = document(ppes_copy)
    # A different run of the site: other shifts, so that the two entries do change.
    edit_config(ppes_copy, '\nfit = "none"', '\nshifts = [0.0, 1.0, 2.0]\nfit = "none"')
    ppes.PpesStep().analyse(ppes_copy, SYSTEM, finish_ppes(ppes_copy))
    after = document(ppes_copy)
    assert list(after['ppes']['regular']['data']['min_site']) == ['shift_0,0', 'shift_1,0',
                                                                  'shift_2,0']
    own = (('ppes', 'regular', 'data', 'min_site'), ('ppes', 'regular', 'energy', 'min_site'))
    assert without(after, *own) == without(before, *own)
    for entry, value in marks.items():
        node = after
        for key in entry:
            node = node[key]
        assert node == value


def test_the_other_site_is_computed_beside_the_first(ppes_copy):
    edit_config(ppes_copy, 'site = "min_site"', 'site = "max_site"')
    assert not ppes.PpesStep().done(ppes_copy, SYSTEM)
    before = document(ppes_copy)
    calc_ids = finish_ppes(ppes_copy)
    assert calc_ids[0] == 'ppes/max_site_-0.75'
    ppes.PpesStep().analyse(ppes_copy, SYSTEM, calc_ids)
    doc = ppes_doc(ppes_copy)
    assert set(doc['data']) == set(doc['energy']) == {'min_site', 'max_site'}
    assert doc['data']['min_site'] == before['ppes']['regular']['data']['min_site']
    assert doc['energy']['min_site'] == before['ppes']['regular']['energy']['min_site']
    assert ppes.PpesStep().done(ppes_copy, SYSTEM)
    # Going back to the first site is done at once: its entries and calculations are there.
    edit_config(ppes_copy, 'site = "max_site"', 'site = "min_site"')
    assert ppes.PpesStep().done(ppes_copy, SYSTEM)


# --- only the fit changes: the calculations are reused (P4) -------------------------------------------

def files_of(run_dir):
    """Bytes of every file of the calculations of the step."""
    base = system_dir(run_dir, SYSTEM) / 'ppes'
    return {str(path.relative_to(base)): path.read_bytes() for path in sorted(base.rglob('*'))
            if path.is_file()}


@pytest.mark.parametrize('old, new', [('none', 'uber'), ('uber', 'none')])
def test_changing_only_the_fit_runs_the_analysis_again(ppes_copy, old, new):
    if old == 'uber':
        edit_config(ppes_copy, 'fit = "none"', 'fit = "uber"')
        ppes.PpesStep().analyse(ppes_copy, SYSTEM, finish_ppes(ppes_copy, prepare=False))
    step = ppes.PpesStep()
    assert step.done(ppes_copy, SYSTEM)
    plan, files, before = step.plan(ppes_copy, SYSTEM), files_of(ppes_copy), document(ppes_copy)

    edit_config(ppes_copy, f'fit = "{old}"', f'fit = "{new}"')
    # Not done any more, with the same calculations: nothing to launch.
    assert not step.done(ppes_copy, SYSTEM)
    assert step.plan(ppes_copy, SYSTEM) == plan
    calc_ids = step.prepare(ppes_copy, SYSTEM)
    assert calc_ids == list(plan)
    assert files_of(ppes_copy) == files
    # The analysis alone: the files of the calculations are not touched, the entries are written.
    step.analyse(ppes_copy, SYSTEM, calc_ids)
    assert files_of(ppes_copy) == files
    assert step.done(ppes_copy, SYSTEM)
    energy = ppes_doc(ppes_copy)['energy']['min_site']
    assert ('uber' in energy) == (new == 'uber')
    after = document(ppes_copy)
    own = ('ppes', 'regular', 'energy', 'min_site')
    assert without(after, own) == without(before, own)
    assert {k: v for k, v in energy.items() if k != 'uber'} == \
        {k: v for k, v in before['ppes']['regular']['energy']['min_site'].items() if k != 'uber'}


# --- provenance and the state of the step ------------------------------------------------------

def test_done_only_after_the_analysis(adhered_copy):
    step = ppes.PpesStep()
    assert step.done(adhered_copy, SYSTEM) is False
    calc_ids = finish_ppes(adhered_copy)
    assert step.done(adhered_copy, SYSTEM) is False
    step.analyse(adhered_copy, SYSTEM, calc_ids)
    assert step.done(adhered_copy, SYSTEM)


def test_not_done_without_the_adhesion(surfaces_copy):
    # Not even the inputs are read.
    assert ppes.PpesStep().done(surfaces_copy, SYSTEM) is False


@pytest.mark.parametrize('old, new', [
    ('kdens = 7.5', 'kdens = 8.0'),
    ('site = "min_site"', 'site = "max_site"'),
    ('\nfit = "none"', '\nshifts = [0.0, 1.0, 2.0]\nfit = "none"'),
])
def test_done_follows_the_settings(ppes_copy, old, new):
    edit_config(ppes_copy, old, new)
    assert not ppes.PpesStep().done(ppes_copy, SYSTEM)


def test_done_follows_the_results(ppes_copy):
    change_energy(ppes_copy, 'ppes/min_site_0.5')
    assert not ppes.PpesStep().done(ppes_copy, SYSTEM)


def test_done_follows_the_interface(ppes_copy):
    # The adhesion was analysed again and wrote another interface.
    with SystemWriter(system_dir(ppes_copy, SYSTEM)) as writer:
        writer.document.set(('structure', 'opt', 'regular', 'interface_min', 'charge'), 1)
        writer.save()
    assert not ppes.PpesStep().done(ppes_copy, SYSTEM)


# --- the chain behind the step (P1) ------------------------------------------------------------------

def test_done_follows_the_results_of_the_surface(ppes_copy):
    assert ppes.PpesStep().done(ppes_copy, SYSTEM)
    # The relaxed surface was completed again with another energy and not analysed: the entries
    # of the adhesion and of the step keep what they read, and the step is not done any more.
    change_energy(ppes_copy, surface_step('pes').calculations(ppes_copy, SYSTEM)[0].calc_id)
    assert adhesion.AdhesionStep().provenance(ppes_copy, SYSTEM)
    assert not adhesion.AdhesionStep().done(ppes_copy, SYSTEM)
    assert not ppes.PpesStep().done(ppes_copy, SYSTEM)


def test_done_follows_match(ppes_copy):
    edit_config(ppes_copy, 'max_area = 160.0', 'max_area = 150.0')
    assert not ppes.PpesStep().done(ppes_copy, SYSTEM)


@pytest.mark.parametrize('change', ['surface', 'match'])
def test_analysis_stops_when_the_chain_is_not_current(ppes_copy, change):
    calc_ids = finish_ppes(ppes_copy, prepare=False)
    if change == 'surface':
        change_energy(ppes_copy, surface_step('pes').calculations(ppes_copy, SYSTEM)[0].calc_id)
    else:
        edit_config(ppes_copy, 'max_area = 160.0', 'max_area = 150.0')
    before = system_bytes(ppes_copy)
    with pytest.raises(PrerequisiteError, match='upstream of adhesion not done or not current'):
        ppes.PpesStep().analyse(ppes_copy, SYSTEM, calc_ids)
    assert system_bytes(ppes_copy) == before


def test_analysis_stops_when_the_adhesion_is_not_current(ppes_copy):
    calc_ids = finish_ppes(ppes_copy, prepare=False)
    change_energy(ppes_copy, 'adhesion/interface_min')
    before = system_bytes(ppes_copy)
    with pytest.raises(PrerequisiteError, match=r'entries missing or not current: '
                                                r'structure\.opt\.regular\.interface_min'):
        ppes.PpesStep().analyse(ppes_copy, SYSTEM, calc_ids)
    assert system_bytes(ppes_copy) == before


# --- message of the analysis -------------------------------------------------------------------------

def test_message_of_the_analysis(adhered_copy, caplog):
    edit_config(adhered_copy, 'fit = "none"', 'fit = "uber"')
    calc_ids = finish_ppes(adhered_copy)
    with caplog.at_level(logging.INFO, logger=ppes.__name__):
        ppes.PpesStep().analyse(adhered_copy, SYSTEM, calc_ids)
    energy = ppes_doc(adhered_copy)['energy']['min_site']
    jm2 = energy['ppes_jm2']
    low, high = min(jm2, key=lambda row: row[1]), max(jm2, key=lambda row: row[1])
    fit = energy['uber']
    lines = [record.getMessage() for record in caplog.records if record.name == ppes.__name__]
    assert lines == [
        f'{SYSTEM}: ppes: regular PPES at min_site, 15 shifts of the slab on top from -0.75 to 3.0 '
        'angstrom (slab_from_scratch)',
        f'{SYSTEM}: total energy per area: lowest {low[1]:.6f} J/m^2 at the shift {low[0]}, '
        f'highest {high[1]:.6f} J/m^2 at the shift {high[0]}',
        f'{SYSTEM}: fit uber: g = {fit["g"]:.6g} +- {fit["sigma_g"]:.3g} eV per cell, '
        f'l = {fit["l"]:.6g} +- {fit["sigma_l"]:.3g} angstrom',
    ]


def test_message_without_a_fit(adhered_copy, caplog):
    calc_ids = finish_ppes(adhered_copy)
    with caplog.at_level(logging.INFO, logger=ppes.__name__):
        ppes.PpesStep().analyse(adhered_copy, SYSTEM, calc_ids)
    lines = [record.getMessage() for record in caplog.records if record.name == ppes.__name__]
    assert len(lines) == 3 and lines[-1] == f'{SYSTEM}: no fit (fit = "none")'


# --- energies of different models are not combined ----------------------------------------------------

@pytest.mark.parametrize('change', ['version', 'pseudopotential', 'k-points', 'spin', 'cutoff',
                                    'smearing'])
def test_energies_of_different_models_are_not_combined(adhered_copy, change):
    calc_ids = finish_ppes(adhered_copy)
    sdir = system_dir(adhered_copy, SYSTEM)
    record = read_record(sdir, calc_ids[3])
    changed = change_model(record.energy_model, change)
    assert changed != record.energy_model
    write_record(sdir, dataclasses.replace(record, energy_model=changed))
    before = system_bytes(adhered_copy)
    with pytest.raises(StoreError, match='different energy models'):
        ppes.PpesStep().analyse(adhered_copy, SYSTEM, calc_ids)
    assert system_bytes(adhered_copy) == before


# --- errors of the analysis ------------------------------------------------------------------------

def test_incomplete_record_stops_the_analysis(adhered_copy):
    calc_ids = finish_ppes(adhered_copy)
    sdir = system_dir(adhered_copy, SYSTEM)
    record = read_record(sdir, calc_ids[4])
    write_record(sdir, dataclasses.replace(
        record, status=Status('ended', 'converged', 'not_converged', None), results=None,
        observed=None))
    before = system_bytes(adhered_copy)
    with pytest.raises(StoreError, match=rf'{re.escape(calc_ids[4])}: not complete'):
        ppes.PpesStep().analyse(adhered_copy, SYSTEM, calc_ids)
    assert system_bytes(adhered_copy) == before


def test_calculations_of_the_settings_only(adhered_copy):
    calc_ids = finish_ppes(adhered_copy)
    with pytest.raises(ValueError, match='not those of the settings'):
        ppes.PpesStep().analyse(adhered_copy, SYSTEM, calc_ids[:-1])


def test_missing_record(adhered_copy):
    calc_ids = finish_ppes(adhered_copy)
    record_path(system_dir(adhered_copy, SYSTEM), calc_ids[0]).unlink()
    with pytest.raises(StoreError, match=rf'{re.escape(calc_ids[0])}: no record'):
        ppes.PpesStep().analyse(adhered_copy, SYSTEM, calc_ids)


def test_a_calculation_without_energy(adhered_copy):
    calc_ids = finish_ppes(adhered_copy, energy=lambda shift: None if shift == 0.5 else -1.0)
    before = system_bytes(adhered_copy)
    with pytest.raises(StoreError, match='no energy in the results of the shift 0.5'):
        ppes.PpesStep().analyse(adhered_copy, SYSTEM, calc_ids)
    assert system_bytes(adhered_copy) == before


def test_failure_after_the_writes_saves_nothing(adhered_copy, monkeypatch):
    calc_ids = finish_ppes(adhered_copy)
    before = system_bytes(adhered_copy)
    seen = []

    def fail(store_, ids, kind, site, shifts, fit):
        seen.append(store_.document.get(('ppes', kind, 'data', site)))
        raise RuntimeError('arrange_data failed')

    monkeypatch.setattr(ppes, 'arrange_data', fail)
    with pytest.raises(RuntimeError, match='arrange_data failed'):
        ppes.PpesStep().analyse(adhered_copy, SYSTEM, calc_ids)
    # The results of the shifts were in the document being written; none reached the disk.
    assert seen and len(seen[0]) == 15
    assert system_bytes(adhered_copy) == before


def test_failure_after_the_emptying_saves_nothing(ppes_copy, monkeypatch):
    calc_ids = finish_ppes(ppes_copy, prepare=False)
    before = system_bytes(ppes_copy)
    seen = []

    def fail(store_, ids, kind, site, calc_id_, *args):
        seen.append(store_.document.get(('ppes', kind, 'data', site)))
        raise RuntimeError('move_results failed')

    monkeypatch.setattr(ppes, 'move_results', fail)
    with pytest.raises(RuntimeError, match='move_results failed'):
        ppes.PpesStep().analyse(ppes_copy, SYSTEM, calc_ids)
    # The entries of the site had been emptied in memory, not on the disk.
    assert seen == [{}]
    assert system_bytes(ppes_copy) == before


# --- the command line and the orchestrator -------------------------------------------------------------

def orchestrated_chain(tmp_path, fake_slurm):
    """match, the static surface, the adhesion and the step: a pass at a time.

    Returns
    -------
    run_dir, orchestrator, settle
        With the four steps completed; ``settle(calc_ids, status)`` completes calculations.
    """
    run_dir = pes_run(tmp_path, steps=('ppes', 'adhesion', 'pes_scf', 'match'))
    configure(run_dir, 'pes_scf', 'regular')
    orch = build_orchestrator(run_dir, groups=2)
    assert orch.order == ['match', 'pes_scf', 'adhesion', 'ppes']

    def statuses():
        return {name: orch.entry(SYSTEM, name)['status'] for name in orch.order}

    def settle(calc_ids, status):
        """The workers have run these calculations: results and finished jobs."""
        for calc_id in calc_ids:
            complete(run_dir, calc_id, status=status)
        fake_slurm.states.update({job_id: 'COMPLETED' for job_id in orch.manager.jobs})

    orch.run_pass()
    assert statuses() == {'match': 'completed', 'pes_scf': 'running', 'adhesion': 'waiting',
                          'ppes': 'waiting'}
    settle([c.calc_id for c in pes.StaticPesStep().calculations(run_dir, SYSTEM)],
           status_of('pes_scf'))
    finish(run_dir, 'pes_scf', prepare=False)
    orch.run_pass()
    assert statuses() == {'match': 'completed', 'pes_scf': 'completed', 'adhesion': 'running',
                          'ppes': 'waiting'}
    four = [f'adhesion/{label}' for label in adhesion.LABELS]
    settle(four, RELAX_DONE)
    finish_adhesion(run_dir, prepare=False)
    orch.run_pass()
    assert statuses() == {'match': 'completed', 'pes_scf': 'completed', 'adhesion': 'completed',
                          'ppes': 'running'}
    shifts = [calc.calc_id for calc in ppes.PpesStep().calculations(run_dir, SYSTEM)]
    assert shifts == [f'ppes/{label}' for label in SHIFT_LABELS]
    settle(shifts, status_of('pes_scf'))
    finish_ppes(run_dir, prepare=False)
    report = orch.run_pass()
    assert report.success and set(statuses().values()) == {'completed'}
    assert ppes.PpesStep().done(run_dir, SYSTEM)
    assert set(document(run_dir)['ppes']) == {'regular'}
    return run_dir, orch


def statuses_of(orch):
    return {name: orch.entry(SYSTEM, name)['status'] for name in orch.order}


def test_the_orchestrator_runs_the_chain(tmp_path, fake_slurm):
    orchestrated_chain(tmp_path, fake_slurm)


def test_the_orchestrator_does_not_keep_the_step_of_a_surface_not_current(tmp_path, fake_slurm):
    run_dir, orch = orchestrated_chain(tmp_path, fake_slurm)
    # The static surface is to be computed on reduced slabs: it is no longer current, and
    # neither is what was computed from it, whatever the entries that were read.
    edit_config(run_dir, 'short = false', 'short = true')
    assert not pes.StaticPesStep().done(run_dir, SYSTEM)
    orch.run_pass()
    statuses = statuses_of(orch)
    assert statuses['match'] == 'completed'
    assert statuses['pes_scf'] != 'completed'
    assert statuses['adhesion'] != 'completed'
    assert statuses['ppes'] != 'completed'


def test_changing_only_the_fit_runs_the_analysis_alone(tmp_path, fake_slurm):
    run_dir, orch = orchestrated_chain(tmp_path, fake_slurm)
    jobs, files = dict(orch.manager.jobs), files_of(run_dir)
    edit_config(run_dir, 'fit = "none"', 'fit = "uber"')
    assert not ppes.PpesStep().done(run_dir, SYSTEM)
    report = orch.run_pass()
    # No calculation to run and no job: the step goes to its analysis at once.
    assert report.success and set(statuses_of(orch).values()) == {'completed'}
    assert orch.manager.jobs == jobs and files_of(run_dir) == files
    assert 'uber' in ppes_doc(run_dir)['energy']['min_site']
    assert ppes.PpesStep().done(run_dir, SYSTEM)


def cli(run_dir, *argv):
    return main([*argv, '-r', str(run_dir)])


def test_the_step_from_the_command_line(tmp_path, capsys):
    run_dir = pes_run(tmp_path, steps=('match', 'pes', 'adhesion', 'ppes'))
    assert cli(run_dir, 'analyse', 'match') == 0
    for name in ('pes',):
        assert cli(run_dir, 'prepare', name) == 0
        for calc in pes.RelaxedPesStep().calculations(run_dir, SYSTEM):
            complete(run_dir, calc.calc_id, status=status_of(name))
        finish(run_dir, name, prepare=False)
        assert cli(run_dir, 'analyse', name) == 0
    assert cli(run_dir, 'prepare', 'adhesion') == 0
    for label in adhesion.LABELS:
        complete(run_dir, f'adhesion/{label}', status=RELAX_DONE)
    finish_adhesion(run_dir, prepare=False)
    assert cli(run_dir, 'analyse', 'adhesion') == 0
    capsys.readouterr()
    assert cli(run_dir, 'prepare', 'ppes') == 0
    assert '15 calculations prepared' in capsys.readouterr().out
    for shift in PPES_SHIFTS:
        complete(run_dir, f'ppes/min_site_{shift}', status=status_of('pes_scf'))
    finish_ppes(run_dir, prepare=False)
    assert cli(run_dir, 'analyse', 'ppes') == 0
    assert f'{SYSTEM}/ppes: completed' in capsys.readouterr().out
    assert ppes.PpesStep().done(run_dir, SYSTEM)
    assert set(document(run_dir)['ppes']['regular']['data']) == {'min_site'}


def test_the_step_waits_for_the_steps_behind_it(tmp_path, capsys):
    run_dir = pes_run(tmp_path, steps=('match', 'pes', 'adhesion', 'ppes'))
    assert cli(run_dir, 'analyse', 'match') == 0
    capsys.readouterr()
    # The surface is not there yet: the steps behind the adhesion are named.
    assert cli(run_dir, 'prepare', 'ppes') == 1
    out = capsys.readouterr().out
    assert f'{SYSTEM}/ppes: blocked' in out
    assert 'steps upstream of adhesion not done or not current: pes; run them first' in out
    # The surface is there and the adhesion is not: its interface is.
    assert cli(run_dir, 'prepare', 'pes') == 0
    for calc in pes.RelaxedPesStep().calculations(run_dir, SYSTEM):
        complete(run_dir, calc.calc_id, status=status_of('pes'))
    finish(run_dir, 'pes', prepare=False)
    assert cli(run_dir, 'analyse', 'pes') == 0
    capsys.readouterr()
    assert cli(run_dir, 'prepare', 'ppes') == 1
    out = capsys.readouterr().out
    assert 'entries missing or not current: structure.opt.regular.interface_min' in out