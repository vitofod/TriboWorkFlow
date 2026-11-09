# Author: Vito Foderà
# Created: 2026-10-23

"""Tests of the TOML settings (``triboworkflow.core.settings``).

The examples load, the pw.x keys pass as given, and every validation error
is reported with the TOML path of the field.
"""

import copy
import dataclasses
import re
import tomllib
from pathlib import Path

import pytest

from triboworkflow.core.settings import (Settings, SettingsError, load_settings,
                                         settings_from_dict, validate)
from triboworkflow.core.settings.schema import (CHARGE_OPERATIONS, CHARGE_SITES,
                                                INTER_PARAMS_DEFAULTS, INTERFACE_STEPS,
                                                PPES_SHIFTS, SHEAR_DELTA, SHEAR_MEP, SHEAR_SEED,
                                                AdhesionParams, AnalysisParams, ChargeParams,
                                                GenerationParams, InterParams, PesScfParams,
                                                PpesParams, ShearMepParams, ShearParams, key_name)
from triboworkflow.execution.orchestrator import INTERFACE_GRAPH
from triboworkflow.vendored.highput.utils.tasktools import read_default_params
from triboworkflow.vendored.physics.tribology.mep import MEP

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / 'examples'
# Defaults of the interface matching, read by the vendored read_default_params.
MATCHING_DEFAULTS = ROOT / 'triboworkflow' / 'vendored' / 'highput' / 'firetasks' / 'defaults.json'

DELETE = object()


def read_example(name):
    with open(EXAMPLES / name, 'rb') as f:
        return tomllib.load(f)


WORKFLOW = read_example('interface.toml')
SINGLE = read_example('single_input.toml')


def edited(base, edits):
    """Copy of ``base`` with ``{'a.b[0].c': value}`` applied (DELETE removes the key).

    Path parts in double quotes are taken whole, e.g. 'namelists.SYSTEM."london_c6(1)"'.
    """
    data = copy.deepcopy(base)
    for path, value in edits.items():
        keys = [part.strip('"') for part in re.findall(r'"[^"]*"|[^.\[\]]+', path)]
        keys = [int(k) if k.isdigit() else k for k in keys]
        node = data
        for key in keys[:-1]:
            node = node.setdefault(key, {}) if isinstance(node, dict) else node[key]
        if value is DELETE:
            del node[keys[-1]]
        else:
            node[keys[-1]] = value
    return data


# --- examples -------------------------------------------------------------------

def test_interface_example_complete():
    settings = load_settings(EXAMPLES / 'interface.toml')
    settings.require_complete()
    assert settings.source == EXAMPLES / 'interface.toml'
    assert [slab.mid for slab in settings.interface.slabs] == ['mp-30', 'mp-134']
    assert settings.namelists.SYSTEM['ecutwfc'] == 50.0
    assert settings.namelists.CONTROL['tprnfor'] is True
    assert [s.label for s in settings.atomic_species] == ['Cu', 'Al']
    assert settings.k_points.mode == 'kdens'
    assert settings.interface.engine_for('pes') == 'qe'


def test_single_input_template_placeholders():
    settings = load_settings(EXAMPLES / 'single_input.toml')
    assert settings.missing_values() == [
        'atomic_species[0].label', 'atomic_species[0].pseudopotential', 'structure.file',
        'namelists.SYSTEM.ecutwfc', 'namelists.SYSTEM.ecutrho']
    with pytest.raises(SettingsError, match='fill in before generating an input: '
                                            r'atomic_species\[0\].label, '):
        settings.require_complete()


def test_placeholders_in_namelists():
    data = edited(WORKFLOW, {'namelists.CONTROL.pseudo_dir': '', 'namelists.SYSTEM.ECUTRHO': 0})
    assert settings_from_dict(data).missing_values() == [
        'namelists.CONTROL.pseudo_dir', 'namelists.SYSTEM.ECUTRHO']


@pytest.mark.parametrize('line', [
    'grid = [, , ]',
    'overrides = [ { atom = 6, 7, 8; if_pos = [0, 0, 0] } ]',
], ids=['grid', 'override'])
def test_invalid_toml_of_the_original_template(tmp_path, line):
    path = tmp_path / 'settings.toml'
    path.write_text(f'schema_version = 1\n[table]\n{line}\n')
    with pytest.raises(SettingsError, match=r'settings.toml: invalid TOML: .*line 3'):
        load_settings(path)


def test_missing_file(tmp_path):
    with pytest.raises(SettingsError, match='cannot read the settings'):
        load_settings(tmp_path / 'missing.toml')


def test_error_names_the_file(tmp_path):
    path = tmp_path / 'bad.toml'
    path.write_text((EXAMPLES / 'interface.toml').read_text().replace('scheduler = "slurm"',
                                                                       'scheduler = "slurmx"'))
    with pytest.raises(SettingsError, match=r"bad.toml: execution.scheduler: 'slurmx' is not allowed"):
        load_settings(path)


def test_integers_accepted_for_reals_of_the_schema():
    settings = settings_from_dict(edited(WORKFLOW, {'atomic_species[0].mass': 63}))
    assert isinstance(settings.atomic_species[0].mass, float)


def test_settings_from_dict_returns_settings():
    assert isinstance(settings_from_dict(copy.deepcopy(SINGLE)), Settings)


def test_generation_defaults_with_structure_only():
    single = settings_from_dict(edited(SINGLE, {'generation': DELETE}))
    assert single.generation == GenerationParams()
    assert single.generation.file_prefix == 'input.pwi' and single.generation.overwrite is False
    assert settings_from_dict(copy.deepcopy(WORKFLOW)).generation is None


def test_validate_after_changes():
    settings = settings_from_dict(copy.deepcopy(WORKFLOW))
    settings.interface.slabs.pop()
    with pytest.raises(SettingsError, match='give two slabs'):
        validate(settings)


# --- pw.x keys pass as given -----------------------------------------------------

def test_namelists_written_as_given():
    # Grimme-D2 with only vdw_corr works as in pw.x, which uses its defaults;
    # the per-species parameters are written as in a pwi, indexed by species.
    data = edited(WORKFLOW, {'namelists.SYSTEM.vdw_corr': 'grimme-d2',
                             'namelists.SYSTEM."london_c6(1)"': 159.927,
                             'namelists.SYSTEM."london_rvdw(1)"': 3.014,
                             'namelists.ELECTRONS.mixing_ndim': 12})
    namelists = settings_from_dict(data).namelists
    assert namelists.SYSTEM['vdw_corr'] == 'grimme-d2'
    assert (namelists.SYSTEM['london_c6(1)'], namelists.SYSTEM['london_rvdw(1)']) == (159.927, 3.014)
    assert 'london_s6' not in namelists.SYSTEM and 'london_rcut' not in namelists.SYSTEM
    assert namelists.ELECTRONS['mixing_ndim'] == 12


def test_namelist_values_keep_their_type():
    settings = settings_from_dict(edited(WORKFLOW, {'namelists.SYSTEM.ecutwfc': 50}))
    assert settings.namelists.SYSTEM['ecutwfc'] == 50 and isinstance(settings.namelists.SYSTEM['ecutwfc'], int)


def test_namelist_get_ignores_case():
    settings = settings_from_dict(edited(SINGLE, {'namelists.CONTROL.calculation': DELETE,
                                                  'namelists.CONTROL.Calculation': 'scf'}))
    assert settings.namelists.get('CONTROL', 'calculation') == 'scf'
    assert key_name(' London_C6(2) ') == 'london_c6'


# Values that pw.x, the interface matching or the user judge: the settings take them as given.
ACCEPTED = [
    ('d2-only', W := WORKFLOW, {'namelists.SYSTEM.vdw_corr': 'grimme-d2'}),
    ('d2-keys-ignored', W, {'namelists.SYSTEM.london_s6': 0.75, 'namelists.SYSTEM.dftd3_version': 4}),
    ('d3-threebody-v2', W, {'namelists.SYSTEM.vdw_corr': 'grimme-d3', 'namelists.SYSTEM.dftd3_version': 2,
                            'namelists.SYSTEM.dftd3_threebody': True}),
    ('vdw-alias', W, {'namelists.SYSTEM.vdw_corr': 'DFT-D'}),
    ('c6-negative', W, {'namelists.SYSTEM."london_c6(2)"': -1.0}),
    # Changed by the relaunches: checked, but only what makes no sense is refused.
    ('nstep-large', W, {'namelists.CONTROL.nstep': 500}),
    ('nstep-1', W, {'namelists.CONTROL.nstep': 1}),
    ('mixing-beta-1', W, {'namelists.ELECTRONS.mixing_beta': 1}),
    ('mixing-beta-small', W, {'namelists.ELECTRONS.mixing_beta': 0.01}),
    ('smearing', W, {'namelists.SYSTEM.smearing': 'cold'}),
    ('occupations', W, {'namelists.SYSTEM.occupations': 'fixed'}),
    ('no-ecut', W, {'namelists.SYSTEM': {}}),
    ('no-namelists', W, {'namelists': DELETE}),
    ('typo-in-pw-key', W, {'namelists.SYSTEM.ecutwf': 50.0}),
    # Hubbard is not implemented, but its keys go to pw.x like the others.
    ('hubbard-keys', W, {'namelists.SYSTEM.lda_plus_u': True, 'namelists.SYSTEM."Hubbard_U(1)"': 4.2,
                         'namelists.SYSTEM."Hubbard_alpha(1)"': 0.1}),
    ('alternatives-to-ibrav', W, {'namelists.SYSTEM.space_group': 225,
                                  'namelists.SYSTEM."starting_ns_eigenvalue(1,1,1)"': 0.0}),
    ('nspin1-magnetization', W, {'namelists.SYSTEM.nspin': 1}),
    ('no-magnetization', W, {'atomic_species[1].starting_magnetization': DELETE}),
    ('magnetization-range', W, {'atomic_species[0].starting_magnetization': 3.0}),
    ('mass-0', W, {'atomic_species[0].mass': 0.0}),
    ('label', W, {'atomic_species[0].label': 'Xx12'}),
    ('grid-with-gamma', W, {'k_points': {'mode': 'gamma', 'grid': [4, 4, 1]}}),
    ('grid-values', W, {'k_points': {'mode': 'automatic', 'grid': [0, 8, 1], 'offset': [2, 0, 0]}}),
    ('kdens-0', W, {'k_points.kdens': 0.0}),
    ('vectors-with-file', S := SINGLE, {'structure.cell.vectors': [[1, 0, 0], [0, 1, 0], [1, 1, 0]]}),
    ('frame', S, {'structure.frame': -1}),
    ('if-pos-values', S, {'structure.constraints.default_if_pos': [2, 1, 1]}),
    ('override-repeated', S, {'structure.constraints.overrides': [{'atoms': [2, 2], 'if_pos': [0, 0, 0]}]}),
    ('generation-in-workflow', W, {'generation': {}}),
    ('max-seconds', W, {'namelists.CONTROL.max_seconds': 90000}),
    ('wfcdir-inside', W, {'namelists.CONTROL.wfcdir': 'WFC/node'}),
    ('bulk-modulus-unused', W, {'interface.set_bm': False}),
    ('weights-with-set-bm', W, {'interface.inter_params.strain_weight_1': 1.0}),
    ('inter-params-values', W, {'interface.inter_params.max_area': 0.0,
                                'interface.inter_params.max_angle_tol': -1.0}),
    ('bulk-modulus-0', W, {'interface.slabs[0].bulk_modulus': 0.0}),
    ('miller-zero', W, {'interface.slabs[1].miller': [0, 0, 0]}),
    ('formula-and-name', W, {'interface.slabs[0].formula': 'Cu', 'interface.slabs[0].name': 'a',
                             'interface.slabs[1].formula': 'Al', 'interface.slabs[1].name': 'b'}),
    ('qe-without-manifest', W, {'qe.upf_md5': DELETE}),
    ('match-only-without-qe', W, {'qe': DELETE, 'interface.steps': ['match', 'shear']}),
    ('single-without-qe', S, {'qe': DELETE}),
    ('single-manifest-only', S, {'qe.version': DELETE, 'qe.upf_md5': 'pseudo.md5'}),
]


@pytest.mark.parametrize('base, edits', [case[1:] for case in ACCEPTED], ids=[case[0] for case in ACCEPTED])
def test_accepted_as_given(base, edits):
    assert isinstance(settings_from_dict(edited(base, edits)), Settings)


@pytest.mark.parametrize('edits, notes', [
    ({}, []),
    ({'namelists.CONTROL.nstep': 250, 'namelists.ELECTRONS.mixing_beta': 0.1}, []),
    ({'namelists.CONTROL.nstep': 251}, [
        'namelists.CONTROL.nstep = 251 is above 250: the recommended value is 150; the '
        'relaunches never lower it']),
    ({'namelists.ELECTRONS.mixing_beta': 0.05}, [
        'namelists.ELECTRONS.mixing_beta = 0.05 is below 0.1: the recommended value is 0.4; the '
        'relaunches never raise it']),
])
def test_advice_on_the_values_changed_by_the_relaunches(edits, notes, caplog):
    caplog.set_level('WARNING')
    settings_from_dict(edited(WORKFLOW, edits))
    assert [record.getMessage() for record in caplog.records] == notes


@pytest.mark.parametrize('version, warned', [
    ('7.1', False), ('7.2', False), ('7.3', False), ('7.1.2', False), ('7.3.1', False),
    ('7.0', True), ('7.4', True), ('8.0', True), ('8.1.2', True), ('6.8', True),
])
def test_qe_versions_accepted_with_a_warning_outside_those_supported(version, warned, caplog):
    caplog.set_level('WARNING')
    settings = settings_from_dict(edited(WORKFLOW, {'qe.version': version}))
    assert settings.qe.version == version
    notes = [record.getMessage() for record in caplog.records]
    assert notes == ([f"qe.version = '{version}': the workflow is built to work with QE 7.1.x, "
                      f'7.2.x, 7.3.x; with version {version} the results are not guaranteed and '
                      'may hold errors: check them'] if warned else [])


@pytest.mark.parametrize('k_points', [
    {'mode': 'gamma'},
    {'mode': 'automatic', 'grid': [8, 8, 1], 'offset': [1, 1, 0]},
    {'mode': 'kdens', 'kdens': 4.0, 'gamma_for_unshifted_1x1x1': False},
])
def test_k_points_modes(k_points):
    settings = settings_from_dict(edited(WORKFLOW, {'k_points': k_points}))
    assert settings.k_points.mode == k_points['mode']


def test_strain_weights_without_bulk_moduli():
    data = edited(WORKFLOW, {
        'interface.set_bm': False,
        'interface.slabs[0].bulk_modulus': DELETE, 'interface.slabs[1].bulk_modulus': DELETE,
        'interface.inter_params.strain_weight_1': 1, 'interface.inter_params.strain_weight_2': 2})
    params = settings_from_dict(data).interface.inter_params.given()
    assert (params['strain_weight_1'], params['strain_weight_2']) == (1.0, 2.0)


def test_max_seconds_margin_default():
    settings = settings_from_dict(edited(WORKFLOW, {'execution.max_seconds_margin': DELETE}))
    assert settings.execution.max_seconds_margin == 600


def test_job_report_wait_default():
    settings = settings_from_dict(edited(WORKFLOW, {'execution.job_report_wait': DELETE}))
    assert settings.execution.job_report_wait == 3600


def test_pp_command_is_optional():
    assert settings_from_dict(WORKFLOW).execution.pp_command is None
    settings = settings_from_dict(edited(WORKFLOW, {'execution.pp_command': 'srun pp.x'}))
    assert settings.execution.pp_command == 'srun pp.x'


def test_analysis_job_is_optional():
    # Without [execution.analysis] the costly analyses are left to the user.
    assert settings_from_dict(WORKFLOW).execution.analysis is None
    settings = settings_from_dict(edited(WORKFLOW, {'execution.analysis.jobscript': 'analysis.sh'}))
    assert settings.execution.analysis == AnalysisParams(jobscript='analysis.sh')


def test_analysis_jobscript_placeholder():
    # As the jobscript of [execution], an empty one is a placeholder.
    data = edited(WORKFLOW, {'execution.analysis.jobscript': ''})
    assert settings_from_dict(data).missing_values() == ['execution.analysis.jobscript']


def test_engine_per_step():
    settings = settings_from_dict(edited(WORKFLOW, {'interface.engines': {'ppes': 'qe'}}))
    assert settings.interface.engine_for('ppes') == 'qe'


# --- steps and groups ---------------------------------------------------------------

def test_interface_steps_required():
    with pytest.raises(SettingsError, match=r'interface\.steps'):
        settings_from_dict(edited(WORKFLOW, {'interface.steps': DELETE}))


def test_interface_steps_kept_in_the_given_order():
    settings = settings_from_dict(edited(WORKFLOW, {'interface.steps': ['adhesion', 'match']}))
    assert settings.interface.steps == ['adhesion', 'match']


@pytest.mark.parametrize('steps, message', [
    (['match', 'shearstr'],
     "interface.steps.*'shearstr'.*"
     "match, pes, pes_scf, plotpes, plotpes_scf, shear, adhesion, ppes, charge"),
    (['pes', 'pes'], r'interface\.steps.*repeated.*pes'),
    ([], r'interface\.steps.*at least one'),
])
def test_interface_steps_errors(steps, message):
    with pytest.raises(SettingsError, match=message):
        settings_from_dict(edited(WORKFLOW, {'interface.steps': steps}))


def test_interface_steps_match_the_graph():
    assert INTERFACE_STEPS == tuple(INTERFACE_GRAPH)


@pytest.mark.parametrize('steps', [['match', 'pes'], ['match', 'pes_scf'],
                                   ['match', 'pes', 'pes_scf']])
def test_relaxed_and_static_pes_are_two_steps(steps):
    # Requested one, the other or both; no key chooses between them.
    assert settings_from_dict(edited(WORKFLOW, {'interface.steps': steps})).interface.steps == steps


def test_static_pes_on_complete_slabs_by_default():
    assert settings_from_dict(WORKFLOW).interface.pes_scf == PesScfParams(short=False, layers=None)
    settings = settings_from_dict(edited(WORKFLOW, {'interface.pes_scf': DELETE}))
    assert settings.interface.pes_scf == PesScfParams(short=False, layers=None)


def test_static_pes_on_reduced_slabs():
    settings = settings_from_dict(edited(WORKFLOW, {'interface.pes_scf.short': True}))
    assert settings.interface.pes_scf == PesScfParams(short=True, layers=None)
    settings = settings_from_dict(edited(WORKFLOW, {'interface.pes_scf.short': True,
                                                    'interface.pes_scf.layers': [4, 2]}))
    assert settings.interface.pes_scf == PesScfParams(short=True, layers=[4, 2])


def test_adhesion_from_the_relaxed_pes_by_default():
    # The defaults of the original workflow: the relaxed surface, complete slabs.
    default = AdhesionParams(pes='pes', type='regular')
    assert settings_from_dict(WORKFLOW).interface.adhesion == default
    settings = settings_from_dict(edited(WORKFLOW, {'interface.adhesion': DELETE}))
    assert settings.interface.adhesion == default


@pytest.mark.parametrize('edits, expected', [
    ({'interface.adhesion.pes': 'pes_scf'}, AdhesionParams(pes='pes_scf', type='regular')),
    ({'interface.adhesion.pes': 'pes_scf', 'interface.adhesion.type': 'short',
      'interface.pes_scf.short': True}, AdhesionParams(pes='pes_scf', type='short')),
    ({'interface.adhesion.pes': 'pes', 'interface.adhesion.type': 'regular'},
     AdhesionParams(pes='pes', type='regular')),
])
def test_adhesion_combinations_allowed(edits, expected):
    assert settings_from_dict(edited(WORKFLOW, edits)).interface.adhesion == expected


def test_ppes_defaults_are_the_command_line_of_the_original_workflow():
    # The minimum site, the 15 shifts of the command line (they skip 2.5) and no fit.
    default = PpesParams(site='min_site', shifts=list(PPES_SHIFTS), fit='none')
    assert PPES_SHIFTS == (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
                           2.25, 2.75, 3.0)
    assert 2.5 not in PPES_SHIFTS
    assert settings_from_dict(WORKFLOW).interface.ppes == default
    settings = settings_from_dict(edited(WORKFLOW, {'interface.ppes': DELETE}))
    assert settings.interface.ppes == default
    # The defaults of a table are not shared between the runs.
    assert PpesParams().shifts == default.shifts
    assert PpesParams().shifts is not PpesParams().shifts


@pytest.mark.parametrize('edits, expected', [
    ({'interface.ppes.site': 'max_site'},
     PpesParams(site='max_site', shifts=list(PPES_SHIFTS), fit='none')),
    ({'interface.ppes.fit': 'uber'}, PpesParams(site='min_site', shifts=list(PPES_SHIFTS), fit='uber')),
    # Integers are read as floats, in the order given.
    ({'interface.ppes.shifts': [3, 0, -1, 0.5]}, PpesParams(shifts=[3.0, 0.0, -1.0, 0.5])),
    ({'interface.ppes.shifts': [1.0]}, PpesParams(shifts=[1.0])),
])
def test_ppes_choices_allowed(edits, expected):
    ppes = settings_from_dict(edited(WORKFLOW, edits)).interface.ppes
    assert ppes == expected
    assert all(type(shift) is float for shift in ppes.shifts)


def test_ppes_does_not_change_the_other_tables():
    settings = settings_from_dict(edited(WORKFLOW, {'interface.ppes.fit': 'uber'}))
    reference = settings_from_dict(WORKFLOW)
    assert settings.interface.adhesion == reference.interface.adhesion
    assert settings.interface.pes_scf == reference.interface.pes_scf
    assert settings.interface.engine_for('ppes') == 'qe'


def test_charge_defaults_are_the_command_line_of_the_original_workflow():
    # The minimum site and the operations '[-, -]' of the command line: interface minus the
    # bottom slab minus the top slab.
    default = ChargeParams(site='min_site', operations=['-', '-'])
    assert (CHARGE_SITES, CHARGE_OPERATIONS) == (('min_site', 'max_site'), ('+', '-'))
    assert settings_from_dict(WORKFLOW).interface.charge == default
    settings = settings_from_dict(edited(WORKFLOW, {'interface.charge': DELETE}))
    assert settings.interface.charge == default
    # The defaults of a table are not shared between the runs.
    assert ChargeParams().operations == ['-', '-']
    assert ChargeParams().operations is not ChargeParams().operations


@pytest.mark.parametrize('edits, expected', [
    ({'interface.charge.site': 'max_site'}, ChargeParams(site='max_site', operations=['-', '-'])),
    ({'interface.charge.operations': ['+', '-']},
     ChargeParams(site='min_site', operations=['+', '-'])),
    ({'interface.charge.site': 'max_site', 'interface.charge.operations': ['-', '+']},
     ChargeParams(site='max_site', operations=['-', '+'])),
])
def test_charge_choices_allowed(edits, expected):
    assert settings_from_dict(edited(WORKFLOW, edits)).interface.charge == expected


def test_charge_does_not_change_the_other_tables():
    settings = settings_from_dict(edited(WORKFLOW, {'interface.charge.site': 'max_site'}))
    reference = settings_from_dict(WORKFLOW)
    assert settings.interface.adhesion == reference.interface.adhesion
    assert settings.interface.ppes == reference.interface.ppes
    assert settings.interface.engine_for('charge') == 'qe'


def test_shear_defaults_are_the_values_that_the_original_workflow_used():
    # The MEP of the original ran with the values of its own file of defaults, not with those
    # that the command line announced (h 0.001, tol 1e-7): they are the recommended ones. The
    # delta of the shear strength is the one of the command line.
    effective = read_default_params(MEP.defaults, 'zerotemp', {})
    assert effective == {'npts': 101, 'extent': [1.5, 1.5], 'h': 0.01, 'nstepmax': 99999,
                         'tol': 1e-8, 'delta': 1e-4}
    assert dataclasses.asdict(ShearMepParams()) == effective
    assert SHEAR_MEP == {**effective, 'extent': (1.5, 1.5)}
    assert (SHEAR_DELTA, SHEAR_SEED) == (0.01, 0)
    default = ShearParams(delta=0.01, seed=0, mep=ShearMepParams(**effective))
    assert ShearParams() == default
    assert settings_from_dict(WORKFLOW).interface.shear == default
    settings = settings_from_dict(edited(WORKFLOW, {'interface.shear': DELETE}))
    assert settings.interface.shear == default
    # The defaults of a table are not shared between the runs.
    assert ShearParams().mep.extent == [1.5, 1.5]
    assert ShearParams().mep.extent is not ShearParams().mep.extent
    assert ShearParams().mep is not ShearParams().mep


@pytest.mark.parametrize('edits, expected', [
    ({'interface.shear.delta': 0.02}, ShearParams(delta=0.02)),
    # An integer is read as a float.
    ({'interface.shear.delta': 1}, ShearParams(delta=1.0)),
    ({'interface.shear.seed': 7}, ShearParams(seed=7)),
    # The seed of NumPy goes from 0 to 2**32 - 1.
    ({'interface.shear.seed': 0}, ShearParams(seed=0)),
    ({'interface.shear.seed': 4294967295}, ShearParams(seed=4294967295)),
    ({'interface.shear.mep.npts': 51}, ShearParams(mep=ShearMepParams(npts=51))),
    # The least that the entries of the shear strength can be written with (A98).
    ({'interface.shear.mep.npts': 6}, ShearParams(mep=ShearMepParams(npts=6))),
    ({'interface.shear.mep.extent': [1, 2.5]}, ShearParams(mep=ShearMepParams(extent=[1.0, 2.5]))),
    ({'interface.shear.mep.h': 0.001, 'interface.shear.mep.tol': 1e-7,
      'interface.shear.mep.delta': 0.01},
     ShearParams(mep=ShearMepParams(h=0.001, tol=1e-7, delta=0.01))),
    ({'interface.shear.mep.nstepmax': 1}, ShearParams(mep=ShearMepParams(nstepmax=1))),
    ({'interface.shear.delta': 0.5, 'interface.shear.seed': 3, 'interface.shear.mep.npts': 11,
      'interface.shear.mep.extent': [2.0, 2.0], 'interface.shear.mep.h': 0.1,
      'interface.shear.mep.nstepmax': 5, 'interface.shear.mep.tol': 1e-3,
      'interface.shear.mep.delta': 1e-3},
     ShearParams(delta=0.5, seed=3, mep=ShearMepParams(npts=11, extent=[2.0, 2.0], h=0.1,
                                                       nstepmax=5, tol=1e-3, delta=1e-3))),
])
def test_shear_choices_allowed(edits, expected):
    shear = settings_from_dict(edited(WORKFLOW, edits)).interface.shear
    assert shear == expected
    assert type(shear.delta) is float and type(shear.mep.delta) is float
    assert all(type(side) is float for side in shear.mep.extent)
    assert type(shear.seed) is int and type(shear.mep.npts) is int
    assert type(shear.mep.nstepmax) is int


def test_the_two_deltas_of_the_shear_are_told_apart():
    # [interface.shear] delta is the step of the derivative of the shear strength; the delta of
    # the MEP, in [interface.shear.mep], the step of the derivative inside the string method.
    settings = settings_from_dict(edited(WORKFLOW, {'interface.shear.delta': 0.5}))
    assert (settings.interface.shear.delta, settings.interface.shear.mep.delta) == (0.5, 1e-4)
    settings = settings_from_dict(edited(WORKFLOW, {'interface.shear.mep.delta': 0.5}))
    assert (settings.interface.shear.delta, settings.interface.shear.mep.delta) == (0.01, 0.5)


def test_shear_does_not_change_the_other_tables():
    settings = settings_from_dict(edited(WORKFLOW, {'interface.shear.seed': 3}))
    reference = settings_from_dict(WORKFLOW)
    assert settings.interface.adhesion == reference.interface.adhesion
    assert settings.interface.ppes == reference.interface.ppes
    assert settings.interface.charge == reference.interface.charge
    assert settings.interface.pes_scf == reference.interface.pes_scf
    # A step that only analyses has no engine.
    with pytest.raises(SettingsError, match="'shear' is not allowed"):
        settings_from_dict(edited(WORKFLOW, {'interface.engines': {'shear': 'qe'}}))


def test_groups_optional():
    assert settings_from_dict(edited(WORKFLOW, {'execution.groups': DELETE})).execution.groups is None


def test_groups_positive_when_given():
    with pytest.raises(SettingsError, match=r'execution\.groups'):
        settings_from_dict(edited(WORKFLOW, {'execution.groups': 0}))


# --- inter_params of the interface matching -----------------------------------------

def test_inter_params_keys_are_the_matching_keys():
    defaults = read_default_params(str(MATCHING_DEFAULTS), 'inter_params', {})
    assert InterParams().given() == {}
    assert set(InterParams.__dataclass_fields__) == set(defaults)


def test_inter_params_defaults_filled_by_read_default_params():
    # The workflow sets its own defaults, then read_default_params the others.
    requested = InterParams().requested()
    assert requested == INTER_PARAMS_DEFAULTS == {'max_area': 160.0, 'interface_distance': 2.0,
                                                  'max_angle_tol': 0.4}
    params = read_default_params(str(MATCHING_DEFAULTS), 'inter_params', requested)
    assert params == {'max_area': 160.0, 'interface_distance': 2.0, 'max_angle_tol': 0.4,
                      'max_length_tol': 0.02, 'max_area_ratio_tol': 0.05,
                      'strain_weight_1': 1, 'strain_weight_2': 1, 'best_match': 'area'}
    # Floats, as the historic command line passes them (the defaults file has the integer 2).
    assert type(params['max_area']) is type(params['interface_distance']) is float


def test_inter_params_given_replace_the_defaults_of_the_workflow():
    data = edited(WORKFLOW, {'interface.inter_params': {'max_area': 150, 'max_angle_tol': 0.2}})
    inter_params = settings_from_dict(data).interface.inter_params
    assert inter_params.given() == {'max_area': 150.0, 'max_angle_tol': 0.2}
    assert inter_params.requested() == {'max_area': 150.0, 'interface_distance': 2.0,
                                        'max_angle_tol': 0.2}


def test_inter_params_of_the_example():
    inter_params = settings_from_dict(copy.deepcopy(WORKFLOW)).interface.inter_params
    assert inter_params.requested() == {'max_area': 160.0, 'max_angle_tol': 0.4,
                                        'interface_distance': 2.0, 'max_length_tol': 0.02,
                                        'max_area_ratio_tol': 0.05, 'best_match': 'area'}


# --- errors -------------------------------------------------------------------------

CONTROL, SYSTEM = 'namelists.CONTROL', 'namelists.SYSTEM'

ERRORS = [
    # loader: schema version, unknown keys, types
    ('version-missing', W, {'schema_version': DELETE}, 'schema_version', 'missing; supported: 1'),
    ('version-2', W, {'schema_version': 2}, 'schema_version', 'unsupported value 2'),
    ('version-bool', W, {'schema_version': True}, 'schema_version', 'unsupported value True'),
    ('unknown-top', W, {'cluster': {}}, 'cluster', 'unknown key (allowed keys: schema_version, '),
    ('unknown-in-list', W, {'atomic_species[1].colour': 'red'}, 'atomic_species[1].colour',
     'allowed keys: label, mass, pseudopotential, starting_magnetization'),
    ('unknown-nested', S, {'structure.cell.size': 3}, 'structure.cell.size', 'allowed keys: source, '),
    ('unknown-namelist', W, {'namelists.CELL': {'cell_dynamics': 'bfgs'}}, 'namelists.CELL',
     'allowed keys: CONTROL, SYSTEM, ELECTRONS, IONS'),
    ('c6-in-species', W, {'atomic_species[0].london_c6': 1.0}, 'atomic_species[0].london_c6',
     'write "london_c6(i)" in [namelists.SYSTEM]'),
    ('rvdw-in-species', W, {'atomic_species[0].london_rvdw': 1.0}, 'atomic_species[0].london_rvdw',
     'write "london_rvdw(i)" in [namelists.SYSTEM]'),
    ('str-for-float', W, {'atomic_species[0].mass': '63.5'}, 'atomic_species.mass',
     "expected float, got '63.5'"),
    ('bool-for-int', W, {'execution.groups': True}, 'execution.groups', 'expected int, got True'),
    ('bool-for-float', W, {'k_points.kdens': True}, 'k_points.kdens', 'expected float, got True'),
    ('bool-in-list', S, {'structure.constraints.default_if_pos': [True, True, False]},
     'structure.constraints.default_if_pos', 'expected int'),
    ('str-for-bool', W, {'interface.set_bm': 'yes'}, 'interface.set_bm', "expected bool, got 'yes'"),
    ('scalar-for-table', W, {'namelists': 5}, 'namelists', 'expected a table, got 5'),
    ('scalar-for-namelist', W, {f'{CONTROL}': 5}, CONTROL, 'expected a table, got 5'),
    ('missing-key', W, {'atomic_species[0].mass': DELETE}, 'atomic_species.mass', 'required key missing'),
    ('missing-table', W, {'k_points': DELETE}, 'k_points', 'required key missing'),
    # namelists
    ('array-value', W, {f'{SYSTEM}.london_c6': [1.0, 2.0]}, f'{SYSTEM}.london_c6', 'one key per element'),
    ('table-value', W, {f'{SYSTEM}.hubbard': {'u': 1.0}}, f'{SYSTEM}.hubbard', 'one key per element'),
    ('ibrav', W, {f'{SYSTEM}.ibrav': 0}, f'{SYSTEM}.ibrav', 'ibrav = 0 always'),
    ('ibrav-case', W, {f'{SYSTEM}.IBRAV': 4}, f'{SYSTEM}.IBRAV', 'ibrav = 0 always'),
    ('celldm', W, {f'{SYSTEM}."celldm(1)"': 6.8}, f'{SYSTEM}.celldm(1)', 'ibrav = 0 always'),
    ('cell-a', W, {f'{SYSTEM}.A': 3.6}, f'{SYSTEM}.A', 'ibrav = 0 always'),
    ('cell-cosab', W, {f'{SYSTEM}.cosAB': 0.5}, f'{SYSTEM}.cosAB', 'ibrav = 0 always'),
    ('nat', W, {f'{SYSTEM}.nat': 4}, f'{SYSTEM}.nat', 'set by the generator from the structure'),
    ('ntyp', W, {f'{SYSTEM}.ntyp': 2}, f'{SYSTEM}.ntyp', 'from [[atomic_species]]'),
    ('magnetization-in-system', W, {f'{SYSTEM}."starting_magnetization(1)"': 0.5},
     f'{SYSTEM}.starting_magnetization(1)', 'for each species in [[atomic_species]]'),
    ('calculation', S, {f'{CONTROL}.calculation': 'vc-relax'}, f'{CONTROL}.calculation',
     "'vc-relax' is not allowed; allowed values: 'scf', 'relax'"),
    *[(f'nstep-{value!r}', W, {f'{CONTROL}.nstep': value}, f'{CONTROL}.nstep',
       f'an integer >= 1 is required, got {value!r}') for value in (0, -50, 150.0, '150', True)],
    ('nstep-case', W, {f'{CONTROL}.nstep': DELETE, f'{CONTROL}.NSTEP': 0}, f'{CONTROL}.NSTEP',
     'an integer >= 1 is required'),
    *[(f'mixing-beta-{value!r}', W, {'namelists.ELECTRONS.mixing_beta': value},
       'namelists.ELECTRONS.mixing_beta', f'a number in (0, 1] is required, got {value!r}')
      for value in (0, 0.0, -0.3, 1.5, 2.0, '0.4', False)],
    *[(f'wfcdir-{value!r}', W, {f'{CONTROL}.wfcdir': value}, f'{CONTROL}.wfcdir',
       'a relative directory inside the calculation directory is required')
      for value in ('/scratch/local', '.', './', '../shared', 'WFC/../..', 'C:\\tmp', 1)],
    # whole file
    ('both-modes', W, {'structure': {'file': 'a.xyz'}}, '', 'give either [structure]'),
    ('no-mode', W, {'interface': DELETE}, '', 'give either [structure]'),
    *[(f'workflow-{name}', W, {f'{CONTROL}.{name}': value}, f'{CONTROL}.{name}', 'not allowed in a workflow')
      for name, value in (('calculation', 'scf'), ('title', 'x'), ('prefix', 'x'),
                          ('restart_mode', 'from_scratch'), ('outdir', 'OUT'), ('PREFIX', 'x'))],
    ('no-species', W, {'atomic_species': []}, 'atomic_species', 'at least one species'),
    ('same-label', W, {'atomic_species[1].label': 'Cu'}, 'atomic_species[1].label',
     "'Cu' is already atomic_species[0]"),
    # [structure]
    ('cell-source', S, {'structure.cell.source': 'auto'}, 'structure.cell.source', "'auto' is not allowed"),
    ('cell-units', S, {'structure.cell.units': 'bohr'}, 'structure.cell.units', "'bohr' is not allowed"),
    ('cell-vectors-shape', S, {'structure.cell.source': 'config', 'structure.cell.vectors': [[1, 0, 0]]},
     'structure.cell.vectors', 'three row vectors'),
    ('override-from-1', S, {'structure.constraints.overrides': [{'atoms': [0, 1], 'if_pos': [0, 0, 0]}]},
     'structure.constraints.overrides[0].atoms', 'counted from 1, got 0'),
    ('override-if-pos', S, {'structure.constraints.overrides': [{'atoms': [1], 'if_pos': [0, 0]}]},
     'structure.constraints.overrides[0].if_pos', 'three values'),
    ('override-twice', S, {'structure.constraints.overrides': [{'atoms': [1, 3], 'if_pos': [0, 0, 0]},
                                                                {'atoms': [3], 'if_pos': [0, 0, 1]}]},
     'structure.constraints.overrides[1].atoms', 'atom 3 is already in overrides[0]'),
    ('override-key', S, {'structure.constraints.overrides': [{'atom': 6, 'if_pos': [0, 0, 0]}]},
     'structure.constraints.overrides[0].atom', 'allowed keys: atoms, if_pos'),
    ('default-if-pos', S, {'structure.constraints.default_if_pos': [1, 1]},
     'structure.constraints.default_if_pos', 'three values'),
    ('format', S, {'structure.format': 'vasp'}, 'structure.format', "'vasp' is not allowed"),
    ('xyz-units', S, {'structure.xyz_units': 'bohr'}, 'structure.xyz_units', "'bohr' is not allowed"),
    ('atom-order', S, {'structure.preserve_atom_order': False}, 'structure.preserve_atom_order',
     'only true'),
    ('species-labels', S, {'structure.preserve_species_labels': False},
     'structure.preserve_species_labels', 'only true'),
    # [k_points]
    ('k-mode', W, {'k_points': {'mode': 'tpiba'}}, 'k_points.mode', "'tpiba' is not allowed"),
    ('k-grid-missing', W, {'k_points': {'mode': 'automatic', 'offset': [0, 0, 0]}}, 'k_points.grid',
     "required with mode = 'automatic'"),
    ('k-grid-length', W, {'k_points': {'mode': 'automatic', 'grid': [8, 8], 'offset': [0, 0, 0]}},
     'k_points.grid', 'three values'),
    ('k-offset-missing', W, {'k_points': {'mode': 'automatic', 'grid': [8, 8, 1]}}, 'k_points.offset',
     "required with mode = 'automatic'"),
    ('k-offset-length', W, {'k_points': {'mode': 'automatic', 'grid': [8, 8, 1], 'offset': [0]}},
     'k_points.offset', 'three values'),
    ('k-kdens-missing', W, {'k_points': {'mode': 'kdens'}}, 'k_points.kdens', "required with mode = 'kdens'"),
    # [generation]
    ('positions-units', S, {'generation.atomic_positions_units': 'bohr'},
     'generation.atomic_positions_units', "'bohr' is not allowed"),
    ('cell-parameters-units', S, {'generation.cell_parameters_units': 'alat'},
     'generation.cell_parameters_units', "'alat' is not allowed"),
    # [execution]
    ('scheduler', W, {'execution.scheduler': 'Slurm'}, 'execution.scheduler', "'Slurm' is not allowed"),
    ('groups', W, {'execution.groups': 0}, 'execution.groups', 'must be > 0'),
    ('walltime', W, {'execution.walltime': 0}, 'execution.walltime', 'must be > 0'),
    ('polling', W, {'execution.polling_interval': -5}, 'execution.polling_interval', 'must be > 0'),
    ('margin-0', W, {'execution.max_seconds_margin': 0}, 'execution.max_seconds_margin', 'must be > 0'),
    ('report-wait-0', W, {'execution.job_report_wait': 0}, 'execution.job_report_wait',
     'must be > 0'),
    ('margin-walltime', W, {'execution.max_seconds_margin': 86400}, 'execution.max_seconds_margin',
     'must be < walltime (86400), got 86400'),
    ('margin-float', W, {'execution.max_seconds_margin': 600.0}, 'execution.max_seconds_margin',
     'expected int, got 600.0'),
    ('pp-command-empty', W, {'execution.pp_command': '  '}, 'execution.pp_command',
     'must not be empty'),
    ('pp-command-type', W, {'execution.pp_command': 1}, 'execution.pp_command',
     'expected str, got 1'),
    # [execution.analysis]: the job of the analyses that need a compute node.
    ('analysis-jobscript-missing', W, {'execution.analysis': {}}, 'execution.analysis.jobscript',
     'required key missing'),
    ('analysis-jobscript-type', W, {'execution.analysis.jobscript': 1},
     'execution.analysis.jobscript', 'expected str, got 1'),
    ('analysis-key', W, {'execution.analysis': {'jobscript': 'analysis.sh', 'memory': '200G'}},
     'execution.analysis.memory', 'unknown key (allowed keys: jobscript)'),
    ('analysis-scalar', W, {'execution.analysis': 'analysis.sh'}, 'execution.analysis',
     'expected a table'),
    # [interface]
    ('one-slab', W, {'interface.slabs': WORKFLOW['interface']['slabs'][:1]}, 'interface.slabs',
     'give two slabs; got 1'),
    ('slab-key', W, {'interface.slabs[0].thickness': 3}, 'interface.slabs[0].thickness', 'unknown key'),
    # formula and name of both slabs, or none of them.
    ('formula-without-name', W, {'interface.slabs[0].formula': 'Cu', 'interface.slabs[1].formula': 'Al'},
     'interface.slabs[0].name', 'give formula and name of both slabs, or none of them (the ids of '
     'the interface are then made from the slabs)'),
    ('name-without-formula', W, {'interface.slabs[0].name': 'a', 'interface.slabs[1].name': 'b'},
     'interface.slabs[0].formula', 'give formula and name of both slabs, or none of them'),
    ('ids-of-one-slab', W, {'interface.slabs[1].formula': 'Al', 'interface.slabs[1].name': 'b'},
     'interface.slabs[0].formula', 'give formula and name of both slabs'),
    ('miller-length', W, {'interface.slabs[0].miller': [1, 1]}, 'interface.slabs[0].miller', 'three values'),
    ('engine', W, {'interface.engine': 'mace'}, 'interface.engine', 'not available yet'),
    ('engines-step', W, {'interface.engines': {'shear': 'qe'}}, 'interface.engines', "'shear' is not allowed"),
    ('engines-engine', W, {'interface.engines': {'pes': 'cp2k'}}, 'interface.engines.pes', 'not available yet'),
    ('set-bm-missing', W, {'interface.slabs[1].bulk_modulus': DELETE}, 'interface.slabs[1].bulk_modulus',
     'required with set_bm = true'),
    ('inter-params-key', W, {'interface.inter_params.vacuum': 15.0}, 'interface.inter_params.vacuum',
     'allowed keys: max_area, '),
    # Only the match of smallest area is available.
    ('best-match-mismatch', W, {'interface.inter_params.best_match': 'mismatch'},
     'interface.inter_params.best_match',
     "'mismatch' is not allowed; allowed values: 'area'; the matching always returns the match "
     'of smallest area'),
    ('best-match-other', W, {'interface.inter_params.best_match': 'strain'},
     'interface.inter_params.best_match', "'strain' is not allowed"),
    # [interface.pes_scf]: reduced slabs of the static PES; the relaxed PES has no table.
    ('pes-scf-layers-without-short', W, {'interface.pes_scf.layers': [3, 2]},
     'interface.pes_scf.layers', 'only with short = true: the complete slabs keep all their layers'),
    ('pes-scf-layers-one', W, {'interface.pes_scf.short': True, 'interface.pes_scf.layers': [3]},
     'interface.pes_scf.layers', 'give two values, for the bottom and the top slab; got [3]'),
    ('pes-scf-layers-zero', W, {'interface.pes_scf.short': True, 'interface.pes_scf.layers': [3, 0]},
     'interface.pes_scf.layers[1]', 'must be >= 1, got 0'),
    ('pes-scf-layers-bool', W, {'interface.pes_scf.short': True,
                                'interface.pes_scf.layers': [3, True]},
     'interface.pes_scf.layers', 'expected int'),
    ('pes-scf-short-int', W, {'interface.pes_scf.short': 1}, 'interface.pes_scf.short',
     'expected bool, got 1'),
    ('pes-scf-key', W, {'interface.pes_scf.top_layers': 3}, 'interface.pes_scf.top_layers',
     'allowed keys: short, layers'),
    ('pes-table', W, {'interface.pes': {'short': True}}, 'interface.pes',
     'unknown key (allowed keys: slabs, steps, '),
    # [interface.adhesion]: one surface and one kind of adhesion for the run.
    ('adhesion-pes', W, {'interface.adhesion.pes': 'plotpes'}, 'interface.adhesion.pes',
     "'plotpes' is not allowed; allowed values: 'pes', 'pes_scf'"),
    ('adhesion-type', W, {'interface.adhesion.type': 'long'}, 'interface.adhesion.type',
     "'long' is not allowed; allowed values: 'regular', 'short'"),
    ('adhesion-pes-number', W, {'interface.adhesion.pes': 1}, 'interface.adhesion.pes',
     'expected str, got 1'),
    ('adhesion-key', W, {'interface.adhesion.layers': 3}, 'interface.adhesion.layers',
     'allowed keys: pes, type'),
    # The relaxed surface has only complete slabs.
    ('adhesion-short-on-relaxed-pes', W, {'interface.adhesion.type': 'short',
                                          'interface.pes_scf.short': True},
     'interface.adhesion.type', "'short' needs pes = 'pes_scf'"),
    ('adhesion-short-on-complete-slabs', W, {'interface.adhesion.pes': 'pes_scf',
                                             'interface.adhesion.type': 'short'},
     'interface.adhesion.type', 'set short = true in [interface.pes_scf]'),
    # [interface.ppes]: one site, the shifts without repetitions, the fits that work.
    ('ppes-site', W, {'interface.ppes.site': 'both'}, 'interface.ppes.site',
     "'both' is not allowed; allowed values: 'min_site', 'max_site'"),
    ('ppes-fit-name', W, {'interface.ppes.fit': 'UBER'}, 'interface.ppes.fit',
     "'UBER' is not allowed; allowed values: 'none', 'uber'"),
    ('ppes-fit-uber0', W, {'interface.ppes.fit': 'uber0'}, 'interface.ppes.fit',
     "'uber0' is not available because the fit function fails with it"),
    ('ppes-fit-false', W, {'interface.ppes.fit': False}, 'interface.ppes.fit',
     'expected str, got False'),
    ('ppes-fit-number', W, {'interface.ppes.fit': 1}, 'interface.ppes.fit', 'expected str, got 1'),
    ('ppes-shifts-empty', W, {'interface.ppes.shifts': []}, 'interface.ppes.shifts',
     'give at least one shift'),
    ('ppes-shifts-repeated', W, {'interface.ppes.shifts': [0.0, 0.5, 0.0]},
     'interface.ppes.shifts[2]', 'repeated shift 0.0'),
    ('ppes-shifts-repeated-as-integer', W, {'interface.ppes.shifts': [1, 1.0]},
     'interface.ppes.shifts[1]', 'repeated shift 1.0'),
    ('ppes-shifts-negative-zero', W, {'interface.ppes.shifts': [0.0, -0.0]},
     'interface.ppes.shifts[1]', 'repeated shift -0.0'),
    ('ppes-shifts-nan', W, {'interface.ppes.shifts': [0.0, float('nan')]},
     'interface.ppes.shifts[1]', 'must be a finite number, got nan'),
    ('ppes-shifts-inf', W, {'interface.ppes.shifts': [float('inf')]},
     'interface.ppes.shifts[0]', 'must be a finite number, got inf'),
    ('ppes-shifts-text', W, {'interface.ppes.shifts': ['0.5']}, 'interface.ppes.shifts',
     'expected a list of float'),
    ('ppes-shifts-not-a-list', W, {'interface.ppes.shifts': 0.5}, 'interface.ppes.shifts',
     'expected a list of float, got 0.5'),
    ('ppes-key', W, {'interface.ppes.type': 'short'}, 'interface.ppes.type',
     'allowed keys: site, shifts, fit'),
    # [interface.charge]: one site, the two operations of the algebra of the three densities.
    ('charge-site', W, {'interface.charge.site': 'both'}, 'interface.charge.site',
     "'both' is not allowed; allowed values: 'min_site', 'max_site'"),
    ('charge-site-number', W, {'interface.charge.site': 1}, 'interface.charge.site',
     'expected str, got 1'),
    ('charge-operations-one', W, {'interface.charge.operations': ['-']},
     'interface.charge.operations', "give two operations, one between the interface and the "
     "bottom slab and one with the top slab; got ['-']"),
    ('charge-operations-three', W, {'interface.charge.operations': ['-', '-', '-']},
     'interface.charge.operations', 'give two operations'),
    ('charge-operations-empty', W, {'interface.charge.operations': []},
     'interface.charge.operations', 'give two operations'),
    ('charge-operation-other', W, {'interface.charge.operations': ['-', '*']},
     'interface.charge.operations[1]', "'*' is not allowed; allowed values: '+', '-'"),
    ('charge-operation-both', W, {'interface.charge.operations': ['+-', '-']},
     'interface.charge.operations[0]', "'+-' is not allowed; allowed values: '+', '-'"),
    ('charge-operations-text', W, {'interface.charge.operations': '[-, -]'},
     'interface.charge.operations', 'expected a list of str'),
    ('charge-operations-number', W, {'interface.charge.operations': ['-', 1]},
     'interface.charge.operations', 'expected a list of str'),
    # The integration range, the axis and the number of points are the ones of the original
    # command line: they are not settings.
    ('charge-integrate-in', W, {'interface.charge.integrate_in': [8.0, 10.0]},
     'interface.charge.integrate_in', 'allowed keys: site, operations'),
    ('charge-rescale', W, {'interface.charge.rescale': 10000}, 'interface.charge.rescale',
     'allowed keys: site, operations'),
    ('charge-type', W, {'interface.charge.type': 'short'}, 'interface.charge.type',
     'allowed keys: site, operations'),
    # [interface.shear] and [interface.shear.mep]: numbers that the algorithm can use. The
    # method, the optimizations, the surface and the entry are not settings.
    ('shear-delta-zero', W, {'interface.shear.delta': 0.0}, 'interface.shear.delta',
     'must be a finite number > 0, got 0.0'),
    ('shear-delta-negative', W, {'interface.shear.delta': -0.01}, 'interface.shear.delta',
     'must be a finite number > 0, got -0.01'),
    ('shear-delta-nan', W, {'interface.shear.delta': float('nan')}, 'interface.shear.delta',
     'must be a finite number > 0, got nan'),
    ('shear-delta-inf', W, {'interface.shear.delta': float('inf')}, 'interface.shear.delta',
     'must be a finite number > 0, got inf'),
    ('shear-delta-text', W, {'interface.shear.delta': '0.01'}, 'interface.shear.delta',
     "expected float, got '0.01'"),
    ('shear-delta-bool', W, {'interface.shear.delta': True}, 'interface.shear.delta',
     'expected float, got True'),
    ('shear-seed-negative', W, {'interface.shear.seed': -1}, 'interface.shear.seed',
     'must be between 0 and 4294967295 (the seed of NumPy), got -1'),
    ('shear-seed-too-large', W, {'interface.shear.seed': 4294967296}, 'interface.shear.seed',
     'must be between 0 and 4294967295 (the seed of NumPy), got 4294967296'),
    ('shear-seed-float', W, {'interface.shear.seed': 1.5}, 'interface.shear.seed',
     'expected int, got 1.5'),
    ('shear-seed-bool', W, {'interface.shear.seed': True}, 'interface.shear.seed',
     'expected int, got True'),
    # Fewer than six points make the writing of the entries fail after the string is computed.
    ('shear-npts-small', W, {'interface.shear.mep.npts': 3}, 'interface.shear.mep.npts',
     'must be >= 6 (the entries of the shear strength hold the first six points of the '
     'string), got 3'),
    ('shear-npts-four', W, {'interface.shear.mep.npts': 4}, 'interface.shear.mep.npts',
     'must be >= 6 (the entries of the shear strength hold the first six points of the '
     'string), got 4'),
    ('shear-npts-five', W, {'interface.shear.mep.npts': 5}, 'interface.shear.mep.npts',
     'must be >= 6 (the entries of the shear strength hold the first six points of the '
     'string), got 5'),
    ('shear-npts-float', W, {'interface.shear.mep.npts': 101.0}, 'interface.shear.mep.npts',
     'expected int, got 101.0'),
    ('shear-npts-bool', W, {'interface.shear.mep.npts': True}, 'interface.shear.mep.npts',
     'expected int, got True'),
    ('shear-extent-one', W, {'interface.shear.mep.extent': [1.5]}, 'interface.shear.mep.extent',
     'give two numbers, one for x and one for y; got [1.5]'),
    ('shear-extent-three', W, {'interface.shear.mep.extent': [1.5, 1.5, 1.5]},
     'interface.shear.mep.extent', 'give two numbers, one for x and one for y'),
    ('shear-extent-zero', W, {'interface.shear.mep.extent': [1.5, 0.0]},
     'interface.shear.mep.extent[1]', 'must be a finite number > 0, got 0.0'),
    ('shear-extent-negative', W, {'interface.shear.mep.extent': [-1.5, 1.5]},
     'interface.shear.mep.extent[0]', 'must be a finite number > 0, got -1.5'),
    ('shear-extent-inf', W, {'interface.shear.mep.extent': [1.5, float('inf')]},
     'interface.shear.mep.extent[1]', 'must be a finite number > 0, got inf'),
    ('shear-extent-text', W, {'interface.shear.mep.extent': '1.5'}, 'interface.shear.mep.extent',
     'expected a list of float'),
    ('shear-extent-text-item', W, {'interface.shear.mep.extent': [1.5, 'a']},
     'interface.shear.mep.extent', 'expected a list of float'),
    ('shear-h-zero', W, {'interface.shear.mep.h': 0.0}, 'interface.shear.mep.h',
     'must be a finite number > 0, got 0.0'),
    ('shear-tol-negative', W, {'interface.shear.mep.tol': -1e-8}, 'interface.shear.mep.tol',
     'must be a finite number > 0, got -1e-08'),
    ('shear-mep-delta-nan', W, {'interface.shear.mep.delta': float('nan')},
     'interface.shear.mep.delta', 'must be a finite number > 0, got nan'),
    ('shear-nstepmax-zero', W, {'interface.shear.mep.nstepmax': 0}, 'interface.shear.mep.nstepmax',
     'must be >= 1, got 0'),
    ('shear-nstepmax-float', W, {'interface.shear.mep.nstepmax': 1000.0},
     'interface.shear.mep.nstepmax', 'expected int, got 1000.0'),
    ('shear-key', W, {'interface.shear.method': 'zerotemp'}, 'interface.shear.method',
     'allowed keys: delta, seed, mep'),
    ('shear-key-pes', W, {'interface.shear.pes': 'pes_scf'}, 'interface.shear.pes',
     'allowed keys: delta, seed, mep'),
    ('shear-mep-key', W, {'interface.shear.mep.optimization': 'x'},
     'interface.shear.mep.optimization', 'allowed keys: npts, extent, h, nstepmax, tol, delta'),
    ('shear-mep-scalar', W, {'interface.shear.mep': 101}, 'interface.shear.mep',
     'expected a table'),
    # [qe]
    ('qe-version', W, {'qe.version': '7.2MaX'}, 'qe.version',
     "'7.2MaX' is not a version of pw.x: give major.minor or major.minor.patch"),
    ('qe-version-one-part', W, {'qe.version': '7'}, 'qe.version', 'not a version of pw.x'),
    ('qe-version-v', W, {'qe.version': 'v.7.2'}, 'qe.version', 'not a version of pw.x'),
    ('qe-version-number', W, {'qe.version': 7.2}, 'qe.version', 'expected str, got 7.2'),
    ('qe-version-missing', W, {'qe.version': DELETE}, 'qe.version',
     'required by the steps that run pw.x'),
    ('qe-missing', W, {'qe': DELETE}, 'qe.version', 'required by the steps that run pw.x'),
    ('qe-key', W, {'qe.md5': 'pseudo.md5'}, 'qe.md5', 'allowed keys: version, upf_md5'),
]


@pytest.mark.parametrize('base, edits, path, message', [case[1:] for case in ERRORS],
                         ids=[case[0] for case in ERRORS])
def test_validation_error(base, edits, path, message):
    with pytest.raises(SettingsError) as error:
        settings_from_dict(edited(base, edits))
    assert error.value.path == path
    assert message in error.value.message


def test_case_ids_unique():
    for cases in (ERRORS, ACCEPTED):
        ids = [case[0] for case in cases]
        assert len(ids) == len(set(ids))
