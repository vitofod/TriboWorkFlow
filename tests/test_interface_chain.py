# Author: Vito Foderà
# Created: 2026-11-09

"""The nine steps of the interface workflow end to end, with a stand-in for pw.x and pp.x.

The run (``chain_helpers.py``, the fixture ``chain`` of ``conftest.py``) goes through the
orchestrator, the workers with the stand-in pw.x and pp.x, and the jobs of the analyses; the
tests read what it left. The stand-in pw.x gives the energies of ASE's EMT, the pp.x a uniform
density: what the run proves is that the steps hand each other the right data, not that the
numbers are those of pw.x.
"""

import json

import numpy as np
import pytest

from chain_helpers import CUCU, SHIFTS, STEPS, posix_only
from triboworkflow.core.store import read_record
from triboworkflow.execution import layout
from triboworkflow.execution.jobs import OUTCOME_FILE
from triboworkflow.workflows.interface import registry


@posix_only
def test_all_the_steps_are_completed(chain):
    assert chain.report.success, chain.report.steps
    assert {s.step: s.status for s in chain.report.steps} == dict.fromkeys(STEPS, 'completed')
    assert chain.orch.order == ['match', 'pes', 'pes_scf', 'plotpes', 'plotpes_scf', 'shear',
                                'adhesion', 'ppes', 'charge']


@posix_only
def test_another_pass_does_nothing(chain):
    jobs = dict(chain.orch.manager.jobs)
    report = chain.orch.run_pass()
    assert report.success and report.submitted == [] and chain.orch.manager.jobs == jobs
    assert all(step().done(chain.run_dir, CUCU) for step in registry.STEPS.values())


@posix_only
def test_every_calculation_is_complete_and_of_one_energy_model_per_step(chain):
    sdir = layout.system_dir(chain.run_dir, CUCU)
    records = sorted(path.parent.relative_to(sdir).as_posix()
                     for path in sdir.glob(f'*/*/{layout.INPUT_FILE}'))
    assert records, 'no calculation folder found'
    steps = {}
    for calc_id in records:
        record = read_record(sdir, calc_id)
        assert record.is_complete, calc_id
        assert record.status.parsing == 'ok', calc_id
        steps.setdefault(record.step, []).append(record)
    assert sorted(steps) == ['adhesion', 'charge', 'pes', 'pes_scf', 'ppes']
    assert {step: len(items) for step, items in steps.items()} == {
        'pes': 4, 'pes_scf': 4, 'adhesion': 4, 'ppes': len(SHIFTS), 'charge': 3}
    for step, items in steps.items():
        assert len({json.dumps(r.energy_model, sort_keys=True) for r in items}) == 1, step


@posix_only
def test_the_surfaces_vary_from_point_to_point(chain):
    doc = chain.document
    for name in ('pes', 'pes_scf'):
        energies = [row[3] for row in doc[name]['energy']['grid']['hs_unique_ev']]
        assert len(energies) == 4 and max(energies) - min(energies) > 1e-6, name
        assert doc[name]['energy']['min_site'] != doc[name]['energy']['max_site'], name


@posix_only
@pytest.mark.parametrize('name', ['plotpes', 'plotpes_scf'])
def test_the_interpolations_ran_in_jobs_of_their_own(chain, name):
    group = layout.job_dir(chain.run_dir, f'{CUCU}.{name}.1')
    outcome = json.loads((group / OUTCOME_FILE).read_text())
    assert (outcome['result'], outcome['error']) == ('analysed', None)
    assert chain.orch.entry(CUCU, name)['job'] is not None
    pes_name = {'plotpes': 'pes', 'plotpes_scf': 'pes_scf'}[name]
    # The 4 unique points of the Cu/Cu interface are 8 in the cell.
    assert len(chain.document[pes_name]['energy']['pes_ev']) == 8
    assert list(layout.system_dir(chain.run_dir, CUCU).glob(f'{name}/*.png'))


@posix_only
def test_the_shear_strength_is_written_along_the_three_paths(chain):
    group = layout.job_dir(chain.run_dir, f'{CUCU}.shear.1')
    assert json.loads((group / OUTCOME_FILE).read_text())['result'] == 'analysed'
    strengths = chain.document['shearstr']
    assert sorted(strengths) == ['along_bs_line', 'along_x', 'along_y']
    for entry in strengths.values():
        assert np.isfinite(entry['shear_strength_GPa'])
        assert 0 <= entry['nstep'] <= 2  # nstepmax = 2: the index of the last step


@posix_only
def test_the_adhesion_the_perpendicular_pes_and_the_charge(chain):
    doc = chain.document
    adhesion = doc['adhesion']['regular']
    assert np.isfinite(adhesion['min_site']['adhesion_energy_ev'])
    assert np.isfinite(adhesion['corrugation_ev'])
    ppes = doc['ppes']['regular']['energy']['min_site']['ppes_ev']
    assert [row[0] for row in ppes] == SHIFTS and len({row[1] for row in ppes}) == len(SHIFTS)
    displacement = doc['charge']['regular']['displacement']['min_site']['z']
    # The stand-in pp.x writes a uniform density of one electron per atom, and the interface
    # has the atoms of its two slabs.
    assert displacement['interface']['integral'] == pytest.approx(0.0, abs=1e-6)
    assert displacement['all']['integral'] == pytest.approx(0.0, abs=1e-6)


@posix_only
def test_the_logs_of_the_jobs_are_in_the_folders_of_the_groups(chain):
    logs = sorted((chain.run_dir / 'jobs').glob('*/job.*.err'))
    assert logs
    shear = [p for p in logs if '.shear.' in p.parent.name]
    assert shear and 'shear strength along' in shear[0].read_text()
