"""Golden (characterisation) values of the TribChem physics modules.

Every case calls TribChem code under a root package, 'tribchem' in the
historic environment or 'triboworkflow.vendored' in the development one,
with the arguments that the TribChem Firetasks and workflows pass, and
returns plain JSON data. The custom_slab cases instead compare a Firetask that
is not vendored (FT_StoreCustomSlab) with its port in triboworkflow.core.structures,
the match_step cases the Firetasks of the matching with the step match
(triboworkflow.workflows.interface.match), the pes_step cases the
Firetasks of the PES with the steps pes and pes_scf
(triboworkflow.workflows.interface.pes), starting from the document that the
match_step golden calls build, and the plotpes_step cases the Firetasks of
the interpolation with the steps plotpes and plotpes_scf
(triboworkflow.workflows.interface.plotpes), starting from the document that
the match_step and pes_step golden calls build; the key image that TribChem
writes (the PNG of the figure encoded again) is compared by its pixels with
the figure of the port, which writes no image (DEV-42), and the adhesion_step
cases the Firetasks of the adhesion with the step adhesion
(triboworkflow.workflows.interface.adhesion), starting from the document that
the match_step and pes_step golden calls build.
The golden values are the results obtained in the
historic environment, stored in tests/golden/<case>.json:
tests/historic/test_golden.py checks that they are reproduced there and
tests/test_golden.py that the vendored modules give the same results.

The input data are copies of TribChem files, kept in tests/data/tribchem/
under their path inside the tribchem package and checked against the source
manifest (envs/historic/tribchem-source.sha256).

Usage, in the historic environment only:
    python tests/golden_cases.py [<case> ...]
writes the golden files of the given cases (default: all of them).
"""

import ast
import contextlib
import copy
import importlib
import io
import json
import math
import os
import re
import sys
import tempfile
import time
import warnings
from pathlib import Path

# OpenBLAS results depend on its number of threads at round-off level (PES
# and RBF values differ between 1, 8 and 12 threads): the golden values are
# produced and checked with one thread, set before numpy is imported.
BLAS_THREADS = {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
if 'numpy' in sys.modules and any(os.environ.get(k) != v for k, v in BLAS_THREADS.items()):
    raise RuntimeError(f'numpy was imported before setting {BLAS_THREADS}')
os.environ.update(BLAS_THREADS)

import numpy as np  # noqa: E402

import vendoring  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'tests' / 'data' / 'tribchem'
GOLDEN = ROOT / 'tests' / 'golden'
HISTORIC, VENDORED = 'tribchem', 'triboworkflow.vendored'

# Seed set by the cases before every MEP calculation: get_bs_line adds
# unseeded np.random noise to the starting string (A05).
SEED = 0

UUID4 = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}')

MATCHING_SCRIPT = 'physics/tribology/tests/test_matching_interfaces.py.py'
PPES_DATA = 'data/homogeneous/ppes_data'


class TribChem:
    """The TribChem modules under a root package."""

    def __init__(self, root):
        self.root = root

    def __getitem__(self, module):
        return importlib.import_module(f'{self.root}.{module}')


# =============================================================================
# Input data
# =============================================================================

def data_files():
    """Paths, relative to DATA, of the copied TribChem files."""
    return sorted(p.relative_to(DATA).as_posix() for p in DATA.rglob('*') if p.is_file())


def check_data(rel, manifest, data=DATA):
    """Problems of <data>/<rel>; an empty list if it equals the TribChem file."""
    expected = manifest.get('tribchem/' + rel)
    if expected is None:
        return [f'tribchem/{rel} is not among the TribChem sources']
    if vendoring.sha256((Path(data) / rel).read_bytes()) != expected:
        return ['differs from the original']
    return []


def slab_dicts():
    """The <name>_dict slabs assigned in the TribChem matching script, read
    with ast.literal_eval without running the script."""
    tree = ast.parse((DATA / MATCHING_SCRIPT).read_text())
    slabs = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.endswith('_dict')):
            slabs[node.targets[0].id.removesuffix('_dict')] = ast.literal_eval(node.value)
    return slabs


def load_slab(name):
    from pymatgen.core.surface import Slab
    return Slab.from_dict(slab_dicts()[name])


# =============================================================================
# Conversion to JSON data and comparison
# =============================================================================

def to_data(obj):
    """Plain JSON data of obj: pymatgen objects through as_dict, arrays as
    {'ndarray': values, 'dtype': dtype}."""
    if hasattr(obj, 'as_dict'):
        return to_data(obj.as_dict())
    if isinstance(obj, dict):
        bad = [k for k in obj if not isinstance(k, str)]
        if bad:
            raise TypeError(f'dictionary keys that are not strings: {bad!r}')
        return {k: to_data(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_data(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return {'ndarray': to_data(obj.tolist()), 'dtype': str(obj.dtype)}
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f'cannot convert {type(obj).__name__}')


def outcome(func, *args, **kwargs):
    """{'result': data} of func(*args, **kwargs), or {'error': ...} if it raises."""
    try:
        return {'result': to_data(func(*args, **kwargs))}
    except Exception as error:
        return {'error': type(error).__name__, 'message': str(error)}


def strip_uuid(tag):
    """The tag with its uuid4 replaced by '<uuid4>'; it must contain one."""
    new, n = UUID4.subn('<uuid4>', tag)
    if n != 1:
        raise ValueError(f'tag without exactly one uuid4: {tag!r}')
    return new


def numbers_differ(a, b, tolerance):
    if a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)):
        return False
    if tolerance is None or isinstance(a, bool) or isinstance(b, bool):
        return True
    return not math.isclose(a, b, rel_tol=tolerance['rtol'], abs_tol=tolerance['atol'])


def compare(expected, actual, tolerance=None, path=''):
    """Differences between two JSON data trees, as 'path: message' strings.

    Numbers must be equal; with tolerance = {'rtol': ..., 'atol': ...} floats
    may differ within math.isclose. Integers and floats are told apart.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        diffs = [f'{path}/{k}: missing' for k in expected if k not in actual]
        diffs += [f'{path}/{k}: unexpected' for k in actual if k not in expected]
        for k in expected:
            if k in actual:
                diffs += compare(expected[k], actual[k], tolerance, f'{path}/{k}')
        return diffs
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f'{path}: length {len(actual)} instead of {len(expected)}']
        diffs = []
        for i, (e, a) in enumerate(zip(expected, actual)):
            diffs += compare(e, a, tolerance, f'{path}/{i}')
        return diffs
    if type(expected) is not type(actual):
        return [f'{path}: {actual!r} instead of {expected!r}']
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return [f'{path}: {actual!r} instead of {expected!r}'] \
            if numbers_differ(expected, actual, tolerance) else []
    return [] if expected == actual else [f'{path}: {actual!r} instead of {expected!r}']


# =============================================================================
# Parameters of the cases
# =============================================================================

# test_matching_interfaces.py.py: InterfaceMatcher(slab_2=Ag(111), slab_1=Pt(111), ...).
SCRIPT_MATCH = {'strain_weight_1': 1, 'strain_weight_2': 1, 'max_area': 100.0,
                'max_length_tol': 0.01, 'max_angle_tol': 1.5, 'max_area_ratio_tol': 1,
                'best_match': 'area', 'interface_distance': 'auto'}

# highput/firetasks/defaults.json, 'inter_params', as FT_GetHighSymmetryPts
# uses them at the first attempt (factor 1): max_angle_tol forced to 0.4
# (slabs.py:579-582, A02).
FIRETASK_MATCH = {'max_area': 150, 'interface_distance': 2, 'max_angle_tol': 0.4,
                  'max_length_tol': 0.02, 'max_area_ratio_tol': 0.05,
                  'strain_weight_1': 1, 'strain_weight_2': 1, 'best_match': 'area'}

# (slab_1, slab_2, matching parameters)
PAIRS = {
    'pt111_ag111': ('pt111', 'ag111', SCRIPT_MATCH),
    'cu111_al111': ('cu111', 'al111', FIRETASK_MATCH),
    'cu111_cu111': ('cu111', 'cu111', FIRETASK_MATCH),
    'cu111_al111_no_match': ('cu111', 'al111', FIRETASK_MATCH | {'max_area': 20}),
}

# Computational parameters of an interface, with the keys produced by
# generate_interface_comp_params (manipulate_struct.py).
COMP_PARAMS = {'functional': 'PBE', 'encut': 400, 'kdens': 6, 'is_metal': True,
               'use_spin': False, 'vdw': 'dftd2'}

# highput/firetasks/defaults.json, 'StartPPES': shifts and calc_type.
PPES_SHIFTS = [-0.5, -0.25, 0.0, 0.25, 0.5, 2.5, 3.0, 4.0, 5.0, 7.5]
PPES_CALC_TYPE = 'slab_from_scratch'

# highput/firetasks/defaults.json, 'CalcShearStrength'; shearstr.py:113-127.
SHEAR = {'method': 'zerotemp', 'delta': 0.01, 'npts': 101}
SHEAR_PES = {'replicate_of': (5, 5), 'density': 20, 'tol': 1e-4}

# PES data of physics/tribology/tests: (file, columns, cell, make_pes arguments)
PES_GRIDS = {
    # test_pes.py: Ti3C2O2 with replicate_of=(n, n) from the command line
    'ti3c2o2': ('physics/tribology/tests/Ti3C2O2_pes.txt', slice(None),
                np.array([[1, 0], [-0.5, np.sqrt(3) / 2]]) * 2.974964365,
                {'replicate_of': (5, 5), 'orthorombize': False, 'theta': False,
                 'density': 20, 'tol': 1e-4}),
    # test_mep.py: phosphorene, rectangular cell (given there as its two sides)
    'phosphorene': ('physics/tribology/tests/grid.txt', slice(1, None),
                    np.diag([3.31420142, 4.449501367]),
                    {'replicate_of': (2, 2), 'orthorombize': False, 'theta': False,
                     'density': 10, 'tol': 1e-4}),
    # test_pes.py, commented block: PtSe2, orthorombized and rotated
    'ptse2': ('physics/tribology/tests/ptse2.txt', slice(None),
              np.array([[3.7100000381, 0.000000000, 0.000000000],
                        [-1.8550000191, 3.2129542811, 0.000000000],
                        [0.00000000000, 0.000000000, 30.000000000]]),
              {'replicate_of': (3, 3), 'orthorombize': True, 'theta': True,
               'density': 20, 'tol': 1e-4}),
}


# =============================================================================
# Building blocks, as in the Firetasks
# =============================================================================

def match(tc, pair):
    """FT_GetHighSymmetryPts.generate_interface (slabs.py:584-590), one attempt."""
    name_1, name_2, params = PAIRS[pair]
    matcher = tc['physics.tribology.interface_matcher'].InterfaceMatcher(
        slab_1=load_slab(name_1), slab_2=load_slab(name_2), **params)
    top_aligned, bot_aligned = matcher.get_centered_slabs()
    interface = matcher.get_interface()
    structure = tc['highput.utils.manipulate_struct'].slab_to_structure(interface)
    return matcher, bot_aligned, top_aligned, interface, structure


def high_symmetry(tc, bot_slab, top_slab):
    """FT_GetHighSymmetryPts.get_hspts (slabs.py:620-676) without the prints."""
    hsym = tc['physics.tribology.highsym']
    flipped_top = tc['physics.tribology.interface_matcher'].flip_slab(top_slab)
    top_hsp_unique, top_hsp_all = hsym.get_slab_hs(flipped_top)
    bottom_hsp_unique, bottom_hsp_all = hsym.get_slab_hs(bot_slab)
    cell = bot_slab.lattice.matrix
    hsp_unique = hsym.get_interface_hs(bottom_hsp_unique, top_hsp_unique, cell)
    hsp_all = hsym.get_interface_hs(bottom_hsp_all, top_hsp_all, cell)
    c_hsp_u, c_hsp_a = hsym.fix_hs_dicts(hsp_unique, hsp_all, top_slab, bot_slab)
    hs = {
        'bot_unique': hsym.pbc_hs_points(bottom_hsp_unique, cell),
        'bot_all': hsym.pbc_hs_points(bottom_hsp_all, cell),
        'top_unique': hsym.pbc_hs_points(top_hsp_unique, cell),
        'top_all': hsym.pbc_hs_points(top_hsp_all, cell),
        'inter_unique': c_hsp_u,
        'inter_all': c_hsp_a}
    steps = {'top_hsp_unique': top_hsp_unique, 'top_hsp_all': top_hsp_all,
             'bottom_hsp_unique': bottom_hsp_unique, 'bottom_hsp_all': bottom_hsp_all,
             'hsp_unique': hsp_unique, 'hsp_all': hsp_all}
    return hs, steps


_HIGH_SYMMETRY = {}


def matched_high_symmetry(tc, pair):
    """Aligned slabs, hs and steps of high_symmetry for a pair. They are
    computed once per root package and pair, and copies are returned."""
    key = (tc.root, pair)
    if key not in _HIGH_SYMMETRY:
        _, bot_slab, top_slab, _, _ = match(tc, pair)
        _HIGH_SYMMETRY[key] = (bot_slab, top_slab, *high_symmetry(tc, bot_slab, top_slab))
    return copy.deepcopy(_HIGH_SYMMETRY[key])


def pes_setup(tc, pair):
    """Aligned slabs and high-symmetry points, as stored by the matching step:
    the shifts are JSON-sanitised (lists), as read back from the database."""
    from monty.json import jsanitize
    bot_slab, top_slab, hs, _ = matched_high_symmetry(tc, pair)
    return bot_slab, top_slab, jsanitize(hs['inter_unique']), jsanitize(hs['inter_all'])


def tag_prefix(pair):
    """Tag prefix as built in highput/workflows/interface.py:166-167."""
    name_1, name_2, _ = PAIRS[pair]
    return f'{name_1[:-3]}(1, 1, 1)_{name_2[:-3]}(1, 1, 1)'


def vis_data(vis):
    return {'class': type(vis).__name__, 'incar': vis.incar, 'kpoints': vis.kpoints}


def inputs_data(structures, vis, tags):
    return {'structures': structures, 'vis': [vis_data(v) for v in vis],
            'tags': [strip_uuid(t) for t in tags]}


def synthetic_energies(hs_unique, cell):
    """[[label, x, y, E], ...] sorted by E, as FT_ComputePES builds them
    (pes.py:194-209), with eV and J/m^2 lists. E is a smooth periodic
    function of the fractional shift, in place of DFT energies."""
    inverse = np.linalg.inv(np.asarray(cell)[:2, :2])
    values = []
    for label, points in hs_unique.items():
        x, y = points[0][:2]
        f = np.array([x, y]) @ inverse
        energy = -0.5 + 0.05 * (np.cos(2 * np.pi * f[0]) + np.cos(2 * np.pi * f[1]))
        values.append([label, x, y, float(energy)])
    return sorted(values, key=lambda v: v[3])


def run_mep(tc, rbf, cell, **mep_kwargs):
    """The three MEP and shear strength calculations of FT_CalcShearStrength
    (shearstr.py:119-130), with np.random seeded before each one."""
    MEP = tc['physics.tribology.mep'].MEP
    results = {}
    for opt in ['bs_line', 'x', 'y']:
        np.random.seed(SEED)
        mep = MEP(rbf=rbf, cell=cell)
        mep_line, mep_data = mep.get_mep(method=SHEAR['method'], optimization=opt, **mep_kwargs)
        shear_strength, data_ss_xy = mep.get_shear_strenght(
            delta=SHEAR['delta'], npts=SHEAR['npts'], optimization=opt)
        results[opt] = {'mep': mep_line, 'mep_data': mep.mep_data,
                        'bsmep': mep.bsmep, 'bsmep_data': mep.bsmep_data,
                        'shear_strength': shear_strength, 'data_ss_xy': data_ss_xy}
    return results


def probe_points(cell_2d, n=7):
    """n x n points spread over the 2D cell, where the interpolants are read."""
    f = (np.arange(n) + 0.5) / n
    fx, fy = np.meshgrid(f, f, indexing='ij')
    return np.column_stack([fx.ravel(), fy.ravel()]) @ np.asarray(cell_2d)[:2, :2]


# =============================================================================
# Cases
# =============================================================================

def case_matcher(tc, pair):
    matcher, bot_aligned, top_aligned, interface, structure = match(tc, pair)
    return {'top_aligned': top_aligned, 'bot_aligned': bot_aligned,
            'interface': interface, 'interface_structure': structure,
            'interface_distance': matcher.get_interface_distance()}


def case_highsym(tc, pair):
    _, _, hs, steps = matched_high_symmetry(tc, pair)
    return {'hs': hs, 'steps': steps}


def case_pes_inputs(tc, pair, calc_type):
    """highput/workflows/interface.py:176-177."""
    bot_slab, top_slab, shifts, _ = pes_setup(tc, pair)
    return inputs_data(*tc['highput.utils.manipulate_struct'].generate_pes_inputs(
        bot_slab, top_slab, shifts, COMP_PARAMS, tag_prefix(pair), calc_type, 2))


def case_interface_shifts(tc, pair):
    """apply_interface_shifts with one shift, as FT_StartPPES (run.py:1409-1410),
    and with all the unique shifts at once."""
    ms = tc['highput.utils.manipulate_struct']
    bot_slab, top_slab, shifts, _ = pes_setup(tc, pair)
    every = np.vstack([np.array(v)[:, :2] for v in shifts.values()])
    return {'single': {k: ms.apply_interface_shifts(bot_slab, top_slab, v) for k, v in shifts.items()},
            'all': ms.apply_interface_shifts(bot_slab, top_slab, every)}


def first_interface(tc, pair):
    """Interface at the first unique shift, as FT_StartPPES builds the
    interfaces of the PPES (run.py:1409)."""
    bot_slab, top_slab, shifts, all_shifts = pes_setup(tc, pair)
    interface = tc['highput.utils.manipulate_struct'].apply_interface_shifts(
        bot_slab, top_slab, next(iter(shifts.values())))
    return interface, shifts, all_shifts


def case_ppes_inputs(tc, pair):
    """highput/workflows/interface.py:429-431."""
    return inputs_data(*tc['highput.utils.manipulate_struct'].generate_ppes_inputs(
        first_interface(tc, pair)[0], PPES_SHIFTS, COMP_PARAMS, 'middle', tag_prefix(pair),
        PPES_CALC_TYPE))


def case_chargedisp_inputs(tc, pair):
    """highput/workflows/electronics.py:80 and 92-93."""
    ms = tc['highput.utils.manipulate_struct']
    interface = first_interface(tc, pair)[0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        inter_fixed = ms.ManipulateStruct(interface).fix_interface(interface)
    return inputs_data(*ms.generate_chargedisp_inputs(
        inter_fixed, COMP_PARAMS, tag_prefix(pair), 'zero', 'slab_from_scratch'))


def pes_from_hs(tc, pair):
    """FT_ComputePES.get_data and compute_pes (pes.py:194-236) on synthetic energies."""
    get_cell_info = tc['physics.base.solidstate'].get_cell_info
    Converter = tc['physics.base.units'].Converter
    structure, shifts, all_shifts = first_interface(tc, pair)
    cell = structure.lattice.matrix
    sorted_values = synthetic_energies(shifts, cell)
    area, _ = get_cell_info(structure.lattice.matrix)
    sorted_values_jm2 = [[s, x, y, Converter.eV_to_Jm2(v, area * 1e-20)]
                         for s, x, y, v in sorted_values]
    get_pes = tc['physics.tribology.potensurf'].get_pes
    _, v_list, data, rbf = get_pes(all_shifts, sorted_values, cell, title=pair, to_fig=None)
    _, v_list_jm2, data_jm2, rbf_jm2 = get_pes(all_shifts, sorted_values_jm2, cell,
                                               title=pair, to_fig=None)
    return cell, {'energies_ev': sorted_values, 'energies_jm2': sorted_values_jm2,
                  'hs_all_ev': v_list, 'hs_all_jm2': v_list_jm2,
                  'pes_ev': data, 'pes_jm2': data_jm2,
                  'rbf_jm2_at_probes': rbf_jm2(probe_points(cell))}


def case_get_pes(tc, pair):
    return pes_from_hs(tc, pair)[1]


def case_shear_from_hs(tc, pair):
    """FT_CalcShearStrength (shearstr.py:110-127) on the PES of case_get_pes."""
    cell, pes = pes_from_hs(tc, pair)
    solver = tc['physics.tribology.pes'].PES(np.array(pes['pes_jm2']), cell)
    solver.make_pes(**SHEAR_PES)
    return run_mep(tc, solver._rbf, cell)


def pes_grid(tc, name):
    rel, columns, cell, params = PES_GRIDS[name]
    data = np.genfromtxt(DATA / rel)[:, columns]
    solver = tc['physics.tribology.pes'].PES(data.copy(), cell)
    solver.make_pes(**params)
    return solver, cell


def case_pes(tc, name):
    solver, _ = pes_grid(tc, name)
    return {'pes': solver.pes, 'cell_2d': solver.cell_2d, 'cell': solver.cell,
            'rbf_at_probes': solver._rbf(probe_points(solver.cell_2d))}


def case_shear(tc, name, pes_params=SHEAR_PES, mep_kwargs={}):
    """FT_CalcShearStrength (shearstr.py:113-127) on a PES of the test data."""
    rel, columns, cell, _ = PES_GRIDS[name]
    data = np.genfromtxt(DATA / rel)[:, columns]
    solver = tc['physics.tribology.pes'].PES(data, cell)
    solver.make_pes(**pes_params)
    return run_mep(tc, solver._rbf, cell, **mep_kwargs)


def case_uber(tc):
    """FT_FitPPES.arrange_data (pes.py:358-360): fit_ppes(values, 'uber') on
    every PPES of data/homogeneous/ppes_data, with the distances of the files
    ('absolute') and with the distances minus the one of minimum energy
    ('relative'), like the shifts from the equilibrium distance that
    FT_FitPPES receives."""
    fit_ppes = tc['physics.tribology.ppes'].fit_ppes
    results = {'absolute': {}, 'relative': {}}
    for path in sorted((DATA / PPES_DATA).rglob('*.txt')):
        data = np.loadtxt(path)
        relative = data.copy()
        relative[:, 0] -= data[np.argmin(data[:, 1]), 0]
        key = path.relative_to(DATA / PPES_DATA).as_posix()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results['absolute'][key] = outcome(fit_ppes, data.tolist(), 'uber')
            results['relative'][key] = outcome(fit_ppes, relative.tolist(), 'uber')
    return results


def case_uber0(tc):
    """fit_ppes(values, 'uber0') on one PPES (A06)."""
    values = np.loadtxt(DATA / PPES_DATA / 'min' / '111' / 'Cu.txt').tolist()
    return outcome(tc['physics.tribology.ppes'].fit_ppes, values, 'uber0')


# Slabs read from a file as the custom-slab workflow stores them: in the
# historic environment FT_StoreCustomSlab.from_input_files on POSCAR, KPOINTS
# and INCAR, then Slab.from_dict as FT_StartInterfaceMatch (run.py:1028); in
# the development one its port, triboworkflow.core.structures.custom_slab, on
# the same POSCAR read by read_structure. Cells: hexagonal with a along x,
# the same rotated by 30 degrees about z, rectangular with opposite moments,
# c tilted, left-handed; the last two are rejected by workflow_slab.
CUSTOM_SLAB_MILLER = [1, 1, 1]
_HEX = np.array([[2.556, 0.0, 0.0], [-1.278, 2.556 * math.sqrt(3) / 2, 0.0], [0.0, 0.0, 25.0]])
_HEX_FRAC = [[0.0, 0.0, 0.1], [1 / 3, 2 / 3, 0.18], [2 / 3, 1 / 3, 0.26]]
_ROT30 = np.array([[math.cos(math.pi / 6), math.sin(math.pi / 6), 0.0],
                   [-math.sin(math.pi / 6), math.cos(math.pi / 6), 0.0], [0.0, 0.0, 1.0]])
CUSTOM_SLABS = {
    'hex': {'lattice': _HEX, 'species': ['Cu'], 'counts': [3], 'frac': _HEX_FRAC,
            'flags': [[False] * 3, [False] * 3, [True] * 3]},
    'rotated': {'lattice': _HEX @ _ROT30, 'species': ['Cu'], 'counts': [3], 'frac': _HEX_FRAC},
    'magmom': {'lattice': np.diag([2.87, 4.0588, 20.0]), 'species': ['Fe'], 'counts': [4],
               'frac': [[0, 0, 0.1], [0.5, 0.5, 0.1], [0, 0.5, 0.17], [0.5, 0, 0.17]],
               'magmom': [2.5, -2.5, 2.5, -2.5]},
    'tilted': {'lattice': np.array([[2.556, 0, 0], [-1.278, 2.556 * math.sqrt(3) / 2, 0], [0.5, 0.3, 25.0]]),
               'species': ['Cu'], 'counts': [3], 'frac': _HEX_FRAC},
    'lefthanded': {'lattice': np.array([[2.556, 0, 0], [-1.278, -2.556 * math.sqrt(3) / 2, 0], [0, 0, 25.0]]),
                   'species': ['Cu'], 'counts': [3], 'frac': _HEX_FRAC},
}


def poscar_text(lattice, species, counts, frac, flags=None, **_):
    """POSCAR with Direct coordinates and, with flags, Selective dynamics."""
    lines = ['custom slab', '1.0']
    lines += ['  '.join(f'{x:.10f}' for x in v) for v in lattice]
    lines += [' '.join(species), ' '.join(map(str, counts))]
    lines += ['Selective dynamics'] if flags else []
    lines.append('Direct')
    for i, coords in enumerate(frac):
        sd = '  ' + ' '.join('T' if f else 'F' for f in flags[i]) if flags else ''
        lines.append('  '.join(f'{x:.10f}' for x in coords) + sd)
    return '\n'.join(lines) + '\n'


def case_custom_slab(tc, name):
    """The Slab that the interface workflow receives for a custom slab."""
    from pymatgen.core.surface import Slab
    slab_data = CUSTOM_SLABS[name]
    Path('POSCAR').write_text(poscar_text(**slab_data))
    magmom = slab_data.get('magmom')
    if tc.root == HISTORIC:
        Path('KPOINTS').write_text('Automatic mesh\n0\nMonkhorst\n8 8 1\n0 0 0\n')
        incar = 'ENCUT = 400\nISMEAR = 1\nISPIN = 2\n'
        incar += f'MAGMOM = {" ".join(map(str, magmom))}\n' if magmom else ''
        Path('INCAR').write_text(incar)
        params = {'miller': CUSTOM_SLAB_MILLER, 'input_dir': '.', 'reconstructed': None,
                  'bulk_modulus': None, 'magmom_from': 'INCAR'}
        store = tc['highput.firetasks.custom_slab'].FT_StoreCustomSlab
        slab = Slab.from_dict(store.from_input_files(None, params)['structure']['opt'])
    else:
        from triboworkflow.core.structures import custom_slab, read_structure
        slab = custom_slab(read_structure('POSCAR'), CUSTOM_SLAB_MILLER, magmom)
    return {'slab': slab.as_dict(), 'labels': slab.labels}


# Step match: the Firetasks of TribChem (historic) and the step of
# triboworkflow (port) on the same slabs and parameters, compared through the
# database calls that they make. The matching parameters are those that
# TribChem's command line passes to FT_StartInterfaceMatch
# (core/cli/calc_interface_cli.py:151-167: floats, best_match dropped), with
# max_angle_tol = 0.4, the value that the historic Firetasks use at every
# attempt and the default of the port (DEV-01).
MATCH_CLI = {'max_area': 160.0, 'interface_distance': 2.0, 'max_angle_tol': 0.4,
             'max_length_tol': 0.02, 'max_area_ratio_tol': 0.05}
MATCH_MIDS = {'cu111': 'mp-30', 'al111': 'mp-134', 'pt111': 'mp-126', 'ag111': 'mp-124'}
MATCH_MODULI = {'cu111': 140.0, 'al111': 76.0, 'pt111': 230.0, 'ag111': 100.0}  # GPa
MATCH_COLLECTION = 'PBE.interface_elements'
# (slab 1, slab 2, inter_params, set_bm)
MATCH_STEPS = {
    'cu111_al111': ('cu111', 'al111', MATCH_CLI, True),
    'al111_cu111': ('al111', 'cu111', MATCH_CLI, True),
    'cu111_cu111': ('cu111', 'cu111', MATCH_CLI, True),
    # Larger tolerances: with those of the command line the match has 150
    # atoms, and the high-symmetry points take minutes.
    'pt111_ag111': ('pt111', 'ag111', MATCH_CLI | {'max_length_tol': 0.05,
                                                   'max_area_ratio_tol': 0.1}, False),
    'cu111_al111_enlarged': ('cu111', 'al111', MATCH_CLI | {'max_area': 20.0}, True),
    'cu111_al111_not_converged': ('cu111', 'al111', MATCH_CLI | {'max_area': 0.0}, True),
}


def match_step_historic(tc, names, inter_params, set_bm):
    """FT_StartInterfaceMatch (inputs and document of the interface), then the
    two Firetasks of InterfaceWF.match_interface, each on its own copy of the
    parameters as FireWorks gives them, with the database calls recorded.

    The database returns the slabs (with comp_params), the bulk moduli and no
    interface document. The comp_params of the inserted document are dropped:
    the port does not write them (DEV-35). The recursion of
    FT_GenerateInterface fails at the first failed attempt (A73): the final
    parameters come from the recursion of FT_GetHighSymmetryPts (DEV-33),
    and FT_GenerateInterface runs with them, matching at its first attempt.
    """
    run_ft = tc['highput.firetasks.run']
    core = tc['highput.firetasks.core']
    errors = tc['highput.utils.errors']
    interface_wf = tc['highput.workflows.interface'].InterfaceWF
    dicts = slab_dicts()
    mids = [MATCH_MIDS[n] for n in names]
    calls = []

    def query_db(collection, db_file=None, database=None, entry=None, fltr=None, **kwargs):
        if collection in ('slabs_1', 'slabs_2'):
            return {'comp_params': COMP_PARAMS}, copy.deepcopy(dicts[names[int(collection[-1]) - 1]])
        if collection == 'PBE.bulk_elements':
            return None, MATCH_MODULI[names[mids.index(fltr['mid'])]]
        if collection == MATCH_COLLECTION:
            return None, None
        raise AssertionError(f'unexpected query of {collection}')

    def update_db(self, data, collection, entry=None, db_file=None, database=None, fltr=None,
                  update_op='$set', upsert=False, to_mongodb=True, **kwargs):
        calls.append({'call': 'update_db', 'data': copy.deepcopy(data), 'collection': collection,
                      'entry': copy.deepcopy(entry), 'fltr': copy.deepcopy(fltr),
                      'update_op': update_op, 'to_mongodb': to_mongodb})

    def insert_db(self, field, collection, db_file=None, database=None, duplicates=False,
                  message=None):
        field = copy.deepcopy(field)
        field.pop('comp_params')
        calls.append({'call': 'insert_db', 'field': field, 'collection': collection})

    saved = core.FireTaskTribChem.update_db, core.FireTaskTribChem.insert_db
    core.FireTaskTribChem.update_db, core.FireTaskTribChem.insert_db = update_db, insert_db
    try:
        start = run_ft.FT_StartInterfaceMatch(
            mid=mids, miller=[[1, 1, 1], [1, 1, 1]], collection_slab1='slabs_1',
            collection_slab2='slabs_2', collection_inter=MATCH_COLLECTION,
            inter_params=copy.deepcopy(inter_params), set_bm=set_bm, db_file='golden')
        start.query_db = query_db
        params = start.read_runtask('StartInterfaceMatch', {})
        errors.InterfaceMatchError.check_inputs(params)
        params = start.query_structure(params)
        start.set_interface_document(params)
        wf = start.run_func(interface_wf.match_interface, **params)
        tasks = {type(t).__name__: t for fw in wf.fws for t in fw.tasks}
        generate = copy.deepcopy(tasks['FT_GenerateInterface'])
        highsym = copy.deepcopy(tasks['FT_GetHighSymmetryPts'])
        probe = copy.deepcopy(tasks['FT_GetHighSymmetryPts'])
        probe_params = probe.read_runtask('GetHighSymmetryPts', {})
        try:
            probe.generate_interface(probe_params)
        except errors.InterfaceMatchError as error:
            return {'calls': calls, 'error': type(error).__name__, 'message': str(error)}
        generate['inter_params'] = probe_params['inter_params']
        generate.run_task({})
        highsym.run_task({})
    finally:
        core.FireTaskTribChem.update_db, core.FireTaskTribChem.insert_db = saved
    return {'calls': calls}


def match_step_port(names, inter_params, set_bm):
    """The inputs of the step from settings with the slabs written to files,
    then run_match on a store that records the database calls."""
    import tomllib

    from triboworkflow.core.settings.loader import settings_from_dict
    from triboworkflow.core.store import SystemDocument
    from triboworkflow.core.store.legacy import LegacyStore
    from triboworkflow.workflows.interface import match as step

    class Recording(LegacyStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = []

        def update_db(self, data, collection, *, entry=None, fltr=None, update_op='$set',
                      to_mongodb=True):
            self.calls.append({'call': 'update_db', 'data': copy.deepcopy(data),
                               'collection': collection, 'entry': copy.deepcopy(entry),
                               'fltr': copy.deepcopy(fltr), 'update_op': update_op,
                               'to_mongodb': to_mongodb})
            super().update_db(data, collection, entry=entry, fltr=fltr, update_op=update_op,
                              to_mongodb=to_mongodb)

        def insert_db(self, field, collection, *, duplicates=False):
            self.calls.append({'call': 'insert_db', 'field': copy.deepcopy(field),
                               'collection': collection})
            super().insert_db(field, collection, duplicates=duplicates)

    data = tomllib.loads((ROOT / 'examples' / 'interface.toml').read_text(encoding='utf-8'))
    dicts = slab_dicts()
    slabs = []
    for i, name in enumerate(names, 1):
        Path(f'slab_{i}.json').write_text(json.dumps(dicts[name]))
        slab = {'file': f'slab_{i}.json', 'mid': MATCH_MIDS[name], 'miller': [1, 1, 1]}
        if set_bm:
            slab['bulk_modulus'] = MATCH_MODULI[name]
        slabs.append(slab)
    data['interface'].update({'slabs': slabs, 'set_bm': set_bm, 'steps': ['match'],
                              'inter_params': dict(inter_params)})
    inputs = step.match_inputs(settings_from_dict(data))
    store = Recording('.', SystemDocument(), step.COLLECTION)
    try:
        step.run_match(inputs, store)
    except step.InterfaceMatchError as error:
        return {'calls': store.calls, 'error': type(error).__name__, 'message': str(error)}
    return {'calls': store.calls}


def case_match_step(tc, name):
    """Database calls of the step match: document, interface, parameters, points."""
    slab_1, slab_2, inter_params, set_bm = MATCH_STEPS[name]
    names = [slab_1, slab_2]
    if tc.root == HISTORIC:
        return match_step_historic(tc, names, inter_params, set_bm)
    return match_step_port(names, inter_params, set_bm)


# Steps pes and pes_scf: FT_StartPES, InterfaceWF.calculate_pes, the
# FT_MoveTagResults of its points and FT_PotentialEnergySurface (historic)
# against pes_inputs and run_pes_analysis of the steps (port), on the same
# document of the interface: the one that the golden calls of the step match
# build, with the comp_params that TribChem's document held (DEV-35: the port
# takes kdens from the settings, here COMP_PARAMS['kdens']). The results of the
# points come from task documents made by the case (pes_task_document), the
# same in both: the energies of synthetic_energies, the structure of each
# point as input and final structure, and the calc_id <step>/<point> as
# task_label, task_id and dir_name (atomate gave a tag with a uuid4 and a
# path). The cases compare the points, structures and k-points of the
# calculations and the database writes. Interface distance 2.0, as TribChem's
# command line passes it (A76). The pairs are those whose ids TribChem makes
# again equal to those of match (A80, DEV-39).
PES_STEP_CALC_TYPES = {'pes': 'interface_z_relax', 'pes_scf': 'slab_from_scratch'}
PES_STEP_DISTANCE = 2.0
# (pair of MATCH_STEPS, step, short): short None for complete slabs, 'rule'
# for the layers of TribChem's command line (calc_interface.py:151) and of the
# port without layers, or the layers [bottom, top] given.
PES_STEPS = {
    'cu111_al111': ('cu111_al111', 'pes', None),
    'scf_cu111_al111': ('cu111_al111', 'pes_scf', None),
    'scf_short_cu111_al111': ('cu111_al111', 'pes_scf', 'rule'),
    'scf_short_2_1_cu111_al111': ('cu111_al111', 'pes_scf', [2, 1]),
    'cu111_cu111': ('cu111_cu111', 'pes', None),
    'scf_cu111_cu111': ('cu111_cu111', 'pes_scf', None),
    'pt111_ag111': ('pt111_ag111', 'pes', None),
}


def mongo_set(document, data, entry, to_mongodb):
    """A historic update_db applied to a document as a $set: the dicts merged
    leaf by leaf with to_mongodb (empty dicts write nothing), otherwise the
    top-level key of the entry replaced."""
    path = [entry] if isinstance(entry, str) else list(entry)
    value = copy.deepcopy(data)
    for key in reversed(path[1:]):
        value = {key: value}
    if not to_mongodb:
        document[path[0]] = value
        return

    def leaves(prefix, item):
        if isinstance(item, dict):
            for k, v in item.items():
                yield from leaves(prefix + [k], v)
        else:
            yield prefix, item

    for keys, item in leaves(path[:1], value):
        node = document
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = item


def pes_document(pair):
    """Document of the interface after match, built from the golden calls of
    match_step_<pair>, with COMP_PARAMS."""
    document = None
    for call in read_golden(f'match_step_{pair}')['calls']:
        if call['call'] == 'insert_db':
            document = copy.deepcopy(call['field'])
        else:
            mongo_set(document, call['data'], call['entry'], call['to_mongodb'])
    document['comp_params'] = copy.deepcopy(COMP_PARAMS)
    return document


def pes_energies(document):
    """{point: energy} of synthetic_energies on the cell of the interface."""
    cell = document['structure']['init']['interface']['lattice']['matrix']
    shifts = document['highsym']['inter_unique']
    return {label: energy for label, _, _, energy in synthetic_energies(shifts, cell)}


def pes_task_document(step, point, structure, energy):
    """Task document of the calculation of a point, as the tasks collection
    gives it to FT_MoveTagResults: the structure of the point as input and
    final structure."""
    from monty.json import jsanitize
    s = jsanitize(structure.as_dict(), strict=False)
    n = len(s['sites'])
    calc_id = f'{step}/{point}'
    return {'nsites': n,
            'output': {'structure': copy.deepcopy(s), 'density': float(structure.density),
                       'energy': energy, 'energy_per_atom': energy / n},
            'input': {'structure': s},
            'task_label': calc_id, 'task_id': calc_id, 'dir_name': calc_id}


def pes_inputs_data(points, structures, kpoints):
    return {'points': list(points), 'structures': structures, 'kpoints': kpoints}


def update_call(data, collection, entry, fltr, update_op, to_mongodb):
    return {'call': 'update_db', 'data': copy.deepcopy(data), 'collection': collection,
            'entry': copy.deepcopy(entry), 'fltr': copy.deepcopy(fltr),
            'update_op': update_op, 'to_mongodb': to_mongodb}


def pes_step_historic(tc, pair, step, short):
    """FT_StartPES up to InterfaceWF.calculate_pes, then the FT_MoveTagResults
    and the FT_PotentialEnergySurface of its workflow, on a database emulated
    with TribChem's prepare_data_to_update ($set of the update dicts)."""
    run_ft = tc['highput.firetasks.run']
    core = tc['highput.firetasks.core']
    query = tc['highput.database.query']
    dbtools = tc['highput.database.dbtools']
    interface_wf = tc['highput.workflows.interface'].InterfaceWF
    document = pes_document(pair)
    tasks = {}
    calls = []

    def matching(fltr):
        if any(document.get(k) != v for k, v in fltr.items()):
            raise AssertionError(f'filter {fltr} does not match the document')

    def query_db(self, collection, db_file=None, database=None, entry=None, fltr=None,
                 **kwargs):
        if collection == 'tasks':
            field = copy.deepcopy(tasks[fltr['task_label']])
        elif collection == MATCH_COLLECTION:
            matching(fltr)
            field = copy.deepcopy(document)
        else:
            raise AssertionError(f'unexpected query of {collection}')
        return field, dbtools.read_multiple_entry(field, entry)

    def update_db(self, data, collection, entry=None, db_file=None, database=None, fltr=None,
                  update_op='$set', upsert=False, to_mongodb=True, **kwargs):
        if collection != MATCH_COLLECTION:
            raise AssertionError(f'unexpected update of {collection}')
        matching(fltr)
        calls.append(update_call(data, collection, entry, fltr, update_op, to_mongodb))
        updates = query.prepare_data_to_update(copy.deepcopy(data), entry, update_op, to_mongodb)
        for update in updates if isinstance(updates, list) else [updates]:
            (op, values), = update.items()
            if op != '$set':
                raise AssertionError(f'unexpected update operator {op}')
            for key, value in values.items():
                *parents, last = key.split('.')
                node = document
                for k in parents:
                    node = node.setdefault(k, {})
                node[last] = value

    layers = None
    if short == 'rule':
        layers = [3 if m == [1, 1, 1] else 2 for m in document['miller']]
    elif short is not None:
        layers = list(short)
    saved = core.FireTaskTribChem.query_db, core.FireTaskTribChem.update_db
    core.FireTaskTribChem.query_db, core.FireTaskTribChem.update_db = query_db, update_db
    try:
        start = run_ft.FT_StartPES(
            mid=document['mid'].split('_'), miller=document['miller'],
            collection=MATCH_COLLECTION, calc_type=PES_STEP_CALC_TYPES[step], layers=layers,
            interface_distance=PES_STEP_DISTANCE, db_file='golden')
        params = start.read_runtask('StartPES', {})
        params['main_folder'] = 'golden'  # get_main_folder reads the database directly (A47)
        _, params = start.query_interface(params)
        wf = start.run_func(interface_wf.calculate_pes, **params)
        fw_tasks = [t for fw in wf.fws for t in fw.tasks]
        run_vasp = next(t for t in fw_tasks if type(t).__name__ == 'FT_RunVaspOnTheFly')
        points = list(params['shifts'])
        energies = pes_energies(document)
        for point, structure, tag in zip(points, run_vasp['structure'], run_vasp['tag']):
            tasks[tag] = pes_task_document(step, point, structure, energies[point])
        for name in ('FT_MoveTagResults', 'FT_PotentialEnergySurface'):
            for task in fw_tasks:
                if type(task).__name__ == name:
                    task.run_task({})
    finally:
        core.FireTaskTribChem.query_db, core.FireTaskTribChem.update_db = saved
    return {'inputs': pes_inputs_data(points, run_vasp['structure'],
                                      [v.kpoints for v in run_vasp['vis']]),
            'calls': calls}


def port_settings(pair, mids, steps, short, adhesion=None):
    """Settings of a run of the port on the slabs of a pair of MATCH_STEPS,
    written as JSON files in the current folder, with the parameters of the
    PES of the golden cases (distance, k-points of COMP_PARAMS, short slabs)
    and, when given, [interface.adhesion]."""
    import tomllib

    from triboworkflow.core.settings.loader import settings_from_dict

    data = tomllib.loads((ROOT / 'examples' / 'interface.toml').read_text(encoding='utf-8'))
    dicts = slab_dicts()
    slabs = []
    for i, (name, mid) in enumerate(zip(MATCH_STEPS[pair][:2], mids), 1):
        Path(f'slab_{i}.json').write_text(json.dumps(dicts[name]))
        slabs.append({'file': f'slab_{i}.json', 'mid': mid, 'miller': [1, 1, 1]})
    pes_scf = {'short': short is not None}
    if isinstance(short, list):
        pes_scf['layers'] = list(short)
    data['interface'].update({'slabs': slabs, 'set_bm': False, 'steps': list(steps),
                              'pes_scf': pes_scf})
    data['interface']['inter_params']['interface_distance'] = PES_STEP_DISTANCE
    if adhesion is not None:
        data['interface']['adhesion'] = dict(adhesion)
    data['k_points'].update({'mode': 'kdens', 'kdens': float(COMP_PARAMS['kdens'])})
    return settings_from_dict(data)


def pes_step_port(tc, pair, step, short):
    """pes_inputs of the step from settings and a run with the document of
    match, then run_pes_analysis on a store that records the database calls
    and gives the task documents."""
    from triboworkflow.core.store import SystemWriter
    from triboworkflow.core.store.legacy import TASKS, LegacyStore
    from triboworkflow.execution.layout import system_dir
    from triboworkflow.workflows.interface import pes

    dbtools = tc['highput.database.dbtools']
    tasks = {}

    class Recording(LegacyStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = []

        def query_db(self, collection, *, entry=None, fltr=None):
            if collection == TASKS:
                field = copy.deepcopy(tasks[fltr['task_label']])
                return field, dbtools.read_multiple_entry(field, entry)
            return super().query_db(collection, entry=entry, fltr=fltr)

        def update_db(self, data, collection, *, entry=None, fltr=None, update_op='$set',
                      to_mongodb=True):
            self.calls.append(update_call(data, collection, entry, fltr, update_op, to_mongodb))
            super().update_db(data, collection, entry=entry, fltr=fltr, update_op=update_op,
                              to_mongodb=to_mongodb)

    document = pes_document(pair)
    settings = port_settings(pair, document['mid'].split('_'), ['match', step], short)

    system = document['mid']
    sdir = system_dir(Path('run'), system)
    with SystemWriter(sdir) as writer:
        writer.document.create(document)
        writer.save()
    inputs = pes.pes_inputs(Path('run'), system, settings, step)
    energies = pes_energies(document)
    for point, structure, _ in inputs.points:
        tasks[f'{step}/{point}'] = pes_task_document(step, point, structure, energies[point])
    with SystemWriter(sdir) as writer:
        store = Recording(sdir, writer.document, pes.COLLECTION)
        pes.run_pes_analysis(inputs, store, [f'{step}/{point}' for point, _, _ in inputs.points])
    return {'inputs': pes_inputs_data([p for p, _, _ in inputs.points],
                                      [s for _, s, _ in inputs.points],
                                      [k for _, _, k in inputs.points]),
            'calls': store.calls}


def case_pes_step(tc, name):
    """Calculations and database writes of the steps pes and pes_scf."""
    pair, step, short = PES_STEPS[name]
    if tc.root == HISTORIC:
        return pes_step_historic(tc, pair, step, short)
    return pes_step_port(tc, pair, step, short)


# Interpolation of the PES: the cases of PES_STEPS whose PES is interpolated,
# with the step that interpolates it. Not on Cu/Al: its 756 points left after
# the duplicates, replicated 12 x 12, need a dense matrix of about 88 GiB.
PLOTPES_STEPS = ('cu111_cu111', 'scf_cu111_cu111', 'pt111_ag111')
PLOTPES_OF = {'pes': 'plotpes', 'pes_scf': 'plotpes_scf'}


def plotpes_document(name):
    """Document of the interface after match and the PES of the case
    pes_step_<name>, built from the golden calls of both. A call with a list
    of entries and as many values sets each value in its entry, as
    write_multiple_entry (dbtools.py:235-301) splits it."""
    document = pes_document(PES_STEPS[name][0])
    for call in read_golden(f'pes_step_{name}')['calls']:
        if call['update_op'] != '$set':
            raise AssertionError(f'unexpected update operator {call["update_op"]}')
        entry, data = call['entry'], call['data']
        if (isinstance(entry, list) and all(isinstance(n, list) for n in entry)
                and isinstance(data, list) and len(data) == len(entry)):
            for value, path in zip(data, entry):
                mongo_set(document, value, path, call['to_mongodb'])
        else:
            mongo_set(document, data, entry, call['to_mongodb'])
    return document


def png_pixels(data):
    """Mode, size and SHA-256 of the pixels of a PNG given as bytes: the same
    for the same image, whatever its encoding."""
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return {'mode': image.mode, 'size': list(image.size),
                'sha256': vendoring.sha256(image.tobytes())}


def plotpes_step_historic(tc, name):
    """FT_StartInterpolation as TribChem's command line builds it, then the
    FT_PESInterpolation of its detour (arrange_data and compute_pes, as its
    run_task) on the document emulated as for the PES. The key image written,
    the PNG of the figure encoded again with PIL, is replaced by its pixels."""
    run_ft = tc['highput.firetasks.run']
    core = tc['highput.firetasks.core']
    dbtools = tc['highput.database.dbtools']
    step = PES_STEPS[name][1]
    document = plotpes_document(name)
    calls = []

    def matching(fltr):
        if any(document.get(k) != v for k, v in fltr.items()):
            raise AssertionError(f'filter {fltr} does not match the document')

    def query_db(self, collection, db_file=None, database=None, entry=None, fltr=None,
                 **kwargs):
        if collection != MATCH_COLLECTION:
            raise AssertionError(f'unexpected query of {collection}')
        matching(fltr)
        field = copy.deepcopy(document)
        return field, dbtools.read_multiple_entry(field, entry)

    def update_db(self, data, collection, entry=None, db_file=None, database=None, fltr=None,
                  update_op='$set', upsert=False, to_mongodb=True, **kwargs):
        if collection != MATCH_COLLECTION:
            raise AssertionError(f'unexpected update of {collection}')
        matching(fltr)
        calls.append(update_call(data, collection, entry, fltr, update_op, to_mongodb))

    saved = core.FireTaskTribChem.query_db, core.FireTaskTribChem.update_db
    core.FireTaskTribChem.query_db, core.FireTaskTribChem.update_db = query_db, update_db
    try:
        # calc_interface_cli.py:191-197 (set_pes), calc_interface.py:184-190
        start = run_ft.FT_StartInterpolation(
            mid=document['mid'].split('_'), miller=document['miller'],
            collection=MATCH_COLLECTION, calc_type=PES_STEP_CALC_TYPES[step], check_entry=step,
            db_file='golden')
        detours = start.run_task({}).detours
        (wf,) = detours if isinstance(detours, (list, tuple)) else [detours]
        (task,) = [t for fw in wf.fws for t in fw.tasks]
        params = task.read_runtask('PESInterpolation', {})
        hs_all, values, cell = task.arrange_data(params)
        task.compute_pes(hs_all, values, cell, params)
    finally:
        core.FireTaskTribChem.query_db, core.FireTaskTribChem.update_db = saved
    (call,) = calls
    image = call['data'][params['pes_type']].pop('image')
    return {'inputs': {'inter_all': hs_all, 'values': values, 'cell': cell},
            'figure': sorted(p.name for p in Path('.').glob('*.png')),
            'image': png_pixels(image), 'calls': calls}


def plotpes_step_port(tc, name):
    """plotpes_inputs of the step from settings and a run with the document
    after the PES, then interpolate and write_interpolation on a store that
    records the database calls; the pixels of the figure written."""
    from triboworkflow.core.store import SystemWriter
    from triboworkflow.core.store.legacy import LegacyStore
    from triboworkflow.execution.layout import system_dir
    from triboworkflow.workflows.interface import plotpes

    class Recording(LegacyStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = []

        def update_db(self, data, collection, *, entry=None, fltr=None, update_op='$set',
                      to_mongodb=True):
            self.calls.append(update_call(data, collection, entry, fltr, update_op, to_mongodb))
            super().update_db(data, collection, entry=entry, fltr=fltr, update_op=update_op,
                              to_mongodb=to_mongodb)

    pair, step, short = PES_STEPS[name]
    document = plotpes_document(name)
    settings = port_settings(pair, document['mid'].split('_'),
                             ['match', step, PLOTPES_OF[step]], short)
    sdir = system_dir(Path('run'), document['mid'])
    with SystemWriter(sdir) as writer:
        writer.document.create(document)
        inputs = plotpes.plotpes_inputs(sdir, writer.document, settings, PLOTPES_OF[step])
        figure = plotpes.figure_path(sdir, inputs.name, inputs.title)
        figure.parent.mkdir(parents=True)
        interpolation = plotpes.interpolate(inputs, figure)
        store = Recording(sdir, writer.document, plotpes.COLLECTION)
        plotpes.write_interpolation(store, inputs, interpolation)
    return {'inputs': {'inter_all': inputs.all_shifts, 'values': inputs.values,
                       'cell': inputs.cell},
            'figure': [figure.name], 'image': png_pixels(figure.read_bytes()),
            'calls': store.calls}


def case_plotpes_step(tc, name):
    """Inputs, database writes and figure of the steps plotpes and plotpes_scf."""
    if tc.root == HISTORIC:
        return plotpes_step_historic(tc, name)
    return plotpes_step_port(tc, name)


# Step adhesion: FT_StartAdhesion up to InterfaceWF.adhesion_energy, the
# FT_MoveTagResults of its four calculations and FT_AdhesionEnergy (historic)
# against adhesion_inputs and run_adhesion_analysis of the step (port), on the
# document that the golden calls of the step match and of the PES of the case
# build (plotpes_document, with the comp_params of TribChem's document: DEV-35).
# Each case is a PES case of PES_STEPS: the relaxed PES gives the regular
# adhesion, the static one on complete slabs the regular adhesion on the slabs
# stacked at its sites (without a distance: A86), the static one on reduced
# slabs the short adhesion, which TribChem's command line pairs with it. The
# results of the four calculations come from task documents made by the case
# (pes_task_document), the same in both: the energies of ADHESION_ENERGIES, the
# structure of each calculation as input and final structure, and the calc_id
# adhesion/<label> as task_label, task_id and dir_name. The cases compare the
# structures, calc_types and k-points of the four calculations and the database
# writes, the structure.init.short ones included.
ADHESION_LABELS = ['bot_slab', 'top_slab', 'interface_min', 'interface_max']
ADHESION_CALC_TYPES = ['slab_pos_relax', 'slab_pos_relax', 'slab_pos_relax', 'interface_z_relax']
ADHESION_ENERGIES = {'bot_slab': -1000.25, 'top_slab': -1008.5, 'interface_min': -2010.125,
                     'interface_max': -2009.0}
ADHESION_STEPS = tuple(PES_STEPS)


def adhesion_kind(short):
    """Kind of adhesion paired with a PES case: short for reduced slabs, else regular."""
    return 'regular' if short is None else 'short'


def emulated_database(tc, document, tasks, calls):
    """The query_db and update_db of FireTaskTribChem on a database emulated with the
    document: queries give a copy of it, or the task document with the given tag,
    and the updates are recorded in calls and applied with TribChem's
    prepare_data_to_update ($set of the update dicts)."""
    query = tc['highput.database.query']
    dbtools = tc['highput.database.dbtools']

    def matching(fltr):
        if any(document.get(k) != v for k, v in fltr.items()):
            raise AssertionError(f'filter {fltr} does not match the document')

    def query_db(self, collection, db_file=None, database=None, entry=None, fltr=None,
                 **kwargs):
        if collection == 'tasks':
            field = copy.deepcopy(tasks[fltr['task_label']])
        elif collection == MATCH_COLLECTION:
            matching(fltr)
            field = copy.deepcopy(document)
        else:
            raise AssertionError(f'unexpected query of {collection}')
        return field, dbtools.read_multiple_entry(field, entry)

    def update_db(self, data, collection, entry=None, db_file=None, database=None, fltr=None,
                  update_op='$set', upsert=False, to_mongodb=True, **kwargs):
        if collection != MATCH_COLLECTION:
            raise AssertionError(f'unexpected update of {collection}')
        matching(fltr)
        calls.append(update_call(data, collection, entry, fltr, update_op, to_mongodb))
        updates = query.prepare_data_to_update(copy.deepcopy(data), entry, update_op, to_mongodb)
        for update in updates if isinstance(updates, list) else [updates]:
            (op, values), = update.items()
            if op != '$set':
                raise AssertionError(f'unexpected update operator {op}')
            for key, value in values.items():
                *parents, last = key.split('.')
                node = document
                for k in parents:
                    node = node.setdefault(k, {})
                node[last] = value

    return query_db, update_db


def adhesion_inputs_data(labels, calc_types, structures, kpoints):
    return {'labels': list(labels), 'calc_types': list(calc_types),
            'structures': structures, 'kpoints': kpoints}


def adhesion_step_historic(tc, name):
    """FT_StartAdhesion up to InterfaceWF.adhesion_energy, then the FT_MoveTagResults
    of the four calculations and the FT_AdhesionEnergy of its workflow, on a
    database emulated with TribChem's prepare_data_to_update."""
    run_ft = tc['highput.firetasks.run']
    core = tc['highput.firetasks.core']
    interface_wf = tc['highput.workflows.interface'].InterfaceWF
    pair, step, short = PES_STEPS[name]
    document = plotpes_document(name)
    tasks, calls = {}, []
    query_db, update_db = emulated_database(tc, document, tasks, calls)
    saved = core.FireTaskTribChem.query_db, core.FireTaskTribChem.update_db
    core.FireTaskTribChem.query_db, core.FireTaskTribChem.update_db = query_db, update_db
    try:
        # calc_interface.py:192-199 (check_entry not given), set_adh of the command line
        start = run_ft.FT_StartAdhesion(
            mid=document['mid'].split('_'), miller=document['miller'],
            collection=MATCH_COLLECTION, pes_type=step, adhesion_type=adhesion_kind(short),
            db_file='golden')
        params = start.read_runtask('StartAdhesion', {})
        params['main_folder'] = 'golden'  # get_main_folder reads the database directly (A47)
        _, params = start.query_interface(params)
        wf = start.run_func(interface_wf.adhesion_energy, **params)
        fw_tasks = [t for fw in wf.fws for t in fw.tasks]
        run_vasp = [t for t in fw_tasks if type(t).__name__ == 'FT_RunVaspOnTheFly']
        for label, task in zip(ADHESION_LABELS, run_vasp):
            if f'_{label}_' not in task['tag']:
                raise AssertionError(f'{task["tag"]} is not the calculation {label}')
            tasks[task['tag']] = pes_task_document('adhesion', label, task['structure'],
                                                   ADHESION_ENERGIES[label])
        for kind in ('FT_MoveTagResults', 'FT_AdhesionEnergy'):
            for task in fw_tasks:
                if type(task).__name__ == kind:
                    task.run_task({})
    finally:
        core.FireTaskTribChem.query_db, core.FireTaskTribChem.update_db = saved
    return {'inputs': adhesion_inputs_data(ADHESION_LABELS, [t['calc_type'] for t in run_vasp],
                                           [t['structure'] for t in run_vasp],
                                           [t['vis'].kpoints for t in run_vasp]),
            'calls': calls}


def adhesion_step_port(tc, name):
    """adhesion_inputs of the step from settings and a run with the document
    after match and the PES, then run_adhesion_analysis on a store that records
    the database calls and gives the task documents."""
    from triboworkflow.core.store import SystemWriter
    from triboworkflow.core.store.legacy import TASKS, LegacyStore
    from triboworkflow.execution.layout import system_dir
    from triboworkflow.workflows.interface import adhesion

    dbtools = tc['highput.database.dbtools']
    tasks = {}

    class Recording(LegacyStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = []

        def query_db(self, collection, *, entry=None, fltr=None):
            if collection == TASKS:
                field = copy.deepcopy(tasks[fltr['task_label']])
                return field, dbtools.read_multiple_entry(field, entry)
            return super().query_db(collection, entry=entry, fltr=fltr)

        def update_db(self, data, collection, *, entry=None, fltr=None, update_op='$set',
                      to_mongodb=True):
            self.calls.append(update_call(data, collection, entry, fltr, update_op, to_mongodb))
            super().update_db(data, collection, entry=entry, fltr=fltr, update_op=update_op,
                              to_mongodb=to_mongodb)

    pair, step, short = PES_STEPS[name]
    document = plotpes_document(name)
    settings = port_settings(pair, document['mid'].split('_'), ['match', step, 'adhesion'], short,
                             adhesion={'pes': step, 'type': adhesion_kind(short)})
    system = document['mid']
    sdir = system_dir(Path('run'), system)
    with SystemWriter(sdir) as writer:
        writer.document.create(document)
        writer.save()
    inputs = adhesion.adhesion_inputs(Path('run'), system, settings)
    for label, _, structure, _ in inputs.points:
        tasks[f'adhesion/{label}'] = pes_task_document('adhesion', label, structure,
                                                       ADHESION_ENERGIES[label])
    with SystemWriter(sdir) as writer:
        store = Recording(sdir, writer.document, adhesion.COLLECTION)
        adhesion.run_adhesion_analysis(inputs, store,
                                       [f'adhesion/{label}' for label in ADHESION_LABELS])
    return {'inputs': adhesion_inputs_data([p[0] for p in inputs.points],
                                           [p[1] for p in inputs.points],
                                           [p[2] for p in inputs.points],
                                           [p[3] for p in inputs.points]),
            'calls': store.calls}


def case_adhesion_step(tc, name):
    """Calculations and database writes of the step adhesion."""
    if tc.root == HISTORIC:
        return adhesion_step_historic(tc, name)
    return adhesion_step_port(tc, name)


# Charge density. The workflow reads three densities on the same grid and in
# the same cell (interface, bottom slab, top slab), averages them along each
# axis and integrates |interface - bottom - top| (chgcar_integrator, on
# CHGCAR files). analyzer.py reads the files that pp.x writes; no Firetask
# calls it, its cases characterise it for the comparison of Q18. The
# densities are synthetic (tests/charge_files.py): Gaussian layers of a toy
# Cu/Al interface, with charge moved between the slabs in the interface.
CHARGE_PARTS = ('interface', 'bot_slab', 'top_slab')
CHARGE_GRID = (6, 5, 40)
CHARGE_CELLS = {
    'orthorhombic': np.diag([2.5, 2.8, 20.0]),
    'hexagonal': np.array([[2.556, 0.0, 0.0], [-1.278, 2.556 * math.sqrt(3) / 2, 0.0],
                           [0.0, 0.0, 20.0]]),
}
CHARGE_LAYERS = {'bot_slab': [(4.0, 11.0), (6.1, 11.0), (8.2, 11.0)],
                 'top_slab': [(10.6, 3.0), (12.9, 3.0), (15.2, 3.0)]}
CHARGE_SPECIES = {'bot_slab': ('Cu', 11.0), 'top_slab': ('Al', 3.0)}
CHARGE_TRANSFER = (9.4, 0.4, 0.5)
# integrate_in of the workflow: the two layers facing each other.
CHARGE_INTEGRATE_IN = (8.2, 10.6)
CHARGE_ECUTWFC, CHARGE_DUAL = 40.0, 4.0


def charge_part(cell, part):
    """Layers, species, counts, fractional positions and density of a part."""
    import charge_files
    slabs = ['bot_slab', 'top_slab'] if part == 'interface' else [part]
    layers = [layer for s in slabs for layer in CHARGE_LAYERS[s]]
    transfer = CHARGE_TRANSFER if part == 'interface' else None
    values = charge_files.slab_density(cell, CHARGE_GRID, layers, transfer=transfer)
    species = [CHARGE_SPECIES[s] for s in slabs]
    counts = [len(CHARGE_LAYERS[s]) for s in slabs]
    inplane = [(0.0, 0.0), (0.5, 0.5), (0.0, 0.5)]
    frac = [(*inplane[i % 3], z / cell[2, 2]) for s in slabs
            for i, (z, _) in enumerate(CHARGE_LAYERS[s])]
    return species, counts, frac, values


def write_chgcars(cell_name):
    """CHGCAR files of the three parts in the working folder; their paths."""
    import charge_files
    cell = CHARGE_CELLS[cell_name]
    paths = []
    for part in CHARGE_PARTS:
        species, counts, frac, values = charge_part(cell, part)
        Path(part).mkdir()
        Path(part, 'CHGCAR').write_text(charge_files.chgcar_text(
            lattice=cell, species=[s for s, _ in species], counts=counts, frac=frac,
            values=values))
        paths.append(f'{part}/CHGCAR')
    return paths


def filplot_of(cell, part, title='', nrx=None):
    """pp.x filplot text of a part (ibrav 0, alat the length of a1)."""
    import charge_files
    species, counts, frac, values = charge_part(cell, part)
    bohr = charge_files.BOHR_TO_ANGSTROM
    alat = np.linalg.norm(cell[0]) / bohr
    tau = np.asarray(frac) @ cell / (alat * bohr)
    ityp = [nt for nt, n in enumerate(counts, 1) for _ in range(n)]
    nr = CHARGE_GRID
    if nrx is not None:
        nrx = tuple(n + d for n, d in zip(nr, nrx))
    return charge_files.filplot_text(
        values=values * bohr ** 3, celldm=[alat], species=species,
        atoms=list(zip(tau.tolist(), ityp)), ibrav=0, at=cell / (alat * bohr), nrx=nrx,
        gcutm=CHARGE_DUAL * CHARGE_ECUTWFC / (2 * math.pi / alat) ** 2,
        dual=CHARGE_DUAL, ecut=CHARGE_ECUTWFC, title=title)


def write_filplots(cell_name, **kwargs):
    """Filplots of the three parts in the working folder; their paths."""
    paths = []
    for part in CHARGE_PARTS:
        Path(part).write_text(filplot_of(CHARGE_CELLS[cell_name], part, **kwargs))
        paths.append(part)
    return paths


def chgcar_summary(chgcar):
    """What pymatgen read: cell, volume, species and the total density."""
    return {'lattice': chgcar.structure.lattice.matrix, 'volume': chgcar.structure.volume,
            'species': [str(s) for s in chgcar.structure.species],
            'total': chgcar.data['total']}


def case_chgcar_average(tc, cell_name):
    """Averages along x, y and z of the three densities, as FT_AverageCharge."""
    ci = tc['physics.electronics.charge.chgcar_integrator']
    paths = write_chgcars(cell_name)
    chgcars = ci.extract_chgcar_data(paths)
    data = {'read': chgcar_summary(chgcars[0]),
            'single': chgcar_summary(ci.extract_chgcar_data(paths[0])),
            'average': {part: {str(ax): ci.chgcar_average_along_axis(chg, ax, to_plot=False)
                               for ax in (0, 1, 2)}
                        for part, chg in zip(CHARGE_PARTS, chgcars)}}
    return data


def case_chgcar_displacement(tc, cell_name):
    """Charge displacement of interface - bottom - top, as FT_ChargeDisplacement,
    with the variants of the integration limits and of the algebra."""
    ci = tc['physics.electronics.charge.chgcar_integrator']
    paths = write_chgcars(cell_name)
    operations, axis = ['-', '-'], 2
    combined = ci.chgcar_algebra_from_file(paths, operations, axis)
    p1, p2 = CHARGE_INTEGRATE_IN

    def summary(result, curve=False):
        integral, width, xy, curve_integral = result
        out = {'integral': integral, 'width': width, 'xy': xy}
        if curve:
            out['curve_integral'] = curve_integral
        return out

    data = {
        'combined': combined.data['total'],
        'interface': summary(ci.charge_displacement(combined, axis, p1, p2, 10000, False)),
        'all': summary(ci.charge_displacement(combined, axis, rescale=10000, to_plot=False)),
        'interface_101': summary(ci.charge_displacement(combined, axis, p1, p2, 101), True),
        'swapped_101': summary(ci.charge_displacement(combined, axis, p2, p1, 101)),
        'outside_101': summary(ci.charge_displacement(combined, axis, -1.0, 1000.0, 101)),
    }
    from_list = ci.chgcar_algebra_from_list(ci.extract_chgcar_data(paths), operations, axis)
    data['from_list'] = from_list.data['total']
    total = ci.chgcar_algebra_from_list(ci.extract_chgcar_data(paths), ['+', '+'])
    data['sum'] = total.data['total']
    data['bad_operation'] = outcome(ci.chgcar_algebra_from_list,
                                    ci.extract_chgcar_data(paths), ['*', '-'])
    Path('not_a_chgcar').write_text('this is not a CHGCAR\n')
    data['not_a_chgcar'] = outcome(ci.extract_chgcar_data, 'not_a_chgcar')
    data['not_a_chgcar_in_list'] = outcome(ci.extract_chgcar_data, [paths[0], 'not_a_chgcar'])
    data['wrong_type'] = outcome(ci.extract_chgcar_data, 3)
    return data


def analyzer_state(analyzer):
    return {'params': analyzer.params, 'cell': analyzer.cell, 'filetype': analyzer.filetype,
            'grid': analyzer.charge_grid}


def case_qecharge_analyzer(tc, cell_name):
    """QEChargeAnalyzer on the pp.x filplot of the interface: reading, cell
    and planar averages; then the variants of the file header."""
    an = tc['physics.electronics.charge.analyzer']
    path = write_filplots(cell_name)[0]
    analyzer = an.QEChargeAnalyzer(path, unpack=False)
    data = {'read': analyzer_state(analyzer)}
    analyzer.average_charge('xyz')
    analyzer.average_charge('xy', shift=0, rescale=200, get_inter=True)
    data['charge'] = analyzer.charge
    cell = CHARGE_CELLS[cell_name]

    def variant(name, **kwargs):
        Path(name).write_text(filplot_of(cell, 'interface', **kwargs))
        result = outcome(lambda: analyzer_state(an.QEChargeAnalyzer(name, unpack=False)))
        if 'result' in result:
            result['result'] = {'params': result['result']['params'],
                                'grid_shape': list(np.shape(result['result']['grid']['ndarray']))}
        return result

    data['titled'] = variant('titled', title='charge density of the interface')
    data['padded_x'] = variant('padded_x', nrx=(1, 0, 0))
    data['padded_z'] = variant('padded_z', nrx=(0, 0, 1))
    return data


def case_qecharge_ibrav(tc):
    """QEChargeAnalyzer on filplots with ibrav != 0 (cell from celldm): fcc
    (ibrav 2, the layout of the pp.x example of silicon) and simple cubic."""
    import charge_files
    an = tc['physics.electronics.charge.analyzer']
    n, alat = 8, 10.2
    k, j, i = np.meshgrid(*[np.arange(n)] * 3, indexing='ij')
    modulation = 1 + 0.3 * np.cos(2 * math.pi * i / n) * np.cos(2 * math.pi * j / n) \
        * np.cos(2 * math.pi * k / n)
    data = {}
    for ibrav, volume in ((2, alat ** 3 / 4), (1, alat ** 3)):
        name = f'si_charge_{ibrav}'
        Path(name).write_text(charge_files.filplot_text(
            values=8.0 / volume * modulation, celldm=[alat], species=[('Si', 4.0)],
            atoms=[((0.0, 0.0, 0.0), 1), ((0.25, 0.25, 0.25), 1)], ibrav=ibrav,
            gcutm=72.0 / (2 * math.pi / alat) ** 2, dual=4.0, ecut=18.0))

        def read_and_average():
            analyzer = an.QEChargeAnalyzer(name, unpack=False)
            state = analyzer_state(analyzer)
            analyzer.average_charge('xyz')
            analyzer.average_charge('xy', shift=0, rescale=100, get_inter=True)
            return {'read': state, 'charge': analyzer.charge}

        data[str(ibrav)] = outcome(read_and_average)
    return data


def case_qecharge_processor(tc, cell_name):
    """QEChargeProcessor on the three filplots: planar averages and charge
    displacement; then the same difference with QEChargeAnalyzer objects."""
    an = tc['physics.electronics.charge.analyzer']
    paths = write_filplots(cell_name)
    processor = an.QEChargeProcessor(paths, unpack=False)
    processor.average_charge(axis='xy', rescale=200, get_inter=True)
    processor.displace_charge(axis='xy', rescale=200, save_charge=False)
    data = {'charge': processor.charge, 'charge_arrays': processor.charge_arrays,
            'charge_diff': processor.charge_diff}
    analyzers = [an.QEChargeAnalyzer(p, unpack=False) for p in paths]
    difference = analyzers[0] - analyzers[1] - analyzers[2]
    difference.average_charge('xy', shift=0, rescale=200, get_inter=True)
    data['difference'] = {'grid': difference.charge_grid, 'charge': difference.charge}
    return data


CASES = {}
for _pair in PAIRS:
    CASES[f'matcher_{_pair}'] = (case_matcher, {'pair': _pair})
for _pair in ['pt111_ag111', 'cu111_al111', 'cu111_cu111']:
    CASES[f'highsym_{_pair}'] = (case_highsym, {'pair': _pair})
CASES['pes_inputs_cu111_al111'] = (case_pes_inputs, {'pair': 'cu111_al111', 'calc_type': 'interface_z_relax'})
CASES['pes_inputs_cu111_cu111'] = (case_pes_inputs, {'pair': 'cu111_cu111', 'calc_type': 'interface_z_relax'})
CASES['pes_scf_inputs_cu111_cu111'] = (case_pes_inputs, {'pair': 'cu111_cu111', 'calc_type': 'slab_from_scratch'})
CASES['interface_shifts_cu111_al111'] = (case_interface_shifts, {'pair': 'cu111_al111'})
CASES['ppes_inputs_cu111_al111'] = (case_ppes_inputs, {'pair': 'cu111_al111'})
CASES['chargedisp_inputs_cu111_al111'] = (case_chargedisp_inputs, {'pair': 'cu111_al111'})
# Not on Cu/Al: its 756 unfolded points, replicated 12 x 12 by get_pes,
# make the RBF interpolation too large for a test.
CASES['get_pes_cu111_cu111'] = (case_get_pes, {'pair': 'cu111_cu111'})
CASES['shear_cu111_cu111'] = (case_shear_from_hs, {'pair': 'cu111_cu111'})
for _name in PES_GRIDS:
    CASES[f'pes_{_name}'] = (case_pes, {'name': _name})
CASES['shear_ti3c2o2'] = (case_shear, {'name': 'ti3c2o2'})
# Phosphorene with the parameters of test_mep.py: with those of the Firetask
# its 100 points replicated 5 x 5 give 10000 RBF centres, and the MEP takes hours.
CASES['shear_phosphorene'] = (case_shear, {
    'name': 'phosphorene', 'pes_params': PES_GRIDS['phosphorene'][3],
    'mep_kwargs': {'nstepmax': 1000, 'tol': 1e-6}})
CASES['uber_ppes_data'] = (case_uber, {})
CASES['uber0_cu111'] = (case_uber0, {})
for _name in CUSTOM_SLABS:
    CASES[f'custom_slab_{_name}'] = (case_custom_slab, {'name': _name})
for _name in MATCH_STEPS:
    CASES[f'match_step_{_name}'] = (case_match_step, {'name': _name})
for _name in PES_STEPS:
    CASES[f'pes_step_{_name}'] = (case_pes_step, {'name': _name})
for _name in PLOTPES_STEPS:
    CASES[f'plotpes_step_{_name}'] = (case_plotpes_step, {'name': _name})
for _name in ADHESION_STEPS:
    CASES[f'adhesion_step_{_name}'] = (case_adhesion_step, {'name': _name})
for _cell in CHARGE_CELLS:
    for _case, _func in (('chgcar_average', case_chgcar_average),
                         ('chgcar_displacement', case_chgcar_displacement),
                         ('qecharge_analyzer', case_qecharge_analyzer),
                         ('qecharge_processor', case_qecharge_processor)):
        CASES[f'{_case}_{_cell}'] = (_func, {'cell_name': _cell})
CASES['qecharge_ibrav'] = (case_qecharge_ibrav, {})


# =============================================================================
# Running and storing
# =============================================================================

def run(name, root, workdir=None):
    """JSON data of case <name> run with the modules under <root>, in workdir
    (a temporary folder by default), with stdout discarded."""
    func, kwargs = CASES[name]
    if any(os.environ.get(k) != v for k, v in BLAS_THREADS.items()):
        raise RuntimeError(f'golden cases run with {BLAS_THREADS}')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(workdir or tmp)
        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'FigureCanvasAgg is non-interactive')
                data = to_data(func(TribChem(root), **kwargs))
        finally:
            os.chdir(cwd)
            plt.close('all')
    return json.loads(json.dumps(data))


ENVIRONMENT = GOLDEN / 'environment.json'
RECORDED = ['numpy', 'scipy', 'pymatgen', 'ase', 'matplotlib', 'monty', 'tribchem']


def golden_path(name):
    return GOLDEN / f'{name}.json'


def golden_names():
    return sorted(p.stem for p in GOLDEN.glob('*.json') if p != ENVIRONMENT)


def read_golden(name):
    return json.loads(golden_path(name).read_text())


def write_golden(name, data):
    golden_path(name).write_text(json.dumps(data, indent=1) + '\n')


def environment():
    from importlib import metadata
    return {'python': sys.version.split()[0],
            **{name: metadata.version(name) for name in RECORDED}}


def main(argv):
    names = argv[1:] or list(CASES)
    unknown = [n for n in names if n not in CASES]
    if unknown:
        print(f'unknown cases: {unknown}')
        return 1
    importlib.import_module(HISTORIC)  # golden values come from the historic environment only
    GOLDEN.mkdir(exist_ok=True)
    ENVIRONMENT.write_text(json.dumps(environment(), indent=1) + '\n')
    for name in names:
        start = time.perf_counter()
        write_golden(name, run(name, HISTORIC))
        print(f'written  {golden_path(name).relative_to(ROOT)}  '
              f'({time.perf_counter() - start:.1f} s)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
