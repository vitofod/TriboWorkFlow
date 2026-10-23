# Environments

Two locked environments, both built with [uv](https://docs.astral.sh/uv/)
in WSL2 (Ubuntu 22.04, x86_64):

| | historic | development |
|---|---|---|
| Purpose | runs the original TribChem code and produces the golden values | runs TriboWorkFlow and its tests |
| venv | `~/.venvs/tribchem-storico` | `~/.venvs/tribchem` |
| Python | 3.12.4 | 3.12 (3.12.13 from the system `python3.12`) |
| Lock | `historic/requirements.lock` (105 packages) | `dev/requirements.lock` (43 packages) |
| Build | `historic/create_env.sh <TribChem source root>` | `dev/create_env.sh` |
| Tests | `python -m pytest tests/historic` | `python -m pytest` |

Both scripts and `compile_locks.sh` take the uv executable from `UV`
(default: `uv` on `PATH`). uv 0.12.23 was used, installed in its own venv:

```bash
python3.12 -m venv ~/.venvs/uv
~/.venvs/uv/bin/pip install uv==0.12.23
export UV=~/.venvs/uv/bin/uv
```

## Historic environment

TribChem's `requirements.txt` (branch `devel-align_elisa_w_devel`) gives
only upper bounds. They are taken as exact versions:

| Package | requirements.txt | Locked | Upload date of the bound |
|---|---|---|---|
| python | ==3.12.4 | 3.12.4 | — |
| numpy | <=1.26.4 | 1.26.4 | 2024-02-05 |
| scipy | <=1.14.0 | **1.13.1** | 2024-06-24 |
| pymatgen | <=2024.6.10 | 2024.6.10 | 2024-06-11 |
| ase | <=3.23.0 | 3.23.0 | 2024-05-31 |
| atomate | <=1.1.0 | 1.1.0 | 2023-08-25 |
| fireworks | <=2.0.3 | 2.0.3 | 2022-04-26 |
| mp-api | <=0.41.2 | 0.41.2 | 2024-03-15 |
| matplotlib | <=3.9.1 | 3.9.1 | 2024-07-04 |
| pandas | <=2.2.2 | 2.2.2 | 2024-04-10 |
| ruamel.yaml | <=0.17.40 | 0.17.40 | 2023-10-20 |
| seaborn | <=0.13.2 | 0.13.2 | 2024-01-25 |
| setuptools | ==70.3.0 | 70.3.0 | 2024-07-09 |
| pip | ==24.0 | 24.0 | 2024-02-03 |

All the bounds are compatible with a single snapshot taken between
2024-07-09 and 2024-07-16: setuptools 71.0.0 and pymatgen 2024.7.18 were
uploaded on 2024-07-17 and 2024-07-18. The transitive dependencies are
therefore resolved with `--exclude-newer 2024-07-17T00:00:00Z`.

Exceptions and additions:

- **SciPy 1.13.1 instead of 1.14.0.** SciPy 1.14.0 removed
  `scipy.integrate.simps`, which
  `tribchem/physics/electronics/charge/analyzer.py` calls with
  `even='avg'`. With 1.13.1 the call works and emits two DeprecationWarnings.
  If the bounds are the versions actually installed, that environment had
  SciPy 1.14.0 and the charge analyzer could not run in it. The golden
  values (none of them involves the charge analyzer) are identical with
  SciPy 1.14.0: checked in a temporary copy of this environment with only
  SciPy changed.
- **`importlib_metadata`**: imported by TribChem's `setup.py`, missing from
  `requirements.txt`.
- **`wheel`**: TribChem's installation guide builds the environment with
  micromamba from conda-forge, whose pip 24.0 depends on setuptools and
  wheel. With wheel present, `pip install -e .` runs the legacy
  `setup.py develop`, which puts the project root on `sys.path`. Without it,
  pip uses PEP 660 and, since `setup.py` declares `py_modules=[]` and no
  packages, `tribchem` is not importable.
- **`pytest`**: test runner for this environment and for the golden tests.

`create_env.sh` also:

1. copies the TribChem sources to `~/tribchem-storico/src/` (`pip install -e`
   writes `tribchem.egg-info` into the project folder, so the original
   folder is never installed) and checks the copy against
   `historic/tribchem-source.sha256`, the SHA-256 of the 985 files of the
   source tree used for the golden values;
2. applies `historic/atomate-1.1.0-vdw-kernel.patch` to atomate in
   site-packages: the change described in TribChem's installation guide
   (`OptimizeFW` and `StaticFW` copy the VASP vdW kernel). It does not touch
   the physics modules.

### What the tests check

`tests/historic/` (skipped where TribChem is not importable): Python
version, installed packages equal to the lock, sources equal to the
manifest, atomate change, `simps(..., even='avg')`, `distutils` (used by
`tribchem/highput/database/filetools.py`, available through setuptools),
`ZSLGenerator` and `AdsorbateSiteFinder`, and the import of every module of
`tribchem.physics` outside the `tests` and `scripts` folders. Six modules do
not import, and the tests expect them to fail:

| Module | Error |
|---|---|
| `tribology.simple_pes` | TabError: mixed tabs and spaces |
| `ml.bayes` | imports `tribchem.physics.base.qeio`, which does not exist |
| `ml.deepmd.deepmd_io` | needs `dpdata`, not in `requirements.txt` |
| `dft.lammps` | script: reads `tests/IO_files/md_ph_iron.pwi` on import |
| `electronics.charge.vasp_pymatgen` | script: reads `tests/Charges/CHGCAR` on import |
| `electronics.dos.plotdos` | script: reads `DOS.pdos_tot` on import |

Three of TribChem's own test scripts under `tribchem/physics` run in this
environment and are executed by the tests: `tribology/tests/test_matching_interfaces.py.py`
(`InterfaceMatcher` on Pt(111)/Ag(111)), `tribology/tests/test_pesmodel.py`
and `base/tests/test_replicate.py`, whose five output files are
byte-identical to those stored in the TribChem repository. The other scripts
there need input files that are not in the source tree (`IO_files/`,
`XYZTest/`, `Charges/`), command-line arguments (`tribology/tests/test_pes.py`),
or APIs that no longer exist (`ChemAds`, `tribchem.physics.base.plotting`,
`tribology.innovative_mep`, `tribchem.physics.base.qeio`, `pymatgen.io.cube`),
or call `PES.rbf` with fewer arguments than it takes (`tribology/tests/test_mep.py`).

### Known limits

- The exact environment of the reference results is not known yet: this is
  a reconstruction from `requirements.txt`.
- Packages come from PyPI wheels, not from conda-forge as in the
  installation guide: compiled libraries (BLAS included) differ, so results
  may differ at round-off level.
- Python 3.12.4 is the python-build-standalone build installed by uv
  (Clang), not the conda-forge one.
- matplotlib 3.9.1 is yanked on PyPI because of its Windows wheels; the
  manylinux wheels used here are not affected.

## Golden values

`tests/golden_cases.py` runs the TribChem physics modules with the
arguments the Firetasks pass (interface matching, high-symmetry points,
the `generate_*_inputs`, PES and RBF, MEP and shear strength, UBER fits)
and stores the results of the historic environment in
`tests/golden/<case>.json`. They are written only on purpose, in the
historic environment:

```bash
~/.venvs/tribchem-storico/bin/python tests/golden_cases.py [<case> ...]
```

`tests/historic/test_golden.py` checks that the historic environment
reproduces them, `tests/test_golden.py` that the vendored modules give the
same results in the development environment. Numbers are compared exactly.

- **One OpenBLAS thread.** PES and RBF values change at round-off level
  with the number of OpenBLAS threads (1, 8 and 12 give different results),
  so `tests/conftest.py` and `golden_cases.py` set
  `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1` before numpy is imported.
- **Seed.** The MEP starting string gets unseeded `np.random` noise in
  TribChem; the cases call `np.random.seed(0)` before every MEP.
- **Tags.** The uuid4 of the calculation tags is replaced by `<uuid4>`.
- **Inputs.** The data files are copies of TribChem files in
  `tests/data/tribchem/`, at their path inside the `tribchem` package and
  checked against `historic/tribchem-source.sha256`.
- `tests/golden/environment.json` records the versions used.

## Development environment

`dev/requirements.lock` is compiled with the historic lock as constraints,
ASE excluded (`dev/historic-constraints.txt`, generated by
`compile_locks.sh`): every package in common has the same version, and ASE is
3.26.0, the version required by Xsorb. Packages that are not in the
historic lock would also be resolved with the 2024-07-17 cutoff (ASE: its
own upload date); at present there are none. `tests/test_dev_environment.py`
checks that the venv equals the lock and that the two locks differ only in
ASE.

`dev/create_env.sh` synchronises the venv with the lock (packages not in
the lock are removed) and installs TriboWorkFlow in editable mode.

## Updating the locks

Edit `historic/requirements.in` or `dev/requirements.in`, then run
`bash envs/compile_locks.sh` and rebuild the environments. The golden values
depend on the historic environment: any change to it must be justified and
the golden tests rerun.
