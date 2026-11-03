# Author: Vito Foderà
# Created: 2026-11-02

"""Classes of the pw.x namelist keys, for the fingerprint and the energy model.

Every key of ``&CONTROL``, ``&SYSTEM``, ``&ELECTRONS`` and ``&IONS`` documented
for pw.x 7.2 (``PW/Doc/INPUT_PW.def``, 194 keys) is in one class:

- ``'model'`` (``MODEL``): the key changes the energy, so two energies are
  compared only if they have the same value. These keys enter the energy
  model and the fingerprint. Among them, besides the cutoffs, occupations,
  spin, functional, vdW, fields and symmetry: ``angle1`` and ``angle2``,
  which with the ``starting_magnetization`` of the species set the initial
  magnetization and the magnetic constraints (all the starting
  magnetizations zero make a noncollinear calculation non-magnetic);
- ``'input'`` (``INPUT``): the key changes what is computed or kept, not the
  energy model: kind of calculation, convergence thresholds, output, start,
  every key of ``&IONS``. These keys enter the fingerprint only;
- ``'attempt'`` (``ATTEMPT``): the key changes how a run gets to the result
  (steps, mixing, diagonalization); the relaunches may change it. Out of the
  fingerprint: the values of each run are in its input file;
- ``'resource'`` (``RESOURCE``): time limit, paths and names. Out of the
  fingerprint; the pseudopotentials are identified by their checksums, not by
  ``pseudo_dir``;
- ``'generated'`` (``GENERATED``): written from the structure and the species
  (``ibrav = 0``, ``nat``, ``ntyp``, ``starting_magnetization(i)``) or not
  allowed by the settings (``celldm``, ``a``, ...).

A key outside these lists, e.g. one that pw.x reads but does not document
(``B_field``, ``ts_vdw``, ...), is in ``'model'``: two calculations with the
same value of it can be compared, and none is compared with one without it.
"""

from triboworkflow.core.settings.schema import key_name

MODEL = frozenset('''
    tefield dipfield lelfield gate twochem lfcp trism
    ecutwfc ecutrho ecutfock nr1 nr2 nr3 nr1s nr2s nr3s
    occupations smearing degauss one_atom_occupations
    nspin noncolin lspinorb tot_magnetization constrained_magnetization fixed_magnetization
    lambda angle1 angle2
    tot_charge
    input_dft exx_fraction screening_parameter exxdiv_treatment x_gamma_extrapolation ecutvcut
    nqx1 nqx2 nqx3 localization_thr ecfixed qcutz q2sigma
    vdw_corr london london_s6 london_c6 london_rvdw london_rcut dftd3_version dftd3_threebody
    ts_vdw_isolated xdm xdm_a1 xdm_a2
    nosym nosym_evc noinv no_t_rev force_symmorphic use_all_frac
    assume_isolated esm_bc esm_w esm_efield esm_nfit edir emaxpos eopreg eamp zgate relaxz
    block block_1 block_2 block_height nextffield lgcscf gcscf_mu
    nbnd_cond degauss_cond nelec_cond
    sic_gamma pol_type sic_energy sci_vb sci_cb dmft
    hubbard_occ hubbard_alpha hubbard_beta
    efield efield_cart efield_phase tqr real_space
'''.split())

INPUT = frozenset('''
    calculation restart_mode etot_conv_thr forc_conv_thr
    tstress tprnfor verbosity iprint wf_collect disk_io
    dt nberrycyc lorbm lberry gdir nppstr
    nbnd starting_charge starting_spin_angle lforcet ace report ensemble_energies dmft_prefix
    gcscf_conv_thr ts_vdw_econv_thr
    starting_ns_eigenvalue space_group uniqueb origin_choice rhombohedral
    conv_thr scf_must_converge adaptive_thr conv_thr_init conv_thr_multi startingwfc startingpot
    ion_positions ion_velocities ion_dynamics pot_extrapolation wfc_extrapolation
    remove_rigid_rot ion_temperature tempw tolp delta_t nraise refold_pos upscale bfgs_ndim
    trust_radius_max trust_radius_min trust_radius_ini w_1 w_2
    fire_alpha_init fire_falpha fire_nmin fire_f_inc fire_f_dec fire_dtmax
'''.split())

ATTEMPT = frozenset('''
    nstep gcscf_beta
    electron_maxstep exx_maxstep mixing_mode mixing_beta mixing_ndim mixing_fixed_ns
    diagonalization diago_thr_init diago_cg_maxiter diago_ppcg_maxiter diago_david_ndim
    diago_rmm_ndim diago_rmm_conv diago_gs_nblock diago_full_acc
'''.split())

RESOURCE = frozenset('max_seconds outdir wfcdir prefix title pseudo_dir lkpoint_dir'.split())

GENERATED = frozenset('ibrav celldm a b c cosab cosac cosbc nat ntyp starting_magnetization'.split())

CLASSES = {'model': MODEL, 'input': INPUT, 'attempt': ATTEMPT, 'resource': RESOURCE,
           'generated': GENERATED}


def key_class(key):
    """Class of a namelist key.

    Parameters
    ----------
    key : str
        Key as written, with or without index, in any case (e.g.
        ``"London_C6(1)"``).

    Returns
    -------
    str
        ``'model'``, ``'input'``, ``'attempt'``, ``'resource'`` or
        ``'generated'``; ``'model'`` for a key in no list.

    Examples
    --------
    >>> [key_class(k) for k in ('ecutwfc', 'conv_thr', 'mixing_beta', 'prefix', 'B_field(3)')]
    ['model', 'input', 'attempt', 'resource', 'model']
    """
    name = key_name(key)
    for cls, names in CLASSES.items():
        if name in names:
            return cls
    return 'model'


def normal_key(key):
    """A key in lower case and without blanks, e.g. ``"London_C6( 1 )"`` → ``"london_c6(1)"``."""
    return ''.join(key.split()).lower()
