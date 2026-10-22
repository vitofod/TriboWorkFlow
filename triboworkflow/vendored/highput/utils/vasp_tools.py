# --- TribChem code vendored by TriboWorkFlow ---------------------------------
# Unchanged except for the imports. Do not edit: tests/vendoring.py
# compares this file with the original.
# Source: tribchem/highput/utils/vasp_tools.py
# Version: tribchem_devel-align_elisa_w_devel
# Source SHA-256: f7f86af8783b34ea283891fd622b084f228a1f600225425fcac5fcb673d02b8b
# Copied: 2026-10-22
# Licence: CC BY-SA 4.0, see NOTICE
# Kind: whole
# Imports: tribchem.* -> triboworkflow.vendored.*
# --- end of the TriboWorkFlow header -----------------------------------------
import os
import subprocess
import math

import numpy as np
import pymatgen.core.structure, pymatgen.core.surface
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.sets import (
    MPStaticSet, 
    MPRelaxSet, 
    MPScanStaticSet, 
    MPScanRelaxSet
    )

from triboworkflow.vendored.highput.utils.tasktools import read_default_params, read_json
from triboworkflow.vendored.highput.utils.io_files import remove_matching_files


currentdir = os.path.dirname(__file__)


class MeshFromDensity:
    """
    Class to find classic Monkhorst-Pack meshes which may
    or my not be Gamma-Centred from a given k-point density.
    Provides also capabilites to check if meshes generated from similar
    densities are equivalent. Meshes for slabs are adapted according to
    the lower symmetry conditions.
    
    """

    def __init__(self,
                 structure,
                 target_density,
                 compare_density=10.0,
                 material_type='Auto',
                 min_vac=6.0,
                 force_gamma=False):
        """Initializes the class and sets internal variables.

        Parameters
        ----------
        structure : pymatgen.core.structure.Structure
            The pymatgen representation of a structure
        target_density : float
            Desired minimal density of kpoints along each reciprocal lattice
            vector in 1/Angstrom.
        compare_density : float
            Density for which the mesh of the target density is to be compared
            to. in 1/Angstom. The default is 10.0
        material_type : bool/str
            If the passed structure is to be considered to be a slab. If "True"
            it is considered to be a slab, if "False" it is considered to be
            bulk, if "Auto", it is attempted to find out. The default is
            "Auto".
        min_vac : float
            Minimum vacuum distance for the slab detection if is_slab is set to
            "Auto". The default is 6.0
        force_gamma : bool
            If Gamma-centred meshes are to be forces (not applicable for
            generalized meshes!). The default is False.

        """

        # Initialize the attributes        
        self.structure = structure.copy()
        self.dens = target_density
        self.compare_dens = compare_density
        self.min_vac = min_vac
        self.force_gamma = force_gamma
        self.klm  = structure.lattice.reciprocal_lattice.abc
        
        # Assign or infer the material type
        if material_type in ['slab', 'Slab', 'interface', 'Interface']:
            self.material_type = 'slab'
        elif material_type in ['atom', 'atoms', 'Atom', 'Atoms', 'molecule',
                               'molecules', 'Molecule', 'Molecules']:
            self.material_type = 'molecule'
        else:
            self._infer_material_type()
    
    def __make_mesh(self, density):
        """Return the subdivisions along each lattice vector.
        
        Consider also if the structure is a slab.

        Parameters
        ----------
        density : float
            Desired minimal density of kpoints along each reciprocal lattice
            vector in 1/Angstrom.

        Returns
        -------
        k1 : int
            Kpoint devisions along b1
        k2 : int
            Kpoint devisions along b2
        k3 : int
            Kpoint devisions along b3

        """
        
        k, l, m = self.klm
        k1 = round(k*density)  # math.round(k*density)
        k2 = round(l*density)  # math.ceil(l*density)
        k3 = round(m*density)  # math.ceil(m*density)
        
        if self.material_type == 'slab':
            k3 = 1
        elif self.material_type == 'molecule':
            k1, k2, k3 = (1, 1, 1)

        return (k1, k2, k3)
       
    def _infer_material_type(self):
        """
        Figures out the type of the passed structure, which can be either
        a slab, an interface, an atom, or a molecule. This method is called
        by the constructore, for an automatic detection of the material type.
        Be careful, the automatic detection might fail for structure that are 
        set up in a none standard way, e.g. for slabs.

        """

        self.material_type = None
        
        # Infer for a slab or an interface
        z_axis = self.structure.lattice.c
        z_coords = []
        for s in self.structure.sites:
            z_coords.append(s.coords[2])

        thickness = max(z_coords) - min(z_coords)
        if z_axis - thickness >= self.min_vac:
            self.material_type = 'slab'
    
    def get_kpoints(self):
        """Return a Kpoint object with the desired density of kpoints.

        Returns
        -------
        kpoints : pymatgen.io.vasp.Kpoints
            Monkhorst-Pack or Gamma centered mesh.

        """
        
        mesh = self.__make_mesh(self.dens)
        is_hexagonal = self.structure.lattice.is_hexagonal()
        #has_odd = any(i % 2 == 1 for i in mesh)
        
        if is_hexagonal or self.force_gamma:
            kpoints = Kpoints.gamma_automatic(kpts=mesh)
        else:
            kpoints = Kpoints.monkhorst_automatic(kpts=mesh)

        return kpoints

    def are_meshes_the_same(self):
        """Compares conventional Monkhorst-Pack meshes and Gamma centered meshes.
        
        To test if a different target density actually provides a different
        mesh than a reference density.

        Returns
        -------
        bool
            True if meshes are the same, False otherwise.

        """

        mesh_1 = self.__make_mesh(self.dens)
        mesh_2 = self.__make_mesh(self.compare_dens)
        if mesh_1 == mesh_2:
            return True
        else:
            return False

def get_emin(potcar):
    """
    Return the minimal recommended  energy cutoff for a given Potcar object.
    
    Unfortunately I don not think that pymatgen Potcars can access the data
    for the recommended cutoffs directly, so I write out a file first and
    then scan for the ENMAX lines. The largest EMIN value found is returned
    in the end.

    Parameters
    ----------
    potcar : pymatgen.io.vasp.inputs.Potcar
        The pymatgen representation of a VASP POTCAR file.

    Returns
    -------
    float
        The largest EMIN value of all species present in the potcar.

    """

    potcar.write_file('temp_potcar')
    with open('temp_potcar', 'r') as pot:
        emin = []
        for l in pot:
            if l.strip().startswith('ENMAX'):
                emin.append(float(l.split()[-2]))
    os.remove('temp_potcar')
    return max(emin)
 
def get_generalized_kmesh(structure, k_dist, rmsym=False, vasp6=True):
    """Get a generalized Monkhorst Pack mesh for a given structure.
    
    Prepares the necessary files (POSCAR, PRECALC, and, if the structure has
    a 'magmom' site property also INCAR) for the K-Point Grid Generator of the
    Mueller group at John Hopkins http://muellergroup.jhu.edu/K-Points.html
    Runs the getKPoints script and reads the KPOINT file produced into a
    pymatgen Kpoints object.
    
    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        Structure for which the kpoints grid is to be generated.
    k_dist : float
        The minimum allowed distance between lattice points on the real-space
        superlattice. This determines the density of the k-point grid. A larger
        value will result in denser grids.

    Returns
    -------
    KPTS : pymatgen.io.vasp.inputs.Kpoints
        Pymatgen Kpoints object representing a generalized MP mesh.

    """

    precalc = ['MINDISTANCE = ' + str(k_dist),
               'MINTOTALKPOINTS = 4',
               'GAPDISTANCE = 6 ',
               'MONOCLINIC_SEARCH_DEPTH = 2500',
               'TRICLINIC_SEARCH_DEPTH = 1500']
    if vasp6:
        precalc.append('WRITE_LATTICE_VECTORS = True')
                
    if rmsym in ['STRUCTURAL', 'TIME_REVERSAL', 'ALL']:
        precalc.append('REMOVE_SYMMETRY = ' + rmsym)
    with open('PRECALC', 'w') as out:
        for line in precalc:
            out.write(line+'\n')

    magmom_list = structure.site_properties.get('magmom')
    if magmom_list:
        with open('INCAR', 'w') as out:
            out.write('ISPIN = 2')
            out.write('MAGMOM = ' + ' '.join(str(m) for m in magmom_list))

    structure.to(fmt='poscar', filename='POSCAR')
    get_kpoints_file = subprocess.Popen('getKPoints')
    get_kpoints_file.communicate()
    kpts = Kpoints().from_file('KPOINTS')
    remove_matching_files(['KPOINTS*', 'POSCAR*', 'INCAR', 'PRECALC'])

    return kpts

class VaspInputSet:
    
    static_types = ['bulk_from_scratch', 'bulk_follow_up', 'bulk_nscf',
                    'slab_from_scratch', 'slab_follow_up', 'slab_nscf']

    relax_types = ['bulk_full_relax', 'bulk_vol_relax', 'bulk_pos_relax',
                   'bulk_shape_relax', 'slab_shape_relax', 'bulk_pos_shape_relax',
                   'slab_pos_relax', 'interface_shape_relax', 'interface_pos_relax',
                   'interface_z_relax']
    
    atomic_types = ['atom']
    
    allowed_types = static_types + relax_types + atomic_types
    
    #ELISA: I added from 'dftd2' to 'df2' to the list of vdw_types. Tested for 'dftd2' and worked.
    vdw_types = ['rVV10', 'optB86b', 'dftd2', 'dftd3', 'dftd3-bj', 'ts', 'ts-hirshfeld', 'mbd@rsc', 'ddsc', 'df', 'optpbe', 'optb88', 'df2']
    
    def __init__(self, structure, comp_params, calc_type, encut_def=None, 
                 kdens_def=8, vis_def='vis.json'):
        """
        Initialize a VaspInputSet object to create a set of parameters to be
        used to start a VASP simulation.

        Parameters
        ----------
        structure : pymatgen.core.structure.Structure or pymatgen.surface.slab.Slab
            Structure to be treated.
    
        comp_params : dict
            Computational parameters dictionary containing information used to
            customize the input parameters to be used by VASP.

        calc_type : str
            The tyope ype of calculation to be run with VASP.

        encut_def : float or None, optional
            Default encut to be used by VASP, if not present in comp_params and
            if nothing is provided, ptimal cutoff values for the various species 
            will be taken from POTCAR files. The default is None.

        kdens_def : float, optional
            Specifies the default kpoint density if no kdens or kspacing key
            is found in the comp_params dictionary. The default is 8.

        vis_def : str, optional
            Path to the json file containing the default Vasp Input Sets to be
            chosen for the customization. The default is 'vis.json'.

        """

        # Read the default values for 
        self.structure, self.calc_type, self.comp_params = self.__read_init(
            structure, calc_type, comp_params)
        
        # Set default variables
        self.encut_default = encut_def
        self.kdens_default = kdens_def
        self.vis_default = read_json(currentdir + '/' + vis_def)

    def __read_init(self, structure, calc_type, comp_params):
        """ Read and check the various input arguments for __init__.
        """
        
        # Check the type of the structure to be pymatgen-coherent
        if not isinstance(structure, (pymatgen.core.structure.Structure, pymatgen.core.surface.Slab)):
            raise ValueError('Wrong value for structure, a pymatgen object '
                             'is required')
        
        # Check the calc_type to see if it makes sense
        if not (calc_type in self.allowed_types):
            raise ValueError('Wrong value for calc_type, it must be one among '
                            '{}'.format(self.allowed_types))

        # Check the comp_params and set the default ones
        comp_params = read_default_params(currentdir + '/../firetasks/defaults.json', 
                                          'comp_params', comp_params,
                                          allow_unknown=True)

        return structure, calc_type, comp_params
    
    def get_vasp_settings(self):
        """ Make and return a custom vasp input settings object.
        """
        
        # Set the common uis 
        uis = self.vis_default['common']


        # Appply spin polarization and/or VdW corrections
        # ==================================
        
        if self.comp_params['use_spin']:
            uis['ISPIN'] = 2  # Spin polarized calculation

        # Set the van der Waals functional

        # Convert 'None' string to actual None
        if self.comp_params['vdw'] == 'None':
            vdw = None
            
        elif self.comp_params['vdw'] is not None:
            vdw = self.comp_params['vdw']
            
            # WARNING! SCAN simulations now can be used only with rVV10
            if self.comp_params['functional'] == 'SCAN':
                vdw = 'rVV10'
            
            # Check consistency for vdw type
            print("vdw in vasp_tools.py", type(self.comp_params['vdw']))
            #if not vdw in self.vdw_types:
            if vdw is not None and vdw not in self.vdw_types:
                raise ValueError('Error in defining the VdW correction, allowed '
                                 'in: {}'.format(self.vdw_types))
        else:
            vdw = None

        # Appply other common conditions
        # ==================================

        if any(np.unique(self.structure.lattice.matrix.flatten()) > 50.0):
            uis['AMIN'] = 0.05  # Adjust mixing for very large boxes

        # Handle specific SCAN case, these are very specific parameters
        # ==================================

        if self.comp_params['functional'] == 'SCAN':
            uis['ISMEAR'] = 0
            uis['SIGMA'] = 0.1
            uis['METAGGA'] = 'SCAN'
            uis['ALGO'] = 'All'
            uis['LELF'] = False  #otherwise KPAR >1 crashes

        # Define the specific options for each calculation
        # ==================================
        
        # Understand the type of calculation and update the parameters
        if self.calc_type in self.static_types:
            uis = self.__get_static(uis)
        elif self.calc_type in self.relax_types:
            uis = self.__get_relax(uis)
        elif self.calc_type in self.atomic_types:
            uis = self.__get_atomic(uis)
        else:
            raise ValueError('Something weird is happening, calc_type attribute '
                             'is not an allowed type')

        # Handle the ISMEAR based on the type of material under study
        # ==================================
        
        if self.comp_params['is_metal'] is None:
            uis['SIGMA'] = 0.05
            uis['ISMEAR'] = 0

        elif self.comp_params['is_metal']:
            uis['SIGMA'] = 0.1
            uis['ISMEAR'] = 1

        else:
            uis['SIGMA'] = 0.05
            uis['ISMEAR'] = -5
            if any(np.unique(self.structure.lattice.matrix.flatten()) > 50.0):
                uis['SIGMA'] = 0.03
                uis['ISMEAR'] = 0

        # Define the kinetic energy cutoff
        # ==================================
        
        # Set it if provided within comp_params or as input argument to the
        # class constructor. If nothing is provided VASP will use it from POTCAR
        if self.comp_params.get('encut', None) is not None:
            uis['ENCUT'] = self.comp_params['encut']
        elif self.encut_default is not None:
            uis['ENCUT'] = self.encut_default

        # Define the kpoints grid
        # ==================================
        
        # Understand if the structure is a slab or an interface
        if self.calc_type.startswith('slab_') or self.calc_type.startswith('interface_'):
            material_type = 'slab'
        elif self.calc_type in self.atomic_types:
            material_type = 'molecule'
        else:
            material_type = 'auto'

        if 'kspacing' in self.comp_params.keys():  # Define kgrid from spacing
            uis['KSPACING'] = self.comp_params['kspacing']
            uis['KGAMMA'] = True
            kpoints = None
        else:
            if 'kdens' in self.comp_params.keys():  # Define kgrid from density
                kdens = self.comp_params['kdens']
            else:
                kdens = self.kdens_default
            KPTS = MeshFromDensity(self.structure,
                                   kdens,
                                   material_type=material_type,
                                   force_gamma=True)
            kpoints = KPTS.get_kpoints()
        
        # Final kpoints grid object
        uks = kpoints

        # Create the object for the VASP calculation and return it
        # ==================================

        if self.comp_params.get('functional') == 'SCAN':
            if self.calc_type in self.static_types:
                func = MPScanStaticSet
            else:
                func = MPScanRelaxSet

        else:
            if self.calc_type in self.static_types:
                func = MPStaticSet
            else:
                func = MPRelaxSet
        
        # Select the correct functional
        upf = 'LDA_54' if self.comp_params['functional'] == 'LDA' else 'PBE_54'

        vis = func(self.structure, user_incar_settings=uis, vdw=vdw, 
                   user_kpoints_settings=uks, user_potcar_functional=upf,
                   validate_magmom=False)

        return vis

    def __get_static(self, uis):
        """ Customize the VIS for a static calculation case.
        """
        
        # Build the dictionary with common static values and update uis
        common = [p for p in self.vis_default['static'].items() if not isinstance(p[1], dict)]
        uis.update(dict(common))

        # Handle specific cases based on simulation type
        # ==================================
        
        # From scratch case
        if self.calc_type.endswith('from_scratch'):
            uis.update(self.vis_default['static']['from_scratch'])

        if self.calc_type.endswith('follow_up'):
            uis.update(self.vis_default['static']['follow_up'])

        elif self.calc_type.endswith('nsfc'):
            uis.update(self.vis_default['static']['nscf'])

        # Handle specific cases based on material type
        # ==================================

        if self.calc_type.startswith('slab_'):
            uis.update(self.vis_default['static']['slab_interface'])

        elif self.comp_params.get('functional') == 'SCAN':
            uis['NELMDL'] = -10
        
        return uis

    def __get_relax(self, uis):
        """ Customize the VIS for a relax calculation case.
        """

        # Build the dictionary with common static values and update uis
        common = [p for p in self.vis_default['relax'].items() if not isinstance(p[1], dict)]
        uis.update(dict(common))

        # Handle specific cases based on simulation type
        # ==================================
        
        print("sel.calc_type in vasp_tools.py", self.calc_type)
        if self.calc_type.endswith('full_relax'):
            uis.update(self.vis_default['relax']['full_relax'])

        elif self.calc_type.endswith('pos_relax'):  #ELISA: altrimenti interface_min era F F T (site properties from pes)
            uis.update(self.vis_default['relax']['pos_relax'])
            #Set up selective dynamics array for the structrues site property
            sd_array = []
            for i in range(len(self.structure.sites)):
                sd_array.append([True, True, True])
            self.structure.add_site_property('selective_dynamics', sd_array)

        elif self.calc_type.endswith('z_relax'):
            uis.update(self.vis_default['relax']['z_relax'])
            #Set up selective dynamics array for the structrues site property
            sd_array = []
            ## START NEW BLOCK
            for site in self.structure.sites:
                if site.specie.symbol == 'H':
                    sd_array.append([True, True, True])  # H: libera in x, y, z
                else:
                    sd_array.append([False, False, True])  # altre specie: libera solo in z
            ## END NEW BLOCK
            #for i in range(len(self.structure.sites)):
            #    sd_array.append([True, True, True])
            self.structure.add_site_property('selective_dynamics', sd_array)

        elif self.calc_type.endswith('vol_relax'):
            uis.update(self.vis_default['relax']['vol_relax'])
            
        elif self.calc_type == 'bulk_pos_shape_relax':
            uis['ISIF'] = 4

        elif self.calc_type.endswith('shape_relax'):
            uis.update(self.vis_default['relax']['shape_relax'])

        # Handle specific cases based on material type
        # ==================================

        if self.calc_type.startswith('slab_') or self.calc_type.startswith('interface_'):
            uis.update(self.vis_default['relax']['slab_interface'])
            # Turn on linear mixing
            # uis['AMIX'] = 0.2
            # uis['BMIX'] = 0.0001
            # uis['AMIX_MAG'] = 0.8
            # uis['BMIX_MAG'] = 0.0001

        return uis
    
    def __get_atomic(self, uis):        
        """ Customize the VIS for an atomic calculation case.
        """
        
        uis = {}
        common = [p for p in self.vis_default['atom'].items() if not isinstance(p[1], dict)]
        uis.update(dict(common))
        
        return uis
