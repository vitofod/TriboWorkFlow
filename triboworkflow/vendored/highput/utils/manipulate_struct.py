# --- TribChem code vendored by TriboWorkFlow ---------------------------------
# Unchanged except for the imports. Do not edit: tests/vendoring.py
# compares this file with the original.
# Source: tribchem/highput/utils/manipulate_struct.py
# Version: tribchem_devel-align_elisa_w_devel
# Source SHA-256: 632d2edc014f9399f22496972bee0b5feb2b0f541f4c9654714222c208d26ca3
# Copied: 2026-10-22
# Licence: CC BY-SA 4.0, see NOTICE
# Kind: whole
# Imports: tribchem.* -> triboworkflow.vendored.*
# Import line 16: tribchem.highput.database.query -> triboworkflow.vendored.highput.database.dbtools
# --- end of the TriboWorkFlow header -----------------------------------------
import os
import warnings
from uuid import uuid4

import numpy as np
#from mpinterfaces.transformations import get_aligned_lattices
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.core.sites import PeriodicSite

from triboworkflow.vendored.highput.utils.tasktools import get_miller_str
from triboworkflow.vendored.highput.utils.vasp_tools import VaspInputSet
from triboworkflow.vendored.physics.base.solidstate import get_cell_info
from triboworkflow.vendored.physics.base.shaper import Shaper
from triboworkflow.vendored.highput.utils.tasktools import read_default_params
from triboworkflow.vendored.highput.database.dbtools import read_multiple_entry


currentdir = os.path.dirname(__file__) + '/../firetasks'


class ManipulateStruct:
    """
    Class containing different methods and tools to modify and manipulate
    pymatgen objects, i.e. structures and slabs.

    """
    
    def __init__(self, structure):
        """
        Read a pymatgen object related to an atomic structure within a lattice
        cell and store relevant information. You can pass either the pymatgen
        object itself or the corresponding dictionary (generated with .as_dict
        method). 

        Parameters
        ----------
        structure : Structure or Slab objects (pymatgen), or dicts
            Atomic structure to be read.

        """

        # Check the types of structure and slab
        if isinstance(structure, (Structure, Slab)):
            structure = structure.as_dict()
        elif not isinstance(structure, dict):
            raise ValueError('Wrong type for input argument: structure')

        self.structure = structure
        self.converter = Slab.from_dict if structure['@class']=='Slab' else Structure.from_dict

    def estimate_interface(self, inter_type='zero', axis=2):
        """
        Estimate the edges of the interface of the structure along axis.
        The kind of infert can be specified with inter_type.
        
        Options:
            - "zero": The interface must be at z = 0, the closest atoms above
                      and below are considered to delimit the interface.
            - "even": The interface is considered to be between the n/2 and 
                      n/2+1 atoms. Only works for an even number of atoms.

        Parameters
        ----------
        inter_type : str, optional
            Type of the interface. The default is 'zero'.
        
        axis : int, optional
            Axis orthogonal to the interface, a common choice is to choose a
            vertically oriented structure, i.e. interface in the xy plane.
            Integers values refer to the matrix columns (x: 0, y: 1, z: 2).
            The default value is 2.
        
        Returns
        -------
        list :
            List of two elements containing the lower and upper delimiter
            of the interface, along axis.

        """
        
        # Extract the coordinates along axis from the dictionary
        sites = np.array([s['xyz'] for s in self.structure['sites']])
        coords = np.sort(sites[:, axis])

        print("coords: ", coords)
        
        # Identify the elements of the structure
        if inter_type == 'zero':  # Default case, interface at z=0
            low = max(coords[coords < 0])
            up  = min(coords[coords > 0])

            print("low: ", low)
            print("up: ", up)

        elif inter_type == 'even':  # Interface is between n/2 and n/2+1 atom
            n = len(coords)

            if n % 2 == 0:
                low = coords[int(n/2)]
                up  = coords[int(n/2) + 1]
            else:
                raise ValueError('Your structure as an odd number of atoms, '
                                 'inter_type="even" does not make sense for it')

        return [low, up]

    def estimate_vacuum(self, kind='default', axis=2):
        """
        Estimate the vacuum for the cell along axis. Allowed estimation kinds:
            - 'default': The structure is fully inside to the cell and the
                         coordinates are continuos (no negative, no outside cell)
            - 'zero': The interface is centered on the zero for axis, so the 
                      two constituent slabs are at the cell extremities.
        
        WARNING: This works only with "squared" lattices, such as cubic and
        orthorombic-like structures.

        Parameters
        ----------
        kind : str, optional
            Kind of estimation. The default is 'default'.

        """
        
        # Group the coordinates of the atoms along axis
        c_axis = []
        sites = self.structure['sites']
        for s in sites:
            c_axis.append(s['xyz'][axis])
        c_axis = np.array(c_axis)

        # Estimate the vacuum for the various cases
        edge = self.structure['lattice']['matrix'][axis][axis]
        if kind == 'default':
            vacuum = edge - abs(max(c_axis) - min(c_axis))

        elif kind == 'zero':
            vacuum = edge - abs(max(c_axis[c_axis > 0]) - min(c_axis[c_axis < 0]))
        
        return vacuum

    def decompose_structure(self, inter_type='zero', as_dict=False):
        """
        Extract the constituents of an interface object, based on the interface 
        type that is passed. Always return something like [bot_slab, top_slab].
        
        The options provided for `inter_type` are:
            - "zero" : The interface is set on the bottom xy plane of the cell.
                       Top slab: atoms with z > 0. Bot slab: atoms with z < 0.
                       It is the default choice.
            - "middle-zero" : The same of "zero", but here PBC have been
                              applied along z, and therefore the atoms of the
                              two slabs are identified by looking if z > c/2,
                              where c is the vertical cell length.

        Parameters
        ----------
        inter_type : str, optional
            Type of the interface. The default is 'zero'.
        
        as_dict : bool, optional
            Dedice whether to return dictionaries or pymatgen objects.
        
        Returns
        -------
        result : list
            Variable list, containing the elements composing the interface.

        """

        # Extract the sites from the dictionary
        sites = self.structure['sites']

        # Identify the elements of the structure
        bot_sites = []
        top_sites = []
        for i, site in enumerate(sites):
            if inter_type == 'zero':  # Default case, interface at z=0
                if site['xyz'][2] > 0:
                    top_sites.append(site)
                elif site['xyz'][2] < 0:
                    bot_sites.append(site)
                else:
                    raise ValueError('One atom found with z=0, perhaps '
                                     'inter_type=0 is not the right choice')
            elif inter_type == 'middle-zero':
                if site['xyz'][2] >= self.structure.lattice.c / 2.:
                    top_sites.append(site)
                else:
                    bot_sites.append(site)
        
        # Decompose the structure
        bot_slab = self.structure.copy()
        top_slab = self.structure.copy()
        bot_slab['sites'] = bot_sites
        top_slab['sites'] = top_sites
        result = [bot_slab, top_slab]

        if not as_dict:
            result = [self.converter(r) for r in result]
        
        return result

    def make_vacuum_around(self, vacuum, vac_ext='mid', vac_kind='default', 
                           axis=2, inplace=False, as_dict=False):
        """
        Set the vacuum of a cell along axis. It increases or reduces the cell size
        in order to have a certain level of vacuum.

        WARNING: It works only for orthorombic-like and cubic cells.
    
        Parameters
        ----------
        structure : pymatgen structure or dict
            Structure to be modified.
            
        vac_ext : str, optional
            How to extend the vacuum around axis, allowed values are: 
                'up': Increase or decrease the vacuum in the direction of positive and
                      higher values for the atomic coordinates along axis.
                'dw': The same, but increases the cell from the side of lower
                      coordinate values.
                'mid': Try to modify the vacuum equally from both the sides of
                       the lattice cell, along the axis direction. It is defalt
        
        vac_kind : float or int, optional
            Kind of vacuum, and so of atomic structure, that is provided. It is
            used to estimate the initial existing vacuum of the cell.
            
        inplace : bool, optional
            If set to True, the instance object is modified, otherwise a new
            modified structure is returned.
        
        as_dict : bool, optional
            Decide to return a pymatgen object or a dict. It only makes sense
            in the case inplace=False. The default is False.
    
        """

        # Estimate and evaluate vacuum of the cell along axis
        vacuum_old = self.estimate_vacuum(vac_kind, axis)
        cell_edge = np.array(self.structure.copy()['lattice']['matrix'])[axis, axis]
        slab_thick = cell_edge - vacuum_old
        
        # Apply the vacuum to the cell structure
        structure = self.structure.copy()
        lattice = structure['lattice']
        # Change the axis length
        k = 'a' if axis == 0 else 'b' if axis == 1 else 'c'
        lattice[k] = slab_thick + vacuum
        # Change the lattice matrix
        new_cell_edge = slab_thick + vacuum
        lattice['matrix'][axis][axis] = new_cell_edge
        # Change the volume
        lattice['volume'], _ = get_cell_info(lattice['matrix'])
        structure['lattice'] = lattice

        # Rescale xyz coordinates, to stay coherent with the new lattice edge
        sites = structure['sites'].copy()
        for i, s in enumerate(sites):
            v = s['abc'][axis] * cell_edge / new_cell_edge
            sites[i]['abc'][axis] = v
        structure['sites'] = sites

        if inplace:
            self.structure = structure
        else:
            if not as_dict:
                structure = self.converter(structure)
            return structure

    def _rescale_sites(self, structure, cell_edge, new_cell_edge, axis=2):
        sites = structure['sites'].copy()
        for i, s in enumerate(sites):
            sites[i]['abc'][axis] = s['abc'][axis] * cell_edge / new_cell_edge
        structure['sites'] = sites
        return structure
    
    @staticmethod
    def fix_interface(interface_opt, interface_init=None, as_dict=False):
        """
        Fix the interface after the adhesion workflow. Interface_init should 
        have the interface centered at zero, atoms have positive and
        negative z-coordinates. Interface_opt instead should be the result of
        a simulation made with VASP that have applied PBC to the structures,
        thus removing the z>0 distinction of the coordiantes.
        This function will set the interface back to z=0, identifying correctly
        the belonging of the atoms of its own slab, even in case of
        cross-slab migration.

        """
        
        if not isinstance(interface_opt, dict):
            interface_opt = interface_opt.as_dict()

        # Simplest case: interface_init is not provided. I will loop over the
        # sites and subtract c the coordinates with z>c/2.
        if interface_init is None:
            warnings.warn('Interface_init not provided in the fix. '
                          'A loop over the sites is done and c will be sutracted '
                          'to the atomic coordinates when z>c/2.')
            c = interface_opt['lattice']['c']
            for s in interface_opt['sites']:
                if s['xyz'][2] >= c/2:
                    s['xyz'][2] -= c
                    s['abc'][2] -= 1
            
            # Convert the dictionary to a Structure if required
            if not as_dict:
                interface_opt = Structure.from_dict(interface_opt)
            
            return interface_opt
        
        if not isinstance(interface_init, dict):
            interface_init = interface_init.as_dict()

        # Identify initial upper and lower slab
        up_index, dw_index = [], []
        test_up, test_dw = [], []
        
        for i, s in enumerate(zip(interface_init['sites'], interface_opt['sites'])):
            s1, s2 = s
            if s1['xyz'][2] > 0: up_index.append(i) 
            else: dw_index.append(i)
            
            if s2['xyz'][2] > 0: test_up.append(i) 
            else: test_dw.append(i)

        if test_up == up_index or test_dw == dw_index:  # No fix is required in this case
            return interface_opt

        # Cycle over the final sites and check for cross-migration
        up_cross_migration, dw_cross_migration = [], []
        c = interface_init['lattice']['c']
        for i, s in enumerate(interface_opt['sites']):
            if s['xyz'][2] >= c/2:
                s['xyz'][2] -= c
                s['abc'][2] -= 1
                if not i in dw_index:
                    up_cross_migration.append(i) #contiene atomi che erano su e ora sono giù
            elif s['xyz'][2] >= 0:
                if not i in up_index:
                    dw_cross_migration.append(i)
            else:
                raise ValueError('Something strange occurring. The opt z-coords '
                                 'should be all positive due to applied PBC')

        # Handle the cross migration case
        if any([up_cross_migration, dw_cross_migration]): 
            print(f"dw_cross_migration: ",dw_cross_migration) 

            if all([up_cross_migration, dw_cross_migration]):
                
                # Check if the slabs have been exchanged by Atomate
                nat = len(interface_opt['sites'])
                if len(up_cross_migration + dw_cross_migration) == nat:
                    elements = []
                    for s in interface_opt['sites']:  # Find the atomic types
                        elements.append(s['species'][0]['element'])
                    # Grep the unique ones ordered and count elements
                    count = {}
                    seen = set()
                    composition = [x for x in elements if not (x in seen or seen.add(x))]
                    for e in composition:
                        count[e] = elements.count(e)
                    
                    # Exchange the slabs in the interface dictionary
                    species_1 = interface_opt['sites'][:count[composition[0]]].copy()
                    species_2 = interface_opt['sites'][count[composition[0]]:].copy()
                    interface_opt['sites'] = species_2 + species_1

                    ManipulateStruct.fix_interface(interface_opt,
                                                   interface_init,
                                                   as_dict=as_dict)

                else:
                    warnings.warn('Interface is uncoverable: the atoms of both '
                                  'slabs have cross-migrated. Input '
                                  'interface is returned')
                    return interface_opt

            # Atoms from lower slab is now part of upper ones.
            dw_z, up_z = [], []
            if dw_cross_migration != []:
                for i in dw_cross_migration: #qua dentro ci va
                    dw_z.append(interface_opt['sites'][i]['xyz'][2])
                print(f"up_index: ", up_index) #è vuoto
                for j in up_index:
                    print(f"up_index: ", up_index)
                    print(f"questo è errore adesione: ",j) #non va qua dentro
                    up_z.append(interface_opt['sites'][j]['xyz'][2])
                max_dw_z = max(dw_z)
                print(f"max_dw_z: ",max_dw_z)
                min_up_z = min(up_z) #errore qui min() senza argomenti ELISA --> up_z vuoto
                
                if max_dw_z > min_up_z:
                    raise ValueError('Interface is uncoverable: an atom from '
                                     'the lower slab is now part of the upper slab')
                else:
                    shift = max_dw_z + (min_up_z - max_dw_z)/2  # Put interface at z=0
            else:
                for i in up_cross_migration:
                    up_z.append(interface_opt['sites'][i]['xyz'][2])
                for j in dw_index:
                    dw_z.append(interface_opt['sites'][j]['xyz'][2])
                max_dw_z = max(dw_z)
                min_up_z = min(up_z)
               
                print(f"max_dw_z: {max_dw_z} > min_up_z: {min_up_z}")
                if max_dw_z > min_up_z:
                    raise ValueError('Interface is uncoverable: an atom from '
                                     'the upper slab is now part of the lower slab')
                else:
                    shift = min_up_z - (min_up_z - max_dw_z)/2  # Put interface at z=0
            
            # Shift all the sites to put the interface at z=0
            for s in interface_opt['sites']:
                s['xyz'][2] -= shift
                s['abc'][2] -= (shift / c)
        
        # Convert the dictionary to a Structure if required
        if not as_dict:
            interface_opt = Structure.from_dict(interface_opt)

        return interface_opt

    @staticmethod
    def create_interface(bot_slab, top_slab, vacuum=15, inter_params={}):
        """
        TG: remove mpinterface dependency (this method is currently not used).

        Match an interface out of two slabs passed as input.

        Parameters
        ----------
        bot_slab : TYPE
            DESCRIPTION.
        top_slab : TYPE
            DESCRIPTION.
        vacuum : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # Define the variables to be returned
        bot_aligned = None
        top_aligned = None
        interface = None

        # Set missing interface_params
        dfl = currentdir + '/defaults.json'
        inter_params = read_default_params(dfl, 'inter_params', inter_params)
        
        # Add safe vacuum to the slabs and try to align them
        d = inter_params.copy()
        _ = d.pop('interface_distance')
        bot_slab = ManipulateStruct(bot_slab).make_vacuum_around(vacuum=50)
        top_slab = ManipulateStruct(top_slab).make_vacuum_around(vacuum=50)
        bot_aligned, top_aligned = get_aligned_lattices(bot_slab, top_slab, **d)

        # If the structures do exist, then the interface can be matched
        if bot_aligned and top_aligned:     

            # Center bot and top slab around 0 and form an interface
            top_aligned, bot_aligned = recenter_aligned_slabs(
                top_aligned, bot_aligned, d=inter_params['interface_distance'])
            interface = stack_aligned_slabs(bot_aligned, top_aligned)
            interface = clean_up_site_properties(interface)
            
            # Assign the desired vacuum to the interface and rescale slabs
            obj = ManipulateStruct(interface)
            obj.make_vacuum_around(vacuum=vacuum, vac_kind='zero', inplace=True)
            bot_aligned, top_aligned = obj.decompose_structure(inter_type='zero')
            interface = obj.converter(obj.structure)

            # Convert the obtained structure back to slabs objects
            bot_aligned = structure_to_slab(bot_aligned, bot_slab)
            top_aligned = structure_to_slab(top_aligned, top_slab)

        return bot_aligned, top_aligned, interface
    
    @staticmethod
    def shorten_interface_slabs(interface, bot_layers, top_layers, tol=0.1,
                                inter_type='zero', as_dict=False):
        """
        Reduce the slabs forming an interface to a minimum number of layers
        up and down. You need to provide an interface object or dict and the
        DESIRED number of layers for the bottom and top slabs, respectively.

        """
        
        if not isinstance(interface, dict):
            interface = interface.as_dict()
        
        obj = ManipulateStruct(interface)
        bot_slab, top_slab = obj.decompose_structure(inter_type, as_dict=False)

        # REDUCE THE BOTTOM SLAB FROM BELOW
        rm_bot = len(Shaper._get_layers(bot_slab, tol).keys()) - bot_layers
        if rm_bot > 0:
            bot_slab = Shaper._remove_layers(bot_slab, rm_bot, method='layers',
                                             position='bottom', center=False)
        bot_slab = bot_slab.as_dict()
        
        # REDUCE THE TOP SLAB FROM ABOVE
        rm_top = len(Shaper._get_layers(top_slab, tol).keys()) - top_layers
        if rm_top > 0:
            top_slab = Shaper._remove_layers(top_slab, rm_top, method='layers',
                                             position='top', center=False)
        top_slab = top_slab.as_dict()

        # Reassign
        interface['sites'] = top_slab['sites'] + bot_slab['sites']
        
        if not as_dict:
            interface = Structure.from_dict(interface)
            bot_slab = Structure.from_dict(bot_slab)
            top_slab = Structure.from_dict(top_slab)

        return interface, bot_slab, top_slab

def generate_chemabs_inputs(field, coverage, sites='all', tag=None,
                            calc_type='interface_z_relax'):
    """
    Generate the inputs for the adatom chemisorption workflow, by passing the
    document field of MongoDB queried from the Database.

    """

    # Prepare the list of sites we are interested to simulate
    sites_entry = ['structure', 'coverage_'+str(coverage)]
    data_dict = read_multiple_entry(field, sites_entry)
    if sites == 'all':
        sites = data_dict.keys()
    sites_entry = [sites_entry] * len(sites)

    # Extract the structures from the dictionary
    structures = []
    for i, s in enumerate(sites):
        structures.append(Slab.from_dict(data_dict[s]['init']))
        sites_entry[i].extend([s])

    vis = []
    tags = []
    comp_params = field['comp_params']
    for s in structures:
        # Define the custom tag label for the structure s
        if tag is None:
            label = str(uuid4())
        else:
            label = tag + '_' + str(uuid4())

        # Calculate VASP input set
        v = VaspInputSet(s, comp_params, calc_type).get_vasp_settings()

        # Store the data for each simulation
        vis.append(v)
        tags.append(label)

    return structures, vis, tags

def generate_chargedisp_inputs(interface, comp_params, tag=None, inter_type='zero',
                               calc_type='slab_from_scratch'):
    """
    Generate a set of inputs to calculate the Charge Displacement of an interface

    """
    
    # Extract the bot and top slabs and shift the interface to the center
    # of the cell. This is necessary for a nice plot. Currently only
    # inter_type=0 is implemented.
    if inter_type == 'zero':
        c = interface.lattice.c
        components = ManipulateStruct(interface).decompose_structure(inter_type)
        slabs = [interface] + components
        
        for s in slabs:
            s.translate_sites(indices=np.arange(len(s.sites)), 
                              vector=[0, 0, c/2],
                              frac_coords=False, 
                              to_unit_cell=False)
    else:
        raise ValueError('Invalid inter_type')

    # Generate the vasp inputs and tags
    tags = []
    vis = []
    for s, l in zip(slabs, ['interface', 'bot_slab', 'top_slab']):
        # Define the custom tag label for the structure s
        if tag is None:
            label = str(uuid4())
        else:
            label = tag + '_' + l + '_' + str(uuid4())

        # Calculate VASP input set
        v = VaspInputSet(s, comp_params, calc_type).get_vasp_settings()

        # Store the data for each simulation
        vis.append(v)
        tags.append(label)

    return slabs, vis, tags

def generate_ppes_inputs(structure, shifts, comp_params, inter_type='middle', tag=None,
                         calc_type='slab_from_scratch', combine=False):
    """
    Prepare the interface inputs to calculate the PPES. At the moment the bot
    and top slabs are identified by looking at the z coordinate, i.e. the 
    interface should be centered at zero.

    """

    structures = []
    vis = []
    tags = []

    structure_wrapped = Structure(
        lattice=structure.lattice,
        species=structure.species,
        coords=[site.frac_coords for site in structure.sites],
        coords_are_cartesian=False,
        to_unit_cell=True,
    )
    
    # Identify the top_slab and the bot_slab by looking at atomic z coordinates
    sites_to_shift = []
    for i, site in enumerate(structure_wrapped.sites):
        if inter_type == 'zero':
            if site.coords[2] > 0:
                sites_to_shift.append(i)
        elif inter_type == 'middle':
            #if site.coords[2] < structure.lattice.c / 2:
            if site.frac_coords[2] < 0.5:
                print("ELISA site to shift: ", i)
                sites_to_shift.append(i)

    # Loop over all the shifts
    for s in shifts:
        # Define the custom tag label for shift s
        if tag is None:
            label = str(uuid4())
        else:
            label = tag + '_' + str(s) + '_' + str(uuid4())
        
        # Clean site properties removing NoneType, and translate upper slab
        inter = clean_up_site_properties(structure.copy())
        inter.translate_sites(indices=sites_to_shift, 
                              vector=[0, 0, s],
                              frac_coords=False, 
                              to_unit_cell=False)
        
        # Calculate VASP input set
        v = VaspInputSet(inter, comp_params, calc_type).get_vasp_settings()
        
        # Store the data for each simulation
        structures.append(inter)
        vis.append(v)
        tags.append(label)

    # Convert to a single list of lists to eventually feed run_vasp_wf
    if combine:
        inputs = [[s, v, t] for s, v, t in zip(structures, vis, tags)]
        return inputs
    
    return structures, vis, tags

def generate_pes_inputs(bot_slab, top_slab, shifts, comp_params, tag=None,
                        calc_type='interface_z_relax', interface_distance=2, combine=False):
    """
    Prepare the inputs to run VASP on the fly for the different relative
    lateral positions that are asked for the interface.

    """

    structures = []
    vis = []
    tags = []

    # Loop over all the shifts in order to do them one by one
    for s in shifts.keys():
        
        # Define the custom tag label for shift s
        if tag is None:
            label = str(uuid4())
        else:
            label = tag + '_' + s + '_' + str(uuid4())

        # Build the interface by applying a single shifts
        inter = apply_interface_shifts(bot_slab, top_slab, shifts[s], interface_distance)
        
        # Calculate the vasp input set for all the simulations
        v = VaspInputSet(inter, comp_params, calc_type).get_vasp_settings()
        
        # Store the data for each simulation
        structures.append(inter)
        vis.append(v)
        tags.append(label)

    # Convert to a single list of lists to eventually feed run_vasp_wf
    if combine:
        inputs = [[s, v, t] for s, v, t in zip(structures, vis, tags)]
        return inputs
    
    return structures, vis, tags

def apply_interface_shifts(bot_slab, top_slab, shifts, interface_distance=2):
    """
    Create a list of interfaces, by combining a bot_slab (substrate) to a 
    top_slab (coating) and placing them in different relative lateral
    positions by applying different lateral shift to the upper slab. 

    Parameters
    ----------
    bot_slab : pymatgen.core.surface.Slab
        Bottom slab structure (substrate).

    top_slab : pymatgen.core.surface.Slab
        Top slab structure (coating).

    shifts : np.ndarray or list of lists
        Lateral shifts that needs to be applied to the upper slab on top of the
        lower one, in order to achieve all the relative lateral configurations
        between the two structures. It can be either a list of two-elements
        lists or a numpy matrix of shape nx2.
        The structure should resemble: [[x0, y0], [x1, y1], ..., [xn, yn]].

    Returns
    -------
    interfaces: (list of) pymatgen.core.surface.Slab
        All the interfaces created matching the slabs according to shifts.

    """

    # Convert shifts to a numpy array
    shifts = np.array(shifts)
    
    # Recenter the slabs to the center of the cell
    top_slab, bot_slab = recenter_aligned_slabs(top_slab, bot_slab, d=interface_distance)
    
    # Create and save the different interfaces
    interfaces = []
    for s in shifts:
        inter_struct = stack_aligned_slabs(bot_slab, top_slab, [s[0], s[1], 0])
        interfaces.append(clean_up_site_properties(inter_struct))
    
    # Just return the interface object if only a shift is required
    if shifts.shape[0] == 1:
        interfaces = interfaces[0]
    
    return interfaces

def generate_interface_comp_params(params):
    """
    Generate the computational parameters for an interface, by comparing the
    values found from a list of parameters for the interface components and 
    selecting the highest priority values. Important data is:
        - functional
        - encut
        - kdens
        - is_metal
        - use_spin
        - vdw

    """

    # Create a single dict where each element is a list with all the data
    params = list_of_dict_to_dict_of_list(params)
    
    # Check the elements of params
    keys=['functional', 'encut', 'kdens', 'is_metal', 'use_spin', 'vdw']
    if not set(keys).issubset(set(params.keys())):
        raise ValueError('The key values of params ({}) is not subset of {}'
                         .format(params.keys(), keys))
    
    # Start to prepare interface parameters
    out_params = {}
    
    # Select the unique functional
    if len(set(params['functional'])) > 1:
        raise ValueError('Wrong selected slabs. YOU MUST HAVE THE SAME FUNCTIONAL')
    out_params['functional'] = params['functional'][0]
    
    out_params['encut'] = max(params['encut'])
    out_params['kdens'] = max(params['kdens'])
    out_params['is_metal'] = any(params['is_metal'])
    out_params['use_spin'] = any(params['use_spin'])
    
    vdw = set(params['vdw'])
    if vdw == set([None]):
        out_params['vdw'] = None
    else:
        if None in vdw:
            vdw.remove(None)
        if len(vdw) > 1:
            raise ValueError('Different VdW corrections have been used for the '
                             'slabs. I do not know what to choose')
        out_params['vdw'] = list(vdw)[0]
        
    return out_params

def list_of_dict_to_dict_of_list(params):
    """
    Convert a list of dictionary to a dictionary with lists for values.
    All the dictionaries in the list should have the same keys, missing values
    are substituted with None.

    """

    out_params = {}
    keys = params[0].keys()
    
    for k in keys:
        values = []
        for p in params:
            values.append(p.get(k, None))
        out_params[k] = values

    return out_params

def material_id_from_list(params, index=0):
    """
    Return the ID specific for a certain material, from a params dictionary
    where you have lists of IDs. Useful to deal with input for calculations
    with materials pair, two or many slabs, interfaces.

    """
    
    p = {'mid': params['mid'][index]}
    
    if 'formula' in params.keys():
        if params['formula'] is not None:
            p.update({'formula': params['formula'][index]})
    if 'name' in params.keys():
        if params['name'] is not None:
            p.update({'name': params['name'][index]})
    if 'miller' in params.keys():
        if params['miller'] is not None:
            p.update({'miller': params['miller'][index]})
    
    return p

def generate_interface_ids(params, slabs=None):
    """
    Generate up to four main identifier for an interface, namely, the mid,
    the formula, and the name. The ids are generated by combining the three
    same values for the two slabs.

    Parameters
    ----------
    params : dict
        Parameters dictionary that should contain the identifier values for
        the two slabs matched to form an interface. The structure might be:
            {
                "mid" : ['mid-1', 'mid-2'],
                "formula" : ['formula_1', 'formula_2'],
                "name" : ['name_1', 'name_2'],
                "miller" : [[miller_1], [miller_2]]
            }
        where the pedix is referred to the first slab (substrate) or the second 
        slab (coating). Types: mid, formula, name are str; miller is a list.
        Formula and name could be equal to None.
    
    slabs : tuple (or list) of pymatgen.core.surface.Slab, optional
        Force the creation of a formula out of the pymatgen slab objects, if 
        no formula is provided in params. It should be: (bot_slab, top_slab).
        The default is None.

    Returns
    -------
    ids : tuple
        Tuple containing the identifier of the interface.

    """
    
    # Define the mid
    mid = params['mid'][0] + '_' + params['mid'][1]
    
    # Eventually define the interface formula, default taken from slabs
    if params['formula'] is None:
        if slabs is not None:
            try:
                bot_slab, top_slab = slabs
                f_1 = slabs[0].composition.reduced_formula
                f_2 = slabs[1].composition.reduced_formula
                formula = f_1 + '_' + f_2 if (bool(f_1) and bool(f_2)) else None
            except:
                raise ValueError('Input argument slabs not defined correctly')
        else:
            formula = None
    else:
        formula = params['formula'][0] + '_' + params['formula'][1]

    # Define the combined miller indexes
    miller = [params['miller'][0], params['miller'][1]]

    # Eventually define the interface name
    if params['name'] is None:
        if slabs is not None and formula is not None:
            name = formula.split('_')[0] + get_miller_str(miller[0]) + '-' + \
                   formula.split('_')[1] + get_miller_str(miller[1])
        else:
            name = None
    else:
        name = params['name'][0] + '_' + params['name'][1]

    # Define final identifiers
    ids = (mid, formula, name, miller)

    return ids    

def structure_to_slab(structure, slab, as_dict=False, transfer_keys=['miller_index', 
                      'oriented_unit_cell', 'energy', 'reconstruction', 
                      'scale_factor', 'shift']):
    """
    Convert a structure object to a slab object by working directly on their 
    dictionaries. This is useful when using Atomate to run VASP simulations.
    In that case, despite you start with a slab, you end up having a structure
    in the `tasks` collection. With this function you can integrate the final
    structure object with missing dictionary keys from its reference slab.
    The `structure` and `slab` can be either pymatgen objects or their dicts.

    Parameters
    ----------
    structure : dict or pymatgen.core.structure.Structure
        Structure to be converted to a Slab.

    slab : dict or pymatgen.core.surface.Slab
        Reference slab to be used to integrate the structure keys.
    
    transfer_magmoms : bool or str, optional
        Decide whether to transfer the magnetic moments properties from the
        reference slab to the structure. If set to 'all', it will transfer all
        the properties that might be present in the slabs, such as: 'magmom',
        'bulk_equivalent', 'bulk_wyckoff'. The default is False.
    
    as_dict : bool, optional
        Return a dictionary or a pymatgen Slab object. The default is False.
    
    transfer_keys: dict of str, optional
        List of attributes that should be transferred as is from the reference
        slab to the structure. 
        The default is: ['miller_index', 'oriented_unit_cell', 'energy', 
                         'reconstruction', 'scale_factor', 'shift']

    Returns
    -------
    out_slab : pymatgen.core.surface.Slab
        The converted slab already as pymatgen object.

    """

    # Check the types of structure and slab
    if isinstance(structure, Structure):
        structure = structure.as_dict()
    elif not isinstance(structure, dict):
        raise ValueError('Wrong type for input argument: structure')
    if isinstance(slab, Slab):
        slab = slab.as_dict()
    elif not isinstance(slab, dict):
        raise ValueError('Wrong type for input argument: slab')

    # Update the structure and convert it back to a slab
    out_slab = structure.copy()
    out_slab['@class'] = 'Slab'
    out_slab['@module'] = 'pymatgen.core.surface'
    for k in transfer_keys:
        out_slab[k] = slab[k]

    if not as_dict:
        out_slab = Slab.from_dict(out_slab)

    return out_slab

def slab_to_structure(slab, as_dict=False):
    """
    Convert a pymatgen Slab back to a pymatgen Structure by removing all the
    attributes that are not needed.

    Parameters
    ----------
    slab : pymatgen.core.surface.Slab or dictionary
        Slab object or dictionary to clean.

    as_dict : bool, optional
        Set the returned cleaned slab as dict or object. The default is False.

    Returns
    -------
    None.

    """

    # Check the types of structure and slab
    if not isinstance(slab, dict):
        if not isinstance(slab, Slab): 
            return slab
        else:
            slab = slab.as_dict()
    
    transfer_keys=['miller_index', 'oriented_unit_cell', 'energy', 
                   'reconstruction', 'scale_factor', 'shift']
    
    # Update the slab to a structure by removing unnecessary items
    out_struct = slab.copy()
    out_struct['@class'] = 'Structure'
    out_struct['@module'] = 'pymatgen.core.structure'
    
    rm_keys = list(set(transfer_keys).intersection(set(slab.keys())))
    for k in rm_keys:
        out_struct.pop(k)

    if not as_dict:
        out_struct = Structure.from_dict(out_struct)

    return out_struct

def clean_up_site_properties(structure):
    """
    Cleans up site_properties of structures that contain NoneTypes.
    
    If an interface is created from two different structures, it is possible
    that some site properties like magmom are not set for both structures.
    This can lead later to problems since they are replaced by None.
    This function replaces NoneTypes with 0.0 for magmom and deletes all other
    site_properties if None entries are found in it.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        Input structure

    Returns
    -------
    struct : pymatgen.core.structure.Structure
        Output structure

    """
    struct = structure.copy()
    for key in struct.site_properties.keys():
        if key == 'magmom':
            new_magmom = []
            for m in struct.site_properties[key]:
                if m == None:
                    new_magmom.append(0.0)
                else:
                    new_magmom.append(m)
            struct.add_site_property('magmom', new_magmom)
        else:
            if any(struct.site_properties[key]) is None:
                struct.remove_site_property(key)
    return struct

def stack_aligned_slabs(bottom_slab, top_slab, top_shift=[0,0,0]):
    """
    Combine slabs that are centered around 0 into a single structure.
    
    Optionally shift the top slab by a vector of cartesian coordinates.

    Parameters
    ----------
    bottom_slab : pymatgen.core.structure.Structure or pymatgen.core.surface.Slab
        Bottom slab.
    top_slab : pymatgen.core.structure.Structure or pymatgen.core.surface.Slab
        Top slab.
    top_shift : list of 3 floats, optional
        Vector of caresian coordinates with which to shift the top slab.
        The default is [0,0,0].

    Returns
    -------
    interface : pymatgen.core.structure.Structure or pymatgen.core.surface.Slab
                depending on type of bottom_slab
        An interface structure of two slabs with an optional shift of the top
        slab.

    """
    interface = bottom_slab.copy()
    t_copy = top_slab.copy()
    
    t_copy.translate_sites(indices=range(len(t_copy.sites)),
                           vector=top_shift,
                           frac_coords=False, to_unit_cell=False)
    
    for s in t_copy.sites:
        new_site = PeriodicSite(lattice=interface.lattice,
                                coords=s.frac_coords,
                                coords_are_cartesian=False,
                                species=s.species,
                                properties=s.properties)
        interface.sites.append(new_site)
    
    return interface

def recenter_aligned_slabs(top_slab, bot_slab, d=2.5):
    """
    Center two slabs around z=0 and give them the distance d if provided.

    Parameters
    ----------
    top_slab : pymatgen.core.structure.Structure
        The slab that should be on top.

    bot_slab : pymatgen.core.structure.Structure
        The slab that should be on the bottom.

    d : float, optional
        The desired distance between the slabs. The default is 2.5.

    Returns
    -------
    t_copy : pymatgen.core.structure.Structure
        Top slab that is shifted so that the lowest atom is at +d/2
    b_copy : pymatgen.core.structure.Structure
        Bottom slab that is shifted so that the topmost atom is at -d/2

    """
    t_copy = top_slab.copy()
    b_copy = bot_slab.copy()
    top_zs=[]
    bot_zs=[]
    for s in t_copy.sites:
        top_zs.append(s.coords[-1])
    top_shift = -min(top_zs) + d/2
    
    for s in b_copy.sites:
        bot_zs.append(s.coords[-1])
    bot_shift = -max(bot_zs) - d/2

    t_copy.translate_sites(indices=range(len(t_copy.sites)),
                           vector=[0, 0, top_shift],
                           frac_coords=False, to_unit_cell=False)
    b_copy.translate_sites(indices=range(len(b_copy.sites)),
                           vector=[0, 0, bot_shift],
                           frac_coords=False, to_unit_cell=False)
    return t_copy, b_copy
