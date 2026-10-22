# --- TribChem code vendored by TriboWorkFlow ---------------------------------
# Unchanged except for the imports. Do not edit: tests/vendoring.py
# compares this file with the original.
# Source: tribchem/highput/utils/errors.py
# Version: tribchem_devel-align_elisa_w_devel
# Source SHA-256: fe2816fa1a1c29dbd80eef91eeb7ce9395c0841ce0249ab815824e4bae596e1d
# Copied: 2026-10-22
# Licence: CC BY-SA 4.0, see NOTICE
# Kind: whole
# Imports: tribchem.* -> triboworkflow.vendored.*
# --- end of the TriboWorkFlow header -----------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:03:22 2021

Custom errors for any Firetasks and Workflow.

@author: glosi000
"""

__author__ = 'Gabriele Losi'
__copyright__ = 'Prof. M.C. Righi, University of Bologna'
__contact__ = 'clelia.righi@unibo.it'
__date__ = 'February 8th, 2021'


from pymatgen.core.surface import Slab


# ============================================================================
# Database Core Errors
# ============================================================================

class DatabaseError(Exception):
    """ General error 
    """
    pass

class NavigatorError(DatabaseError):
    """ Errors in database initialization and queries, for Navigator class.
    """
    pass

class StructureNavigatorError(NavigatorError):
    """ Errors in calling the StructureNavigator child class.
    """
    pass

class NavigatorMPError(DatabaseError):
    """ Errors in querying the online Materials Project Database.
    """
    pass


# ============================================================================
# Firetasks Errors
# ============================================================================

class GeneralErrorFT(Exception):

    @staticmethod
    def check_slab_type(slab, message):
        """ Check the type of the slab
        """
        try:
            assert isinstance(slab, Slab)
            _ = list(slab.miller_index)
        except:
            raise GeneralErrorFT("You have used {} as an input for <{}>.\n"
                                 "Please use <class 'pymatgen.core.surface.Slab'>"
                                 " instead".format(type(slab), message))

    @staticmethod
    def check_collection(collection, functional=['PBE', 'SCAN'], 
                         data=['bulk', 'slab', 'interface']):
        
        allowed_collections = [f + '.' + d for f in functional for d in data]

        if collection not in allowed_collections:
            raise GeneralErrorFT('Wrong value for collection. Allowed '
                                 'values: "PBE/SCAN.bulk", "PBE/SCAN.slab_data", '
                                 '"PBE/SCAN.interface"')

class InitError(GeneralErrorFT):
    """ Initialization error
    """

class MoveTagResultsError(GeneralErrorFT):
    """ Errors when moving data between two different db_file/database/collection.
    """
    
    @staticmethod
    def check_entry(entry, msg='entry'):

        if isinstance(entry, list):
            if not all([isinstance(n, list) for n in entry]):
                raise MoveTagResultsError("Wrong type for arguments: {}. "
                                          "Allowed type: list".format(msg))
        
        # elif not isinstance(entry, str):
        #     raise MoveTagResultsError("Wrong argument: {}. "
        #                               "Allowed types: str, list".format(msg))

class RunVaspSimulationError(GeneralErrorFT):
    """ Error when running a relax calculation
    """

    @staticmethod
    def is_data(structure, mp_id, functional):
        if structure is None:
            formula = structure.composition.reduced_formula
            raise RunVaspSimulationError('No entry found in DB {} for a '
                                      'structure with mpid: {}, functional: {}'
                                      .format(formula, mp_id, functional))


    @staticmethod
    def check_entries(entry_check, entry_to, entry_from):

        MoveTagResultsError.check_entry(entry_check, 'entry_check')
        MoveTagResultsError.check_entry(entry_to, 'entry')
        MoveTagResultsError.check_entry(entry_from, 'entry_tag')

        if len(list(entry_to)) != len(list(entry_from)):
            raise MoveTagResultsError("Wrong arguments: entry or entry_tag. If "
                                      "converted to str, should have same length")

class GenerateSlabsError(GeneralErrorFT):
    """ Error in generating slabs.
    """

    @staticmethod
    def check_name(name):
        if name is not None:
            if not isinstance(name, (str, list)):
                raise GenerateSlabsError("Wrong type for argument: name. "
                                         "Allowed types: str, list, None")

    @staticmethod
    def check_miller(miller):
        if not isinstance(miller, list):
            raise GenerateSlabsError("Wrong type for argument: miller. "
                                     "Allowed types: list")
        elif any([isinstance(x, list) for x in miller]) and not all([isinstance(x, list) for x in miller]):
            raise GenerateSlabsError("Wrong type for elements of list: "
                                     "miller. Allowed types: list")

    @staticmethod
    def check_thickness(thickness):
        if not isinstance(thickness, (list, float, int)):
            raise GenerateSlabsError("Wrong type for argument: thickness. "
                                     "Allowed types: list, float, int")
        if isinstance(thickness, list):
            if not all([isinstance(x, (float, int)) for x in thickness]):
                raise GenerateSlabsError("Wrong type for elements of list: "
                                         "miller. Allowed types: int, float")

    @staticmethod
    def check_vacuum(vacuum):
        if not isinstance(vacuum, (list, float, int)):
            raise GenerateSlabsError("Wrong type for argument: vacuum. "
                                     "Allowed type: list")
        if isinstance(vacuum, list):
            if not all([isinstance(x, (float, int)) for x in vacuum]):
                raise GenerateSlabsError("Wrong type for elements of list: "
                                         "vacuum. Allowed types: list")

    @staticmethod
    def check_entry(entry):
        if not isinstance(entry, (str, list)):
            raise GenerateSlabsError("Wrong type for argument: entry. "
                                     "Allowed types: list of str, str.")
        if isinstance(entry, list):
            if not all([isinstance(x, (str, list)) for x in entry]):
                raise GenerateSlabsError("Wrong type for elements of list: "
                                         "entry. Allowed types: str, list")
            
    @staticmethod
    def check_comp_params(cp):
        if cp is not None:
            if not isinstance(cp, (dict, list)):
                raise GenerateSlabsError("Wrong type for argument: comp_params. "
                                         "Allowed types: str, list, None")
            if isinstance(cp, list):
                if not all([isinstance(x, (dict)) for x in cp]):
                    raise GenerateSlabsError("Wrong type for elements of list: "
                                             "comp_params. Allowed types: dict")

    @staticmethod
    def check_inputs(name, miller, thickness, vacuum, entry, comp_params):

        GenerateSlabsError.check_name(name)
        GenerateSlabsError.check_miller(miller)
        GenerateSlabsError.check_thickness(thickness)
        GenerateSlabsError.check_vacuum(vacuum)
        GenerateSlabsError.check_entry(entry)
        GenerateSlabsError.check_comp_params(comp_params)
        
        # Check if miller, names, comp_params are compatible
        if not all([isinstance(x, list) for x in miller]):
            miller = [miller]
        if name is not None:
            name = [name] if not isinstance(name, list) else name
            if len(miller) != len(name):
                raise GenerateSlabsError("Wrong arguments: miller, name. " 
                                         "They should have the same length if lists")
        if comp_params is not None:
            comp_params = [comp_params] if not isinstance(comp_params, list) else comp_params
            if len(miller) != len(comp_params):
                raise GenerateSlabsError("Wrong arguments: miller, comp_params. " 
                                         "They should have the same length if lists")

        # If one is a list, both of them should be
        if isinstance(thickness, list):
            if not isinstance(entry, list):
                raise GenerateSlabsError("Wrong type for arguments: thickness, "
                                         "entry. If one is a list, both "
                                         "should be.")
            if len(thickness) != len(entry):
                raise GenerateSlabsError("Wrong arguments: thickness, entry. " 
                                         "They should have the same length if lists")

class SurfaceEnergyError(GeneralErrorFT):
    """ Error in surface energy Firetask.
    """
    
    @staticmethod
    def check_entry(entry):
        """ At least two elements are needed to calculate the surface energy
        """
        if len(entry) < 2:
            raise SurfaceEnergyError('Wrong argument: entry. At least two '
                                     'elements are needed to calculate the '
                                     'surface energy, i.e. OUC bulk and a slab. '
                                     'You provided: {}'.format(entry))
    
    @staticmethod
    def check_output(output):
        for el in output:
            if el is None:
                raise SurfaceEnergyError('Error with output data. Some data is '
                                         'missing. Cannot proceed with surface '
                                         'energy calculation')
        
        if not 'energy_per_atom' in output[0].keys():
                raise SurfaceEnergyError('Error with output data. The first dict '
                                         'should contain the OUC bulk, and have '
                                         'a key: "energy_per_atom". Your keys '
                                         'are: {}'.format(output[0].keys()))
        else:
            for i, out in enumerate(output[1:]):
                k = out.keys()
                val = None
                if not 'energy' in k:
                    val = '"energy"'
                if not 'nsites' in k: 
                    if val is not None:
                        val = val + ', "nsites"'
                    else:
                        val = '"nsites"'
                if val is not None:
                    raise SurfaceEnergyError('Error with output data. The {} dict '
                                             'should contain a slab, and have '
                                             'keys: "energy", "nsites". No {} '
                                             'is provided. Your keys '
                                             'are: {}'.format(i+1, val, k))


# ============================================================================
# Workflow and Subworkflows Errors
# ============================================================================

class WorkflowError(Exception):
    """ Error in reading the subworkflow parameters.
    """

class EncutConvoError(WorkflowError):
    """ Error in Energy cutoff convergence workflows and dependencies.
    """
    
    @staticmethod
    def entry(entry):
        raise EncutConvoError('Structure has not been found in entry: {}'.format(entry))

class KpointsConvoError(WorkflowError):
    """ Error in Kpoints convergence workflows and dependencies.
    """
    pass

class SlabOptThickError(WorkflowError): 
    """ Error in Slab Optimal Thickness workflows and dependencies.
    """
    pass

class InterfaceMatchError(WorkflowError):
    """ Error in Interface Match workflows and dependencies.
    """

    @staticmethod
    def check_inputs(params):
        try:
            assert len(params['mid']) == 2
            assert len(params['miller']) == 2
        except:
            raise InterfaceMatchError('mid, structure or miller is wrong, '
                                      'a list of length two for them is expected')
        
        if not isinstance(params['formula'], list) or not isinstance(params['name'], list) :
            if params['formula'] is not None or params['name'] is not None:
                raise InterfaceMatchError('formula or name is wrong, if not '
                                          'None, a list of length two is expected')

class RunSimulationPESError(WorkflowError):
    
    @staticmethod
    def check_lateral_shifts(lateral_shifts):
        """ Check if the lateral shifts are present.
        """
        if not lateral_shifts:
            raise SystemExit('Lateral shifts not found in the fw_spec')

class DFTSimulationError(WorkflowError):
    """ General error in performing DFT simulations.
    """
    
    @staticmethod
    def check_types(structure, mid, comp_params):
        
        if isinstance(mid, list):
            n = len(mid)
           
            if not isinstance(mid, list):
                raise DFTSimulationError('Wrong input argument. When `structure` '
                                        'is a list, `mid` should be a list too')
            elif len(mid) != n:
                raise DFTSimulationError('Wrong input argument: `mid` has not '
                                        'the correct length')

            if not isinstance(comp_params, list):
                raise DFTSimulationError('Wrong input argument. When `structure` '
                                        'is a list, `comp_params` should be a list too')
            elif len(comp_params) != n:
                raise DFTSimulationError('Wrong input argument: `comp_params` '
                                        'has not the correct length')

class AdhesionError(WorkflowError):
    """ Error in calculating the adhesion.
    """

class ChargeDispError(WorkflowError):
    """ Errors in calculating the charge displacement
    """
    
    @staticmethod
    def check_integration():
        pass

# ============================================================================
# Errors in Tools Functions, Parameters Handling, and I/O Errors
# ============================================================================

class ReadParamsError(WorkflowError):
    """ Errors when reading, parsing or retrieving data or parameters in FTs.
    """
    pass

class WriteParamsError(Exception):
    """ Errors when writing data to dictionary to be stored in DB.
    """
    pass
