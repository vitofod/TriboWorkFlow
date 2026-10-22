# --- TribChem code vendored by TriboWorkFlow ---------------------------------
# Part of the original file: the statements listed below, copied
# unchanged except for the imports. The rest needs MongoDB,
# FireWorks or atomate and is left out. Do not edit:
# tests/vendoring.py compares them with the original.
# Source: tribchem/highput/database/dbtools.py
# Version: tribchem_devel-align_elisa_w_devel
# Source SHA-256: 216cc46c4b68b36b4399b3064e798020c667ed05e274a3497aa2a20a134c6bc8
# Copied: 2026-10-22
# Licence: CC BY-SA 4.0, see NOTICE
# Kind: partial
# Imports: tribchem.* -> triboworkflow.vendored.*
# Statement lines 3-41: 30b04619bf8e90eb84becc1052d42ebe0fb4bcb9ea22c9de63dd4e649719f7a9
# Statement lines 43-43: 480ba43e60c55749c9ffe31b2b9906dbb00b977d134dc5befe5125b4b5fcbdee
# Statement lines 44-44: 18f4b438f6c0d9f38102ae110dc28d295ee6cbd840321760da10bec4c9fb5c12
# Statement lines 45-45: 3d733e2358e342859da118675966817f4f850202b3e09d30ea3abe443dad2713
# Statement lines 46-46: bbe18a6cf879186198bf336d0cfa99fc835977944198434488461288411f0235
# Statement lines 56-59: c8e897f8e7a761b5acd5a9e0058ee6010afcc7182ed8bbad022e622ef2109614
# Statement lines 67-113: 9cbd2a4bc94b03dc4ff704087d736297aacee46d40612c827578cb74ebba3be7
# Statement lines 115-162: 5477101c12ff6e22a29518e1da4e9efce7fd39ecaae7b7045809240043289610
# --- end of the TriboWorkFlow header -----------------------------------------

"""
Created on Fri Jan 15 15:26:19 2021

This module contains functions to handle dictionaries. They are used to read 
fields that are extracted from a database query and to write dictionaries, 
consistently with the update notation of MongoDB, to be stored in a database.
The module also contains tools to convert images and store them to a database.

The module contains the following functions:

    - read_one_entry
    - read_multiple_entry
    - write_one_entry
    - write_multiple_entry
    - image_bytes_converter
    - create_filter
    - prepare_data_to_update

    Author: Gabriele Losi (glosi000)
    Copyright 2021, Prof. M.C. Righi, TribChem, University of Bologna


    - convert_dict_to_mongodb
    
    Author: Gabriele Losi (glosi000)
    Copyright 2021, Prof. M.C. Righi, TribChem, University of Bologna
    Credits: Code based on StackOverflow, Creative Commons licence 3.0
    https://creativecommons.org/licenses/by-sa/3.0/
    https://stackoverflow.com/questions/29267519/mongodb-update-dictionary-in-document


    - convert_bytes_to_image
    - convert_image_to_bytes

    Author: Gabriele Losi (glosi000)
    Credits: Code based on the original work of Michael Wolloch, TriboFlow package, Wien University
    Copyright 2021, Prof. M.C. Righi, TribChem, University of Bologna

"""

__author__ = 'Gabriele Losi'
__copyright__ = 'Prof. M.C. Righi, University of Bologna'
__contact__ = 'clelia.righi@unibo.it'
__date__ = 'February 15th, 2021'

from triboworkflow.vendored.highput.utils.errors import (
    ReadParamsError,
    WriteParamsError
)


def read_one_entry(input_dict, entry):
    """
    Extract subdictionaries or values from an input dictionary by providing a
    list containing the key to be read one after the other.

    Parameters
    ----------
    input_dict : dict
        Input dictionary to be read.

    entry : str or list
        Contains the key(s) that should be read in sequence from the dictionary.
        The keys should be "innested" within the dictionary or an error raises.
        
        Example:
            1.  entry = 'key'
                function returns input_dict[key]

            2.  entry = ['key1', 'key2', 'key3']
                function returns input_dict['key1']['key2']['key3']

    Returns
    -------
    dict
        Subdictionary containing the desired information.

    """

    # If entry is None, then there is no data to be read
    if entry is None:
        info = None

    # Simply read a dictionary key
    if isinstance(entry, str):
        info = input_dict.get(entry, None)
    
    # You can have multiple nested keys
    elif isinstance(entry, list):
        info = input_dict.copy()
        for n in entry:
            if info is not None:
                info = info.get(n, None)  # Read one key after the other

    else:
        ReadParamsError('Error in reading input_dict, entry is wrong.')

    return info


def read_multiple_entry(input_dict, entry):
    """
    Extract multiple values from an input dictionary by providing a list of
    lists where each element contains the key(s) to be read.

    Parameters
    ----------
    input_dict : dict
        Input dictionary to be read.

    entry : str or list or list of lists
        Contains the key(s) that should be read in sequence from the dictionary.
        The keys should be "innested" within the dictionary or an error raises.
        If a list of lists is passed the dictionary is read multiple times.
        
        Example:
            1.  entry = 'key' or ['key1', 'key2']
                function returns:
                    input_dict[key] or input_dict['key1']['key2']

            2.  entry = [['key1', 'key2'], ['key3']]
                function returns:
                    [ 
                        input_dict['key1']['key2'],
                        input_dict['key3']
                    ]

    Returns
    -------
    dict or list of dicts
        Subdictionary containing the desired information.

    """

    # Extract many info at the same time 
    if isinstance(entry, list):
        if all([isinstance(x, list) for x in entry]):
            info = []
            for n in entry:
                info.append(read_one_entry(input_dict, n))

        else:
            info = read_one_entry(input_dict, entry)
    
    else:
        info = read_one_entry(input_dict, entry)
    
    return info
