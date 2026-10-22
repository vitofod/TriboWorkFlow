# --- TribChem code vendored by TriboWorkFlow ---------------------------------
# Part of the original file: the statements listed below, copied
# unchanged except for the imports. The rest needs MongoDB,
# FireWorks or atomate and is left out. Do not edit:
# tests/vendoring.py compares them with the original.
# Source: tribchem/highput/utils/tasktools.py
# Version: tribchem_devel-align_elisa_w_devel
# Source SHA-256: 8a86e3450c0787a18cda0b1b6e62835ca0edd25ded8d72a81df5b1fb040107f8
# Copied: 2026-10-22
# Licence: CC BY-SA 4.0, see NOTICE
# Kind: partial
# Imports: tribchem.* -> triboworkflow.vendored.*
# Statement lines 3-22: c39684facd2d2135ae105d78ad12e52c8a7147d751f5f0029eeb52f4cc30b5ed
# Statement lines 24-24: 480ba43e60c55749c9ffe31b2b9906dbb00b977d134dc5befe5125b4b5fcbdee
# Statement lines 25-25: 6539201855057424b7d22fda3676a2416c7b4a41ce569f6526365de46bd2e70c
# Statement lines 26-26: 3d733e2358e342859da118675966817f4f850202b3e09d30ea3abe443dad2713
# Statement lines 27-27: b77481cae0e1bee4ebcabe50adeabf20f7d95acefe89a3fa290741c0bd357033
# Statement lines 30-30: 261a9112a2918e2aba03e19a2f221a027ab4b6f9e946a6c39814f54e18e07dfc
# Statement lines 32-32: bfc1294e2831a4708101009636bb0e78fbe437440b3a644d1e695b0e1b01cecb
# Statement lines 35-35: 24d8d0319d8b483fc0f11efec902382f0ab4bcf64be61141b53347c8ec98eb2f
# Statement lines 39-39: 233f055d94bee86eec442d36ca70ea94e9e4ec0fec37287b3084a7c5dec45a82
# Statement lines 51-59: 23c93e6ee71c3997d080b9ef8cf0ef21a76f87a59fe6d29740cf1cc1f6c442b3
# Statement lines 119-169: 3b561a86f83bd67c1ee923c07a318dd8c7e6e8830d5b9161400d6c5129e6cfa4
# Statement lines 176-189: dc1bb9e07cca8d1f2d207ed5fb29848449e3539a17065be6a1bd74fc08d028f7
# Statement lines 191-196: 14bb764e6e5af633988c80d77649ab7bbe107ec60292daa2af70e4da451e0c12
# Statement lines 198-213: 515e93b6c51ce733d4f920fa61e6c5c792208fa9d46045e4ae9b407d058e71fc
# --- end of the TriboWorkFlow header -----------------------------------------

"""
Created on Mon Feb 22 10:49:54 2021

This module contains core functionalities and features common to almost all
the Firetasks. It provides tools to manage the input parameters of a Firetask
and to manipulate the dictionaries to be retrieved and stored to the database

The module contains the following functions:

    - read_json
    - read_runtask_params
    - read_default_params
    - create_tags
    - get_miller_str
    - select_struct_func

    Author: Gabriele Losi (glosi000)
    Copyright 2021, Prof. M.C. Righi, TribChem, ERC-SLIDE, University of Bologna

"""

__author__ = 'Gabriele Losi'
__copyright__ = 'Copyright 2021, Prof. M.C. Righi, TribChem, ERC-SLIDE, University of Bologna'
__contact__ = 'clelia.righi@unibo.it'
__date__ = 'February 22nd, 2021'

import json
from uuid import uuid4
from pymatgen.core.surface import Structure, Slab
from triboworkflow.vendored.highput.utils.errors import ReadParamsError


def read_json(jsonfile):
    """
    Shortcut to easily read a json file.
        
    """
    
    with open(jsonfile, 'r') as f:
        data = json.load(f)  
    return data


def read_default_params(default_file, default_key, dict_params, allow_unknown=False):
    """
    Read the default argument from a JSON file and compare them with the keys 
    of a dictionary. If some data is missing, is substituted with default
    values. It is a generalization of `read_runtask_params`, because it takes a
    dictionary as input and can be used outside Firetasks.

    Parameters
    ----------
    default_file : str
        Path to the JSON file containing the default values.

    default_key : str
        Key of the JSON dict containing the default parameters.

    dict_params : dict
        Dictionary containing the parameters to be updated with defaults. It
        should be a subset of the dictionary extracted from `default_file`.

    Returns
    -------
    dict
        Final dictionary containing the elements of `dict_params` if present,
        else the default values read from JSON file.

    Raises
    ------
    ReadParamsError
        When unknown keys are present ind `dict_params`.

    """

    # Read the JSON file with defaults and extract the corresponding key
    defaults = read_json(default_file)
    defaults = defaults[default_key]

    # Check if there are some unknown parameters
    if not allow_unknown:
        if not set(dict_params.keys()).issubset(set(defaults.keys())):
            raise ReadParamsError("The values passed as kwargs are not known. "
                                  "Allowed values for '{}' method are: {}"
                                  .format(default_key, list(defaults.keys())))
        params = {}
    else:
        params = dict_params.copy()

    # Set the parameters, missing parameters are substituted with defaults
    for key, value in defaults.items():
        params[key] = dict_params.get(key, value)

    return params


def create_tags(prefix):
    """
    Generate a tag out of a prefix.

    """

    # Create a list of tags
    if isinstance(prefix, list):
        tag = [n + '_' + str(uuid4()) for n in prefix]

    else:
        tag = prefix + '_' + str(uuid4())
    
    return tag


def get_miller_str(miller):
    """
    Convert a miller index from list to string.

    """
    return ''.join(str(s) for s in miller)


def select_struct_func(struct_kind):
    """
    Select a function to work on pymatgen structure, depending on the value
    of `struct_kind` it returns either `pymatgen.core.surface.Structure` or 
    `pymatgen.core.surface.Slab`.

    """

    if struct_kind in ['Structure', 'structure', 'bulk']:
        func = Structure
    elif struct_kind in ['Slab', 'slab', 'interface']:
        func = Slab
    else:
        ValueError("Wrong argument: struct_kind. Allowed values: "
                   "'bulk', 'slab'. Given value: {}".format(struct_kind)) 
    return func
