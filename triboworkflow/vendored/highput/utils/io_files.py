# --- TribChem code vendored by TriboWorkFlow ---------------------------------
# Part of the original file: the statements listed below, copied
# unchanged except for the imports. The rest needs MongoDB,
# FireWorks or atomate and is left out. Do not edit:
# tests/vendoring.py compares them with the original.
# Source: tribchem/highput/utils/io_files.py
# Version: tribchem_devel-align_elisa_w_devel
# Source SHA-256: fde1a226e6aea3eeb18801d257e1af96aef02068b4f03d34c7e2cd8855eaa212
# Copied: 2026-10-22
# Licence: CC BY-SA 4.0, see NOTICE
# Kind: partial
# Imports: tribchem.* -> triboworkflow.vendored.*
# Statement lines 1-1: b4c134e5a1cc263ffc9a67ebd339184432371aa0ecfe0768ec5f83dad1cb4c0f
# Statement lines 6-16: efa57229ae5a8bc53ac52dd655b4c0e38f6e9eb2622c02d39748f29f025f1269
# --- end of the TriboWorkFlow header -----------------------------------------

import os, glob                                                                                                                       


def remove_matching_files(list_of_patterns):                                                                                          
    """Remove all files matching the patterns in the directory."""                                                                    
    remove_list=[]                                                                                                                    
    for pattern in list_of_patterns:                                                                                                  
        remove_list.extend(glob.glob(pattern))                                                                                        
    if remove_list is []:                                                                                                             
        return
    else:
        for file_to_remove in remove_list:
            os.remove(file_to_remove)
        return
