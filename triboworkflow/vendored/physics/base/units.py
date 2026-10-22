# --- TribChem code vendored by TriboWorkFlow ---------------------------------
# Unchanged except for the imports. Do not edit: tests/vendoring.py
# compares this file with the original.
# Source: tribchem/physics/base/units.py
# Version: tribchem_devel-align_elisa_w_devel
# Source SHA-256: 2b032f3253ed7136cd22fa7b5214150fbcf9b834c2c13cab46f6309bbf113bfa
# Copied: 2026-10-22
# Licence: CC BY-SA 4.0, see NOTICE
# Kind: whole
# Imports: tribchem.* -> triboworkflow.vendored.*
# --- end of the TriboWorkFlow header -----------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 11:13:09 2020

This module contains a class that will provide all the useful physical 
constants and conversion factors between units

@author: glosi000
"""


class PhysConstants:
    
    # Numerical constants
    k_load = 2.4275587028e7
    mol = 6.02214179e23

    # Energy
    ry_to_eV = 13.6056980659
    J_to_eV = 1.602176565e-19
    ha_to_eV = 27.211396
    ha_to_ry = 2
    kjmol_to_eVA = 0.0103642688
    
    # Lenght
    bohr_to_A = 0.529177208
    
    # Force
    rybohr_to_eVA = 25.71104309541616
    habohr_to_eVA = 51.42208619083232

    # Pressure
    eVA3_to_GPa = 160.21766208

class Converter:

    @staticmethod
    def load2qeforces(load, area, nat=1):
        """
        Calculate the forces (Ry/a.u. units) to be applied to atoms in a pw
        simulation, to realize a certain load on the cell.

        Parameters
        ----------
        load : float
            Starting load of the interface, it must be expressed in GPa.

        area : float
            Base area of the cell of interest, it must be expressed in A^2.

        nat : int, optional
            Number of atoms on which the forces are applied. The default is 1.

        Returns
        -------
        float
            Force in Ry/a.u. unit per atom.
        """

        return (load * 1e9) * (area * 1e-20) * 2.4275587028e7 / nat

    @staticmethod
    def qeforces2load(force, area, nat=1):
        return force * nat / 2.4275587028e7 / (area * 1e-20) / 1e9
    
    @staticmethod
    def ry_to_Jm2(energy, area):        
        return energy * PhysConstants.ry_to_eV * PhysConstants.J_to_eV / area
    
    @staticmethod
    def eV_to_Jm2(energy, area):
        return energy * PhysConstants.J_to_eV / area
