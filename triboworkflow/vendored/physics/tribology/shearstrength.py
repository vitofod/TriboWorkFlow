# --- TribChem code vendored by TriboWorkFlow ---------------------------------
# Unchanged except for the imports. Do not edit: tests/vendoring.py
# compares this file with the original.
# Source: tribchem/physics/tribology/shearstrength.py
# Version: tribchem_devel-align_elisa_w_devel
# Source SHA-256: 2272cfe1abebc81a13f8a59942ea98780b5cf8cbf9dede0e190900cc8fc48c14
# Copied: 2026-10-22
# Licence: CC BY-SA 4.0, see NOTICE
# Kind: whole
# Imports: tribchem.* -> triboworkflow.vendored.*
# --- end of the TriboWorkFlow header -----------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Oct 28 16:44:44 2020

Python functions to get the Shear Strength (SS) of an interface

@author: glosi000
"""

import numpy as np

from triboworkflow.vendored.physics.base.math import rbf_derive_line


# =============================================================================
# EVALUATION OF THE SHEAR STRENGTH
# =============================================================================

class ShearStrength:
    
    @staticmethod
    def get_shear_strength(x, y, rbf, delta=0.01):
        """
        Calculate the shear strength given a path and a potential energy surface.

        Parameters
        ----------
        coords : numpy.ndarray
            Coordinates [x, y] of the path along which you evaluate shear strength.

        rbf : scipy.interpolate.rbf.Rbf
            Contain the information of the interpolation of the potential energy.

        delta : TYPE, optional
            discretized step along x and y for integration. Tuning this value may
            vary slightly the final result. The default is 0.01.

        Returns
        -------
        data_ss_mep : numpy.ndarray
            Profile of potential energy and forces along the MEP.

        ss : float
            The shear strenth along the MEP.

        """

        n = len(x)

        dx = x - np.roll(x,1) 
        dy = y - np.roll(y,1)
        dx[0] = 0.
        dy[0] = 0.
        tx = 0.5 * (np.roll(x, -1) - np.roll(x, 1))
        ty = 0.5 * (np.roll(y, -1) - np.roll(y, 1))    
        # potential computed as integral of projection of gradV on string tangent
        Vz = np.zeros(n)
        #derivative of the potential
        x += delta
        tempValp = rbf(np.column_stack((x,y)))
        x -= 2.*delta
        tempValm = rbf(np.column_stack((x,y)))
        dVx = 0.5*(tempValp - tempValm) / delta
        x += delta
        y += delta
        tempValp = rbf(np.column_stack((x, y)))
        y -= 2. * delta
        tempValm = rbf(np.column_stack((x, y)))
        y += delta
        dVy = 0.5 * (tempValp - tempValm) / delta
        
        #tforce = projection of dV on the local direction of the curve (given by the tangent vector)
        tforce= - (tx * dVx + ty * dVy) #scalar product of the potential gradient and the tangent
        force= tforce / np.sqrt(tx**2 + ty**2) #normalization (la tangente non è un versore, quindi qui lo sta normalizzando)
        
        for i in range(n - 1):
            Vz[i + 1] = Vz[i] - 0.5 * (tforce[i] + tforce[i + 1])
            
        Vz -= np.min(Vz) #min a 0
        Ve = rbf(np.column_stack((x, y)))
        Ve -= np.min(Ve) #min a 0
        lxy  = np.cumsum(np.sqrt(dx**2 + dy**2)) #incrementi sulla retta corrispondono all'incremento effettivo sulla curva
        data_ss = np.stack((lxy, dVx, dVy, Vz, Ve, force), axis=-1)
        
        ss_min = 10.*np.min(force)
        ss_max = 10.*np.max(force)
        ss = max(abs(ss_min), abs(ss_max)) #min force potrebbe essere negativa
        
        # TODO : check what is stored in data_ss_mep and keep only important stuff
        return ss, data_ss
    
    @staticmethod
    def get_shear_strength_xy(cell, rbf, delta=0.01, npoints=101):   
        """
        Calculate the shear strength along the x and y directions of the cell.
        Simplified version of GetShearStrength.
        
        TODO : generalize the function in order to calculate the SS along any 
        straight line
        
        """
    
        #delta = 0.01
        #npoints = 101
    #
        #if params != None and isinstance(params, dict):
        #    for k in params:
        #        if k == 'delta':
        #            delta = params[k]
        #        elif k == 'npoints':
        #            npoints = params[k]   

        cell = np.array(cell)
        alat_x = cell[0, 0]
        alat_y = cell[1, 1]
    
        x = np.linspace(-1.5 * alat_x, 1.5 * alat_x, npoints)
        y = np.linspace(-1.5 * alat_y, 1.5 * alat_y, npoints)   
        zdev_x = np.zeros(len(x))
        zdev_y = np.zeros(len(y))
     
        for i in range(len(x)):
            coordx = x[i]
            coordy = y[i]
            zdev_x[i] = rbf_derive_line(rbf, coordx, coordy, m=0, delta=delta)
            zdev_y[i] = rbf_derive_line(rbf, coordx, coordy, m=None, delta=delta)
    
        #Shear strength in GPa    
        ss_x = np.amax(np.abs(zdev_x)) * 10.0
        ss_y = np.amax(np.abs(zdev_y)) * 10.0
        ss_xy = (ss_x, ss_y)
        print(f'Shear strength along x: {ss_x} GPa, along y: {ss_y} GPa')
        data_ss_xy = np.stack((zdev_x, zdev_y), axis=-1)
    
        return ss_xy, data_ss_xy
