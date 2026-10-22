# --- TribChem code vendored by TriboWorkFlow ---------------------------------
# Unchanged except for the imports. Do not edit: tests/vendoring.py
# compares this file with the original.
# Source: tribchem/physics/tribology/ppes.py
# Version: tribchem_devel-align_elisa_w_devel
# Source SHA-256: a30656816e0d9de5b8aa5e74f122547d6aa748336139c5396945441ff3982af8
# Copied: 2026-10-22
# Licence: CC BY-SA 4.0, see NOTICE
# Kind: whole
# Imports: tribchem.* -> triboworkflow.vendored.*
# --- end of the TriboWorkFlow header -----------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:30:35 2021

Math models to parametrize the Potential Energy Surface.

Author: Gabriele Losi (glosi000)
Copyright 2021, Prof. M.C. Righi, TribChem, University of Bologna

"""


from inspect import signature

import numpy as np
from scipy.optimize import curve_fit


class PPES:
    
    def __init__(self, data):

        self.data = np.array(data)
        self.x = self.data[:, 0]
        self.y = self.data[:, 1]

    def fit(self, method='n', p0=None, maxfev=1000):
        """
        Fit the PPES data with a selected method or a custom function.
        Return the coefficient of the fitting, with the standard deviations.

        Parameters
        ----------
        method : TYPE, optional
            DESCRIPTION. The default is 'n'.
        p0 : TYPE, optional
            DESCRIPTION. The default is None.
        maxfev : TYPE, optional
            DESCRIPTION. The default is 1000.

        Returns
        -------
        popt : list of floats
            Coefficients resulting from the fitting.
    
        perr : float
            One standard deviation error on the obratined coefficients.

        """
        
        f = self.__select_fitting(method)
        p0 = self.__select_p0(f, p0)
        
        popt, perr = self.__fit(method, f, p0, maxfev)
        
        return popt, perr

    def __select_fitting(self, method):
        """
        Select the fitting option based on the provided method from input.

        """
        
        # Check if method is a custom function
        if hasattr(method, '__call__'):
            return method

        # Else try to infer the desired fitting function
        else:
            if method in ['potential_n', 'ppes_n', 'n']:
                func = PPESModel.potential_n
            elif method in ['potential_n4', 'ppes_n4', 'n4']:
                func = PPESModel.potential_n4
            elif method in ['uber', 'u']:
                func = PPESModel.uber
            elif method in ['uber0', 'u0']:
                func = PPESModel.uber0
            else:
                raise ValueError('Wrong input argument: method. Allowed '
                                 'values: n, n4, u, u0.')
        return func

    def __select_p0(self, f, p0):
        
        params = list(signature(f).parameters)
        n = len(params) - 1
        
        if p0 is not None:
            if len(p0) != n:
                raise ValueError('Wrong number of guessing parameters. It must '
                                 'be: len(p0) = len(kwargs) -1 for a function '
                                 'of the kind f(x, *kwargs). You provided p0 = '
                                 '{} for a function with kwargs = {}'.format(
                                 p0, params[1:]))
        else:
            p0 = np.array([1] * n)
        
        return p0

    def __fit(self, method, f, p0, maxfev):
        
        if method in ['uber', 'uber0', 'u', 'u0']:
            
            # Get the correct elements to make the fit
            fit_data = self.data.copy()[self.data[:, 0] >= 0.0, :]
            z = fit_data[:, 0]
            e = fit_data[:, 1] - min(fit_data[:, 1])

            # Set correctly the initial p0
            p0[:2] = np.array([-min(e), 0.75])

        else:
            z = self.x.copy()
            e = self.y.copy()
        
        # Make the fit        
        popt, pcov = curve_fit(f, z, e, p0=p0, maxfev=maxfev)
        
        # Calculate the standard deviation for each fit coefficient
        perr = np.sqrt(np.diag(pcov))
        
        return popt, perr

class PPESModel(PPES):
    """
    Collection of functions to be used to fit a ppes

    """

    @staticmethod
    def potential_n(z, c0, c1, c2, n):
        return c0 * np.exp(-z * c1) - c2 / np.power(z, n)

    @staticmethod
    def potential_n4(z, c0, c1, c2):
        return c0 * np.exp(-z * c1) - c2 / np.power(z, 4)
    
    @staticmethod
    def uber(x, g, l, x0=0):
        """
        Universal binding energy relation, to link potential energy surface
        with 
    
        Parameters
        ----------
        x : float or array of float
            Distance(s).
    
        g : float
            Binding energy at UBER minimum.
    
        l : float
            Critcal length (location of inflection point).
        
        x0 : float, optional
            Position of the minimum of the curve.
    
        Returns
        -------
        float
            UBER binding energy at position(s) x.

        """
        return g * (1 - (1 + (x - x0)/l) * np.exp(-(x - x0)/l))
    
    @staticmethod
    def uber_0(x, g, l):
        """ UBER curve, simplified for data shifted on zero. See uber method.
        """        
        return g * (1 - (1 + x/l) * np.exp(-x/l))
    

# =============================================================================
# Tools for the PES
# =============================================================================

def fit_ppes(values, fit='uber'):
    """
    Perform the fit of the PPES.
    
    Parameters
    ----------
    data : list
        Array containing the energy at the several distances.
    
    Return
    ------
    fit_dict : dictionary containing the dict
    
    """
    
    # Old calculation
    # d_E_array=np.asarray(distance_energy_array)
    # Fit_data = d_E_array[d_E_array[:,0] >= 0.0,:]
    # z = Fit_data[:,0]
    # E = Fit_data[:,1]-min(Fit_data[:,1])
    # Initial_guess = [-min(E),0.75]
    # popt, pcov = curve_fit(UBER, z, E, p0=Initial_guess)
    # perr = np.sqrt(np.diag(pcov))
    # TODO: double check if here is correct, and delete the previous lines.
    
    data = np.array(values)

    # Prepare the data to perform the fit
    if fit == 'uber':
        fit_data = data[data[:, 0] >= 0.0, :]
        solver = PPES(fit_data)
        p0 = [-min(fit_data[:, 1]), 0.75, 0]
    elif fit == 'uber0':
        fit_data = data[data[:, 0] >= 0.0, :]
        solver = PPES(fit_data)
        e = fit_data[:, 1] - min(fit_data[:, 1])
        p0 = [-min(e), 0.75]

    # Execute the fit
    popt, perr = solver.fit(method=fit, p0=p0)
    
    if 'uber' in fit:
        fit_dict = {
                'g': popt[0],
                'l': popt[1],
                'sigma_g': perr[0],
                'sigma_l': perr[1]
            }
    
    return fit_dict
