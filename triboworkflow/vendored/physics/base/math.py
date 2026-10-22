# --- TribChem code vendored by TriboWorkFlow ---------------------------------
# Unchanged except for the imports. Do not edit: tests/vendoring.py
# compares this file with the original.
# Source: tribchem/physics/base/math.py
# Version: tribchem_devel-align_elisa_w_devel
# Source SHA-256: e31aea7c59f0e20499958936a582fec9009d33dc26980e4509aad7761bccff0a
# Copied: 2026-10-22
# Licence: CC BY-SA 4.0, see NOTICE
# Kind: whole
# Imports: tribchem.* -> triboworkflow.vendored.*
# --- end of the TriboWorkFlow header -----------------------------------------
"""
Created on Fri Nov 20 16:23:41 2020

Useful physics models and math functions to interpolate, integrate, etc...
Try to build an innovative way of calculating the mep, based on GA

@author: glosi000
"""


import numpy as np
#from geneticalgorithm import geneticalgorithm as ga


def rbf_derive_line(rbf, coordx, coordy, m=None, delta=0.01):
    """
    Calculate the x derivative or the y derivative along a given curve. If m
    is not set as None or zero it calculates the derivative 
    
    Calculate the derivative of a straight line for a (x, y) set of points.

    Parameters
    ----------
    x : np.ndarray or list-like
        Abscissa coordinates.

    y : np.ndarray or list-like
        Ordinate coordinates.

    rbf : function or interpolation object
        Surface function to be used. It can be either a standard function
        or an interpolation object behaving like a function, e.g. an
        interpolation with radial basis function rbf = Rbf(x, y).

    m : str or None, optional
        Type derivative. It can be a float (y = mx line) or None (for making a
        derivative along y). The default is None.

    delta : float, optional
        Discretized step to make the derivative. The default is 0.01.
    
    Returns
    -------
    zdev : float
        Value of the derivative along the straight line.

    """
    
    if m is None:  # Derive along y
        coordx_1 = coordx
        coordx_2 = coordx
        coordy_1 = coordy - delta
        coordy_2 = coordy + delta
    elif m < 1e-10 :  # Derive along x
        coordx_1 = coordx - delta
        coordx_2 = coordx + delta
        coordy_1 = coordy
        coordy_2 = coordy
    else:  # Derive along the straight line with slope m
        coordx_1 = coordx - m * delta 
        coordx_2 = coordx + m * delta
        coordy_1 = coordy
        coordy_2 = coordy
    
    # Calculate the derivative
    V_1 = rbf(np.column_stack((coordx_1, coordy_1)))
    V_2 = rbf(np.column_stack((coordx_2, coordy_2)))
    zdev = 0.5 * (V_2 - V_1) / delta

    return zdev

def poly(x, *params):
    return sum([p * (x**i) for i, p in enumerate(params)])

def curve(x, cpoly=(0, 1), csin=None):
    
    if not isinstance(x, (int, float, list, tuple, np.ndarray)):
        try:
            x = np.array(x)
        except: ValueError('x should be a number, a tuple, a list or an array')
    
    if not isinstance(cpoly, list):
        try:
            cpoly = list(cpoly)
        except:
            cpoly = [cpoly]
    
    # Calculate the polynomials part of the function
    f = poly(x, *cpoly)
    
    # Calculate the sinusoidal part of the function
    if csin is not None:
        f += np.sin(x * csin[1]) * csin[0]

    return f
           
# def function(c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, cs1, cs2, rbf, x):
    
#     cpoly = (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10)
#     csins = (cs1, cs2)
#     y = curve(x, cpoly, csins)
#     _, ss = get_shear_strength(np.column_stack((x, y), rbf))
    
#     return ss

# def run_ga(function, dimension, var):
    
#     model = ga(function, dimension, variable_type='real', variable_boundaries=var)
#     model.run()
    
#     return model
