# --- TribChem code vendored by TriboWorkFlow ---------------------------------
# Unchanged except for the imports. Do not edit: tests/vendoring.py
# compares this file with the original.
# Source: tribchem/physics/base/plotter.py
# Version: tribchem_devel-align_elisa_w_devel
# Source SHA-256: 3204913ab76bd75c32bd92903bafe9bb5bc14a090d60982012c663af69aaf18d
# Copied: 2026-10-22
# Licence: CC BY-SA 4.0, see NOTICE
# Kind: whole
# Imports: tribchem.* -> triboworkflow.vendored.*
# --- end of the TriboWorkFlow header -----------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:46:24 2020

Utility tools to calculate the High Simmetry (HS) points for slab and interface

@author: glosi000
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PLOTTING TOOLS
# =============================================================================


def plot_pes(data, lattice, to_fig=None):
    """
    Plot the PES and eventually save it

    """

    a = lattice[0]
    b = lattice[1]
    x = data[0]
    y = data[1]
    E = data[2]

    fact=1.
    level= 43
    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    anglerot='vertical'
    shrin=1.
    zt1=plt.contourf(x, y, E, level, extent=(-fact*a, fact*a, -fact*b, fact*b), cmap=plt.cm.RdYlBu_r)
    cbar1=plt.colorbar(zt1,ax=ax,orientation=anglerot,shrink=shrin)
    cbar1.set_label(r'$E_{adh} (J/m^2)$', rotation=270, labelpad=20,fontsize=15,family='serif')

    ax.quiver(0. , 0., 1., 0.,scale=1.,scale_units='inches',width=0.01,color='white')
    ax.quiver(0. , 0., 0., 1.,scale=1.,scale_units='inches',width=0.01,color='white')
    ax.plot(0.,0.,'w.',ms=7)
    ax.text(0.5,-0.5,'[1 0 1]',rotation='horizontal',color='white', fontsize=14)
    ax.text(-0.5,1.,'[1 2 1]',rotation='vertical',color='white', fontsize=14)
    ax.axis([-fact*min(a), fact*max(a), -fact*min(b), fact*max(b)])
    plt.xlabel(r"distance ($\AA$)",fontsize=12,family='serif')
    plt.ylabel(r"distance ($\AA$)",fontsize=12,family='serif')

    for zt1 in zt1.collections:
       zt1.set_edgecolor("face")
       zt1.set_linewidth(0.000000000001)

    if to_fig != None:
        plt.title("PES for " + str(to_fig), fontsize=18, family='serif')
        plt.savefig('PES_' + str(to_fig) + '.pdf', dpi=300)

    
def plot_uniform_grid(grid, cell, n_a, n_b):
    """
    Plot an uniform grid of n_aXn_b points on the planar base of a lattice 
    
    """
    
    a = cell[0, :]
    b = cell[1, :]
    v = np.cross(a, b)
    
    mod_a = np.sqrt(a[0]**2. + a[1]**2. + a[2]**2.)
    mod_b = np.sqrt(b[0]**2. + b[1]**2. + b[2]**2.)
    A = np.sqrt(v[0]**2. + v[1]**2. + v[2]**2.)
    
    N = n_a * n_b
    density = N / A
    
    # Print information
    print("1st vector:  {:} -> norm: {:.3f}".format(a, mod_a))
    print("2nd vector:  {:} -> norm: {:.3f}".format(b, mod_b))
    print("N pts: {:}   Area: {:.3f}   Density: {:.3f}".format(N, A, density))
    print("\nUniform {0}x{1} grid\n".format(n_a, n_b))
    print(grid)      
    
    # Projection on the plane, top view
    plt.title("Projection on xy plane")
    plt.plot(grid[:, 0], grid[:, 1], 'o')
    
    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(grid[:,0], grid[:,1], grid[:,2], 
               c='r', marker='o', label="3D grid")
    
    # Plot the lattice edge of the plane
    x = [0, a[0], a[0]+b[0], b[0],0]
    y = [0, a[1], a[1]+b[1], b[1],0]
    z = [0, a[2], a[2]+b[2], b[2],0]
    
    ax.plot(x, y, z)
    plt.show()

class BasePlotter():
    def __init__(self):
        pass

class ChargePlotter(BasePlotter):

    def __init__(self):
        pass

    @staticmethod
    def center_interface():
        """
        Method that should center properly the data at zero, in order
        to be able to plot a nice figure
    
        """

        pass
        #n = int(num/2)
        #average = np.mean(data, dtype=np.float, axis=axis)
        #dv = v_len / n
        #v = np.concatenate((v[n:]-max(v)-dv, v[:n])) - shift
        #average = np.concatenate((average[n:], average[:n]))
