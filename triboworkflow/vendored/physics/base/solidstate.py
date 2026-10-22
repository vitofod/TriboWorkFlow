# --- TribChem code vendored by TriboWorkFlow ---------------------------------
# Unchanged except for the imports. Do not edit: tests/vendoring.py
# compares this file with the original.
# Source: tribchem/physics/base/solidstate.py
# Version: tribchem_devel-align_elisa_w_devel
# Source SHA-256: 9742c6e5690f43f8b4a60e25a9b697baa23262ead8191787385ff536528a249c
# Copied: 2026-10-22
# Licence: CC BY-SA 4.0, see NOTICE
# Kind: whole
# Imports: tribchem.* -> triboworkflow.vendored.*
# --- end of the TriboWorkFlow header -----------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Oct  7 15:46:24 2020

Functions to manage crystalline structures and solid state physics.

The module contains the following functions:
    - orient_bulk
    - generate_slabs

    Author: Gabriele Losi (glosi000)
    Copyright 2021, Prof. M.C. Righi, TribChem, ERC-SLIDE, University of Bologna

"""


__author__ = 'Gabriele Losi'
__copyright__ = 'Copyright 2021, Prof. M.C. Righi, TribChem, ERC-SLIDE, University of Bologna'
__contact__ = 'clelia.righi@unibo.it'
__date__ = 'October 7th, 2021'


import math as m
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from pymatgen.core.structure import Structure, Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from triboworkflow.vendored.physics.base.shaper import transfer_average_magmoms
from triboworkflow.vendored.physics.base.shaper import Shaper
from triboworkflow.vendored.physics.base.units import PhysConstants
from triboworkflow.vendored.physics.base.plotter import plot_uniform_grid
#from tribchem.physics.tribology.highsym import dict_converter_hs


# =============================================================================
# Work on lattice cells
# =============================================================================

def get_cell_qe(ibrav, celldm, cell=None, is_bohr=True):
    """
    Calculate the lattice cell for a crystalline material given the typical
    QE notation for the cells. It returns volume, area, and lattice cell.
    
    Information about QE implementation: shorturl.at/kyEOQ
    
    TODO : 1) Generalize to any ibrav, i.e. give the possibility to celldm to be
    an array of either 3 or 5 elements. 2) Check if there's the possibility to 
    insert any possible unit of measure (bohr, angstrom, crystal, alat).

    Parameters
    ----------
    ibrav : int
        The Bravais-lattice index as defined on the pw input page.
        
    celldm : list or numpy.ndarray
        Array-like object containing the celldm vector from a pwi input.
        See the Quantum Espresso documentation for more information about it.
        For the current implementation, when ibrav!=0, it is a 3-element list.
        
    cell : list or numpy.ndarray, optional
        Lattice cell of the system, required when ibrav=0. The default is None.
    
    is_bohr : bool, optional
        Determine whether alat (celldm[0]) is bohr or angstrom.
        The default is True.

    Returns
    -------
    cell : np.array
        The lattice cell of the system.
    
    area : float
        The area of the cell.
        
    volume : float
        The volume of the cell.

    """
    
    if is_bohr:
        bohr_to_angstrom = PhysConstants.bohr_to_A
    else:
        bohr_to_angstrom = 1
        
    alat = celldm[0] * bohr_to_angstrom

    if ibrav == 0:
        if not cell:
            raise IOError('If ibrav=0, a lattice cell (alat) have to be provided')
        
        else:
            a = cell[0, :] * alat
            b = cell[1, :] * alat
            c = cell[2, :] * alat
            area = np.linalg.norm( np.cross(a, b) )
            volume = np.abs( np.dot( (np.cross(a, b)), c ) )
            
            return cell, area, volume
    
    else:
        
        if not ibrav or not celldm:
            raise IOError('ibrav or celldm are not provided correctly')
        
        # Check type and manipulate data as needed
        if isinstance(celldm, list):
            celldm = np.array(celldm)
        
        b_a = celldm[1]
        c_a = celldm[2]
        cos_ab = celldm[3]
        
        # For triclinic cell
        #cos_bc = celldm[3]
        #cos_ac = celldm[4]
        #cos_ab = celldm[5]
        
        # Simple cubic
        if ibrav == 1:
            a = np.array([1, 0, 0]) * alat
            b = np.array([0, 1, 0]) * alat
            c = np.array([0, 0, 1]) * alat
        
        # Hexagonal and Trigonal P
        elif ibrav == 4:
            a = np.array([1, 0, 0]) * alat
            b = np.array([-0.5, m.sqrt(3)/2, 0]) * alat
            c = np.array([0, 0, c_a]) * alat
        
        # Tetragonal
        elif ibrav == 6:
            a = np.array([1, 0, 0]) * alat
            b = np.array([0, 1, 0]) * alat
            c = np.array([0, 0, c_a]) * alat
            
        # Orthorombic
        elif ibrav == 8:
            a = np.array([1, 0, 0]) * alat
            b = np.array([0, b_a, 0]) * alat
            c = np.array([0, 0, c_a]) * alat
        
        # Orthorhombic base-centered(bco)
        elif ibrav==9:
            a = np.array([ 0.5, 0.5*b_a, 0])*alat
            b = np.array([-0.5, 0.5*b_a, 0])*alat
            c = np.array([0, 0, c_a]) * alat
        
        # Orthorhombic base-centered(bco) - alternative description
        elif ibrav == -9:
            a = np.array([0.5, -0.5*b_a, 0]) * alat
            b = np.array([0.5,  0.5*b_a, 0]) * alat
            c = np.array([0, 0, c_a]) * alat
        
        # Monoclinic P, unique axis c
        elif ibrav == 12:
            sin_ab = m.sin( m.acos(cos_ab) )
            a = np.array([1, 0, 0]) * alat
            b = np.array([b_a*cos_ab, b_a*sin_ab, 0]) * alat
            c = np.array([0, 0, c_a]) * alat
    
    cell = np.vstack([a, b, c])
    area = np.linalg.norm( np.cross(a, b) )
    volume = np.abs( np.dot( (np.cross(a, b)), c ) )
    
    
    return cell, area, volume

def get_cell_info(cell):
    """
    Calculate the area and volume of 3x3 cell in unit Angstrom.

    """

    # Convert to numpy and assign lattice vectors
    a, b, c = np.array(cell)
    
    # Calculate area and volume
    area = np.linalg.norm( np.cross(a, b) )
    volume = np.abs( np.dot( (np.cross(a, b)), c ) )
    
    return area, volume

def reciprocal_cell(cell):
    """
    Calculate the reciprocal lattice vectors of a given cell

    Parameters
    ----------
    cell : np.ndarray or list
        Lattice cell of interest.

    Returns
    -------
    

    """
    
    redv = 2 * np.pi / np.dot(cell[0, :], np.cross(cell[1, :], cell[2, :]))
    
    b0 = redv * np.cross(cell[1, :], cell[2, :])
    b1 = redv * np.cross(cell[2, :], cell[0, :])
    b2 = redv * np.cross(cell[0, :], cell[1, :])
    
    return np.vstack((b0, b1, b2))

def reciprocal_cell_2d(cell):
    
    q = np.array([[0, -1], [1, 0]])
    
    b0 = 2*np.pi * np.dot(q, cell[1, :]) / np.dot(cell[0, :], np.dot(q, cell[1, :]))
    b1 = 2*np.pi * np.dot(q, cell[0, :]) / np.dot(cell[1, :], np.dot(q, cell[0, :]))
    
    return np.vstack((b0, b1))
    

# =============================================================================
# TOOLS FOR PERIODIC BOUNDARY CONDITIONS
# =============================================================================

def pbc_coordinates(data, cell, to_array=True, scaled_positions=False):
    """
    Apply Periodic Boundary Conditions to a set of atoms (data) in a given 
    lattice cell (cell). PBC are applied by creating a "fake" molecule.
    Return a numpy array containing the atomic sites within cell.

    """
    
    # Check the types of the input parameters
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("data must be a numpy array or a list")
    if not isinstance(cell, (list, np.ndarray)):
        raise TypeError("cell must be a numpy array or a list")
    
    # Manage the situation where you provide only x, y coordinates
    data = np.array(data)
    two_col = False
    pbc = [1, 1, 1]
    if data.shape[1] == 2:
        two_col = True
        pbc = [1, 1, 0]
        data = np.column_stack([data, np.zeros(len(data))])
    
    # Create a fake atomic structures and apply PBC
    if scaled_positions:
        atoms_fake = Atoms(scaled_positions=data, cell=cell, pbc=pbc)
    else:
        atoms_fake = Atoms(positions=data, cell=cell, pbc=pbc)
        
    # Get the coordinates
    data_new = atoms_fake.get_positions(wrap=True, pbc=True)
    if two_col:
        data_new = data_new[:, :2]
    
    if not to_array:
        data_new = data_new.tolist()

    return data_new  


# =============================================================================
# TOOLS TO REPLICATE POINTS IN A LATTICE CELL
# =============================================================================

def replicate(data, cell, replicate_of=(1, 1, 1), symm=False, to_list=False):
    """
    Replicate a set of points or atomic sites in a (n,m,l)-size lattice cell.
    Only work if data and cell are provided in Angstrom units.
    
    TODO: FIX A BUG. When symm is True, not-unique points are generated and 
    you need to call rm_duplicates_2d

    Parameters
    ----------
    data : list or numpy.ndarray
        (n, 3) matrix, of array-like type, containing the coordinates.
        The shape of data can be (n, m), with m>3. In that case, the other 
        columns are assumed to contain physical quantities that you want to 
        track during the replica of the points.
        
        Format:
            x0  y0  z0
            y1  y1  z1
            .   .   .
            .   .   .
            .   .   .
        
    cell : list or numpy.ndarray
        Initial lattice cell to be duplicated. Must be an array-like type
        
    replicate_of : tuple, optional
        Number of times to replicate along x, y, z. The default is (1, 1, 1).
        
    symm : bool, optional
        To have a more symmetric coordinates and cell, centered around the
        origin. Ex. n=replicate_of[0] and symm=True, replicates of (-n,n) in x.
        The default is False.
    
    to_list : bool, optional
        Whether to return the replicated data as lists. The default is False.
    
    units : string, optional
        To be implemented. The default is angstrom.

    Returns
    -------
    data_new : list or numpy.ndarray
        Replicated coordinates.
    
    cell_new : list or numpy.ndarray
        Replicated cell.

    """
    
    # Check wether the number inserted are correct
    le = len(replicate_of)
    n = int(replicate_of[0]) if le >= 1 else 1
    m = int(replicate_of[1]) if le >= 2 else 1
    l = int(replicate_of[2]) if le == 3 else 1
    if n<=0: n=1
    if m<=0: m=1
    if l<=0: l=1
    
    # Data and cell are not replicated if the indexes are 1
    if (n, m, l) == (1, 1, 1) and not symm:
        return data, cell
    
    # Initialize coordinates and cells
    else: 
        cell = np.array(cell)
        a = cell[0, :]
        b = cell[1, :]
        c = cell[2, :]
        
        data = np.array(data)
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        
        x_new = np.array([])
        y_new = np.array([])
        z_new = np.array([])
        
        # Check for other physical quantities and initialize them 
        n_col = data.shape[1]
        if n_col > 3:
            e = data[:, 3:]
            e_new = np.array([])

        for i in range(-n * int(symm), n):
                for j in range(-m * int(symm), m):
                    for k in range(-l * int(symm), l):

                        # Replicate the x-, y-, z- coordinates
                        x_add = x + a[0] * i + b[0] * j + c[0] * k
                        y_add = y + a[1] * i + b[1] * j + c[1] * k
                        z_add = z + a[2] * i + b[2] * j + c[2] * k
                        
                        # Collect coordinates and other physical quantities
                        x_new = np.append(x_new, x_add)
                        y_new = np.append(y_new, y_add)
                        z_new = np.append(z_new, z_add)
                        if n_col > 3:
                            e_new = np.append(e_new, e)
    
    data_new = np.column_stack([x_new, y_new, z_new])
    if n_col > 3:
        data_new = np.column_stack([data_new, e_new])
    
    cell_new = np.vstack([a * n, b * m, c * l])
        
    if to_list:
        data_new = data_new.tolist()
        cell_new = cell_new.tolist()

    return data_new, cell_new

def replicate_2d(data, cell, replicate_of=(1, 1), symm=False, to_list=False):
    """
    Replicate a set of points or atomic sites in a 2D (n,m)-size lattice cell.
    The data contains x, y coordinates to replicate and may also contain other
    columns with physical quantities to be tracked with the replicated points. 
    The cell is a 2D array like. Useful to replicate data and interpolate PES.
    
    Format:   
        data:
            x0  y0  E0  Y0  ... 
            y1  y1  E1  Y1  ...
            .   .   .   .
            .   .   .   .
            .   .   .   .
            
        cell:
            a0  a1
            b0  b1
            
    """
    
    # Check wether the number inserted are correct
    le = len(replicate_of)
    n = int(replicate_of[0]) if le >= 1 else 1
    m = int(replicate_of[1]) if le >= 2 else 1
    if n<=0: n=1
    if m<=0: m=1
    
    # Data and cell are not replicated if the indexes are 1
    if n == 1 and m == 1 and not symm:
        return data, cell
    
    # Check the cell dimension and transform it into a 3x3 cell
    cell = zfill_cell(cell)
    
    # Handle data and insert a column of zeros for the z-coordinates
    data = np.array(data)
    if len(data.shape) == 2:
        if data.shape[1] >= 2:
            data = np.insert(data, 2, values=np.zeros(data.shape[0]), axis=1)
    
    # Replicate the data
    data_new, cell = replicate(data, cell, replicate_of=(n, m, 1), 
                               symm=symm, to_list=to_list)
    data_new = np.delete(data_new, 2, 1)
        
    return data_new, cell

def generate_uniform_grid(cell, density=1, pts_a=None, to_plot=False):
    """
    Generate a 2D-uniform grid of points with a given density on the
    basal lattice plane of a cell (a X b), i.e. lattice[0,:] X lattice[1,:].
    You can set a uniform density or provide the points along a.

    Parameters
    ----------
    lattice : numpy.ndarray
        Vectors of the lattice cell. A uniform grid of points is generated on
        the surface spanned by the first and second vector, i.e. a X b.
        lattice shape is (2, 3) or (3, 3); lattice is in Angstrom units.

    density : float, optional
        Density of the grid of points that will cover the planar surface of 
        the lattice cell. Units: number of points per unit Angstrom^2
        
    pts_a : int, optional
        If this value is provided, the grid will contain pts_a points along 
        the first vector and (b/a)*pts_a along the second vector. a and b are 
        the lengths of the planar lattice vectors. The default is None.
                
    to_plot : bool, optional
        Wether to display the grid of points inside the lattice cell. 
        Plot is redirected to standard output. The default is False.

    Returns
    -------
    matrix : numpy.ndarray
        Grid of points spanning the entire lattice plane.
        
        Format:
            
            x0  y0  z0
            y1  y1  z1
            .   .   .
            .   .   .
            .   .   .

    """
        
    a = cell[0, :]
    b = cell[1, :]
    a_mod = np.sqrt(a[0]**2. + a[1]**2. + a[2]**2.)
    b_mod = np.sqrt(b[0]**2. + b[1]**2. + b[2]**2.)
    ratio = b_mod/a_mod
    
    # Calculate the number of points for each lattice vector
    if pts_a == None:
        N_tot = round(density * a_mod * b_mod)
        n_a = int(round( np.sqrt( N_tot/ratio )))
        n_b = int(round( ratio*n_a ))
    else:
        n_a = pts_a
        n_b = int(round( ratio*n_a ))
    
    # Obtain the displacements along a and b
    dist_a_x = a[0]/n_a 
    dist_a_y = a[1]/n_a
    dist_a_z = a[2]/n_a
    dist_b_x = b[0]/n_b
    dist_b_y = b[1]/n_b
    dist_b_z = b[2]/n_b
    
    # Create the grid
    matrix = np.zeros((n_a*n_b, 3))
    k = 0
    for i in range(0, n_a):
        for j in range(0, n_b):
            matrix[k, 0] = i*dist_a_x + j*dist_b_x
            matrix[k, 1] = i*dist_a_y + j*dist_b_y
            matrix[k, 2] = i*dist_a_z + j*dist_b_z
            k += 1
    if to_plot:
        plot_uniform_grid(matrix, cell, n_a, n_b)

    return matrix

def zfill_cell(cell):
    """
    Fill an uncomplete lattice cell and return a 3x3 cell.

    """

    cell = np.array(cell)
    
    if cell.shape == (3, 3):
        return cell

    elif cell.shape == (2, 2):
        return np.array([[cell[0, 0], cell[0, 1], 0], 
                         [cell[1, 0], cell[1, 1], 0], 
                         [0, 0, 1]])

    elif cell.shape == (2,):
        return np.array([[cell[0], 0, 0], 
                         [0, cell[1], 0], 
                         [0, 0, 1]])

    elif cell.shape == (3,):
        return np.array([[cell[0], 0, 0],
                         [0, cell[1], 0],
                         [0, 0, cell[2]]])


# =============================================================================
# TOOLS TO MODIFY THE LATTICE CELL
# =============================================================================

def orthorombize_cell(data, cell, tol=1e-4, to_plot=True):
    """
    At the moment this function is a clone of orthorombize_2d, but generalized
    to orthorombize the basis of any cell, providing the xyz coordinates of
    the atoms in the cell and the 3D cell. 
    Use orthorombize_2d only for the PES! While this should be used just for cells.

    """

    # Select the cell
    a = cell[0, :]
    b = cell[1, :]
    c_back = cell[2, :]

    # Create a rectangular cell out of a general 2d shape
    if np.sign(a[0]) == np.sign(b[0]):
        if a[0] > 0:
            x_up = a[0] + b[0]
            x_dw = 0
        else:
            x_up = 0
            x_dw = a[0] + b[0]
    else:
        x_up =  max(a[0], b[0])
        x_dw =  min(a[0], b[0])
    if np.sign(a[1]) == np.sign(b[1]):
        if a[1] > 0:
            y_up = a[1] + b[1]
            y_dw = 0
        else:
            y_up = 0
            y_dw = a[1] + b[1]
    else:
        y_up =  max(a[1], b[1])
        y_dw =  min(a[1], b[1])    

    # Prepare the final cell by placing all the data in the origin (0, 0)
    data_ort, _ = replicate_2d(data, cell, replicate_of=(3, 3), symm=True)
    
    plt.plot([0, cell[0, 0], cell[0, 0]+cell[1, 0], cell[1, 0], 0], 
             [0, cell[0, 1], cell[0, 1]+cell[1, 1], cell[1, 1], 0])
    plt.plot(data_ort[:, 0], data_ort[:, 1], 'o')
    plt.show()

    # Replicate the points and fill a rectangular cell, remove duplicates
    is_inside_x = (data_ort[:, 0] <= (x_up + tol)) * (data_ort[:, 0] >= (x_dw - tol))
    is_inside_y = (data_ort[:, 1] <= (y_up + tol)) * (data_ort[:, 1] >= (y_dw - tol))
    orthorombic = rm_duplicates(data_ort[is_inside_x * is_inside_y])

    # Orthorombize the cell and shift the data to (0, 0)
    cell_2d = np.array([[x_up, y_dw], [x_dw, y_up]])
    if cell_2d[0, 1] != 0:
        y = cell_2d[0, 1]
        cell_2d[:, 1] -= y
        orthorombic[:, 1] -= y
    if cell_2d[1, 0] != 0:
        x = cell_2d[1, 0]
        cell_2d[:, 0] -= x
        orthorombic[:, 0] -= x
    
    # Create a 3D cell and apply PBC to the orthorombized cell
    cell = cell_2d.copy()
    cell = np.column_stack((cell, np.zeros(2)))
    cell = np.vstack((cell, c_back))

    # Catch and remove the replicated atoms on the edges
    index = catch_replica_orthorombic_2d(orthorombic, cell, tol)
    rm_list = []
    for i in index:
        rm_list.append(min(i))
    orthorombic = np.delete(orthorombic, rm_list, axis=0)
    
    if to_plot:
        d, _ = replicate_2d(orthorombic, cell, replicate_of=(1, 1), symm=False)
        plt.plot([0, cell[0, 0], cell[0, 0]+cell[1, 0], cell[1, 0], 0], 
                 [0, cell[0, 1], cell[0, 1]+cell[1, 1], cell[1, 1], 0])
        plt.plot(d[:, 0], d[:, 1], 'o')
        plt.show()

    return orthorombic, cell_2d

def orthorombize_2d(data, cell, tol=1e-4, to_plot=True):
    """
    Take the replicated points of the pes and cut them in a squared shape.
    TODO : Improve the code and VECTORIZE

    """

    # Select the cell
    a = cell[0, :]
    b = cell[1, :]

    # Create a rectangular cell out of a general 2d shape
    if np.sign(a[0]) == np.sign(b[0]):
        if a[0] > 0:
            x_up = a[0] + b[0]
            x_dw = 0
        else:
            x_up = 0
            x_dw = a[0] + b[0]
    else:
        x_up =  max(a[0], b[0])
        x_dw =  min(a[0], b[0])
    if np.sign(a[1]) == np.sign(b[1]):
        if a[1] > 0:
            y_up = a[1] + b[1]
            y_dw = 0
        else:
            y_up = 0
            y_dw = a[1] + b[1]
    else:
        y_up =  max(a[1], b[1])
        y_dw =  min(a[1], b[1])    

    # Prepare the final cell by placing all the data in the origin (0, 0)
    data_ort, _ = replicate_2d(data, cell, replicate_of=(3, 3), symm=True)
    
    plt.plot([0, cell[0, 0], cell[0, 0]+cell[1, 0], cell[1, 0], 0], 
             [0, cell[0, 1], cell[0, 1]+cell[1, 1], cell[1, 1], 0])
    plt.plot(data_ort[:, 0], data_ort[:, 1], 'o')
    plt.show()

    # Replicate the points and fill a rectangular cell, remove duplicates
    is_inside_x = (data_ort[:, 0] <= (x_up + tol)) * (data_ort[:, 0] >= (x_dw - tol))
    is_inside_y = (data_ort[:, 1] <= (y_up + tol)) * (data_ort[:, 1] >= (y_dw - tol))
    orthorombic = rm_duplicates_2d(data_ort[is_inside_x * is_inside_y])

    # Orthorombize the cell and shift the data to (0, 0)
    cell_2d = np.array([[x_up, y_dw], [x_dw, y_up]])
    if cell_2d[0, 1] != 0:
        y = cell_2d[0, 1]
        cell_2d[:, 1] -= y
        orthorombic[:, 1] -= y
    if cell_2d[1, 0] != 0:
        x = cell_2d[1, 0]
        cell_2d[:, 0] -= x
        orthorombic[:, 0] -= x
    
    # Create a 3D cell and apply PBC to the orthorombized cell
    cell = cell_2d.copy()
    cell = np.column_stack((cell, np.zeros(2)))
    cell = np.vstack((cell, np.array([0, 0, 10])))

    # Catch and remove the replicated atoms on the edges
    index = catch_replica_orthorombic_2d(orthorombic, cell, tol)
    rm_list = []
    for i in index:
        row1 = orthorombic[min(i)]
        row2 = orthorombic[max(i)]
        if not abs(row1[2] - row2[2]) < tol:
             raise ValueError('Energies of replicated atoms are different, but'
                              ' they should not be')
        rm_list.append(min(i))
    orthorombic = np.delete(orthorombic, rm_list, axis=0)
    
    if to_plot:
        d, _ = replicate_2d(orthorombic, cell, replicate_of=(1, 1), symm=False)
        plt.plot([0, cell[0, 0], cell[0, 0]+cell[1, 0], cell[1, 0], 0], 
                 [0, cell[0, 1], cell[0, 1]+cell[1, 1], cell[1, 1], 0])
        plt.plot(d[:, 0], d[:, 1], 'o')
        plt.show()

    return orthorombic, cell_2d

def catch_replica_orthorombic_2d(data, cell, tol=1e-4):

    index_set = []
    a = cell[0, 0]
    b = cell[1, 1]

    for i, row1 in enumerate(data):
        x1, y1 = np.abs(row1[0:2])
        data_check = np.delete(data, (i), axis=0)
            
        for j, row2 in enumerate(data_check):
            x2, y2 = np.abs(row2[0:2])           
            
            # Consider the different statement for a rectangular shape
            if (x1 < tol) and (y1 < tol):  # (0, 0)
                state = ((abs(x2 - a) < tol) and ((y2 < tol) or (abs(y2 - b) < tol))) or\
                        ((x2 < tol) and (abs(y2 - b) < tol))
            
            elif (x1 < tol) and (abs(y1 - b) < tol):
                state = ((abs(x2 - a) < tol) and ((y2 < tol) or (abs(y2 - b) < tol))) or\
                        ((x2 < tol) and (y2 < tol))

            elif (abs(x1 - a) < tol) and (y1 < tol): #ok
                state = ((x2 < tol) and ((y2 < tol) or (abs(y2 - b) < tol))) or\
                        ((abs(x2 - a) < tol) and (abs(y2 - b) < tol))
            
            elif (abs(x1 - a) < tol) and (abs(y1 - b) < tol):
                state = ((x2 < tol) and ((y2 < tol) or (abs(y2 - b) < tol))) or\
                        ((abs(x2 - a) < tol) and (y2 < tol))

            elif x1 < tol:
                state = ((abs(y2 - y1) < tol) and (abs(x2 - a) < tol))
            
            elif abs(x1 - a) < tol:
                state = ((abs(y2 - y1) < tol) and (x2 < tol))
            
            elif y1 < tol:
                state = ((abs(x2 - x1) < tol) and (abs(y2 - b) < tol))
            
            elif abs(y1 - b) < tol:
                state = ((abs(x2 - x1) < tol) and (y2 < tol))

            else:
                state = False
            
            if state:
                ind = np.where(np.all(data==row2,axis=1))[0].tolist()
                for k in ind:
                    index_set.append(set([i, k]))

    index = np.unique(index_set)

    return index.tolist()

def rm_duplicates(data, rounding_decimal=5):
    """
    Remove the duplicates of a set of x, y, z coordinates and round them up.

    """
    xyz = np.round(data[:, :3], decimals = int(rounding_decimal))
    xyz_unique = np.unique(xyz, axis=0)
    
    return xyz_unique

def rm_duplicates_2d(data, rounding_decimal=5):
    """
    Round a list of data points to a certain number of rounding decimals and
    remove the duplicate rows

    Parameters
    ----------
    data : np.ndarray
        Data matrix to be checked and cleaned for duplicates rows.

    rounding_decimal : int or float, optional
        Number of significative numbers to be kept in data. The default is 5.

    Returns
    -------
    np.ndarray
        Data cleaned from duplicates.

    """
        
    xy = np.round(data[:, :2], decimals = int(rounding_decimal))
    e_dict = {}
    for i in xy:
        duplicates = np.where((xy == i).all(axis = 1))[0]
        e_list = []
        for j in duplicates:
            e_list.append(data[j, 2])
        e_dict[str(i)]=np.mean(e_list)
    xy_unique = np.unique(xy, axis=0)
    e = []
    for i in xy_unique:
        e.append([e_dict[str(i)]])
    
    return np.hstack((xy_unique, np.array(e)))


# ============================================================================
# Functions to deal with crystalline slabs
# ============================================================================

def orient_bulk(structure, miller, thickness, primitive=False, lll_reduce=False, 
                in_unit_planes=True):
    """
    Orient a bulk unit cell along a direction identified by Miller indexes.

    """

    # Generate the oriented bulk
    slabgen = SlabGenerator(initial_structure=structure,
                            miller_index=miller,
                            primitive=primitive,
                            lll_reduce=lll_reduce,
                            in_unit_planes=in_unit_planes,
                            min_slab_size=thickness,
                            min_vacuum_size=0)
    
    bulk_miller = slabgen.oriented_unit_cell

    return bulk_miller

def atom_in_a_box(species, alat=15):
    """
    Build a box containing a single atomic species, placed in the center of it.

    Parameters
    ----------
    species : list of str or str
        List of strings or a string containing the species to be used.

    alat : int or float, optional
        Edge of the cubic box to be constructed. The default is 40.

    Returns
    -------
    box : list of pymatgen.core.structure.Structure
        List of structures containing the molecules in a box.

    """

    # Convert species to a list
    if not isinstance(species, list):
        species = [species]

    # Create the big lattice cell
    lattice = Lattice.from_parameters(a=alat, b=alat, c=alat, 
                                      alpha=90, beta=90, gamma=90)

    # Cycle over the species and construct a structure
    box = [0] * len(species)
    for i, atom in enumerate(species):
        box[i] = Structure(lattice, [str(atom)], [[0.5, 0.5, 0.5]])
    
    return box

def generate_slabs(structure, miller, thickness, vacuum, thick_bulk=12,
                   center_slab=True, primitive=True, lll_reduce=True,
                   in_unit_planes=True, ext_index=0, bonds=None, ftol=0.1, 
                   tol=0.1, repair=False, max_broken_bonds=0, symmetrize=False):
    """
    Create and return a single slab or a list of slabs out of a structure.
    To return a list of slabs you need to provide `miller` as a list of lists,
    while `vacuum` and `thickness` can be either a single value (if you want all
    the slabs to have same thickness and vacuum) or a list of values.
    If `thickness` is 0, an oriented bulk is returned. The other values are
    passed as arguments to `SlabGenerator` to build the slabs.

    Examples
    --------
        miller : [0, 0, 1] or [[0, 0, 1], [1, 1, 1], [1, 1, 0]].

    """

    # Manage the arguments type in order to have lists
    if isinstance(miller, list) and not all([isinstance(m, list) for m in miller]):
        miller = [miller]
    if not isinstance(thickness, list):
        thickness = [thickness]
    if not isinstance(vacuum, list):
        vacuum = [vacuum]

    # Manage the length of the lists
    n = len(thickness)
    if len(vacuum) != n:
        vacuum *= n
    if len(miller) != n:
        miller *= n
    # SlabGenerator expects conventional unit cell so we convert the structure accordingly.
    # As a result, we require input structure to be the primitive standard structure.
    conv_structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
    structure = transfer_average_magmoms(structure, conv_structure)

    slabs = []
    for hkl, thk, vac in zip(miller, thickness, vacuum):
        
        # If thk is zero, then we want to construct an oriented unit bulk by
        # our conversion, so we define a "fake" thickness > 1 to be used to
        # build the slabs, otherwise errors are raised by pymatgen
        thk_gen = 1 if thk == 0 else thk

        slabgen = SlabGenerator(initial_structure=structure,
                                miller_index=hkl,
                                center_slab=center_slab,
                                primitive=primitive,
                                lll_reduce=lll_reduce,
                                max_normal_search=max([abs(index) for index in hkl]),
                                in_unit_planes=in_unit_planes,
                                min_slab_size=thk_gen,
                                min_vacuum_size=vac)

        s = slabgen.get_slabs(bonds=bonds,
                              ftol=ftol,
                              tol=tol,
                              repair=repair,
                              max_broken_bonds=max_broken_bonds,
                              symmetrize=symmetrize)

        # Case of an oriented bulk
        if thk == 0:
            s = s[ext_index].oriented_unit_cell.get_primitive_structure(
                constrain_latt={'a': s[ext_index].lattice.a,
                                'b':s[ext_index].lattice.b,
                                'gamma':s[ext_index].lattice.gamma})

        # Case of a slab
        else:
            s = [Shaper.reconstruct(slab, thk, vac) for slab in s]
            s = s[ext_index]

        slabs.append(s)
    
    return slabs

def rotate(data, mod='z', theta=0.):
    """
    Rotate a set of points. Theta should be degrees

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    mod : TYPE, optional
        DESCRIPTION. The default is 'z'.
    theta : TYPE, optional
        DESCRIPTION. The default is 0..

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    a = data[:, 0]
    b = data[:, 1]
    c = data[:, 2]

    if theta == 0:
       return data
   
    theta = np.pi / 180 * theta

    R = np.zeros((3,3))
    R[0,:] = [1,0,0]
    R[1,:] = [0,1,0]
    R[2,:] = [0,0,1]

    if mod == 'x':
        R[1,:] = [0, np.cos(theta), -np.sin(theta)]
        R[2,:] = [0, np.sin(theta), np.cos(theta)]
    elif mod == 'y':
        R[0,:] = [np.cos(theta), 0, np.sin(theta)]
        R[2,:] = [-np.sin(theta), 0, np.cos(theta)]
    elif mod == 'z':
        R[0,:] = [np.cos(theta), -np.sin(theta), 0]
        R[1,:] = [np.sin(theta), np.cos(theta), 0]

    a_rot = np.zeros(len(a))
    b_rot = np.zeros(len(b))
    c_rot = np.zeros(len(c))

    for j in range(len(a)):
        x = [a[j], b[j], c[j]]
        x_rot = np.dot(R, x)
        a_rot[j] = x_rot[0]
        b_rot[j] = x_rot[1]
        c_rot[j] = x_rot[2]

    return np.column_stack((a_rot, b_rot, c_rot))
