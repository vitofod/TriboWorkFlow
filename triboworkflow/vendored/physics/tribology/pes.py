# --- TribChem code vendored by TriboWorkFlow ---------------------------------
# Unchanged except for the imports. Do not edit: tests/vendoring.py
# compares this file with the original.
# Source: tribchem/physics/tribology/pes.py
# Version: tribchem_devel-align_elisa_w_devel
# Source SHA-256: 82895fbeada949be19d70d0399033dfeb99f8cd8fadf7228d63c7db072519f80
# Copied: 2026-10-22
# Licence: CC BY-SA 4.0, see NOTICE
# Kind: whole
# Imports: tribchem.* -> triboworkflow.vendored.*
# --- end of the TriboWorkFlow header -----------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 15:07:18 2021

Manage the calculation of the Potential Energy Surface (PES) through an high
level class interface. A PES object can be created and the Minimum Energy Path 
(MEP) and the Shear Strength can be evaluated.

This module contains:
    - PES
        Interpolate a 3D Surface with radial basis functions and calculate the
        less stipest path on top of it.
        
        Methods:
        - rbf
        - make_pes
        - __make_rbf
        - __infer_theta
        - to_file
        - from_file
        - from_hs
        - plot

    - PESModel

    - unfold_pes
        Unfold the energies calculated for the unique HS points of an 
        interface/surface, associating them to the replicated HS points 
        covering the whole surface of the cell.


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

Version 1.2.1
-------------------------------------------------------------------------------
- Slightly improve the PES plotting, now make a symmetrized plot around zero
- Add a distinction between plot_name and fig_name in PES plotting

Version 1.2.0
-------------------------------------------------------------------------------
- Add a class to store the PES mathematical models. Contains the methods:
    - v, ci, u
- Add the AnalyticPES class, which initialize a PES using mathematical models.
  Contains the methods and properties:
    - __make_fit, make_pes, v, corrugation, plot_ppes, plot

Version 1.1.0
-------------------------------------------------------------------------------
- Bug Fix: now the code of the PES works fine
- Implement correctly the orthorombize features when making the PES
- Implement the rotation of the provided lattice cell and a method that can
  infer the correct angle. This is the only way to keep the geometry and
  orthorombize correctly an hexagonal lattice cell
- Add a method to plot the initial cell and points

Version 1.0.0
-------------------------------------------------------------------------------
- First module release, contains class: PES, functions: unfold_pes

"""

__author__ = 'Gabriele Losi'
__version__ = "1.1.0"
__credits__ = 'Code based on the `interface-workflow` package, MIT License. '\
              'https://github.com/mcrighi/interface-workflow'
__copyright__ = 'Copyright 2021, Prof. M. Clelia Righi, University of Bologna'
__contact__ = 'clelia.righi@unibo.it'
__date__ = 'March 11th, 2021'


import numpy as np
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt

from triboworkflow.vendored.physics.tribology.mep import MEP
from triboworkflow.vendored.physics.tribology.ppes import PPES, PPESModel
from triboworkflow.vendored.physics.base.solidstate import (
    pbc_coordinates,
    rm_duplicates_2d,
    replicate_2d,
    generate_uniform_grid,
    zfill_cell,
    orthorombize_2d,
    rotate
)


class PES(MEP):
    """
    This class takes a data grid of positions and values, distributed within a
    certain lattice cell. The PES is obtained by replicating the points, and 
    then interpolating the given values with radial basis functions.
    It is also possible to orthorombize the cell before interpolation, however
    in this case it is necessary to ensure that the cell is initially correctly
    oriented, so that the periodicity is granted before the interpolation.

    """

    def __init__(self, data, cell):
        """
        Initialize the class, by passing the data to be interpolated, as a 
        Nx3 matrix, and the lattice cell of the system, as a 3x3 or 2x2 matrix.
        
        The input argument `data` should have the form:
            x1 y1 E1
            x2 y2 E2
             .  .  .
             .  .  .
             .  .  .

        The third column can have whatever units, e.g. in the case of energy you
        might use eV or J/m^2. However, the first two columns need to be Ansgtrom.
        
        The input argument `cell` should have the form:
            x1 y1 z1
            x2 y2 z2
            x3 y3 z3
        or:
            x1 y1
            x2 y2
        
        The lattice cell coordinates need to be Angstrom. If a 3x3 matrix is 
        provided, only the 2x2 planar inner matrix is considered to build the
        PES, being it a two-dimensional function, i.e. f(x, y).
        When a 2x2 matrix is provided, the cell is still filled to a 3x3 with 
        default numbers, to keep the informatic consistency of the code.

        Parameters
        ----------
        data : numpy.array
            Data grid, with shape Nx3, containing all the cartesian displacements
            or coordinates on the first two columns and the values to be
            interpolated on the third column.

        cell : numpy.array
            Lattice cell of the system, as a 2x2 matrix in Angstrom units.

        """

        self.data = data
        self.cell = zfill_cell(cell)

    def make_pes(self, replicate_of=(2, 2), orthorombize=False, theta=None,
                 density=20, tol=1e-4, set_emin_xy=True):
        """
        Parameters
        ----------
        replicate_of : TYPE, optional
            DESCRIPTION. The default is (3, 3).
        function : TYPE, optional
            DESCRIPTION. The default is 'cubic'.
        density : TYPE, optional
            DESCRIPTION. The default is 20.

        """

        output = self.__make_rbf(self.data, self.cell, replicate_of, 
                                 orthorombize, theta, density, tol, set_emin_xy)
        self._rbf, self.pes, self.cell_2d = output

    def __make_rbf(self, data, cell, replicate_of=(2, 2), orthorombize=False,
                   theta=None, density=20, tol=1e-4, set_emin_xy=True):
        """
        Effective calculation of the PES by means of radial basis function
        interpolation. It is wrapped by the `make_pes` method.

        """

        # Rotate the cell in order to have it with the diagonal // to x
        if bool(theta):
            if not type(theta) in (int, float):
                theta = self.__infer_theta()
            fake_d = np.column_stack((data[:, :2], np.zeros(data.shape[0])))
            data[:, :2] = rotate(fake_d, mod='z', theta=theta)[:, :2]
            cell = rotate(cell, mod='z', theta=theta)

        # Apply pbc and make the cell to be orthorombic
        if orthorombize:
           data, cell_2d = orthorombize_2d(data, cell, tol, to_plot=False)
        else:
           cell_2d = cell[:2, :2]

        # Be sure points are not represented twice by ensuring rows in data are 
        # unique. Check also that x and y coordinates are inside the unit cell.
        fake_cell = np.vstack((np.column_stack((cell_2d, [0, 0])), [0, 0, 1]))
        data[:, :2] = pbc_coordinates(data[:, :2], fake_cell, to_array=True)
        data = rm_duplicates_2d(data)
        print('ED: RBF data points fake_cell after rm_duplicates = ', data, flush=True) #rimangono gli stessi shift ed energie OK

        # Scale the energies by setting the minimum to zero
        emin = min(data[:, 2])
        print('ED: minimum energy in the PES = ', emin, flush=True)
        if set_emin_xy:  # Set also the zero to zero
            xmin, ymin = data[np.argwhere(data[:, 2] == emin)][0][0][:2] 
            data[:, 0] -= xmin
            data[:, 1] -= ymin
        data[:, 2] -= emin
        print('ED: RBF data points before replication = ', data, flush=True)
        # Replicate points and interpolate
        print('TG: known points before replication = ', len(data[:, 0]), flush=True)
        data, _ = replicate_2d(data, cell_2d, replicate_of, symm=True) # with symm=True range(-n, n)⟶[-3, -2, -1, 0, 1, 2] → 6 replicas per direction
        print('TG: known points before rm_duplicates = ', len(data[:, 0]), flush=True)
        data = rm_duplicates_2d(data)
        print('TG: RBF known points = ', len(data[:, 0]), flush=True)
        old_options = np.get_printoptions()
        np.set_printoptions(threshold=np.inf)
        print('ED: DATA = ', data, flush=True)
        np.set_printoptions(**old_options)
        rbf = RBFInterpolator(data[:, :2], data[:, 2], kernel='thin_plate_spline', smoothing=1e-3)

        # Calculate the PES on a very dense and uniform grid. Useful for further 
        # analysis (MEP, shear strength) and to plot the PES
        if density is not None:
            coordinates = generate_uniform_grid(cell, density, to_plot=False)
            print('TG: RBF interpolation grid points = ', len(coordinates[:,0]), flush=True)
            energy= rbf(coordinates[:, :2])
            pes = np.column_stack([coordinates[:, :2], energy])
            # Print max energy
            emax = np.max(energy)
            print('ED: maximum energy in the PES =', emax, flush=True)
            idx_max = np.argmax(energy)
            xmax, ymax = coordinates[idx_max, :2]
            print('ED: maximum energy located at x =', xmax, ', y =', ymax, flush=True)
        else:
            pes = data

        print("ED: cell_2d = ", cell_2d, flush=True)
        return rbf, pes, cell_2d

    def __infer_theta(self, tol=1e-4):

        a = np.linalg.norm(self.cell[0, :2])
        b = np.linalg.norm(self.cell[1, :2])
        
        alfa = 180 / np.pi * np.arccos(np.dot(self.cell[0, :2], [1, 0]) / a)
        beta = 180 / np.pi * np.arccos(np.dot(self.cell[1, :2], [1, 0]) / b)
        if alfa > 90:
            alfa = 180 - alfa
        if beta > 90:
            beta = 180 - beta

        if not abs(alfa - beta) < tol:
            theta = (alfa + beta) / 2.
        else:
            theta = 0.

        return theta
    
    def rbf(self, x, y):
        grid = np.vstack((x, y)).T
        return self._rbf(grid)
    
    def plot_cell(self, replicate_of=(1, 1), is_2d=False):
        
        if is_2d:
            cell = self.cell_2d.copy()
        else:
            cell = self.cell.copy()
        
        data, _ = replicate_2d(self.data, cell, replicate_of)
        
        plt.plot([0, cell[0, 0], cell[0, 0]+cell[1, 0], cell[1, 0], 0], 
             [0, cell[0, 1], cell[0, 1]+cell[1, 1], cell[1, 1], 0])
        plt.plot(data[:, 0], data[:, 1], 'o')
        plt.show()

    def to_file(self, fname='pes.dat'):
        """
        Save the pes data to file.

        """
        np.savetxt(fname, self.pes)

    def plot(self, extent=(1, 1), mpts=(200j, 200j), figsize=(7, 7), symm=False,
             level=40, shrink=1, mep=None, show_grid=False, title=None, 
             add_axis=None, to_fig=None):
        """
        Plot the PES and eventually save it

        """

        # Create a mesh grid for plotting
        a = np.sum(np.abs(self.cell_2d[:, 0]))
        b = np.sum(np.abs(self.cell_2d[:, 1]))

        mptx, mpty = (mpts, mpts) if isinstance(mpts, complex) else mpts

        x, y = np.mgrid[- extent[0] * a * int(symm) : extent[0] * a : mptx,
                        - extent[1] * b * int(symm) : extent[1] * b : mpty] #la griglia parte da 0 perchè symm è False
        xf = x.reshape(1, -1)
        yf = y.reshape(1, -1)
        nx = int(np.imag(mptx))
        ny = int(np.imag(mpty))
        z = self.rbf(xf, yf).reshape(nx, ny)
        print("ED: min value of PES in plot", np.min(z))
        z -= np.min(z)

        # Create the plot
        fig = plt.figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        anglerot='vertical'
        zt1=plt.contourf(x, y, z, level, cmap=plt.cm.RdYlBu_r)
        
        # Add legend center the plot and write axis labels
        ax.axis([-extent[0] * a * int(symm), extent[0] * a, 
                 -extent[1] * b * int(symm), extent[1] * b])
        plt.xlabel(r"distance ($\AA$)", fontsize=12, family='serif')
        plt.ylabel(r"distance ($\AA$)", fontsize=12, family='serif')
        cbar1=plt.colorbar(zt1, ax=ax, orientation=anglerot, shrink=shrink)
        cbar1.set_label(r'$E_{adh} (J/m^2)$', rotation=270, labelpad=20,
                        fontsize=15, family='serif')
        if mep is not None:
            ax.plot(mep[:, 0], mep[:, 1], '.-', c='black', ms=2)
        if title is not None:
            plt.title("PES for " + str(title), fontsize=18, family='serif')
        
        if show_grid:
            plt.plot(self.data[:, 0], self.data[:, 1], 'o')
        
        # Add axis to the origin, to indicate the two crystalline orientations
        # in the plane (along cartesian x and y).
        if add_axis is not None:
            ax.plot(0.,0., 'w.', ms=7)
            ax.quiver(0, 0, 1, 0, scale=1., scale_units='inches', width=0.01, 
                      color='white')
            ax.quiver(0, 0, 0, 1, scale=1., scale_units='inches', width=0.01, 
                      color='white')
            ax.text(0.25, -0.5, str(add_axis[0]).replace(',', '').strip(), 
                    rotation='horizontal', color='white', fontsize=14)
            ax.text(-0.5, 0.25, str(add_axis[1]).replace(',', '').strip(), 
                    rotation='vertical', color='white', fontsize=14)
        
        # Small bug in matplotlib, improve the plot lines and graphics
        for zt1 in zt1.collections:
            zt1.set_edgecolor("face")
            zt1.set_linewidth(0.000000000001)
        
        # Save the figure
        if to_fig is not None:
            plt.savefig(str(to_fig), dpi=300)
        
        self.pes_plot = fig
    
    def calculate_mep(self, starting_point=None):
        pass

    @classmethod
    def from_file(cls, fname, cell):
        """
        Generate the pes out of an existing set of data which has

        """
        pes = np.genfromtxt(fname, dtype=float)
        return cls(pes, cell)

    @classmethod
    def from_hs(cls, hs, energy, cell, to_fig=None, point_density=20):
        """
        Create a PES object out of a list of high symmetry points and energies.
        
        Main function to get the Potential Energy Surface (PES) for an interface. 
        The points are replicated to span a 3x3 lattice cell and are interpolated
        by using Radial Basis Functions (cubic function).
        In the output data the energy is normalized so that the absolute minimum
        is 0. Furthermore it is made sure that the lateral point are inside the
        unit cell before they are replicated.

        The entire set of HS points covering the interface associated with 
        the corresponding energy.
                x[0]  y[0]  E[0]
                x[1]  y[1]  E[1]
                .     .     .
                .     .     .
                .     .     .
        
        Parameters
        ----------        
        hs : dict
            Unfolded HS points of the interface, covering all the surface.
            
        E : dict
            Contains the energy calculated for each unique surfacial HS site.
        
        cell : np.ndarray
            Vectors of the lattice cell of the interface.
            
        to_fig : string, optional CURRENTLY NOT IMPLEMENTED
            Name of the image that you want to save, it will be: 'to_fig'+'.pdf' 
            Suggested name: name = 'PES_' + 'Name of the interface'.
            The default is None and no image is saved.

        Returns
        -------
        rbf : scipy.interpolate.rbf.Rbf
            Object containing the information of the interpolation of the potential
            energy. Call it on a set of [x, y] coordinates to obtain the energy 
            values for those points. Usage: rbf([x, y])
            
        pes_dict : dict
            Dictionary containing the points and the associated energies divided
            by type. Useful to keep track of the type of the HS sites and their
            energies. To be saved to DB (?)
        
    pes_data : np.ndarray
        The entire set of HS points covering the interface with the 
        corresponding energies.
        Format:
            x[0]  y[0]  E[0]
            x[1]  y[1]  E[1]
             .     .     .
             .     .     .
             .     .     .
     
        """
        
        # Unfold the PES points and create the data array to feed the PES class
        _, data = unfold_pes(hs, energy)
        return cls(data, cell)


# =============================================================================
# Analytic PES
# =============================================================================

class PESModel:

    # Optimal coefficients for graphene PPES fit
    gr_min = np.array([1.63096e6, 3.347616, 8184.7, 4])  # meV, A^-1, meV/A^4
    gr_max = np.array([2.75075e6, 3.349208, 8258.11, 4])  # meV, A^-1, meV/A^4

    def __init__(self, cmin, cmax, g, n=4):

        self.cmin = np.array(cmin)
        self.cmax = np.array(cmax)
        self.g = g
        self.n = n

    def v(self, x, y, z, n):

        v_rep = self.ci(x, y, 0) * np.exp(-z * self.ci(x, y, 1))
        v_att = - self.ci(x, y, 2) / np.power(z, n)

        return  v_rep, v_att

    def ci(self, x, y, i=0):
        return self.cmax[i] - (self.cmax[i] - self.cmin[i]) * self.u(x, y)

    def u(self, x, y):
        
        sum_cos = 0
        for g in self.g:
            sum_cos += np.cos(np.dot(g, np.array([x, y])))

        return 2/9 * (3 - sum_cos)

class AnalyticPES(PESModel):
    
    def __init__(self, vmin, vmax, g, n=4, p0=None, maxfev=1000, to_plot=True):
        
        # Store the PPES data on top of the minimum and maximum energy points
        self.ppmin = np.array(vmin)
        self.ppmax = np.array(vmax)
        self.g = g
        
        # Fit the ppes and calculate the coefficients
        self.__make_fit(self.ppmin, self.ppmax, g, n, p0, maxfev, to_plot)

    def __make_fit(self, vmin, vmax, g, n, p0, maxfev, to_plot):
        
        # Calculate the coefficients of the min PPES
        obj_min = PPES(self.ppmin)
        cmin, _ = obj_min.fit('n' + str(n), p0, maxfev)
        
        # Calculate the coefficients of the max PPES
        obj_max = PPES(self.ppmax)
        cmax, _ = obj_max.fit('n' + str(n), p0, maxfev)
        super().__init__(cmin, cmax, g, n)

        # Plot the points and the fitting results
        if to_plot:
            self.plot_ppes()

        return cmin, cmax

    def make_pes(self, extent=(5, 5), mpts=(200j, 200j), to_plot=True):
        
        grid = np.mgrid[- extent[0] : extent[0] : mpts[0],
                        - extent[1] : extent[1] : mpts[1]]
        x = grid[0].flatten()
        y = grid[1].flatten()
        
        # Calculate the energy and shift the minimum to zero, stack the data
        energy = self.v(x, y, to_print=False)
        energy -= min(energy)
        data = np.column_stack((x, y, energy))

        self.pes = data
        self.pes_shape = grid.shape[1:]
        
    def v(self, x, y, z=None, to_print=True):
        
        # Define a default z if not provided, correspond to the z of min energy
        if z is None:
            z = self.ppmin[:, 0][self.ppmin[:, 0] == min(self.ppmin[:, 0])]
        
        # Calculate the repulsive and attractive terms
        v_rep, v_att = super().v(x, y, z, n=self.n)
        
        if to_print:
            print('PES model evaluation in ({.4f}, {.4f}, {.4f})'.format(x, y, z))
            print('Repulsive term  : {}'.format(v_rep))
            print('Attractive term : {}'.format(v_att))

        return v_rep + v_att
    
    @property
    def corrugation(self):
        return min(self.ppmax[:, 1]) - min(self.ppmin[:, 1])
    
    def plot_ppes(self):

        x = np.arange(min(self.ppmin[:, 0]), max(self.ppmin[:, 0]), 0.01)
        y = PPESModel.potential_n4(x, self.cmin[0], self.cmin[1], self.cmin[2])
        plt.plot(self.ppmin[:, 0], self.ppmin[:, 1], 'o', label='PPES min - points')
        plt.plot(x, y, label='PPES min - fit')

        x = np.arange(min(self.ppmax[:, 0]), max(self.ppmax[:, 0]), 0.01)
        y = PPESModel.potential_n4(x, self.cmax[0], self.cmax[1], self.cmax[2])
        plt.plot(self.ppmax[:, 0], self.ppmax[:, 1], 'o', label='PPES max - points')
        plt.plot(x, y, label='PPES max - fit')
        
        plt.legend()
        plt.show()
    
    def plot(self, level=40, shrink=1):
        
        if not hasattr(self, 'pes'):
            raise ValueError('The pes has not been calculated yet. Call the '
                             ' make_pes method on the instance!')
        
        x = self.pes[:, 0].reshape(self.pes_shape)
        y = self.pes[:, 1].reshape(self.pes_shape)
        e = self.pes[:, 2].reshape(self.pes_shape)
        
        # Create the plot
        fig = plt.figure(figsize=(7, 7), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        anglerot='vertical'
        zt1 = plt.contourf(x, y, e, level, cmap=plt.cm.RdYlBu_r)
        
        # Add legend center the plot and write axis labels
        plt.xlabel(r"distance ($\AA$)", fontsize=12, family='serif')
        plt.ylabel(r"distance ($\AA$)", fontsize=12, family='serif')
        cbar1=plt.colorbar(zt1, ax=ax, orientation=anglerot, shrink=shrink)
        cbar1.set_label(r'$E_{adh} (J/m^2)$', rotation=270, labelpad=20,
                        fontsize=15, family='serif')

        plt.show()


# =============================================================================
# Tools for the PES
# =============================================================================

def unfold_pes(hs_u, energy_u):
    """
    Unfold the energies calculated for the unique HS points of an interface,
    associating them to the replicated HS points covering the whole surface of 
    the cell. It is supposed that you provide a set of unique HS points and the
    corresponding energies, to be replicated to full the entire lattice cell.

    Parameters
    ----------
        
    hs_all : dict
        Surfacial HS sites that has been unfolded (replicated) across the whole
        lattice cell of slab. 
        Ex. To the key 'ontop_1 + bridge_1' will correspond n points, spanning
        the entire plane axb of the lattice cell. Data is a (n, 3) numpy array.
    
    E : list
        Contains the energy calculated for each unique interfacial HS site.
        The energy are calculated by means of ab initio simulations (VASP).
        E must have the following structure:
            
            [ [label_1, x_1, y_1, E_1], 
              [label_2, x_2, y_2, E_2], 
              ...                      ]
        
        Ex. label should corresponds to the keys in hs_all, associated to a 
        certain shit between the lower and upper slab, e.g. 'ontop_1+bridge_1'.
        
    Returns
    -------
    E_list : list
        It's basically the same as E_unique but contains all the HS points 
        replicated on the whole interface. The structure of the list is:
            
            [ [label_1, x_1, y_1, E_1], 
              [label_2, x_2, y_2, E_2], 
              ...                      ]
        
    E_array : np.ndarray
        Numpy matrix containing the coordinates and the energy useful to 
        interpolate a PES. It's E_list without labels and with array type. 
        The structure of the matrix is:
            
            np.array([ [x_1, y_1, E_1], 
                       [x_2, y_2, E_2], 
                       ...             ])

    """

    # Initialize lists for the result
    e_list = []
    e_array = []

    # Extract the element
    for element in energy_u:
       label  = element[0]
       energy = element[3]

       # Associate each Energy to all the corresponding HS values
       for row in hs_u[label]:
          x_shift = row[0]
          y_shift = row[1]

          e_list.append([label, x_shift, y_shift, energy])
          e_array.append([x_shift, y_shift, energy])

    e_array = np.array(e_array)      
    
    return e_list, e_array

def static_tribo(hs, E, cell):  
    """
    Main function to calculate the PES, MEP, and shear strength of an interface

    Parameters
    ----------        
    hs : dict
        Unfolded HS points of the interface, covering all the surface.        

    E : dict
        Contains the energy calculated for each unique surfacial HS site.
        
    cell : numpy.ndarray
        Lattice parameter of the interface you want to study

    Returns
    -------
    pes_info : list
        Relevant information about PES
        
    mep_info : list
        Relevant information about MEP
        
    ss_info : list
        Relevant information about Shear Strength
    """
    
    from triboworkflow.vendored.physics.tribology.potensurf import get_pes, orthorombize
    from triboworkflow.vendored.physics.tribology.mep import get_bsmep, get_mep
    from triboworkflow.vendored.physics.tribology.shearstrength import (
        get_shear_strength_xy,
        get_shear_strength
    )
    
    # Get the PES
    rbf, pes_dict, pes_data = get_pes(hs, E, cell)
    data_ortho, cell_ortho = orthorombize(pes_data, cell)
    
    # Get the MEP on the potential energy surface starting from a guess
    bsmep, ss_bsmep, theta = get_bsmep(cell_ortho, rbf)
    mep, mep_convergency = get_mep(cell_ortho, rbf, theta)
    
    # Calculate the MEP along the x, y, and MEP directions
    data_ss_xy, ss_xy = get_shear_strength_xy(cell_ortho, rbf)
    data_ss_mep, ss_mep  = get_shear_strength(mep, rbf)
    
    pes_info = [pes_dict, pes_data, rbf]
    mep_info = [data_ss_mep, mep_convergency, mep, bsmep]
    ss_info = [ss_xy, ss_bsmep, ss_mep]
    
    return pes_info, mep_info, ss_info
