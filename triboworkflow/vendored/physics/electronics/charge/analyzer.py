# --- TribChem code vendored by TriboWorkFlow ---------------------------------
# Unchanged except for the imports. Do not edit: tests/vendoring.py
# compares this file with the original.
# Source: tribchem/physics/electronics/charge/analyzer.py
# Version: tribchem_devel-align_elisa_w_devel
# Source SHA-256: c61562800aeea3e2a273af6d7bfca730116892994a8c15e976c03f081d3824d8
# Copied: 2026-10-22
# Licence: CC BY-SA 4.0, see NOTICE
# Kind: whole
# Imports: tribchem.* -> triboworkflow.vendored.*
# --- end of the TriboWorkFlow header -----------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The module defines the QEChargeAnalyzer class for analyzing chargefiles from
QE pp. It is currently capable to read both filplot and fileout charge files.
It defines also the ChargeProcessor class for analyzing multiple chargefiles.

Filplot are the raw charge data (by &INPUTPP), while fileout is the processed
charge data (by &PLOT). The first takes into account the effect of valence
electrons.

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

Version 1.3.2
-------------------------------------------------------------------------------
- Now the integrated charge is always normalized by the integration width, 
  in order to keep the units as e-/A^3.
- Add the possibility to normalize rho(i) with area and valence electrons
- Improve the information printed to screen

Version 1.3.1
-------------------------------------------------------------------------------
- Minor improvements in coding style (PEP8)

Version 1.3.0
-------------------------------------------------------------------------------
- Introduce new functions to work on final charge arrays from file
- Improve the error handling in _process_average, print more details
- Rename: Classes names are now QEChargeAnalyzer and QEChargeProcessor
- Bug Fix: Calculation of area/volume for ibrav=0 with filplot charge inputs
- Bug Fix: Saving files when unpack is True and no zipped files are found
- Bug Fix: Reading error when processing more than 9 files with name charge_#
- Bug Fix: The interface is now correctly detected when ibrav=0
- Minor improvements in coding style (PEP8)

Version 1.2.0
-------------------------------------------------------------------------------
- Add terminal gui and functions to run charge displacement analysis easily
- Add a new method to print a charge displacement summary in ChargeProcessor
- Now the integration of the charge displacement is made with absolute values
- Fixed a bug that prevented the charge attribute from being properly saved
- Minor fix and improvements

Version 1.1.0
-------------------------------------------------------------------------------
- Add new methods to get info about structures and charge in ChargeAnalyzer
- Fixed a critical bug in ChargeProcessor regarding IO to files
- The documentation of ChargeProcessor has been completed
- Minor improve to the code

Version 1.0.0
-------------------------------------------------------------------------------
- First release of the program, basic ChargeAnalyzer, ChargeProcessor classes

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

TODO :
    - Create a class that read and store physical constants and conversions
      between different units of measure. Call it in QEChargeAnalyzer.
    - Create a Plotter class and use it to make the plots in the save_average
      method.
    - Write the center_interface method, in order to concatenate charge arrays
      aroung the zero and not having problems with PBC. Important for plotting
      charge arrays in a clean way and making nice figures.
    - Add the possibility to have linear averages, i.e. axis is 'x', 'y', 'z',
      by writing the __process_linear_average method, 
    - Generalize the extraction of atoms and coordinates in __clean_filplot, 
      by using pandas in a smarter way to work directly on the portion of
      interest of the DataFrame. (same as in __clean_fileout).
    - Add the possibility to average the charge on any custom direction and 
      plane, by passing the parameters of a general straigth line or plane.
    - Add methods to analyze cube files and eventually other extensions.
    - Detour_average is not working with map at the moment, debug and fix it.
    - Possibly rewrite the code employing cls, staticmethods, and abc.

"""
              
__author__ = 'Gabriele Losi'
__version__ = "1.3.2"
__copyright__ = 'Prof. M.C. Righi, University of Bologna'
__contact__ = 'clelia.righi@unibo.it'
__related__ = 'Sci Rep 9, 17062 (2019); Comput. Mater. Sci. 154 517–29 (2018)'
__date__ = 'October 22nd, 2020'

import sys
import os  
import datetime
import pickle
import json  

import scipy.integrate as integrate
from monty.json import jsanitize
from copy import deepcopy 
import pandas as pd
import numpy as np

from triboworkflow.vendored.physics.base.solidstate import get_cell_info, get_cell_qe                               


# =============================================================================
# Definition of the classes: QEChargeAnalyzer and QEChargeProcessor
# =============================================================================

class QEChargeAnalyzer():
    """
    Use this class to load and manipulate charge density files that have been 
    written by Quantum Espresso post processing tools (pp.x).

    Attributes
    ----------
    charge_grid : np.ndarray
        Contain the charge grid with the raw data.
    params : dict
        Relevant parameters collected from the charge files read from input.
    filepath : str
        Path to the uploaded charge files.
    cell : np.ndarray
        Lattice cell of the system.
    filetype : str
        Type of the input file, either filplot or fileout.
    charge : list of dict
        Contains information related to the averaged charge of the system.

    Methods
    -------
    read_charge(filename=None, to_file=True)
        Read an external data file and upload its data.
    average_charge(axis='xy', shift=0., rescale=1000, get_inter=True,
                   valence=True, to_file=True, to_plot=True)
        Average the charge within the cell or along a specific axis or plane.
    detour_average(axis=['xyz', 'xy'], shift=0, rescale=1000, get_inter=True, 
                   to_file=True, to_plot=True)
        Run different averages at the same time.
    get_info(verbose=False)
        Print information about the average charge.
    information(obj, verbose=False)
        Static method working on a object, print a summary on average charge.
    save_average(data, filename=None, to_file=True, to_plot=(True))
        Save the averaged data to file and plot.
    to_file(filename=None)
        Export charge grid and parameters to files.
    from_file(filename=None)
        Upload charge grid and parameters from files.
    
    """
    
    bohr_to_angstrom = 0.529177249
    
    def __init__(self, chargefile=None, unpack=True):
        """
        Initialize a QEChargeAnalyzer object by passing a file path, and also 
        trying to read it from previous zipped data files to be faster.
        The object could be initialized without specifying any parameter and
        initialized later calling the methods from_file or read_charge.
        
        Parameters
        ----------
        chargefile : string, optional
            String containing the file path to charge output file of the post
            processing of Quantum Espresso.
            It can be a "raw" file from QE or an "elaborated" one, i.e. the 
            files named after defining filplot and fileout in the pp input.
            It is always possible to initialize an object without specifying a 
            file name. The default is None.
            
        unpack : bool, optional
            Try loading data from previously exported data files.
            By default, it looks for: chargefile.ext file. The default is True.
            
            Extension are:
                - .pkl for pickle binary files. Contain grid for charge data.
                - .json for dicts with parameters and integration results.
                
            WARNING : It is recommended that you always switch unpack to True, 
            unless you have a valid reason not to. There is a speed up factor 
            of 10^2 between unzipping exported data vs uploading charge files.
        
        Raises
        ------
        IOError
            When input charge files are not found.
        
        """
        
        if chargefile is not None:

            if unpack:
                data = self.from_file(filename=chargefile)

                # Assign attributes
                if data is not False:
                    
                    grid, json, filename = data        
                    self.charge_grid = grid
                    self.params, self.charge = json 
                    self.filepath = filename
                    self.cell = np.array(self.params['crystal']['cell'])                
                    self.filetype = self.params['filetype']
                
                else:
                    self.read_charge(chargefile)
            else:
                self.read_charge(chargefile)
       
    def read_charge(self, filename=None, to_file=True):
        """
        High level method managing the reading of external charge files.

        It is called automatically if a chargefile is passed to the 
        constructor, otherwise it can be called as an object method.   

        Parameters
        ----------
        filename : str, optional
            Single input file containing charge data. The default is None.
            
        to_file : bool or str, optional
            Decide whether to save the processed data to be faster the next 
            time an instance is created. A string containing the path and the
            filename where to save, otherwise the default filepath is used.
            The default is True.

        """
        
        self.charge_grid, self.params = self._process_charge(filename)
        
        # Generate object attributes
        self.cell = np.array(self.params['crystal']['cell'])
        self.filetype = self.params['filetype']
        self.filepath = filename
        self.charge = {}
        
        # Save to file if to_file is neither None or False
        # Use default name if to_file is True or '', to_file string otherwise
        if to_file == '' or to_file:
            self.to_file()
        elif isinstance(to_file, str):
            self.to_file(filename=to_file)

    def average_charge(self, axis='xy', shift=0., rescale=1000, get_inter=True,
                       geometry_norm=False, valence=True, to_file=True, to_plot=True):
        """
        Average charge data along specific directions or planes.
        By default it averages the charge in xy plane, i.e. rho(x,y,z)->rho(z).
        It is capable to make whole cell and planar averages at the moment.

        Parameters
        ----------
        axis : str, optional
            String containing the axis to be averaged, e.g. 'x', 'xy', 'xyz'.
            The string are converted to tuples (z=0, y=1, x=2), to select
            axis to average. It is possible to make different kind of averages:
                - To make a 3D average over the lattice cell use 'xyz'.
                - To make a planar average pass a 2-str element, e.g. 'xy'.
                - To make a linear average pass a 1-str element, e.g. 'x'.
            The default is 'xy'.
        
        shift : float, optional
            Shift to be applied along the remaining direction after planar 
            averages in order to center the interface to the zero. 
            The default value is 0. 
        
        rescale : int (or tuple of int), optional
            Set the number of points to be used when creating the final array
            containing the data that can be plotted. Useful either when 
            averaging 1D or 2D. The default value is 1000.
                - If averaging along planes, rescale should be an int. If a 
                  tuple is passed only the first value is taken.
                - If averaging along lines, rescale could be either a tuple or
                  an integer. In the latter case, (rescale, rescale) is assumed
                  to construct a grid of points in the two directions.
        
        get_inter : bool, optional
            Control whether to collect information about the interface.
            If set to True, it saves additional information to charge attribute, 
            such as the interface position, width, and charge.
            The default is True.
        
        valence : bool, optional
            To take into account the valence electrons when calculating the 
            average of the charge. Only works for a filplot filetype, 
            does nothing in other cases. The default is True.
            
        to_file : bool (or str), optional
            Save data arrays to file to be plotted later by other app, 
            e.g. gnuplot. Useful only when making planar or linear averages. 
            A string containing the name of the file to be saved can be passed,
            if it is set to True a default name is used instead.
            The Default value is None.
            
        to_plot : bool (or str), optional
            Wheter to save data arrays to nice plots. Useful only when making 
            planar or linear averages. The name of the file can be passed as 
            a string, otherwise if set to True a default name is used.            
            The default value is None.

        """                                
        
        # Define the path where to save data
        path = ''
        for x in self.filepath.split('/')[:-1]:
            path = path + str(x) + '/'
            
        average, p = self._process_average(self.charge_grid, self.params, 
                                           axis, path, shift, rescale, get_inter,
                                           geometry_norm, valence, to_file, to_plot)

        if p is not None:
            self.charge.update(p)

    def detour_average(self, axis=['xyz', 'xy'], shift=0., rescale=1000,
                       get_inter=True, geometry_norm=False, valence=True, 
                       to_file=True, to_plot=True):
        """
        Run a detour of averages, calling average_charge over a list of axis.
        By default it is called by switching on all extra saving options.
        The numeric and boolean arguments will be applied at all the runs.
        For a more detailed description of the arguments, see average_charge.

        Parameters
        ----------
        axis : list of str (or str), optional
            List containing the axis along which you want to average the charge.
            If all' is passed everything is done. The default is ['xyz', 'xy'].
        
        shift : float or int, optional
            Shift applied to the averaged charge, in order to center the 
            interface to zero. The default is 0.
            
        rescale : int, optional
            Rescale the final array of averaged data. The default is 1000.
            
        get_inter : bool, optional
            Decide whether to calculate the interfacial charge properties
            automatically. The default is True.
        
        valence : bool, optional
            To take into account valence electrons when averaging the charge. 
            Meaningful only for filplot filetypes. The default is True.
            
        to_file : bool, optional
            Export calculated data to file. The default is True.
            
        to_plot : bool, optional
            Plot average charge. The default is True.

        """
        
        # Select the runs to be done
        list_all = ['xyz', 'xy', 'xz', 'yz', 'x', 'y', 'z']
        if axis != 'all':
            list_all = axis
        
        # Run the calculation
        for axis in list_all:
            self.average_charge(axis, shift, rescale, get_inter, geometry_norm,
                                valence, to_file, to_plot)

        # MAP is strangely not working at the moment, fix it
        # n = len(list_all)
        # map(self.average_charge, list_all, [shift]*n, [rescale]*n, [get_inter]*n,
        #     [valence]*n, [to_file]*n, [to_plot]*n)
    
    def save_average(self, data, filename=None, to_file=True, to_plot=True):
        """
        Save arrays containing the averaged charge to file.

        Parameters
        ----------
        data : np.ndarray
            Array to1 be saved to file.
            
        filename : str, optional
            Name of the file to be saved. When nothing is passed, a default 
            filename is adopted instead. The default is None.
            
        to_file : bool, optional
            Export to file the given array. The default is True.
            
        to_plot : bool, optional
            Call the plotter and plot the charge. The default is True.

        """
        
        if isinstance(to_file, str) and to_file != '':
            np.savetxt(self.filepath + '.dat', data)
        elif to_file:
            np.savetxt(filename + '.dat', data)            
        
        if isinstance(to_plot, str) and to_plot != '':
            pass
        elif to_plot:
            pass 
        
    def to_file(self, filename=None):
        """
        Export the charge grid as a pickle file, and params as a json dict.
        
        Parameters
        ----------
        filename : str, optional
            Base filename to save the data to file. Data will be saved as 
            filename+'.pkl' and filename+'.json'. A default name based on the
            object filepath is used if nothing is passed. The default is None.

        """
        
        if filename is None or not isinstance(filename, str):
            filename = self.filepath
        
        f_pkl = filename + '.pkl'
        f_json = filename + '.json'
        
        # Save the picke binary file
        with open(f_pkl, 'wb') as f:
            pickle.dump(self.charge_grid, f)
            
        # Save the json dictionary
        with open(f_json, 'w') as f:
            json.dump([jsanitize(self.params), jsanitize(self.charge)], f)
        
    def from_file(self, filename=None):
        """
        Load charge matrix from a pickle file and params from a json dict.
        
        Parameters
        ----------
        filename : str, optional
            Files to be read to upload the data. It will search for files as: 
            filename+'.pkl' and filename+'.json'. A default name based on the
            object filepath is used if nothing is passed. The default is None.
    
        """
        
        # Assign default values if filename not provided
        if filename is None or not isinstance(filename, str):
            try:
                filename = self.filepath
            except:
                return False
        
        f_pkl = filename + '.pkl'
        f_json = filename + '.json'
        
        # Check if input files exist and are files
        exists_pkl = os.path.exists(f_pkl) and os.path.isfile(f_pkl)
        exists_json = os.path.exists(f_json) and os.path.isfile(f_json)
            
        if not exists_pkl or not exists_json:
            return False
        
        # Load the pickle binary file
        with open(f_pkl, 'rb') as f:
            charge_grid = pickle.load(f)
        
        # Load the json dictionary
        with open(f_json, 'r') as f:
            json_dicts = json.load(f)
            
        return (charge_grid, json_dicts, filename)
    
    def get_info(self, verbose=False):
        """
        Method to print to stdout a summary of information about the system
        under study and the processed average charge.

        Parameters
        ----------
        verbose : bool, optional
            Plot more information about the system. The default is False.

        """
        
        _, _ = self.information(self, verbose=verbose)
    
    @staticmethod
    def information(obj, verbose=False):        
        """
        Extract the information of a QEChargeAnalyzer object.

        Parameters
        ----------
        obj : tribchem.electronics.charge.analyzer.QEChargeAnalyzer
            QEChargeAnalyzer object of which you want to colelct information.
            
        verbose : bool, optional
            Plot much more information about the system. The default is False.

        Raises
        ------
        ValueError
            If the object is not initialized yet.

        Returns
        -------
        grid : np.ndarray
            Charge grid containing the points.
            
        chg : dict
            Contains the average data associated with the correpsponding axis.

        """
        
        # Check if the object has been initialized
        attributes = obj.__dict__
        if not attributes:
            raise ValueError('The object {} is not initialized'.format(obj))
        
        p = obj.params
        
        # Print information about charge input file
        print('')
        print('\033[4m' + 'General Information' + '\033[0m\n')
        print('Filename at: {}'.format(obj.filepath))
        print('Filetype is: {}'.format(obj.filetype))
        
        # Print parameter information
        print("Charge grid: {}x{}x{}".format(p['params']['nx'], 
                                             p['params']['ny'], 
                                             p['params']['nz']))
        print("Number of atoms   : {}".format(p['params']['nat']))
        print("Number of species : {}\n".format(p['params']['ntyp']))
        
        # Print crystal information
        print('\033[4m' + 'Crystal Information' + '\033[0m\n')
        if verbose:
            print('Lattice cell: ')
            for row in obj.cell:
                print('{:.4f} {:.4f} {:.4f}'.format(row[0], row[1], row[2]))
            print('Area   : {0:.3f}'.format(p['crystal']['area']))
            print('Volume : {0:.3f}\n'.format(p['crystal']['volume']))
        
        # Print atoms information
        print('Atoms in the cell: ')
        n_atoms = []
        sp = p['atoms']['species']
        for at in set(sp):
            n_atoms.append((at, sp.count(at)))
        for el in n_atoms:
            print('{} : #{}'.format(el[0], el[1]))
            
        if verbose:
            atomic_coords = np.column_stack((np.asarray(sp),
                                             np.array(p['atoms']['coordinates'])))
            print('Atomic coordinates:')
            for row in atomic_coords:
                print('{} {:.6f} {:.6f} {:.6f}'.format(row[0], 
                                                       float(row[1]), 
                                                       float(row[2]), 
                                                       float(row[3])))
        
        # Evaluate the charge average
        print('')
        print('\033[4m' + 'Charge information' + '\033[0m')
        if not attributes['charge']:
            chg = None
            print('\nAverage charge along plane/axis is not available')
            
        else:
            chg = {}
            keys = list(obj.charge.keys())
            print('')
            
            # Check if the whole cell is present
            if 'cell' in obj.charge.keys():
                inter = obj.charge['cell']
                print('Cell\n')
                print('Total charge: {0:.6f} e-'.format(inter['rho_tot']))
                print('Mean charge : {0:.6f} e-/\u212B^3\n'.format(inter['rho_mean']))
                keys.remove('cell')
            
            # Print averaged data along axis and planes
            for kind in keys: 
                for ax in obj.charge[kind].keys():
                    # Extract data
                    inter = obj.charge[kind][ax]['interface']
                    chg[ax] = obj.charge[kind][ax]['data']
                    print('\u03C1({})\n'.format(ax))
                    
                    # Cell information
                    if inter['position'] is not None:
                        print('Interface position to cell: {0:.3f} \u212B'.format(inter['position']))
                    if inter['width'] is not None:
                        print('Interface width: {0:.3f} \u212B'.format(inter['width']))
                    if inter['heigth'] is not None:
                        print('Atomic thickness: {0:.3f} \u212B'.format(inter['heigth']))
                    
                    # Charge information
                    if inter['rho_tot'] is not None:
                        print('Total charge ((1/\u0394{0})\u222b\u03C1({1})d{2}): {3:.8f} e-/\u212B^3'
                              .format(ax, ax, ax, inter['rho_tot']))
                    if inter['rho_0'] is not None:
                        print('Zero charge (\u03C1({0}=0)): {1:.8f} e-/\u212B^3'
                              .format(ax, inter['rho_0']))
                    if inter['rho_interface'] is not None:
                        print('Interfacial charge ((1/\u0394{0})\u222b\u03C1({1})d{2}): '
                              '{3:.8f} e-/\u212B^3'.format(ax, ax, ax, inter['rho_interface']))
                    if 'normalized_by_valence' in inter.keys():
                        print('Normalized by valence: {}'.format(inter['normalized_by_valence']))
                    if inter['normalized_lenght_tot'] is not None:
                        print('Normalized length for total charge: {0:.3f} \u212B'
                              .format(inter['normalized_lenght_tot']))
                    if not bool(inter['normalized_by_area']):
                        print('Normalized by area: {}\n'.format(inter['normalized_by_area']))   
                    else:
                        print('Normalized by area: {0:.3f} \u212B\n'
                              .format(inter['normalized_by_area']))     
        
        grid = obj.charge_grid
        
        return grid, chg

    def _process_charge(self, inp_file=None):
        """
        Upload charge files as pandas Dataframes, clean the tables, and store
        all the relevant data. Dependency of read_charge.

        """
        
        if not inp_file:
            raise IOError('No input file is provided!')
        
        # Check if input exists and is a files
        else:
            exists = os.path.exists(inp_file)
            isfile = os.path.isfile(inp_file)
            
            if not exists:
                IOError('Input "{}" does not exists'.format(inp_file))
                
            elif not isfile:
                IOError('Input "{}" is not a file'.format(inp_file))
        
        # Get relevant data
        table, params, indexes = self._clean_table(inp_file)
        charge = self._load_data_grid(table, params, indexes)
            
        return charge, params        
        
    def _clean_table(self, inp_file=None):
        """
        Subfunction managing the cleaning of charge data saved as pd.DataFrame.
        Dependency of process_charge.

        """
        
        if not inp_file:
            raise IOError('Something weird is happening')
        
        table = pd.read_table(inp_file)
        first_row = table.columns[0].split()
        
        # Read header of "raw" data from QE pp (output from filplot)
        if len(first_row) > 1 and len(first_row) == 8:
            table_type = 'filplot'
            index, params = self._clean_filplot(table, first_row)
        
        # Read header of "processed" data from QE pp (output from fileout)
        elif len(first_row) == 1 and first_row[0] == 'CRYSTAL':    
            table_type = 'fileout'
            index, params = self._clean_fileout(table)
            
        else:
            raise IOError('You input file is not recognized.'
                          'Provide the outcome of QE post processing as input')
        
        # Prepare the indexes to extract data
        if table_type == 'filplot':
            indexes = [index, None]
        else:
            indexes = [index, -2]
        
        params.update({'filetype': table_type})

        return table, params, indexes
    
    def _clean_filplot(self, table, columns):
        """
        Clean the raw data from a QE post processing. It is the file defined 
        by the `filplot` flag in namelist &INPUTPP. Dependency of __clean_table.

        """
        
        # Collect data from the first row (DataFrame column names)
        columns = [int(i) for i in columns]
        nx, ny, nz, nnx, nny, nnz, nat, ntyp = columns
        
        # Collect data from the second row of the file
        row_2 = [float(i) for i in list(table.loc[0])[0].split()]
        ibrav = int(row_2[0])
        celldm = row_2[1:]      
        
        # If ibrav is zero there are three more lines containing cell vectors
        if ibrav == 0:
            alat = celldm[0] * QEChargeAnalyzer.bohr_to_angstrom
            a1, a2, a3 = [float(i) for i in list(table.loc[1])[0].split()]
            b1, b2, b3 = [float(i) for i in list(table.loc[2])[0].split()]
            c1, c2, c3 = [float(i) for i in list(table.loc[3])[0].split()]
            cell = np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]) * alat
            
            # Calculate volume and area
            c = np.array(cell)
            area = np.linalg.norm(np.cross(c[:, 0], c[:, 1]))
            volume = np.abs(np.dot((np.cross(c[:, 0], c[:, 1])), c[:, 2]))
            index = 5
        
        else:
            cell, area, volume = get_cell_qe(ibrav, celldm)
            cell = cell.tolist()
            index = 2
        
        # Save the different types of atoms
        i = 0
        elements = {'species': {}}
        atoms_index = {}
        elements_list = []
        while i < ntyp:
            n_atom, name, valence = [k for k in 
                                     list(table.loc[index + i])[0].split()]
            atoms_index.update({int(n_atom): str(name)})
            elements_list.append(name)
            elements['species'].update(
                {str(name): {'number': 0, 'valence': float(valence)}})
            i += 1
            
        # Read the atomic coordinates and save coordinates and species
        index += i
        j = 0
        coordinates = []
        species = []        
        while j < nat: 
            coords = [float(k) for k in list(table.loc[index + j])[0].split()]
            coordinates.append(coords[1:4])
            species.append(atoms_index[int(coords[4])])
            j += 1
        
        # Find total number of electrons and update dict
        electrons = 0
        for key in elements['species'].keys():
            number = species.count(key)
            elements['species'][key].update({'number': number})
            electrons += number * elements['species'][key]['valence']
        mean_valence = electrons / nat
        elements.update({'electrons': electrons, 
                         'mean_valence': mean_valence})
            
        coordinates = (np.array(coordinates) * \
                       celldm[0] * QEChargeAnalyzer.bohr_to_angstrom).tolist()
        
        # Define variables to be returned
        index += j
        params = {
            'params': 
                {
                    'nx': nx, 'ny': ny, 'nz': nz,
                    'nnx': nnx, 'nny': nny, 'nnz': nnz,
                    'nat': nat, 'ntyp': ntyp
                },
                                   
            'crystal': 
                {
                    'ibrav': ibrav, 
                    'celldm': celldm,
                    'cell': cell,
                    'area': area,
                    'volume': volume
                },
                
            'atoms':
                {
                    'electronics': elements,
                    'species': species,
                    'coordinates': coordinates
                }
            }
        
        return index, params
        
    def _clean_fileout(self, table):
        """
        Clean the final data from a QE post processing. It is the file defined 
        by the `fileout` flag in namelist &PLOT. Dependency of __clean_table.

        """

        # Save the cell        
        matrix = table.loc[[1, 2, 3]]
        matrix = [x.split() for x in matrix[" CRYSTAL"]]
        matrix = pd.DataFrame(matrix, columns=['x', 'y', 'z'])
        
        cell = np.array(matrix, dtype=float)
        area, volume = get_cell_info(cell)
        cell = cell.tolist()
        
        # Read how many atomic species
        nat, ntyp = [int(k) for k in list(table.loc[5])[0].split()]
        
        # Collect elements and 
        index = 6
        coords = table.loc[list(range(index, index + nat))]
        coords = [x.split() for x in coords[" CRYSTAL"]]
        coords = pd.DataFrame(coords, columns=['element', 'x', 'y', 'z'])
        
        elements = np.array(coords['element'], dtype=str)
        coordinates = np.array(coords[['x', 'y', 'z']], dtype=float)
        
        elements = elements.tolist()
        coordinates = coordinates.tolist()
        
        # Read the grid used
        index += nat + 3
        nx, ny, nz = [int(k) for k in list(table.loc[index])[0].split()]
        
        # Define variables to be returned
        index += 5
        params = {
            'params': 
                {
                    'nx': nx, 'ny': ny, 'nz': nz,
                    'nat': nat, 'ntyp': ntyp
                },
                                   
            'crystal': 
                {
                    'cell': cell,
                    'area': area,
                    'volume': volume
                },
                
            'atoms':
                {
                    'species': elements,
                    'coordinates': coordinates
                }
                 }
            
        return index, params

    def _load_data_grid(self, table, params, index):
        """
        Load the data grid present in filplot or fileout tables.
        Index is a list containing the starting index and final index to be 
        used to read the points, whereas table is a pandas DataFrame.
        Dependency of process_charge.
        
        """
        
        # Setup indexes
        index_1 = index[0]
        index_2 = index[1]
        
        if params['filetype'] == 'filplot':
            rows = list(range(5))
        elif params['filetype'] == 'fileout':
            rows = list(range(6))
        else:
            raise IOError('Something weird is happening')
        
        # Clean the data grid
        data = table.iloc[index_1:index_2]
        data = [x.split() for x in data[data.columns[0]]]
        data = pd.DataFrame(data, columns=rows)
        
        # Extract the new grid for reshaping
        nx = params['params']['nx']
        ny = params['params']['ny']
        nz = params['params']['nz']
        
        # Get the final array and reshape
        charge = np.array(data[rows], dtype=float).flatten()
        charge = charge[np.logical_not(np.isnan(charge))]  # Remove any np.nan
        charge = np.reshape(charge, (nz, ny, nx)) / \
                 QEChargeAnalyzer.bohr_to_angstrom**3

        return charge

    def _process_average(self, data, params, axis, path=None, shift=0, 
                         rescale=1000, get_inter=True, geometry_norm=False,
                         valence=True, to_file=True, to_plot=True):
        """
        Process the average request of the charge data along specific axis.
        Dependency of average_charge
        
        """
        
        # Initialize variables
        average = None
        p = None        
        
        if isinstance(axis, (int, float)):
            axis = [axis]
        
        axis_dict = {
            'x': 2, 'y': 1, 'z': 0, 'xy': (1, 2), 'yx': (1, 2), 
            'xz': (0, 2), 'zx': (0, 2), 'yz': (0, 1), 'zy': (0, 1),
            'xyz': (0, 1, 2), 'xzy': (0, 1, 2), 'yxz': (0, 1, 2),
            'yzx': (0, 1, 2), 'zxy': (0, 1, 2), 'zyx': (0, 1, 2)
            }
        axis_str = axis
        axis = axis_dict[axis]
        axis_dependence = 'xyz'.replace(axis_str, '')
        
        # Average the charge within the entire lattice cell
        if set(axis) == {0, 1, 2}:
            p = self._process_cell_average(data, params)
            
        # Make a planar average
        elif (set(axis) == {0, 1} or \
              set(axis) == {0, 2} or \
              set(axis) == {1, 2}):
            average, p = self._process_planar_average(data, params, axis, 
                                                      shift, rescale, get_inter,
                                                      geometry_norm, valence)

        # Make a linear average
        elif (set(axis) == {0} or \
              set(axis) == {1} or \
              set(axis) == {2}):
            average, p = self._process_linear_average()
        
        else:
            raise ValueError('A tuple containing the axis to be averaged '
                             'should be passed. Check the documentation.')
        
        # Save information and plot to files
        if isinstance(average, (list, np.ndarray)):
            filename = path + 'rho_' + axis_dependence
            self.save_average(average, filename, to_file, to_plot)
        
        return average, p

    def _process_cell_average(self, data, params):
        """
        Calculate the total and average charge within the cell
        """
        
        # nat = params['params']['nat']
        volume = params['crystal']['volume']
        dv = volume / (params['params']['nx'] * \
                       params['params']['ny'] * \
                       params['params']['nz'])
        
        # Normalize by number of electrons
        # el_valences = params['atoms']['electronics'].get('valence', nat)
        
        rho_tot = np.sum(data) * dv
        rho_mean = rho_tot / volume
        
        p = {
            'cell':
                {
                    'rho_tot': rho_tot,
                    'rho_mean': rho_mean
                }
             }
        
        return p 
    
    def _process_planar_average(self, data, params, axis=(1, 2), shift=0, 
                                rescale=1000, get_inter=True, geometry_norm=False,
                                valence=True):
        """
        Make the numerical average within a given plane, in this way a one-
        dimensional dependent charge is left, i.e. rho(i), with i=x,y,z.
        
        """
        
        # Define variables to extract data from the dictionary
        grid_index = {0: 'nz', 1: 'ny', 2: 'nx'}
        read_ax = list({0, 1, 2} - set(axis))[0]
        line_index = {'nx': 0, 'ny': 1, 'nz': 2}    
        ax = line_index[grid_index[read_ax]]
        
        # Extract relevant parameters
        num = params['params'][grid_index[read_ax]]
        cell = np.array(params['crystal']['cell'])
        coordinates = np.array(params['atoms']['coordinates'])
        # species = np.asarray(params['atoms']['species'])
        
        # Extract and calculate geometric quantities
        ar_ax = tuple({0, 1, 2} - {ax})
        area = np.linalg.norm(np.cross(cell[ar_ax[0], :], cell[ar_ax[1], :]))
        
        # Get the interface position, width, and charge
        if get_inter:
            shift, width, heigth = self._estimate_interface(coordinates, 
                                                            shift, ax)
        else:
            width = None
            heigth = None
            
        # Try to stimate the interface width in a simpler way
        if width is None:
            z = coordinates[:, ax] - shift
            try:
                lower_index = z[z < 0].argmax()
                upper_index = z[z > 0].argmin()
                width = z[z > 0][upper_index] - z[z < 0][lower_index]
            except:
                pass

        # Calculate the abscissa axis
        v_len = np.linalg.norm(cell[:, ax])
        v = np.linspace(0, v_len, num=num)
        v -= shift
               
        # Make planar average and shift everything around zero.
        # Interpolate and rescale points     
        average = np.mean(data, dtype=float, axis=axis)
        x = np.linspace(min(v), max(v), num=rescale)
        y = np.interp(x, v, average)

        # Normalize with average valence and check that charge is ok.
        # The next card makes sense only when filetype is filplot
        if valence:
            el_valences = params['atoms'].get('electronics', None)
            norm_val = True if bool(el_valences) else False

            # Normalize by the average valence of the atoms
            if bool(el_valences):
                mean_val = el_valences['mean_valence']
                y /= mean_val

                # DEBUG the code
                electrons = el_valences['electrons']
                volume = params['crystal']['volume']
                dv = volume / (params['params']['nx'] * \
                               params['params']['ny'] * \
                               params['params']['nz'])
                tot_chr_sum = np.sum(data) * dv
                tot_chr = integrate.simps(y, x, even='avg') * area * mean_val
                
                if abs(electrons - tot_chr) / electrons >= 0.025 or \
                   abs(electrons - tot_chr_sum) / electrons >= 0.0001:
                       
                    print('')
                    print('Total charge-electrons numbers difference is >2.5%'
                          ' or raw data-electrons difference is >0.01%.')
                    print('Due to an integration error or a charge '
                          'displacement calculation.')
                    print('Total integrated charge: {:.4f}'.format(tot_chr))
                    print('Summed up raw charge: {:.4f}'.format(tot_chr_sum))
                    print('Nominal charge: {:.4f}'.format(electrons))
                    print('')

        else:
            norm_val = False
        
        # Make a normalization of the values by the area
        if geometry_norm:
            y /= area
            norm_area = area
        else:
            norm_area = False

        # Finally save the data array
        data_l = np.stack((x, y), axis=1)
        
        # Detect where the interface is located and extract interfacial charge.
        # Calculate the interfacial, global charges, and its value at x=0.
        # At this point the interface is at zero, this either comes from user
        # input or from interface estimation by _estimate_interface
        charge_tot = integrate.simps(data_l[:, 1], data_l[:, 0], even='avg')
        charge_0 = np.interp(0, data_l[:, 0], data_l[:, 1])
        
        # Normalize charge_tot
        pts_def = len(data_l[:, 0][data_l[:, 1] > max(data_l[:, 1]) * 1e-2])
        step = abs(data_l[:, 0][0] - data_l[:, 0][1])
        norm_len = step * pts_def
        charge_tot /= norm_len

        if width is not None:
            lower_bound = - width / 2.
            upper_bound = + width / 2.
            x_inter = np.linspace(lower_bound, upper_bound, num=rescale)
            y_inter = np.interp(x_inter, data_l[:, 0], data_l[:, 1])
            charge_inter = integrate.simps(y_inter, x_inter, even='avg')
            charge_inter /= width  # Normalize charge_inter

        else:
            charge_inter = None

        # Write final dictionary
        p = {
            'axis':
                {
                    {0: 'z', 1: 'y', 2: 'x'}[read_ax]: 
                        {
                            'interface': 
                                {
                                    'position': shift,
                                    'width': width,
                                    'heigth': heigth,
                                    'rho_tot': charge_tot,
                                    'rho_0': charge_0,
                                    'rho_interface': charge_inter,
                                    'normalized_by_valence': norm_val,
                                    'normalized_by_area': norm_area,
                                    'normalized_lenght_tot': norm_len
                                },
                            'data': data_l.tolist()
                    }
                }
              }

        return data_l, p 
    
    @staticmethod
    def _estimate_interface(atoms, shift, axis):
        """
        Estimate the position of the interface and some relevant parameters.
        It returns: 
            - shift : interface position with respect to the initial disposal
                      of the atoms within the lattice cell. If the number of 
                      atoms is odd, it returns the initial provided shift.
            - width : the extension of the interface being defined by the 
                      coordinates of the atoms immediately adjacent to the 
                      interface.
            - heigth : Maximum extension of the atomic slabs.
       
        """
        
        if isinstance(atoms, list):
            atoms = np.array(atoms)
        # if isinstance(species, list):
        #    species = np.asarray(species)
        
        # Initialize variables to be returned
        width = None
        heigth = None
        
        # Get Interface information if number of atoms is even
        n = len(atoms)
        
        if n % 2 == 0:
            # species = species.reshape((n,))
            # atomic_coords = np.column_stack(atoms, species)
            coords = np.sort(atoms[:, axis])
            
            lower_bound = coords[int(n / 2 - 1)]
            upper_bound = coords[int(n / 2)]
            
            # Calculate parameters
            shift = (lower_bound + upper_bound) / 2
            width = upper_bound - lower_bound
            heigth = (coords[-1] - coords[0])
        
            # atoms[:, axis] -= shift
            
        return shift, width, heigth

    def _process_linear_average(self):
        """
        Process a linear average and returns a 2D dependent rho(i, j)
        
        """
        pass
  
    def __add__(self, other):
        """
        Define the sum between two instances as the sum of charge_data

        """
        
        if not ((self.cell == other.cell).all()):
            raise ValueError('You can not sum objects with different cells')
        
        elif not (self.charge_grid.shape == other.charge_grid.shape):
            raise ValueError('Data grid are different, you need to integrate')
        
        # Make the effective addiction
        res = deepcopy(self)
        res.charge_grid += other.charge_grid
        
        return res
    
    def __sub__(self, other):
        """
        Define two instances substraction as the subtraction of charge_data

        """
                
        if not ((self.cell == other.cell).all()):
            raise ValueError('You can not sum objects with different cells')
        
        elif not (self.charge_grid.shape == other.charge_grid.shape):
            raise ValueError('Data grid are different, integrate first')
        
        # Make the effective subtraction
        res = deepcopy(self)
        res.charge_grid -= other.charge_grid
        
        return res
    
    def __copy__(self):
        """
        Copy method for the class, can be called as: self.__copy__()

        """
        return deepcopy(self)


class QEChargeProcessor(QEChargeAnalyzer):
    """
    This class constitutes an extension of QEChargeAnalyzer and can be used to
    load/manipulate multiple charge density files and calculate different 
    charge displacements.
    
    Attributes
    ----------
    charge_grid : list of np.ndarray
        Contain the charge grid with the raw data.
    params : list of dict
        Relevant parameters collected from the charge files read from input.
    cell : list of np.ndarray
        Path to the uploaded charge files.
    filetype : list of str
        Lattice cell of the system.
    filepath : list of str
        Type of the input file, either filplot or fileout.
    charge : list of dict
        Contains relevant information related to the averaged charges.
    charge_diff : dict
        Containg information about the interface after the charge displacement
        has occurred.        
    charge_arrays: list of np.ndarray
        List containing all the numpy arrays with the averaged data. The 
        charge displacement array is appended here after the method call.

    Methods
    -------   
    read_charge(args=None, to_file=True)
        Read multiple input files and upload their data.
    multiread(args=None, to_file=True)
        Read multiple charge files at the same time, either uploading previously
        zipped data with from_file (if found) or reading charge files from scratch.
    average_charge(axis='xy', shift=0., rescale=1000, get_inter=True, valence=True)
        Average the charge of the systems in the cell or along specific axis/planes.
    displace_charge(axis='xy', shift=0., rescale=1000, operations='diff', 
                    filename=None, save_charge=True, get_inter=True, valence=True)
        Execute the charge displacement by combining all the charge arrays
        which has been previously averaged.
    get_info()
        Print information about the charge displacement
    information(obj)
        Static method working on a object, print a summary on charge displacement.
    to_file(filename=None)
        Export charge grid and parameters to files.
    save_average(data, filename=None, to_file=True, to_plot=(True))
        Save the averaged data to file and plot.
    from_file(filename=None)
        Upload charge grid and parameters from files.
    reset()
        Reset the attributes to empty values.
    
    """
    
    def __init__(self, args=None, unpack=True):
        """
        Initialize a QEChargeProcessor object, by passing a list containing all
        the paths to files to be processed.

        Parameters
        ----------
        args : list of str, optional
            List containing the files to be read for be later processed. 
            The default is None.
            
        unpack : bool, optional
            If True it will try to load each filedata from previously processed 
            and zipped data, if these are not found it will read them from
            raw charge data. The default is True.

        """
        
        # Initialize multivariables
        self.reset()
        
        # Read data if filenames are provided
        if args is not None and args != []:
            if unpack:
                self.multiread(args)
            else:
                self.read_charge(args, to_file=False)

    def read_charge(self, args=None, to_file=True):
        """
        Read charge data from Quantum Espresso charge data files for a provided
        list of arguments. It process all the files from scratch, so it can be 
        extremly slow especially if many files are processed. Use it only if 
        no zipped files has been previously saved. For a standard use, it is 
        suggested to use multiread instead.

        Parameters
        ----------
        args : list of str, optional
            List containing all the files to be processed. The default is None.
            
        to_file : bool, optional
            Whether to export the read data to zip files. Useful to be faster
            the next time the object is initialized, by calling from_file or
            multiread methods. The default is True.

        Raises
        ------
        IOError
            If no input files are provided.

        """
        
        if not args:
            raise IOError('No input file has been provided!')
       
        # Check if inputs exists and are files
        exists = list(map(os.path.exists, args))
        isfile = list(map(os.path.isfile, args))
        
        if not all(exists) or not all(isfile):
            raise IOError('Some inputs does not exists or are not files')
                
        # Read the input files passed by args
        data = list(map(self._process_charge, args))
        
        for i in range(len(data)):
            grid, params = list(data[i])
            self.charge_grid.append(grid)
            self.charge.append({})
            self.params.append(params)
            self.cell.append(params['crystal']['cell'])
            self.filetype.append(params['filetype'])
            self.filepath.append(args[i])
        
        # Save data to compressed files
        if to_file == '' or to_file:
            self.to_file(filename=self.filepath)
        elif isinstance(to_file, list):
            self.to_file(filename=to_file)

    def multiread(self, args=None, to_file=True):
        """
        Advanced read method, which tries to upload first the charge data by
        seeking for previously exported files (.json and .pkl files), and only
        if nothing is found call the general and heavy read method.
        It works on a list of filepaths passed by arguments and apply the analysis
        described above to each of them.

        Parameters
        ----------
        args : list of str, optional
            List containing all the files to be processed. The default is None.
            
        to_file : bool, optional
            Whether to export the read data to zip files. Useful to be faster
            the next time the object is initialized. The default is True.

        Raises
        ------
        IOError
            If no input files are provided.

        """
        
        if not args: 
            raise IOError('No input file has been provided!')
        
        for filename in args:
            # Try to read previously saved files
            bol = self.from_file(args=[filename])
            
            # Read raw charge data from QE pp
            if not bol:
                self.read_charge(args=[filename], to_file=False)
        
        if to_file:
            self.to_file(filename=self.filepath)
    
    def average_charge(self, axis='xy', shift=0, rescale=1000, get_inter=True,
                       geometry_norm=False, valence=True):
        """
        Average charge data along specific directions or planes.
        By default it averages the charge in the xy plane, i.e. keep rho(z).
        All the systems will be averaged adopting the same input arguments and
        the collected data are saved in charge and charge_arrays attributes.
        The last one is used by charge_displace for further elaborations.
        It only process whole cell and planar averages at the moment.

        Parameters
        ----------
        axis : str, optional
            String containing the axis to be averaged, e.g. 'x', 'xy', 'xyz'.
            The default is 'xy'.
            
        shift : int or float, optional
            Shift to be applied along the remaining direction after planar 
            averages in order to center all the interfaces around zero. 
            The default is 0.
            
        rescale : int, optional
            Rescale the final array in order to have a coherent number of points
            between all the updated systems. Very important to being able to 
            sum charge arrays with different starting grids. The default is 1000.
            
        get_inter : bool, optional
            Control whether to collect information about the interface.
            If set to True, it saves additional information to charge attribute, 
            such as the interface position, width, and charge.
            The default is True.
            
        valence : bool, optional
            To take into account the valence electrons when calculating the 
            average of the charge. Meaningful for filplot filetypes.
            The default is True.

        """
        
        # Define the path where to save data
        paths = []
        for j, p in enumerate(self.filepath):
            filepath = ''
            for x in p.split('/')[:-1]:
                filepath = filepath + str(x) + '/'
            paths.append(filepath + str())
        
        # Calculate the shift to be used for all the interfaces, based on the
        # first uploaded system
        n = len(paths)
        if get_inter:
            coordinates = np.array(self.params[0]['atoms']['coordinates'])
            
            # Calculate the right axis to extract the shift
            grid_index = {0: 'nz', 1: 'ny', 2: 'nx'}
            read_ax = list({0, 1, 2} - set(axis))[0]
            line_index = {'nx': 0, 'ny': 1, 'nz': 2}    
            ax = line_index[grid_index[read_ax]]
            
            shift, _, _ = self._estimate_interface(coordinates, shift, ax)
            get_inter = [True] + [False] * (n - 1)
        
        else:
            get_inter = [False] * n
            shift = 0        
        
        # Run the average on all the files
        data = list(map(self._process_average, self.charge_grid, self.params,
                         [axis] * n, paths, [shift] * n, [rescale] * n,
                         get_inter, [geometry_norm] * n, [valence] * n,
                         [False] * n, [False] * n))

        # Initialize charge attributes and update params dictionary
        self.charge = [{}] * n
        self.charge_arrays = []
        for i, el in enumerate(data):
            average, p = el
            
            # Update arrays and params dictionary
            self.charge_arrays.append(average)
            self.charge[i] = p

    def displace_charge(self, axis='xy', shift=0, rescale=1000, 
                        operations='diff', filename=None, save_charge=True,
                        get_inter=True, geometry_norm=False, valence=True):
        """
        It calculates the charge displacement between the arrays stored in
        charge_arrays. If this attribute is not found, average_charge is
        automatically called. To set the analysis, tune the operations argument.
        This is the only way to calculate the charge displacement for charge 
        data with different grids.
        Some relevant parameters concerning the processed charge are saved in
        the charge_diff attribute, as well as the final displacement array, 
        which is appended to charge_arrays.
        
        Parameters
        ----------
        axis : str, optional
            String containing the axis to be averaged, e.g. 'x', 'xy', 'xyz'.
            The default is 'xy'.
            
        shift : int or float, optional
            Shift to be applied along the remaining direction after planar 
            averages in order to center all the interfaces around zero. 
            The default is 0.
            
        rescale : int, optional
            Rescale the final array in order to have a coherent number of points
            between all the updated systems. Very important to being able to 
            sum charge arrays with different starting grids. The default is 1000.
            
        operations : str (or list of str), optional
            Set of operations to be done on averaged charge arrays. It process
            the arrays contained in charge_arrays (ca) attribute, always starting
            from 0th element, i.e. charge_arrays[0]. The default is 'diff'. 
            
            A summation or custom operations can be done. Allowed values:                
                - 'diff', '-', 'default' : ca[0] - ca[1] - ca[2] - ...
                - 'sum', '+' : ca[0] + ca[1] + ca[2] + ...
                - list : custom list of dimension `len(ca)-1`, contains the 
                         set of operations to be made on the ca sequence, e.g.
                         ['+', '-', '+'] -> ca[0] + ca[1] - ca[2] + ca[3]
            
        filename : str, optional
            Name of the file containing the final charge displacement array
            to be saved. Meaningful only is save_charge is set to True.
            The default is None.
            
        save_charge : bool, optional
            Decide if the charge displacement array should be saved to file. 
            The default is True.
            
        get_inter : bool, optional
            Control whether to collect information about the interface.
            If set to True, it saves additional information to charge attribute, 
            such as the interface position, width, and charge.
            The default is True.
            
        valence : bool, optional
            To take into account the valence electrons when calculating the 
            average of the charge. Meaningful for filplot filetypes.
            The default is True.

        Raises
        ------
        ValueError
            When provided input arguments are not of the correct type.
            
        """
        
        # Check input arguments for axis
        if not isinstance(axis, (str)):
            raise ValueError('Axis is wrong, check carefully input arguments')
        
        elif axis not in ['xy', 'xz', 'yz']:
            raise ValueError("Displace only works for axis = 'xy', 'xz', 'yz'")
        
        # Call averaging method if not already done   
        bol = [bool(d) for d in self.charge]
        if not all(bol):
            self.average_charge(axis, shift, rescale, get_inter, geometry_norm, valence)

        # Check if data is missing, call displace in recursive way is so
        try:
            n = len(self.charge_arrays)
            assert n > 0 
            assert n == len(self.charge)
        except:
            self.average_charge(axis, shift, rescale, get_inter, geometry_norm, valence)
            self.displace_charge(axis, shift, rescale, operations, filename, 
                                 save_charge, get_inter, geometry_norm, valence)
        
        data, p = self._process_displacement(self.charge_arrays, self.charge[0],
                                             axis, operations, shift, rescale)

        # Collect data and save interface information
        self.charge_arrays.append(data)
        self.charge_diff = p
        
        if save_charge:
            self.save_charge(data, filename)

    def get_info(self):
        """
        Print to stdout a summary about charge displacement.
        
        """
        
        _, _ = self.information(self)
    
    @staticmethod
    def information(obj):        
        """
        Extract information from charge_arrays and charge_diff attributes.

        Parameters
        ----------
        obj : tribchem.electronics.charge.analyzer.QEChargeProcessor
            QEChargeProcessor object of which you want to colelct information.

        Raises
        ------
        ValueError
            If the object is not initialized yet.

        Returns
        -------
        array : np.ndarray
            Charge array containing the points.
            
        chg : dict
            Contains the average data associated with the correpsponding axis.

        """
        
        # Print charge displacement informatioon
        print('')
        print('\033[4m' + 'Charge displacement' + '\033[0m')
        
        # Verify that the displacement has been done
        try:
            assert isinstance(obj.charge_arrays, list)
            assert(len(obj.charge_arrays) == len(obj.filepath) + 1)
            assert isinstance(obj.charge_diff, dict)
        except:
            raise ValueError('Charge displacement not calculated yet')
        print('') 

        # Extract and print data
        for ax in obj.charge_diff.keys():            
            inter = obj.charge_diff[ax]['interface']
            print('\u03C1({})\n'.format(ax))
            
            # Cell information
            if inter['position'] is not None:
                print('Interface position to cell: {0:.3f} \u212B'.format(inter['position']))
            if inter['width'] is not None:
                print('Interface width: {0:.3f} \u212B'.format(inter['width']))
            if inter['heigth'] is not None:
                print('Atomic thickness: {0:.3f} \u212B'.format(inter['heigth']))

            # Charge information
            if inter['rho_tot'] is not None:
                print('Total charge ((1/\u0394{0})\u222b|\u03C1({1})|d{2}): {3:.8f} e-/\u212B^3'
                      .format(ax, ax, ax, inter['rho_tot']))
            if inter['rho_0'] is not None:
                print('Zero charge (\u03C1({0}=0)): {1:.8f} e-/\u212B^3'
                      .format(ax, inter['rho_0']))
            if inter['rho_interface'] is not None:
                print('Interfacial charge ((1/\u0394{0})\u222b|\u03C1({1})|d{2}): '
                      '{3:.8f} e-/\u212B^3'.format(ax, ax, ax, inter['rho_interface']))
            if 'normalized_by_valence' in inter.keys():
                print('Normalized by valence: {}'.format(inter['normalized_by_valence']))
            if inter['normalized_lenght_tot'] is not None:
                print('Normalized length for total charge: {0:.3f} \u212B\n'
                      .format(inter['normalized_lenght_tot']))
        
        array = obj.charge_arrays[-1]
        chg = obj.charge_diff
        
        return array, chg
    
    def to_file(self, filename=None):
        """
        Export charge matrix as a pickle fil, and params as a json dict.

        Parameters
        ----------
        filename : list of str, optional
            List containing all the filename to be used to export charge grid,
            params, and charge attributes to file. If nothing is passed try
            to use a default name from filepaths. The default is None.

        """
        
        if filename is None or not isinstance(filename, list):
            self.to_file(filename=self.filepath)
        
        else:
            f_pkl = [f + '.pkl' for f in filename]
            f_json = [f + '.json' for f in filename]
            if self.charge == []:
                chg = [{}] * len(filename)
            else:
                chg = self.charge.copy()
            
            # Save the picke binary file and the json dictionary
            for i in range(len(filename)):
                
                with open(f_pkl[i], 'wb') as f:
                    pickle.dump(self.charge_grid[i], f)
                    
                with open(f_json[i], 'w') as f:
                    json.dump([jsanitize(self.params[i]), 
                               jsanitize(chg[i])], f)

    def save_charge(self, charge, filename=None):
        """
        Save a charge array to file, useful to save displacement arrays.
        It can be called automatically when running charge_displace method.
        In principle it can be used also on any array passed as charge input.

        Parameters
        ----------
        charge : np.ndarray or list
            Array to be saved to file, you can call it on object attributes.
            It is suggested to call it on the elements of the charge_array 
            attribute.
            
        filename : str, optional
            Filename where to save the array. If nothing is passed try to use a 
            default name. The default is None.

        Raises
        ------
        ValueError
            If charge is not a valid element to be exported to file.

        """
          
        # Define a default path and filename to save files
        if filename is None or not isinstance(filename, str):
            filename = self.filepath[0]
            path = ''
            
            for x in filename.split('/')[:-1]:
                path += str(x) + '/'
            file_dat = path + 'rho_disp' + '.dat'
            
        elif isinstance(filename, str):
            file_dat = filename
        
        else:
            raise ValueError('filename should be a string!')

        try:
            np.savetxt(file_dat, charge)
        except:
            raise ValueError('No charge displacement has been evaluated yet. '
                             'Run first the displace_charge method')          

    def from_file(self, args=None):
        """
        Load charge matrix from a pickle file and params from a json dict.
        
        Parameters
        ----------
        args : list of str, optional
            List containing all the files to be read. The default is None.
        
        """
        
        # Check the type of inputs and return False if not correct
        if args is None:
            return False
        
        elif not isinstance(args, list):
            try: 
                args = list(args)
            except:
                return False
            
            if not all(isinstance(x, str) for x in args):
                return False
        
        data = list(map(super().from_file, args))
        bol = [True if not d else False for d in data]
        
        # All the data exists
        if not all(bol):
            for i in range(len(args)):
                grid, json, filename = data[i]
                
                self.charge_grid.append(grid)
                self.filepath.append(filename)
                
                # Read only the params dictionary
                if isinstance(json, list):
                    js = json[0]
                self.params.append(js)
                self.cell.append(np.array(js['crystal']['cell']))
                self.filetype.append(js['filetype'])  
                
                # Append the charge dictionary
                self.charge.append(json[1])
        
        # Some inputs files are not found
        else:
            return False
            
        return True

    def reset(self):
        """
        Reset the attributes of the object so it can be rerun from scratch.

        """

        self.charge_grid = []
        self.params = []
        self.cell = []
        self.filetype = []
        self.filepath = []
        self.charge = []
    
    def _process_displacement(self, data, charge, axis='xy', operations='diff',
                              shift=0., rescale=1000):
        """
        Process the charge displacement by performing a set of operations on
        the list of charge attributes. Dependency of displace_charge.

        """
        
        # Calculate the set of operations to be done
        n = len(data)
        if operations == 'diff' or operations == '-' or operations == 'default':
            op = ['-'] * (n - 1)
        elif operations == 'sum' or operations == '+':
            op = ['+'] * (n - 1)
        elif isinstance(operations, (list, tuple)) and len(operations) == n - 1:
            op = operations
        else:
            raise ValueError('Specify a set of operations correctly. '
                             'See the documentation for more information')
        
        # Calculate the displacement from the set of given operations
        displacement = data[0].copy()
        for i in range(n - 1):
            command = op[i]
            
            if command == '-':
                displacement[:, 1] -= data[i + 1][:, 1]
            elif command == '+':
                displacement[:, 1] += data[i + 1][:, 1]
            else:
                raise ValueError('Specify a set of operations correctly. See '
                                 'the documentation for more information.')
        
        # Extract information about the new interface
        p = self._process_planar_displacement(displacement, charge, axis, 
                                              rescale)        
       
        return displacement, p
     
    def _process_planar_displacement(self, data, charge, axis='xy', rescale=1000):
        """
        Calculate the charge displacement along a given plane. Dependency of
        _process_displacement.

        """
        
        # Grep the correct index
        read_ax = list({0, 1, 2} - set(axis))[0]
        ax = 'xyz'.replace(axis, '')
        
        # Extract data from the base element of the displacement
        shift = charge['axis'][ax]['interface']['position']
        width = charge['axis'][ax]['interface']['width']
        heigth = charge['axis'][ax]['interface']['heigth']

        # Calculate the total integral of the curve and get charge at z=0
        charge_tot = integrate.simps(np.abs(data[:, 1]), data[:, 0], even='avg')       
        charge_0 = np.interp(0, data[:, 0], data[:, 1])

        # Normalize charge_tot
        pts_def = len(data[:, 0][abs(data[:, 1]) > max(abs(data[:, 1])) * 1e-2])
        step = abs(data[:, 0][0] - data[:, 0][1])
        norm_len = step * pts_def
        charge_tot /= norm_len
        
        if width is not None:
            lower_bound = - width / 2.
            upper_bound = + width / 2.
            x_inter = np.linspace(lower_bound, upper_bound, num=rescale)
            y_inter = np.interp(x_inter, data[:, 0], data[:, 1])
            charge_inter = integrate.simps(np.abs(y_inter), x_inter, even='avg')
            charge_inter /= width  # Normalize charge_inter
            
        else:
            charge_inter = None

        # Write final dictionary
        p = {
            {0: 'z', 1: 'y', 2: 'x'}[read_ax]: 
                {
                    'interface': 
                        {
                            'position': shift,
                            'width': width,
                            'heigth': heigth,
                            'rho_tot': charge_tot,
                            'rho_0': charge_0,
                            'rho_interface': charge_inter,
                            'normalized_lenght_tot': norm_len
                        },
                    'data': data.tolist()
                 }
             }
        
        return p
    
    def _process_linear_displacement(self):
        """
        Process the charge displacement of planes of charge, e.g. rho(i,j) where
        i, j are two among x, y, z.

        """
        pass


# =============================================================================
# Functions and Tools
# =============================================================================      
    
def run_displacement(files, operations='default', **kwargs):
    """
    Run a fast charge displacement and save any possible array. 
    Parameters allowed by kwargs are: unpack, axis, shift, rescale, filename, 
    save_charge, get_inter, valence.

    Parameters
    ----------
    files : list of str
        Contains the files to be processed.
        
    operations : str or list, optional
        Contain the operations to be done among the average charges. See the
        QEChargeProcessor documentation for more info. The default is 'default'.

    Returns
    -------
    chg_tot : tribchem.electronics.charge.analyzer.QEChargeAnalyzer
        Object containing the charge_tot file
        
    obj : tribchem.electronics.charge.analyzer.QEChargeProcessor
        Object containing the charge displacement    
    
    """
    
    # Define default values from kwargs
    unpack = kwargs.get('unpack', True)
    axis = kwargs.get('axis', 'xy')
    shift = kwargs.get('shift', 0) 
    rescale = kwargs.get('rescale', 1000)
    
    # remain = 'xyz'.replace(axis, '')
    
    chg_tot = QEChargeAnalyzer(files[0], unpack)
    chg_tot.detour_average(['xyz', axis], shift, rescale, get_inter=True, valence=True, 
                           to_file=False, to_plot=False)
    print(''.center(50, '*'))
    print('')
    print('TOTAL CHARGE')
    chg_tot.get_info(verbose=False)
    
    obj = QEChargeProcessor(files, unpack)
    obj.average_charge(axis, shift, rescale, get_inter=True, valence=True)
    obj.displace_charge(axis, shift, rescale, operations, save_charge=False,
                        get_inter=True, valence=True)
    
    print(''.center(50, '*'))
    print('')
    print('FINAL INTERFACE')
    obj.get_info()
    
    # Save average data to files for further processing
    obj.save_charge(obj.charge_arrays[0], filename='rho_tot.dat')
    for k, array in enumerate(obj.charge_arrays[1:-1]):
        obj.save_charge(obj.charge_arrays[k + 1], filename='rho_' + str(k + 1) + '.dat')
    obj.save_charge(obj.charge_arrays[-1], filename='rho_disp.dat')
    
    return chg_tot, obj


def gui_chgdisp():
    """
    Minimal terminal interface to run a simple charge displacement calculation.
    Search for name_tot and name_# files, where name is passed by input.

    Returns
    -------
    None.

    """
    
    # Header
    if sys.stdout.isatty():
        rows, columns = os.popen('stty size', 'r').read().split()
    else:
        columns = 75
    
    print('')
    print(''.center(int(columns), '='))
    print(str(' Program QEChargeAnalyzer v.' + str(__version__) + ' ')
          .center(int(columns), '='))
    print(''.center(int(columns), '='))
    print('')
    
    print('This program is part of the Tribchem python package for high throughput '
          'simulations of materials.\n')
    
    print('')
    print('Author      : ' + __author__)
    print('Copyright   : ' + __copyright__)
    print('Contact     : ' + __contact__)
    print('Please cite : ' + __related__)
    print(''.center(int(columns), '='))
    print('')
    
    now = datetime.datetime.now()
    print('Program starts on ' + str(now.day) + ' ' + str(now.strftime("%B")) + \
          ' ' + str(now.year) + ' at ' + str(now.hour) + ':' + str(now.minute) + \
          ':' + str(now.second))
    print('')
    
    name = str(input('Chargefile Prefix [charge] : ') or 'charge')
    op = str(input('Type of analysis [default] : ') or 'default')
    
    if op not in ['default', 'diff', 'sum', '-', '+'] and not isinstance(op, list):
        print('')
        sys.exit('Wrong type of analysis')
    
    # Read the files from folder and isolate the correct ones
    p = os.getcwd()
    files = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
    charges = files.copy()
    for f in files:
        if not f.startswith(name + '_'):
            charges.remove(f)
        elif f != 'charge_tot':
            residual = f.split('_')[1]
            try:
                residual = int(residual)
            except:
                charges.remove(f)
    
    # Display the files that have been found and ask of proceeding
    print('')
    if charges == []:
        sys.exit('Seeking for ' + name + '_tot and ' + name + '_#. No files found.')
    else:
        print('The following files have been found:')
        for el in charges:
            print(el)
        proceed = str(input('\nDo you want to proceed? [y|n] ') or 'y')
    print('')
        
    # Run the calculation
    if proceed == 'y':
        # Put the object in the correct order
        charges.sort(reverse=True)
        numbers = np.sort(charges[1:]).tolist()
        chg = [charges[0]] + numbers
        data = run_displacement(chg, operations=op)
    else:
        sys.exit('Analysis interrupted by user')
    
    return data


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    chg, obj = gui_chgdisp()
