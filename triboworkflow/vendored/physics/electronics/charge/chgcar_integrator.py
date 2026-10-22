# --- TribChem code vendored by TriboWorkFlow ---------------------------------
# Unchanged except for the imports. Do not edit: tests/vendoring.py
# compares this file with the original.
# Source: tribchem/physics/electronics/charge/chgcar_integrator.py
# Version: tribchem_devel-align_elisa_w_devel
# Source SHA-256: 5a9d0d009fb6dd3cb046db56108a861d5d2946d2c81fc9c0b9065170119ea3ae
# Copied: 2026-10-22
# Licence: CC BY-SA 4.0, see NOTICE
# Kind: whole
# Imports: tribchem.* -> triboworkflow.vendored.*
# --- end of the TriboWorkFlow header -----------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 14:23:46 2021

Module containing different functions and tools to read and analyze CHGCAR files.

Author: Jacopo Mascitelli (jmascitelli)
Copyright 2021, Prof. M.C. Righi, TribChem, ERC-SLIDE, University of Bologna

"""

__author__ = 'Jacopo Mascitelli'
__copyright__ = 'Copyright 2021, Prof. M.C. Righi, TribChem, ERC-SLIDE, University of Bologna'
__contact__ = 'clelia.righi@unibo.it'
__date__ = 'April 12th, 2020'


from pymatgen.io.vasp.outputs import Chgcar
import matplotlib.pyplot as plt
import numpy as np 
import scipy.integrate as intgrt


def extract_chgcar_data(input_chgcar):
	"""
	Loads CHGCAR input file(s) using Chgcar class from pymatgen library.

	Parameters
    ----------
    input_chgcar: str or list (of str)
        This is the location of the single CHGCAR file, or a list containting 
		the location of more CHGCAR files.

	Returns
    -------
    A Chgcar class object or a list of Chgcar class objects

	"""
    
    # Read a single CHGCAR file
	if isinstance(input_chgcar, str):
		try:
			chgcar = Chgcar.from_file(input_chgcar)
		except:
			raise ValueError('Wrong element: the input file provided is not a '
							 'CHGCAR file')

    # Read a list of CHGCAR files
	elif isinstance(input_chgcar, list):
		chgcar=[]
		for i, el in enumerate(input_chgcar):
			try:
				chgcar.append(Chgcar.from_file(el))
			except:
				raise ValueError('Wrong element: {}th input file provided is not '
								 'a CHGCAR file'.format(i))

	else:
		raise TypeError('Wrong input file: input_chgcar should be either a str '
						'or a list. Provided: {}'.format(type(input_chgcar)))

	return chgcar

def chgcar_average_along_axis(chgcar_object, axis, to_plot=False):
	"""
	Finds average along the given axis and eventually plots the final curve.

	Parameters
    ----------
    input_chgcar: Chgcar class object
        Could be one object or a combination of them e.g. 
        input_chgcar=chgcar_obj_1 + chgcar_obj_2 - chgcar_obj_3.

	axis: int
		Specifies along which axis to find the average value (0:x, 1:y, 2:z)

	to_plot: bool, optional
		If True, a plot of the obtained curve is shown

    Returns
    -------
    	A list of two lists. The first list contains the values of the selected
    	axis going from 0 to the length of the corresponding lattice vectors.
    	The second list contains the average charge density (in e/A^3)along 
    	the specified axis. Both lists have the same length so as to allow integration.

	"""
    
    # Read the CHGCAR objects
	try:
		lattice = chgcar_object.structure.lattice.abc
		vol = chgcar_object.structure.volume
		avg = chgcar_object.get_average_along_axis(axis)
	except:
		raise ValueError('Wrong element: the input object is not a '
						 'Chgcar pymatgen object')	
    
    # Make the average
	avg /= vol
	x = np.linspace(0, lattice[axis], len(avg))

	# Make plot
	if to_plot:
		plt.plot(x, avg)
		plt.show()

	return [x, avg]

def charge_displacement(chgcar_object, axis, p1=None, p2=None, rescale=10000, 
                        to_plot=False):
    """
    Given a Chgcar object, find the average charge density along the specified 
    axis and integrates it between two points.
    
    Parameters
    ----------
    input_chgcar: Chgcar class object
        Could be one object or a combination of them e.g. 
        input_chgcar = chgcar_obj_1 - chgcar_obj_2 - chgcar_obj_3.
    
    axis: int
        Specifies along which axis to find the average value (0:x, 1:y, 2:z)
    
    p1, p2: float
        Points along the given axis which will serve as the integration limits.
    
    rescale: int
        Number of points used to perform integration. Default is 10,000
    
    to_plot: bool, optional
        If True, a plot of the average charge density is shown.
    
    Returns
    -------
    tuple
        The integral of the charge density averaged along an axis
        ided by the width of the integrated region. The result is in e/A^3.
        Also the x,y average points of the curve and the interface width are
        returned.
    
    """
    
    # Make the planar average
    xy = chgcar_average_along_axis(chgcar_object, axis, to_plot)
    min_x = np.min(xy[0])
    max_x = np.max(xy[0])

    # Set the integration points
    p1 = min_x if (p1 is None or p1 < min_x or p1 > max_x) else p1
    p2 = max_x if (p2 is None or p2 < min_x or p2 > max_x) else p2
    if p1 > p2: p1, p2 = p2, p1

    # Do the integration in the selected interval
    x_new = np.linspace(p1, p2, rescale)
    y_new = abs(np.interp(x_new, xy[0], xy[1]))
    integral = intgrt.simpson(y_new, x=x_new) / (p2 - p1)

    # Define other parameters to be returned
    curve_integral = [x_new, y_new]
    width = p2 - p1

    return integral, width, xy, curve_integral

def chgcar_algebra_from_list(chgcar_list, operation_list, axis=2, to_plot=False):
    """
    Given a list of Chgcar objects and a list of operations ('+' or '-'),
    performs the operations and returns the resulting chgcar object.

    Parameters
    ----------
    chgcar_list: list 
        List of Chgcar objects.
	
    operation_list: list
        List of strings. Allowed elements are '+' and '-'.
        The n-th element of this list is the operation that will be performed
        between the n-th and the (n+1)-th element of chgcar_list.

    axis: int
        Specifies along which axis to find the average value (0:x, 1:y, 2:z)

    plot: bool, optional
        If True, a plot of the obtained curve is shown

    Returns
    -------
        A Chgcar object, combination of the objects given in chgcar_list.

    """

    chgcar_object = chgcar_list[0]

    for n in range(len(operation_list)):
        if '+' in operation_list[n]:
            chgcar_object += chgcar_list[n+1]
        elif '-' in operation_list[n]:
            chgcar_object -= chgcar_list[n+1]		
        else:			
            raise ValueError('Wrong element: expected "+" or "-"')

    return chgcar_object

def chgcar_algebra_from_file(chgcar_list, operation_list, axis=2, to_plot=False):
    """
    Given a list of CHGCAR files and a list of operations ('+' or '-'),
    performs the operations and returns the resulting chgcar object.

    Parameters
    ----------
    chgcar_list: list of str
        List of CHGCAR paths to files.

    operation_list: list
        List of strings. Allowed elements are '+' and '-'.
        The n-th element of this list is the operation that will be performed
        between the n-th and the (n+1)-th element of chgcar_list.

    axis: int
       	Specifies along which axis to find the average value (0:x, 1:y, 2:z)

    plot: bool, optional
    	If True, a plot of the obtained curve is shown

    Returns
    -------
       	A Chgcar object, combination of the objects given in chgcar_list.

    """

    chgcar_object_list = extract_chgcar_data(chgcar_list)
    chgcar_object = chgcar_algebra_from_list(chgcar_object_list, operation_list, 
                                             axis, to_plot=False)

    return chgcar_object
