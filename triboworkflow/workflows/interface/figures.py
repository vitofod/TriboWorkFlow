# Author: Vito Foderà
# Created: 2026-11-15

"""Figures of the interface workflow, sized for one column of an article.

Every figure is laid out for one column of a two-column article: no title, a
width of 8.5 cm, serif text of 8 points (ticks, legends) and 9 points (labels
of the axes). Each is saved as a PNG of 600 dpi and as a PDF, whose vector
paths go into LaTeX as they are, both cropped to what is drawn.

- ``pes_figure``: the potential energy surface interpolated by the steps
  ``plotpes`` and ``plotpes_scf`` (energies in J/m^2) as a filled contour map
  over two cells along each axis, on a grid of 300 x 300 points, with 40
  levels of the colour map ``RdYlBu_r`` and the minimum of the map at zero:
  the colour bar gives the energy above the minimum, Delta E in J/m^2, as tall
  as the map. With ``paths``, the minimum energy paths of the step ``shear``
  are drawn on it, the map then covering them.
- ``profile_figure``: Delta E and the shear stress along each minimum energy
  path (step ``shear``).
- ``ppes_figure``: the perpendicular PES, the energy against the distance
  between the slabs, with the UBER curve when it was fitted (step ``ppes``).
- ``charge_figure``: the planar average of the charge displacement along z,
  with the region between the slabs shaded (step ``charge``).

Notes
-----
To remove: the grid, the levels, the colours and the labels of the axes of
the map are those of TribChem's ``PES.plot`` (``physics/tribology/pes.py
:287-356``), as ``get_pes`` called it (``potensurf.py:111-113``: two cells,
300 x 300 points); ``PES.plot`` drew one path on it, in black (``mep``,
lines 326-327). Its figure (7 x 7 inches, title, text of 12 and 15 points, colour
bar taller than the map) is not drawn: the step calls ``get_pes`` without a
file for it (DEV-60). The PPES and the charge displacement follow TribChem's
``plot_ppes`` and ``plot_charge`` (``highput/utils/plotting.py:571-820``),
which drew the database of many interfaces outside the workflow.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# Cells along x and y, points of the grid along each, levels and colour map of the map.
# To remove: get_pes called PES.plot with extent=(2, 2) and mpts=(300j, 300j); level=40 and
# RdYlBu_r are the defaults of PES.plot.
EXTENT = (2, 2)
POINTS = 300
LEVELS = 40
COLOUR_MAP = 'RdYlBu_r'
AXIS_LABEL = r'distance ($\AA$)'
# The map is the energy above its minimum, not the adhesion energy.
# To remove: PES.plot labels the bar E_adh (J/m^2) (DEV-60).
BAR_LABEL = r'$\Delta E$ (J/m$^2$)'
# One column of a two-column article, in inches, and the height of the figures of curves.
COLUMN_WIDTH = 8.5 / 2.54
CURVE_HEIGHT = 0.75 * COLUMN_WIDTH
# Points: axis labels and label of the colour bar, tick labels and legends.
LABEL_SIZE = 9
TICK_SIZE = 8
# Every text in serif, the mathematical text too.
STYLE = {'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'}
# Inches around the map: left (label and ticks of y), bottom (label and ticks
# of x), top; then the gap between map and colour bar, its width and the room
# for its ticks and label on the right.
LEFT, BOTTOM, TOP = 0.50, 0.42, 0.08
GAP, BAR, RIGHT = 0.08, 0.12, 0.55
PNG_DPI = 600
# The three minimum energy paths of the step shear: line, colour and label of each. Colours
# apart from those of the map, since two paths can run on the same line.
PATHS = {'bs_line': ('-', 'black', 'best straight line'), 'x': ('--', '#e7298a', 'along x'),
         'y': (':', '#1b9e77', 'along y')}
# Margin of the map of the paths around them, as a fraction of their larger extent.
PATH_MARGIN = 0.05
# Colour of the curves; of the UBER curve; of the region between the slabs.
INK, FIT_COLOUR, REGION_COLOUR = 'black', 'tab:red', '0.85'


def figure_paths(png):
    """PNG and PDF of a figure: ``<name>.png`` and ``<name>.pdf``."""
    return png, png.with_suffix('.pdf')


def save_figure(fig, png, pdf, stamp=None):
    """Save a figure as a PNG of ``PNG_DPI`` dots per inch and as a PDF, cropped to what is drawn.

    The PDF has no creation date, so that the same figure gives the same file.
    ``stamp``, the stamp of the analysis (``outputs.analysis_stamp``), goes in
    the metadata of both: ``Comment`` of the PNG, ``Subject`` of the PDF.
    """
    fig.savefig(str(png), dpi=PNG_DPI, bbox_inches='tight', pad_inches=0.02,
                metadata={'Comment': stamp})
    fig.savefig(str(pdf), format='pdf', bbox_inches='tight', pad_inches=0.02,
                metadata={'CreationDate': None, 'Subject': stamp})


def _label(ax, xlabel, ylabel):
    """Labels of the axes and size of the ticks of the figures for an article."""
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)


def _legend_above(ax):
    """The legend of the paths in one row above the axes, where it hides no curve."""
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncols=len(PATHS), frameon=False,
              fontsize=TICK_SIZE - 1, handlelength=1.8, columnspacing=1.0, borderaxespad=0.2)


def pes_map(solver, bounds=None, rows=False):
    """Grid and energies of the map of a PES.

    Parameters
    ----------
    solver : triboworkflow.vendored.physics.tribology.pes.PES
        An interpolation of the PES with its cell (``cell_2d``).
    bounds : tuple of float or None, optional
        ``(x0, x1, y0, y1)`` of the grid, in angstrom; by default from 0 to
        ``EXTENT`` times the extent of the cell along x and along y.
    rows : bool, optional
        Evaluate the interpolation one row of the grid at a time: the memory
        of one evaluation grows with the points times the centres of the
        interpolation. False: all the grid at once.

    Returns
    -------
    x, y : numpy.ndarray
        Coordinates of the grid, ``(POINTS, POINTS)``.
    z : numpy.ndarray
        Energies of the interpolation on the grid, minus their minimum.
    """
    if bounds is None:
        a = np.sum(np.abs(solver.cell_2d[:, 0]))
        b = np.sum(np.abs(solver.cell_2d[:, 1]))
        bounds = (0, EXTENT[0] * a, 0, EXTENT[1] * b)
    x0, x1, y0, y1 = bounds
    steps = complex(0, POINTS)
    x, y = np.mgrid[x0:x1:steps, y0:y1:steps]
    if rows:
        z = np.array([solver.rbf(xr.reshape(1, -1), yr.reshape(1, -1)).ravel()
                      for xr, yr in zip(x, y)])
    else:
        z = solver.rbf(x.reshape(1, -1), y.reshape(1, -1)).reshape(POINTS, POINTS)
    return x, y, z - np.min(z)


def path_bounds(paths):
    """Bounds of a map that holds every point of ``paths``, with a margin around them.

    Parameters
    ----------
    paths : dict
        Name of a path -> array ``(n, 2)`` of its points x, y in angstrom.

    Returns
    -------
    tuple of float
        ``(x0, x1, y0, y1)``: the extent of the points, larger on each side by
        ``PATH_MARGIN`` times the larger of their extents along x and y.
    """
    points = np.vstack([np.asarray(path)[:, :2] for path in paths.values()])
    low, high = points.min(axis=0), points.max(axis=0)
    margin = PATH_MARGIN * max(high - low)
    return low[0] - margin, high[0] + margin, low[1] - margin, high[1] + margin


def pes_figure(solver, png, pdf, paths=None, stamp=None):
    """Draw the map of a PES for one column of an article and save it twice.

    Parameters
    ----------
    solver : triboworkflow.vendored.physics.tribology.pes.PES
        An interpolation of the PES in J/m^2: the one of ``get_pes``, or the
        one on which the step ``shear`` found its paths.
    png, pdf : path-like
        Files written: a PNG of ``PNG_DPI`` dots per inch and a PDF.
    paths : dict or None, optional
        Minimum energy paths drawn on the map: name (a key of ``PATHS``) ->
        array ``(n, 2)`` of the points x, y in angstrom. The map then covers
        them, with a margin (``path_bounds``), is evaluated one row at a time
        (``pes_map``) and has their legend above it.
    stamp : str or None, optional
        Stamp of the analysis, in the metadata of the files (``save_figure``).

    Notes
    -----
    The map keeps the ratio of its axes: the height of the figure follows
    from the width of one column. The colour bar is placed by hand next to
    the map, with its height. The files are cropped to what is drawn, so
    that no label is cut: they can be a little narrower than the column, and
    LaTeX scales them to it. The figure is closed once saved.
    """
    if paths:
        x, y, z = pes_map(solver, path_bounds(paths), rows=True)
    else:
        x, y, z = pes_map(solver)
    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    width = COLUMN_WIDTH - LEFT - GAP - BAR - RIGHT
    height = width * (y1 - y0) / (x1 - x0)
    total = BOTTOM + height + TOP
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(COLUMN_WIDTH, total))
        try:
            ax = fig.add_axes([LEFT / COLUMN_WIDTH, BOTTOM / total, width / COLUMN_WIDTH,
                               height / total])
            cax = fig.add_axes([(LEFT + width + GAP) / COLUMN_WIDTH, BOTTOM / total,
                                BAR / COLUMN_WIDTH, height / total])
            filled = ax.contourf(x, y, z, LEVELS, cmap=COLOUR_MAP)
            # As in the original figure: no white lines between the levels in the PDF.
            filled.set_edgecolor('face')
            filled.set_linewidth(1e-12)
            for name, path in (paths or {}).items():
                style, colour, label = PATHS[name]
                path = np.asarray(path)
                ax.plot(path[:, 0], path[:, 1], style, color=colour, lw=1.3, label=label)
            if paths:
                _legend_above(ax)
            ax.set_aspect('equal')
            ax.axis([x0, x1, y0, y1])
            _label(ax, AXIS_LABEL, AXIS_LABEL)
            bar = fig.colorbar(filled, cax=cax)
            bar.set_label(BAR_LABEL, rotation=270, labelpad=12, fontsize=LABEL_SIZE)
            bar.ax.tick_params(labelsize=TICK_SIZE)
            save_figure(fig, png, pdf, stamp)
        finally:
            plt.close(fig)


def profile_figure(profiles, png, pdf, stamp=None):
    """Delta E and the shear stress along each minimum energy path, in two panels.

    Parameters
    ----------
    profiles : dict
        Name of a path (a key of ``PATHS``) -> its profile, the array of
        ``shear.output``: columns x, y, lxy, dvx, dvy, vz, ve, force. The
        panel above draws ``ve`` (J/m^2, minimum at zero), the one below the
        shear stress, 10 times ``force`` (J/m^2 per angstrom is 10 GPa), both
        against ``lxy``, the length along the path in angstrom.
    png, pdf : path-like
    stamp : str or None, optional
        Stamp of the analysis (``save_figure``); the same in the two figures below.

    Notes
    -----
    This figure and the two below are drawn without ``pyplot``, so without a
    backend that needs a display: the orchestrator draws them on the login
    node.
    """
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(COLUMN_WIDTH, 1.2 * CURVE_HEIGHT), layout='constrained')
        top, bottom = fig.subplots(2, 1, sharex=True)
        for name, profile in profiles.items():
            style, colour, label = PATHS[name]
            profile = np.asarray(profile)
            top.plot(profile[:, 2], profile[:, 6], style, color=colour, lw=1.0, label=label)
            bottom.plot(profile[:, 2], 10 * profile[:, 7], style, color=colour, lw=1.0)
        bottom.axhline(0, color='0.6', lw=0.5)
        _legend_above(top)
        _label(top, '', BAR_LABEL)
        _label(bottom, r'length along the path ($\AA$)', r'shear stress (GPa)')
        fig.align_ylabels((top, bottom))
        save_figure(fig, png, pdf, stamp)


def ppes_figure(x, energy, png, pdf, xlabel, ylabel, curve=None, stamp=None):
    """The perpendicular PES: the energy of each shift, and the UBER curve when fitted.

    Parameters
    ----------
    x, energy : array_like
        Distance between the slabs (or shift of the slab on top) in angstrom,
        and energy in J/m^2, one value per calculation.
    png, pdf : path-like
    xlabel, ylabel : str
        Labels of the axes.
    curve : tuple of array_like or None, optional
        ``(x, energy)`` of the UBER curve fitted, drawn dashed; None: no fit.
    """
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(COLUMN_WIDTH, CURVE_HEIGHT), layout='constrained')
        ax = fig.subplots()
        ax.axhline(0, color='0.6', lw=0.5)
        ax.plot(x, energy, 'o-', color=INK, lw=0.8, ms=3, label='calculations')
        if curve is not None:
            ax.plot(*curve, '--', color=FIT_COLOUR, lw=1.0, label='UBER fit')
            ax.legend(fontsize=TICK_SIZE - 1, frameon=False, loc='lower right')
        _label(ax, xlabel, ylabel)
        save_figure(fig, png, pdf, stamp)


def charge_figure(z, displacement, region, png, pdf, stamp=None):
    """The planar average of the charge displacement along z, the region between the slabs shaded.

    Parameters
    ----------
    z, displacement : array_like
        Position along z in angstrom, with the contact plane of the slabs at
        0, and the planar average of the charge displacement in e/angstrom^3.
    region : tuple of float
        Lower and upper limit of the region between the slabs, in angstrom,
        on the same z.
    png, pdf : path-like
    """
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(COLUMN_WIDTH, CURVE_HEIGHT), layout='constrained')
        ax = fig.subplots()
        ax.axvspan(*region, color=REGION_COLOUR, lw=0)
        ax.axhline(0, color='0.6', lw=0.5)
        ax.plot(z, displacement, '-', color=INK, lw=0.8)
        ax.set_xlim(np.min(z), np.max(z))
        _label(ax, r'$z$ ($\AA$)', r'$\Delta\rho$ (e/$\AA^3$)')
        save_figure(fig, png, pdf, stamp)
