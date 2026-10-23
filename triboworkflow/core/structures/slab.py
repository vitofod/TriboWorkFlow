# Author: Vito Foderà
# Created: 2026-10-23

"""Slabs for the workflows: the pymatgen ``Slab`` the interface workflow uses.

The interface matching and the high-symmetry analysis (vendored modules)
work on pymatgen ``Slab`` objects and use their ``miller_index``,
``oriented_unit_cell``, ``shift``, ``scale_factor``, ``reconstruction`` and
site properties. A slab file becomes such a ``Slab`` along one of two paths:

- ``database``: a pymatgen JSON slab, used as stored: ``Slab.from_dict``;
- ``custom_slab``: any other structure file, turned into a ``Slab`` by
  ``custom_slab`` (Miller indices from the settings, ``oriented_unit_cell``
  equal to the structure, ``shift = 0``, ``scale_factor = (1, 1, 1)``, only
  ``magmom`` among the site properties), stored as a dict without lattice
  matrix and read back with ``Slab.from_dict``.

Both rebuild the lattice with ``Lattice.from_parameters``, which the ``Slab``
constructor already does (``reorient_lattice=True``, pymatgen
``core/surface.py:86``, ``147-158``): c along z, a along x. Both lose the
labels. ``workflow_slab`` makes these transformations explicit and keeps the
structure as read:

- the cell must be right-handed with c perpendicular to a and b; any other
  cell would be mirrored or tilted by ``from_parameters``, and is an error;
- the rotation from the frame of the file to that of the slab is returned,
  with the data dropped and added;
- QE labels other than the element (e.g. ``Fe1``, ``Fe2``) are an error: the
  workflow code does not carry them along the chain. This also excludes, for
  now, magnetic cases with several QE species of the same element.

Notes
-----
To remove: the two paths reproduce how TribChem's interface workflow reads
its slabs from the database (``highput/firetasks/run.py:1028``, ``1036``)
and how ``FT_StoreCustomSlab`` stores a custom slab
(``highput/firetasks/custom_slab.py:118-270``). Golden tests compare
``custom_slab`` with that Firetask. The error for left-handed and tilted
cells is a deviation: TribChem transforms them.
"""

from dataclasses import dataclass, field

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab

from .model import MAGMOM, StructureError
from .readers import read_structure

# |cos| of the angles between c and a, b above which c is not perpendicular.
ORTHOGONALITY_TOL = 1e-6
# Largest deviation of R R^T from the identity, and of R from the identity
# for a slab that is not rotated.
ROTATION_TOL = 1e-9


@dataclass
class WorkflowSlab:
    """A slab read for a workflow: as read, as the workflow uses it, and how they differ.

    Attributes
    ----------
    original : pymatgen.core.structure.Structure
        The structure as read from the file, with all its data.
    slab : pymatgen.core.surface.Slab
        The slab given to the workflow.
    path : {'database', 'custom_slab'}
        How ``slab`` was obtained (see the module docstring).
    rotation : numpy.ndarray, shape (3, 3)
        Proper rotation with ``slab.lattice.matrix = original.lattice.matrix @ rotation``.
    rotated : bool
        Whether ``rotation`` differs from the identity.
    dropped : list of str
        Data of ``original`` missing from ``slab`` (site properties,
        ``properties.<key>``, ``reconstruction``).
    added : list of str
        Site properties of ``slab`` missing from ``original`` (e.g. ``magmom``).
    """

    original: Structure
    slab: Slab
    path: str                    # 'database' or 'custom_slab'
    rotation: np.ndarray         # slab.lattice.matrix = original.lattice.matrix @ rotation
    rotated: bool
    dropped: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)

    def provenance(self):
        """JSON data of the transformation, for the records.

        Returns
        -------
        dict
            Path, rotation, lattice matrices before and after, data dropped
            and added.
        """
        return {'path': self.path, 'rotated': self.rotated,
                'rotation': self.rotation.tolist(),
                'original_matrix': self.original.lattice.matrix.tolist(),
                'slab_matrix': self.slab.lattice.matrix.tolist(),
                'dropped': list(self.dropped), 'added': list(self.added)}


def read_workflow_slab(path, miller, format='auto'):
    """Read a slab file for a workflow.

    Parameters
    ----------
    path : str or pathlib.Path
        The slab file.
    miller : sequence of int
        Miller indices of the surface.
    format : str, optional
        As in ``read_structure``.

    Returns
    -------
    WorkflowSlab

    Raises
    ------
    StructureError
        As ``read_structure`` and ``workflow_slab``, with the file name.
    """
    try:
        return workflow_slab(read_structure(path, format), miller)
    except StructureError as error:
        if str(error).startswith(str(path)):
            raise
        raise StructureError(f'{path}: {error}') from None


def workflow_slab(structure, miller):
    """The ``Slab`` the workflow uses for a structure, with its checks.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        As read by ``read_structure``; a ``Slab`` follows the ``database``
        path, any other structure the ``custom_slab`` path.
    miller : sequence of int
        Miller indices of the surface; for a ``Slab`` they must equal its
        ``miller_index``.

    Returns
    -------
    WorkflowSlab

    Raises
    ------
    StructureError
        For QE labels other than the element, a left-handed cell, c not
        perpendicular to a and b, or Miller indices that differ from those of
        the slab.
    """
    miller = [int(m) for m in miller]
    custom = [i + 1 for i, site in enumerate(structure) if site.label != site.species_string]
    if custom:
        labels = sorted({structure[i - 1].label for i in custom})
        raise StructureError(f'QE labels other than the element ({", ".join(labels)}) are not '
                             'supported in the workflows: the workflow code does not carry them')
    check_slab_cell(structure.lattice.matrix)
    if isinstance(structure, Slab):
        if [int(m) for m in structure.miller_index] != miller:
            raise StructureError(f'miller = {miller} in the settings, but the slab has '
                                 f'miller_index = {list(structure.miller_index)}')
        path, slab = 'database', Slab.from_dict(structure.as_dict())
    else:
        # The moments of the file are passed apart from the rest of the
        # structure, which then carries no moments in oriented_unit_cell.
        poscar = structure.copy()
        if MAGMOM in poscar.site_properties:
            poscar.remove_site_property(MAGMOM)
        magmom = structure.site_properties.get(MAGMOM)
        path, slab = 'custom_slab', custom_slab(poscar, miller, magmom)
    rotation = np.linalg.solve(structure.lattice.matrix, slab.lattice.matrix)
    if (np.abs(rotation @ rotation.T - np.eye(3)).max() > ROTATION_TOL
            or np.linalg.det(rotation) <= 0):
        raise StructureError('the slab lattice is not a rotation of the cell of the file')
    original_props, slab_props = set(structure.site_properties), set(slab.site_properties)
    dropped = sorted(original_props - slab_props)
    dropped += [f'properties.{k}' for k in sorted(structure.properties)]
    if getattr(structure, 'reconstruction', None) is not None:
        dropped.append('reconstruction')
    return WorkflowSlab(structure, slab, path, rotation,
                        rotated=bool(np.abs(rotation - np.eye(3)).max() > ROTATION_TOL),
                        dropped=dropped, added=sorted(slab_props - original_props))


def check_slab_cell(matrix):
    """Check that a slab cell keeps its shape in the workflow.

    Parameters
    ----------
    matrix : array_like, shape (3, 3)
        Lattice vectors as rows.

    Raises
    ------
    StructureError
        Unless the cell is right-handed with c perpendicular to a and b
        (``ORTHOGONALITY_TOL``).
    """
    a, b, c = np.asarray(matrix, float)
    if np.linalg.det(matrix) <= 0:
        raise StructureError('left-handed cell (a x b points away from c): the slab of the '
                             'workflows (Lattice.from_parameters) would be its mirror image')
    for name, v in (('a', a), ('b', b)):
        cos = np.dot(v, c) / (np.linalg.norm(v) * np.linalg.norm(c))
        if abs(cos) > ORTHOGONALITY_TOL:
            angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            raise StructureError(f'c is not perpendicular to {name} (angle {angle:.6f} degrees): '
                                 'the slab of the workflows (c along z) would have a tilted surface')


def custom_slab(structure, miller, magmom=None, reconstructed=None):
    """A ``Slab`` built from any structure, as stored and read back by the workflow.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        The slab structure, without moments.
    miller : list of int
        Miller indices of the surface.
    magmom : sequence of float, optional
        Initial moment of every atom; zeros when not given.
    reconstructed : str, optional
        Name of the reconstruction, stored in the dict (``Slab.from_dict``
        does not read it back).

    Returns
    -------
    pymatgen.core.surface.Slab
        In the orientation of ``Lattice.from_parameters``, with ``magmom`` as
        its only site property. The cell is not checked: see
        ``workflow_slab``.

    Notes
    -----
    To remove: port of ``FT_StoreCustomSlab.from_input_files`` of TribChem
    (``highput/firetasks/custom_slab.py:118-130``, ``200-205``, ``229-267``)
    without the database: ``structure`` replaces the POSCAR and ``magmom`` the
    MAGMOM of the INCAR (``magmom_from = 'INCAR'``, the default of
    ``defaults.json:274``). The final ``Slab.from_dict`` is the reading of
    ``run.py:1028``.
    """
    slab = Slab(
        lattice=structure.lattice,
        species=structure.species,
        coords=structure.frac_coords,
        miller_index=miller,
        oriented_unit_cell=structure,
        shift=0,
        scale_factor=np.array((1, 1, 1)),
        site_properties=structure.site_properties
    )

    # Moments of the file, or zeros.
    if magmom is None:
        magmom = [0] * len(structure)
    slab.add_site_property('magmom', magmom)

    # The slab as stored: lattice parameters without matrix, magmom as the
    # only site property.
    opt = {
        "@class": "Slab",
        "@module": "pymatgen.core.surface",
        "charge": None,
        "energy": None,
        "lattice": {
            "a": slab.lattice.a,
            "alpha": slab.lattice.alpha,
            "b": slab.lattice.b,
            "beta": slab.lattice.beta,
            "c": slab.lattice.c,
            "gamma": slab.lattice.gamma,
            "volume": slab.lattice.volume,
        },
        "miller_index": miller,
        "oriented_unit_cell": slab.oriented_unit_cell.as_dict(),
        "reconstruction": reconstructed,
        "scale_factor": slab.scale_factor.tolist(),
        "shift": slab.shift,
        "sites": [
            {
                "species": [
                    {
                        "element": sp.symbol,
                        "occu": site.species[sp]  # Occupation of the site
                    } for sp in site.species
                ],
                "abc": list(site.frac_coords),
                "xyz": list(site.coords),
                "label": str(site.specie),
                "properties": {
                    "magmom": site.properties.get("magmom")
                }
            }
            for site in slab.sites
        ]
    }

    # Read back as the workflow reads a stored slab.
    return Slab.from_dict(opt)
