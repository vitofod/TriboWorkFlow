"""Slabs for the workflows: the pymatgen ``Slab`` that TribChem would use.

The interface workflow of TribChem reads each slab from the database with
``Slab.from_dict`` (``highput/firetasks/run.py:1028``, ``1036``) and its code
needs a ``Slab`` (``check_slab_type``, ``flip_slab``: ``miller_index``,
``oriented_unit_cell``, ``shift``, ``scale_factor``, ``reconstruction``,
``site_properties``). Two historic paths are reproduced:

- ``database``: a pymatgen JSON slab, e.g. the ``structure.opt`` of a TribChem
  document, read back with ``Slab.from_dict`` as in ``run.py:1028``;
- ``custom_slab``: any other structure file, stored as ``FT_StoreCustomSlab``
  stores a POSCAR (``highput/firetasks/custom_slab.py:118-270``) and read
  back with ``Slab.from_dict``.

Both rebuild the lattice with ``Lattice.from_parameters``, already in the
``Slab`` constructor (``reorient_lattice=True``, pymatgen
``core/surface.py:86``, ``147-158``): c along z, a along x. The custom path
keeps only ``magmom`` among the site properties (``custom_slab.py:261-263``),
and both lose the labels. ``workflow_slab`` makes these
transformations explicit and keeps the structure as read:

- the cell must be right-handed with c perpendicular to a and b; any other
  cell would be mirrored or tilted by ``from_parameters``, and is an error
  here, where TribChem transforms it;
- the rotation from the file frame to the slab frame is returned, with the
  data dropped and added;
- custom QE labels (e.g. ``Fe1``, ``Fe2``) are an error: TribChem does not
  carry them, so neither do the workflows for now. This also excludes
  magnetic cases with several QE species of the same element.
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
    """A slab read for a workflow: as read, as TribChem uses it, and how they differ."""

    original: Structure
    slab: Slab
    path: str                    # 'database' or 'custom_slab'
    rotation: np.ndarray         # slab.lattice.matrix = original.lattice.matrix @ rotation
    rotated: bool
    dropped: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)

    def provenance(self):
        """JSON data of the transformation, for the records."""
        return {'path': self.path, 'rotated': self.rotated,
                'rotation': self.rotation.tolist(),
                'original_matrix': self.original.lattice.matrix.tolist(),
                'slab_matrix': self.slab.lattice.matrix.tolist(),
                'dropped': list(self.dropped), 'added': list(self.added)}


def read_workflow_slab(path, miller, format='auto'):
    """``workflow_slab`` of the structure in ``path``."""
    try:
        return workflow_slab(read_structure(path, format), miller)
    except StructureError as error:
        if str(error).startswith(str(path)):
            raise
        raise StructureError(f'{path}: {error}') from None


def workflow_slab(structure, miller):
    """The ``Slab`` TribChem would use for ``structure``, with the checks above."""
    miller = [int(m) for m in miller]
    custom = [i + 1 for i, site in enumerate(structure) if site.label != site.species_string]
    if custom:
        labels = sorted({structure[i - 1].label for i in custom})
        raise StructureError(f'QE labels other than the element ({", ".join(labels)}) are not '
                             'supported in the workflows: TribChem does not carry them')
    check_slab_cell(structure.lattice.matrix)
    if isinstance(structure, Slab):
        if [int(m) for m in structure.miller_index] != miller:
            raise StructureError(f'miller = {miller} in the settings, but the slab has '
                                 f'miller_index = {list(structure.miller_index)}')
        path, slab = 'database', Slab.from_dict(structure.as_dict())
    else:
        # The moments of the file stand for the MAGMOM of the INCAR, the rest
        # of the structure for the POSCAR, which carries no moments.
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
    """StructureError unless the cell is right-handed with c perpendicular to a and b."""
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
    """The slab of ``FT_StoreCustomSlab`` as ``FT_StartInterfaceMatch`` reads it.

    Port of ``FT_StoreCustomSlab.from_input_files`` (``custom_slab.py:118-130``,
    ``200-205``, ``229-267``) without the database: ``structure`` replaces the
    POSCAR and ``magmom`` the MAGMOM of the INCAR (``magmom_from = 'INCAR'``,
    the default of ``defaults.json:274``); without it the moments are zeros, as
    with an INCAR without MAGMOM. Then ``Slab.from_dict`` as in ``run.py:1028``.
    No checks: see ``workflow_slab``.
    """
    # custom_slab.py:121-130
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

    # custom_slab.py:200-205, MAGMOM of the INCAR or zeros
    if magmom is None:
        magmom = [0] * len(structure)
    slab.add_site_property('magmom', magmom)

    # custom_slab.py:229-267, the "structure.opt" entry of the database
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

    # run.py:1028
    return Slab.from_dict(opt)
