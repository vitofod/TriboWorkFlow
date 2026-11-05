# Interface workflow: step `match`

The interface workflow starts from two slabs, given in `[interface]` of the
settings ([configuration.md](configuration.md)). Its first step, `match`,
runs no calculation: it builds the interface cell of the two slabs and the
high-symmetry points of the two facing surfaces, which the next steps use. In
this version `match` is the only step available, so a run lists only `match`
in `[interface].steps`:

```bash
triboworkflow analyse match
```

In a pass of the orchestrator the step, having no calculation, is analysed as
soon as it starts.

## What the step does

1. **Inputs.** The two slab files are read as the workflow uses them
   ([structures.md](structures.md)); a slab rotated to the frame of the
   workflow, or with data dropped, is reported with a warning. The matching
   parameters are the keys of `[interface.inter_params]`, then the defaults
   (`max_area` 160.0, `interface_distance` 2.0, `max_angle_tol` 0.4, then the
   defaults of the matching code); with `set_bm = true` the strain weights are the bulk moduli of the
   two slabs. The ids of the interface are `mid1_mid2`, `formula1_formula2`
   and `name1_name2`, or, without formula and name, made from the slabs
   (`Cu_Al`, `Cu111-Al111`).
2. **Document.** The document of the system (`systems/<mid1_mid2>/system.json`,
   [store.md](store.md)) is created when missing, with `mid`, `formula`,
   `miller`, an empty `structure` and `name`. An existing document is kept: it
   must have the same ids, otherwise the step stops with an error naming the
   field that differs.
3. **Matching.** The slab whose formula and Miller indices come first in
   alphabetical order goes on top (Al(111) on Cu(111)); with the same formula
   and indices, slab 1 goes on top. The supercells of the two slabs are
   matched with the algorithm of Zur and McGill, and both are strained to the
   common cell, the average of the two cells weighted by the strain weights
   (with the bulk moduli, the stiffer slab is strained less). When no match is
   found, `max_area`,
   `max_length_tol` and `max_area_ratio_tol` are multiplied by 1.01, then by
   1.02, and so on (the products accumulate), until a match is found; when
   the factor reaches 1.5 the step stops with `Error in generating the
   interface. Matching not converged`. `max_angle_tol` is used unchanged at
   every attempt. 15 Å of vacuum are set around the interface.
4. **High-symmetry points.** The matching is done again from the parameters
   of step 1, without vacuum, and the high-symmetry points of the top surface
   of the bottom slab and of the bottom surface of the top slab are found;
   their combinations are the lateral shifts of the interface.
5. **Writing.** The entries below and the files are written together, at the
   end: when any part fails, nothing is written.

## Entries and files

| Entry of `system.json` | Content |
|---|---|
| `structure.init.interface` | the interface, with vacuum (pymatgen `Structure` dict) |
| `structure.init.bot_slab`, `structure.init.top_slab` | the two slabs strained to the common cell, placed as in the interface (pymatgen `Slab` dicts) |
| `inter_params` | the matching parameters of the attempt that found the match |
| `highsym` | `bot_unique`, `bot_all`, `top_unique`, `top_all`: the points of each surface, by label (`ontop_1`, `bridge_1`, `hollow_1`, …); `inter_unique`, `inter_all`: the lateral shifts of the interface, by the labels of the two points (e.g. `hollow_1-bridge_1`), the unique ones and all |

Each of the five entries records its provenance: the ids and Miller indices
of the slabs, the complete matching parameters, `set_bm`, the vacuum and the
content of the two slab files. Before writing, the entries already in the
document are emptied, so that nothing of an earlier interface (e.g. points
whose labels no longer exist) is left among the new data.

In `systems/<mid1_mid2>/match/`:

| File | Content |
|---|---|
| `interface.extxyz` | the interface, with vacuum |
| `interface.json` | the same, as a pymatgen `Structure` dict |

## Message

The step reports, in the terminal and in `cli.log`:

```text
mp-30_mp-134: interface mp-30_mp-134: Al(111) (slab 2, mp-134) on top of Cu(111) (slab 1, mp-30)
mp-30_mp-134: match found at attempt 1
mp-30_mp-134: parameters used: max_area = 160 angstrom^2, max_length_tol = 0.02, max_area_ratio_tol = 0.05, max_angle_tol = 0.4 (never enlarged), interface_distance = 2.0, strain weights = 140.0, 76.0 (slab 1, slab 2)
mp-30_mp-134: strain of the top slab: a +1.150 %, b +1.150 %, gamma -0.000 deg; of the bottom slab: a -0.614 %, b -0.614 %, gamma -0.000 deg
mp-30_mp-134: high-symmetry points: 6 lateral shifts (inter_unique), 6 in all (inter_all)
mp-30_mp-134: interface written in match/interface.extxyz and match/interface.json
```

The strain is that of the lattice vectors a and b of each slab, in percent,
and of the angle between them, in degrees; two slabs that are already
aligned have no strain. What the matching code prints goes to the log at
debug level only.

## When the step runs again

The step is done when the five entries exist and their provenance is the one
that the settings and the slab files give now. Changing a slab file, a key
of `[interface.inter_params]`, a bulk modulus with `set_bm = true`, or
`set_bm`, makes it run again at the next `analyse match` or pass of the
orchestrator; the entries of the later steps computed from the old interface
are then no longer current. Changing `formula`, `name` or `mid` gives
another interface: `mid` names another system; `formula` and `name` differ
from the existing document, which is an error (remove `system.json` of the
system, or restore the ids).

Notes
-----
To remove: port of TribChem's `FT_StartInterfaceMatch`, `FT_GenerateInterface`
and `FT_GetHighSymmetryPts` (`highput/firetasks/run.py`, `slabs.py`), chained
as in `InterfaceWF.match_interface`. The golden cases `match_step_*` compare
the database calls of the step with those of the Firetasks. Deviations:
DEV-01 (`max_angle_tol` of the user at every attempt, not 0.4 forced), DEV-33
(the recursion of `FT_GenerateInterface` works; its debug prints broke it,
A73), DEV-34 (`best_match` only `"area"`), DEV-35 (no `comp_params` in the
document), DEV-36 (entries emptied before being written again). Not ported:
`override`, `override_inter_db` (A45) and the bulk moduli read from the
database (`bm_custom`).
