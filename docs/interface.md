# Interface workflow: steps `match`, `pes`, `pes_scf`, `plotpes` and `plotpes_scf`

The interface workflow starts from two slabs, given in `[interface]` of the
settings ([configuration.md](configuration.md)). Its first step, `match`,
runs no calculation: it builds the interface cell of the two slabs and the
high-symmetry points of the two facing surfaces, which the next steps use.
The steps `pes` and `pes_scf` compute the potential energy surface of the
interface from those points, relaxed along z and static; `plotpes` and
`plotpes_scf` interpolate the one and the other over the whole plane. In
this version these five steps are available, so a run lists only them in
`[interface].steps`:

```bash
triboworkflow analyse match
triboworkflow prepare pes_scf
```

In a pass of the orchestrator the step `match`, having no calculation, is
analysed as soon as it starts. `plotpes` and `plotpes_scf` have no
calculation either, but are analysed only by hand (*Where they run*).

## Step `match`

### What the step does

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

### Entries and files

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

### Message

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

### When the step runs again

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

## Steps `pes` and `pes_scf`

The potential energy surface (PES) of the interface is its energy as a
function of the lateral position of the top slab on the bottom one. It is
sampled at the lateral shifts written by `match` (`highsym.inter_unique`):
one pw.x calculation per point, with the top slab shifted laterally by the
point and placed at `interface_distance` (of `[interface.inter_params]`,
default 2.0 Å) from the bottom one. Two steps compute it; each runs when
listed in `[interface].steps`, also both, and neither is the default:

| Step | Calculation of each point | Calculations | Entry |
|---|---|---|---|
| `pes` | `relax` in which the atoms move only along z, the hydrogen atoms in every direction (`calc_type` `interface_z_relax`) | `pes/<point>` | `pes` |
| `pes_scf` | `scf` (`calc_type` `slab_from_scratch`), on the complete slabs or on slabs reduced to their layers next to the interface | `pes_scf/<point>` | `pes_scf` |

Both need only `match`: one can be computed without the other, and either
first. Each has its own commands:

```bash
triboworkflow prepare pes_scf
triboworkflow submit pes_scf
triboworkflow analyse pes_scf
```

### Calculations

1. **Inputs.** The entries of `match` that the steps read
   (`structure.init.interface`, `.bot_slab`, `.top_slab`, `highsym`) must
   be done and current: otherwise the step is blocked with what is missing.
   The k-point grid comes from its density: `[k_points] mode = "kdens"` is
   required, since the cell of the interface is known only after the
   matching; another mode stops the step with an error on `k_points.mode`.
2. **Reduced slabs** (`pes_scf` only, `short = true` in
   `[interface.pes_scf]`). The bottom and the top slab keep only their
   `layers` next to the interface; without `layers`, 3 for a (111) slab and 2
   for the others, by the Miller indices of the slab below and of the slab
   above. In the example, Cu(111) below Al(111) gives 3 and 3 layers;
   `layers = [2, 1]` keeps 2 layers of Cu and 1 of Al. The relaxed PES always
   uses the complete slabs.
3. **Points.** For each point the top slab is moved by the first lateral
   shift of the point and stacked on the bottom one; the constraints and the
   k-point grid are those of the calculation type, and the pw.x input is
   written as for every calculation of the workflow
   ([configuration.md](configuration.md)). The two steps give the same
   structures and grids on the same slabs; they differ in the calculation and
   the constraints.

### Analysis

`analyse <step>`, or the orchestrator once every calculation is completed,
uses the results only when each calculation is complete with the
fingerprint that the settings give now, and all share one energy model
(engine, pseudopotentials, cutoffs, k-points, smearing, spin, vdW); otherwise
it stops with the calculations at fault and writes nothing. Then it writes:

| Entry of `system.json` | Content |
|---|---|
| `<step>.data.<point>` | the results of the calculation of the point: `output.nsites`, `output.density` (g/cm^3), `output.energy` and `output.energy_per_atom` (eV, the total energy of pw.x), `output.forces`, `output.stress`, `output.structure` (the final structure) and `input.structure`; `output.task_label`, `output.task_id` and `output.dir_name` hold the calculation (`pes/hollow_1-bridge_1`). The keys that pw.x does not give (`bandgap`, `cbm`, `vbm`, `is_gap_direct`, `pseudo_potential`, and `forces` or `stress` when not printed) are null |
| `<step>.energy.grid.hs_unique_ev` | `[point, x, y, energy]` of every point, sorted by energy (eV); `x, y` (Å) is the first lateral shift of the point; equal energies keep the order of the points |
| `<step>.energy.grid.hs_unique_jm2` | the same, with the energies in J/m^2 on the area of the cell of the minimum |
| `<step>.energy.min_site`, `<step>.energy.max_site` | the points of lowest and highest energy |
| `structure.opt.<step>.min`, `structure.opt.<step>.max` | the final structures of those two points |

The energies are total energies: only their differences have a meaning, in
eV and in J/m^2 alike. The empty dicts of the structures (`properties: {}`)
are not stored; the structures are read back the same.

Before writing, the three entries of the step (`<step>`,
`structure.opt.<step>.min` and `.max`) are emptied, so that nothing of an
earlier analysis (e.g. the points of an earlier interface, or their
interpolation) is left among the new data; the other PES and the entries of
`match` are not touched. Everything is written together, at the end: when any
part fails, nothing is written, the emptying included. Each of the seven
entries of the table (`<step>.data` as one) records its provenance: the parameters of the
step (calculation type, k-point density, `interface_distance`, layers), the
calculations with their fingerprints and the results used, and the content
of the entries of `match` read. The interpolation, written by `plotpes` or
`plotpes_scf` next to them in `<step>.energy`, leaves them current.

### Message

```text
<system>: <step>: <n> high-symmetry points, <kind> (<calc_type>)
<system>: minimum at <point>: <energy> eV; maximum at <point>: <energy> eV
<system>: corrugation (maximum - minimum): <difference> eV, <difference> J/m^2
```

`<kind>` is `relaxed along z` for `pes`; `static, on the complete slabs` or
`static, on the slabs reduced to <bottom> (bottom) and <top> (top) layers`
for `pes_scf`. The energies have 6 decimals. What the code that builds the
points prints goes to the log at debug level only.

### When the steps run again

A PES step is done when its seven entries exist and their provenance is the
one that the settings, the calculations and the entries of `match` give now.
Changing `kdens`, `interface_distance`, a parameter of the calculations
(e.g. `ecutwfc`) or the interface (a new `match`) makes both run again;
`short` or `layers` only `pes_scf`; new results of a calculation, run again
with the same input, only the PES that uses it. A PES analysed again removes
its interpolation, which is then done again.

The later steps (`shear`, `adhesion`) are not available in this version:
which of the two PES each one reads is decided with them.

Notes
-----
To remove: port of TribChem's `FT_StartPES` (with
`FireTaskInterface.query_interface`), `InterfaceWF.calculate_pes`, the
`FT_MoveTagResults` of its points and `FT_PotentialEnergySurface`
(`arrange_data`, `_update_opt_structure`); the interpolation, commented out
there, is not part of these steps. The two steps are the two values of `-pt`
of TribChem's command line, which ran one PES per invocation, `pes` by default
(DEV-37). The golden cases `pes_step_*` compare the calculations and the
database writes of the steps with those of the Firetasks. Deviations: DEV-35
(the k-point density from the settings, not from `comp_params` of the
document), DEV-38 (the layers of the reduced slabs are those of the bottom and
of the top slab, and the default rule takes the Miller indices of the slab
below and above, where TribChem's command line took them in the order given,
A81), DEV-39 (the writes use the ids of the document of `match`, A80), DEV-40
(the three entries emptied before being written again). TribChem's command
line always passed the layers, so its static PES was always on reduced slabs;
here the complete slabs are the default (D07). Not ported: `override`,
`run_vasp`, `to_file`, `get_main_folder` (A47) and `FT_SaveVASPOutputs` (the
files of each calculation stay in its directory).

## Steps `plotpes` and `plotpes_scf`

Each step interpolates one PES over the whole plane of the interface, from
the energies of its high-symmetry points; neither runs a calculation:

| Step | PES interpolated | Entries read | Figure |
|---|---|---|---|
| `plotpes` | `pes`, relaxed along z | `highsym` (of `match`), `pes.data` | `plotpes/pes_<name>.png` |
| `plotpes_scf` | `pes_scf`, static | `highsym` (of `match`), `pes_scf.data` | `plotpes_scf/pes_<name>.png` |

Each runs when listed in `[interface].steps`, also both, and neither is the
default. `<name>` is the name of the interface in the document
(`Cu111-Al111`).

### What the steps do

1. **Inputs.** The entries read must be done and current: otherwise the step
   is blocked with what is missing. The points are `highsym.inter_unique`,
   each with its first lateral shift `x, y`, and its energy
   `<pes>.data.<point>.output.energy` (eV); they are sorted by energy
   (`[point, x, y, energy]`, equal energies in the order of the points), and
   converted to J/m^2 on the area of the cell of the final structure of the
   point of lowest energy, whose cell is the one of the interpolation.
2. **Interpolation.** Every replica of each point in the cell
   (`highsym.inter_all`) takes the energy of its point; the duplicates are
   removed, the points brought into the cell and the minimum set to zero.
   The points are replicated 6 times on each side of the cell (12 x 12
   copies) and interpolated with radial basis functions (thin-plate spline,
   smoothing 1e-3), then evaluated on a grid of 50 points per Å^2. This is
   done twice, as in TribChem: with the energies in eV, then in J/m^2, the
   second time drawing the figure (2 x 2 cells, 300 x 300 points, 300 dpi).
3. **Writing.** The entries and the figure are written together, at the end:
   when any part fails, or the entries read change during the
   interpolation, nothing is written and the figure is not left.

### Entries and files

| Entry of `system.json` | Content |
|---|---|
| `<pes>.energy.grid.hs_all_ev` | `[point, x, y, energy]` of every replica of each point in the cell (eV) |
| `<pes>.energy.grid.hs_all_jm2` | the same, in J/m^2 |
| `<pes>.energy.pes_ev` | `[x, y, energy]` of the points interpolated: in the cell, without duplicates, the minimum at zero (eV) |
| `<pes>.energy.pes_jm2` | the same, in J/m^2 |

Each of the four entries records its provenance: the calculation type and
`pes_type` of the PES, and the content of the entries read (replicas,
sorted energies, cell). They are written next to the entries of the PES
step, which stay current. The figure of the PES in J/m^2 is a file of the
folder of the system, not an entry: TribChem also stored it in the document
(`image`), the same image encoded again; the two steps write in two folders,
so their figures never overwrite each other.

### Where they run

The interpolation solves a dense linear system on all the points
replicated, whose memory grows with the square of their number (*Cost*). In
this version it never runs in the orchestrator, which lives on the login
node, nor in any other automation: once ready, the step is left **manual**,
with the command that analyses it, and the steps that depend on it wait.
Run it by hand where there is enough memory, with the orchestrator ended or
stopped ([cli.md](cli.md), *The interpolation of the PES*):

```bash
triboworkflow analyse plotpes
triboworkflow analyse plotpes_scf
```

The command holds the lock of the run for the whole interpolation. An
interpolation stopped before its end writes nothing, and the next `analyse`
starts it from the beginning; an interpolation completed is recognised from
its entries, their provenance and its figure, without interpolating again.
A job of its own on a compute node, followed by the orchestrator, is for a
later version; the interpolation stays the same.

### Cost

Measured with `tests/plotpes_cost.py` on the interfaces of the tests, with
synthetic energies, on a laptop with the linear algebra on one thread. The
centres are the points replicated 12 x 12; the matrix is their dense matrix
(8 bytes per element), close to the peak memory measured:

| Interface | Points (unique, in the cell) | Centres | Matrix | Interpolation in eV | In J/m^2 with the figure | Construction | Figure |
|---|---|---|---|---|---|---|---|
| Cu(111)/Cu(111) | 4, 8 | 1 152 | 0.01 GiB | 0.22 s | 1.35 s | 0.22 s | 1.08 s |
| Pt(111)/Ag(111) | 6, 12 | 1 728 | 0.02 GiB | 0.40 s | 1.78 s | 0.38 s | 1.40 s |
| Cu(111)/Al(111), `max_area` 20 | 6, 144 | 20 736 | 3.2 GiB (peak 3.5 GiB) | 122 s | 132 s | 115 s | 14 s |
| Cu(111)/Al(111) of the example | 6, 756 | 108 864 | 88.3 GiB | not measured | | | |

On the small interfaces the figure takes most of the time: its 90 000
points are evaluated on all the centres. On the large ones the construction
does (the dense linear system), and it grows faster than the matrix. The
second call repeats the whole interpolation of the first, then draws the
figure.

### Message

```text
<step>: interpolation in eV: <seconds> s
<step>: interpolation in J/m^2 and figure: <seconds> s
<system>: <step>: PES <pes> (<calc_type>) interpolated: <n> high-symmetry points, <m> in the cell, <k> after removing the duplicates
<system>: interpolation in eV <seconds> s, in J/m^2 with the figure <seconds> s
<system>: figure written in <step>/pes_<name>.png
```

What the interpolation code prints goes to the log at debug level only.

### When the steps run again

An interpolation step is done when its four entries exist, their provenance
is the one the entries read give now, and its figure exists. Analysing its
PES again (new calculations, other settings, a new `match`) removes the
interpolation, which is then done again; a figure removed makes the step run
again.

Notes
-----
To remove: port of TribChem's `FT_StartInterpolation` (with
`FireTaskInterface.query_interface`), `InterfaceWF.pes_interpolation` and
`FT_PESInterpolation` (`arrange_data`, `compute_pes`). The two steps are the
two values of `-pt` of TribChem's command line, which interpolated one PES
per invocation, `pes` by default (DEV-41). The golden cases `plotpes_step_*`
compare the entries read and written and the pixels of the figure with
those of the Firetasks, for both PES. Deviations: DEV-42 (the figure stays a
file, `image` is not written), DEV-39 (the writes use the ids of the
document of `match`); the analysis only by hand until the job of its own
(D54). The figure is closed once saved (A84); the interpolation is computed
twice, as there (A07). Not ported: `override`, `check_entry` (the provenance
decides), `to_file` (A83), `db_file`, `database`.
