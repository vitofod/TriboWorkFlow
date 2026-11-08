# Interface workflow: steps `match`, `pes`, `pes_scf`, `plotpes`, `plotpes_scf`, `adhesion` and `ppes`

The interface workflow starts from two slabs, given in `[interface]` of the
settings ([configuration.md](configuration.md)). Its first step, `match`,
runs no calculation: it builds the interface cell of the two slabs and the
high-symmetry points of the two facing surfaces, which the next steps use.
The steps `pes` and `pes_scf` compute the potential energy surface of the
interface from those points, relaxed along z and static; `plotpes` and
`plotpes_scf` interpolate the one and the other over the whole plane;
`adhesion` computes the adhesion energy of the interface at the site of
lowest and at the site of highest energy of one of the two surfaces; `ppes`
computes the energy of the interface that `adhesion` relaxed as its two slabs
move apart or closer along z. In this version these seven steps are
available, so a run lists only them in `[interface].steps`:

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
its interpolation, which is then done again. A PES is done only while `match`
is done and current too: the provenance holds the content that the step read,
so changing the settings of `match` and not analysing it again leaves the
PES not done, even if its own settings did not change.

The step `shear` is not available in this version: which of the two PES it
reads is decided with it.

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
replicated, whose memory grows with the square of their number (*Cost*). It
never runs in the orchestrator, which lives on the login node. It runs in
one of two ways, with the same code and the same results:

- **in a job of its own** on a compute node, with `[execution.analysis]` in
  the settings ([configuration.md](configuration.md)): once the step is
  ready the orchestrator writes its group of jobs, submits it with the
  jobscript of that table (memory, walltime, partition) and follows the job;
  the job interpolates and writes the entries, and the step is completed
  once they are done. A job that ends without them (out of memory, walltime)
  makes the step **failed**, with the state of the job and its outcome, and
  is never submitted again automatically: give the jobscript more memory or
  time, then `triboworkflow retry plotpes` ([execution.md](execution.md),
  *Analyses in a job of their own*);
- **by hand**, without that table: once ready, the step is left **manual**,
  with the command that analyses it, and the steps that depend on it wait.
  Run it where there is enough memory, with the orchestrator ended or
  stopped ([cli.md](cli.md), *The interpolation of the PES*):

```bash
triboworkflow analyse plotpes
triboworkflow analyse plotpes_scf
```

The command holds the lock of the run for the whole interpolation; the job
does not take it, and writes only the entries of its step and its outcome.
An interpolation stopped before its end writes nothing, and the next one
starts it from the beginning; an interpolation completed is recognised from
its entries, their provenance and its figure, without interpolating again.

The memory of the job is about the size of the matrix (*Cost*): 8·N² bytes
for N centres, N = 144 × the points left after the duplicates, with some
more for the rest (the peak measured on the 20 736 centres below is about
1.1 times the matrix). The linear algebra uses the threads of NumPy's BLAS:
set them in the jobscript (e.g. `export OMP_NUM_THREADS=8`) to the cores
asked for.

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
again. An interpolation is done only while its PES and `match` are done and
current too.

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

## Step `adhesion`

The adhesion energy of an interface is the energy that binds its two slabs:
the energy of the interface minus those of the two slabs alone. The step
computes it with four pw.x calculations, for the interface at the site of
lowest energy and at the site of highest energy of a PES, and takes the
difference of the two as the corrugation. It runs when listed in
`[interface].steps`, and has its own commands like the PES steps:

```bash
triboworkflow prepare adhesion
triboworkflow submit adhesion
triboworkflow analyse adhesion
```

### Which surface and which slabs

`[interface.adhesion]` ([configuration.md](configuration.md)) chooses, for the
whole run, the PES that gives the two sites and the kind of adhesion. The
step waits for that PES (and for `match`), not for the other one:

| `pes` | `type` | Interfaces of the two sites | Slabs |
|---|---|---|---|
| `pes` (default) | `regular` (default) | the final structures of the minimum and of the maximum of the relaxed PES (`structure.opt.pes.min`, `.max`) | the complete slabs of `match` |
| `pes_scf` | `regular` | the complete slabs of `match`, stacked at the sites of lowest and highest energy of the static PES (`pes_scf.energy.min_site`, `.max_site`, with the shifts of `highsym.inter_unique`) | the complete slabs of `match` |
| `pes_scf` | `short` | the input structures of the static PES at those two sites (`pes_scf.data.<site>.input.structure`) | the atoms below and above the plane z = 0 of the interface at the lowest site, taken as slabs again |

`short` needs `short = true` in `[interface.pes_scf]` (the reduced slabs of the
PES), and `pes = "pes"` with `short` is an error: the relaxed PES always
uses the complete slabs. The interface of the second row is stacked at a
distance of 2 Å between the slabs, whatever `interface_distance` of
`[interface.inter_params]` is. One variant per run: to compute another, make
another run, or change the settings and replace the calculations
(`prepare adhesion --replace`, [cli.md](cli.md)).

### Calculations

| Calculation | System | `calc_type` | Atoms that move |
|---|---|---|---|
| `adhesion/bot_slab` | the bottom slab alone | `slab_pos_relax` | all |
| `adhesion/top_slab` | the top slab alone | `slab_pos_relax` | all |
| `adhesion/interface_min` | the interface at the site of lowest energy | `slab_pos_relax` | all |
| `adhesion/interface_max` | the interface at the site of highest energy | `interface_z_relax` | only along z, the hydrogen atoms in every direction |

All four are `relax` calculations, with the k-point grid from its density as
for the PES (`[k_points] mode = "kdens"` is required) and the pw.x input
written as for every calculation of the workflow. The entries read must be
done and current, otherwise the step is blocked with what is missing: the
slabs of `match` (`structure.init.bot_slab`, `.top_slab`) and, by variant,
`structure.opt.pes.min` and `.max` of `pes`; or `pes_scf.energy.min_site` and
`.max_site` of `pes_scf` together with `highsym` of `match` (`regular`) or
with `pes_scf.data` (`short`).

### Analysis

`analyse adhesion`, or the orchestrator once every calculation is
completed, uses the results only when each calculation is complete with the
fingerprint that the settings give now, and the four share one energy model
(engine, pseudopotentials, cutoffs, k-points, smearing, spin, vdW): energies
of different models are never combined, and the step stops with the
calculations at fault and writes nothing. Then it writes, with `<type>` the
`type` of `[interface.adhesion]`:

| Entry of `system.json` | Content |
|---|---|
| `data.<label>` (`data.short.<label>` with `short`) | the results of each calculation: `nsites`, `density` (g/cm^3), `energy` and `energy_per_atom` (eV, the total energy of pw.x), `forces`, `stress`, `dir_name`, `task_id`, `task_label` (the calculation, `adhesion/bot_slab`); the keys that pw.x does not give (`bandgap`, `cbm`, `vbm`, `is_gap_direct`, `pseudo_potential`, and `forces` or `stress` when not printed) are null |
| `structure.opt.<type>.<label>` | the final structure of each calculation; the two slabs are slabs again (pymatgen `Slab` dicts, with the Miller indices and the other keys of the slabs they started from), the two interfaces stay structures |
| `adhesion.<type>` | `min_site` and `max_site`, each with `adhesion_energy_ev` and `adhesion_energy_jm2`; `corrugation_ev` and `corrugation_jm2` |
| `structure.init.short.bot_slab`, `.top_slab`, `.interface` | only with `short`: the reduced slabs and the interface at the lowest site, written with the rest |

The adhesion energy at a site is the energy of the interface at that site
minus those of the top slab and of the bottom slab: **negative for an
interface that binds**. The corrugation is the energy of the interface at
the highest site minus that at the lowest one. The energies are in eV; the
J/m^2 divide them by the area of the cell of the final structure of
`adhesion/interface_min`. The empty dicts of the structures (`properties:
{}`) are not stored; the structures are read back the same.

Everything is written together, at the end: when any part fails, nothing is
written. The writes replace the same keys at every analysis, so nothing of an
earlier one is left, and the entries of the other kind (`regular` or `short`)
are not touched. Each entry of the table (`data.<label>` and the other
entries by label) records its provenance: the parameters of the step (the
surface, the kind, the `calc_type` of each calculation, the k-point density),
the calculations with their fingerprints and the results used, and the
content of the entries read.

### Message

```text
<system>: adhesion: <type> adhesion <on the complete slabs|on the reduced slabs>, from the PES <pes> (<relaxed along z|static>), at <site> (lowest energy) and <site> (highest energy)
<system>: adhesion energy at <site>: <energy> eV, <energy> J/m^2
<system>: adhesion energy at <site>: <energy> eV, <energy> J/m^2
<system>: corrugation (maximum - minimum): <difference> eV, <difference> J/m^2
```

The energies have 6 decimals. What the code that builds the structures
prints goes to the log at debug level only.

### When the step runs again

The step is done when its entries exist and their provenance is the one that
the settings, the calculations and the entries read give now. Changing
`[interface.adhesion]`, `kdens`, a parameter of the calculations (e.g.
`ecutwfc`), or analysing the PES or `match` again (new calculations, other
settings) makes it run again: the structures of the calculations differ, so
`prepare adhesion` stops with the list and `prepare adhesion --replace`
prepares them again. New results of a calculation, run again with the same
input, make the step run again at the next `analyse` or pass. The step is
done only while the PES it reads and `match` are done and current too: after
a change to the PES that is not analysed again, `adhesion` is not done and its
analysis stops, naming the steps to run first.

Notes
-----
To remove: port of TribChem's `FT_StartAdhesion` (with
`FireTaskInterface.query_interface`), `InterfaceWF.adhesion_energy`, the
`FT_MoveTagResults` of its four calculations and `FT_AdhesionEnergy`
(`calculate_adhesion`, `_change_slabs`). TribChem read the surface and the
kind of adhesion as two options, `pes_type` (`pes` by default) and
`adhesion_type` (`regular`), with `regular` on `pes` and `short` on `pes_scf`
reachable from its command line, `regular` on `pes_scf` only in the code, and
`short` on `pes` refused (D56). The golden cases `adhesion_step_*` compare the
structures, `calc_type` and k-points of the four calculations and the database
writes with those of the Firetasks, for the three variants. Differences:
DEV-35 (the k-point density from the settings, not from `comp_params` of the
document), DEV-39 (the writes use the ids of the document of `match`); the
adhesion was never found done there, since `check_entry` was not passed
(A57), and here it is done when its entries and their provenance are current;
`structure.init.short` is written with the results, not before the
calculations. The distance of the interface of `pes_scf` with `regular` is
that of the function that stacks the slabs, as there (A86). Not ported:
`override`, `run_vasp`, `get_main_folder` (A47), `FT_SaveVASPOutputs` (the
files of each calculation stay in its directory) and the other `calc_type`
that `adhesion_energy` accepts at one's own risk.

## Step `ppes`

The perpendicular PES (PPES) is the energy of an interface as its two slabs
move apart or closer along z, rigidly. The step starts from the interface that
`adhesion` relaxed, moves the slab on top by each of the shifts of
`[interface.ppes]` and runs one static pw.x calculation for each, with no
atom free to move. It runs when listed in `[interface].steps`, and has its own
commands like the other steps with calculations:

```bash
triboworkflow prepare ppes
triboworkflow submit ppes
triboworkflow analyse ppes
```

### Which interface

The step waits for `adhesion` and, behind it, for its PES and `match`. One
run covers one kind and one site:

| Setting | Chooses | Entry read |
|---|---|---|
| `type` of `[interface.adhesion]` (`regular` or `short`) | the kind of interface, the one that `adhesion` computed | `structure.opt.<type>.interface_min` or `.interface_max` |
| `site` of `[interface.ppes]` (`min_site` or `max_site`) | the interface at the site of lowest energy or of highest energy | `interface_min` for `min_site`, `interface_max` for `max_site` |

The entry must be written by `adhesion` and current, otherwise the step is
blocked with what is missing. To compute the other site, change `site` and run
again: the calculations are other ones, and the results of the two sites stay
side by side in the document. For the other kind make another run, as for the
adhesion.

### Calculations

One calculation per shift, `ppes/<site>_<shift>`, with the shift in angstrom as
it is written in `shifts`: `ppes/min_site_-0.75`, `ppes/min_site_0.25`. All are
`scf` calculations (`calc_type` `slab_from_scratch`) of the relaxed interface
with the slab on top moved by the shift along z, a negative shift bringing the
slabs closer: shift 0 is the interface as `adhesion` left it. No atom is
fixed, whatever the relaxation of `adhesion` fixed. The k-point grid comes from
its density (`[k_points] mode = "kdens"` is required) and the pw.x input is
written as for every calculation of the workflow. With the 15 default shifts
([configuration.md](configuration.md)) a site needs 15 calculations.

The atoms to move are those of the slab on top, told by the slab each atom
belongs to and not by its position in the cell: the site property
`interface_region` (`bot_slab` or `top_slab`) that `match` gives to every atom of
an interface and that the pw.x adapter returns with the final structure. Each
shift moves the whole slab on top by the same distance along z, whatever the
thickness of the two slabs, the origin of the cell and whether the coordinates
are wrapped; the cell and the atoms of the bottom slab stay where they are. The
slabs are not told by their elements, which may be the same in both.

The slab of each atom is checked before any calculation is prepared. The
interface of the document must carry it, and must come from the input of the
calculation of the adhesion that gave it. The document does not keep that input
but the record of the calculation does (`adhesion/interface_min` or
`adhesion/interface_max`): the interface must have the same number of atoms,
the same species in the same order and the same slab for each atom (the
positions are not compared, since the calculation moved the atoms). Otherwise
nothing is prepared and the step says why; in a run it is `failed`, with the
message among its errors, and no job is submitted:

| Case | Message of the error |
|---|---|
| the interface carries no `interface_region` | `structure.opt.<type>.interface_min: the atoms carry no 'interface_region' (the slab each belongs to); run adhesion again` |
| a value other than `bot_slab` or `top_slab`, or no atom in one of the two slabs | `…: atom 2 is of the slab 'middle_slab', not 'bot_slab' or 'top_slab'`, `…: no atom belongs to the slab 'top_slab'` |
| the record of the adhesion is missing, or has no input structure, or the input carries no `interface_region` | `<system>: the record adhesion/interface_min is missing: …; run adhesion again` |
| the number of atoms, their species in order or the slab of an atom differ between the interface and the input | `…: the interface has 62 atoms and the input of adhesion/interface_min has 63; …`, `…: atom 3 is of the slab 'top_slab' in the interface and of the slab 'bot_slab' in …` |

Results computed with another selection of the atoms are not reused: the
structures of the calculations change where the selection does, and so do their
fingerprints, so a step whose results were computed before is no longer done and
`prepare ppes --replace` computes it again. Where the selection by position
picked the slab on top (the slabs thin compared with the cell, the contact plane
at the origin) the structures, and the results, are the same.

### Analysis

`analyse ppes`, or the orchestrator once every calculation is completed, uses
the results only when each calculation is complete with the fingerprint that
the settings give now, and all share one energy model (engine,
pseudopotentials, cutoffs, k-points, smearing, spin, vdW): energies of
different models are never combined, and the step stops with the calculations
at fault and writes nothing. The energies are not combined with those of the
adhesion either. Then it writes, with `<type>` the `type` of
`[interface.adhesion]` and `<site>` the `site` of `[interface.ppes]`:

| Entry of `system.json` | Content |
|---|---|
| `ppes.<type>.data.<site>.shift_<shift>` | the results of each calculation, in `output` (`nsites`, `density` in g/cm^3, `energy` and `energy_per_atom` in eV, `forces`, `stress`, `dir_name`, `task_id`, `task_label`, and the final `structure`; the keys that pw.x does not give are null) and the input `structure`; the key has a comma for the point of the shift (`shift_-0,75`) |
| `ppes.<type>.energy.<site>.ppes_ev` | `[shift, energy]` for each shift, the total energy of the cell in eV |
| `ppes.<type>.energy.<site>.ppes_jm2` | the same, per area of the cell of the final structure, in J/m^2 |
| `ppes.<type>.energy.<site>.uber` | only with `fit = "uber"`: `g`, `l`, `sigma_g`, `sigma_l` |

The energy is the total energy of the cell, **not referred to the lowest
one**. Everything is written together, at the end: when any part fails,
nothing is written. Before writing, the two entries of the type and site run
(`data.<site>` and `energy.<site>`) are emptied, so that the shifts or the fit
of an earlier analysis do not stay under the provenance of this one; the other
site, the other type and the entries of the other steps are not touched. Each of
the two entries records its provenance: the parameters of the step (type, site,
shifts, fit, the `calc_type`, the k-point density), the calculations with
their fingerprints and the results used, and the content of the interface read.

The fit is the universal binding energy relation (UBER). It is made on the
shifts and the energies in eV, and only on the shifts that are zero or
positive, so:

- `g` is the depth of the curve in **eV per cell**, not in J/m^2, `l` is its
  length in **angstrom**, and `sigma_g` and `sigma_l` are their standard
  deviations, in the same units;
- with three shifts that are zero or positive the fit runs but the covariance
  cannot be estimated: the deviations are infinite and the step says so with a
  warning (they are written as such);
- with fewer than three it fails, and so does the analysis: it names the fit
  and the number of shifts it had, writes nothing and keeps the cause of the
  error; list more shifts or use `fit = "none"`;
- with four or more the deviations are finite.

### Message

```text
<system>: ppes: <type> PPES at <site>, <n> shifts of the slab on top from <min> to <max> angstrom (slab_from_scratch)
<system>: total energy per area: lowest <energy> J/m^2 at the shift <shift>, highest <energy> J/m^2 at the shift <shift>
<system>: fit uber: g = <g> +- <sigma_g> eV per cell, l = <l> +- <sigma_l> angstrom
```

The last line is `no fit (fit = "none")` without the fit. The energies have
6 decimals. What the code that builds the structures prints goes to the log at
debug level only.

### When the step runs again

The step is done when its two entries exist and their provenance is the one
that the settings, the calculations and the interface read give now. It is not
done, and its analysis stops with the steps to run first, when `adhesion`, the
PES it reads or `match` is not done and current: the provenance holds the
content that each step read, not whether that content is still current.

- Changing only `fit` computes nothing again: the fingerprints of the
  calculations do not depend on it, so no calculation is prepared or
  submitted, and the step only analyses the same results anew.
- Changing `shifts` or `site` makes the step not done and asks for the
  calculations of the shifts or of the site that are new; those that stay keep
  their results.
- Changing `kdens` or a parameter of the calculations (e.g. `ecutwfc`), or
  analysing `adhesion` again (a new interface), changes the calculations
  already prepared: `prepare ppes` stops with the list and `prepare ppes
  --replace` prepares them again.
- New results of a calculation, run again with the same input, make the step
  run again at the next `analyse` or pass.

Notes
-----
To remove: port of TribChem's `FT_StartPPES` (with
`FireTaskInterface.query_interface`), `InterfaceWF.calculate_ppes`,
`generate_ppes_inputs`, the `FT_MoveTagResults` of its calculations and
`FT_FitPPES`. The golden cases `ppes_step_*` compare the structures, `calc_type`
and k-points of the calculations and the database writes, the energies and the
fit included, with those of the Firetasks, at both sites and with both kinds;
the fit of the step on the files of `data/homogeneous/ppes_data` is compared
with the golden values of the fit function (`uber_ppes_data`). Differences:
DEV-35 (the k-point density from the settings, not from `comp_params` of the
document), DEV-39 (the writes use the ids of the document of `match`), DEV-43
(the two entries of the type and site run are emptied before the writes;
TribChem merged the new keys into the old ones). TribChem's command line never
passed the fit, so no fit was run (A88): `fit = "none"` is the default here
too. The fit function works only with `"uber"` (A06, A89), and the other names
are refused when the settings are read. The shifts of its command line are the
default ones, with the jump over 2.5 (A90). The type of the PPES was an option
independent of the adhesion in TribChem; here it is the one of
`[interface.adhesion]`, and the PPES of a site was skipped there once the other
site was done (A59), where here "done" is the entries of the site with their
provenance (D57). TribChem chose the atoms to move by their fractional z, once
wrapped into the cell, below one half: the slab on top only while each slab lies
in one half of the cell, otherwise a slab thicker than half the cell was
deformed (a Cu(111)/Al(111) interface with slabs of different thickness moved 82
of the 105 atoms of the slab on top) and, with the origin moved by half a cell,
the bottom slab was moved (A91). Here the atoms are chosen by the slab each
belongs to and checked against the input of the adhesion (DEV-44); the rest of
the operations on the structures, the translation of the atoms and the input set
among them, are the original ones. Not ported: `override`, `run_vasp`, `get_main_folder`
(A47), `FT_SaveVASPOutputs`, the other values of `ppes_type` and the figure and
the csv of the PPES.
