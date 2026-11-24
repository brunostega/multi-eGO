# cmap_builder

Builds the backbone CMAPs (`multi-ego-basic.ff/cmap/<RES>.cmap`) that make
blocked dipeptides (Ace-X-NMe) reproduce target Ramachandran distributions,
by default those of Choi & Pappu, JCTC 15, 1355 (2019), with maps that run
at the multi-eGO 5 fs time step.

## Workflow

1. Generate each dipeptide with `--backbone cmap` (no phi/psi dihedrals) and
   the same interaction matrix as the final model.
2. Run it without any CMAP; extract `gmx rama -f traj.xtc -s topol.tpr -o RES.xvg`.
3. Build the maps:

   ```bash
   python tools/cmap_builder/build_cmap.py --sim runs/ --target choi_pappu/ \
       --temperature 300 --out multi-ego-basic.ff/cmap
   ```

4. Read `multi-ego-basic.ff/cmap/README.md`. Residues marked **iterate**
   should be rerun *with* the maps (OpenMM backend, `cmap_dir=...`) and
   rebuilt from those runs:

   ```bash
   python tools/cmap_builder/build_cmap.py --sim runs_iter2/ --target choi_pappu/ \
       --temperature 300 --sampled-maps multi-ego-basic.ff/cmap --out cmap_iter2
   ```

Any change to the matrix, the bonded terms or the temperature requires new
runs and new maps.

## Method

With a map U active, the dipeptide samples exactly
`P_sim(phi, psi) exp(-U(phi, psi)/kT)`, where P_sim comes from the run without
a phi/psi potential. So the map is fitted on the sampled frames by
reweighting; no new simulation is needed for the fit.

1. Start from a smoothed Boltzmann inversion `kT ln(P_sim / P_target)`.
2. Minimise the cross-entropy of the target against the reweighted frames
   (5 deg bins) over the 72 x 72 node values, plus a thin-plate smoothness
   term (`--smoothness`, default 1e-4) and a penalty on curvature above
   `--max-curvature` (500 kJ/mol/rad^2). OpenMM's bicubic CMAP is exactly
   the tensor-product periodic cubic spline through the nodes, so the map and
   its derivatives on any grid are matrix products with exact gradients
   (`cmap_math.ConstrainedFit`; verified against OpenMM to 1e-13).
3. Held-out check: maps fitted on one half of the frames are applied to the
   other half. This estimates what a new simulation will show, and decides
   the verdict in the report.

The target is never smoothed. Moving whole basins up or down costs almost no
curvature, so basin populations are fitted to the original target; only fine
detail inside basins is traded for smoothness.

## Why the regularisation: the 5 fs time step

The Choi & Pappu grids are histograms: about 60 % of the 5 deg cells are
empty and the tails are single counts. Tested on Ace-X-NMe in OpenMM
(Langevin, 5 fs, 0.04/ps, 300 K):

| map | Ala | Asn |
|---|---|---|
| none | stable, 294-297 K | |
| fit to the raw target, no regularisation (max curvature ~11,000) | blows up after 50-140 ps (stable at 2 fs) | |
| raw target, curvature capped at 500 only | stable but 367 K; PII 52 % (target 59) | 310 K; PII 25 % (target 33) |
| target smoothed by 15 deg | stable, 300 K | basins up to 7 points off (PII/beta boundary blurred) |
| **thin-plate 1e-4 + cap 500 (default)** | **299 K; populations within statistical error** | **306 K; within statistical error** |
| smooth analytic well, curvature 400 | 298 K | |

The last row shows that stiffness alone is not the problem (GROMOS angle
terms are as stiff); roughness, i.e. rapidly changing curvature, is.

## Other choices

| option | default | reason |
|---|---|---|
| `--grid` | 72 (5 deg) | the targets are sharp (half of Ala's mass in 25 cells of 5 deg) |
| `--max-curvature` | 500 | safety cap; the thin-plate term does most of the work |
| `--smoothness` | 1e-4 | 1e-3 starts to lose populations (Asn PII -2.5 points); 1e-5 is rougher |

Effective sample sizes are low by nature (Asn ~5k, Trp ~9k of 1M frames)
because the targets are much narrower than the unbiased sampling.

## Files

- `build_cmap.py`: command line tool and fit.
- `cmap_math.py`: histograms, OpenMM-exact map evaluation, constrained fit,
  geometry.
