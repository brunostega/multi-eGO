# cmap_builder

Builds the backbone CMAPs (`multi-ego-basic.ff/cmap/<RES>.cmap`) that make
blocked dipeptides (Ace-X-NMe) reproduce target Ramachandran distributions,
by default those of Choi & Pappu, JCTC 15, 1355 (2019).

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

- Boltzmann inversion on bins centred on the map nodes:
  `U = U_sampled + kT [ln(n_sim + a) - ln(N p_target + a)]`.
- Periodic Gaussian smoothing (5 deg) and a ceiling (30 kJ/mol above the
  minimum; regions above it hold < 0.1 % of the target).
- Refinement by reweighting the same frames: the distribution with the map is
  exactly `P_sim exp(-(U - U_sampled)/kT)`, with `U` evaluated exactly as
  OpenMM evaluates it (`cmap_math.BicubicMap`, tested against OpenMM to 1e-10).
  Corrections are computed in raw-frame units with pseudo-count `a = 20`.
- Held-out check: maps fitted on one half of the frames are applied to the
  other half. Reweighting the fitted frames is optimistic; with a small
  pseudo-count and many iterations a map can fit them perfectly and still dig
  spurious wells that a new simulation falls into. The held-out errors
  decide the verdict in the report.

## Time step: sharp maps need 2 fs

The Choi & Pappu targets are histograms: about 60 % of the 5 deg cells are
empty and the tails are single counts. Fitted as they are (`--target-sigma 0`),
the maps reproduce the targets almost exactly but reach curvatures of
5,000-13,000 kJ/mol/rad^2 at the edges of the empty regions (the omega term
peaks at 134). Tested on Ace-Ala-NMe (OpenMM, Langevin, 5 fs):

| map | max curvature | 5 fs | 2 fs | basins vs target | TV@20 deg |
|---|---|---|---|---|---|
| no map | - | stable, 294-297 K | | | |
| `--target-sigma 0` | 11,100 | blows up after 50-140 ps | stable, 294 K | exact | 0.008 |
| `--target-sigma 5` | 5,500 | blows up after 250-340 ps | | | |
| `--target-sigma 7.5 --sigma 7.5` | 2,600 | one of two seeds blows up | | | |
| `--target-sigma 10 --sigma 10` | 1,350 | stable, 312-316 K (too hot) | | within 0.4 pt | 0.23 |
| `--target-sigma 15 --sigma 15` | 540 | stable, 300 K | | within 1.5 pt | 0.36 |

So either keep sharp maps and run with a 2 fs step, or smooth the targets
(about 15 deg) and keep 5 fs with broader basins. The report lists each map's
maximum curvature.

## Defaults and why

| option | default | reason |
|---|---|---|
| `--grid` | 72 (5 deg) | the targets are sharp (half of Ala's mass in 25 cells of 5 deg); 10 deg grids fit worse at 2.5 and 10 deg resolution |
| `--pseudo` | 20 | best held-out errors; 1 overfits badly (Ala held-out aR 4.7 % vs 10.4 %) |
| `--umax` | 30 | same fit as 50, half the steepest gradients (~300 vs ~650 kJ/mol/rad) |
| `--iterations` | 60 | aL converges slowly |
| `--target-sigma` | 0 | faithful to the target; see the time-step section |

## Files

- `build_cmap.py`: command line tool and fit.
- `cmap_math.py`: histograms, OpenMM-exact bicubic evaluation, geometry.
