# Running multi-eGO models with OpenMM

`multiego.py` writes, next to `topol_mego.top` and `ffnonbonded.itp`, the
OpenMM parameter file `mego_params.h5` (`--backend both`, the default;
`--backend openmm` writes only this file). It holds the same force field:
every bonded term, the pairs and exclusions, the nonbonded c6/c12 (mg prior
per pair of chemical types plus the learned contacts), and, for models built
with `--backbone cmap`, the backbone CMAPs of the residues present
(from `multi-ego-basic.ff/cmap`, or `--cmap_dir`).

```bash
python multiego.py --system gpref --egos mg --backbone cmap
python openmm/run.py --params outputs/gpref/mg_1 --conf gb1.gro \
    --temperature 300 --nsteps 1000000 --minimize --output gb1
```

`run.py` follows `mdps/ff_aa.mdp`: Langevin dynamics (sd), 5 fs, friction
1/25 ps^-1, 2.5 nm cutoff with potential shift, no dispersion correction,
centre-of-mass motion removal every 20 steps, no constraints; trajectory and
log every 50000 steps.

| option | meaning |
|---|---|
| `--molecules GB1:10,BNZ:5` | composition in coordinate order (the GROMACS `[ molecules ]`); default one copy of each molecule type |
| `--minimize` | energy minimisation first (tolerance `--em-tolerance`, 10 kJ/mol/nm) |
| `--posre 1000` | heavy-atom position restraints to the starting positions (ff_aa-posre.mdp) |
| `--restart run.chk` | continue a run (positions, velocities, step); outputs are appended |
| `--platform`, `--device`, `--precision`, `--threads` | OpenMM platform settings |

Outputs: `<output>.xtc`, `<output>.log` (energies, temperature, speed),
`<output>_energy.csv` (every term, like `gmx energy`), `<output>.chk`,
`<output>_final.gro` and `.pdb`.

Velocities are generated at the start temperature (`.gro` velocities are not
read); use `--restart` to continue a run. OpenMM's Langevin integrator and
GROMACS `sd` sample the same ensemble; trajectories are not identical.
