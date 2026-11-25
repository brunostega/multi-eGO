# Running multi-eGO models with OpenMM

`multiego.py` writes, next to `topol_mego.top` and `ffnonbonded.itp`, the
OpenMM parameter file `mego_params.h5` (`--backend both`, the default;
`--backend openmm` writes only this file). It holds the same force field:
every bonded term, the pairs and exclusions, the nonbonded c6/c12 (mg prior
per pair of chemical types plus the learned contacts), and, for models built
with `--backbone cmap`, the backbone CMAPs of the residues present
(from `multi-ego-basic.ff/cmap`, or `--cmap_dir`), and, for models built
with `--hbond`, the geometric backbone hydrogen bond.

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

## Geometric hydrogen bond (`--hbond`)

```bash
python multiego.py --system gpref --egos mg --hbond --hbond_epsilon_local 10 --hbond_epsilon_nonlocal 5
```

Every backbone N-H...O=C pair gets

    E = -eps * f(r) * A_don(N-H...O) * A_acc(H...O=C)

with r the H...O distance, f = 1 up to `--hbond_r0` (0.20 nm) and a Gaussian
of width `--hbond_width` (0.035 nm) beyond, A_don a Gaussian centred at 180
degrees (width 40), A_acc a Gaussian centred at 147 degrees (width 25), both
zero below 90 degrees. eps is `--hbond_epsilon_local` for donor and acceptor
3-4 residues apart and `--hbond_epsilon_nonlocal` (default: the local value)
for 5 or more residues apart or different chains; pairs closer than 3
residues keep their Lennard-Jones. The term replaces the H-O Lennard-Jones
interaction; its repulsive wall is the prior H-O c12, never a learned value.

With `--hbond` the prior of the backbone H-O, N-O, C-N, N-N and O-O pairs
becomes purely repulsive (c6 = 0, the default repulsive c12 of the pair) in
every output, `ffnonbonded.itp` included, since the GROMACS files of the
reference model define the prior of the next production step. C-O keeps its
prior. Learned contacts are generated as usual, except backbone H-O and N-O
between atoms 3 or more residues apart or on different chains: these describe
the hydrogen bond itself and keep the repulsive prior (they are not learned,
so the hydrogen bond is not put back into the isotropic matrix). Learned local
H-O / N-O contacts are written in `[ pairs ]`, so that the matrix entry, which
also applies between chains, keeps the prior. The GROMACS files have no
hydrogen-bond term. The energy appears as `H-bond` in `<output>_energy.csv`.

The term is an OpenMM `CustomHbondForce`. On GPU its cost is about linear in
the system size (blocks of donors and acceptors farther apart than the
cutoff are skipped): about 12% of the step time for 1000 GB1 copies 5 nm
apart, 5% when the copies touch. On the CPU platform it evaluates all
donor-acceptor pairs and becomes expensive beyond a few tens of chains.

Velocities are generated at the start temperature (`.gro` velocities are not
read); use `--restart` to continue a run. OpenMM's Langevin integrator and
GROMACS `sd` sample the same ensemble; trajectories are not identical.
