# Backbone CMAP corrections

One `<RESNAME>.npz` per residue type, used with `multiego.py --cmap cmaps`.

Each file holds `cmap` (kJ/mol, 36 x 36, indexed `[phi, psi]`, nodes at
`-180 + 10 i` degrees), the `target` distribution it was fitted to and the
distribution `sampled` with the final map, at `temperature` (300 K).

Targets: Choi & Pappu, J. Chem. Theory Comput. 15, 1355 (2019),
doi:10.1021/acs.jctc.8b00572.  Fit: iterative Boltzmann inversion on the
blocked amino acid Ace-Xaa-NMe (`tools/cmap_fit/fit_cmap.py`).

Status per residue (blocked amino acid at 300 K; overlap = shared probability
between the target and the distribution sampled with the final map; GLY also
populates alphaL, which is not shown):

| Residue | Overlap | target aR / PPII / beta (%) | model aR / PPII / beta (%) | Status |
|---|---|---|---|---|
| PRO | 0.88 | 38 / 62 / 0 | 39 / 61 / 1 | final |
| GLY | 0.84 | 32 / 8 / 11 | 39 / 6 / 8 | final |
| GLU | 0.83 | 5 / 56 / 38 | 2 / 60 / 37 | final |
| LYS | 0.82 | 5 / 52 / 42 | 1 / 52 / 46 | final |
| LEU | 0.82 | 10 / 57 / 32 | 4 / 63 / 32 | final |
| TYR | 0.82 | 7 / 45 / 48 | 7 / 47 / 47 | final |
| GLN | 0.81 | 8 / 42 / 49 | 5 / 38 / 54 | refit pending (fitted before the pseudo-count fix) |
| ILE | 0.81 | 3 / 47 / 50 | 3 / 46 / 51 | final |
| SER | 0.80 | 4 / 48 / 47 | 3 / 50 / 45 | final |
| ARG | 0.80 | 7 / 52 / 41 | 6 / 46 / 46 | final |
| MET | 0.80 | 4 / 51 / 45 | 4 / 53 / 43 | final |
| PHE | 0.79 | 6 / 42 / 51 | 5 / 46 / 49 | final |
| VAL | 0.78 | 3 / 48 / 49 | 7 / 44 / 49 | final |
| ALA | 0.77 | 10 / 58 / 31 | 16 / 48 / 29 | refit pending (fitted before the pseudo-count fix) |
| CYS | 0.76 | 4 / 38 / 58 | 5 / 35 / 54 | final |
| HIS | 0.75 | 4 / 38 / 57 | 5 / 41 / 54 | final |
| THR | 0.74 | 4 / 39 / 57 | 16 / 34 / 49 | longer refit pending (excess alphaR) |
| ASP | 0.72 | 5 / 36 / 58 | 11 / 33 / 56 | longer refit pending (excess alphaR) |
| TRP | 0.68 | 2 / 52 / 46 | 4 / 41 / 54 | longer refit pending (PPII/beta inverted) |
| ASN | 0.67 | 3 / 29 / 67 | 0 / 13 / 85 | refit pending (not converged) |

Residues without a file get no correction.  Only the 20 standard L-amino
acids are covered: CYS2 and D-amino acids are deliberately not included.

Reproduce a map with
`GMXLIB=$PWD tools/cmap_fit/fit_all.sh WORKDIR TARGETDIR ITERATIONS NS_PER_ITERATION RESNAME ...`
(ITERATIONS 6, NS 5 for the maps above, 8 x 12 ns for PRO).
