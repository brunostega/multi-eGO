# Backbone CMAP corrections

One `<RESNAME>.npz` per residue type, used with `multiego.py --cmap cmaps`.

Each file holds `cmap` (kJ/mol, 36 x 36, indexed `[phi, psi]`, nodes at
`-180 + 10 i` degrees), the `target` distribution it was fitted to and the
distribution `sampled` with the final map, at `temperature` (300 K).

Targets: Choi & Pappu, J. Chem. Theory Comput. 15, 1355 (2019),
doi:10.1021/acs.jctc.8b00572.  Fit: iterative Boltzmann inversion on the
blocked amino acid Ace-Xaa-NMe (`tools/cmap_fit/fit_cmap.py`).

Status per residue (blocked amino acid at 300 K; overlap = shared probability
between the target and the distribution sampled with the final map; aL error =
model minus target free energy of alphaL relative to alphaR, negative = alphaL
too favourable; "nan" where the validation run did not visit alphaL):

| Residue | Overlap | target aR / PPII / beta / aL (%) | model aR / PPII / beta / aL (%) | aL error (kJ/mol per residue) | Status |
|---|---|---|---|---|---|
| ALA | 0.90 | 10 / 58 / 31 / 0.3 | 11 / 59 / 30 / 0.0 | ~ -0.1 (fit), see text | provisional: flattened fit (depth 25), matrix EMAX 0.14 / EMAX_SC_BKBN 0.09 / EMAX_BKBN 0.08; aL from the fit 9.1 vs 9.2 kJ/mol |
| ARG | 0.80 | 7 / 52 / 41 / 0.3 | 6 / 46 / 46 / 1.2 | -3.7 | refit with --flatten pending (aL under-penalized) |
| ASN | 0.67 | 3 / 29 / 67 / 0.3 | 0 / 13 / 85 / 0.4 | -6.4 | refit with --flatten pending (aL under-penalized) |
| ASP | 0.72 | 5 / 36 / 58 / 0.3 | 11 / 33 / 56 / 0.2 | +3.2 | refit with --flatten pending (aL under-penalized) |
| CYS | 0.76 | 4 / 38 / 58 / 0.3 | 5 / 35 / 54 / 2.1 | -4.2 | refit with --flatten pending (aL under-penalized) |
| GLN | 0.81 | 8 / 42 / 49 / 0.3 | 5 / 38 / 54 / 1.2 | -4.5 | refit with --flatten pending (aL under-penalized) |
| GLU | 0.83 | 5 / 56 / 38 / 0.2 | 2 / 60 / 37 / 0.2 | -2.6 | refit with --flatten pending (aL under-penalized) |
| GLY | 0.84 | 32 / 8 / 11 / 29.4 | 39 / 6 / 8 / 32.0 | +0.3 | refit with --flatten pending (aL correct) |
| HIS | 0.75 | 4 / 38 / 57 / 0.1 | 5 / 41 / 54 / 0.3 | -2.0 | refit with --flatten pending (aL under-penalized) |
| ILE | 0.81 | 3 / 47 / 50 / 0.0 | 3 / 46 / 51 / 0.0 | +nan | refit with --flatten pending (aL under-penalized) |
| LEU | 0.82 | 10 / 57 / 32 / 0.3 | 4 / 63 / 32 / 0.6 | -4.3 | refit with --flatten pending (aL under-penalized) |
| LYS | 0.82 | 5 / 52 / 42 / 0.3 | 1 / 52 / 46 / 0.3 | -3.4 | refit with --flatten pending (aL under-penalized) |
| MET | 0.80 | 4 / 51 / 45 / 0.2 | 4 / 53 / 43 / 0.0 | +nan | refit with --flatten pending (aL under-penalized) |
| PHE | 0.79 | 6 / 42 / 51 / 0.2 | 5 / 46 / 49 / 0.0 | +5.6 | refit with --flatten pending (aL under-penalized) |
| PRO | 0.88 | 38 / 62 / 0 / 0.0 | 39 / 61 / 1 / 0.0 | +nan | refit with --flatten pending (aL under-penalized) |
| SER | 0.80 | 4 / 48 / 47 / 0.2 | 3 / 50 / 45 / 0.8 | -4.1 | refit with --flatten pending (aL under-penalized) |
| THR | 0.74 | 4 / 39 / 57 / 0.0 | 16 / 34 / 49 / 0.0 | +nan | refit with --flatten pending (aL under-penalized) |
| TRP | 0.68 | 2 / 52 / 46 / 0.1 | 4 / 41 / 54 / 0.7 | -3.2 | refit with --flatten pending (aL under-penalized) |
| TYR | 0.82 | 7 / 45 / 48 / 0.2 | 7 / 47 / 47 / 0.0 | +nan | refit with --flatten pending (aL under-penalized) |
| VAL | 0.78 | 3 / 48 / 49 / 0.0 | 7 / 44 / 49 / 0.0 | +nan | refit with --flatten pending (aL under-penalized) |

ALA (provisional).  Fitted on Ace-Ala-NMe generated with the matrix
parameters EMAX = 0.14, EMAX_SC_BKBN = 0.09, EMAX_BKBN = 0.08 (not the
defaults of type_definitions.py), with the flattening bias at depth 25
kJ/mol (`fit_cmap.py --flatten 25`).  Validation:
* main basins: final run of the fit 10.7 / 59.5 / 29.8 % (alphaR / PPII /
  beta, target 10.3 / 58.4 / 30.7); an independent 20 ns run gave alphaR
  15.6 +- 3.0 %: populations of this dipeptide have long correlation times;
* alphaL: 9.04 and 9.10 kJ/mol in the last fit iterations (target 9.23);
  the independent reweighted estimate is not reliable yet (bilinear vs
  bicubic interpolation of the bias in the reweighting);
* handedness on ideal Ac-(Ala)15-NMe helices: left-handed +107.7 kJ/mol
  above right-handed (experimental estimate ~ +130; the previous map gave
  about -12 on (AAQAA)3).

All other maps were fitted without the flattening bias and under-penalize
alphaL by 3-6 kJ/mol per residue for most residues: with the achiral
hydrogen-bond term this can make helices of L residues left-handed.  Refit
them with `tools/cmap_fit/fit_all.sh` (flattening on by default).

Residues without a file get no correction.  Only the 20 standard L-amino
acids are covered: CYS2 and D-amino acids are deliberately not included.

Reproduce a map with
`GMXLIB=$PWD tools/cmap_fit/fit_all.sh WORKDIR TARGETDIR ITERATIONS NS_PER_ITERATION RESNAME ...`
(ITERATIONS 6, NS 5 for the maps above, 8 x 12 ns for PRO).
