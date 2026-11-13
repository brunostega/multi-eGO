# Backbone CMAP corrections

One `<RESNAME>.npz` per residue type, used with `multiego.py --cmap cmaps`.

Each file holds `cmap` (kJ/mol, 36 x 36, indexed `[phi, psi]`, nodes at
`-180 + 10 i` degrees), the `target` distribution it was fitted to and the
distribution `sampled` with the final map, at `temperature` (300 K).

Targets: Choi & Pappu, J. Chem. Theory Comput. 15, 1355 (2019),
doi:10.1021/acs.jctc.8b00572.  Fit: iterative Boltzmann inversion on the
blocked amino acid Ace-Xaa-NMe (`tools/cmap_fit/fit_cmap.py`).

Status: preliminary.  ALA and GLN were fitted before the pseudo-count fix of
the fitting algorithm and should be redone; THR, ASP and TRP need longer
sampling (5 ns per iteration was used).  Residues without a file get no
correction.
