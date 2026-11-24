#!/usr/bin/env python3
"""
Build backbone CMAPs from dipeptide simulations run WITHOUT any phi/psi
potential (models generated with ``--backbone cmap``), so that the dipeptides
reproduce target Ramachandran distributions (Choi & Pappu, JCTC 2019), with
maps smooth enough for the multi-eGO 5 fs time step.

Method
------
With the map U active, the dipeptide samples exactly

    P(phi, psi) ~ P_sim(phi, psi) exp(-U(phi, psi)/kT)

(P_sim: the run without phi/psi potential; U only depends on phi, psi). So the
map is fitted on the sampled frames by reweighting:

1. Start: Boltzmann inversion U0 = kT ln(P_sim / P_target), heavily smoothed.
2. Minimise the cross-entropy of the target against the reweighted frames
   (5 degree bins) over the 72 x 72 node values, plus a thin-plate smoothness
   term (``--smoothness``) and a penalty on curvature above
   ``--max-curvature`` (500 kJ/mol/rad^2). The map is evaluated exactly as
   OpenMM evaluates it (tensor-product periodic cubic spline, identical to
   OpenMM's bicubic CMAP).

The regularisation is what the multi-eGO 5 fs time step needs. Maps that
follow the targets exactly (60 % of the target cells are empty, the tails are
single counts) reach ~11,000 kJ/mol/rad^2 and blow up; maps that are only
capped in curvature are rough and heat the system (367 K for Ala). The
target itself is not smoothed: moving whole basins up or down costs almost
no curvature, so basin populations are still fitted; only fine detail inside
basins is traded for smoothness.

A held-out check (fit on one half of the frames, test on the other) estimates
what a new simulation with the map will show.

Usage::

    python tools/cmap_builder/build_cmap.py --sim runs/ --target choi_pappu/ \\
        --temperature 300 --out multi-ego-basic.ff/cmap

``--sim`` holds ``<RES>.xvg`` from ``gmx rama`` (one dipeptide per file;
psi = N-CA-C-N(i+1)). ``--target`` holds ``<RES>.csv.gz`` (or ``.csv``), the
144 x 144 Choi & Pappu grids, rows = psi, columns = phi, 2.5 degree bins.
"""

import argparse
import datetime
import gzip
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))
from multiego.openmm_backend.cmap import read_cmap, write_cmap  # noqa: E402
from cmap_math import (  # noqa: E402
    KB,
    BicubicMap,
    ConstrainedFit,
    coarsen_target,
    node_histogram,
    smooth_periodic,
)

RESIDUES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]  # fmt: skip


# ----------------------------------------------------------------------- input
def read_rama(path):
    """phi, psi (degrees) from a ``gmx rama`` .xvg or a two-column text file."""
    import pandas as pd

    with open(path) as fh:
        lines = [line for line in fh if line.strip() and line[0] not in "@#"]
    data = pd.read_csv(io.StringIO("".join(lines)), sep=r"\s+", header=None, usecols=[0, 1])
    phi, psi = data[0].to_numpy(float), data[1].to_numpy(float)
    if len(phi) == 0:
        raise ValueError(f"{path}: no data")
    return phi, psi


def read_target(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        grid = np.loadtxt(fh, delimiter=",")
    if grid.shape != (144, 144):
        raise ValueError(f"{path}: expected a 144x144 grid, found {grid.shape}")
    if np.any(grid < 0) or grid.sum() <= 0:
        raise ValueError(f"{path}: weights must be non-negative")
    return grid / grid.sum()


# --------------------------------------------------------------------- basins
def basin_masks(phi, psi):
    """Basins used for reporting (degrees)."""
    up = (psi > 50) | (psi < -120)
    return {
        "aR": (phi < 0) & ~up,
        "PII": (phi >= -100) & (phi < 0) & up,
        "beta": (phi < -100) & up,
        "aL": (phi >= 0) & (psi > -90) & (psi < 90),
        "phi>0 other": (phi >= 0) & ~((psi > -90) & (psi < 90)),
    }


def basin_populations(phi, psi, weights):
    total = weights.sum()
    return {k: weights[m].sum() / total for k, m in basin_masks(phi, psi).items()}


def target_basins(target144):
    c = -180.0 + 1.25 + 2.5 * np.arange(144)
    phi, psi = np.meshgrid(c, c)  # rows psi, cols phi
    return basin_populations(phi.ravel(), psi.ravel(), target144.ravel())


def dg_al_ar(pops, kt):
    """Free energy of aL relative to aR, kJ/mol."""
    if pops["aL"] <= 0 or pops["aR"] <= 0:
        return float("inf")
    return -kt * np.log(pops["aL"] / pops["aR"])


# ------------------------------------------------------------------------ fit
def max_curvature(grid, step_deg=0.5):
    """Largest |d2U/dphi2| or |d2U/dpsi2| of the map as OpenMM applies it, kJ/mol/rad^2."""
    a = np.arange(-180.0, 180.0, step_deg)
    phi, psi = np.meshgrid(a, a)
    e = BicubicMap(grid).energy(phi, psi)
    h2 = np.radians(step_deg) ** 2
    d_phi = (np.roll(e, -1, 1) - 2 * e + np.roll(e, 1, 1)) / h2
    d_psi = (np.roll(e, -1, 0) - 2 * e + np.roll(e, 1, 0)) / h2
    return float(max(np.abs(d_phi).max(), np.abs(d_psi).max()))


def _sampled_energy(phi, psi, sampled_map):
    if sampled_map is None:
        return np.zeros(len(phi))
    return BicubicMap(sampled_map - sampled_map.min()).energy(phi, psi)


def reweight(phi, psi, grid, temperature, sampled_map=None):
    """Weights turning the sampled frames into the distribution with ``grid`` active."""
    de = BicubicMap(grid).energy(phi, psi) - _sampled_energy(phi, psi, sampled_map)
    return np.exp(-(de - de.min()) / (KB * temperature))


def inversion(phi, psi, target144, n=72, temperature=300.0, pseudo=20.0, sigma_deg=15.0, umax=30.0, sampled_map=None):
    """
    Smoothed Boltzmann inversion on node-centred bins (starting point of the fit):
    U = U_sampled + kT [ln(n_sim + a) - ln(N p_target + a)], smoothed, clamped.
    """
    kt = KB * temperature
    p_t = coarsen_target(target144, n)
    w0 = None
    if sampled_map is not None:
        e_s = _sampled_energy(phi, psi, sampled_map)
        w0 = np.exp((e_s - e_s.max()) / kt)
        w0 *= len(phi) / w0.sum()  # in frame units
    h = node_histogram(phi, psi, n, w0)
    corr = kt * (np.log(h + pseudo) - np.log(len(phi) * p_t + pseudo))
    u = np.minimum(smooth_periodic(corr - corr.min(), sigma_deg * n / 360.0), umax)
    return u - u.min()


def fit_map(
    phi,
    psi,
    target144,
    n=72,
    temperature=300.0,
    max_curvature_limit=500.0,
    lam=1e-7,
    smoothness=1e-4,
    sampled_map=None,
    maxiter=2000,
    fine_deg=0.5,
):
    """
    Curvature-limited fit of a map to frames (phi, psi).

    sampled_map : (n, n) grid or None
        Map active when the frames were sampled (None: no phi/psi potential).

    Returns
    -------
    grid : (n, n) kJ/mol, rows = psi, cols = phi, min 0
    info : dict with diagnostics (weights of the frames with the new map, ...)
    """
    kt = KB * temperature
    weights0 = None
    if sampled_map is not None:
        e_s = _sampled_energy(phi, psi, sampled_map)
        weights0 = np.exp((e_s - e_s.max()) / kt)  # undo the sampled map
    u0 = inversion(phi, psi, target144, n=n, temperature=temperature, sampled_map=sampled_map)
    cf = ConstrainedFit(phi, psi, target144, n=n, temperature=temperature, fine_deg=fine_deg, weights0=weights0)
    u, res = cf.fit(u0, kappa=max_curvature_limit, lam=lam, maxiter=maxiter, mu=smoothness)
    w = reweight(phi, psi, u, temperature, sampled_map)
    p_t = coarsen_target(target144, n)
    w_sampled = np.ones(len(phi)) if weights0 is None else weights0
    return u, {
        "frames": len(phi),
        "unbiased_basins": basin_populations(phi, psi, w_sampled),
        "effective_frames": w.sum() ** 2 / (w * w).sum(),
        "total_variation_initial": 0.5 * np.abs(node_histogram(phi, psi, n, w_sampled) / w_sampled.sum() - p_t).sum(),
        "total_variation_final": 0.5 * np.abs(node_histogram(phi, psi, n, w) / w.sum() - p_t).sum(),
        "iterations": res.nit,
        "max_curvature": max_curvature(u),
        "weights": w,
    }


def cross_validate(phi, psi, target144, temperature, **kw):
    """
    Fit on one half of the frames, evaluate on the other, both ways.

    Returns dict with the max basin error (aR, PII, beta; probability) and the
    aL-aR dG error (kJ/mol) for each direction.
    """
    kt = KB * temperature
    t = target_basins(target144)
    half = len(phi) // 2
    out = {"basin": [], "dg": []}
    for fit, test in ((slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))):
        grid, _ = fit_map(phi[fit], psi[fit], target144, temperature=temperature, **kw)
        w = reweight(phi[test], psi[test], grid, temperature, kw.get("sampled_map"))
        r = basin_populations(phi[test], psi[test], w)
        out["basin"].append(max(abs(r[k] - t[k]) for k in ("aR", "PII", "beta")))
        out["dg"].append(dg_al_ar(r, kt) - dg_al_ar(t, kt))
    return out


def block_errors(phi, psi, w, nblocks=10):
    """Standard errors of the basin populations from contiguous blocks."""
    vals = [
        basin_populations(a, b, c)
        for a, b, c in zip(np.array_split(phi, nblocks), np.array_split(psi, nblocks), np.array_split(w, nblocks))
    ]
    return {k: np.std([v[k] for v in vals], ddof=1) / np.sqrt(nblocks) for k in vals[0]}


# --------------------------------------------------------------------- driver
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", required=True, help="directory with <RES>.xvg (gmx rama)")
    ap.add_argument("--target", required=True, help="directory with <RES>.csv.gz target grids")
    ap.add_argument("--temperature", type=float, required=True, help="K, of the dipeptide simulations")
    ap.add_argument("--out", required=True, help="output directory for <RES>.cmap")
    ap.add_argument("--report", default=None, help="markdown report (default: <out>/README.md)")
    ap.add_argument("--residues", nargs="*", default=RESIDUES)
    ap.add_argument("--grid", type=int, default=72, help="map nodes per axis (144/grid must be even)")
    ap.add_argument(
        "--max-curvature",
        type=float,
        default=500.0,
        help="curvature limit, kJ/mol/rad^2; 500 is integrable at the multi-eGO 5 fs time step",
    )
    ap.add_argument(
        "--smoothness",
        type=float,
        default=1e-4,
        help="weight of the thin-plate smoothness term. 1e-4 keeps basin populations and runs at the right "
        "temperature at 5 fs; 0 (curvature cap only) makes rough maps that heat the system",
    )
    ap.add_argument(
        "--sampled-maps",
        default=None,
        help="directory with the maps active in the --sim runs (second iterations); "
        "omit for runs without any phi/psi potential",
    )
    ap.add_argument("--no-cross-validation", action="store_true", help="skip the half/half check")
    ap.add_argument("--note", default="", help="free text stored in each map header (e.g. model, matrix)")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    kt = KB * args.temperature
    rows = []
    stamp = datetime.date.today().isoformat()
    for res in args.residues:
        sim_path = os.path.join(args.sim, f"{res}.xvg")
        tgt_path = next(
            (
                p
                for p in (os.path.join(args.target, f"{res}.csv.gz"), os.path.join(args.target, f"{res}.csv"))
                if os.path.exists(p)
            ),
            None,
        )
        if not os.path.exists(sim_path) or tgt_path is None:
            print(f"{res}: missing simulation or target data, skipped", flush=True)
            continue
        phi, psi = read_rama(sim_path)
        target = read_target(tgt_path)
        sampled = None
        if args.sampled_maps:
            _, sampled = read_cmap(os.path.join(args.sampled_maps, f"{res}.cmap"))
            if sampled.shape[0] != args.grid:
                raise ValueError(f"{res}: sampled map grid {sampled.shape[0]} differs from --grid {args.grid}")
        kw = dict(n=args.grid, max_curvature_limit=args.max_curvature, smoothness=args.smoothness, sampled_map=sampled)
        grid, info = fit_map(phi, psi, target, temperature=args.temperature, **kw)
        info["cv"] = None if args.no_cross_validation else cross_validate(phi, psi, target, args.temperature, **kw)
        w = info["weights"]
        t = target_basins(target)
        r = basin_populations(phi, psi, w)
        se = block_errors(phi, psi, w)
        write_cmap(
            os.path.join(args.out, f"{res}.cmap"),
            res,
            grid,
            {
                "method": "curvature-limited fit by reweighting of a run "
                + ("with the maps in " + args.sampled_maps if args.sampled_maps else "without phi/psi potential"),
                "target": "Choi & Pappu, JCTC 15, 1355 (2019), " + os.path.basename(tgt_path),
                "simulation": f"{os.path.basename(sim_path)}, {info['frames']} frames",
                "temperature": f"{args.temperature} K",
                "max curvature": f"{info['max_curvature']:.0f} kJ/mol/rad^2 (limit {args.max_curvature:g}, "
                f"smoothness {args.smoothness:g})",
                "fit": f"total variation vs target {info['total_variation_initial']:.4f} sampled -> "
                f"{info['total_variation_final']:.4f} reweighted; effective frames {info['effective_frames']:.0f}",
                "created": stamp,
                **({"note": args.note} if args.note else {}),
            },
        )
        print(
            f"{res}: max curvature {info['max_curvature']:.0f}, "
            + " ".join(f"{k} {100 * r[k]:.1f}/{100 * t[k]:.1f}" for k in ("aR", "PII", "beta", "aL")),
            flush=True,
        )
        rows.append((res, info, t, r, se))

    report = args.report or os.path.join(args.out, "README.md")
    write_report(report, rows, args, kt)
    print(f"maps in {args.out}, report {report}")


def write_report(path, rows, args, kt):
    keys = ["aR", "PII", "beta", "aL"]
    out = [
        "# Backbone CMAPs",
        "",
        f"Built {datetime.date.today().isoformat()} with `tools/cmap_builder/build_cmap.py` at {args.temperature} K, "
        f"grid {args.grid} ({360 / args.grid:g} deg), curvature limit {args.max_curvature:g} kJ/mol/rad^2 "
        "(multi-eGO 5 fs time step), "
        + (
            f"runs sampled with the maps in `{args.sampled_maps}`."
            if args.sampled_maps
            else "runs without phi/psi potential."
        ),
        "",
        "Populations in % : target / sampled run / **with the map** (reweighted frames) ± block SE. "
        "Basins: aR phi<0, -120<psi<50; PII -100<=phi<0 and beta phi<-100 (psi outside the aR band); "
        "aL phi>=0, -90<psi<90. dG(aL-aR) in kJ/mol.",
        "",
        "Held-out check: a map fitted on one half of the frames, applied to the other half, both ways; it "
        "estimates what a new simulation with the map will show. Verdict **ok** if the held-out basin error "
        "is <= 2 points and the held-out dG(aL-aR) error <= 1 kJ/mol (aL only where the target has >= 0.1 %); "
        "otherwise **iterate**: rerun the dipeptide with the map and rebuild with `--sampled-maps`.",
        "",
        "| res | "
        + " | ".join(keys)
        + " | dG(aL-aR) tgt / map | held-out basin err (pt) | held-out dG err | eff. frames | max curvature | verdict |",
        "|---" * (len(keys) + 7) + "|",
    ]
    for res, info, t, r, se in rows:
        s = info["unbiased_basins"]
        cells = [f"{100 * t[k]:.1f} / {100 * s[k]:.1f} / **{100 * r[k]:.1f}** ± {100 * se[k]:.1f}" for k in keys]
        dt, dr = dg_al_ar(t, kt), dg_al_ar(r, kt)
        cv = info.get("cv")
        if cv:
            b_err = max(cv["basin"])
            dg_err = max(cv["dg"], key=abs)
            has_al = t["aL"] >= 1e-3
            ok = b_err <= 0.02 and (not has_al or abs(dg_err) <= 1.0)
            cv_cells = f"{100 * b_err:.1f} | " + (f"{dg_err:+.2f}" if has_al else "n/a")
            verdict = "ok" if ok else "iterate"
        else:
            cv_cells, verdict = "- | -", "not checked"
        out.append(
            f"| {res} | "
            + " | ".join(cells)
            + f" | {dt:.2f} / {dr:.2f} | {cv_cells} | {info['effective_frames']:,.0f}"
            + f" | {info['max_curvature']:,.0f} | {verdict} |"
        )
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
