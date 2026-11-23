#!/usr/bin/env python3
"""
Build backbone CMAPs from dipeptide simulations run WITHOUT any phi/psi
potential (models generated with ``--backbone cmap``), so that the dipeptides
reproduce target Ramachandran distributions (Choi & Pappu, JCTC 2019).

Method
------
1. Boltzmann inversion on node-centred bins::

       U0 = kT [ ln(n_sim + a) - ln(N p_target + a) ]

   n_sim: frames per bin, N: total frames, a: pseudo-count (frames). Without
   a phi/psi term in the sampled model this is exact where sampling is good.
2. Periodic Gaussian smoothing and an upper clamp.
3. Refinement by reweighting the same frames: the distribution the model will
   have with the map is exactly P_sim(phi, psi) exp(-U(phi, psi)/kT), with U
   evaluated as OpenMM evaluates it (bicubic, ``cmap_math.BicubicMap``).
   The map is corrected with kT ln(P_reweighted / P_target) until the
   reweighted distribution matches the target. This removes the smoothing and
   interpolation errors; it cannot fix bins the simulation never visited.

No new simulations are needed; validating with a new run is still advised.

Usage::

    python tools/cmap_builder/build_cmap.py --sim runs/ --target choi_pappu/ \\
        --temperature 300 --out multi-ego-basic.ff/cmap --report cmap_report.md

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
from cmap_math import KB, BicubicMap, coarsen_target, node_histogram, smooth_periodic  # noqa: E402

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


# ----------------------------------------------------------------------- fit
def regularize_target(target144, sigma_deg):
    """
    Periodic Gaussian smoothing of a target histogram (2.5 degree bins).

    About 60 % of the target cells are empty and the tails are single counts;
    without smoothing the map digs spikes and cliffs there, with curvatures that
    the 5 fs integrator cannot follow. Returns a normalised 144 x 144 grid.
    """
    p = np.maximum(smooth_periodic(target144, sigma_deg / 2.5), 0.0)
    return p / p.sum()


def fit_map(
    phi,
    psi,
    target144,
    n=72,
    temperature=300.0,
    pseudo=20.0,
    sigma_deg=5.0,
    umax=30.0,
    iterations=60,
    tol=1e-3,
    min_frames=1,
    sampled_map=None,
    target_sigma_deg=0.0,
    log=print,
):
    """
    Fit a map to frames (phi, psi) so that they reproduce the target.

    sampled_map : (n, n) grid or None
        Map that was active when the frames were sampled (None: sampled with
        no phi/psi potential). The new map is then U_sampled + correction and
        frames are reweighted with exp(-(U_new - U_sampled)/kT).

    Returns
    -------
    grid : (n, n) kJ/mol, rows = psi, cols = phi, min 0
    info : dict with diagnostics
    """
    kt = KB * temperature
    nframes = len(phi)
    # The targets are histograms: 60 % of the cells are empty and the tails are
    # single counts. Smoothing them removes spikes the map would otherwise dig.
    if target_sigma_deg > 0:
        target144 = regularize_target(target144, target_sigma_deg)
    p_t = coarsen_target(target144, n)
    sigma = sigma_deg * n / 360.0

    h = node_histogram(phi, psi, n)
    e_sampled = 0.0
    u_sampled = np.zeros((n, n))
    if sampled_map is not None:
        if sampled_map.shape != (n, n):
            raise ValueError(f"sampled map is {sampled_map.shape}, expected {(n, n)}")
        u_sampled = sampled_map - sampled_map.min()
        e_sampled = BicubicMap(u_sampled).energy(phi, psi)
    corr0 = kt * (np.log(h + pseudo) - np.log(nframes * p_t + pseudo))
    u = np.minimum(u_sampled + smooth_periodic(corr0 - corr0.min(), sigma), umax)
    u -= u.min()

    history = []
    for it in range(iterations + 1):
        de = BicubicMap(u).energy(phi, psi) - e_sampled
        w = np.exp(-(de - de.min()) / kt)
        ess = w.sum() ** 2 / (w * w).sum()
        p_r = node_histogram(phi, psi, n, w) / w.sum()
        tv = 0.5 * np.abs(p_r - p_t).sum()
        history.append((it, tv, ess))
        log(f"    iteration {it:2d}: total variation vs target {tv:.4f}, effective frames {ess:,.0f}")
        if it == iterations or (it > 0 and history[-2][1] - tv < tol * history[-2][1]):
            break
        # Correction in raw-frame units, as in the first step: n_raw frames in a
        # bin vs the raw frames it would need (at its mean weight) to carry the
        # target probability. Using effective frames instead would let the
        # pseudo-count freeze sparse but well-sampled regions such as aL.
        w_bin = node_histogram(phi, psi, n, w)
        wbar = np.where(h > 0, w_bin / np.maximum(h, 1), 1.0)
        n_needed = p_t * w.sum() / wbar
        # bins with few raw frames get no correction: their reweighted
        # probability is too noisy and correcting it digs spurious wells
        corr = np.where(h >= max(min_frames, 1), kt * (np.log(h + pseudo) - np.log(n_needed + pseudo)), 0.0)
        u = np.minimum(u + smooth_periodic(corr, sigma / 2), umax)
        u -= u.min()

    info = {
        "unbiased_basins": basin_populations(
            phi, psi, np.exp(-(e_sampled - np.min(e_sampled)) / kt) if sampled_map is not None else np.ones_like(phi)
        ),
        "frames": nframes,
        "effective_frames": ess,
        "total_variation_initial": 0.5 * np.abs(h / nframes - p_t).sum(),
        "total_variation_final": tv,
        "iterations": it,
        "weights": w,
        "history": history,
        "clamped_nodes": int(np.sum(u >= umax - u.min() - 1e-9)),
    }
    return u, info


def cross_validate(phi, psi, target144, temperature, **kw):
    """
    Fit on one half of the frames, evaluate on the other, both ways.

    Reweighting the frames a map was fitted to is optimistic; the held-out
    half shows what a new simulation with the map is likely to give.

    Returns
    -------
    dict: max basin error (aR, PII, beta; probability) and the aL-aR dG error
    (kJ/mol) for each direction.
    """
    kt = KB * temperature
    t = target_basins(target144)
    half = len(phi) // 2
    sampled = kw.pop("sampled_map", None)
    out = {"basin": [], "dg": []}
    for fit, test in ((slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))):
        grid, _ = fit_map(
            phi[fit], psi[fit], target144, temperature=temperature, sampled_map=sampled, log=lambda *a: None, **kw
        )
        de = BicubicMap(grid).energy(phi[test], psi[test])
        if sampled is not None:
            de = de - BicubicMap(sampled - sampled.min()).energy(phi[test], psi[test])
        w = np.exp(-(de - de.min()) / kt)
        r = basin_populations(phi[test], psi[test], w)
        out["basin"].append(max(abs(r[k] - t[k]) for k in ("aR", "PII", "beta")))
        out["dg"].append(dg_al_ar(r, kt) - dg_al_ar(t, kt))
    return out


def max_curvature(grid, step_deg=0.5):
    """Largest |d2U/dphi2| or |d2U/dpsi2| of the map as OpenMM applies it, kJ/mol/rad^2."""
    a = np.arange(-180.0, 180.0, step_deg)
    phi, psi = np.meshgrid(a, a)
    e = BicubicMap(grid).energy(phi, psi)
    h2 = np.radians(step_deg) ** 2
    d_phi = (np.roll(e, -1, 1) - 2 * e + np.roll(e, 1, 1)) / h2
    d_psi = (np.roll(e, -1, 0) - 2 * e + np.roll(e, 1, 0)) / h2
    return float(max(np.abs(d_phi).max(), np.abs(d_psi).max()))


def block_errors(phi, psi, w, nblocks=10):
    """Standard errors of the basin populations from contiguous blocks."""
    vals = [
        basin_populations(a, b, c)
        for a, b, c in zip(np.array_split(phi, nblocks), np.array_split(psi, nblocks), np.array_split(w, nblocks))
    ]
    return {k: np.std([v[k] for v in vals], ddof=1) / np.sqrt(nblocks) for k in vals[0]}


def dg_al_ar(pops, kt):
    """Free energy of aL relative to aR, kJ/mol."""
    if pops["aL"] <= 0 or pops["aR"] <= 0:
        return float("inf")
    return -kt * np.log(pops["aL"] / pops["aR"])


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
    ap.add_argument("--pseudo", type=float, default=20.0, help="pseudo-count, frames per bin")
    ap.add_argument("--sigma", type=float, default=5.0, help="smoothing, degrees")
    ap.add_argument("--umax", type=float, default=30.0, help="clamp, kJ/mol above the map minimum")
    ap.add_argument("--iterations", type=int, default=60, help="reweighting refinement steps")
    ap.add_argument(
        "--sampled-maps",
        default=None,
        help="directory with the maps active in the --sim runs (second and later iterations); "
        "omit for runs without any phi/psi potential",
    )
    ap.add_argument("--no-cross-validation", action="store_true", help="skip the half/half validation")
    ap.add_argument(
        "--target-sigma",
        type=float,
        default=0.0,
        help="smooth the target histograms (degrees) before fitting. 0 reproduces the sharp targets but "
        "the maps then need a 2 fs time step; about 15 makes them integrable at 5 fs (see README)",
    )
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
            print(f"{res}: missing simulation or target data, skipped")
            continue
        print(f"{res}:")
        phi, psi = read_rama(sim_path)
        target = read_target(tgt_path)
        sampled = None
        if args.sampled_maps:
            meta, sampled = read_cmap(os.path.join(args.sampled_maps, f"{res}.cmap"))
            if sampled.shape[0] != args.grid:
                raise ValueError(f"{res}: sampled map grid {sampled.shape[0]} differs from --grid {args.grid}")
        kw = dict(
            n=args.grid,
            pseudo=args.pseudo,
            sigma_deg=args.sigma,
            umax=args.umax,
            iterations=args.iterations,
            target_sigma_deg=args.target_sigma,
        )
        grid, info = fit_map(phi, psi, target, temperature=args.temperature, sampled_map=sampled, **kw)
        info["max_curvature"] = max_curvature(grid)
        info["cv"] = (
            None
            if args.no_cross_validation
            else cross_validate(phi, psi, target, args.temperature, sampled_map=sampled, **kw)
        )
        w = info["weights"]
        t = target_basins(target)
        s = info["unbiased_basins"]
        r = basin_populations(phi, psi, w)
        se = block_errors(phi, psi, w)
        write_cmap(
            os.path.join(args.out, f"{res}.cmap"),
            res,
            grid,
            {
                "method": "Boltzmann inversion of an unbiased run without phi/psi potential + reweighting refinement",
                "target": "Choi & Pappu, JCTC 15, 1355 (2019), " + os.path.basename(tgt_path),
                "simulation": f"{os.path.basename(sim_path)}, {info['frames']} frames"
                + (f", sampled with maps from {args.sampled_maps}" if args.sampled_maps else ", no phi/psi potential"),
                "temperature": f"{args.temperature} K",
                "parameters": f"pseudo {args.pseudo}, sigma {args.sigma} deg, umax {args.umax} kJ/mol, "
                f"target sigma {args.target_sigma} deg, {info['iterations']} refinement steps",
                "max curvature": f"{info['max_curvature']:.0f} kJ/mol/rad^2",
                "fit": f"total variation vs target {info['total_variation_initial']:.4f} unbiased -> "
                f"{info['total_variation_final']:.4f} reweighted; effective frames {info['effective_frames']:.0f}",
                "created": stamp,
                **({"note": args.note} if args.note else {}),
            },
        )
        rows.append((res, info, t, s, r, se))

    report = args.report or os.path.join(args.out, "README.md")
    write_report(report, rows, args, kt)
    print(f"maps in {args.out}, report {report}")


def write_report(path, rows, args, kt):
    keys = ["aR", "PII", "beta", "aL"]
    out = [
        "# Backbone CMAPs",
        "",
        f"Built {datetime.date.today().isoformat()} with `tools/cmap_builder/build_cmap.py` at {args.temperature} K, "
        f"grid {args.grid} ({360 / args.grid:g} deg), pseudo-count {args.pseudo}, smoothing {args.sigma} deg, "
        f"clamp {args.umax} kJ/mol, target smoothing {args.target_sigma} deg, {args.iterations} refinement steps"
        + (
            f", runs sampled with the maps in `{args.sampled_maps}`."
            if args.sampled_maps
            else ", runs without phi/psi potential."
        ),
        "",
        "Populations in % : target / sampled run / **reweighted with the new map** ± block SE. "
        "Basins: aR phi<0, -120<psi<50; PII -100<=phi<0 and beta phi<-100 (psi outside the aR band); "
        "aL phi>=0, -90<psi<90. dG(aL-aR) in kJ/mol.",
        "",
        "Held-out check: a map fitted on one half of the frames, applied to the other half, both ways. "
        "Reweighting the frames a map was fitted to is optimistic; the held-out errors estimate what a new "
        "simulation with the map will show. Verdict: **ok** if the held-out basin error is <= 2 points and the "
        "held-out dG(aL-aR) error <= 1 kJ/mol (aL only where the target has >= 0.1 %); otherwise "
        "**iterate**: rerun the dipeptide with the map and rebuild with `--sampled-maps`.",
        "",
        "Max curvature: largest second derivative of the map (kJ/mol/rad^2). On Ace-Ala-NMe at 5 fs, maps "
        "above ~1500 heated or blew up; ~540 ran at the right temperature. With sharp maps use a 2 fs step.",
        "",
        "| res | "
        + " | ".join(keys)
        + " | dG(aL-aR) tgt / map | held-out basin err (pt) | held-out dG err | eff. frames | max curvature | verdict |",
        "|---" * (len(keys) + 7) + "|",
    ]
    for res, info, t, s, r, se in rows:
        cells = [f"{100 * t[k]:.1f} / {100 * s[k]:.1f} / **{100 * r[k]:.1f}** ± {100 * se[k]:.1f}" for k in keys]
        dt, dr = dg_al_ar(t, kt), dg_al_ar(r, kt)
        cv = info.get("cv")
        if cv:
            b_err = max(cv["basin"])
            dg_err = max(cv["dg"], key=abs)
            ok = b_err <= 0.02 and (t["aL"] < 1e-3 or abs(dg_err) <= 1.0)
            cv_cells = f"{100 * b_err:.1f} | {dg_err:+.2f}" if t["aL"] >= 1e-3 else f"{100 * b_err:.1f} | n/a"
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
