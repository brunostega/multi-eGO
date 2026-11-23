#!/usr/bin/env python3
"""
Regenerate the GROMACS reference energies and forces used by
``tests/test_openmm_backend.py``.

For every case in ``cases.py`` this runs ``gmx grompp`` and
``gmx mdrun -rerun`` with ``mdps/ff_aa.mdp`` on the stored frames and saves
per-term energies and total forces to ``<case>.npz`` next to this script.

Run it again whenever a model in ``tests/test_outputs`` changes (the test
reports a stale reference by comparing file hashes)::

    python tests/openmm_reference/make_gromacs_reference.py [--gmx gmx] [case ...]
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cases  # noqa: E402

TERMS = ["Bond", "Angle", "Proper Dih.", "Improper Dih.", "LJ-14", "LJ (SR)", "Potential"]
MDP_OVERRIDES = {
    "nsteps": "0",
    "ref_t": "300",
    "gen_temp": "300",
    "nstcalcenergy": "1",
    "nstenergy": "1",
    "nstfout": "1",
    "nstlog": "1",
}


def write_mdp(path):
    src = os.path.join(os.path.dirname(cases.TESTS), "mdps", "ff_aa.mdp")
    out = []
    with open(src) as fh:
        for line in fh:
            key = line.split("=", 1)[0].strip().lower().replace("-", "_")
            if "=" in line and key in MDP_OVERRIDES:
                line = f"{key:24s} = {MDP_OVERRIDES[key]}\n"
            out.append(line)
    with open(path, "w") as fh:
        fh.writelines(out)


def run(cmd, cwd, stdin=None):
    res = subprocess.run(cmd, cwd=cwd, input=stdin, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
    return res


def read_xvg(path):
    legends, rows = [], []
    with open(path) as fh:
        for line in fh:
            m = re.match(r'@ s\d+ legend "(.*)"', line)
            if m:
                legends.append(m.group(1))
            elif line.strip() and line[0] not in "@#":
                rows.append([float(x) for x in line.split()])
    return legends, np.array(rows)


def make(name, gmx):
    with tempfile.TemporaryDirectory() as wd:
        top, gro, frames, box = cases.prepare(name, wd)
        write_mdp(os.path.join(wd, "rerun.mdp"))
        run([gmx, "grompp", "-f", "rerun.mdp", "-c", "conf.gro", "-p", "topol_mego.top", "-o", "rerun.tpr"], wd)
        run([gmx, "mdrun", "-s", "rerun.tpr", "-rerun", "conf.gro", "-deffnm", "rerun", "-nt", "1"], wd)
        run(
            [gmx, "energy", "-f", "rerun.edr", "-o", "energy.xvg"],
            wd,
            stdin="\n".join(t.replace(" ", "-") for t in TERMS) + "\n\n",
        )
        legends, data = read_xvg(os.path.join(wd, "energy.xvg"))
        energies = {t: data[:, 1 + legends.index(t)] for t in TERMS}
        run([gmx, "traj", "-f", "rerun.trr", "-s", "rerun.tpr", "-of", "forces.xvg"], wd, stdin="0\n")
        _, fdata = read_xvg(os.path.join(wd, "forces.xvg"))
        forces = fdata[:, 1:].reshape(len(fdata), -1, 3)
        version = run([gmx, "--version"], wd).stdout
        version = next(
            (v.split(":", 1)[1].strip() for v in version.splitlines() if v.startswith("GROMACS version")), "?"
        )

    if forces.shape != frames.shape:
        sys.exit(f"{name}: forces {forces.shape} vs frames {frames.shape}")
    np.savez_compressed(
        cases.reference_path(name),
        frames=frames,
        box=box,
        forces=forces,
        terms=np.array(TERMS),
        energies=np.array([energies[t] for t in TERMS]),
        input_hash=cases.input_hash(name),
        gromacs_version=version,
    )
    print(f"{name}: {frames.shape[1]} atoms, {len(frames)} frames, GROMACS {version}")
    for t in TERMS:
        print(f"   {t:14s} {energies[t][0]:14.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*", default=list(cases.CASES))
    ap.add_argument("--gmx", default="gmx")
    args = ap.parse_args()
    for name in args.cases:
        make(name, args.gmx)


if __name__ == "__main__":
    main()
