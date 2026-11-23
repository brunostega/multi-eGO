"""
Test systems for the OpenMM backend, shared by the tests and by
``make_gromacs_reference.py``.

Each case is a multi-eGO model already produced by the regression tests
(``tests/test_outputs``) plus coordinates from ``tests/test_inputs``. The
two-copy case is built on the fly: same molecule twice, second copy shifted
so that the copies are in contact, in a larger box.
"""

import hashlib
import os
import shutil

import numpy as np

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(TESTS, "test_outputs")
INP = os.path.join(TESTS, "test_inputs")
REF_DIR = os.path.dirname(os.path.abspath(__file__))

# name -> (model dir, gro file, copies)
CASES = {
    "gb1_mg": ("gpref/case_1", "gpref/gb1_mego.gro", 1),
    "abeta_production": ("abetaref/case_1", "abetaref/ab42_mego.gro", 1),
    "lyso_bnz": ("lyso-bnz_ref/case_1", "lyso-bnz_ref/lyso_bnz_mego.gro", 1),
    "ttr_two_copies": ("ttrref/case_4", "ttrref/ttr_mego.gro", 2),
}

N_FRAMES = 3  # input frame + 2 perturbed frames
NOISE = 0.004  # nm, Gaussian perturbation of the extra frames
TWO_COPY_BOX = 12.0  # nm
CONTACT_GAP = 0.45  # nm, gap along x between the two copies


def read_gro(path):
    """Positions (n, 3) and box diagonal (3,) in nm; rectangular boxes only."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    n = int(lines[1])
    atom_lines = lines[2:][:n]
    pos = np.array([[float(line[20:28]), float(line[28:36]), float(line[36:44])] for line in atom_lines])
    box = np.array([float(x) for x in lines[2 + n].split()[:3]])
    return pos, box, atom_lines


def write_gro(path, frames, box, atom_lines, title="multi-eGO OpenMM test"):
    """Write frames (f, n, 3) with the residue/atom columns of atom_lines."""
    with open(path, "w") as fh:
        for k, pos in enumerate(frames):
            fh.write(f"{title} frame {k}\n{len(pos)}\n")
            for line, (x, y, z) in zip(atom_lines, pos):
                fh.write(f"{line[:20]}{x:8.3f}{y:8.3f}{z:8.3f}\n")
            fh.write(f"{box[0]:10.5f}{box[1]:10.5f}{box[2]:10.5f}\n")


def prepare(name, workdir):
    """
    Write ``topol_mego.top``, ``ffnonbonded.itp`` and ``conf.gro`` (all frames)
    for a case into workdir.

    Returns
    -------
    top_path, gro_path, frames (f, n, 3) rounded to the .gro precision, box (3,)
    """
    model, gro, copies = CASES[name]
    src = os.path.join(OUT, model)
    os.makedirs(workdir, exist_ok=True)
    shutil.copy(os.path.join(src, "ffnonbonded.itp"), workdir)
    top_path = os.path.join(workdir, "topol_mego.top")
    with open(os.path.join(src, "topol_mego.top")) as fh:
        top_text = fh.read()
    if copies != 1:
        top_text = _set_copies(top_text, copies)
    with open(top_path, "w") as fh:
        fh.write(top_text)

    pos, box, atom_lines = read_gro(os.path.join(INP, gro))
    if copies == 2:
        shift = pos[:, 0].max() - pos[:, 0].min() + CONTACT_GAP
        pos = np.vstack([pos, pos + [shift, 0.0, 0.0]])
        pos += 0.5 * TWO_COPY_BOX - pos.mean(axis=0)
        box = np.full(3, TWO_COPY_BOX)
        atom_lines = atom_lines * 2
    elif copies != 1:
        raise ValueError("only 1 or 2 copies are implemented")

    rng = np.random.default_rng(20261123)
    frames = [pos] + [pos + rng.normal(0.0, NOISE, pos.shape) for _ in range(N_FRAMES - 1)]
    frames = np.round(np.array(frames), 3)
    gro_path = os.path.join(workdir, "conf.gro")
    write_gro(gro_path, frames, box, atom_lines)
    return top_path, gro_path, frames, box


def _set_copies(top_text, copies):
    lines = top_text.splitlines()
    k = next(i for i, line in enumerate(lines) if line.strip().lower().replace(" ", "") == "[molecules]")
    body = [line for line in lines[k:][1:] if line.strip() and not line.strip().startswith(";")]
    if len(body) != 1:
        raise ValueError("copies are only supported for single-molecule topologies")
    name = body[0].split()[0]
    return "\n".join(lines[: k + 1] + [f"{name}\t{copies}"]) + "\n"


def input_hash(name):
    """sha256 of the model and coordinate files a reference was computed from."""
    model, gro, _ = CASES[name]
    h = hashlib.sha256()
    for path in (
        os.path.join(OUT, model, "topol_mego.top"),
        os.path.join(OUT, model, "ffnonbonded.itp"),
        os.path.join(INP, gro),
    ):
        with open(path, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def reference_path(name):
    return os.path.join(REF_DIR, f"{name}.npz")
