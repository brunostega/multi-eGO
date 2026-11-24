"""
OpenMM parameter file (``mego_params.h5``) written by ``multiego.py`` and
``openmm/run.py``.

The parameter file must describe exactly the model of the GROMACS files of the
same run: both are generated and the two OpenMM Systems (one from each) must
give the same energies and forces term by term.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
h5py = pytest.importorskip("h5py")

from multiego.openmm_backend import FORCE_GROUPS, load, load_params  # noqa: E402
from multiego.openmm_backend import paramfile  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS = os.path.join(ROOT, "tests", "test_inputs")
REFERENCE = openmm.Platform.getPlatformByName("Reference")


def generate(tmp, system, *extra):
    cmd = [sys.executable, os.path.join(ROOT, "multiego.py"), "--system", system, "--egos", "mg", "--no_header"]
    cmd += ["--explicit_name", "t", "--inputs_dir", INPUTS, "--outputs_dir", str(tmp), "--backend", "both", *extra]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    return os.path.join(str(tmp), system, "t_1")


def boxed_gro(src, dst, box=6.0):
    """Copy a .gro centred in a cubic box (the test inputs have tight boxes)."""
    lines = open(src).read().splitlines()
    n = int(lines[1])
    atom_lines = lines[2:][:n]
    xyz = np.array([[float(line[20:28]), float(line[28:36]), float(line[36:44])] for line in atom_lines])
    xyz += box / 2 - xyz.mean(axis=0)
    with open(dst, "w") as f:
        f.write(f"boxed\n{n}\n")
        for line, (x, y, z) in zip(atom_lines, xyz):
            f.write(f"{line[:20]}{x:8.3f}{y:8.3f}{z:8.3f}\n")
        f.write(f"{box:10.5f}{box:10.5f}{box:10.5f}\n")
    return dst


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gen")
    return {
        "ttr": (generate(tmp, "ttrref"), os.path.join(INPUTS, "ttrref", "ttr_mego.gro"), None),
        "aladip_cmap": (
            generate(tmp, "aladip", "--backbone", "cmap"),
            boxed_gro(os.path.join(INPUTS, "aladip", "aladip_mego.gro"), str(tmp / "aladip.gro")),
            os.path.join(ROOT, "multi-ego-basic.ff", "cmap"),
        ),
    }


def _energies_forces(system, positions):
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), REFERENCE)
    ctx.setPositions(positions)
    e = {k: ctx.getState(getEnergy=True, groups={g}).getPotentialEnergy()._value for k, g in FORCE_GROUPS.items()}
    return e, ctx.getState(getForces=True).getForces(asNumpy=True)._value


@pytest.mark.parametrize("name", ["ttr", "aladip_cmap"])
def test_paramfile_equals_gromacs_files(models, name):
    out, gro, cmap_dir = models[name]
    s_gmx, _, pos = load(os.path.join(out, "topol_mego.top"), gro, cmap_dir=cmap_dir)
    s_h5, _, _ = load_params(os.path.join(out, paramfile.DEFAULT_FILENAME), gro)
    assert s_gmx.getNumParticles() == s_h5.getNumParticles()
    e_gmx, f_gmx = _energies_forces(s_gmx, pos)
    e_h5, f_h5 = _energies_forces(s_h5, pos)
    for term in e_gmx:
        # GROMACS files print 6 significant digits; the parameter file keeps full precision
        assert e_h5[term] == pytest.approx(e_gmx[term], rel=1e-5, abs=1e-4), term
    assert np.abs(f_h5 - f_gmx).max() < 1e-5 * np.abs(f_gmx).max()
    if cmap_dir:
        assert e_h5["CMAP"] != 0.0


def test_cmap_stored_only_with_backbone_cmap(models):
    p = paramfile.read(os.path.join(models["ttr"][0], paramfile.DEFAULT_FILENAME))
    assert p.cmap is None and p.attrs["backbone"] == "gromos"
    p = paramfile.read(os.path.join(models["aladip_cmap"][0], paramfile.DEFAULT_FILENAME))
    assert p.attrs["backbone"] == "cmap"
    assert p.cmap.map_order() == ["ALA"]  # only the maps the system uses
    assert len(p.cmap.residues["index"]) == 1 and not p.cmap.residues["mirrored"][0]


def test_roundtrip(models, tmp_path):
    src = os.path.join(models["aladip_cmap"][0], paramfile.DEFAULT_FILENAME)
    p = paramfile.read(src)
    paramfile.write(str(tmp_path / "copy.h5"), p)
    q = paramfile.read(str(tmp_path / "copy.h5"))
    assert list(q.molecules) == list(p.molecules)
    for name in p.molecules:
        for term in paramfile.BONDED_COLUMNS:
            for col in paramfile.BONDED_COLUMNS[term]:
                np.testing.assert_array_equal(
                    q.molecules[name].bonded[term][col[0]], p.molecules[name].bonded[term][col[0]]
                )
    np.testing.assert_array_equal(q.nonbonded.dense_tables()[0], p.nonbonded.dense_tables()[0])
    np.testing.assert_array_equal(q.cmap.maps["ALA"], p.cmap.maps["ALA"])


def test_old_format_rejected(models, tmp_path):
    path = str(tmp_path / "old.h5")
    with h5py.File(os.path.join(models["ttr"][0], paramfile.DEFAULT_FILENAME), "r") as f, h5py.File(path, "w") as g:
        for k in f:
            f.copy(k, g)
        for k, v in f.attrs.items():
            g.attrs[k] = v
        g.attrs["format_version"] = 3
    with pytest.raises(ValueError, match="format version"):
        paramfile.read(path)


def test_composition(models):
    out, gro, _ = models["ttr"]
    path = os.path.join(out, paramfile.DEFAULT_FILENAME)
    with pytest.raises(ValueError, match="unknown molecule"):
        load_params(path, gro, molecules=[("XYZ", 1)])
    with pytest.raises(ValueError, match="composition"):
        load_params(path, gro, molecules=[("TTR", 2)])


def test_generator_side_needs_no_openmm():
    """multiego.py writes the parameter file without OpenMM installed."""
    code = "import sys; sys.modules['openmm'] = None; import multiego.openmm_backend.paramfile, multiego.fileio"
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    # run outside the repository: there, the top-level multiego.py would shadow the package
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=os.path.dirname(ROOT)
    )
    assert res.returncode == 0, res.stderr


# ----------------------------------------------------------------- openmm/run.py
def test_run_py_and_restart(models, tmp_path):
    out, gro, _ = models["aladip_cmap"]
    run = os.path.join(ROOT, "openmm", "run.py")
    base = [sys.executable, run, "--params", out, "--conf", gro, "--temperature", "300", "--platform", "Reference"]
    base += ["--log-interval", "100", "--xtc-interval", "100", "--checkpoint-interval", "100", "--seed", "3"]
    prefix = str(tmp_path / "out" / "ala")
    res = subprocess.run(base + ["--nsteps", "300", "--minimize", "--output", prefix], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    assert "CMAP on 1 residues" in res.stdout
    for ext in (".xtc", ".log", ".chk", "_energy.csv", "_final.gro", "_final.pdb"):
        assert os.path.getsize(prefix + ext) > 0, ext
    header = open(prefix + "_energy.csv").readline().strip().split(",")
    assert header[:2] == ["step", "time_ps"] and "CMAP" in header and header[-1] == "potential"
    rows = np.loadtxt(prefix + "_energy.csv", delimiter=",", skiprows=1)
    assert rows.shape == (3, len(header))
    np.testing.assert_allclose(rows[:, 2:-1].sum(axis=1), rows[:, -1], rtol=1e-8, atol=1e-6)
    res = subprocess.run(
        base + ["--nsteps", "200", "--restart", prefix + ".chk", "--output", prefix], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr[-2000:]
    steps = np.loadtxt(prefix + "_energy.csv", delimiter=",", skiprows=1)[:, 0]
    np.testing.assert_array_equal(steps, [100, 200, 300, 400, 500])
