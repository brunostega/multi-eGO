"""
Backbone CMAP maps: file format, residue matching and OpenMM conversion.

File format (``<RES>.cmap``, plain text)
---------------------------------------
Lines starting with ``#`` are header lines. Header lines of the form
``# key: value`` are metadata; ``residue`` and ``grid`` are required::

    # multi-eGO CMAP
    # residue: ALA
    # grid: 72
    # layout: rows = psi, columns = phi; node k at -180 + k*360/grid degrees
    # units: kJ/mol
    # ... provenance ...
    <grid rows of grid values>

Row ``i`` is psi = -180 + i*360/N, column ``j`` is phi = -180 + j*360/N, with
phi = C(i-1)-N-CA-C and psi = N-CA-C-N(i+1) in degrees. The same layout as the
Choi & Pappu target grids, so a map can be compared with a target directly.

OpenMM convention
-----------------
``CMAPTorsionForce.addMap(size, energy)`` takes ``energy[i + size*j]`` at
first torsion (phi) = ``i*2*pi/size`` and second torsion (psi) =
``j*2*pi/size``, i.e. origin at 0, not -180 degrees. :func:`to_openmm_map`
converts; the conversion is pinned by an asymmetric test.
"""

from __future__ import annotations

import os

import numpy as np

# residue name -> (map residue, mirrored). D-residues use the L map with
# U_D(phi, psi) = U_L(-phi, -psi).
_ALIASES = {"CYS2": "CYS", "HISD": "HIS", "HISE": "HIS", "HISH": "HIS", "HIS1": "HIS", "HISA": "HIS", "HISB": "HIS"}


def map_residue(resname):
    """
    Name of the map a residue uses, and whether the map is mirrored.

    ``DALA`` -> ("ALA", True); ``CYS2`` -> ("CYS", False);
    ``DCYS2`` -> ("CYS", True).
    """
    mirrored = False
    name = resname.upper()
    if len(name) == 4 and name.startswith("D") and name not in _ALIASES:
        name, mirrored = name[1:], True
    elif len(name) == 5 and name.startswith("D") and name[1:] in _ALIASES:
        name, mirrored = name[1:], True
    return _ALIASES.get(name, name), mirrored


def grid_nodes(n):
    """Node angles in degrees: -180, -180 + 360/n, ..."""
    return -180.0 + np.arange(n) * 360.0 / n


def read_cmap(path):
    """
    Read a map file.

    Returns
    -------
    meta : dict  header metadata (strings); always has "residue" and "grid"
    grid : (n, n) ndarray, rows = psi, columns = phi, kJ/mol
    """
    meta, rows = {}, []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                body = s[1:].strip()
                if ":" in body:
                    key, value = body.split(":", 1)
                    meta.setdefault(key.strip().lower(), value.strip())
                continue
            rows.append([float(x) for x in s.split()])
    for key in ("residue", "grid"):
        if key not in meta:
            raise ValueError(f"{path}: missing '# {key}:' header line")
    n = int(meta["grid"])
    grid = np.array(rows, dtype=float)
    if grid.shape != (n, n):
        raise ValueError(f"{path}: expected a {n}x{n} grid, found {grid.shape}")
    if n % 2:
        raise ValueError(f"{path}: grid size must be even")
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"{path}: non-finite values")
    return meta, grid


def write_cmap(path, residue, grid, meta=None):
    """Write a map file; ``meta`` adds ``# key: value`` provenance lines."""
    grid = np.asarray(grid, dtype=float)
    n = grid.shape[0]
    if grid.shape != (n, n) or n % 2:
        raise ValueError("grid must be square with an even size")
    lines = [
        "# multi-eGO CMAP",
        f"# residue: {residue}",
        f"# grid: {n}",
        "# layout: rows = psi, columns = phi; node k at -180 + k*360/grid degrees",
        "# units: kJ/mol",
    ]
    for key, value in (meta or {}).items():
        lines.append(f"# {key}: {value}")
    lines += [" ".join(f"{x:.6f}" for x in row) for row in grid]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def mirror(grid):
    """U(phi, psi) -> U(-phi, -psi) on the node grid (node -180 maps to itself)."""
    return np.roll(grid[::-1, ::-1], 1, axis=(0, 1))


def to_openmm_map(grid):
    """
    Flat energy list for ``CMAPTorsionForce.addMap``.

    Our row i (psi) / column j (phi) start at -180; OpenMM indices start at 0
    degrees, so both axes are rolled by n/2. Row-major flattening then gives
    ``energy[i_phi + n*j_psi]`` as OpenMM expects.
    """
    n = grid.shape[0]
    return np.roll(grid, -n // 2, axis=(0, 1)).ravel().tolist()


def load_maps(cmap_dir, names):
    """Read ``<name>.cmap`` for each map name; missing files are skipped."""
    maps = {}
    for name in sorted(names):
        path = os.path.join(cmap_dir, f"{name}.cmap")
        if os.path.exists(path):
            meta, grid = read_cmap(path)
            if meta["residue"].upper() != name:
                raise ValueError(f"{path}: header residue {meta['residue']} does not match the file name")
            maps[name] = grid
    return maps


def backbone_quintuplets(moltype):
    """
    Residues of a molecule type that have a full (phi, psi) pair.

    A residue qualifies if it has atoms N, CA and C, its N is bonded to an atom
    named C of another residue, and its C is bonded to an atom named N of
    another residue.

    Returns
    -------
    list of (resname, resnum, (c_prev, n, ca, c, n_next)) with 0-based indices
    """
    atoms = moltype.atoms
    neighbours = [[] for _ in atoms]
    for i, j, *_ in moltype.bonds:
        neighbours[i].append(j)
        neighbours[j].append(i)

    by_residue = {}
    order = []
    for a in atoms:
        key = (a.resnum, a.resname)
        if key not in by_residue:
            by_residue[key] = {}
            order.append(key)
        by_residue[key].setdefault(a.name, a.index)

    out = []
    for key in order:
        names = by_residue[key]
        if not all(x in names for x in ("N", "CA", "C")):
            continue
        n, ca, c = names["N"], names["CA"], names["C"]
        prev = [b for b in neighbours[n] if atoms[b].name == "C" and atoms[b].resnum != key[0]]
        nxt = [b for b in neighbours[c] if atoms[b].name == "N" and atoms[b].resnum != key[0]]
        if len(prev) == 1 and len(nxt) == 1:
            out.append((key[1], key[0], (prev[0], n, ca, c, nxt[0])))
    return out
