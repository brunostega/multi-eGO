"""
Canonical, implementation-independent view of the parameters of a System.

Why this module exists
----------------------
:mod:`multiego.omm.energy` compares two Systems through energies and forces.
That catches structural errors (a missing term, a wrong functional form, a
wrong cut-off), but it is *not* sensitive to a small error in a single
parameter: e.g. one wrong LJ pair out of ~120,000 changes the total
nonbonded energy far below any sensible tolerance.

This module extracts every parameter of a System into a canonical form, keyed
by the atoms involved, so two Systems can be compared parameter by parameter:

=====================  ===============================  ========================
term                   key                              value(s)
=====================  ===============================  ========================
bonds                  (i, j), i < j                    (r0 [nm], k)
angles                 (i, j, k), i < k                 (theta0 [rad], k)
proper_dihedrals       (i, j, k, l), i < l              sorted [(n, phase, k)]
improper_dihedrals     (i, j, k, l), i < l              (theta0 [rad], k)
pairs                  (i, j), i < j                    (c6, c12)
nonbonded              dense N x N matrices             c6, c12, exclusions
=====================  ===============================  ========================

A dihedral and its reverse (l, k, j, i) define the same angle, so reversing
the atom order is a valid canonicalisation.  Terms with a zero force constant
have no effect and are dropped, so implementations may differ in whether they
write them.

Supported parameterisations
---------------------------
* ``openmm.app.GromacsTopFile``: LJ table functions ``acoef = sqrt(c12)`` and
  ``bcoef = c6``; pair bonds with per-bond parameters ``C = c6`` and
  ``A = c12``; impropers as ``CustomTorsionForce`` with ``theta0, k``.
* The native multi-eGO builder: LJ table functions ``c6tab``/``c12tab``; pair
  bonds with per-bond parameters ``c6``/``c12``; impropers as above.
"""

import numpy as np
import openmm as mm

from multiego.omm.energy import classify_force


def _canonical(key):
    """Reverse an atom tuple if needed so that the first index is the smaller one."""
    return key if key[0] < key[-1] else tuple(reversed(key))


def _value(quantity):
    """Strip OpenMM units (values are then in OpenMM default units: nm, rad, kJ/mol)."""
    return quantity._value if hasattr(quantity, "_value") else quantity


def _per_bond_parameter_names(force):
    """Names of the per-bond parameters of a CustomBondForce, in order."""
    return [force.getPerBondParameterName(i) for i in range(force.getNumPerBondParameters())]


# ---------------------------------------------------------------------------
# Extraction of each kind of force
# ---------------------------------------------------------------------------


def _bonds(force, out):
    for b in range(force.getNumBonds()):
        i, j, r0, k = force.getBondParameters(b)
        if _value(k) != 0.0:
            out[_canonical((i, j))] = (_value(r0), _value(k))


def _angles(force, out):
    for a in range(force.getNumAngles()):
        i, j, k, theta0, kk = force.getAngleParameters(a)
        if _value(kk) != 0.0:
            out[_canonical((i, j, k))] = (_value(theta0), _value(kk))


def _proper_dihedrals(force, out):
    for t in range(force.getNumTorsions()):
        i, j, k, l, n, phase, kk = force.getTorsionParameters(t)
        if _value(kk) != 0.0:
            out.setdefault(_canonical((i, j, k, l)), []).append((n, _value(phase), _value(kk)))
    for key in out:
        out[key] = sorted(out[key])


def _improper_dihedrals(force, out):
    names = [force.getPerTorsionParameterName(p) for p in range(force.getNumPerTorsionParameters())]
    for t in range(force.getNumTorsions()):
        i, j, k, l, params = force.getTorsionParameters(t)
        p = dict(zip(names, params))
        if p["k"] != 0.0:
            out[_canonical((i, j, k, l))] = (p["theta0"], p["k"])


def _pairs(force, out):
    names = _per_bond_parameter_names(force)
    # GromacsTopFile uses C (=c6) and A (=c12); the native builder c6/c12.
    c6_name, c12_name = ("C", "A") if "C" in names else ("c6", "c12")
    for b in range(force.getNumBonds()):
        i, j, params = force.getBondParameters(b)
        p = dict(zip(names, params))
        if p[c6_name] != 0.0 or p[c12_name] != 0.0:
            out[_canonical((i, j))] = (p[c6_name], p[c12_name])


def _nonbonded(force):
    """
    Effective (c6, c12) for every atom pair, as dense N x N matrices, plus the
    set of excluded pairs.  Built from the per-particle type and the type
    tables, i.e. exactly what the force evaluates.
    """
    functions = {
        force.getTabulatedFunctionName(f): force.getTabulatedFunction(f)
        for f in range(force.getNumTabulatedFunctions())
    }

    def table(name):
        nx, ny, values = functions[name].getFunctionParameters()
        # Discrete2DFunction values are stored with x varying fastest.
        return np.array(values).reshape(ny, nx).T

    if "acoef" in functions:  # GromacsTopFile convention
        c6_table, c12_table = table("bcoef"), table("acoef") ** 2
    else:  # native builder convention
        c6_table, c12_table = table("c6tab"), table("c12tab")

    types = np.array([int(force.getParticleParameters(p)[0]) for p in range(force.getNumParticles())])
    exclusions = {_canonical(tuple(force.getExclusionParticles(e))) for e in range(force.getNumExclusions())}
    return {
        "c6": c6_table[np.ix_(types, types)],
        "c12": c12_table[np.ix_(types, types)],
        "exclusions": exclusions,
        "cutoff": _value(force.getCutoffDistance()),
        "long_range_correction": force.getUseLongRangeCorrection(),
    }


def extract(system):
    """
    Extract the canonical parameters of every multi-eGO term of a System.

    Returns
    -------
    dict
        ``{term: dict}`` for the bonded/pair terms (see the module table) and
        ``{"nonbonded": {...}}`` with the dense matrices.  Terms with no
        parameters (e.g. electrostatics) are omitted.
    """
    params = {}
    for force in system.getForces():
        term = classify_force(force)
        if term is None or term == "electrostatics":
            continue
        if term == "nonbonded":
            params["nonbonded"] = _nonbonded(force)
            continue
        out = params.setdefault(term, {})
        if isinstance(force, mm.HarmonicBondForce):
            _bonds(force, out)
        elif isinstance(force, mm.HarmonicAngleForce):
            _angles(force, out)
        elif isinstance(force, mm.PeriodicTorsionForce):
            _proper_dihedrals(force, out)
        elif isinstance(force, mm.CustomTorsionForce):
            _improper_dihedrals(force, out)
        elif isinstance(force, mm.CustomBondForce):
            _pairs(force, out)
        else:
            # e.g. the hydrogen-bond and CMAP terms: compared through
            # energies only.
            params.pop(term)
    return params


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _close(a, b, rtol):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=rtol, atol=0.0)


def compare(reference, candidate, rtol=1e-5, max_reported=5):
    """
    Compare two outputs of :func:`extract`.

    ``rtol`` absorbs the 6-significant-digit rounding of the GROMACS text
    files; everything else must match exactly (same keys, same exclusions,
    same cut-off).

    Returns
    -------
    list of str
        Descriptions of the mismatches (empty if the Systems are identical).
    """
    failures = []
    for term in sorted(set(reference) | set(candidate)):
        if term not in reference or term not in candidate:
            failures.append(f"{term}: present only in {'reference' if term in reference else 'candidate'}")
            continue
        ref, cand = reference[term], candidate[term]

        if term == "nonbonded":
            if ref["exclusions"] != cand["exclusions"]:
                failures.append(
                    f"nonbonded: exclusions differ ({len(ref['exclusions'] - cand['exclusions'])} missing, "
                    f"{len(cand['exclusions'] - ref['exclusions'])} extra)"
                )
            if ref["cutoff"] != cand["cutoff"] or ref["long_range_correction"] != cand["long_range_correction"]:
                failures.append("nonbonded: cut-off or long-range correction settings differ")
            # Excluded pairs are never evaluated: ignore them.
            mask = np.ones(ref["c6"].shape, bool)
            for i, j in ref["exclusions"]:
                mask[i, j] = mask[j, i] = False
            np.fill_diagonal(mask, False)
            for name in ("c6", "c12"):
                bad = mask & ~np.isclose(ref[name], cand[name], rtol=rtol, atol=0.0)
                for i, j in list(zip(*np.nonzero(np.triu(bad))))[:max_reported]:
                    failures.append(
                        f"nonbonded {name} ({i}, {j}): {cand[name][i, j]:.6e} != reference {ref[name][i, j]:.6e}"
                    )
            continue

        missing, extra = set(ref) - set(cand), set(cand) - set(ref)
        if missing or extra:
            failures.append(
                f"{term}: {len(missing)} missing and {len(extra)} extra interactions, e.g. {sorted(missing or extra)[:3]}"
            )
        wrong = [k for k in set(ref) & set(cand) if not _close(ref[k], cand[k], rtol)]
        for key in sorted(wrong)[:max_reported]:
            failures.append(f"{term} {key}: {cand[key]} != reference {ref[key]}")
        if len(wrong) > max_reported:
            failures.append(f"{term}: ... {len(wrong) - max_reported} more")
    return failures
