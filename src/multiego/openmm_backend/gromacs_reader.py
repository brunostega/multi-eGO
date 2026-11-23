"""
Strict reader for the GROMACS topology written by multi-eGO.

Reads ``topol_mego.top`` (and the files it includes, normally
``ffnonbonded.itp``) into plain Python objects. Only the subset of the
GROMACS format that multi-eGO writes is accepted; anything else raises a
``TopologyError`` naming the file and line, so that a change in the generator
cannot be silently mistranslated.

Supported
---------
- ``[ defaults ]``: nbfunc 1, comb-rule 1 (c6/c12), gen-pairs no
- ``[ atomtypes ]``: name, atomic number, mass, charge, ptype A, c6, c12
- ``[ nonbond_params ]``: funct 1 (c6, c12)
- ``[ moleculetype ]``, ``[ atoms ]`` (charge and mass optional)
- ``[ bonds ]`` funct 1, ``[ angles ]`` funct 1
- ``[ dihedrals ]`` funct 1, 4, 9 (periodic) and funct 2 (harmonic improper)
- ``[ pairs ]`` funct 1 (c6, c12), ``[ exclusions ]``
- ``[ system ]``, ``[ molecules ]``
- ``#include``, ``#ifdef``/``#ifndef``/``#else``/``#endif``. ``#define`` is
  rejected because multi-eGO never writes macros.

Units are the GROMACS ones (nm, degrees, kJ/mol); the builder converts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


class TopologyError(ValueError):
    """Raised for any construct outside the multi-eGO dialect."""


@dataclass
class AtomType:
    name: str
    atomic_number: int
    mass: float
    charge: float
    c6: float
    c12: float


@dataclass
class Atom:
    index: int  # 0-based within the molecule
    type: str
    resnum: int
    resname: str
    name: str
    charge: float
    mass: float


@dataclass
class MoleculeType:
    name: str
    nrexcl: int
    atoms: list = field(default_factory=list)  # list[Atom]
    # 0-based atom indices; parameters in GROMACS units
    bonds: list = field(default_factory=list)  # (i, j, b0 [nm], kb [kJ/mol/nm^2])
    angles: list = field(default_factory=list)  # (i, j, k, theta0 [deg], k [kJ/mol/rad^2])
    propers: list = field(default_factory=list)  # (i, j, k, l, phi0 [deg], k [kJ/mol], mult)
    impropers: list = field(default_factory=list)  # (i, j, k, l, xi0 [deg], k [kJ/mol/rad^2])
    pairs: list = field(default_factory=list)  # (i, j, c6, c12)
    exclusions: list = field(default_factory=list)  # (i, j), i < j


@dataclass
class GromacsTopology:
    atomtypes: dict  # name -> AtomType, in file order
    nonbond_params: dict  # (type_a, type_b) sorted tuple -> (c6, c12)
    moleculetypes: dict  # name -> MoleculeType
    molecules: list  # [(moleculetype name, count)] in file order
    system_name: str = ""
    fudge_lj: float = 1.0

    @property
    def n_atoms(self):
        return sum(len(self.moleculetypes[name].atoms) * count for name, count in self.molecules)


_PERIODIC_DIHEDRALS = {1, 4, 9}


def read_topology(top_path, defines=()):
    """
    Read a multi-eGO GROMACS topology.

    Parameters
    ----------
    top_path : str
        Path to ``topol_mego.top``. ``#include`` paths are resolved relative
        to the including file.
    defines : iterable of str
        Preprocessor symbols considered defined (e.g. ``("POSRES",)``).
        Position restraints are not supported, so in practice leave it empty.

    Returns
    -------
    GromacsTopology
    """
    reader = _Reader(set(defines))
    reader.read_file(top_path)
    return reader.finish()


class _Reader:
    def __init__(self, defines):
        self.defines = defines
        self.section = None
        self.atomtypes = {}
        self.nonbond_params = {}
        self.moleculetypes = {}
        self.molecules = []
        self.system_name = ""
        self.fudge_lj = 1.0
        self.defaults_seen = False
        self.current = None  # current MoleculeType
        self.where = ""

    # ------------------------------------------------------------------ files
    def read_file(self, path):
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise TopologyError(f"{self.where}: included file not found: {path}")
        # Stack of (this block active, parent active, else seen)
        stack = []
        with open(path) as fh:
            for lineno, raw in enumerate(fh, start=1):
                self.where = f"{path}:{lineno}"
                line = raw.split(";", 1)[0].strip()
                if not line:
                    continue
                active = all(s[0] for s in stack)
                if line.startswith("#"):
                    self._directive(line, stack, active, path)
                    continue
                if not active:
                    continue
                if line.startswith("["):
                    self._start_section(line)
                else:
                    self._data_line(line.split())
        if stack:
            raise TopologyError(f"{path}: unterminated #ifdef/#ifndef")

    def _directive(self, line, stack, active, path):
        parts = line.split()
        key = parts[0]
        if key in ("#ifdef", "#ifndef"):
            if len(parts) != 2:
                raise TopologyError(f"{self.where}: malformed {key}")
            defined = parts[1] in self.defines
            stack.append([defined if key == "#ifdef" else not defined, False])
        elif key == "#else":
            if not stack or stack[-1][1]:
                raise TopologyError(f"{self.where}: unexpected #else")
            stack[-1][0] = not stack[-1][0]
            stack[-1][1] = True
        elif key == "#endif":
            if not stack:
                raise TopologyError(f"{self.where}: unexpected #endif")
            stack.pop()
        elif key == "#include":
            if not active:
                return
            if len(parts) != 2:
                raise TopologyError(f"{self.where}: malformed #include")
            name = parts[1].strip('"<>')
            self.read_file(os.path.join(os.path.dirname(path), name))
        elif key == "#define":
            if active:
                raise TopologyError(f"{self.where}: #define is not supported in multi-eGO topologies")
        else:
            raise TopologyError(f"{self.where}: unsupported preprocessor directive {key}")

    # --------------------------------------------------------------- sections
    _MOLECULE_SECTIONS = {"atoms", "bonds", "angles", "dihedrals", "pairs", "exclusions"}
    _GLOBAL_SECTIONS = {"defaults", "atomtypes", "nonbond_params", "moleculetype", "system", "molecules"}

    def _start_section(self, line):
        if not line.endswith("]"):
            raise TopologyError(f"{self.where}: malformed section header {line!r}")
        name = line[1:-1].strip().lower()
        if name not in self._MOLECULE_SECTIONS | self._GLOBAL_SECTIONS:
            raise TopologyError(f"{self.where}: unsupported section [ {name} ]")
        if name in self._MOLECULE_SECTIONS and self.current is None:
            raise TopologyError(f"{self.where}: [ {name} ] outside a [ moleculetype ]")
        if name in ("system", "molecules"):
            self.current = None
        self.section = name

    def _data_line(self, f):
        if self.section is None:
            raise TopologyError(f"{self.where}: data outside any section")
        try:
            getattr(self, "_s_" + self.section)(f)
        except TopologyError:
            raise
        except (ValueError, IndexError) as e:
            raise TopologyError(f"{self.where}: cannot parse [ {self.section} ] line {' '.join(f)!r}: {e}") from None

    def _funct(self, f, pos, allowed):
        funct = int(f[pos])
        if funct not in allowed:
            raise TopologyError(f"{self.where}: [ {self.section} ] function type {funct} is not supported")
        return funct

    def _atom(self, token):
        """1-based atom number in the current molecule -> 0-based index."""
        i = int(token) - 1
        if not 0 <= i < len(self.current.atoms):
            raise TopologyError(f"{self.where}: atom {token} not in molecule {self.current.name}")
        return i

    def _s_defaults(self, f):
        nbfunc, comb = int(f[0]), int(f[1])
        gen_pairs = f[2].lower() if len(f) > 2 else "no"
        if nbfunc != 1 or comb != 1:
            raise TopologyError(f"{self.where}: only nbfunc 1 with comb-rule 1 (c6/c12) is supported")
        if gen_pairs != "no":
            raise TopologyError(f"{self.where}: gen-pairs must be 'no'")
        # fudgeLJ only scales generated pairs, which are disabled; stored for reference.
        self.fudge_lj = float(f[3]) if len(f) > 3 else 1.0
        self.defaults_seen = True

    def _s_atomtypes(self, f):
        # name at.num mass charge ptype c6 c12
        if len(f) != 7:
            raise TopologyError(f"{self.where}: expected 7 fields in [ atomtypes ], got {len(f)}")
        if f[4] != "A":
            raise TopologyError(f"{self.where}: only ptype A is supported")
        if f[0] in self.atomtypes:
            raise TopologyError(f"{self.where}: duplicate atom type {f[0]}")
        self.atomtypes[f[0]] = AtomType(f[0], int(f[1]), float(f[2]), float(f[3]), float(f[5]), float(f[6]))

    def _s_nonbond_params(self, f):
        self._funct(f, 2, {1})
        a, b = f[0], f[1]
        for t in (a, b):
            if t not in self.atomtypes:
                raise TopologyError(f"{self.where}: unknown atom type {t} in [ nonbond_params ]")
        key = (a, b) if a <= b else (b, a)
        if key in self.nonbond_params:
            raise TopologyError(f"{self.where}: duplicate [ nonbond_params ] entry {a} {b}")
        self.nonbond_params[key] = (float(f[3]), float(f[4]))

    def _s_moleculetype(self, f):
        name = f[0]
        if name in self.moleculetypes:
            raise TopologyError(f"{self.where}: duplicate moleculetype {name}")
        self.current = MoleculeType(name=name, nrexcl=int(f[1]))
        self.moleculetypes[name] = self.current

    def _s_atoms(self, f):
        # nr type resnr residue atom cgnr [charge [mass]]
        nr = int(f[0])
        if nr != len(self.current.atoms) + 1:
            raise TopologyError(f"{self.where}: atoms must be numbered consecutively from 1")
        atype = f[1]
        if atype not in self.atomtypes:
            raise TopologyError(f"{self.where}: unknown atom type {atype}")
        at = self.atomtypes[atype]
        charge = float(f[6]) if len(f) > 6 else at.charge
        mass = float(f[7]) if len(f) > 7 else at.mass
        if len(f) > 8:
            raise TopologyError(f"{self.where}: B-state atom parameters are not supported")
        self.current.atoms.append(Atom(nr - 1, atype, int(f[2]), f[3], f[4], charge, mass))

    def _s_bonds(self, f):
        self._funct(f, 2, {1})
        if len(f) != 5:
            raise TopologyError(f"{self.where}: bonds need explicit b0 and kb")
        self.current.bonds.append((self._atom(f[0]), self._atom(f[1]), float(f[3]), float(f[4])))

    def _s_angles(self, f):
        self._funct(f, 3, {1})
        if len(f) != 6:
            raise TopologyError(f"{self.where}: angles need explicit theta0 and k")
        i, j, k = (self._atom(x) for x in f[:3])
        self.current.angles.append((i, j, k, float(f[4]), float(f[5])))

    def _s_dihedrals(self, f):
        funct = self._funct(f, 4, _PERIODIC_DIHEDRALS | {2})
        i, j, k, m = (self._atom(x) for x in f[:4])
        if funct == 2:
            if len(f) != 7:
                raise TopologyError(f"{self.where}: impropers need explicit xi0 and k")
            self.current.impropers.append((i, j, k, m, float(f[5]), float(f[6])))
        else:
            if len(f) != 8:
                raise TopologyError(f"{self.where}: dihedrals need explicit phi0, k and multiplicity")
            self.current.propers.append((i, j, k, m, float(f[5]), float(f[6]), int(f[7])))

    def _s_pairs(self, f):
        self._funct(f, 2, {1})
        if len(f) != 5:
            raise TopologyError(f"{self.where}: pairs need explicit c6 and c12")
        self.current.pairs.append((self._atom(f[0]), self._atom(f[1]), float(f[3]), float(f[4])))

    def _s_exclusions(self, f):
        i = self._atom(f[0])
        for tok in f[1:]:
            j = self._atom(tok)
            if i != j:
                self.current.exclusions.append((min(i, j), max(i, j)))

    def _s_system(self, f):
        self.system_name = " ".join(f) if not self.system_name else self.system_name + " " + " ".join(f)

    def _s_molecules(self, f):
        name, count = f[0], int(f[1])
        if name not in self.moleculetypes:
            raise TopologyError(f"{self.where}: unknown molecule {name} in [ molecules ]")
        if count < 0:
            raise TopologyError(f"{self.where}: negative molecule count")
        self.molecules.append((name, count))

    # ----------------------------------------------------------------- finish
    def finish(self):
        if not self.defaults_seen:
            raise TopologyError("no [ defaults ] section found")
        if not self.molecules:
            raise TopologyError("no [ molecules ] section found")
        return GromacsTopology(
            atomtypes=self.atomtypes,
            nonbond_params=self.nonbond_params,
            moleculetypes=self.moleculetypes,
            molecules=self.molecules,
            system_name=self.system_name,
            fudge_lj=self.fudge_lj,
        )
