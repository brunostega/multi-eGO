"""
Root-level launcher of multiego.omm.inspect: what a parameter file contains
(hydrogen bond? CMAP, on which residues?) and, given a structure, the energy
of every term::

    python mego_inspect.py outputs/NAME/mg_1/mego_params.h5 [conf.gro] [--molecules NAME:N]
"""

import os
import sys

# Same trick as multiego.py: make src/multiego/ win over this directory.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from multiego.omm.inspect import main  # noqa: E402

if __name__ == "__main__":
    main()
