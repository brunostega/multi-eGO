"""
Root-level launcher for OpenMM simulations of multi-eGO models, the
counterpart of ``multiego.py`` for the simulation step::

    python mego_simulate.py --params outputs/SYSTEM/mg_1/mego_params.h5 \
        --conf conf.gro --temperature 300 --nsteps 1000000 --minimize --output run

See ``src/multiego/omm/simulate.py`` (or ``--help``) for all options.  After
``pip install -e .`` the same program is available as ``mego-simulate``.
"""

import os
import sys

# Same trick as multiego.py: make src/multiego/ win over this directory.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from multiego.omm.simulate import main  # noqa: E402

if __name__ == "__main__":
    main()
