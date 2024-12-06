import os
import re
import sys
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..",))

from multiego.resources import type_definitions
from multiego.util import masking
from multiego import io

import argparse
import itertools
import multiprocessing
import numpy as np
import pandas as pd
import parmed as pmd
import time
import warnings
import gzip
import tarfile