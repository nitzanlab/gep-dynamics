#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from enum import Enum

from seaborn import cubehelix_palette

PROJECT_HOME_PATH = {
    'linux': '/cs/labs/mornitzan/yotamcon/gep-dynamics',
    'darwin': '/Users/yotamcon/projects/gep-dynamics'}

EXCEL_FINGERPRINT_TEMPLATE = os.path.join(os.path.dirname(__file__), 'static', 'fingerprint_template.xlsx')

## Constants
# Data parameters
NUMBER_HVG = 2000

# NMF parameters
NMF_TOLERANCE = 1e-6  # ratio of change in objective function from start to end

# Plotting parameters
NON_NEG_CMAP = cubehelix_palette(start=.5, rot=-.3, light=1, as_cmap=True)

# Truncated spearman correlation cutoff
N_COMPARED_RANKED = 1000 # number of top ranked genes to compare

## Classes
class NMFEngine(str, Enum):
    """
    Which engine to use when decomposing data using NMF
    """
    sklearn = "sklearn"
    torchnmf = "torchnmf"
    consensus = "consensus"
    consensus_torch = "consensus_torch"

class Stage(str, Enum):
    INITIALIZED = "Initialized"
    PREPARED = "Prepared"
    DECOMPOSED = "Decomposed"


