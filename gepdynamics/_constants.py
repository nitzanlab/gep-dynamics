#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from seaborn import cubehelix_palette

PROJECT_HOME_PATH = {
    'linux': '/cs/labs/mornitzan/yotamcon/gep-dynamics',
    'darwin': '/Users/yotamcon/projects/gep-dynamics'}

NUMBER_HVG = 2000

# cNMF parameters
NMF_TOLERANCE = 1e-6

# Plotting parameters
NON_NEG_CMAP = cubehelix_palette(start=.5, rot=-.3, light=1, as_cmap=True)

# Truncated spearman correlation cuttoff 
N_COMPARED_RANKED = 1000 