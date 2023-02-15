#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from seaborn import cubehelix_palette


NUMBER_HVG = 2000

# cNMF parameters
NMF_TOLERANCE = 1e-6

# Plotting parameters
NON_NEG_CMAP = cubehelix_palette(start=.5, rot=-.3, light=1, as_cmap=True)
