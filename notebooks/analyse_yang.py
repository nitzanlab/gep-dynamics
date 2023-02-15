# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Downloading and pre-processing Yang et. al 2022 data

# %%
# %load_ext autoreload
# %autoreload 2

import sys
import os
import tarfile
from urllib.request import urlretrieve


import numpy as np
import pandas as pd
import scanpy as sc

from gepdynamics import _utils


print(os.getcwd())
os.chdir('/cs/labs/mornitzan/yotamcon/gep-dynamics')


# %%
data_dir = _utils.set_dir('data')
orig_adata_path = data_dir.joinpath('adata_processed.nt.h5ad')

if not orig_adata_path.exists():
    f_tar = data_dir.joinpath('KPTracer-Data.tar.gz')
    
    if not f_tar.exists():
        ftp_address = 'https://zenodo.org/record/5847462/files/KPTracer-Data.tar.gz?download=1'
        urlretrieve(ftp_address, f_tar)

    f = tarfile.open(f_tar)
    adata_tar_path = 'KPTracer-Data/expression/adata_processed.nt.h5ad'
    f.extract(adata_tar_path, data_dir)
    os.rename(data_dir.joinpath(adata_tar_path), orig_adata_path)


# %%
# !ls data


# %%
adata_nt.obs_keys()
