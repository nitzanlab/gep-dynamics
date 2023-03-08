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

# %%
# %%time
# %load_ext autoreload
# %autoreload 2

import sys
import os
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from gepdynamics import _utils
from gepdynamics import _constants
from gepdynamics import cnmf

# Move to the project's home directory, as defined in _constants
_utils.cd_proj_home()
print(os.getcwd())

# %%
import torch
assert torch.cuda.is_available()
device = 'cuda'

# %%
results_dir = _utils.set_dir('results')
orig_adata_path = results_dir.joinpath('marjanovic_mmLungPlate.h5ad')
split_adatas_dir = _utils.set_dir(results_dir.joinpath('marjanovic_mmLungPlate_split'))

adata = sc.read(orig_adata_path)
sc.external.pl.phate(adata, color=['clusterK12', 'timesimple'])
adata

# %%
k_12 = sc.read_h5ad(split_adatas_dir.joinpath('04_K_12w_ND_GEPs.h5ad'))
k_12

# %%
k_30 = sc.read_h5ad(split_adatas_dir.joinpath('05_K_30w_ND_GEPs.h5ad'))
k_30

# %% [markdown]
# ## Running pfNMF on K30 using the K12 GEPs

# %% [markdown]
# ### Preparing K12 and K30 data on joint highly variable genes (jHVGs)
#

# %%
print("Number of joint HVGs for K12 and K30 datasets = "
      f"{np.sum(k_30.var.highly_variable & k_12.var.highly_variable)}")

# %% tags=[]
joint_K12_K30_var = sc.pp.highly_variable_genes(
    adata[adata.obs.timesimple.isin([k_12.uns['name'], k_30.uns['name']])],
    flavor='seurat_v3', n_top_genes=_constants.NUMBER_HVG, inplace=False)
joint_K12_K30_var

# %%
print("Selecting 2000 joint HVGs, intersection with K12 HVGS is "
      f"{np.sum(joint_K12_K30_var.highly_variable & k_12.var.highly_variable)}"
      ", and with K30 is "
      f"{np.sum(joint_K12_K30_var.highly_variable & k_12.var.highly_variable)}")

# %%
# Variance normalized version of K12 data on the jHVGs
X12 = sc.pp.scale(k_12.X[:, joint_K12_K30_var.highly_variable].toarray(), zero_center=False)
X12[:4, :4]

# %%
# Variance normalized version of K30 data on the jHVGs
X30 = sc.pp.scale(k_30.X[:, joint_K12_K30_var.highly_variable].toarray(), zero_center=False)
X30[:4, :4]

# %% [markdown]
# ### Running NNLS to get K12 GEPs (geps12) on jHVGs

# %%
# Working in the transposed notation to get the programs: X.T ~ H.T @ W.T

nmf_kwargs={'H': k_12.obsm['usages'].T.copy(),
            'update_H': False,
            'tol': _constants.NMF_TOLERANCE,
            'n_iter': 500,
            'beta_loss': 'kullback-leibler'
           }

tens = torch.tensor(X12.T).to(device)

W, H, n_iter = cnmf.nmf_torch(X12.T, nmf_kwargs, tens, verbose=True)

del tens

geps12 = W.T
geps12.shape

# %% [markdown]
# ### Decomposing K30 with geps12 and 0 additional programs

# %%
#  x30 ~ W @ geps12

nmf_kwargs={'H': geps12.copy(),
            'update_H': False,
            'tol': _constants.NMF_TOLERANCE,
            'n_iter': 500,
            'beta_loss': 'kullback-leibler'
           }

tens = torch.tensor(X30).to(device)

W, H, n_iter = cnmf.nmf_torch(X30, nmf_kwargs, tens, verbose=True)

del tens

# %%
#  x30 ~ W @ geps12

nmf_kwargs={
    'n_components': 6,
    'tol': _constants.NMF_TOLERANCE,
    'n_iter': 500,
    'beta_loss': 'kullback-leibler'
   }

tens = torch.tensor(X30).to(device)

W, H, n_iter = cnmf.nmf_torch(X30, nmf_kwargs, tens, verbose=True)

del tens

# %%
