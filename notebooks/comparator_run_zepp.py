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
# # Running the comparator between developmental stages in Zepp et. al.

# %%
# %%time
# %load_ext autoreload
# %autoreload 2

# import sys
import os
import warnings

# from urllib.request import urlretrieve
# from argparse import Namespace
# from importlib import reload


# import numpy as np
# import pandas as pd
import scanpy as sc
# import matplotlib.pyplot as plt
# import seaborn as sns
# import networkx as nx

# from sklearn.decomposition import _nmf as sknmf
from sklearn.exceptions import ConvergenceWarning
# from scipy.stats import rankdata
# from scipy.cluster import hierarchy

from gepdynamics import _utils
# from gepdynamics import _constants
# from gepdynamics import cnmf
# from gepdynamics import pfnmf
from gepdynamics import comparator

# Move to the project's home directory, as defined in _constants
_utils.cd_proj_home()
print(os.getcwd())

# %%
# %%time
results_dir = _utils.set_dir('results_zepp')

split_adatas_dir = results_dir.joinpath('split_adatas')

k_12 = sc.read_h5ad(split_adatas_dir.joinpath('04_K_12w_ND_GEPs.h5ad'))

k_30 = sc.read_h5ad(split_adatas_dir.joinpath('05_K_30w_ND_GEPs.h5ad'))

kp_12 = sc.read_h5ad(split_adatas_dir.joinpath('06_KP_12w_ND_GEPs.h5ad'))

kp_30 = sc.read_h5ad(split_adatas_dir.joinpath('08_KP_30w_ND_GEPs.h5ad'))

# %% [markdown]
# ### Comparing Kras 12 and 30 weeks

# %%
import numpy as np
def load_from_file(filename: _utils.PathLike, adata_a: sc.AnnData, adata_b: sc.AnnData) -> 'Comparator':
    # make sure the file exists
    assert os.path.exists(filename), f'File {filename} does not exist'

    new_instance = np.load(filename, allow_pickle=True)['obj'].item()

    new_instance.adata_a = adata_a
    new_instance.adata_b = adata_b
    return new_instance


# %%

pairs = [(k_12, k_30), (k_30, k_12), (kp_12, kp_30), (kp_30, kp_12), (k_12, kp_12), (k_30, kp_30)]

for adata_a, adata_b in pairs:
    comparison_dir = results_dir.joinpath(f"{adata_a.uns['sname']}_{adata_b.uns['sname']}")

    tst = comparator.Comparator(
        adata_a, adata_a.obsm['usages'], adata_b, comparison_dir, 'torchnmf',
        max_nmf_iter=800, verbosity=1, )

    with warnings.catch_warnings():  # supress convergence warning
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)
        tst.extract_geps_on_jointly_hvgs()

    tst.decompose_b(repeats=10)

    tst.plot_loss_per_cell_histograms(show=False)
    tst.plot_usages_clustermaps(show=False)
    tst.plot_decomposition_comparisons(show=False)
    tst.run_gsea(gprofiler_kwargs=dict(organism='hsapiens', sources=['GO:BP', 'WP', 'REAC', 'KEGG']))

    tst.calculate_fingerprints()

    tst.print_errors()

    tst.save_to_file(comparison_dir.joinpath('comparator.npz'))

# %%
for adata_a, adata_b in [(k_12, k_30), (k_30, k_12), (k_12, kp_12),
                         (k_30, kp_30), (kp_12, kp_30), (kp_30, kp_12)]:
    comparison_dir = results_dir.joinpath(f"{adata_a.uns['sname']}_{adata_b.uns['sname']}", 'comparator.npz')
    
    tst = comparator.Comparator.load_from_file(comparison_dir, adata_a, adata_b)
    for res in tst._all_results:
        res.calculate_prog_labels()
    tst.a_result.calculate_prog_labels()
    
    tst.save_to_file(comparison_dir)
