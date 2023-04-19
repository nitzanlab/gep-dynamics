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
# # Simulating data for the different operator based on K12 decomposition
#
# 1. Decompose a dataset
# 2. Approximate the data, and then sample from a Poisson with these rates (k12_0)
# 3. Remove one of the GEPs from both W and H, approximate the data and then sample from a Poisson with these rates (K12_m_p[#])
# 4. Use NMF to decompose the resampled datasets
# 5. Compare the usages to the original
# 6. Compare the genes to the original
# 7. Perform decomposition of K12_m_p[#] using K12_0 GEPs (loss of GEP operator)
# 8. Perform decomposition of K12_0 using K12_m_p[#] GEPs (gain of GEP operator) 
# 9. Perform pfnmf with 1, 2, 3 extra GEPs of K12_m_p[#] using K12_0 GEPs (loss of GEP operator)
# 10. Perform pfnmf with 1, 2, 3 extra GEPs of K12_0 using K12_m_p[#] GEPs (gain of GEP operator)
#

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ### Imports and Data Loading

# %%
# %%time
# %load_ext autoreload
# %autoreload 2

import sys
import os

from urllib.request import urlretrieve
from argparse import Namespace

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.decomposition import _nmf as sknmf
from sklearn.metrics import jaccard_score
from scipy.stats import rankdata
from scipy.cluster import hierarchy

from gepdynamics import _utils
from gepdynamics import _constants
from gepdynamics import cnmf
from gepdynamics import pfnmf

# Move to the project's home directory, as defined in _constants
_utils.cd_proj_home()
print(os.getcwd())

rng = np.random.default_rng()

# %%
import torch
assert torch.cuda.is_available()
device = 'cuda'

# %%
results_dir = _utils.set_dir('results')
notebook_dir = _utils.set_dir(results_dir.joinpath('marjanovic_operator_simulation'))
orig_adata_path = results_dir.joinpath('marjanovic_mmLungPlate.h5ad')
split_adatas_dir = _utils.set_dir(results_dir.joinpath('marjanovic_mmLungPlate_split'))

# adata = sc.read(orig_adata_path)
# sc.external.pl.phate(adata, color=['clusterK12', 'timesimple'])
# adata

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ### 1. Decomposing the data

# %%
subdata = sc.read_h5ad(split_adatas_dir.joinpath('04_K_12w_ND_GEPs.h5ad'))
sname = 'k12'
subdata

# %%
# Parameters
beta_loss = 'kullback-leibler'
max_iter = 500

# statistics variable:
prog_usage = 100 * subdata.obsm['usages_norm'].sum(axis=0) / subdata.obsm['usages_norm'].sum()


# %%
# Working in the transposed notation to get the GEPs: X.T ~ H.T @ W.T

nmf_kwargs={'H': subdata.obsm['usages'].T.copy(),
            'update_H': False,
            'tol': _constants.NMF_TOLERANCE,
            'max_iter': max_iter,
            'beta_loss': beta_loss
           }

# Variance normalized version of the data
X = sc.pp.scale(subdata.X[:, subdata.var.highly_variable].toarray(), zero_center=False)

tens = torch.tensor(X.T).to(device)

W, H, n_iter = cnmf.nmf_torch(X.T, nmf_kwargs, tens, verbose=True)
print(f'Error per sample = {(sknmf._beta_divergence(X.T, W, H, beta_loss) / subdata.n_obs): .1f}')

del tens

# Normalizing the GEPs so L2 norm == 1
rownorm = np.linalg.norm(W.T, ord=2, axis=1, keepdims=True)

geps = W.T / rownorm
usages = subdata.obsm['usages'] * rownorm.T

geps[geps < np.finfo(geps.dtype).eps**2] = 0
usages[usages < np.finfo(usages.dtype).eps**2] = 0

geps.shape


# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ### 2. Recompose the data and sample counts from Poisson with $WH$ rates

# %%
recompositions = {}
orig_prog_names = [f'{sname}.p{i}' for i in range(subdata.uns['cnmf_params']['k_nmf'])]

# %%
dict_key = f'{sname}_0'
ns = Namespace(name=dict_key, W=usages, H=geps)

ns.rank = ns.W.shape[1]
ns.data_prog_names = orig_prog_names.copy()
ns.nmf_prog_names = [f'{dict_key}.p{i}' for i in range(ns.rank)]

recompositions[dict_key] = ns

ns.data = rng.poisson(lam=(ns.W @ ns.H)).astype(X.dtype)

ns.data_error = sknmf._beta_divergence(ns.data, ns.W, ns.H, beta_loss) / subdata.n_obs
print(f'Resampled data error per sample = {ns.data_error: .1f}')

print(f'non-zero entries of original data is {np.count_nonzero(X) * 100 / np.multiply(*X.shape): .1f}%')
print(f'non-zero entries of resampled data is {np.count_nonzero(ns.data) * 100 / np.multiply(*ns.data.shape): .1f}%')

# print the keys gained per experiment this far
vars(ns).keys()

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ### 3. Recomposing the data without one GEP and sampling counts from Poisson

# %%
print(f"Resampled data error per sample = {recompositions[f'{sname}_0'].data_error: .1f}")
print(f'non-zero entries of original data is {np.count_nonzero(X) * 100 / np.multiply(*X.shape): .1f}%\n')

for dropped_GEP_index in range(usages.shape[1]):
    print(f'Dropping gep {dropped_GEP_index}, counting for'
          f' {prog_usage[dropped_GEP_index]: .1f}% of usage')
    
    dict_key = f'{sname}_m_p{dropped_GEP_index}'
    ns = Namespace(
        name=dict_key,
        dropped_GEP_index=dropped_GEP_index,
        W=np.delete(usages, dropped_GEP_index, axis=1),
        H=np.delete(geps, dropped_GEP_index, axis=0))
    
    # Give some value for very small entries
    threshold = 1e-4
    ns.W[ns.W.sum(axis=1) <= threshold, :] += threshold
    ns.H[:, ns.H.sum(axis=0) <= threshold] += threshold
    
    ns.rank = ns.W.shape[1]
    ns.data_prog_names = orig_prog_names.copy()
    ns.data_prog_names.pop(ns.dropped_GEP_index)
    ns.nmf_prog_names = [f'{dict_key}.p{i}' for i in range(ns.rank + 1)]
    ns.nmf_prog_names.pop(ns.dropped_GEP_index)
    
    ns.data = rng.poisson(lam=(ns.W @ ns.H )).astype(X.dtype)
    
    # turn low counts cell closer to their expactation
    expectation = (ns.W @ ns.H).astype(np.float64)
    expectation[expectation < np.finfo(expectation.dtype).eps**2] = 0
    for cell_index in np.where(ns.data.sum(axis=1) < 10)[0]:
        multinomial_proporitons = expectation[cell_index, :]
        multinomial_proporitons /= multinomial_proporitons.astype(np.float64).sum()
        for missing_count in range(10 - int(ns.data[cell_index].sum())):
            ns.data[cell_index, rng.multinomial(1, multinomial_proporitons)] += 1
        
    # turn low counts genes closer to their expactation
    for gene_index in np.where(ns.data.sum(axis=0) < 5)[0]:
        multinomial_proporitons = expectation[:, gene_index]
        multinomial_proporitons /= multinomial_proporitons.astype(np.float64).sum()
        for missing_count in range(5 - int(ns.data[:, gene_index].sum(axis=0))):
            ns.data[rng.multinomial(1, multinomial_proporitons), gene_index] += 1
        
    ns.data_error = sknmf._beta_divergence(ns.data, ns.W, ns.H, beta_loss) / subdata.n_obs
    
    print(f'Resampled data error per sample = {ns.data_error: .1f}')

    print(f'non-zero entries of resampled data is {np.count_nonzero(ns.data) * 100 / np.multiply(*ns.data.shape): .1f}%\n')
    
    recompositions[dict_key] = ns

# %%
for name, ns in recompositions.items():
    print('working on ', ns.name)
    print(ns.data.sum(axis=0).min())
    print(ns.data.sum(axis=1).min())
    print(ns.H.sum(axis=0).min())
    print(ns.W.sum(axis=1).min())

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ### 4. Use NMF to decompose the resampled datasets

# %%
for name, ns in recompositions.items():
    print('working on ', ns.name)
    
    nmf_kwargs={
        'tol': _constants.NMF_TOLERANCE,
        'max_iter': max_iter,
        'beta_loss': beta_loss,
        'n_components': ns.rank
    }
    
    tens = torch.tensor(ns.data).to(device)
    W, H, n_iter = cnmf.nmf_torch(ns.data, nmf_kwargs, tens, verbose=True)
    
    ns.nmf_error = sknmf._beta_divergence(ns.data, W, H, beta_loss) / subdata.n_obs
    print(f'Error per sample = {ns.nmf_error: .1f}')

    del tens

    # Normalizing the GEPs so L2 norm == 1
    rownorm = np.linalg.norm(H, ord=2, axis=1, keepdims=True)

    ns.H_nmf = H / rownorm
    ns.W_nmf = W * rownorm.T
    
    # Normalized usage
    ns.W_nmf_proportion = ns.W_nmf / ns.W_nmf.sum(axis=0, keepdims=True)
    
    print()

# %%
# Rearranging the programs using a permutation matrix
orig_ns = recompositions[f'{sname}_0']

for name, ns in recompositions.items():
    print('working on ', ns.name)
    perm = np.zeros((ns.rank, ns.rank), dtype=int)

    cc = np.corrcoef(orig_ns.W.T, ns.W_nmf.T)[:orig_ns.rank, orig_ns.rank:]

    if vars(ns).get('dropped_GEP_index', -1) >= 0:
        cc = np.delete(cc, ns.dropped_GEP_index, axis=0)

    df = pd.DataFrame(index=np.arange(ns.rank**2))
    df['row'] = df.index // ns.rank
    df['col'] = df.index % ns.rank
    df['val'] = cc.flatten()
    df.sort_values('val', inplace=True, ascending=False, ignore_index=True)

    for index in range(ns.rank):
        row, col = df.iloc[0, 0], df.iloc[0, 1]
        perm[row, col] = 1
        df.drop(df[df.row == row].index, inplace=True)
        df.drop(df[df.col == col].index, inplace=True)

    # print(perm)
    
    ns.H_nmf = perm @ ns.H_nmf
    ns.W_nmf = ns.W_nmf @ perm.T
    
    # Normalized usage
    ns.W_nmf_proportion = ns.W_nmf_proportion @ perm.T


# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ### 5. Compare the usages to the original
#

# %%
for name, ns in recompositions.items():

    ax = _utils.heatmap_with_numbers(
        np.corrcoef(orig_ns.W.T, ns.W_nmf.T)[:orig_ns.rank, orig_ns.rank:],
        title=f'{name} usages correlations', param_dict={'vmin': 0, 'vmax': 1})

    ax.set_xticks(ticks=ax.get_xticks(), labels=ns.nmf_prog_names)
    ax.set_yticks(ticks=ax.get_yticks(), labels=orig_ns.data_prog_names)
    ax.tick_params(axis='x', labelrotation = 45)

    plt.show()
    plt.close()


# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ### 6. Compare the genes to the original

# %%
# Extracting coefficients:
z_layer = 'cpm_log1p_zscore'

subdata.layers[z_layer] = sc.pp.normalize_total(subdata, target_sum=1e6, inplace=False)['X']
sc.pp.log1p(subdata, layer=z_layer)
sc.pp.scale(subdata, layer=z_layer)

orig_ns.data_gene_coefs = pd.DataFrame(
    _utils.fastols(orig_ns.W_nmf_proportion, subdata.layers[z_layer]).T,
    index=subdata.var.index,
    columns=orig_ns.data_prog_names)

for name, ns in recompositions.items():
    ns.nmf_gene_coefs = pd.DataFrame(
        _utils.fastols(ns.W_nmf_proportion, subdata.layers[z_layer]).T,
        index=subdata.var.index,
        columns=ns.nmf_prog_names)

# %%
for name, ns in recompositions.items():
    concatenated_spectras = pd.concat([orig_ns.data_gene_coefs, ns.nmf_gene_coefs], axis=1)

    n_genes, n_programs = concatenated_spectras.shape

    ranked_coefs = n_genes - rankdata(concatenated_spectras, axis=0)

    ranked_coefs[ranked_coefs > _constants.N_COMPARED_RANKED] = _constants.N_COMPARED_RANKED

    spearman_corr = np.corrcoef(ranked_coefs, rowvar=False)
    
    ax = _utils.heatmap_with_numbers(
        spearman_corr[:orig_ns.rank, orig_ns.rank:],
        title=f'{name} gene coefficients correlations', param_dict={'vmin': 0, 'vmax': 1})

    ax.set_xticks(ticks=ax.get_xticks(), labels=ns.nmf_prog_names)
    ax.set_yticks(ticks=ax.get_yticks(), labels=orig_ns.data_prog_names)
    ax.tick_params(axis='x', labelrotation = 45)

    plt.show()
    plt.close()


# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# ### 7. Perform decomposition of K12_m_p[#] using K12 GEPs (loss of GEP operator)

# %%
#  k12_m_p[].data ~ W @ k12_0.nmf_H

orig_ns = recompositions[f'{sname}_0']

for name, ns in recompositions.items():
    print('working on ', ns.name)
    
    nmf_kwargs={'H': orig_ns.H.copy(),
                'update_H': False,
                'tol': _constants.NMF_TOLERANCE,
                'max_iter': max_iter,
                'beta_loss': beta_loss
               }
    tens = torch.tensor(ns.data).to(device)
    
    W, H, n_iter = cnmf.nmf_torch(ns.data, nmf_kwargs, tens, verbose=True)

    ns.pfnmf_error = []
    ns.pfnmf_error.append(sknmf._beta_divergence(ns.data, W, H, beta_loss) / subdata.n_obs)
    print(f'Error per sample = {ns.pfnmf_error[-1]: .1f}')

    del tens
    
    ns.W_pfnmf = []
    ns.W_pfnmf.append(W)
    
    print()


# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# #### usages comparisons
# Results came out clear as expected

# %%
for name, ns in recompositions.items():
    print('working on ', ns.name)
        
    ax = _utils.heatmap_with_numbers(
        np.corrcoef(orig_ns.W.T, ns.W_pfnmf[0].T)[:orig_ns.rank, orig_ns.rank:],
        title=f'{name} decomposed with k12_0 usages correlations', param_dict={'vmin': 0, 'vmax': 1})

    # ax.set_xticks(ticks=ax.get_xticks(), labels=ns.nmf_prog_names)
    ax.set_yticks(ticks=ax.get_yticks(), labels=orig_ns.data_prog_names)
    ax.tick_params(axis='x', labelrotation = 45)

    plt.show()
    plt.close()

    
    title = f'{name} decomposed with k12_0 GEPs, normalized usages'

    eps_W = ns.W_pfnmf[0] + np.finfo(float).eps**2
    normalized_usages = (eps_W  / (eps_W.sum(axis=1, keepdims=True)))
    normalized_usages[eps_W.sum(axis=1) < np.finfo(float).eps] = np.finfo(float).eps**2

    prog_percent = normalized_usages.sum(axis=0) * 100 / subdata.n_obs
    prog_label_2l = [name + f'\n({prog_percent[i]: 0.1f}%)' for i, name in enumerate(orig_ns.nmf_prog_names)]

    un_sns = _utils.plot_usages_norm_clustermaps(subdata, normalized_usages=normalized_usages,
        columns=prog_label_2l, title='', show=True, sns_clustermap_params={'figsize':(6, 6)})

    plt.close(un_sns.figure)


# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# #### Gene coefficients comparison
# Surprisingly, a lot of information was retained for dropped GEPs both with beta coefficients and jaccard score

# %%
# Extracting gene coefficients from resampled data:

for name, ns in recompositions.items():  
    # The experiment can be repeated for NMF on non-variance stablized data
    obs_mask = ns.data.sum(axis=1) > np.finfo(float).eps
    var_mask = ns.data.std(axis=0) > np.finfo(float).eps
    
    data = ns.data.copy()[obs_mask]
    data *= (500 / data.sum(axis=1, keepdims=True))
    np.log1p(data, out=data)
    data -= data.mean(axis=0, keepdims=True)
    
    data[:, var_mask] /= data[:, var_mask].std(axis=0, keepdims=True) 

    # not perfectly sound, but OK for now
    eps_W = ns.W_pfnmf[0][obs_mask] + np.finfo(float).eps**2
    normalized_usages = (eps_W  / (eps_W.sum(axis=1, keepdims=True)))
    normalized_usages[eps_W.sum(axis=1) < np.finfo(float).eps] = np.finfo(float).eps**2

    ns.gene_coefs__pfnmf = []
    ns.gene_coefs__pfnmf.append(pd.DataFrame(
        _utils.fastols(normalized_usages, data).T,
        index=subdata.var[subdata.var.highly_variable].index,
        columns=orig_ns.nmf_prog_names))


# %%
concatenated_spectras = pd.concat([orig_ns.data_gene_coefs[subdata.var.highly_variable],
                                   orig_ns.gene_coefs__pfnmf[0]], axis=1)

n_genes, n_programs = concatenated_spectras.shape

ranked_coefs = n_genes - rankdata(concatenated_spectras, axis=0)

ranked_coefs[ranked_coefs > 400] = 400

spearman_corr = np.corrcoef(ranked_coefs, rowvar=False)

ax = _utils.heatmap_with_numbers(
    spearman_corr[:orig_ns.rank, orig_ns.rank:],
    title=f'{orig_ns.name} sampled data vs original data, gene coefficients correlations', param_dict={'vmin': 0, 'vmax': 1})

ax.set_xticks(ticks=ax.get_xticks(), labels=orig_ns.nmf_prog_names)
ax.set_yticks(ticks=ax.get_yticks(), labels=orig_ns.data_prog_names)
ax.tick_params(axis='x', labelrotation = 45)

plt.show()
plt.close()


# %%
n_coefs = 400

for name, ns in recompositions.items():
    
    concatenated_spectras = pd.concat([orig_ns.gene_coefs__pfnmf[0],
                                       ns.gene_coefs__pfnmf[0]], axis=1)

    n_genes, n_programs = concatenated_spectras.shape

    ranked_coefs = n_genes - rankdata(concatenated_spectras, axis=0)
    
    ranked_coefs[ranked_coefs > n_coefs] = n_coefs
    
    spearman_corr = np.corrcoef(ranked_coefs, rowvar=False)
    
    ax = _utils.heatmap_with_numbers(
        spearman_corr[:orig_ns.rank, orig_ns.rank:],
        title=f'{name} decomposed with k12_0 GEPs, gene coefficients correlations', param_dict={'vmin': 0, 'vmax': 1})

    ax.set_xticks(ticks=ax.get_xticks(), labels=orig_ns.nmf_prog_names)
    ax.set_yticks(ticks=ax.get_yticks(), labels=orig_ns.nmf_prog_names)
    ax.tick_params(axis='x', labelrotation = 45)

    plt.show()
    plt.close()


# %%
# Calculating jaccard scores on top 10% (200) of coefficients.
n_coefs = 200

maxvals_orig = orig_ns.gene_coefs__pfnmf[0] >= orig_ns.gene_coefs__pfnmf[0].values.max(axis=1, keepdims=True)

for name, ns in recompositions.items():
    maxvals_ns = ns.gene_coefs__pfnmf[0] >= ns.gene_coefs__pfnmf[0].values.max(axis=1, keepdims=True)
    
    concatenated_maxvals = pd.concat([maxvals_orig, maxvals_ns], axis=1)
    
    n_genes, n_programs = concatenated_maxvals.shape
    
    concatenated_spectras = pd.concat([orig_ns.gene_coefs__pfnmf[0],
                                       ns.gene_coefs__pfnmf[0]], axis=1)
    ranked_coefs = n_genes - rankdata(concatenated_spectras, axis=0)
    
    jaccard_scores = _utils.df_jaccard_score(concatenated_maxvals & (ranked_coefs < n_coefs))
    
    ax = _utils.heatmap_with_numbers(
        jaccard_scores[:orig_ns.rank, orig_ns.rank:],
        title=f'{name} decomposed with k12_0 GEPs, maximal gene coefficients Jaccard score', param_dict={'vmin': 0, 'vmax': 1})

    ax.set_xticks(ticks=ax.get_xticks(), labels=orig_ns.nmf_prog_names)
    ax.set_yticks(ticks=ax.get_yticks(), labels=orig_ns.nmf_prog_names)
    ax.tick_params(axis='x', labelrotation = 45)

    plt.show()
    plt.close()

# %% [markdown] tags=[]
# ### 8. Perform decomposition of K12_0 using K12_m_p[#] GEPs (gain of GEP operator) 

# %%
#  k12_0.data ~ W @ k12_m_p[].H_nmf

for name, ns in recompositions.items():
    print('working on ', ns.name)
    
    nmf_kwargs={'H': ns.H_nmf.copy(),
                'update_H': False,
                'tol': _constants.NMF_TOLERANCE,
                'max_iter': max_iter,
                'beta_loss': beta_loss
               }
    tens = torch.tensor(orig_ns.data).to(device)
    
    W, H, n_iter = cnmf.nmf_torch(orig_ns.data, nmf_kwargs, tens, verbose=True)

    ns.gain_pfnmf_error = []
    ns.gain_pfnmf_error.append(sknmf._beta_divergence(orig_ns.data, W, H, beta_loss) / subdata.n_obs)
    print(f'Error per sample = {ns.pfnmf_error[-1]: .1f}')

    del tens
    
    ns.W_gain_pfnmf = []
    ns.W_gain_pfnmf.append(W)
    
    print()

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# #### usages comparisons
# Results: other GEPs seem to "take charge" of cells that were primarily using a dropped GEP

# %%
for name, ns in recompositions.items():
    print('working on ', ns.name)
        
    ax = _utils.heatmap_with_numbers(
        np.corrcoef(orig_ns.W.T, ns.W_gain_pfnmf[0].T)[:orig_ns.rank, orig_ns.rank:],
        title=f'k12_0 decomposed with {name} GEPs, usages correlations', param_dict={'vmin': 0, 'vmax': 1})

    ax.set_xticks(ticks=ax.get_xticks(), labels=ns.nmf_prog_names)
    ax.set_yticks(ticks=ax.get_yticks(), labels=orig_ns.nmf_prog_names)
    ax.tick_params(axis='x', labelrotation = 45)

    plt.show()
    plt.close()

    
    title = f'k12_0 decomposed with {name} GEPs, normalized usages'

    eps_W = ns.W_gain_pfnmf[0] + np.finfo(float).eps**2
    normalized_usages = (eps_W  / (eps_W.sum(axis=1, keepdims=True)))
    normalized_usages[eps_W.sum(axis=1) < np.finfo(float).eps] = np.finfo(float).eps**2

    prog_percent = normalized_usages.sum(axis=0) * 100 / subdata.n_obs
    prog_label_2l = [name + f'\n({prog_percent[i]: 0.1f}%)' for i, name in enumerate(ns.nmf_prog_names)]

    un_sns = _utils.plot_usages_norm_clustermaps(subdata, normalized_usages=normalized_usages,
        columns=prog_label_2l, title='', show=True, sns_clustermap_params={'figsize':(6, 6)})

    plt.close(un_sns.figure)


# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# #### Gene coefficients comparison
# Results: some low correlation arrives from the GEP that is used to "cover" for the dropped one

# %%
# Extracting gene coefficients from resampled data:

obs_mask = orig_ns.data.sum(axis=1)!=0
var_mask = orig_ns.data.std(axis=0)!=0

data = orig_ns.data.copy()[obs_mask]
data *= (500 / data.sum(axis=1, keepdims=True))
np.log1p(data, out=data)
data -= data.mean(axis=0, keepdims=True)

data[:, var_mask] /= data[:, var_mask].std(axis=0, keepdims=True) 

for name, ns in recompositions.items():  
    
    # not perfectly sound, but OK for now
    eps_W = ns.W_gain_pfnmf[0][obs_mask] + np.finfo(float).eps**2
    normalized_usages = (eps_W  / (eps_W.sum(axis=1, keepdims=True)))
    normalized_usages[eps_W.sum(axis=1) < np.finfo(float).eps] = np.finfo(float).eps**2

    ns.gene_coefs_gain_pfnmf = []
    ns.gene_coefs_gain_pfnmf.append(pd.DataFrame(
        _utils.fastols(normalized_usages, data).T,
        index=subdata.var[subdata.var.highly_variable].index,
        columns=ns.nmf_prog_names))

# %%
n_coefs = 400

for name, ns in recompositions.items():
    
    concatenated_spectras = pd.concat([orig_ns.gene_coefs_gain_pfnmf[0],
                                       ns.gene_coefs_gain_pfnmf[0]], axis=1)

    n_genes, n_programs = concatenated_spectras.shape

    ranked_coefs = n_genes - rankdata(concatenated_spectras, axis=0)
    
    ranked_coefs[ranked_coefs > n_coefs] = n_coefs
    
    spearman_corr = np.corrcoef(ranked_coefs, rowvar=False)
    
    ax = _utils.heatmap_with_numbers(
        spearman_corr[:orig_ns.rank, orig_ns.rank:],
        title=f'k12_0 decomposed with {name} GEPs, gene coefficients correlations', param_dict={'vmin': 0, 'vmax': 1})

    ax.set_xticks(ticks=ax.get_xticks(), labels=ns.nmf_prog_names)
    ax.set_yticks(ticks=ax.get_yticks(), labels=orig_ns.nmf_prog_names)
    ax.tick_params(axis='x', labelrotation = 45)

    plt.show()
    plt.close()


# %%
# Calculating jaccard scores on top 10% (200) of coefficients.
n_coefs = 200

maxvals_orig = orig_ns.gene_coefs_gain_pfnmf[0] >= orig_ns.gene_coefs_gain_pfnmf[0].values.max(axis=1, keepdims=True)

for name, ns in recompositions.items():
    maxvals_ns = ns.gene_coefs_gain_pfnmf[0] >= ns.gene_coefs_gain_pfnmf[0].values.max(axis=1, keepdims=True)
    
    concatenated_maxvals = pd.concat([maxvals_orig, maxvals_ns], axis=1)
    
    n_genes, n_programs = concatenated_maxvals.shape
    
    concatenated_spectras = pd.concat([orig_ns.gene_coefs_gain_pfnmf[0],
                                       ns.gene_coefs_gain_pfnmf[0]], axis=1)
    ranked_coefs = n_genes - rankdata(concatenated_spectras, axis=0)
    
    jaccard_scores = _utils.df_jaccard_score(concatenated_maxvals & (ranked_coefs < n_coefs))
    
    ax = _utils.heatmap_with_numbers(
        jaccard_scores[:orig_ns.rank, orig_ns.rank:],
        title=f'k12_0 decomposed with {name} GEPs, maximal gene coefficients Jaccard score', param_dict={'vmin': 0, 'vmax': 1})

    ax.set_xticks(ticks=ax.get_xticks(), labels=ns.nmf_prog_names)
    ax.set_yticks(ticks=ax.get_yticks(), labels=orig_ns.nmf_prog_names)
    ax.tick_params(axis='x', labelrotation = 45)

    plt.show()
    plt.close()

# %% [markdown] tags=[]
# ### 9. Perform pfnmf with 1, 2, 3 extra GEPs of K12_m_p[#] using K12_0 GEPs (loss of GEP operator)
# Results: as expected, additional degrees of freedom add little to the final loss

# %%
# # %%script --no-raise-error false

# pfnmf is written for constant W_1, so we will transpose as needed:
# K12_m_p[#] ~ W_1 @ k12_0.H + W_2 @ H_2  <--> K12_m_p[#].T ~ k12_0.H.T @ W_1.T + H_2.T @ W_2.T

pfnmf_repeats = 3

for name, ns in recompositions.items():
    print('working on ', ns.name, f'original loss was {ns.pfnmf_error[0]: .1f}' )
    ns.pfnmf = [0, 1, 2, 3]
    
    if len(ns.pfnmf_error) == 1:
        ns.pfnmf_error.extend([1, 2, 3])
    
    
    for added_rank in [1, 2, 3]:
        print(f"Working on added rank = {added_rank}")

        best_loss = np.infty
        
        for repeat in range(pfnmf_repeats):
            X = ns.data.T
            X[:, (X.sum(axis=0)==0)] += np.finfo(float).eps
            X[(X.sum(axis=1)==0), :] += np.finfo(float).eps
            
            w1, h1, w2, h2, n_iter = pfnmf.pfnmf(
                X, orig_ns.H_nmf.T, rank_2=added_rank, beta_loss=beta_loss,
                tol=_constants.NMF_TOLERANCE, max_iter=max_iter, verbose=False)

            final_loss = pfnmf.calc_beta_divergence(X, w1, w2, h1, h2, beta_loss)

            if final_loss <= best_loss:
                best_loss = final_loss
                ns.pfnmf[added_rank] = {'w1': w1, 'h1': h1, 'w2': w2, 'h2': h2, 'n_iter': n_iter}
                ns.pfnmf_error[added_rank] = final_loss / subdata.n_obs

                print(f"repeat {repeat}, after {n_iter} iterations reached {final_loss: .4e}"
                     f", per sample loss = {ns.pfnmf_error[added_rank]: .1f}")
    print()


# %% [markdown] tags=[]
# ### 10. Perform pfnmf with 1, 2, 3 extra GEPs of K12_0 using K12_m_p[#] GEPs (gain of GEP operator)
# Results: after adding just one more GEP we get similar loss to the baseline

# %%
# # %%script --no-raise-error false

# pfnmf is written for constant W_1, so we will transpose as needed:
# k12_0 ~ W_1 @ K12_m_p[#].H_nmf + W_2 @ H_2  <--> k12_0.T ~ K12_m_p[#].H_nmf.T @ W_1.T + H_2.T @ W_2.T

pfnmf_repeats = 3

X = orig_ns.data.T
X[:, (X.sum(axis=0)==0)] += np.finfo(float).eps
X[(X.sum(axis=1)==0), :] += np.finfo(float).eps

for name, ns in recompositions.items():
    print('working on ', ns.name, f', original loss was {ns.gain_pfnmf_error[0]: .1f}')
    ns.gain_pfnmf = [0, 1, 2, 3]
    
    if len(ns.gain_pfnmf_error) == 1:
        ns.gain_pfnmf_error.extend([1, 2, 3])
    
    
    for added_rank in [1, 2, 3]:
        print(f"Working on added rank = {added_rank}")

        best_loss = np.infty
        
        for repeat in range(pfnmf_repeats):
            w1, h1, w2, h2, n_iter = pfnmf.pfnmf(
                X, ns.H_nmf.T, rank_2=added_rank, beta_loss=beta_loss,
                tol=_constants.NMF_TOLERANCE, max_iter=max_iter, verbose=False)

            final_loss = pfnmf.calc_beta_divergence(X, w1, w2, h1, h2, beta_loss)

            if final_loss <= best_loss:
                best_loss = final_loss
                ns.gain_pfnmf[added_rank] = {'w1': w1, 'h1': h1, 'w2': w2, 'h2': h2, 'n_iter': n_iter}
                ns.gain_pfnmf_error[added_rank] = final_loss / subdata.n_obs

                print(f"repeat {repeat}, after {n_iter} iterations reached {final_loss: .4e}"
                     f", per sample loss = {ns.gain_pfnmf_error[added_rank]: .1f}")
    print()

# %%
