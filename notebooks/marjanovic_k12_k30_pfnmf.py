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
# # Running pfNMF on Marjanovic sample Kras30 using Kras 12
# To estimates the aplicability of GEPs from the Kras sample at 12 weeks (K12) on the cells from Kras 30 weeks (K30), we use the K12 GEPs readjusted to the jointly highly variable genes (jHVGs) to decompose the K30 dataset using the partially fixed NMF algorithm (pfnmf)
#
# steps:
# 1.  Loading the data
# 2.  Finding jointly highly variable genes
# 3.  Ruuning NNLS to get the K12 GEPs on the jHVGs
# 4.  Decomposing the K30 dataset de-novo at the same rank as K12
# 5.  Decomposing the K30 dataset using pfnmf with 0, 1, 2, 3 additional novel programs
# 6.  Evaluating the usage patterns of the different decompositions
# 7. 

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true
# ## Imports and loading data

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
import networkx as nx

from sklearn.decomposition import _nmf as sknmf
from scipy.stats import rankdata
from scipy.cluster import hierarchy

from gepdynamics import _utils
from gepdynamics import _constants
from gepdynamics import cnmf
from gepdynamics import pfnmf

# Move to the project's home directory, as defined in _constants
_utils.cd_proj_home()
print(os.getcwd())

# %%
import torch
assert torch.cuda.is_available()
device = 'cuda'

# %%
results_dir = _utils.set_dir('results')
notebook_dir = _utils.set_dir(results_dir.joinpath('marjanovic_k12_k30_pfnmf'))
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

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ## Running pfNMF on K30 using the K12 GEPs

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ### Preparing K12 and K30 data on joint highly variable genes (jHVGs)
#

# %% tags=[]
var_subset = (k_12.var.n_cells >= 5) & (k_30.var.n_cells >= 5)
obs_subset = adata.obs.timesimple.isin([k_12.uns['name'], k_30.uns['name']])

joint_K12_K30_var = sc.pp.highly_variable_genes(
    adata[obs_subset, var_subset], flavor='seurat_v3',
    n_top_genes=_constants.NUMBER_HVG, inplace=False)
joint_K12_K30_var

# %%
print("Selecting 2000 joint HVGs, intersection with K12 HVGS is "
      f"{np.sum(joint_K12_K30_var.highly_variable & k_12.var.highly_variable)}"
      ", and with K30 is "
      f"{np.sum(joint_K12_K30_var.highly_variable & k_30.var.highly_variable)}")
joint_K12_K30_HVG = joint_K12_K30_var[joint_K12_K30_var.highly_variable].index

# %%
# Variance normalized version of K12 data on the jHVGs
X12 = sc.pp.scale(k_12[:, joint_K12_K30_HVG].X.toarray(), zero_center=False)
print(f'X12.shape = {X12.shape}')
X12[:4, :4]

# %%
# Variance normalized version of K30 data on the jHVGs
X30 = sc.pp.scale(k_30[:, joint_K12_K30_HVG].X.toarray(), zero_center=False)
print(f'X30.shape = {X30.shape}')
X30[:4, :4]

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ### Running NNLS to get K12 GEPs (geps12) on jHVGs

# %%
#Parameters
beta_loss = 'kullback-leibler'
max_iter = 500

# %%
100 * k_12.obsm['usages_norm'].sum(axis=0) / k_12.n_obs

# %%
# Working in the transposed notation to get the programs: X.T ~ H.T @ W.T

nmf_kwargs={'H': k_12.obsm['usages'].T.copy(),
            'update_H': False,
            'tol': _constants.NMF_TOLERANCE,
            'n_iter': max_iter,
            'beta_loss': beta_loss
           }

tens = torch.tensor(X12.T).to(device)

W, H, n_iter = cnmf.nmf_torch(X12.T, nmf_kwargs, tens, verbose=True)
print(f'Error per sample = {(sknmf._beta_divergence(X12.T, W, H, beta_loss) / k_12.n_obs): .3e}')

del tens

geps12 = W.T
geps12.shape

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ### Decomposing K30 de-novo with same rank as geps12 (on jHVGs)

# %%
pfnmf_results = {}
rank_k12 = geps12.shape[0]

# %% magic_args="--no-raise-error false" language="script"
#
# nmf_kwargs={
#     'n_components': rank_k12,
#     'tol': _constants.NMF_TOLERANCE,
#     'n_iter': max_iter,
#     'beta_loss': beta_loss
#    }
#
# tens = torch.tensor(X30).to(device)
#
# W, H, n_iter = cnmf.nmf_torch(X30, nmf_kwargs, tens, verbose=True)
#
# final_loss = sknmf._beta_divergence(X30, W, H, beta_loss)
# print(f'Error per sample = {(final_loss / k_30.n_obs): .3e}')
#
# pfnmf_results['de_novo'] = {'W': W, 'H': H, 'n_iter': n_iter, 'final_loss': final_loss}
#
# del tens

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ### Decomposing K30 with geps12 and no additional programs

# %% magic_args="--no-raise-error false" language="script"
#
# #  x30 ~ W @ geps12
#
# nmf_kwargs={'H': geps12.copy(),
#             'update_H': False,
#             'tol': _constants.NMF_TOLERANCE,
#             'n_iter': max_iter,
#             'beta_loss': beta_loss
#            }
#
# tens = torch.tensor(X30).to(device)
#
# W, H, n_iter = cnmf.nmf_torch(X30, nmf_kwargs, tens, verbose=True)
#
# final_loss = sknmf._beta_divergence(X30, W, H, beta_loss)
# print(f'Error per sample = {(final_loss / k_30.n_obs): .3e}')
#
# pfnmf_results['k12'] = {'W': W, 'H': H, 'n_iter': n_iter, 'final_loss': final_loss}
#
# del tens

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ### Decomposing K30 with geps12 and additional programs

# %% magic_args="--no-raise-error false" language="script"
#
# # pfnmf is written for constant W_1, so we will transpose as needed:
# # x30 ~ W_1 @ geps12 + W_2 @ H_2  <--> x30.T ~ geps12.T @ W_1.T + H_2.T @ W_2.T
#
# for added_rank in range(1, 5):
#     print(f"Working on added rank = {added_rank}")
#     
#     best_loss = np.infty
#     
#     for repeat in range(20): 
#         w1, h1, w2, h2, n_iter = pfnmf.pfnmf(X30.T, geps12.T, rank_2=added_rank, beta_loss=beta_loss,
#             tol=_constants.NMF_TOLERANCE, max_iter=max_iter, verbose=False)
#
#         final_loss = pfnmf.calc_beta_divergence(X30.T, w1, w2, h1, h2, beta_loss)
#         
#         if final_loss <= best_loss:
#             best_loss = final_loss
#             pfnmf_results[f'k12e{added_rank}'] = {'w1': w1, 'h1': h1, 'w2': w2, 'h2': h2, 'n_iter': n_iter, 'final_loss': final_loss}
#
#             print(f"repeat {repeat}, after {n_iter} iterations reached {final_loss: .4e}"
#                  f", per sample loss = {(final_loss / k_30.n_obs): .3e}")

# %% magic_args="--no-raise-error false" language="script"
#
# np.savez_compressed(results_dir.joinpath('marjanovic_k12_k30_pfnmf.npz'), pfnmf_results=pfnmf_results)

# %%
loaded = np.load(results_dir.joinpath('marjanovic_k12_k30_pfnmf.npz'), allow_pickle=True)
print([key for key in loaded.keys()])

# %%
pfnmf_results = loaded['pfnmf_results'].item()

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Evaluating the added programs

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true
# ### Preparing plotting parameters

# %%
columns_k12 = [f'k12.p{i}' for i in range(geps12.shape[0])]
sname = k_30.uns["sname"]

coloring_scheme = {'de_novo': '#d62728', 'k12': '#2ca02c', 'k12e1': 'limegreen', 'k12e2': 'yellow', 'k12e3': 'orange', 'k12e4': 'chocolate'}

for key in pfnmf_results.keys():
    print(f"{key}\t{pfnmf_results[key]['final_loss'] / 505: .0f}")


# %%
for dict_key, short_name in [('de_novo', 'k30'), ('k12', 'k12')]:
    res_dict = pfnmf_results[dict_key]
    
    res_dict['rank'] = rank_k12
    
    res_dict['norm_usage'] = res_dict['W'] / \
        np.linalg.norm(res_dict['W'], 1, axis=1, keepdims=True)
    
    res_dict['prog_percent'] = res_dict['norm_usage'].sum(axis=0) * 100 / k_30.n_obs

    res_dict['prog_name'] = [f'{short_name}.p{i}' for i in range(rank_k12)]
    
    res_dict['prog_label_2l'] = [name + f'\n({res_dict["prog_percent"][i]: 0.1f}%)' for i, name in enumerate(res_dict['prog_name'])]
    res_dict['prog_label_1l'] = [name + f' ({res_dict["prog_percent"][i]: 0.1f}%)' for i, name in enumerate(res_dict['prog_name'])]   

# %%
for index, dict_key in enumerate(['k12e1', 'k12e2', 'k12e3', 'k12e4']):
    added_rank = index + 1
    
    res_dict = pfnmf_results[dict_key]
    
    res_dict['rank'] = rank_k12 + added_rank
    
    usages = np.concatenate([res_dict['h1'], res_dict['h2']], axis=0).T
    
    res_dict['norm_usage'] = usages / np.linalg.norm(usages, 1, axis=1, keepdims=True)
    
    res_dict['prog_percent'] = res_dict['norm_usage'].sum(axis=0) * 100 / k_30.n_obs

    res_dict['prog_name'] = [f'k12e{added_rank}.p{i}' for i in range(rank_k12)]
    res_dict['prog_name'].extend([f'k12e{added_rank}.e{i}' for i in range(added_rank)])
    
    res_dict['prog_label_2l'] = [name + f'\n({res_dict["prog_percent"][i]: 0.1f}%)' for i, name in enumerate(res_dict['prog_name'])]
    res_dict['prog_label_1l'] = [name + f' ({res_dict["prog_percent"][i]: 0.1f}%)' for i, name in enumerate(res_dict['prog_name'])]   

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true
# ### usages clustermaps

# %%
dict_key = 'de_novo'
res_dict = pfnmf_results[dict_key]

title = f'K_30 normalized usages of de-novo GEPs, k={rank_k12}'

un_sns = _utils.plot_usages_norm_clustermaps(k_30, normalized_usages=res_dict['norm_usage'],
    columns=res_dict['prog_label_2l'], title=title, show=True, sns_clustermap_params={'col_colors': [coloring_scheme[dict_key]] * res_dict['rank']})

# %%
dict_key = 'k12'
res_dict = pfnmf_results[dict_key]

title = f'K_30 normalized usages of k_12 GEPs, k={rank_k12}'

un_sns = _utils.plot_usages_norm_clustermaps(k_30, normalized_usages=res_dict['norm_usage'],
    columns=res_dict['prog_label_2l'], title=title, show=True, sns_clustermap_params={'col_colors': [coloring_scheme[dict_key]] * res_dict['rank']})

# %% [markdown]
# Creating expanded usages DataFrame for added rank 1

# %%
for dict_key in ['k12e1', 'k12e2', 'k12e3', 'k12e4']:
    res_dict = pfnmf_results[dict_key]
    
    title = f'K_30 normalized usages of k_12 GEPs + {res_dict["rank"] - rank_k12} novel'
    
    un_sns = _utils.plot_usages_norm_clustermaps(k_30, normalized_usages=res_dict['norm_usage'],
        columns=res_dict['prog_label_2l'], title=title, show=True, sns_clustermap_params={'col_colors': [coloring_scheme[dict_key]] * res_dict['rank']})


# %%
title = f'K_30 normalized usages of de-novo GEPs and k_12 GEPs'

dict_key0 = 'k12'
res_dict0 = pfnmf_results[dict_key0]

dict_key1 = 'de_novo'
res_dict1 = pfnmf_results[dict_key1]

joint_usages = np.concatenate([res_dict0['norm_usage'], res_dict1['norm_usage']], axis=1)

joint_labels = res_dict0['prog_label_2l'] + res_dict1['prog_label_2l']

joint_colors = [coloring_scheme[dict_key0]] * res_dict0['rank'] + [coloring_scheme[dict_key1]] * res_dict1['rank']

un_sns = _utils.plot_usages_norm_clustermaps(k_30, normalized_usages=joint_usages, columns=joint_labels,
                                             title=title, show=True, sns_clustermap_params={'col_colors': joint_colors})

# %%
dict_key0 = 'de_novo'
res_dict0 = pfnmf_results[dict_key0]

for dict_key1 in ['k12e1', 'k12e2', 'k12e3', 'k12e4']:
    res_dict1 = pfnmf_results[dict_key1]
    
    title = f'K_30 normalized usages of de-novo GEPs and k_12 GEPs + {res_dict1["rank"] - rank_k12} novel'

    joint_usages = np.concatenate([res_dict0['norm_usage'], res_dict1['norm_usage']], axis=1)

    joint_labels = res_dict0['prog_label_2l'] + res_dict1['prog_label_2l']

    joint_colors = [coloring_scheme[dict_key0]] * res_dict0['rank'] + [coloring_scheme[dict_key1]] * res_dict1['rank']

    un_sns = _utils.plot_usages_norm_clustermaps(k_30, normalized_usages=joint_usages, columns=joint_labels,
                                                 title=title, show=True, sns_clustermap_params={'col_colors': joint_colors})


# %% [markdown]
# apply the usages clustermap function

# %%
title = f'K_30 normalized usages of de-novo GEPs and k_12 GEPs {rank_k12} + [1, 2, 3] novel'

dict_key0 = 'de_novo'
res_dict0 = pfnmf_results[dict_key0]

joint_usages = res_dict0['norm_usage'].copy()
joint_labels = res_dict0['prog_label_1l'].copy()
joint_colors = [coloring_scheme[dict_key0]] * res_dict0['rank']

for dict_key1 in ['k12', 'k12e1', 'k12e2', 'k12e3']:
    res_dict1 = pfnmf_results[dict_key1]
    
    joint_usages = np.concatenate([joint_usages, res_dict1['norm_usage']], axis=1)

    joint_labels.extend(res_dict1['prog_label_1l'])

    joint_colors.extend([coloring_scheme[dict_key1]] * res_dict1['rank'])

    
un_sns = _utils.plot_usages_norm_clustermaps(k_30, normalized_usages=joint_usages, columns=joint_labels, title=title,
                                             show=True, sns_clustermap_params={'col_colors': joint_colors, 'figsize': (13, 13)})


# %%
n_programs = joint_usages.shape[1]

pearson_corr = np.corrcoef(joint_usages.T)

un_sns = _utils.sns.clustermap(pd.DataFrame(pearson_corr, index=joint_labels, columns=joint_labels),
                               figsize=(4 + n_programs * 0.43, 4 + n_programs * 0.41),
                               row_colors=joint_colors, col_colors=joint_colors)

un_sns.figure.suptitle('Correlation of GEP usages', fontsize=40, y=1.02)
plt.show()

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ### Extracting usage coefficients over all genes

# %%
z_layer = 'cpm_log1p_zscore'

k_30.layers[z_layer] = sc.pp.normalize_total(k_30, target_sum=1e6, inplace=False)['X']
sc.pp.log1p(k_30, layer=z_layer)
sc.pp.scale(k_30, layer=z_layer)

# %%
for dict_key in pfnmf_results.keys():
    res_dict = pfnmf_results[dict_key]
    
    res_dict['gene_coefs'] = pd.DataFrame(
        _utils.fastols(res_dict['norm_usage'], k_30.layers[z_layer]).T,
        index=k_30.var.index,
        columns=res_dict['prog_name'])


# %%
res_dict['gene_coefs']

# %% [markdown] tags=[]
# ### Running GO

# %%
gp = _utils.MyGProfiler(organism='hsapiens', sources=['GO:BP', 'WP', 'REAC', 'KEGG'])

genes_list = adata.var.loc[k_30.var.index, 'geneSymbol'].to_list()


# %% magic_args="--no-raise-error false" language="script"
#
# program_go_dir = _utils.set_dir(notebook_dir.joinpath('programs_GSEA'))
#
# for dict_key in pfnmf_results.keys():
#     res_dict = pfnmf_results[dict_key]
#     
#     for index in range(res_dict['rank']):
#         ordered_genes_index = res_dict['gene_coefs'].nlargest(
#             columns=[res_dict['prog_name'][index]],
#             n=1000).index
#         ordered_genes = adata.var.loc[ordered_genes_index, 'geneSymbol'].to_list()
#         
#         go_enrichment = gp.profile( 
#             ordered_genes, ordered=True, background=genes_list)
#
#         go_enrichment.to_csv(
#             program_go_dir.joinpath(f"{res_dict['prog_name'][index]}.csv"))
#


# %%
joint_HVG_geneID = set(joint_K12_K30_var[joint_K12_K30_var.highly_variable].index)
k_30_HVG_geneID = set(k_30.var[k_30.var.highly_variable].index)
union_HVG_geneID = k_30_HVG_geneID.union(joint_HVG_geneID)

for dict_key in pfnmf_results.keys():
    res_dict = pfnmf_results[dict_key]
    for index in range(res_dict['rank']):
        
        
        ordered_geneID = set(res_dict['gene_coefs'].nlargest(
            columns=[res_dict['prog_name'][index]], n=1000).index)
        print(res_dict['prog_name'][index],
              len(set.intersection(ordered_geneID, joint_HVG_geneID)),
              len(set.intersection(ordered_geneID, k_30_HVG_geneID)),
              len(set.intersection(ordered_geneID, union_HVG_geneID))
             )


# %%
# # %%script --no-raise-error false

program_go_dir = _utils.set_dir(notebook_dir.joinpath('programs_GSEA_HVG'))

for dict_key in pfnmf_results.keys():
    res_dict = pfnmf_results[dict_key]
    
    for index in range(res_dict['rank']):
        ordered_genes_index = res_dict['gene_coefs'].nlargest(
            columns=[res_dict['prog_name'][index]],
            n=1000).index
        
        ordered_genes_index = ordered_genes_index[ordered_genes_index.isin(joint_HVG_geneID)]
        
        ordered_genes = adata.var.loc[ordered_genes_index, 'geneSymbol'].to_list()
        
        go_enrichment = gp.profile( 
            ordered_genes, ordered=True, background=genes_list)

        go_enrichment.to_csv(
            program_go_dir.joinpath(f"{res_dict['prog_name'][index]}.csv"))
        
#         break
#     break
# go_enrichment

# %% [markdown] tags=[]
# ### Calculating truncated spearman correlation
#

# %%
keys = ['k12', 'k12e1', 'k12e2', 'k12e3', 'de_novo']

concatenated_spectras = pd.concat([
    pfnmf_results[dict_key]['gene_coefs'].copy() for dict_key in keys], axis=1)

n_genes, n_programs = concatenated_spectras.shape


ranked_coefs = n_genes - rankdata(concatenated_spectras, axis=0)

ranked_coefs[ranked_coefs > _constants.N_COMPARED_RANKED] = _constants.N_COMPARED_RANKED

spearman_corr = np.corrcoef(ranked_coefs, rowvar=False)

# spearman figure
fig, ax = plt.subplots(figsize=(4 + ranked_coefs.shape[1] * 0.43,
                                4 + ranked_coefs.shape[1] * 0.41))

_utils.heatmap_with_numbers(
    spearman_corr, ax=ax, param_dict={'vmin': 0, 'vmax': 1})

ax.xaxis.tick_bottom()
ax.set_xticklabels(concatenated_spectras.columns, rotation='vertical')
ax.set_yticklabels(concatenated_spectras.columns)
ax.set_title(f'{_constants.N_COMPARED_RANKED}-Truncated Spearman Correlation',
             size=25, y=1.05, x=0.43)

plt.show(fig)

# fig.savefig(RESULTS_DIR.joinpath(
#     f'correlation_spearman_{utils.N_COMPARED_RANKED}_truncated.png'),
#     dpi=180, bbox_inches='tight')

# plt.close(fig)


# %%
# correlation histogram
fig, ax = plt.subplots(figsize=(6, 5))

plt.hist(spearman_corr[np.triu_indices_from(spearman_corr, k=1)],
         bins=np.linspace(-1, 1, 41))
ax.set_title('Spearman correlation distribution')
plt.show()

# fig.savefig(RESULTS_DIR.joinpath('correlation_histogtam_pearson.png'),
#             dpi=180, bbox_inches='tight')

# plt.close(fig)

# %%
threshold = 0.17

# maping adata short name to layer number
name_map = dict(zip(keys, range(len(keys))))
name_map.update({'k30': name_map.pop('de_novo')})  # for the de_novo
ks = [pfnmf_results[key]['rank'] for key in keys]

# adjacency matrix creation and filtering
adj_df = pd.DataFrame(np.round((spearman_corr), 2),
                      index=concatenated_spectras.columns,
                      columns=concatenated_spectras.columns)

# order
linkage = hierarchy.linkage(
    adj_df, method='average', metric='euclidean')
prog_order = hierarchy.leaves_list(
    hierarchy.optimal_leaf_ordering(linkage, adj_df))

np.fill_diagonal(adj_df.values, 0)
# adj_df.values[adj_df.values <= 0.0] = 0

# keeping only edges between consecutive layers
for i in range(len(ks) - 2):
    adj_df.values[:np.sum(ks[:i + 1]), np.sum(ks[:i + 2]):] = 0
    adj_df.values[np.sum(ks[:i + 2]):, :np.sum(ks[:i + 1])] = 0

adj_df.values[adj_df.values <= threshold] = 0
print(f'Number of edges={np.count_nonzero(adj_df)}')

# ordering the nodes for display
adj_df = adj_df.iloc[prog_order, prog_order]

# create the graph object
G = nx.from_numpy_array(adj_df.values, create_using=nx.Graph)
nx.relabel_nodes(G, lambda i: adj_df.index[i], copy=False)
nx.set_node_attributes(
    G, {node: name_map[node.split('.')[0]] for node in G.nodes}, name='layer')

# prepare graph for display
layout = nx.multipartite_layout(G, subset_key='layer')

edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
edge_width = 15 * np.power(weights, 2)  # visual edge emphesis


for layer in {data['layer'] for key, data in G.nodes.data()}:
    nodes = [node for node in G.nodes if name_map[node.split('.')[0]] == layer]

    angles = np.linspace(-np.pi / 4, np.pi / 4, len(nodes))
    
    for i, node in enumerate(nodes):
        layout[node] = [layer + 2 * np.cos(angles[i]), np.sin(angles[i])]

fig, ax = plt.subplots(1, 1, figsize=(16.4, 19.2), dpi=180)
nx.draw(G, layout, node_size=3000, with_labels=False, edge_color=weights,
        edge_vmin=threshold, edge_vmax=1., width=edge_width, ax=ax)

cmp = plt.matplotlib.cm.ScalarMappable(plt.matplotlib.colors.Normalize(vmin=threshold, vmax=1))
plt.colorbar(cmp, orientation='horizontal', cax=fig.add_subplot(15, 5, 71))

# change color of layers
for key in keys:
    nx.draw_networkx_nodes(
        G, layout, node_color=coloring_scheme[key], node_size=2800,
        nodelist=pfnmf_results[key]['prog_name'], ax=ax)
nx.draw_networkx_labels(G, layout, font_size=11, ax=ax)

ax.set_title(f'Timepoint correlation graph, correlation threshold={threshold}',
             {'fontsize': 25})
plt.show()

# %%
