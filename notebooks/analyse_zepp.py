# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
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
# # Downloading, pre-processing and running cNMF on Zepp et. al 2021 data
# 1. Obtaining the AnnData object and complementary metadata
#
# 2. filtering genes, selecting joint highly variable genes (HVGs) and showing key statistics
# 3. Splitting the dataset by developmental stage, and selecting HVG per stage
# 3. Running consensus NMF (cNMF) per stage
# 4. Selecting parameters for the cNMF 
#
#

# %%
# %%time
# %load_ext autoreload
# %autoreload 2

#debug:
from importlib import reload

import sys
import os

import numpy as np
import pandas as pd
from scipy import sparse
import scanpy as sc
import matplotlib.pyplot as plt

from gepdynamics import _utils
from gepdynamics import _constants
from gepdynamics import cnmf

_utils.cd_proj_home()
print(os.getcwd())
# os.chdir('/cs/labs/mornitzan/yotamcon/gep-dynamics')


# %% [markdown]
# ### Downloading and loading AnnData object
# The adata contains log1p(CP10K) data, so we keep the transformed data in `raw`, and add the original counts as `X`

# %%
results_dir = _utils.set_dir('results_zepp')
data_dir = _utils.set_dir('data')
GSE_dir = _utils.set_dir(data_dir.joinpath('GSE149563'))


# %% magic_args="--no-raise-error false # remove this to run the downloading" language="script"
#
# # Adata downloaded from https://data-browser.lungmap.net/explore/projects/00f056f2-73ff-43ac-97ff-69ca10e38c89/get-curl-command
# # by running this for the adata: 
# !(cd {GSE_dir.as_posix()} && curl --location --fail 'https://service.azul.data.humancellatlas.org/manifest/files?catalog=lm3&format=curl&filters=%7B%22fileFormat%22%3A+%7B%22is%22%3A+%5B%22h5ad%22%5D%7D%2C+%22projectId%22%3A+%7B%22is%22%3A+%5B%2200f056f2-73ff-43ac-97ff-69ca10e38c89%22%5D%7D%2C+%22genusSpecies%22%3A+%7B%22is%22%3A+%5B%22Mus+musculus%22%5D%7D%7D&objectKey=manifests%2Fe42d976a-5137-5422-be32-39008e1d53d7.1ad7b2a4-0d0f-55d3-9d0c-6c37e8d46dc8.curlrc' | curl --config - )
# # and then running this for the metadata: 
# !(cd {GSE_dir.as_posix()} && curl --location --fail 'https://service.azul.data.humancellatlas.org/manifest/files?catalog=lm3&format=curl&filters=%7B%22fileFormat%22%3A+%7B%22is%22%3A+%5B%22csv%22%5D%7D%2C+%22projectId%22%3A+%7B%22is%22%3A+%5B%2200f056f2-73ff-43ac-97ff-69ca10e38c89%22%5D%7D%2C+%22genusSpecies%22%3A+%7B%22is%22%3A+%5B%22Mus+musculus%22%5D%7D%7D&objectKey=manifests%2Fed538a08-689b-530d-a661-e1756132b883.1ad7b2a4-0d0f-55d3-9d0c-6c37e8d46dc8.curlrc' | curl --config -)
#
# download_dir = GSE_dir.joinpath('a078a6cb-a72a-305c-80df-cf35aedd01ff')
# ! mv {download_dir.as_posix()}/* {GSE_dir.as_posix()}
# ! rmdir {download_dir.as_posix()}

# %%
# %%time

# %time adata = sc.read(GSE_dir.joinpath('JZ_Mouse_TimeSeries.h5ad'))
metadata = pd.read_csv(GSE_dir.joinpath('AllTimePoints_metadata.csv'), index_col=0)

adata.obs['celltype'] = metadata.var_celltype
adata.obs['compartment'] = metadata.var_compartment

adata.raw = adata
adata.X = (sparse.csc_matrix(adata.obs.n_molecules.values[:, None] / 10_000).multiply(adata.X.expm1())).rint()
adata


# %%
adata.obs.development_stage = adata.obs.development_stage.cat.rename_categories(
    {'Adult': 'P42', 'E12.5': 'E12', 'E15.5': 'E15', 'E17.5': 'E17'}).cat.reorder_categories(
    ['E12', 'E15', 'E17', 'P3', 'P7', 'P15', 'P42'])

# %%
sc.pl.umap(adata, color=['development_stage', 'compartment'])
sc.pl.umap(adata, color=['celltype'])

adata.uns['development_stage_colors_dict'] = dict(zip(adata.obs['development_stage'].cat.categories, adata.uns['development_stage_colors']))
adata.uns['compartment_colors_dict'] = dict(zip(adata.obs['compartment'].cat.categories, adata.uns['compartment_colors']))
adata.uns['celltype_colors_dict'] = dict(zip(adata.obs['celltype'].cat.categories, adata.uns['celltype_colors']))

# %%
pd.crosstab(adata.obs.development_stage, adata.obs.compartment)

# %%
pd.crosstab(adata.obs.celltype, adata.obs.development_stage)

# %%
pd.crosstab(adata.obs.celltype, adata.obs.compartment)

# %% [markdown]
# ### Filter genes and plot basic statistics

# %%
# %%time

# filtering genes with very low abundance
sc.pp.filter_genes(adata, min_cells=np.round(adata.shape[0] / 1000))

# getting general statistics for counts abundance
sc.pp.filter_genes(adata, min_counts=0)
sc.pp.filter_cells(adata, min_counts=0)
sc.pp.filter_cells(adata, min_genes=0)

sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=_constants.NUMBER_HVG)
adata

# %%
column_of_interest = 'development_stage'

stats_df = adata.obs.loc[:, [column_of_interest, 'n_genes', 'n_counts']].groupby(
    [column_of_interest]).median()

stats_df = pd.concat([adata.obs.groupby([column_of_interest]).count().iloc[:, 0],
                      stats_df], axis=1)
stats_df.columns = ['# cells', 'median # genes', 'median # counts']

stats_df.plot(kind='bar', title=f'{column_of_interest} statistics', log=True, ylim=((5e2, 2e5)))
plt.show()
del column_of_interest, stats_df

# %% [markdown]
# ### Splitting the adata by "development_stage", and creating a normalized variance layer

# %%
# %%time

stages = adata.obs.development_stage.cat.categories
split_adatas_dir = _utils.set_dir(results_dir.joinpath('split_adatas'))

for stage in stages:
    print(f'working on stage {stage}')
    if not split_adatas_dir.joinpath(f'{stage}.h5ad').exists():
        tmp = adata[adata.obs.development_stage == stage].copy()

        tmp.uns['name'] = f'{stage}'   # full name
        tmp.uns['sname'] = f'{stage}'  # short name, here it is the same

        # correcting the gene counts
        sc.pp.filter_genes(tmp, min_cells=0)
        sc.pp.filter_genes(tmp, min_counts=0)

        # calculating per sample HVGs
        sc.pp.highly_variable_genes(tmp, flavor='seurat_v3', n_top_genes=_constants.NUMBER_HVG)

        tmp.write_h5ad(split_adatas_dir.joinpath(f'{stage}.h5ad'))

        del tmp


# %% [markdown]
# ### Running multiple NMF iterations

# %%
# %%time

cnmf_dir = _utils.set_dir(results_dir.joinpath('cnmf'))

ks = [22, 23, 24, 25, 26, 27, 28, 29, 30]#[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

for stage in stages:
    print(f'Starting on {stage}')
    tmp = sc.read_h5ad(split_adatas_dir.joinpath(f'{stage}.h5ad'))
    
    c_object = cnmf.cNMF(cnmf_dir, stage)
    
    # Variance normalized version of the data
    X = sc.pp.scale(tmp.X[:, tmp.var.highly_variable].toarray().astype(np.float32), zero_center=False)
    
    c_object.prepare(X, ks, n_iter=100, new_nmf_kwargs={'tol': _constants.NMF_TOLERANCE,
                                                        'beta_loss': 'kullback-leibler'})
    
    c_object.factorize(0, 1, gpu=True)
    
    # c_object.combine()
    
    del tmp, X
    


# %%
# %%time
for stage in stages:
    print(f'Starting on {stage}')
    c_object = cnmf.cNMF(cnmf_dir, stage)
    for thresh in [0.5, 0.4, 0.3]:
        print(f'working on threshold {thresh}')
        c_object.k_selection_plot(density_threshold=thresh, nmf_refitting_iters=500,
                                  close_fig=True, show_clustering=True, gpu=True)
    


# %%
stage = 'P3'
thresh = 0.45
print(f'Starting on {stage}')
c_object = cnmf.cNMF(cnmf_dir, stage)
print(f'working on threshold {thresh}')
c_object.k_selection_plot(density_threshold=thresh, nmf_refitting_iters=500,
                          close_fig=True, show_clustering=True, gpu=True)


# %% [markdown]
# ### Selecting the decomposition rank utilizing K-selection plots and PCA variance explained

# %%
# %%time
df_var = pd.DataFrame()
df_cumulative_var = pd.DataFrame()

n_components = 50

for stage in stages:
    
    # %time tmp = sc.read_h5ad(split_adatas_dir.joinpath(f'{stage}.h5ad'))
    
    a, b, c, d, = sc.tl.pca(tmp.X[:, tmp.var.highly_variable], n_comps=n_components, return_info=True)

    df_var[f'{stage}'] = c*100
    df_cumulative_var[f'{stage}'] = c.cumsum()*100
    
#     print(f'{stage} - cummulative variance percentages:'),
#     for i in range(n_components):
#         print(f'{c[i]*100: .2f}', end='\t')
#     print()
#     for i in range(n_components):
#         print(f'{c.cumsum()[i]*100: .2f}', end='\t')
#     print()

# %%
plt.plot(range(50), df_var, label=df_var.columns)
plt.title('Variance explained')
plt.legend()
plt.yscale('log')
plt.show()

# %%
plt.plot(df_cumulative_var.index, 100-df_cumulative_var, label=df_var.columns)
plt.yscale('log')
plt.title(f' 1- CDF variance explained')
plt.legend()
plt.show()

# %%
# %%time

selected_cnmf_params = {
    'E12': (16, 0.4),  # 
    'E15': (18, 0.4),  # 
    'E17': (15, 0.4),    # 
    'P3': (16, 0.5),    # rank 16 has a split that improves loss and stability
    'P7': (18, 0.4),   # 
    'P15': (13, 0.4),  # 
    'P42': (13, 0.4)}   # 

split_adatas = {}

for stage, (k, threshold) in selected_cnmf_params.items():
    print(f'Working on {stage} with k={k} and threshold={threshold}')
    # %time tmp = sc.read_h5ad(split_adatas_dir.joinpath(f'{stage}.h5ad'))

    c_object = cnmf.cNMF(cnmf_dir, stage)
    c_object.consensus(k, density_threshold=threshold, gpu=True, verbose=True,
                       nmf_refitting_iters=1000, show_clustering=False)

    usages, spectra = c_object.get_consensus_usages_spectra(k, density_threshold=threshold)

    tmp.uns['cnmf_params'] = {'k_nmf': k, 'threshold': threshold}

    tmp.obsm['usages'] = usages.copy()

    usages_norm = usages / np.sum(usages, axis=1, keepdims=True)
    tmp.obsm['usages_norm'] = usages_norm

    # get per gene z-score of data after TPM normalization and log1p transformation 
    tpm_log1p_zscore = tmp.X.toarray()
    tpm_log1p_zscore /= 1e-6 * np.sum(tpm_log1p_zscore, axis=1, keepdims=True)
    tpm_log1p_zscore = np.log1p(tpm_log1p_zscore)
    tpm_log1p_zscore = sc.pp.scale(tpm_log1p_zscore)

    usage_coefs = _utils.fastols(usages_norm, tpm_log1p_zscore)

    tmp.varm['usage_coefs'] = pd.DataFrame(
        usage_coefs.T, index=tmp.var.index,
        columns=[f'{tmp.uns["sname"]}.p{prog}' for prog in range(usages.shape[1])])
    
    split_adatas[stage] = tmp

    tmp.write_h5ad(split_adatas_dir.joinpath(f'{stage}_GEPs.h5ad'))


# %% Loading GEPs adatas
# %%time
times = adata.obs.timesimple.cat.categories
split_adatas_dir = _utils.set_dir(results_dir.joinpath('marjanovic_mmLungPlate_split'))

split_adatas = {}
for time in times:
    split_adatas[time] = sc.read_h5ad(split_adatas_dir.joinpath(f'{time}_GEPs.h5ad'))

# %% [markdown]
# ### Examening results

# %%
for time in times:
    print(time)
    s = split_adatas[time].obsm['usages_norm'].sum(axis=0)
    with np.printoptions(precision=2, suppress=False):
        print(s * 100 / s.sum())

# %%
# Plotting normalized usages clustermap
for time in times:
    tmp = split_adatas[time]
    sname = tmp.uns["sname"]
    k = tmp.obsm["usages"].shape[1]

    un_sns = _utils.plot_usages_norm_clustermaps(tmp, show=True)
    un_sns.savefig(results_dir.joinpath(
        sname, f'{sname}.usages_norm.k_{k}.png'),
        dpi=180, bbox_inches='tight')
    plt.close(un_sns.fig)

    _utils.plot_usages_norm_violin(
        tmp, 'clusterK12', save_path=results_dir.joinpath(
            sname, f'{sname}_norm_usage_per_lineage_k_{k}.png'))




# %% [markdown]
# ## ToDo - plot all programs on phate
# ## ToDo - consider adding "main program" per cell and plot it on joint tsne and phate

# %%
for time in times:
    tmp = split_adatas[time]
    plt.scatter(tmp.obsm['X_phate'][:, 0], tmp.obsm['X_phate'][:, 1], c=tmp.obsm['usages_norm'][:, 0])
    plt.title(f'{tmp.uns["name"]} Program 0 on Phate coordinates')
    plt.show()
    plt.close()

# %%

# sub-setting the genes:
mutual_hvg = list(set().union(*[tmp.var[tmp.var.highly_variable].index \
                          for time, tmp in split_adatas.items()]))
mutual_hvg = list(set(mutual_hvg).intersection(*[tmp.var[tmp.var.highly_variable].index \
                          for time, tmp in split_adatas.items()]))
# mutual_hvg = adata.var[adata.var.highly_variable].index

concatenated_spectras = pd.concat([
    tmp.varm['usage_coefs'].loc[mutual_hvg].copy() for time, tmp in split_adatas.items()], axis=1)

n_genes, n_programs = concatenated_spectras.shape

pearson_corr = np.corrcoef(concatenated_spectras.T)

# cosine figure
fig, ax = plt.subplots(figsize=(4 + n_programs * 0.43, 4 + n_programs * 0.41))

_utils.heatmap_with_numbers(
    pearson_corr, ax=ax, param_dict={'vmin': 0, 'vmax': 1})

ax.xaxis.tick_bottom()
ax.set_xticklabels(concatenated_spectras.columns, rotation='vertical')
ax.set_yticklabels(concatenated_spectras.columns)
ax.set_title('Pearson correlation',
             size=25, y=1.05, x=0.43)

fig.savefig(results_dir.joinpath('correlation_pearson.png'),
            dpi=180, bbox_inches='tight')
plt.close(fig)


# correlation histogram
fig, ax = plt.subplots(figsize=(6, 5))

plt.hist(pearson_corr[np.triu_indices_from(pearson_corr, k=1)],
         bins=np.linspace(-1, 1, 41))
ax.set_title('Pearson correlation distribution')
plt.show()

fig.savefig(results_dir.joinpath('correlation_histogtam_pearson.png'),
            dpi=180, bbox_inches='tight')

plt.close(fig)

# 3.1b Calculating spearman correlation between usages coefficient

from scipy.stats import rankdata
N_COMPARED_RANKED = 1000

ranked_coefs = n_genes - rankdata(concatenated_spectras, axis=0)

ranked_coefs[ranked_coefs > N_COMPARED_RANKED] = N_COMPARED_RANKED

spearman_corr = np.corrcoef(ranked_coefs, rowvar=False)

# spearman figure
fig, ax = plt.subplots(figsize=(4 + ranked_coefs.shape[1] * 0.43,
                                4 + ranked_coefs.shape[1] * 0.41))

_utils.heatmap_with_numbers(
    spearman_corr, ax=ax, param_dict={'vmin': 0, 'vmax': 1})

ax.xaxis.tick_bottom()
ax.set_xticklabels(concatenated_spectras.columns, rotation='vertical')
ax.set_yticklabels(concatenated_spectras.columns)
ax.set_title(f'{N_COMPARED_RANKED}-Truncated Spearman Correlation',
             size=25, y=1.05, x=0.43)

fig.savefig(results_dir.joinpath(
    f'correlation_spearman_{N_COMPARED_RANKED}_truncated.png'),
    dpi=180, bbox_inches='tight')

plt.close(fig)

# correlation histogram
fig, ax = plt.subplots(figsize=(6, 5))

plt.hist(spearman_corr[np.triu_indices_from(spearman_corr, k=1)],
         bins=np.linspace(0, 1, 21))
ax.set_title('Spearman Correlation distribution')
plt.show()

fig.savefig(results_dir.joinpath(
    f'correlation_histogtam_spearman_{N_COMPARED_RANKED}_truncated.png'),
    dpi=180, bbox_inches='tight')

plt.close(fig)

# %%

import matplotlib as mpl
import networkx as nx
from scipy.cluster import hierarchy

threshold = 0.2

# maping adata short name to layer number
name_map = {}
for i, time in enumerate(times):
    name_map['t'+time[:2]] = i + 1

ks = [tmp.obsm['usages_norm'].shape[1] for time, tmp in split_adatas.items()]

# adjacency matrix creation and filtering

adj_df = pd.DataFrame(np.round((spearman_corr + pearson_corr) / 2, 2),
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
nx.draw(G, layout, node_size=1500, with_labels=False, edge_color=weights,
        edge_vmin=threshold, edge_vmax=1., width=edge_width, ax=ax)

cmp = mpl.cm.ScalarMappable(mpl.colors.Normalize(vmin=threshold, vmax=1))
plt.colorbar(cmp, orientation='horizontal', cax=fig.add_subplot(15, 5, 71))

# change color of layers
for time, tmp in split_adatas.items():
    nx.draw_networkx_nodes(
        G, layout, node_color=adata.uns['timesimple_colors_dict'][time],
        node_size=1400, nodelist=[f'{tmp.uns["sname"]}.p{i}' for i in range(
            tmp.obsm['usages'].shape[1])], ax=ax)
nx.draw_networkx_labels(G, layout, font_size=11, ax=ax)

ax.set_title(f'Timepoint correlation graph, correlation threshold={threshold}',
             {'fontsize': 25})
plt.show()


fig.savefig(results_dir.joinpath(
    f'correlations_graph_threshold_{threshold}.png'),
    dpi=180, bbox_inches='tight')

plt.close()

del name_map, G, edges, weights, i, layer, angles, node, nodes
del fig, ax, layout, prog_order

