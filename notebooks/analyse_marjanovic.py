# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # Downloading, pre-processing and running cNMF on marjanovic et. al 2020 data
# 1. Obtaining the data and creating AnnData object
# 2. filtering genes and selecting joint highly variable genes (HVGs) and showing key statistics
# 3. Splitting the dataset by timepoints, and selecting HVG per timepoint
# 3. Running consensus NMF (cNMF) per timepoint
# 4. Selecting parameters for the cNMF 
#

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

print(os.getcwd())
os.chdir('/cs/labs/mornitzan/yotamcon/gep-dynamics')


# %% [markdown]
# ### Downloading or loading AnnData object

# %%
# %%time
results_dir = _utils.set_dir('results')
orig_adata_path = results_dir.joinpath('marjanovic_mmLungPlate.h5ad')

if not orig_adata_path.exists():  # create the original adata if it doesn't exist
    # directories for file download:
    data_dir = _utils.set_dir('data')
    GSE_dir = _utils.set_dir(data_dir.joinpath('GSE154989'))
    
    # GEO server prefix for mmLungPlate SubSeries GSE154989
    ftp_address = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE154nnn/GSE154989/suppl/'

    #filenames
    f_rawCount = GSE_dir.joinpath('GSE154989_mmLungPlate_fQC_dSp_rawCountOrig.h5')
    f_geneTable = GSE_dir.joinpath('GSE154989_mmLungPlate_fQC_geneTable.csv.gz')
    f_smpTable = GSE_dir.joinpath('GSE154989_mmLungPlate_fQC_smpTable.csv.gz')
    f_smp_annot = GSE_dir.joinpath('GSE154989_mmLungPlate_fQC_dZ_annot_smpTable.csv.gz')

    ftp_address = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE154nnn/GSE154989/suppl/'

    # downloading if needed:
    if not f_rawCount.exists():
        urlretrieve(ftp_address + f_rawCount.name, f_rawCount)
    
    if not f_geneTable.exists():
        urlretrieve(ftp_address + f_geneTable.name, f_geneTable)
    
    if not f_smpTable.exists():
        urlretrieve(ftp_address + f_smpTable.name, f_smpTable)
    
    if not f_smp_annot.exists():
        urlretrieve(ftp_address + f_smp_annot.name, f_smp_annot)
    
    # reading the files
    sparse_counts = _utils.read_matlab_h5_sparse(f_rawCount)
    
    gene_ids = pd.read_csv(f_geneTable, index_col=0)
    smp_ids = pd.read_csv(f_smpTable, index_col=0)
    smp_annotation = pd.read_csv(f_smp_annot, index_col=0)
    
    # constructing the adata
    adata = sc.AnnData(X=sparse_counts, dtype=np.float32, var=gene_ids, obs=smp_ids)
    
    adata.obs['clusterK12'] = smp_annotation.clusterK12.astype('category')
    
    adata.obsm['X_tsne'] = smp_annotation[['tSNE_1', 'tSNE_2']].values
    adata.obsm['X_phate'] = smp_annotation[['phate_1', 'phate_2']].values
    adata.write(orig_adata_path)
else:
    adata = sc.read(orig_adata_path)

adata
# adata_norm_depth = sc.read(data_dir.joinpath('marjanovic_mmLungPlate_depth_normalized.h5ad'))



# %%
sc.external.pl.phate(adata, color=['timesimple', 'clusterK12'])
adata.uns['timesimple_colors_dict'] = dict(zip(adata.obs['timesimple'].cat.categories, adata.uns['timesimple_colors']))
adata.uns['clusterK12_colors_dict'] = dict(zip(adata.obs['clusterK12'].cat.categories, adata.uns['clusterK12_colors']))

# %%
pd.crosstab(adata.obs.timesimple, adata.obs.clusterK12)

# %% [markdown]
# ### Filter genes and plot basic statistics

# %%
# filtering genes with very low abundance
sc.pp.filter_genes(adata, min_cells=np.round(adata.shape[0] / 1000))
sc.pp.filter_genes(adata, min_counts=40)

# getting general statistics for counts abundance
sc.pp.filter_cells(adata, min_counts=0)
sc.pp.filter_cells(adata, min_genes=0)

sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=_constants.NUMBER_HVG)
adata

# %%
column_of_interest = 'timesimple'

stats_df = adata.obs.loc[:, [column_of_interest, 'n_genes', 'n_counts']].groupby(
    [column_of_interest]).median()

stats_df = pd.concat([adata.obs.groupby([column_of_interest]).count().iloc[:, 0],
                      stats_df], axis=1)
stats_df.columns = ['# cells', 'median # genes', 'median # counts']

stats_df.plot(kind='bar', title=f'{column_of_interest} statistics', log=True, ylim=((1e2, 2e6)))

del column_of_interest, stats_df

# %% [markdown]
# ### Splitting the adata by "timesimple", and creating a normalized variance layer

# %%
# %%time

times = adata.obs.timesimple.cat.categories
split_adatas_dir = _utils.set_dir(results_dir.joinpath('marjanovic_mmLungPlate_split'))

for time in times:
    if not split_adatas_dir.joinpath(f'{time}.h5ad').exists():
        tmp = adata[adata.obs.timesimple == time].copy()

        tmp.uns['name'] = f'{time}'   # full name
        tmp.uns['sname'] = f't{time[:2]}'  # short name

        # correcting the gene counts
        sc.pp.filter_genes(tmp, min_cells=0)
        sc.pp.filter_genes(tmp, min_counts=0)
        
        # calculating per sample HVGs
        sc.pp.highly_variable_genes(tmp, flavor='seurat_v3', n_top_genes=_constants.NUMBER_HVG)
        
        tmp.write_h5ad(split_adatas_dir.joinpath(f'{time}.h5ad'))

        del tmp


# %% [markdown]
# ### Running multiple NMF iterations

# %%
# %%time

cnmf_dir = _utils.set_dir(results_dir.joinpath('cnmf'))

ks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

for time in times:
    print(f'Starting on {time}')
    tmp = sc.read_h5ad(split_adatas_dir.joinpath(f'{time}.h5ad'))
    
    c_object = cnmf.cNMF(cnmf_dir, time)
    
    # Variance normalized version of the data
    X = sc.pp.scale(tmp.X[:, tmp.var.highly_variable].toarray(), zero_center=False)
    
    c_object.prepare(X, ks, n_iter=200, new_nmf_kwargs={'tol': _constants.NMF_TOLERANCE})
    
    c_object.factorize(0, 1, gpu=True)
    
    c_object.combine()
    
    del tmp, X


# %%
for time in times:
    print(f'Starting on {time}')
    c_object = cnmf.cNMF(cnmf_dir, time)
    for thresh in [0.5, 0.4, 0.3]:
        c_object.k_selection_plot(density_threshold=thresh, nmf_refitting_iters=500,
                                  close_fig=True, show_clustering=True)


# %% [markdown]
# ### Selecting the decomposition rank utilizing K-selection plots and PCA variance explained

# %%
n_components = 14

for time in times:
    
    tmp = sc.read_h5ad(split_adatas_dir.joinpath(f'{time}.h5ad'))
    
    a, b, c, d, = sc.tl.pca(tmp.X[:, tmp.var.highly_variable], n_comps=n_components, return_info=True)
    print(f'{time} - cummulative variance percentages:'),
    for i in range(n_components):
        print(f'{c[i]*100: .2f}', end='\t')
    print()
    for i in range(n_components):
        print(f'{c.cumsum()[i]*100: .2f}', end='\t')
    print()

# %%
# # %%time

# selected_cnmf_params = {
#     '01_T_early_ND': (2, 0.5),  # rank 3 has slightly better loss, but is much more unstable
#     '02_KorKP_early_ND': (5, 0.5),  # could have chosen 4 as well
#     '04_K_12w_ND': (6, 0.4),    # Program 6 isn't very stable, need to be careful with it
#     '05_K_30w_ND': (8, 0.4),    # rank 8,0.3 better loss, but was less stable, 5 had similar loss with same stability
#     '06_KP_12w_ND': (7, 0.4),   # there are four programs that seem very stable among the 4+ ranks
#     '07_KP_20w_ND': (10, 0.5),  # Ranks 8, 9 also look very good
#     '08_KP_30w_ND': (8, 0.5)}   # Ranks 11+ had better loss but were less stable, and their last programs where garbage

# split_adatas = {}

# for time, (k, threshold) in selected_cnmf_params.items():
#     print(f'Working on {time} with k={k} and threshold={threshold}')
#     tmp = sc.read_h5ad(split_adatas_dir.joinpath(f'{time}.h5ad'))

#     c_object = cnmf.cNMF(cnmf_dir, time)
#     c_object.consensus(k, density_threshold=threshold, gpu=True, verbose=True,
#                        nmf_refitting_iters=1000, show_clustering=False)

#     usages, spectra = c_object.get_consensus_usages_spectra(k, density_threshold=threshold)

#     tmp.uns['cnmf_params'] = {'k_nmf': k, 'threshold': threshold}

#     tmp.obsm['usages'] = usages.copy()

#     usages_norm = usages / np.sum(usages, axis=1, keepdims=True)
#     tmp.obsm['usages_norm'] = usages_norm

#     # get per gene z-score of data after TPM normalization and log1p transformation 
#     tpm_log1p_zscore = tmp.X.toarray()
#     tpm_log1p_zscore /= 1e-6 * np.sum(tpm_log1p_zscore, axis=1, keepdims=True)
#     tpm_log1p_zscore = np.log1p(tpm_log1p_zscore)
#     tpm_log1p_zscore = sc.pp.scale(tpm_log1p_zscore)

#     usage_coefs = _utils.fastols(usages_norm, tpm_log1p_zscore)

#     tmp.varm['usage_coefs'] = pd.DataFrame(
#         usage_coefs.T, index=tmp.var.index,
#         columns=[f'{tmp.uns["sname"]}.p{prog}' for prog in range(usages.shape[1])])
    
#     split_adatas[time] = tmp
    
#     tmp.write_h5ad(split_adatas_dir.joinpath(f'{time}_GEPs.h5ad'))


# %%
# %%time
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
for time in times:
    tmp = split_adatas[time]
    
    un_sns = _utils.plot_usages_norm_clustermaps(tmp, show=True)
    plt.show(un_sns)
    plt.close()
    s = split_adatas[time].obsm['usages_norm'].sum(axis=0)
    with np.printoptions(precision=2, suppress=False):
        print(s * 100 / s.sum())


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

