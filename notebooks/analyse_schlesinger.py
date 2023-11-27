# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Downloading, pre-processing and running cNMF on Schlesinger et. al 2020 data
# 1. Obtaining the count matrix and complementary metadata
#
# 2. Filtering genes, embedding the data, and showing key statistics
#
# 3. Subsettign, splitting the dataset by time after tumor induction, and selecting joint highly variable genes (HVG)
#
# 3. Running consensus NMF (cNMF) per time
#
# 4. Selecting parameters for the cNMF 
#
# 5. Running the comparator

# %%
# %%time
# %load_ext autoreload
# %autoreload 2

#debug code for ide:
from importlib import reload

import sys
import os
import time
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from scipy import sparse
import scanpy as sc
import matplotlib.pyplot as plt

from gepdynamics import _utils
from gepdynamics import _constants
from gepdynamics import cnmf
from gepdynamics import comparator

_utils.cd_proj_home()
print(os.getcwd())

results_dir = _utils.set_dir(_utils.set_dir('results').joinpath('schlesinger'))



# %% [markdown]
# ## 1. Downloading and creating a basic AnnData object

# %%
# %%time

data_dir = _utils.set_dir('data')

unprocessed_adata_path = data_dir.joinpath('GSE141017_ALL.h5ad')

if not unprocessed_adata_path.exists():  # create the original adata if it doesn't exist
    # directories for file download:
    
    GSE_dir = _utils.set_dir(data_dir.joinpath('GSE141017'))
    
    # GEO server prefix for PDAC Series GSE141017
    ftp_address = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE141nnn/GSE141017/suppl/'

    #filenames
    f_raw_data = GSE_dir.joinpath('GSE141017_ALL.csv.gz')
    f_barcodes_ident = GSE_dir.joinpath('GSE141017_ALL_barcode_ident.csv.gz')

    # downloading if needed:
    if not f_raw_data.exists():
        urlretrieve(ftp_address + f_raw_data.name, f_raw_data)
    
    if not f_barcodes_ident.exists():
        urlretrieve(ftp_address + f_barcodes_ident.name, f_barcodes_ident)

    barcodes_ident = pd.read_csv(f_barcodes_ident, index_col=0, sep='\t', 
                                 dtype={'ident': 'category', 'time_point': 'category'})
    
    # fixing the barcode identities dataframe to match raw data
    barcodes_ident.index = barcodes_ident.index.str.replace('.', '-', regex=False).str.replace('^X', '', regex=True)
    
    # changing categories display order
    barcodes_ident.time_point = barcodes_ident.time_point.cat.reorder_categories(
        ['CTRL', '17D', '6W', '3M', '5M', '9M', '15M', ])
    barcodes_ident.ident = barcodes_ident.ident.cat.reorder_categories(
        [str(i) for i in range(27)])
    
    # loading the raw data as np.float32
    dtype_dict = {col: np.float32 for col in barcodes_ident.index}
    raw_data = pd.read_csv(f_raw_data, index_col=0, sep='\t', dtype=dtype_dict)
    
    # constructing the adata
    adata = sc.AnnData(X=raw_data.T, dtype=np.float32, obs=barcodes_ident)
    
    adata.X = sparse.csr_matrix(adata.X)
    
    adata.write(unprocessed_adata_path)
    
    del raw_data, barcodes_ident   
else:
    adata = sc.read(unprocessed_adata_path)

print(f'Density of data = {_utils.calculate_anndata_object_density(adata): .4f}')
adata


# %%
pd.crosstab(adata.obs.ident, adata.obs.time_point)

# %%
sc.pl.highest_expr_genes(adata, n_top=20, )

# %% [markdown]
# ## 2. Filtering genes, embedding the data, and showing key statistics

# %% [markdown]
# ### Filter genes and plot basic statistics
# The cells were pre-filtered to:
# 1. A minimum of 1000 counts and 200 genes.
# 2. Cells with more than 5% (most) or 10% (9M, 15M) mitochondrial genes were removed
# 3. Cells with more than 6000 genes were removed
#
# We add filtering to have:
# 1. Remove cells with more than 40,000 counts
# 2. Remove genes appearing in less than 4 cells
#

# %%
# %%time
print(f'before filtering shape was {adata.X.shape}')

# filtering genes with very low abundance
sc.pp.filter_genes(adata, min_cells=np.round(adata.shape[0] / 10000))

# getting general statistics for counts abundance
sc.pp.filter_genes(adata, min_counts=0)
sc.pp.filter_cells(adata, min_counts=0)
sc.pp.filter_cells(adata, max_counts=4e4)
sc.pp.filter_cells(adata, min_genes=0)

sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=_constants.NUMBER_HVG)

print(f'after filtering shape is {adata.X.shape}')

adata

# %%
column_of_interest = 'time_point'

stats_df = adata.obs.loc[:, [column_of_interest, 'n_genes', 'n_counts']].groupby(
    [column_of_interest]).median()

stats_df = pd.concat([adata.obs.groupby([column_of_interest]).count().iloc[:, 0],
                      stats_df], axis=1)
stats_df.columns = ['# cells', 'median # genes', 'median # counts']

stats_df.plot(kind='bar', title=f'{column_of_interest} statistics', log=True, ylim=((5e2, 3e4)))
plt.show()
del column_of_interest, stats_df

# %%
# From scanpy tutorial:

adata.var['mt'] = adata.var_names.str.startswith('mt-')  # annotate the group of mitochondrial genes as 'mt'
print(f"{np.sum(adata.var['mt'])} mitochondrial genes")
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

sc.pl.violin(adata, ['n_genes', 'total_counts', 'pct_counts_mt',],
             jitter=0.4, multi_panel=True, groupby='time_point')


# %% [markdown]
# ### Creating embedding for the data

# %%
# %%time
bdata = adata[:, adata.var.highly_variable].copy()

sc.pp.normalize_total(bdata, target_sum=1e4)
sc.pp.log1p(bdata)
sc.pp.regress_out(bdata, ['total_counts']) # The pct_counts_mt regression was removed as it correlates with time 
sc.pp.scale(bdata, max_value=20)

sc.tl.pca(bdata, svd_solver='arpack', n_comps=100)
sc.pl.pca(bdata, color=['time_point', 'ident'])
sc.pl.pca(bdata, color=['total_counts', 'pct_counts_mt'])

# %%
print(f"cummulative variance of {bdata.uns['pca']['variance_ratio'].size} PCs is {bdata.uns['pca']['variance_ratio'].sum():.3f}")
sc.pl.pca_variance_ratio(bdata, log=True, n_pcs=50)

# %%
# %%time
sc.pp.neighbors(bdata, n_neighbors=10, n_pcs=100)
sc.tl.umap(bdata)
sc.pl.umap(bdata, color='time_point')
sc.pl.umap(bdata, color='ident')

# %%
sc.pl.umap(bdata, color=['Ptf1a', 'tdTomato'], vmax=[1, 50])


# %%
sc.pl.umap(bdata, color=['Epcam', 'Ptprc'], vmax=[25, 2])


# %% [markdown]
# #### Adding the PCA and UMAP coordinates to the main AnnData object

# %%
adata.obsm = bdata.obsm
adata.obsp = bdata.obsp
adata.uns.update(bdata.uns)
adata.uns.pop('log1p')
adata

# %%
for obs_col in ['time_point', 'ident']:
    adata.uns[obs_col + '_colors_dict'] = dict(zip(adata.obs[obs_col].cat.categories, adata.uns[obs_col + '_colors']))


# %% [markdown]
# ### selecting jointly highly variable genes

# %%
_utils.joint_hvg_across_stages(adata, obs_category_key = 'time_point')

# %% [markdown]
# ### Saving/loading the pre-processed object

# %%
# %%time
pre_processed_adata_file = results_dir.joinpath('full.h5ad')

if not pre_processed_adata_file.exists():
    adata.write(pre_processed_adata_file)
else:
    adata = sc.read(pre_processed_adata_file)

# %% [markdown]
# ## 3. Subsetting and splitting the dataset by time after tumor induction, and selecting joint highly variable genes (HVG)
#

# %% [markdown]
# ### Subsetting relevant epithelial clusters (ductal-0, acinar-9 early 15 late, metaplastic-4 17, endocrine-18)

# %%
column_of_interest = 'time_point'

subset_adata_file = results_dir.joinpath('subset.h5ad')
if not subset_adata_file.exists():
    subset = adata[adata.obs.ident.isin(['0', '4', '9', '15', '17', '18'])].copy()

    # Removing cells that were probably miss-labeled:
    subset = subset[subset.obsp['connectivities'].sum(axis=1)>=2.1]

    sc.pp.filter_genes(subset, min_cells=0)
    sc.pp.filter_genes(subset, min_counts=0)

    _utils.joint_hvg_across_stages(subset, obs_category_key=column_of_interest)

    subset.write(subset_adata_file)
else:
    subset = sc.read(subset_adata_file)
    
sc.pl.umap(subset, color=['time_point', 'ident'])



# %% [markdown]
# ### Splitting the adata by "time_point"

# %%
# %%time

column_of_interest = 'time_point'
categories = subset.obs[column_of_interest].cat.categories

split_adatas_dir = _utils.set_dir(results_dir.joinpath(f'split_{column_of_interest}'))

for cat in categories:
    print(f'working on {cat}')
    if not split_adatas_dir.joinpath(f'{cat}.h5ad').exists():
        tmp = subset[subset.obs[column_of_interest] == cat].copy()

        tmp.uns['name'] = f'{cat}'   # full name
        tmp.uns['sname'] = f'{cat[:3]}'  # short name, truncating CTRL to CTR

        # correcting the gene counts
        sc.pp.filter_genes(tmp, min_cells=0)
        sc.pp.filter_genes(tmp, min_counts=0)

        tmp.write_h5ad(split_adatas_dir.joinpath(f'{cat}.h5ad'))

        del tmp


# %% [markdown]
# ### Running multiple NMF iterations

# %%
cnmf_dir = _utils.set_dir(results_dir.joinpath('cnmf'))


# %%
# %%time

ks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]#, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

for cat in categories:
    print(f'Starting on {cat}')
    tmp = sc.read_h5ad(split_adatas_dir.joinpath(f'{cat}.h5ad'))
    
    c_object = cnmf.cNMF(cnmf_dir, cat)
    
    # Variance normalized version of the data
    X = sc.pp.scale(tmp.X[:, tmp.var.joint_highly_variable].toarray().astype(np.float32), zero_center=False)
    
    c_object.prepare(X, ks, n_iter=100, new_nmf_kwargs={'tol': _constants.NMF_TOLERANCE,
                                                        'beta_loss': 'kullback-leibler'})
    
    c_object.factorize(0, 1, gpu=True)
    
    c_object.combine()
    
    del tmp, X


# %%
# %%time
for cat in categories:
    print(f'Starting on {cat}, time is {time.strftime("%H:%M:%S", time.localtime())}')
    c_object = cnmf.cNMF(cnmf_dir, cat)
    for thresh in [0.5, 0.4, 0.3]:
        print(f'working on threshold {thresh}')
        c_object.k_selection_plot(density_threshold=thresh, nmf_refitting_iters=1000,
                                  close_fig=True, show_clustering=True, gpu=True)


# %% [markdown]
# ### Selecting the decomposition rank utilizing K-selection plots and PCA variance explained

# %%
# %%time
df_var = pd.DataFrame()
df_cumulative_var = pd.DataFrame()

n_components = 50

for cat in categories:    
    # %time tmp = sc.read_h5ad(split_adatas_dir.joinpath(f'{cat}.h5ad'))
    
    scaled_log1p_tp100k = np.log1p(np.array(tmp.X[:, tmp.var.joint_highly_variable] / np.sum(tmp.X, axis=1) * 1e4))
    # remove zero-column (genes):
    scaled_log1p_tp100k = scaled_log1p_tp100k[:, scaled_log1p_tp100k.sum(axis=0) > 0]
    
    scaled_log1p_tp100k -= scaled_log1p_tp100k.mean(axis=0)
    scaled_log1p_tp100k /= scaled_log1p_tp100k.std(axis=0)
    
    a, b, c, d, = sc.tl.pca(scaled_log1p_tp100k, n_comps=n_components, return_info=True)

    df_var[f'{cat}'] = c*100
    df_cumulative_var[f'{cat}'] = c.cumsum()*100
    
    del tmp


# %%
df_var[:5]

# %%
plt.plot(range(50), df_var, label=df_var.columns)
plt.title('Percent variance explained')
plt.legend()
plt.yscale('log')
plt.show()

# %%
plt.plot(df_cumulative_var.index, 1-df_cumulative_var/100, label=df_var.columns)
plt.yscale('log')
plt.title(f' 1- CDF variance explained')
plt.legend()
plt.show()

# %%
# %%time

selected_cnmf_params = {
    'CTRL': (7, 0.5),  # 
    '17D': (6, 0.5),  # 
    '6W': (9, 0.5),    # 
    '3M': (9, 0.5),    # 
    '5M': (9, 0.5),   # 
    '9M': (9, 0.5),  # 
    '15M': (10, 0.5)}   # 

split_adatas = {}

for cat, (k, threshold) in selected_cnmf_params.items():
    print(f'Working on {cat} with k={k} and threshold={threshold}')
    # %time tmp = sc.read_h5ad(split_adatas_dir.joinpath(f'{cat}.h5ad'))

    c_object = cnmf.cNMF(cnmf_dir, cat)
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
    
    split_adatas[cat] = tmp

    tmp.write_h5ad(split_adatas_dir.joinpath(f'{cat}.h5ad'))


# %% [markdown]
# ### Examening results

# %% Loading GEPs adatas
# %%time

split_adatas = {}
for cat in categories:
    split_adatas[cat] = sc.read_h5ad(split_adatas_dir.joinpath(f'{cat}.h5ad'))

# %%
for cat in categories:
    print(cat)
    s = split_adatas[cat].obsm['usages_norm'].sum(axis=0)
    with np.printoptions(precision=2, suppress=False):
        print(s * 100 / s.sum())

# %% [markdown]
# ## Running comparator on the data

# %%
for cat in categories:
    tmp = split_adatas[cat]

    field_1 = 'ident'

    tmp.obsm['row_colors'] = pd.concat([
        tmp.obs[field_1].map(tmp.uns[f'{field_1}_colors_dict']),
        ], axis=1)

# %%
# %%time

pairs = [(categories[i], categories[i + 1]) for i in range(len(categories) - 1)]

for cat_a, cat_b in pairs:
    comparison_dir = _utils.set_dir(results_dir.joinpath(f"unique_genes_{cat_a}_{cat_b}"))
    
    adata_a = split_adatas[cat_a]
    adata_b = split_adatas[cat_b]
    
    if os.path.exists(comparison_dir.joinpath('comparator.npz')):
        cmp = comparator.Comparator.load_from_file(comparison_dir.joinpath('comparator.npz'), adata_a, adata_b)
    else:
        cmp = comparator.Comparator(adata_a, adata_a.obsm['usages'], adata_b, comparison_dir, 'torchnmf', device='cuda', max_nmf_iter=1000, verbosity=1)
        
        print('decomposing')
        cmp.extract_geps_on_jointly_hvgs()
        cmp.decompose_b(repeats = 5)
    
    cmp.examine_adata_a_decomposition_on_jointly_hvgs()
    
    cmp.print_errors()
    cmp.examine_adata_b_decompositions()
    cmp.plot_decomposition_comparisons()
    
    cmp.calculate_fingerprints()
    
    print('running GSEA')
    cmp.run_gsea(gprofiler_kwargs=dict(organism='mmusculus', sources=['GO:BP', 'WP', 'REAC', 'KEGG']))

    cmp.save_to_file(comparison_dir.joinpath('comparator.npz'))

    

# %% [markdown]
# ## ToDo - plot all programs on phate
# ## ToDo - consider adding "main program" per cell and plot it on joint tsne and phate

# %% [markdown]
# # Epithelial compartment decomposition

# %% [markdown]
# ### Splitting the adata by "development_stage", retaining only epithelial cells and creating a normalized variance layer

# %%
# %%time

stages = adata.obs.development_stage.cat.categories
epi_split_adatas_dir = _utils.set_dir(results_dir.joinpath('epi_split_adatas'))

for stage in stages:
    print(f'working on stage {stage}')
    file = epi_split_adatas_dir.joinpath(f'epi_{stage}.h5ad')
    
    if not file.exists():
        tmp = adata[adata.obs.development_stage == stage]
        tmp = tmp[(tmp.obs.compartment == 'epi') & (tmp.obs.celltype != 'unknown3')].copy()
                
        tmp.uns['name'] = f'Epi {stage}'   # full name
        tmp.uns['sname'] = f'{stage}'  # short name, here it is the same

        # correcting the gene counts
        sc.pp.filter_genes(tmp, min_cells=0)
        sc.pp.filter_genes(tmp, min_counts=0)

        # calculating per sample HVGs
        sc.pp.highly_variable_genes(tmp, flavor='seurat_v3', n_top_genes=_constants.NUMBER_HVG)

        tmp.write_h5ad(file)

        del tmp


# %% [markdown]
# ### Running multiple NMF iterations

# %%
# %%time

cnmf_dir = _utils.set_dir(results_dir.joinpath('epi_cnmf'))

ks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

for stage in stages:
    print(f'Starting on epithelial {stage}, time is {time.strftime("%H:%M:%S", time.localtime())}')
    tmp = sc.read_h5ad(epi_split_adatas_dir.joinpath(f'epi_{stage}.h5ad'))
    
    c_object = cnmf.cNMF(cnmf_dir, stage)
    
    # Variance normalized version of the data
    X = sc.pp.scale(tmp.X[:, tmp.var.highly_variable].toarray().astype(np.float32), zero_center=False)
    
    c_object.prepare(X, ks, n_iter=100, new_nmf_kwargs={'tol': _constants.NMF_TOLERANCE,
                                                        'beta_loss': 'kullback-leibler'})
    
    c_object.factorize(0, 1, gpu=True)
    
    c_object.combine()
    
    del tmp, X
    


# %%
# %%time
for stage in stages:
    print(f'Starting on {stage}, time is {time.strftime("%H:%M:%S", time.localtime())}')
    c_object = cnmf.cNMF(cnmf_dir, stage)
    for thresh in [0.5, 0.4, 0.3]:
        print(f'working on threshold {thresh}')
        c_object.k_selection_plot(density_threshold=thresh, nmf_refitting_iters=500,
                                  close_fig=True, show_clustering=True, gpu=True)

    


# %% [markdown]
# ### Selecting the decomposition rank utilizing K-selection plots and PCA variance explained

# %%
# %%time
df_var = pd.DataFrame()
df_cumulative_var = pd.DataFrame()

n_components = 30

for stage in stages:
    
    # %time tmp = sc.read_h5ad(epi_split_adatas_dir.joinpath(f'epi_{stage}.h5ad'))
    
    a, b, c, d, = sc.tl.pca(tmp.X[:, tmp.var.highly_variable], n_comps=n_components, return_info=True)

    df_var[f'{stage}'] = c*100
    df_cumulative_var[f'{stage}'] = c.cumsum()*100


# %%
plt.plot(range(len(df_var)), df_var, label=df_var.columns)
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
    'E12': (7, 0.5),  # 
    'E15': (8, 0.5),  # 
    'E17': (10, 0.5),   # 
    'P3': (9, 0.5),    # 
    'P7': (8, 0.5),   # 
    'P15': (5, 0.5),  # 
    'P42': (5, 0.5)}   # 

epi_split_adatas = {}

for stage, (k, threshold) in selected_cnmf_params.items():
    print(f'Working on epi {stage} with k={k} and threshold={threshold}')
    # %time tmp = sc.read_h5ad(epi_split_adatas_dir.joinpath(f'epi_{stage}.h5ad'))

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
    
    epi_split_adatas[stage] = tmp

    tmp.write_h5ad(epi_split_adatas_dir.joinpath(f'epi_{stage}_GEPs.h5ad'))


# %%
epidata.obsm_keys()

# %%
stages = ['E12', 'E15', 'E17', 'P3', 'P7', 'P15', 'P42']
decomposition_images = _utils.set_dir(results_dir.joinpath("epi_split_adatas", "images"))


for stage in stages:
    epidata = sc.read_h5ad(results_dir.joinpath("epi_split_adatas", f"epi_{stage}_GEPs.h5ad"))
    
    # UMAP
    um = sc.pl.umap(epidata, color='celltype', s=10, return_fig=True, title=f'{stage} epithelial')
    plt.tight_layout()
    um.savefig(decomposition_images.joinpath(f"epi_{stage}_umap_celltype.png"), dpi=300)
    plt.close(um)

    # usages clustermap
    un_sns = _utils.plot_usages_norm_clustermaps(
        epidata, title=f'{stage}', show=False,sns_clustermap_params={
            'row_colors': epidata.obs['celltype'].map(epidata.uns['celltype_colors_dict'])})
    un_sns.savefig(decomposition_images.joinpath(f"epi_{stage}_usages_norm.png"),
                   dpi=180, bbox_inches='tight')
    plt.close(un_sns.fig)

    # usages violin plot
    _utils.plot_usages_norm_violin(
        epidata, 'celltype', save_path=decomposition_images.joinpath(
            f'epi_{stage}_norm_usage_per_lineage.png'))

