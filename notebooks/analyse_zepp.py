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
# 2. filtering genes, and showing key statistics
# 3. Subsetting and splitting the dataset by developmental stage, and selecting joint highly variable genes (HVG)
# 4. Running consensus NMF (cNMF) per stage
# 5. Selecting parameters for the cNMF
# 6. Running the comparator for adjacent steps
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
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import silhouette_samples
import scanpy as sc
import matplotlib.pyplot as plt

from gepdynamics import _utils
from gepdynamics import _constants
from gepdynamics import cnmf
from gepdynamics import comparator

_utils.cd_proj_home()
print(os.getcwd())



# %% [markdown]
# ## 1. Obtaining the AnnData object and complementary metadata
# The adata contains log1p(CP10K) data, we un-transform the data to have the original counts as `X`

# %%
results_dir = _utils.set_dir('results')
results_dir = _utils.set_dir(results_dir.joinpath('zepp'))
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

untransformed = sparse.csr_matrix(adata.obs.n_molecules.values[:, None].astype(np.float32) / 10_000).multiply(adata.X.expm1())
adata.X = sparse.csc_matrix(untransformed).rint()

del untransformed

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
# ## 2. filtering genes, selecting joint highly variable genes (HVGs) and showing key statistics
#

# %%
# %%time
print(f'before filtering shape was {adata.X.shape}')

# filtering genes with very low abundance
sc.pp.filter_genes(adata, min_cells=np.round(adata.shape[0] / 1000))

# getting general statistics for counts abundance
sc.pp.filter_genes(adata, min_counts=0)
sc.pp.filter_cells(adata, min_counts=0)
sc.pp.filter_cells(adata, min_genes=0)

sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=_constants.NUMBER_HVG)

print(f'after filtering shape is {adata.X.shape}')

adata

# %%
column_of_interest = 'development_stage'

stats_df = adata.obs.loc[:, [column_of_interest, 'n_genes', 'n_counts']].groupby(
    [column_of_interest]).median()

stats_df = pd.concat([adata.obs.groupby([column_of_interest]).count().iloc[:, 0],
                      stats_df], axis=1)
stats_df.columns = ['# cells', 'median # genes', 'median # counts']

stats_df.plot(kind='bar', title=f'{column_of_interest} statistics', log=True, ylim=((5e2, 3e4)))
plt.show()
del column_of_interest, stats_df

# %%
_utils.joint_hvg_across_stages(adata, obs_category_key='development_stage', n_top_genes=5000)
adata.var


# %% [markdown]
# ### Saving/loading the pre-processed object

# %%
# %%time
pre_processed_adata_file = results_dir.joinpath('full.h5ad')

if not pre_processed_adata_file.exists():
    adata.write(pre_processed_adata_file)
else:
    adata = sc.read(pre_processed_adata_file)
adata

# %% [markdown]
# ## 3. Subsetting and splitting the dataset by stage, and selecting joint highly variable genes (HVG)
#

# %% [markdown]
# ### Splitting the adata by "development_stage", retaining only epithelial cells and creating a normalized variance layer

# %%
# %%time

column_of_interest = 'development_stage'

subset_adata_file = results_dir.joinpath('subset.h5ad')
if not subset_adata_file.exists():
    subset = adata[(adata.obs.compartment == 'epi') & (adata.obs.celltype != 'unknown3')].copy()

    sc.pp.filter_genes(subset, min_cells=1)
    sc.pp.filter_genes(subset, min_counts=1)

    _utils.joint_hvg_across_stages(subset, obs_category_key=column_of_interest, n_top_genes=5000)

    subset.write(subset_adata_file)
else:
    subset = sc.read(subset_adata_file)

# umap by celltype:
sc.pl.umap(subset, color=[column_of_interest, 'celltype'])

# statistics
stats_df = subset.obs.loc[:, [column_of_interest, 'n_genes', 'n_counts']].groupby(
    [column_of_interest]).median()

stats_df = pd.concat([subset.obs.groupby([column_of_interest]).count().iloc[:, 0],
                      stats_df], axis=1)
stats_df.columns = ['# cells', 'median # genes', 'median # counts']

stats_df.plot(kind='bar', title=f'Subset {column_of_interest} statistics', log=True, ylim=((1e2, 2e4)))
plt.show()
del stats_df

subset

# %%
# %%time

categories = subset.obs[column_of_interest].cat.categories

split_adatas_dir = _utils.set_dir(results_dir.joinpath(f'split_{column_of_interest}'))

for cat in categories:
    print(f'working on {cat}')
    if not split_adatas_dir.joinpath(f'{cat}.h5ad').exists():
        tmp = subset[subset.obs[column_of_interest] == cat].copy()

        tmp.uns['name'] = f'{cat}'   # full name
        tmp.uns['sname'] = f'{cat[:3]}'  # short name, here it is the same

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
    print(f'Starting on {cat}, time is {time.strftime("%H:%M:%S", time.localtime())}')
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
    for thresh in [0.5, 0.4]:
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
    
    a, b, c, d, = sc.tl.pca(tmp.X[:, tmp.var.joint_highly_variable], n_comps=n_components, return_info=True)

    df_var[f'{cat}'] = c*100
    df_cumulative_var[f'{cat}'] = c.cumsum()*100


# %%
plt.plot(range(len(df_var)), df_var, label=df_var.columns)
plt.title('Variance explained - jointly highly variable genes')
plt.legend()
plt.yscale('log')
plt.show()

# %%
plt.plot(df_cumulative_var.index, 100-df_cumulative_var, label=df_var.columns)
plt.yscale('log')
plt.title(f' 1- CDF variance explained joint highly variable')
plt.legend()
plt.show()

# %%
# %%time

selected_cnmf_params = {
    'E12': (4, 0.5),  # 
    'E15': (5, 0.5),  # 
    'E17': (8, 0.5),   # 
    'P3': (6, 0.5),    # 
    'P7': (6, 0.5),   # 
    'P15': (4, 0.5),  # 
    'P42': (5, 0.5)}   # 

split_adatas = {}

for cat, (k, threshold) in selected_cnmf_params.items():
    print(f'Working on epi {cat} with k={k} and threshold={threshold}')
    # %time tmp = sc.read_h5ad(split_adatas_dir.joinpath(f'{cat}.h5ad'))

    tmp.var.joint_highly_variable = subset.var.joint_highly_variable
    
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
# ### Examining results

# %%
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

# %%
decomposition_images = _utils.set_dir(split_adatas_dir.joinpath("images"))

for cat in categories:
    epidata = sc.read_h5ad(split_adatas_dir.joinpath(f"{cat}.h5ad"))
    
    # UMAP
    um = sc.pl.umap(epidata, color='celltype', s=10, return_fig=True, title=f'{cat} epithelial')
    plt.tight_layout()
    um.savefig(decomposition_images.joinpath(f"epi_{cat}_umap_celltype.png"), dpi=300)
    plt.close(um)

    # usages clustermap
    un_sns = _utils.plot_usages_norm_clustermaps(
        epidata, title=f'{cat}', show=False,sns_clustermap_params={
            'row_colors': epidata.obs['celltype'].map(epidata.uns['celltype_colors_dict'])})
    un_sns.savefig(decomposition_images.joinpath(f"{cat}_usages_norm.png"),
                   dpi=180, bbox_inches='tight')
    plt.close(un_sns.fig)

    # usages violin plot
    _utils.plot_usages_norm_violin(
        epidata, 'celltype', save_path=decomposition_images.joinpath(
            f'{cat}_norm_usage_per_lineage.png'))


# %% [markdown]
# ## 5. Running comparator on the data

# %%
for cat in categories:
    tmp = split_adatas[cat]

    field_1 = 'celltype'

    tmp.obsm['row_colors'] = pd.concat([
        tmp.obs[field_1].map(tmp.uns[f'{field_1}_colors_dict']),
        ], axis=1)

# %%
# %%time

pairs = [(categories[i], categories[i + 1]) for i in range(len(categories) - 1)]

for cat_a, cat_b in pairs:
    comparison_dir = _utils.set_dir(results_dir.joinpath(f"same_genes_{cat_a}_{cat_b}"))
    
    adata_a = split_adatas[cat_a]
    adata_b = split_adatas[cat_b]
    
    if os.path.exists(comparison_dir.joinpath('comparator.npz')):
        cmp = comparator.Comparator.load_from_file(comparison_dir.joinpath('comparator.npz'), adata_a, adata_b)
    else:
        cmp = comparator.Comparator(adata_a, adata_a.obsm['usages'], adata_b, comparison_dir,
                                    'torchnmf', device='cuda', max_nmf_iter=1000, verbosity=1,
                                   highly_variable_genes='joint_highly_variable')
        
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

    
