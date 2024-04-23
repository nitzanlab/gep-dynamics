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
# # Downloading, pre-processing and running cNMF on Zepp et. al 2021 data
# 1. Obtaining the AnnData object and complementary metadata
# 2. Filtering cells
# 3. Subsetting and splitting the dataset by developmental stage, and selecting joint highly variable genes (HVG)
# 4. Running consensus NMF (cNMF) per stage
# 5. Selecting parameters for the cNMF
#

# %%
# %%time
# %load_ext autoreload
# %autoreload 2

#debug:
# from importlib import reload

import sys
import os
import time
import warnings

    
import numpy as np
import pandas as pd
from scipy import sparse
# from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

from gepdynamics import _utils, _constants, cnmf, pfnmf, comparator, plotting

warnings.filterwarnings("ignore", category=FutureWarning)

sc.settings.n_jobs=-1

_utils.cd_proj_home()
print(os.getcwd())


# %%
results_dir = _utils.set_dir('results')
results_dir = _utils.set_dir(results_dir.joinpath('zepp'))
data_dir = _utils.set_dir('data')
GSE_dir = _utils.set_dir(data_dir.joinpath('GSE149563'))


# %% [markdown]
# ## 1. Obtaining the AnnData object and complementary metadata
# The adata contains log1p(CP10K) data, we un-transform the data to have the original counts as `X`

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

# %time adata = sc.read_h5ad(GSE_dir.joinpath('JZ_Mouse_TimeSeries.h5ad'))
metadata = pd.read_csv(GSE_dir.joinpath('AllTimePoints_metadata.csv'), index_col=0)

adata.obs['celltype'] = metadata.var_celltype
adata.obs['compartment'] = metadata.var_compartment

untransformed = sparse.csr_matrix(adata.obs.n_molecules.values[:, None].astype(np.float32) / 10_000).multiply(adata.X.expm1())
adata.X = sparse.csr_matrix(untransformed).rint()

del untransformed, metadata

adata


# %%
adata.obs.development_stage = adata.obs.development_stage.cat.rename_categories(
    {'Adult': 'P42', 'E12.5': 'E12', 'E15.5': 'E15', 'E17.5': 'E17'}).cat.reorder_categories(
    ['E12', 'E15', 'E17', 'P3', 'P7', 'P15', 'P42'])

# %%
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    
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
adata.var['mt'] = adata.var_names.str.startswith('mt-')  # annotate the group of mitochondrial genes as 'mt'
print(f"{np.sum(adata.var['mt'])} mitochondrial genes")
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    sc.pl.violin(adata, ['n_genes', 'total_counts', 'pct_counts_mt',],
                 jitter=0.4, multi_panel=True, groupby='development_stage')

# %% [markdown]
# ## 2 Filtering cells and genes

# %% [markdown]
# ### 2.0 Basic removal of cells with <800 gens and genes in <20 cells and showing key statistics

# %%
adata = adata[adata.obs.development_stage.isin(['E12', 'E15', 'E17', 'P3', 'P7'])]
adata = adata[~adata.obs.celltype.str.startswith('unknown')]
adata = adata[adata.obs.celltype!='Ciliated'].copy()

# %%
# %%time
print(f'before filtering shape was {adata.X.shape}')

# filtering cells with low number of genes
sc.pp.filter_cells(adata, min_genes=800)

# filtering genes with very low abundance
sc.pp.filter_genes(adata, min_cells=20)

# getting general statistics for counts abundance
sc.pp.filter_genes(adata, min_counts=0)
sc.pp.filter_cells(adata, min_counts=0)

print(f'after filtering shape is {adata.X.shape}')

_utils.joint_hvg_across_stages(adata, obs_category_key='development_stage', n_top_genes=5000)
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

# %% [markdown]
# ### 2.1 SVM to remove cells with umbiguous compartment assignment
#

# %%
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

# %%
# %%time
k=10

adata.obs[f'predP{k}'] = None
for stage in adata.obs.development_stage.cat.categories:
    dat = adata[adata.obs.development_stage==stage].copy()
    sc.pp.normalize_total(dat, 5_000)
    sc.pp.log1p(dat)
    pca = PCA(n_components=k)
    pcs = pca.fit_transform(dat[:, dat.var.joint_highly_variable].X.toarray())
    svm_model = LinearSVC(C=0.00001, class_weight='balanced', dual=False)
    svm_model.fit(pcs, dat.obs.compartment)
    adata.obs.loc[adata.obs.development_stage==stage, f'predP{k}'] = svm_model.predict(pcs)

# %%
pd.crosstab(adata.obs.compartment, adata.obs.predP10)

# %%
adata.obs['compartment_cleaned'] = 'ambiguous'
adata.obs.loc[adata.obs.compartment==adata.obs.predP10, 'compartment_cleaned'] = adata[adata.obs.compartment==adata.obs.predP10].obs.compartment
adata.obs.compartment_cleaned.value_counts()

# %%
# %%time
k=10

adata.obs[f'predP{k}_twice'] = 'dropped'
for stage in adata.obs.development_stage.cat.categories:
    dat = adata[adata.obs.development_stage==stage].copy()
    sc.pp.normalize_total(dat, 5_000)
    sc.pp.log1p(dat)
    pca = PCA(n_components=k)
    pcs = pca.fit_transform(dat[:, dat.var.joint_highly_variable].X.toarray())
    svm_model = LinearSVC(C=0.00001, class_weight='balanced', dual=False)
    # unabiguous = dat.obs.compartment_cleaned!='ambiguous'
    # svm_model.fit(pcs[unabiguous], dat[unabiguous].obs.compartment_cleaned)
    
    svm_model.fit(pcs, dat.obs.compartment_cleaned)
    adata.obs.loc[dat.obs.index, f'predP{k}_twice'] = svm_model.predict(pcs)

# %%
pd.crosstab(adata.obs.compartment_cleaned, adata.obs.predP10_twice)

# %%
adata.obs.loc[adata.obs.compartment_cleaned!=adata.obs.predP10_twice, 'compartment_cleaned'] = 'ambiguous'

# %%
pred = 'predP10_twice'
pred = 'predP10'

bdata = adata[(adata.obs['predP10']=='epi') | (adata.obs['predP10_twice']=='epi') | (adata.obs.compartment=='epi')]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    sc.pl.umap(bdata, color=['predP10', 'predP10_twice', 'compartment_cleaned'])
    sc.pl.pca(bdata, color=['predP10', 'predP10_twice', 'compartment_cleaned'])

# %%
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    sc.pl.umap(adata[adata.obs.compartment_cleaned=='epi'], color=['celltype'])

# %%
pd.crosstab(adata.obs.compartment_cleaned, adata.obs.compartment)

# %%
pd.crosstab(adata.obs.compartment_cleaned, adata.obs.development_stage)

# %% [markdown]
# ### 2.2 Cleaning the data - doublet removal

# %% [markdown]
# #### Training VAE towards Solo doublet removal

# %%
import torch
torch.set_float32_matmul_precision("high")

# %%
# %%time
import scvi
from lightning.fabric.utilities.warnings import PossibleUserWarning
warnings.filterwarnings("ignore", category=PossibleUserWarning)

# %%
# %%time
subdata = adata[adata.obs.compartment_cleaned!='ambiguous', adata.var.joint_highly_variable].copy()

scvi.model.SCVI.setup_anndata(subdata, labels_key='development_stage')
vae = scvi.model.SCVI(subdata, n_layers=3, n_hidden=256, n_latent=64, dispersion='gene-label')

vae

# %%
# %time vae.train(accelerator='gpu', train_size=0.8, validation_size=0.1, check_val_every_n_epoch=1, batch_size=256, max_epochs=150)
vae

# %%
list(vae.history.keys())

# %%
for stat in ['train_loss_epoch', 'reconstruction_loss_train', 'validation_loss', 'reconstruction_loss_validation']:
    plt.plot(vae.history[stat][20:], label=stat)
plt.legend()
plt.show()



# %%
# model_dir = results_dir.joinpath("scvi_model_batches")
model_dir = results_dir.joinpath("scvi_model_labels")

if vae.is_trained:
    vae.save(model_dir, overwrite=True)

# vae = scvi.model.SCVI.load(model_dir, adata=adata[:, adata.var.joint_highly_variable].copy())
vae

# %%
SCVI_LATENT_KEY = "X_scVI"

latent = vae.get_latent_representation(adata[:, adata.var.joint_highly_variable])
adata.obsm[SCVI_LATENT_KEY] = latent
latent.shape

# %% [markdown]
# Visualizing the SCVI latent space

# %%
# %%time
sc.pp.neighbors(adata, use_rep=SCVI_LATENT_KEY)
vae_adata = sc.tl.umap(adata, min_dist=0.3, copy=True)

# %%
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    sc.pl.umap(vae_adata, color=['development_stage', 'compartment_cleaned'])
    sc.pl.umap(vae_adata, color=['celltype'])

# %% [markdown]
# #### Solo doublet prediction

# %%
SOLO_THRESHOLD = 0.5

stages = adata.obs.development_stage.cat.categories
stages

# %%
adata.obs['doublet_softmax'] = np.nan

# %%

for stage in stages:
    print(f'working on stage {stage}', '\n')
    cells_mask = (adata.obs.development_stage==stage) & (adata.obs.compartment_cleaned != 'ambiguous')
    subdata = adata[cells_mask, adata.var.joint_highly_variable].copy()
    
    solo_batch = scvi.external.SOLO.from_scvi_model(
        vae, subdata, doublet_ratio=20, n_layers=2)

    solo_batch.train(accelerator='gpu', train_size=0.85, validation_size=0.15, batch_size=128, max_epochs=600)

    df = solo_batch.predict()
    df['softmax'] = np.exp(df['doublet'])/np.sum(np.exp(df[['singlet', 'doublet']]), axis=1)
    df['prediction'] = np.where(df['softmax'] > SOLO_THRESHOLD, 'doublet', 'singlet')

    print(df.prediction.value_counts())
    
    df.softmax.hist(bins=np.linspace(0,1,21))
    
    subdata.obs['prediction'] = df.prediction
    subdata.obs['softmax'] = df.softmax
    
    print(pd.crosstab(subdata.obs.compartment, subdata.obs.prediction))
    
    comp = 'epi'
    plot_dat = subdata[(subdata.obs.compartment==comp)]
    dat = vae_adata[adata.obs.development_stage==stage]
    
    dat.obs['softmax'] = df.softmax
    dat.obs['prediction'] = df.prediction
    
    print(pd.crosstab(plot_dat.obs.celltype, plot_dat.obs.prediction))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        sc.pl.umap(plot_dat, color=['celltype'])
        sc.pl.umap(plot_dat, color=['softmax', 'prediction'])
        sc.pl.umap(subdata, color=['softmax', 'prediction'])
        
        sc.pl.umap(dat, color=['softmax', 'prediction'])
    
    adata.obs.loc[subdata.obs_names, 'doublet_softmax'] = subdata.obs.softmax


# %%
adata.obs['doublet_softmax'].hist(bins=np.linspace(0,1,21))
plt.title('doublet softmax probability')
sc.pl.umap(adata, color='doublet_softmax')

# %%
adata.obs['doublet'] = np.where(adata.obs['doublet_softmax'] > SOLO_THRESHOLD, 'doublet', 'singlet')
sc.pl.umap(adata, color='doublet')

# %%
adata.obs['doublet_softmax_r2'] = np.nan

# %%
# %%time
r2_data = adata[(adata.obs.doublet == 'singlet') & (adata.obs.compartment_cleaned != 'ambiguous')]

for stage in stages:
    print(f'working on stage {stage}')
    
    subdata = r2_data[r2_data.obs.development_stage==stage, r2_data.var.joint_highly_variable].copy()
    solo_batch = scvi.external.SOLO.from_scvi_model(
        vae, subdata.copy(), doublet_ratio=20, n_layers=2)

    # %time solo_batch.train(accelerator='gpu', train_size=0.8, validation_size=0.1, batch_size=128, max_epochs=100)

    df = solo_batch.predict()
    df['softmax'] = np.exp(df['doublet'])/np.sum(np.exp(df[['singlet', 'doublet']]), axis=1)
    df['prediction'] = np.where(df['softmax'] > SOLO_THRESHOLD, 'doublet', 'singlet')

    print(df.prediction.value_counts())
    
    df.softmax.hist(bins=np.linspace(0,1,21))
    
    subdata.obs['prediction'] = df.prediction
    subdata.obs['softmax'] = df.softmax
    
    print(pd.crosstab(subdata.obs.compartment, subdata.obs.prediction))
    
    comp = 'epi'
    plot_dat = subdata[(subdata.obs.compartment==comp)]
    dat = vae_adata[adata.obs.development_stage==stage]
    
    dat.obs['softmax'] = df.softmax
    dat.obs['prediction'] = df.prediction
    
    print(pd.crosstab(plot_dat.obs.celltype, plot_dat.obs.prediction))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        sc.pl.umap(plot_dat, color=['celltype'])
        sc.pl.umap(plot_dat, color=['softmax', 'prediction'])
        sc.pl.umap(subdata, color=['softmax', 'prediction'])
        
        sc.pl.umap(dat, color=['softmax', 'prediction'])
    
    adata.obs.loc[subdata.obs_names, 'doublet_softmax_r2'] = subdata.obs.softmax

# %%
adata.obs['doublet_softmax_r2'].hist(bins=np.linspace(0,1,21))
plt.title('doublet softmax round 2 probability')
sc.pl.umap(adata, color='doublet_softmax_r2')

# %%
adata.obs['doublet'] = np.where((adata.obs['doublet_softmax'] > SOLO_THRESHOLD) | (adata.obs['doublet_softmax_r2'] > SOLO_THRESHOLD), 'doublet', 'singlet')
sc.pl.umap(adata, color='doublet')

# %%
plt.scatter(adata.obs.doublet_softmax, adata.obs.doublet_softmax_r2, s=2)

# %%
no_doublets = adata[adata.obs.doublet == 'singlet']
no_doublets = no_doublets[no_doublets.obs.compartment_cleaned == 'epi']
pd.crosstab(no_doublets.obs.celltype, no_doublets.obs.development_stage)

# %%
pd.crosstab(adata.obs.doublet, adata.obs.compartment_cleaned)

# %%
adata = adata[(adata.obs.doublet == 'singlet') & (adata.obs.compartment_cleaned != 'ambiguous' )]
adata.shape

# %% [markdown]
# ### Saving/loading the pre-processed object

# %%
# %%time
pre_processed_adata_file = results_dir.joinpath('full.h5ad')

if not pre_processed_adata_file.exists():
    adata.write(pre_processed_adata_file)
else:
    adata = sc.read_h5ad(pre_processed_adata_file)
adata

# %%
column_of_interest = 'development_stage'
color_obs_by = 'compartment'

with warnings.catch_warnings():  # supress scanpy plotting warning
    warnings.simplefilter(action='ignore', category=UserWarning)
    
    # umap by celltype:
    sc.pl.umap(adata, color=[column_of_interest, color_obs_by])
    
    # statistics
    stats_df = adata.obs.loc[:, [column_of_interest, 'n_genes', 'n_counts']].groupby(
        [column_of_interest]).median()
    
    stats_df = pd.concat([adata.obs.groupby([column_of_interest]).count().iloc[:, 0],
                          stats_df], axis=1)
    stats_df.columns = ['# cells', 'median # genes', 'median # counts']
    
    stats_df.plot(kind='bar', title=f'{column_of_interest} statistics', log=True, ylim=((1e2, 2e4)))
    plt.show()
    del stats_df


# %%
pd.crosstab(adata.obs.celltype, adata.obs.development_stage)

# %% [markdown]
# ## 3. Subsetting and splitting the dataset by stage, and selecting joint highly variable genes (HVG)
# #### Cells filter:
# 1.  Epithelial (not including "Ciliated" as those were almost exclusively doublets and the "unknown 3" celltype)
# 2.  Stages E12-P7
#
# #### Genes filter:
# 1.  Removing ribosomal genes
# 2.  Removing mitochondrial genes
#
# Creating adata object split by developmental stage

# %%
# %%time

subset_adata_file = results_dir.joinpath('epi_subset.h5ad')
column_of_interest = 'development_stage'
color_obs_by = 'celltype'

if not subset_adata_file.exists():
    subset = adata[(adata.obs.celltype.isin(['AT1', 'AT2', 'Club', 'epi progenitor'])) & \
    (adata.obs.development_stage.isin(['E12', 'E15', 'E17', 'P3', 'P7'])),
    ~(adata.var.name.str.contains('^mt-') |
      adata.var.name.str.contains('^Mrp[ls]\d') |
      adata.var.name.str.contains('^Rp[ls]\d'))].copy()

    # remove unutilized genes and recalculate counts statistics
    sc.pp.filter_genes(subset, min_cells=1)
    sc.pp.filter_genes(subset, min_counts=1)
    sc.pp.filter_cells(subset, min_genes=1)
    sc.pp.filter_cells(subset, min_counts=1)
    subset.obs.n_genes_by_counts = subset.obs.n_genes.copy()
    subset.obs.total_counts = subset.obs.n_counts.copy()

    sc.pp.highly_variable_genes(subset, flavor='seurat_v3', n_top_genes=5000)
    
    _utils.joint_hvg_across_stages(subset, obs_category_key=column_of_interest, n_top_genes=5000)

    subset.write(subset_adata_file)
else:
    subset = sc.read_h5ad(subset_adata_file)


with warnings.catch_warnings():  # supress scanpy plotting warning
    warnings.simplefilter(action='ignore', category=UserWarning)
    
    # umap by celltype:
    sc.pl.umap(subset, color=[column_of_interest, color_obs_by])
    
    # statistics
    stats_df = subset.obs.loc[:, [column_of_interest, 'n_genes', 'n_counts']].groupby(
        [column_of_interest]).median()
    
    stats_df = pd.concat([subset.obs.groupby([column_of_interest]).count().iloc[:, 0],
                          stats_df], axis=1)
    stats_df.columns = ['# cells', 'median # genes', 'median # counts']
    
    stats_df.plot(kind='bar', title=f'Subset {column_of_interest} statistics', log=True, ylim=((1e2, 2e4)))
    plt.show()
    del stats_df

subset.shape

# %%
# %%time

categories = subset.obs[column_of_interest].cat.categories

split_adatas_dir = _utils.set_dir(results_dir.joinpath(f'split_{column_of_interest}'))

for cat in categories:
    if not split_adatas_dir.joinpath(f'{cat}.h5ad').exists():
        print(f'working on {cat}')
        tmp = subset[subset.obs[column_of_interest] == cat].copy()

        tmp.uns['name'] = f'{cat}'   # full name
        tmp.uns['sname'] = f'{cat[:3]}'  # short name, here it is the same

        # correcting the gene counts
        sc.pp.filter_genes(tmp, min_cells=0)
        sc.pp.filter_genes(tmp, min_counts=0)
        sc.pp.highly_variable_genes(tmp, flavor='seurat_v3', n_top_genes=5000)

        tmp.write_h5ad(split_adatas_dir.joinpath(f'{cat}.h5ad'))

        del tmp
    else:
        print(f'{cat} split adata exists')

# %% [markdown]
# ## 4. Running consensus NMF iterations

# %%
cnmf_dir = _utils.set_dir(results_dir.joinpath('cnmf'))
beta_loss = 'kullback-leibler'
tpm_target_sum = 5_000
nmf_iterations = 100

# %%
# %%time

ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]#, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

for cat in categories:
    print(f'Starting on {cat}, time is {time.strftime("%H:%M:%S", time.localtime())}')
    tmp = sc.read_h5ad(split_adatas_dir.joinpath(f'{cat}.h5ad'))
    
    c_object = cnmf.cNMF(cnmf_dir, cat)
    
    # Variance-capped normalized version of the data
    X = _utils.subset_and_normalize_for_nmf(tmp, method='variance_cap')
    
    c_object.prepare(X, ks, n_iter=nmf_iterations, new_nmf_kwargs={
        'tol': _constants.NMF_TOLERANCE, 'beta_loss': beta_loss, 'max_iter': 1000})
    
    c_object.factorize(0, 1, gpu=True)
    
    c_object.combine()
    
    del tmp, X

# %%
# %%time
thresh = 0.5

for cat in categories:
    print(f'Starting on {cat}, time is {time.strftime("%H:%M:%S", time.localtime())}')
    c_object = cnmf.cNMF(cnmf_dir, cat)

    c_object.k_selection_plot(density_threshold=thresh, nmf_refitting_iters=1000, 
                              consensus_method='mean',
                              close_fig=True, show_clustering=True, gpu=True)
    # printing the selected knee point
    df = cnmf.load_df_from_npz(c_object.paths['k_selection_stats_dt'] % c_object.convert_dt_to_str(thresh))
    pos = len(df) - 4
    for i in range(4 + 1):
        print(cnmf.find_knee_point(df.prediction_error[:pos + i], df.k_source[:pos + i]), end=", ")
    print()

# %% [markdown]
# ## 5. Selecting decomposition rank for cNMF using knee-point, silhouette and rank dynamics

# %%
column_of_interest = 'development_stage'
color_obs_by = 'celltype'

if 'subset' not in globals():
    subset_adata_file = results_dir.joinpath('epi_subset.h5ad')
    subset = sc.read_h5ad(subset_adata_file)

if 'categories' not in globals():
    categories = subset.obs[column_of_interest].cat.categories

if 'split_adatas' not in globals():
    split_adatas_dir = _utils.set_dir(results_dir.joinpath(f'split_{column_of_interest}'))

    split_adatas = {}
    for cat in categories:
        split_adatas[cat] =  sc.read_h5ad(split_adatas_dir.joinpath(f'{cat}.h5ad'))

if 'decompositions' not in globals():
    decompositions = {}
    for cat in categories:
        decompositions[cat] = {}


# %% [markdown]
# #### Calculating decompositions for a range of ranks around the relevant values

# %%
# %%time
threshold = 0.5

k_min = 3
k_max = 9

for cat in categories:
    print(f'Working on {cat}')
    tmp = split_adatas[cat]
    
    c_object = cnmf.cNMF(cnmf_dir, cat)
    
    for k in range(k_min, k_max + 1):
        if k in decompositions[cat].keys():
            continue

        print(f'processing k={k}')
        try:
            usages, spectra = c_object.get_consensus_usages_spectra(k, density_threshold=threshold)
        except FileNotFoundError:
            print(f'Calculating consensus NMF for k={k}')
            c_object.consensus(k, density_threshold=threshold, gpu=True, verbose=True,
                               consensus_method='mean',
                               nmf_refitting_iters=1000, show_clustering=False)

            usages, spectra = c_object.get_consensus_usages_spectra(k, density_threshold=threshold)

        X = cnmf.load_data_from_npz(c_object.paths['data'])
        # X ~ W @ H, transpose for cells to be columns
        loss_per_cell = pfnmf.calc_beta_divergence(
            X.T, W = spectra.T, H = usages.T, per_column=True)
    
        res = comparator.NMFResult(
            name=f'{tmp.uns["sname"]}_k{k}',
            loss_per_cell=loss_per_cell,
            rank=k,
            W=usages,
            H=spectra)
        
        comparator.NMFResultBase.calculate_gene_coefficients_list(
            tmp, [res], target_sum=tpm_target_sum, target_variance=tmp.var['variances_norm'].values)
        
        decompositions[cat][k] = res
    
    print()

np.savez(results_dir.joinpath('decompositions.npz'), obj=decompositions)

# %% [markdown]
# ### Examining results

# %%
decomposition_images = _utils.set_dir(split_adatas_dir.joinpath("decompositions"))

tsc_threshold: float = 0.35
tsc_truncation_level: int = 1000

for cat in categories:
    results = [decompositions[cat][i] for i in range(k_min, k_max + 1)]
    names_list = [res.name.split('_')[1] for res in results]
    ks, joint_names, joint_usages, joint_labels = comparator.NMFResultBase.aggregate_results(results)
    prog_names_dict = {res.name.split('_')[1]: [name.split('_')[1] for name in res.prog_names] for res in results}
    joint_names = [name.split('_')[1] for name in joint_names]
    
    # genes flow graph
    genes_title = f'{cat} flow chart of gene coefficients correlations for different decomposition ranks'
    genes_filename = f'{cat}_flow_chart_genes_by_rank.png'
    
    tsc = _utils.truncated_spearmans_correlation(pd.concat(
        [res.gene_coefs for res in results], axis = 1),
        truncation_level = tsc_truncation_level, rowvar = False)
    
    genes_adjacency = plotting.get_ordered_adjacency_matrix(
        tsc, joint_names, ks, tsc_threshold, verbose = True)
    
    fig = plotting.plot_layered_correlation_flow_chart(
        names_list, genes_adjacency, prog_names_dict, genes_title, layout_type='fan')
    
    fig.savefig(decomposition_images.joinpath(genes_filename))
    
    plt.close()

    # CDF of correlations
    plt.ecdf(tsc.flatten())
    plt.title(f'{cat}_flow_correlations_CDF.png')
    plt.savefig(decomposition_images.joinpath(f'{cat}_flow_correlations_CDF.png'))
    plt.close()


# %%
color_obs_by = 'celltype'

# Proximal: "Sox2", "Tspan1"
# Club: "Cyp2f2", "Scgb3a1",
# Ciliated: "Rsph1", "Foxj1"
# Distal: "Sox9", "Hopx"
# AT1: "Timp3", 'Aqp5'  
# AT2: 'Sftpa1', 'Sftpb'
# Cell Cycle: "Mki67", "Cdkn3", "Rrm2", "Lig1"
# Lineage markers: "Fxyd3", "Epcam", "Elf3", "Col1a2", "Dcn", "Mfap4", "Cd53", "Coro1a", "Ptprc", "Cldn5", "Clec14a", "Ecscr" 

marker_genes = ["Sox2", "Tspan1", "Cyp2f2", "Scgb3a1", "Rsph1", "Foxj1",
               "Sox9", "Hopx", "Timp3", 'Aqp5', 'Sftpa1', 'Sftpb',
               "Mki67", "Cdkn3", "Rrm2", "Lig1"]

with warnings.catch_warnings():  # supress plotting warnings
    warnings.simplefilter(action='ignore', category=UserWarning)

    for cat in categories:
        tmp = split_adatas[cat]
        for k in range(k_min, k_max + 1):
            res = decompositions[cat][k]
    
            # usages clustermap
            un_sns = _utils.plot_usages_norm_clustermaps(
                tmp, normalized_usages=res.norm_usages, prog_names=res.prog_names,
                title=f'{cat}', show=False, sns_clustermap_params={
                    'row_colors': tmp.obs[color_obs_by].map(tmp.uns[f'{color_obs_by}_colors_dict'])})
            un_sns.savefig(decomposition_images.joinpath(f"{cat}_{k}_usages_norm.png"),
                           dpi=180, bbox_inches='tight')
            plt.close(un_sns.fig)
    
            # usages violin plot
            _utils.plot_usages_norm_violin(
                tmp, color_obs_by, normalized_usages=res.norm_usages, prog_names=res.prog_names,
                save_path=decomposition_images.joinpath(
                    f'{cat}_{k}_norm_usage_per_lineage.png'))

            # Marker genes heatmap
            hm = sns.heatmap(res.gene_coefs.loc[marker_genes].T, cmap='coolwarm', vmin=-2, vmax=2)
            plt.tight_layout()
            hm.figure.savefig(decomposition_images.joinpath(f'{cat}_{k}_marker_genes.png'))
            plt.close()

        # UMAP of cells
        um = sc.pl.umap(tmp, color=color_obs_by, s=10, return_fig=True, title=f'{cat} epithelial')
        plt.tight_layout()
        um.savefig(decomposition_images.joinpath(f"{cat}_umap_{color_obs_by}.png"), dpi=300)
        plt.close(um)


# %% [markdown]
# ### Selecting final parameters

# %%
decompositions = np.load(results_dir.joinpath('decompositions.npz'), allow_pickle=True)['obj'].item()
decompositions

# %%
thresh = 0.5
selected_cnmf_params = {
    'E12': (5, 0.5),
    'E15': (5, 0.5),
    'E17': (5, 0.5),
    'P3': (5, 0.5),
    'P7': (5, 0.5)}

selected_cnmf_params

# %%
# %%time

for cat, (k, threshold) in selected_cnmf_params.items():
    print(f'Working on {cat} with k={k} and threshold={threshold}')

    tmp = split_adatas[cat]
    
    c_object = cnmf.cNMF(cnmf_dir, cat)

    try:
        usages, spectra = c_object.get_consensus_usages_spectra(k, density_threshold=threshold)
    except FileNotFoundError:
        print(f'Calculating consensus NMF for k={k}')
        c_object.consensus(k, density_threshold=threshold, gpu=True, verbose=True,
                           consensus_method='mean',
                           nmf_refitting_iters=1000, show_clustering=False)

        usages, spectra = c_object.get_consensus_usages_spectra(k, density_threshold=threshold)
    
    tmp.uns['cnmf_params'] = {'k_nmf': k, 'threshold': threshold}

    if k not in decompositions[cat]:
        X = cnmf.load_data_from_npz(c_object.paths['data'])
        
        # X ~ W @ H, transpose for cells to be columns
        loss_per_cell = pfnmf.calc_beta_divergence(
            X.T, W = spectra.T, H = usages.T, beta_loss=beta_loss, per_column=True)
    
        res = comparator.NMFResult(
            name=f'{tmp.uns["sname"]}_k{k}',
            loss_per_cell=loss_per_cell,
            rank=k,
            W=usages,
            H=spectra)
        
        comparator.NMFResultBase.calculate_gene_coefficients_list(
            tmp, [res], target_sum=tpm_target_sum, target_variance=tmp.var['variances_norm'].values)
    
        decompositions[cat][k] = res

    tmp.write_h5ad(split_adatas_dir.joinpath(f'{cat}.h5ad'))

    print()

np.savez(results_dir.joinpath('decompositions.npz'), obj=decompositions)


# %%
for cat in categories:
    print(cat)
    res = decompositions[cat][split_adatas[cat].uns['cnmf_params']['k_nmf']]
    with np.printoptions(precision=2, suppress=False):
        print(res.prog_percentages)

# %% [markdown]
# ## 6. Running comparator on the data
#

# %%
# %%time 
# loading general variables

if 'results_dir' not in globals():
    results_dir = _utils.set_dir('results')
    results_dir = _utils.set_dir(results_dir.joinpath('zepp'))

if 'column_of_interest' not in globals():
    column_of_interest = 'development_stage'

if 'categories' not in globals():
    categories = ['E12', 'E15', 'E17', 'P3', 'P7']

if 'split_adatas' not in globals():
    split_adatas_dir = _utils.set_dir(results_dir.joinpath(f'split_{column_of_interest}'))

    split_adatas = {}
    for cat in categories:
        split_adatas[cat] =  sc.read_h5ad(split_adatas_dir.joinpath(f'{cat}.h5ad'))

if 'decompositions' not in globals():
    decompositions = np.load(results_dir.joinpath('decompositions.npz'), allow_pickle=True)['obj'].item()

# %%
# Adding coloring of data rows by cell type

for cat in categories:
    tmp = split_adatas[cat]

    field_1 = 'celltype'

    tmp.obsm['row_colors'] = pd.concat([
        tmp.obs[field_1].map(tmp.uns[f'{field_1}_colors_dict']),
        ], axis=1)

# %%
# %%time

pairs = [(categories[i], categories[i + 1]) for i in range(len(categories) - 1)]

pairs.extend((j, i) for i, j in pairs[::-1])

marker_genes = ["Sox2", "Tspan1", "Cyp2f2", "Scgb3a1", "Rsph1", "Foxj1",
               "Sox9", "Hopx", "Timp3", 'Aqp5', 'Sftpa1', 'Sftpb',
               "Mki67", "Cdkn3", "Rrm2", "Lig1"]


for cat_a, cat_b in pairs:
    print(f'comparing {cat_a} and {cat_b}')
    comparison_dir = _utils.set_dir(results_dir.joinpath(f"comparator_{cat_a}_{cat_b}"))
    
    adata_a = split_adatas[cat_a]
    adata_b = split_adatas[cat_b]
    
    if os.path.exists(comparison_dir.joinpath('comparator.npz')):
        cmp = comparator.Comparator.load_from_file(comparison_dir.joinpath('comparator.npz'), adata_a, adata_b)
    else:
        cmp = comparator.Comparator(
            comparison_dir, adata_a, decompositions[cat_a][adata_a.uns['cnmf_params']['k_nmf']],
            max_added_rank=1,
            highly_variable_genes_key='joint_highly_variable',
            adata_b=adata_b, usages_matrix_b=decompositions[cat_b][adata_b.uns['cnmf_params']['k_nmf']],
            tpm_target_sum=tpm_target_sum,
            nmf_engine='torchnmf', device='cuda', max_nmf_iter=1000, verbosity=2,
            decomposition_normalization_method='variance_cap',
            coefs_variance_normalization='variances_norm')
    
        print('decomposing')
        cmp.extract_geps_on_jointly_hvgs()
        
        # getting cnmf results for the varius ranks
        c_object = cnmf.cNMF(cnmf_dir, cat_b)
        
        threshold = adata_b.uns['cnmf_params']['threshold']
        
        for k in range(cmp.rank_b, cmp.rank_b + cmp.max_added_rank + 1):
            if k in decompositions[cat_b].keys():
                continue  # we have a result object for this decomposition
            
            try:  # getting pre-calculated cNMF results
                usages, spectra = c_object.get_consensus_usages_spectra(k, density_threshold=threshold)
            except FileNotFoundError:
                print(f'Calculating consensus NMF for k={k} and threshold={threshold}')
                c_object.consensus(k, density_threshold=threshold, gpu=True, verbose=True,
                                   consensus_method='mean',
                                   nmf_refitting_iters=1000, show_clustering=False)
    
                usages, spectra = c_object.get_consensus_usages_spectra(k, density_threshold=threshold)

            X = cnmf.load_data_from_npz(c_object.paths['data'])
            
            # X ~ W @ H, transpose for cells to be columns
            loss_per_cell = pfnmf.calc_beta_divergence(
                X.T, W = spectra.T, H = usages.T, beta_loss=beta_loss, per_column=True)
        
            res = comparator.NMFResult(
                name=f'{tmp.uns["sname"]}_k{k}',
                loss_per_cell=loss_per_cell,
                rank=k,
                W=usages,
                H=spectra)
            
            comparator.NMFResultBase.calculate_gene_coefficients_list(
                adata_b, [res], target_sum=cmp.tpm_target_sum,
                target_variance=tmp.var['variances_norm'].values)
            
            decompositions[cat_b][k] = res
            np.savez(results_dir.joinpath('decompositions.npz'), obj=decompositions)
        
        cmp.decompose_b(repeats = 5, precalculated_denovo_usage_matrices={k: res.W for k, res in decompositions[cat_b].items()})
    
        cmp.save_to_file(comparison_dir.joinpath('comparator.npz'))

    cmp.print_errors()
    cmp.calculate_fingerprints()
    
    print('running GSEA')
    cmp.run_gsea(gprofiler_kwargs=dict(organism='mmusculus', sources=['GO:BP', 'WP', 'REAC', 'KEGG']))

    with warnings.catch_warnings():  # supress plotting warnings
        warnings.simplefilter(action='ignore', category=UserWarning)
        warnings.simplefilter(action='ignore', category=FutureWarning)
    
        cmp.examine_adata_a_decomposition_on_jointly_hvgs(35, 3500)
        cmp.examine_adata_b_decompositions(3500, 35, 3500)
        
        cmp.plot_decomposition_comparisons()
    
        cmp.plot_marker_genes_heatmaps(marker_genes)
    
        cmp.plot_usages_violin('celltype', show=False)

        cmp.plot_utilization_scatters('X_umap')
    
    cmp.save_to_file(comparison_dir.joinpath('comparator.npz'))



# %%
for cat_a, cat_b in pairs:
    print(f'comparing {cat_a} and {cat_b}')
    comparison_dir = _utils.set_dir(results_dir.joinpath(f"comparator_{cat_a}_{cat_b}"))
    
    adata_a = split_adatas[cat_a]
    adata_b = split_adatas[cat_b]
    
    if os.path.exists(comparison_dir.joinpath('comparator.npz')):
        cmp = comparator.Comparator.load_from_file(comparison_dir.joinpath('comparator.npz'), adata_a, adata_b)
        print(cmp)


# %%
