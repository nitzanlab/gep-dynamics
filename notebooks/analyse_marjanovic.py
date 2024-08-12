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
# # Downloading, pre-processing and running cNMF on marjanovic et. al 2020 data
# 1. Obtaining the data and creating AnnData object
# 2. filtering genes, showing key statistics and selecting joint highly variable genes (jHVGs)
# 3. Splitting the dataset by timepoints
# 3. Running consensus NMF (cNMF) per timepoint
# 4. Selecting parameters for the cNMF
# 5. Running the comparator for adjacent time points
#
#

# %%
# %%time

#debug jupyter:
# %load_ext autoreload
# %autoreload 2

#debug IDE:
from importlib import reload

import sys
import os
import time
import warnings
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

sc.settings.n_jobs = -1

from gepdynamics import _utils, _constants, cnmf, pfnmf, comparator, plotting

_utils.cd_proj_home()
print(os.getcwd())


# %% [markdown]
# ### 1. Downloading or loading AnnData object

# %%
results_dir = _utils.set_dir('results')
results_dir = _utils.set_dir(results_dir.joinpath('marjanovic'))
data_dir = _utils.set_dir('data')

# %%
# %%time

orig_adata_path = data_dir.joinpath('marjanovic_mmLungPlate.h5ad')

if not orig_adata_path.exists():  # create the original adata if it doesn't exist
    print('Source AnnData object does not exist, creating it')
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
    
    gene_ids = pd.read_csv(f_geneTable, index_col=None)
    gene_ids.index = gene_ids.ensgID.str.split('.').str[0]
    gene_ids.index.name = None
    smp_ids = pd.read_csv(f_smpTable, index_col=0)
    smp_annotation = pd.read_csv(f_smp_annot, index_col=0)
    
    # constructing the adata
    adata = sc.AnnData(X=sparse_counts.astype(np.float32), var=gene_ids, obs=smp_ids)

    # remove genes with 0 counts
    adata = adata[:, adata.X.sum(axis=0) > 0].copy()
    
    adata.obs['clusterK12'] = smp_annotation.clusterK12.astype('category')
    
    adata.obsm['X_tsne'] = smp_annotation[['tSNE_1', 'tSNE_2']].values
    adata.obsm['X_phate'] = smp_annotation[['phate_1', 'phate_2']].values
    adata.write(orig_adata_path)

    del sparse_counts, gene_ids, smp_ids, smp_annotation
else:
    adata = sc.read_h5ad(orig_adata_path)

adata

# %%
adata.obs.timesimple.replace({'01_T_early_ND': '00_All_early', '02_KorKP_early_ND': '00_All_early'}, inplace=True)
adata.obs['timesimple'].cat.categories

# %%

# %%
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    sc.external.pl.phate(adata, color=['clusterK12', 'timesimple'])

adata.uns['timesimple_colors_dict'] = dict(zip(adata.obs['timesimple'].cat.categories, adata.uns['timesimple_colors']))
adata.uns['clusterK12_colors_dict'] = dict(zip(adata.obs['clusterK12'].cat.categories, adata.uns['clusterK12_colors']))

# %%
pd.crosstab(adata.obs.timesimple, adata.obs.clusterK12)

# %%
adata.var['mt'] = adata.var.geneSymbol.str.startswith('mt-')  # annotate the group of mitochondrial genes as 'mt'
print(f"{np.sum(adata.var['mt'])} mitochondrial genes")
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                 jitter=0.4, multi_panel=True, groupby='timesimple')

# %%
column_of_interest = 'timesimple'

stats_df = adata.obs.loc[:, [column_of_interest, 'n_genes_by_counts', 'total_counts']].groupby(
    [column_of_interest]).median()

stats_df = pd.concat([adata.obs.groupby([column_of_interest]).count().iloc[:, 0],
                      stats_df], axis=1)
stats_df.columns = ['# cells', 'median # genes', 'median # counts']

stats_df.plot(kind='bar', title=f'{column_of_interest} statistics', log=True, ylim=((1e2, 2e6)))
plt.show()
del column_of_interest, stats_df

# %%
sc.pl.highest_expr_genes(adata, n_top=20, gene_symbols='geneID')

# %%
# rRNA overlapping gene 
sc.external.pl.phate(adata, color='ENSMUSG00000106106')
sc.pl.violin(adata, keys='ENSMUSG00000106106', groupby='timesimple')

# %% [markdown]
# ### 2. Filter genes and plot basic statistics
# Cells with low number of genes were already filtered

# %%
# %%time
print(f'before filtering shape was {adata.X.shape}')

# filter cells with high amount of mitochondrial genes or extremely high counts
adata = adata[(adata.obs.pct_counts_mt < 10) & (adata.obs.total_counts < 5E6)].copy()

# filter ribosomal and mitochondrial genes:
adata = adata[:, ~(adata.var.geneSymbol.str.contains('^mt-') |
    adata.var.geneSymbol.str.contains('^Mrp[ls]\d') |
    adata.var.geneSymbol.str.contains('^Rp[ls]\d') |
    (adata.var_names == 'ENSMUSG00000106106'))].copy()

# filtering genes with very low abundance
sc.pp.filter_genes(adata, min_cells=np.round(adata.shape[0] / 1000))

# re-setting general statistics for counts abundance
adata.obs.n_genes_by_counts = np.count_nonzero(adata.X.toarray(), axis=1)
adata.obs.total_counts = adata.X.toarray().sum(axis=1)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=_constants.NUMBER_HVG)

print(f'after filtering shape is {adata.X.shape}')


# %%
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                 jitter=0.4, multi_panel=True, groupby='timesimple')

# %%
column_of_interest = 'timesimple'

stats_df = adata.obs.loc[:, [column_of_interest, 'n_genes_by_counts', 'total_counts']].groupby(
    [column_of_interest]).median()

stats_df = pd.concat([adata.obs.groupby([column_of_interest]).count().iloc[:, 0],
                      stats_df], axis=1)
stats_df.columns = ['# cells', 'median # genes', 'median # counts']

stats_df.plot(kind='bar', title=f'{column_of_interest} statistics', log=True, ylim=((1e2, 2e6)))
plt.show()
del column_of_interest, stats_df

# %%
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    sc.external.pl.phate(adata, color=['clusterK12', 'timesimple'])

# %%
pd.crosstab(adata.obs.timesimple, adata.obs.clusterK12)

# %%
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    _utils.joint_hvg_across_stages(adata, obs_category_key='timesimple', n_top_genes=5000)
adata.var


# %%
prepared_adata_path = results_dir.joinpath('full.h5ad')

if not prepared_adata_path.exists():
    # Removing h5ad trouble saving element before saving
    adata.uns.pop('clusterK12_colors_dict', None)
    adata.write_h5ad(prepared_adata_path)
else:
    print('reading adata...')
    adata = sc.read_h5ad(prepared_adata_path)
    # restoring h5ad trouble saving element
    adata.uns['clusterK12_colors_dict'] = dict(zip(adata.obs['clusterK12'].cat.categories, adata.uns['clusterK12_colors']))

adata

# %% [markdown]
# ### 3. Splitting the adata by "timesimple"

# %%
short_names_dict = {
    '00_All_early': 'T0',
    '04_K_12w_ND': 'K12',
    '05_K_30w_ND': 'K30',
    '06_KP_12w_ND': 'KP12',
    '07_KP_20w_ND': 'KP20',
    '08_KP_30w_ND': 'KP30'}

# %%
# %%time

column_of_interest = 'timesimple'
categories = adata.obs[column_of_interest].cat.categories

split_adatas_dir = _utils.set_dir(results_dir.joinpath(f'split_{column_of_interest}'))

for cat in categories:
    if not split_adatas_dir.joinpath(f'{cat}.h5ad').exists():
        print(f'working on {cat}')
        tmp = adata[adata.obs[column_of_interest] == cat].copy()

        tmp.uns['name'] = f'{cat}'   # full name
        tmp.uns['sname'] = short_names_dict[cat]  # short name

        # correcting the gene counts
        sc.pp.filter_genes(tmp, min_cells=0)
        sc.pp.filter_genes(tmp, min_counts=0)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            sc.pp.highly_variable_genes(tmp, flavor='seurat_v3', n_top_genes=5000)

        # Removing h5ad trouble saving element before saving
        tmp.uns.pop('clusterK12_colors_dict', None)
        tmp.write_h5ad(split_adatas_dir.joinpath(f'{cat}.h5ad'))

        del tmp
    else:
        print(f'{cat} split adata exists')


# %% [markdown]
# ### 4. Running multiple NMF iterations

# %%
cnmf_dir = _utils.set_dir(results_dir.joinpath('cnmf'))
beta_loss = 'kullback-leibler'
tpm_target_sum = 1_000_000


# %%
# %%time

ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

for cat in categories:
    print(f'Starting on {cat}, time is {time.strftime("%H:%M:%S", time.localtime())}')
    tmp = sc.read_h5ad(split_adatas_dir.joinpath(f'{cat}.h5ad'))
    
    c_object = cnmf.cNMF(cnmf_dir, cat)
    
    # Variance normalized version of the data
    X = _utils.subset_and_normalize_for_nmf(tmp, method='variance_cap')
    
    c_object.prepare(X, ks, n_iter=150, new_nmf_kwargs={
        'tol': _constants.NMF_TOLERANCE, 'beta_loss': beta_loss, 'max_iter': 1000})
    
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
        c_object.k_selection_plot(density_threshold=thresh, nmf_refitting_iters=500, 
                                  consensus_method='mean',
                                  close_fig=True, show_clustering=True, gpu=True)
        # printing the selected knee point
        df = cnmf.load_df_from_npz(c_object.paths['k_selection_stats_dt'] % c_object.convert_dt_to_str(thresh))
        pos = len(df) - 4
        for i in range(5):
            print(cnmf.find_knee_point(df.prediction_error[:pos + i], df.k_source[:pos + i]), end=", ")
        print()


# %% [markdown]
# ## 5. Selecting decomposition ranks for cNMF using knee-point and silhouette
#

# %%
if 'split_adatas' not in globals():
    print('creating split_adatas')
    
    split_adatas_dir = _utils.set_dir(results_dir.joinpath(f'split_{column_of_interest}'))
    split_adatas = {}
    
    for cat in categories:
        tmp = sc.read_h5ad(split_adatas_dir.joinpath(f'{cat}.h5ad'))
        
        # restoring h5ad trouble saving element
        tmp.uns['clusterK12_colors_dict'] = dict(zip(tmp.obs['clusterK12'].cat.categories, tmp.uns['clusterK12_colors']))
        
        split_adatas[cat] = tmp

if 'decompositions' not in globals():
    decompositions = {}
    for cat in categories:
        decompositions[cat] = {}


# %% [markdown]
# #### Examining programs dynamics by rank
#

# %%
# %%time
threshold = 0.5

k_min = 2
k_max = 8

for cat in categories:
    print(f'Working on {cat}')
    tmp = split_adatas[cat]
    
    c_object = cnmf.cNMF(cnmf_dir, cat)
    
    for k in range(k_min, k_max + 1):
        if k in decompositions[cat].keys():
            continue

        print(f'Working on k={k}')
        try:
            usages, spectra = c_object.get_consensus_usages_spectra(k, density_threshold=threshold)
        except FileNotFoundError:
            print(f'Calculating consensus NMF for k={k}')
            c_object.consensus(k, density_threshold=threshold, gpu=True, verbose=True,
                               consensus_method='mean',
                               nmf_refitting_iters=1000, show_clustering=False)

            usages, spectra = c_object.get_consensus_usages_spectra(k, density_threshold=threshold)
            
        # X ~ W @ H, transpose for cells to be columns
        loss_per_cell = pfnmf.calc_beta_divergence(
            c_object.X.T, W = spectra.T, H = usages.T, per_column=True)
    
        res = comparator.NMFResult(
            name=f'{tmp.uns["sname"]}_k{k}',
            loss_per_cell=loss_per_cell,
            rank=k,
            W=usages,
            H=spectra)
        
        comparator.NMFResultBase.calculate_gene_coefficients_list(
            tmp, [res], target_sum=1_000_000, target_variance=tmp.var['variances_norm'].values)
        
        decompositions[cat][k] = res
    
    print()

np.savez(results_dir.joinpath('decompositions.npz'), obj=decompositions)

# %%
decomposition_images = _utils.set_dir(split_adatas_dir.joinpath("images"))

tsc_threshold: float = 0.3
tsc_truncation_level: int = 500

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
color_obs_by = 'clusterK12'

# Proximal: "Sox2", "Tspan1"
# Club: "Cyp2f2", "Scgb3a1",
# Ciliated: "Rsph1", "Foxj1"
# Distal: "Sox9", "Hopx"
# AT1: "Timp3", 'Aqp5'  
# AT2: 'Sftpa1', 'Sftpb'
# Cell Cycle: "Mki67", "Cdkn3", "Rrm2", "Lig1"
# Lineage markers: "Fxyd3", "Epcam", "Elf3", "Col1a2", "Dcn", "Mfap4", "Cd53", "Coro1a", "Ptprc", "Cldn5", "Clec14a", "Ecscr" 

marker_genes_symbols = ["Sox2", "Tspan1", "Cyp2f2", "Scgb3a1", "Rsph1", "Foxj1",
                        "Sox9", "Hopx", "Timp3", 'Aqp5', 'Sftpa1', 'Sftpb',
                        "Mki67", "Cdkn3", "Rrm2", "Lig1", "H2-Aa", "H2-Ab1",
                        "Fxyd3", "Epcam", "Elf3", "Col1a2", "Dcn", "Mfap4",
                        "Cd53", "Coro1a", "Ptprc", "Cldn5", "Clec14a", "Ecscr"]

marker_genes_ID = [adata.var.index[adata.var['geneSymbol'] == gene].tolist()[0] for gene in marker_genes_symbols]


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
            heatmap_data = res.gene_coefs.loc[marker_genes_ID]
            hm = sns.heatmap(heatmap_data, cmap='coolwarm', vmin=-2, vmax=2)

            plt.yticks(0.5 + np.arange(len(marker_genes_symbols)), marker_genes_symbols)

            plt.title(f'Marker genes coefficients for {res.name}')
            plt.tight_layout()
            
            hm.figure.savefig(decomposition_images.joinpath(f'{cat}_{k}_marker_genes.png'))
            plt.close()

        # Phate
        um = sc.external.pl.phate(tmp, color=color_obs_by, s=10, return_fig=True, title=f'{cat}')
        plt.tight_layout()
        um.savefig(decomposition_images.joinpath(f"{cat}_phate_{color_obs_by}.png"), dpi=300)
        plt.close(um)

# %% [markdown]
# #### selecting final parameters

# %%
selected_cnmf_params = {
    '00_All_early': (4, 0.5),
    '04_K_12w_ND': (4, 0.5),
    '05_K_30w_ND': (4, 0.5),
    '06_KP_12w_ND': (5, 0.5),
    '07_KP_20w_ND': (5, 0.5),
    '08_KP_30w_ND': (6, 0.5)}

selected_cnmf_params

# %%
# %%time


for cat, (k, threshold) in selected_cnmf_params.items():
    print(f'Working on {cat} with k={k} and threshold={threshold}')
    tmp = split_adatas[cat]

    if k not in decompositions[cat].keys():
        c_object = cnmf.cNMF(cnmf_dir, cat)
        c_object.consensus(k, density_threshold=threshold, gpu=True, verbose=True,
                           consensus_method='mean',
                           nmf_refitting_iters=1000, show_clustering=False)
    
        usages, spectra = c_object.get_consensus_usages_spectra(k, density_threshold=threshold)
        
        # X ~ W @ H, transpose for cells to be columns
        loss_per_cell = pfnmf.calc_beta_divergence(
            c_object.X.T, W = spectra.T, H = usages.T, beta_loss=beta_loss, per_column=True)
    
        res = comparator.NMFResult(
            name=f'{tmp.uns["sname"]}_k{k}',
            loss_per_cell=loss_per_cell,
            rank=k,
            W=usages,
            H=spectra)
            
        comparator.NMFResultBase.calculate_gene_coefficients_list(
            tmp, [res], target_sum=tpm_target_sum, target_variance=tmp.var['variances_norm'].values)
        
        decompositions[cat][k] = res

    tmp.uns['cnmf_params'] = {'k_nmf': k, 'threshold': threshold}

    # Saving
    # Removing h5ad trouble saving element before saving
    tmp.uns.pop('clusterK12_colors_dict', None)
    
    tmp.write_h5ad(split_adatas_dir.joinpath(f'{cat}.h5ad'))

    # restoring h5ad trouble saving element
    tmp.uns['clusterK12_colors_dict'] = dict(zip(tmp.obs['clusterK12'].cat.categories, tmp.uns['clusterK12_colors']))
        
    print()

# %%
np.savez(results_dir.joinpath('decompositions.npz'), obj=decompositions)

# %% [markdown]
# #### Reloading the results

# %% Loading GEPs adatas
# %%time

column_of_interest = 'timesimple'
categories = adata.obs[column_of_interest].cat.categories

color_obs_by = 'clusterK12'

if 'split_adatas' not in globals():
    split_adatas_dir = _utils.set_dir(results_dir.joinpath(f'split_{column_of_interest}'))

    split_adatas = {}
    for cat in categories:
        tmp = sc.read_h5ad(split_adatas_dir.joinpath(f'{cat}.h5ad'))
        # restoring h5ad trouble saving element
        tmp.uns['clusterK12_colors_dict'] = dict(zip(tmp.obs['clusterK12'].cat.categories, tmp.uns['clusterK12_colors']))
    
        split_adatas[cat] = tmp

if 'decompositions' not in globals():
    decompositions = np.load(results_dir.joinpath('decompositions.npz'), allow_pickle=True)['obj'].item()

# %% [markdown]
# #### Examening results

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
for cat in categories:
    tmp = split_adatas[cat]

    field_1 = color_obs_by

    tmp.obsm['row_colors'] = pd.concat([
        tmp.obs[field_1].map(tmp.uns[f'{field_1}_colors_dict']),
        ], axis=1)



# %%
categories

# %%
# %%time

pairs = [(categories[i], categories[i + 1]) for i in range(len(categories) - 1)]
pairs.extend((j, i) for i, j in pairs[::-1])

marker_genes_symbols = ["Sox2", "Tspan1", "Cyp2f2", "Scgb3a1", "Rsph1", "Foxj1",
                        "Sox9", "Hopx", "Timp3", 'Aqp5', 'Sftpa1', 'Sftpb',
                        "Mki67", "Cdkn3", "Rrm2", "Lig1", "H2-Aa", "H2-Ab1",
                        "Fxyd3", "Epcam", "Elf3", "Col1a2", "Dcn", "Mfap4",
                        "Cd53", "Coro1a", "Ptprc", "Cldn5", "Clec14a", "Ecscr"]

marker_genes_ID = [adata.var.index[adata.var['geneSymbol'] == gene].tolist()[0] for gene in marker_genes_symbols]


for cat_a, cat_b in pairs:
    print(f'comparing {cat_a} and {cat_b}')
    
    adata_a = split_adatas[cat_a]
    adata_b = split_adatas[cat_b]
    
    comparison_dir = _utils.set_dir(results_dir.joinpath(
        f"comparator_{adata_a.uns['sname']}_{adata_b.uns['sname']}"))
    
    if os.path.exists(comparison_dir.joinpath('comparator.npz')):
        continue
        # cmp = comparator.Comparator.load_from_file(comparison_dir.joinpath('comparator.npz'), adata_a, adata_b)
    else:
        cmp = comparator.Comparator(
            comparison_dir, adata_a, decompositions[cat_a][adata_a.uns['cnmf_params']['k_nmf']],
            highly_variable_genes_key='joint_highly_variable',
            adata_b=adata_b, usages_matrix_b=decompositions[cat_b][adata_b.uns['cnmf_params']['k_nmf']],
            tpm_target_sum=tpm_target_sum,
            nmf_engine='torchnmf', device='cuda', max_nmf_iter=500, verbosity=1,
            decomposition_normalization_method='variance_cap',
            coefs_variance_normalization='variances_norm')
    
        print('decomposing')
        cmp.extract_geps_on_jointly_hvgs()
        
        # getting cnmf results
        c_object = cnmf.cNMF(cnmf_dir, cat_b)

        if not hasattr(c_object, 'X'):
            c_object.X = cnmf.load_data_from_npz(c_object.paths['data'])
        
        threshold = adata_b.uns['cnmf_params']['threshold']
        for k in range(cmp.rank_b, cmp.rank_b + cmp.max_added_rank + 1):
            if k in decompositions[cat_b].keys():
                continue
            
            try:
                usages, spectra = c_object.get_consensus_usages_spectra(k, density_threshold=threshold)
            except FileNotFoundError:
                print(f'Calculating consensus NMF for k={k} and threshold={threshold}')
                c_object.consensus(k, density_threshold=threshold, gpu=True, verbose=True,
                                   consensus_method='mean',
                                   nmf_refitting_iters=1000, show_clustering=False)
    
                usages, spectra = c_object.get_consensus_usages_spectra(k, density_threshold=threshold)
            
            # X ~ W @ H, transpose for cells to be columns
            loss_per_cell = pfnmf.calc_beta_divergence(
                c_object.X.T, W = spectra.T, H = usages.T, beta_loss=beta_loss, per_column=True)
        
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
        
        
        cmp.decompose_b(repeats = 5, precalculated_denovo_usage_matrices={k: res.norm_usages for k, res in decompositions[cat_b].items()})
    
        cmp.save_to_file(comparison_dir.joinpath('comparator.npz'))

    cmp.print_errors()
    
    cmp.examine_adata_a_decomposition_on_jointly_hvgs(35, 3500)
    cmp.examine_adata_b_decompositions(3500, 35, 3500)
    
    cmp.plot_decomposition_comparisons()
    
    cmp.calculate_fingerprints()
    
    print('running GSEA')
    cmp.run_gsea(gene_ids_column_number=2, 
                 gprofiler_kwargs=dict(organism='mmusculus',
                                       sources=['GO:BP', 'WP', 'REAC', 'KEGG']))

    cmp.plot_marker_genes_heatmaps(marker_genes_ID, marker_genes_symbols)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        cmp.plot_usages_violin(color_obs_by, show=False)
    


# %%

for cat_a, cat_b in pairs:
    print(f'comparing {cat_a} and {cat_b}')
    
    adata_a = split_adatas[cat_a]
    adata_b = split_adatas[cat_b]
    
    comparison_dir = _utils.set_dir(results_dir.joinpath(
        f"comparator_{adata_a.uns['sname']}_{adata_b.uns['sname']}"))
    
    cmp = comparator.Comparator.load_from_file(comparison_dir.joinpath('comparator.npz'), adata_a, adata_b)
    
    break
