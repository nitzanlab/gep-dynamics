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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %%time
# %load_ext autoreload
# %autoreload 2

#debug:
from importlib import reload

import sys
import os
import time
import warnings

import numpy as np
import pandas as pd
# from scipy import sparse
# from sklearn.metrics import silhouette_samples
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=FutureWarning)

sc.settings.n_jobs=-1

from gepdynamics import _utils, _constants, cnmf, pfnmf, comparator, plotting

_utils.cd_proj_home()
print(os.getcwd())

# %%

results_dir = _utils.set_dir('results')
marjanovic_results_dir = _utils.set_dir(results_dir.joinpath('marjanovic'))

decompositions = np.load(marjanovic_results_dir.joinpath('decompositions.npz'), allow_pickle=True)['obj'].item()

adata = sc.read_h5ad(marjanovic_results_dir.joinpath('full.h5ad'))

column_of_interest = 'timesimple'
categories = adata.obs[column_of_interest].cat.categories

# %%

split_adatas_dir = _utils.set_dir(marjanovic_results_dir.joinpath(f'split_{column_of_interest}'))

split_adatas = {}
for cat in categories:
    tmp = sc.read_h5ad(split_adatas_dir.joinpath(f'{cat}.h5ad'))

    # restoring h5ad trouble saving element
    tmp.uns['clusterK12_colors_dict'] = dict(zip(adata.obs['clusterK12'].cat.categories, adata.uns['clusterK12_colors']))

    field_1 = 'clusterK12'

    tmp.obsm['row_colors'] = pd.concat([
        tmp.obs[field_1].map(tmp.uns[f'{field_1}_colors_dict']),
        ], axis=1)

    split_adatas[cat] = tmp

# %% getting copies of results with the correct program names

cp_rn = comparator.NMFResultBase.copy_and_rename_programs

res_t0 = cp_rn(decompositions['00_All_early'][4],
               ['T0_AT2', 'T0_Lymphocyte', 'T0_CellCycle', 'T0_Interferon_HPCS'])
res_k12 = cp_rn(decompositions['04_K_12w_ND'][4],
                ["K12_HPCS", "K12_AT2", "K12_CellCycle", "K12_AT1_ECM"])
res_k30 = cp_rn(decompositions['05_K_30w_ND'][4],
                ["K30_HPCS", "K30_AT2", "K30_Metabolism1", "K30_Metabolism2"])
res_kp12 = cp_rn(decompositions['06_KP_12w_ND'][5],
                 ["KP12_Cluster6", "KP12_AT2", "KP12_Cluster8", "KP12_HPCS", "KP12_CellCycle"])
res_kp20 = cp_rn(decompositions['07_KP_20w_ND'][5],
                 ["KP20_HPCS", "KP20_CellCycle", "KP20_AT2_H2", "KP20_Cluster8", "KP20_Cluster9"])
res_kp30 = cp_rn(decompositions['08_KP_30w_ND'][6],
                    ["KP30_Cluster11", "KP30_HPCS", "KP30_Cluster8",
                        "KP30_Clusters_9_10", "KP30_AT2_Cluster12", "KP30_AT1"])

all_res = {'00_All_early': res_t0, '04_K_12w_ND': res_k12, '05_K_30w_ND': res_k30,
           '06_KP_12w_ND': res_kp12, '07_KP_20w_ND': res_kp20, '08_KP_30w_ND': res_kp30}

del cp_rn


#%% log average TP5k expression

for cat in categories:
    res = all_res[cat]

    X = split_adatas[cat].X.toarray()
    X = X * 1_000_000. / X.sum(axis=1, keepdims=True)

    nnls_reg = LinearRegression(positive=True, fit_intercept=False)

    data = nnls_reg.fit(res.norm_usages, X).coef_
    gene_importance = pd.DataFrame(np.log1p(data),
                                   index=res.gene_coefs.index, columns=res.gene_coefs.columns)
    res.gene_importance = gene_importance

    del nnls_reg

# %% Sanky plot

reload(plotting)

plotting.pio.renderers.default = 'browser'
# plotting.pio.renderers.default = 'svg'

plotting.plot_sankey_for_nmf_results(
    [res_t0, res_k12, res_k30, res_kp12, res_kp20, res_kp30],
    gene_list_cutoff=201,
    cutoff=401, # cutoff for coefficient ranks in comparison
    display_threshold_counts=75,
    show_unassigned_genes=True
)

#%% extra plots for all results

pairs = [(categories[i], categories[i + 1]) for i in range(len(categories) - 1)]
pairs.extend((j, i) for i, j in pairs[::-1])

marker_genes_symbols = ["Sox2", "Cyp2f2", "Scgb3a1", "Rsph1",
                        "Sox9", "Timp3", 'Aqp5', 'Sftpa1', 'Sftpc',
                        "Mki67", "Ube2c", "Tigit", "Slc4a11", "H2-Aa", "H2-Q7",
                        "Fxyd3", "Dcn", "Mfap4",
                        "Ptprc", "Cldn5", "Ecscr", "Igfbp4", "Fn1"]

marker_genes_ID = [adata.var.index[adata.var['geneSymbol'] == gene].tolist()[0] for gene in marker_genes_symbols]

for cat_a, cat_b in pairs:
    print(f'comparing {cat_a} and {cat_b}')

    adata_a = split_adatas[cat_a]
    adata_b = split_adatas[cat_b]

    comparator_dir = _utils.set_dir(marjanovic_results_dir.joinpath(
        f"comparator_{adata_a.uns['sname']}_{adata_b.uns['sname']}"))

    cmp = comparator.Comparator.load_from_file(comparator_dir.joinpath('comparator.npz'), adata_a, adata_b)

    cmp.plot_marker_genes_heatmaps(marker_genes_ID, marker_genes_symbols)
    cmp.plot_utilization_scatters('X_phate')


#%% per program marker genes TPM heatmap (no reordering)

for cat in categories:
    res = all_res[cat]
    k = res.rank

    # Marker genes heatmap
    plt.figure(figsize=(6, 6))
    hm = sns.heatmap(res.gene_importance.loc[marker_genes_ID], cmap='BuGn', vmax=12)
    plt.yticks(0.5 + np.arange(len(marker_genes_symbols)), marker_genes_symbols)
    plt.tight_layout()
    hm.figure.savefig(marjanovic_results_dir.joinpath(f'marker_genes_tpm_{cat}_{k}.png'))
    plt.close()

#%% per program marker genes heatmap (reordering)

cat = '06_KP_12w_ND'; k = 5; res = res_kp12
col_order = res.gene_coefs.columns[[0,4,1,2,3]]

# Marker genes heatmap
plt.figure(figsize=(6, 6))
hm = sns.heatmap(res.gene_coefs.loc[marker_genes_ID, col_order], cmap='coolwarm', vmin=-2, vmax=2)
plt.yticks(0.5 + np.arange(len(marker_genes_symbols)), marker_genes_symbols)
plt.tight_layout()
hm.figure.savefig(marjanovic_results_dir.joinpath(f'marker_genes_{cat}_{k}.png'))
plt.close()


#%% plotting marker genes coefs dynamics of AT2 and HPCS programs dynamics

reload(plotting)

empty_column = pd.Series(np.zeros(res_t0.gene_coefs.shape[0]), index=res_t0.gene_coefs.index, name='')

programs_list = [
    res_t0.gene_coefs['T0_Interferon_HPCS'],
    res_k12.gene_coefs['K12_HPCS'],
    res_k30.gene_coefs['K30_HPCS'],
    res_kp12.gene_coefs[['KP12_HPCS', 'KP12_Cluster6']],
    res_kp20.gene_coefs['KP20_HPCS'],
    res_kp30.gene_coefs['KP30_HPCS'],
    empty_column,
    res_t0.gene_coefs['T0_AT2'],
    res_k12.gene_coefs['K12_AT2'],
    res_k30.gene_coefs[['K30_AT2', 'K30_Metabolism2']],
    res_kp12.gene_coefs['KP12_AT2'],
    res_kp20.gene_coefs[['KP20_AT2_H2', 'KP20_Cluster8']],
    res_kp30.gene_coefs[['KP30_AT1', 'KP30_AT2_Cluster12']],
    ]

plotting.plot_marker_genes_heatmaps(
    programs_list, marker_genes_ID, marker_genes_symbols,
    title='Programs dynamic shown by marker genes', show=False,
    save_file=marjanovic_results_dir.joinpath('marker_genes_dynamics.png'))

#%% plotting marker genes TPM dynamics of AT2 and HPCS programs dynamics

programs_list = [
    res_t0.gene_importance['T0_Interferon_HPCS'],
    res_k12.gene_importance['K12_HPCS'],
    res_k30.gene_importance['K30_HPCS'],
    res_kp12.gene_importance[['KP12_HPCS', 'KP12_Cluster6']],
    res_kp20.gene_importance['KP20_HPCS'],
    res_kp30.gene_importance['KP30_HPCS'],
    empty_column,
    res_t0.gene_importance['T0_AT2'],
    res_k12.gene_importance['K12_AT2'],
    res_k30.gene_importance[['K30_Metabolism2', 'K30_AT2']],
    res_kp12.gene_importance['KP12_AT2'],
    res_kp20.gene_importance[['KP20_Cluster8', 'KP20_AT2_H2']],
    res_kp30.gene_importance[['KP30_AT1', 'KP30_AT2_Cluster12']],
    ]

#%%

marker_genes_symbols = ["Sox9",'Sftpa1', 'Sftpb', 'Sftpc', 'Sftpd',
                        'Irx3', 'Cebpa', 'Irx5', 'Etv5', 'Tfcp2l1', 'Dbp',
                        "Tigit", "Slc4a11", "Igfbp4", "Fn1",
'Lyz2',
 'Cbr2',
 'Sftpc',
 'Scd1',
 'Hc',
 'Slc34a2',
 'Chil1',
 'Chia1',
 'Itih4',
 'Lrg1',
 'Kcnj15',
 'Hp',
 'Acoxl',
 'Napsa',
 'Lcn2',
 'Lyz1',
 'Mfsd2a',
 'Nkd1',
 'Sftpa1',
 'Aox3'
                        ]

marker_genes_ID = [adata.var.index[adata.var['geneSymbol'] == gene].tolist()[0] for gene in marker_genes_symbols]

# Marker genes heatmap
plt.figure(figsize=(8, 6))
hm = sns.heatmap(pd.concat(programs_list, axis=1).loc[marker_genes_ID], cmap='BuGn', vmax=12)
plt.yticks(0.5 + np.arange(len(marker_genes_symbols)), marker_genes_symbols)

plt.title('Marker genes log-TPM dynamics', y=1.05)
plt.tight_layout()

hm.figure.savefig(marjanovic_results_dir.joinpath(f'marker_genes_tpm_dynamics 4.png'))
plt.close()

#%%

all_gene_importance = pd.concat([all_res[cat].gene_importance for cat in categories], axis=1)
all_gene_importance.index = adata.var['geneSymbol']

all_gene_importance.columns
AT2_matching_vector = all_gene_importance.columns.str.contains('AT2')
HPCS_matching_vector = all_gene_importance.columns.str.contains('HPCS')

# correlate all rows of all gene importance with the at2 matching vector
vec = HPCS_matching_vector.astype(np.float32)
vec = AT2_matching_vector.astype(np.float32)
vec -= vec.mean()
vec = vec / np.linalg.norm(vec)

rows = all_gene_importance.values.copy()
rows -= rows.mean(axis=1, keepdims=True)
rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)

corrs = pd.DataFrame(data=(rows @ vec * all_gene_importance.std(axis=1)), index=all_gene_importance.index, columns=['AT2_matching'])
corrs.sort_values('AT2_matching', ascending=False, inplace=True)

tmp = all_gene_importance.loc[corrs.index]

#%% comparing programs pairs

reload(comparator)
comparison_dir = _utils.set_dir(marjanovic_results_dir.joinpath('programs_comparisons'))

# create an instance of my gprofiler object for mus-musculus:
gp = _utils.MyGProfiler(organism='mmusculus', sources=['GO:BP', 'WP', 'REAC', 'KEGG'])

# T0 vs K12
for index_a, index_b in [(0,1), (1,2), (2,2), (3,0), (3,3)]:
    # (T0_AT2, K12_AT2), (T0_Lymphocyte, K12_CellCycle), (T0_CellCycle, K12_CellCycle),
    # (T0_Interferon_HPCS, K12_HPCS), (T0_Interferon_HPCS, K12_AT1_ECM)
    comparator.compare_programs(res_t0, index_a, res_k12, index_b, comparison_dir,
                                genes_symbols=adata.var['geneSymbol'], gp=gp, gprofiler_kwargs={'no_iea': True})

# K12 vs K30
for index_b, index_c in [(0,0), (1,1), (1,3), (2,2)]:
    # (K12_HPCS, K30_HPCS), (K12_AT2, K30_AT2), (K12_AT2, K30_Metabolism2), (K12_CellCycle, K30_Metabolism1)
    comparator.compare_programs(res_k12, index_b, res_k30, index_c, comparison_dir,
                                genes_symbols=adata.var['geneSymbol'], gp=gp, gprofiler_kwargs={'no_iea': True})

# K30 vs KP12
for index_c, index_d in [(0,0), (0,3), (1,1), (2,2), (3,1), (3,2)]:
    # (K30_HPCS, KP12_Cluster6), (K30_HPCS, KP12_HPCS), (K30_AT2, KP12_AT2),
    # (K30_Metabolism1, KP12_Cluster8), (K30_Metabolism2, KP12_AT2), (K30_Metabolism2, KP12_Cluster8)
    comparator.compare_programs(res_k30, index_c, res_kp12, index_d, comparison_dir,
                                genes_symbols=adata.var['geneSymbol'], gp=gp, gprofiler_kwargs={'no_iea': True})

# KP12 vs KP20
for index_d, index_e in [(0,0), (0,2), (0,4), (1,2), (1,3), (2,3), (2,2), (3,0), (4,1)]:
    # (KP12_Cluster6, KP20_HPCS), (KP12_Cluster6, KP20_AT2_H2), (KP12_Cluster6, KP20_Cluster9),
    # (KP12_AT2, KP20_AT2_H2), (KP12_AT2, KP20_Cluster8), (KP12_Cluster8, KP20_Cluster8),
    # (KP12_Cluster8, KP20_AT2_H2), (KP12_HPCS, KP20_HPCS), (KP12_CellCycle, KP20_CellCycle)
    comparator.compare_programs(res_kp12, index_d, res_kp20, index_e, comparison_dir,
                                genes_symbols=adata.var['geneSymbol'], gp=gp, gprofiler_kwargs={'no_iea': True})

# KP20 vs KP30
for index_e, index_f in [(0,1), (1,0), (2,5), (2,4), (3,2), (3,4), (4,3)]:
    # (KP20_HPCS, KP30_HPCS), (KP20_AT2_H2, KP30_Cluster11), (KP20_AT2_H2, KP30_AT1),
    # (KP20_AT2_H2, KP30_AT2_Cluster12), (KP20_Cluster8, KP30_Cluster8), (KP20_Cluster8, KP30_AT2_Cluster12),
    # (KP20_Clusters_9, KP30_Clusters_9_10)
    comparator.compare_programs(res_kp20, index_e, res_kp30, index_f, comparison_dir,
                                genes_symbols=adata.var['geneSymbol'], gp=gp, gprofiler_kwargs={'no_iea': True})

# long range AT2 and HPCS comparison t0 vs k30
comparator.compare_programs(res_t0, 0, res_k30, 1, comparison_dir,
                            genes_symbols=adata.var['geneSymbol'], gp=gp, gprofiler_kwargs={'no_iea': True})
comparator.compare_programs(res_t0, 3, res_k30, 0, comparison_dir,
                            genes_symbols=adata.var['geneSymbol'], gp=gp, gprofiler_kwargs={'no_iea': True})
# long range AT2 and HPCS comparison k12 vs kp12
comparator.compare_programs(res_k12, 0, res_kp12, 3, comparison_dir,
                            genes_symbols=adata.var['geneSymbol'], gp=gp, gprofiler_kwargs={'no_iea': True})
comparator.compare_programs(res_k12, 1, res_kp12, 1, comparison_dir,
                            genes_symbols=adata.var['geneSymbol'], gp=gp, gprofiler_kwargs={'no_iea': True})

# long range AT2 and HPCS comparison k30 vs kp20
comparator.compare_programs(res_k30, 0, res_kp20, 0, comparison_dir,
                            genes_symbols=adata.var['geneSymbol'], gp=gp, gprofiler_kwargs={'no_iea': True})
comparator.compare_programs(res_k30, 1, res_kp20, 2, comparison_dir,
                            genes_symbols=adata.var['geneSymbol'], gp=gp, gprofiler_kwargs={'no_iea': True})

# long range AT2 and HPCS comparison kp12 vs kp30
comparator.compare_programs(res_kp12, 1, res_kp30, 4, comparison_dir,
                            genes_symbols=adata.var['geneSymbol'], gp=gp, gprofiler_kwargs={'no_iea': True})
comparator.compare_programs(res_kp12, 3, res_kp30, 1, comparison_dir,
                            genes_symbols=adata.var['geneSymbol'], gp=gp, gprofiler_kwargs={'no_iea': True})


#%% projection of utilization from consecutive stages - AT2

reload(plotting)

colors = plotting.PROJ_COLORS_LIST

sets = [(split_adatas['00_All_early'], res_t0, 0, 'T0 AT2', colors[0]),
        (split_adatas['04_K_12w_ND'], res_k12, 1, 'K12 AT2', colors[1]),
        (split_adatas['05_K_30w_ND'], res_k30, 1, 'K30 AT2', colors[2]),
        (split_adatas['06_KP_12w_ND'], res_kp12, 1, 'KP12 AT2', colors[3]),
        (split_adatas['07_KP_20w_ND'], res_kp20, 2, 'KP20 AT2-H2', colors[4]),
        (split_adatas['08_KP_30w_ND'], res_kp30, 4, 'KP30 AT2-Cluster12', colors[5])]

title = 'Utilization levels of AT2 programs in consecutive stages'

filename = marjanovic_results_dir.joinpath('utilization_AT2_programs.png')

plotting.plot_joint_utilization_projection(sets, title, filename, show=False,
                                           obsm_coordinates='X_phate')

#%% projection of utilization from consecutive stages - HPCS

sets = [(split_adatas['00_All_early'], res_t0, 3, 'T0 HPCS', colors[0]),
        (split_adatas['04_K_12w_ND'], res_k12, 0, 'K12 HPCS', colors[1]),
        (split_adatas['05_K_30w_ND'], res_k30, 0, 'K30 HPCS', colors[2]),
        (split_adatas['06_KP_12w_ND'], res_kp12, 3, 'KP12 HPCS', colors[3]),
        (split_adatas['07_KP_20w_ND'], res_kp20, 0, 'KP20 HPCS', colors[4]),
        (split_adatas['08_KP_30w_ND'], res_kp30, 1, 'KP30 HPCS', colors[5])]

title = 'Utilization levels of HPCS programs in consecutive stages'

filename = marjanovic_results_dir.joinpath('utilization_HPCS_programs.png')

plotting.plot_joint_utilization_projection(sets, title, filename, show=False,
                                           obsm_coordinates='X_phate')


#%% projection of utilization from consecutive stages - cluster 8

sets = [(split_adatas['05_K_30w_ND'], res_k30, 3, 'K30 Metabolism 2', colors[2]),
        (split_adatas['06_KP_12w_ND'], res_kp12, 2, 'KP12 Cluster 8', colors[3]),
        (split_adatas['07_KP_20w_ND'], res_kp20, 3, 'KP20 Cluster 8', colors[4]),
        (split_adatas['08_KP_30w_ND'], res_kp30, 2, 'KP30 Cluster 8', colors[5])]

title = 'Utilization levels of Cluster8 programs in consecutive stages'

filename = marjanovic_results_dir.joinpath('utilization_Cluster8_programs.png')

plotting.plot_joint_utilization_projection(sets, title, filename, show=False,
                                           obsm_coordinates='X_phate')

#%% adding mouse origin coloring

field_1 = 'clusterK12'
field_2 = 'mouseID'

for cat in categories:
    tmp = sc.read_h5ad(split_adatas_dir.joinpath(f'{cat}.h5ad'))

    tmp.uns[f'{field_1}_colors_dict'] = dict(zip(adata.obs[field_1].cat.categories, adata.uns['clusterK12_colors']))
    ### we will use the same colors for the different mice/tumors, but in reverse order
    tmp.uns[f'{field_2}_colors_dict'] = dict(zip(tmp.obs[field_2].cat.categories, adata.uns['clusterK12_colors'][::-1]))

    tmp.obsm['row_colors'] = pd.concat([
        tmp.obs[field_1].map(tmp.uns[f'{field_1}_colors_dict']),
        tmp.obs[field_2].map(tmp.uns[f'{field_2}_colors_dict']),
    ], axis=1)

    split_adatas[cat] = tmp

#%% plotting utilizations clustermaps with new colors (cluster + mouseID):

for cat_a, cat_b in pairs:
    print(f'comparing {cat_a} and {cat_b}')

    adata_a = split_adatas[cat_a]
    adata_b = split_adatas[cat_b]

    comparator_dir = _utils.set_dir(marjanovic_results_dir.joinpath(
        f"comparator_{adata_a.uns['sname']}_{adata_b.uns['sname']}"))

    cmp = comparator.Comparator.load_from_file(comparator_dir.joinpath('comparator.npz'), adata_a, adata_b)

    res = cmp.a_result
    dec_folder = _utils.set_dir(cmp.results_dir.joinpath('decompositions'))

    # clustermap of normalized usages
    title = f"{cmp.a_sname} normalized usages of " \
            f"original GEPs, k={res.rank}"

    row_colors = _utils.expand_adata_row_colors(cmp.adata_a, pd.Series(
        _utils.floats_to_colors(res.loss_per_cell, cmap='RdYlGn_r', vmax=(
                2 * np.median(res.loss_per_cell))),
        name='residual', index=cmp.adata_a.obs.index))

    un_sns = _utils.plot_usages_norm_clustermaps(
        cmp.adata_a, normalized_usages=res.norm_usages,
        prog_names=res.prog_labels_2l, title=title,
        sns_clustermap_params={'row_colors': row_colors})

    un_sns.savefig(dec_folder.joinpath(f'{res.name}_normalized_usages_clustermap_w_mouseID.png'))
    plt.close()


#%% Trying to understand a split in cluster 4

cat_a = '05_K_30w_ND'
cat_b = '06_KP_12w_ND'

adata_a = split_adatas[cat_a]
adata_b = split_adatas[cat_b]

comparator_dir = _utils.set_dir(marjanovic_results_dir.joinpath(
    f"comparator_{adata_a.uns['sname']}_{adata_b.uns['sname']}"))

cmp = comparator.Comparator.load_from_file(comparator_dir.joinpath('comparator.npz'), adata_a, adata_b)

res = decompositions[cat_a][3]
dec_folder = _utils.set_dir(cmp.results_dir.joinpath('decompositions'))

# clustermap of normalized usages
title = f"{cmp.a_sname} normalized usages of " \
        f"original GEPs, k={res.rank}"

row_colors = _utils.expand_adata_row_colors(cmp.adata_a, pd.Series(
    _utils.floats_to_colors(res.loss_per_cell, cmap='RdYlGn_r', vmax=(
            2 * np.median(res.loss_per_cell))),
    name='residual', index=cmp.adata_a.obs.index))

un_sns = _utils.plot_usages_norm_clustermaps(
    cmp.adata_a, normalized_usages=res.norm_usages,
    prog_names=res.prog_labels_2l, title=title,
    sns_clustermap_params={'row_colors': row_colors})

un_sns.savefig(dec_folder.joinpath(f'{res.name}_normalized_usages_clustermap_w_mouseID_k3.png'))
plt.close()

#%% generating cell cycle statistics

adata = sc.read_h5ad(marjanovic_results_dir.joinpath('full.h5ad'))

adata.raw = adata
sc.pp.normalize_total(adata, target_sum=1e6, exclude_highly_expressed=True)
sc.pp.log1p(adata)

#%%

cc_gene_symbles = ['Top2a', 'Cdkn3', 'Mki67', 'Rrm2', 'Lig1', 'Ube2c']
cc_genes_ids = adata.var_names[adata.var.geneSymbol.isin(cc_gene_symbles)]

sc.pl.dotplot(adata, cc_genes_ids, groupby='timesimple', show=False)
ax = plt.gcf().get_axes()[0]

# correct x axis ticks to the gene names
ax.set_xticklabels(cc_gene_symbles, rotation=45, horizontalalignment='right')

# correct y axis ticks to the shorthand time names (t0, k12,...
ax.set_yticklabels(['T0', 'K12', 'K30', 'KP12', 'KP20', 'KP30'])

# figure title on the top center of the figure and not the last axis:
plt.suptitle('Cell cycle genes expression', y=0.75)

plt.tight_layout()
plt.savefig(marjanovic_results_dir.joinpath('cell_cycle_genes_expression.png'),
            bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# df = subset.obs.loc[:, ['S.Score', 'G2M.Score', 'Phase']]
