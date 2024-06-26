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

warnings.filterwarnings("ignore", category=FutureWarning)

sc.settings.n_jobs=-1

from gepdynamics import _utils, _constants, cnmf, pfnmf, comparator, plotting

_utils.cd_proj_home()
print(os.getcwd())

# %%

results_dir = _utils.set_dir('results')
results_dir = _utils.set_dir(results_dir.joinpath('marjanovic'))

decompositions = np.load(results_dir.joinpath('decompositions.npz'), allow_pickle=True)['obj'].item()

adata = sc.read_h5ad(results_dir.joinpath('full.h5ad'))

column_of_interest = 'timesimple'
categories = adata.obs[column_of_interest].cat.categories

# %%

split_adatas_dir = _utils.set_dir(results_dir.joinpath(f'split_{column_of_interest}'))

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


# %% Sanky plot

reload(plotting)

plotting.pio.renderers.default = 'browser'
# plotting.pio.renderers.default = 'svg'

plotting.plot_sankey_for_nmf_results([
    decompositions['00_All_early'][4],
    decompositions['04_K_12w_ND'][4],
    decompositions['05_K_30w_ND'][4],
    decompositions['06_KP_12w_ND'][5],
    decompositions['07_KP_20w_ND'][5],
    decompositions['08_KP_30w_ND'][6]],
    gene_list_cutoff=201,
    cutoff=501, # cutoff for coefficient ranks in comparison
    display_threshold_counts=80,
    show_unassigned_genes=True
)

#%%

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

    comparator_dir = _utils.set_dir(results_dir.joinpath(
        f"comparator_{adata_a.uns['sname']}_{adata_b.uns['sname']}"))

    cmp = comparator.Comparator.load_from_file(comparator_dir.joinpath('comparator.npz'), adata_a, adata_b)

    # cmp.plot_marker_genes_heatmaps(marker_genes_ID, marker_genes_symbols)
    cmp.plot_utilization_scatters('X_phate')


#%% comparing programs pairs

from copy import copy
reload(comparator)

comparison_dir = _utils.set_dir(results_dir.joinpath('programs_comparisons'))

# create an instance of my gprofiler object for mus-musculus:
gp = _utils.MyGProfiler(organism='mmusculus', sources=['GO:BP', 'WP', 'REAC', 'KEGG'])

# Program names

T0_prog_names = ['T0_AT2', 'T0_Lymphocyte', 'T0_Cell_Cycle', 'T0_Interferon_HPCS']
K12_prog_names = ["K12_HPCS", "K12_AT2", "K12_Cell_Cycle", "K12_AT1_ECM"]
K30_prog_names = ["K30_HPCS", "K30_AT2", "K30_Metabolism1", "K30_Metabolism2"]
KP12_prog_names = ["KP12_Cluster8", "KP12_AT2", "KP12_Cluster6", "KP12_HPCS", "KP12_Cluster7"]
KP20_prog_names = ["KP20_HPCS", "KP20_Cluster7", "KP20_AT2_H2", "KP20_Cluster8", "KP20_Cluster9"]
KP30_prog_names = ["KP30_Cluster11", "KP30_HPCS", "KP30_Cluster8",
                   "KP30_Cluster9_Cluster10", "KP30_AT2_Cluster12", "KP30_AT1"]

prog_names_dict = {
    '00_All_early': T0_prog_names,
    '04_K_12w_ND': K12_prog_names,
    '05_K_30w_ND': K30_prog_names,
    '06_KP_12w_ND': KP12_prog_names,
    '07_KP_20w_ND': KP20_prog_names,
    '08_KP_30w_ND': KP30_prog_names
}

pairs_comparisons = {
    ('00_All_early', '04_K_12w_ND'): [(0,1), (1,2), (2,2), (2,0), (3,0)],
    ('04_K_12w_ND', '05_K_30w_ND'): [(0,0), (2,2), (1,3), (1,1)],
    ('05_K_30w_ND', '06_KP_12w_ND'): [(0,0), (0,3), (1,1), (2,2), (2,0), (2,4), (3,2)],
    ('06_KP_12w_ND', '07_KP_20w_ND'): [(0,0), (0,2), (0,4), (1,2), (1,3), (2,3), (3,0), (4,1)],
    ('07_KP_20w_ND', '08_KP_30w_ND'): [(0,1), (1,0), (2,4), (2,5), (3,2), (3,4), (4,3)]
}

#%% comparing all interesting pairs of programs from adjacent time points

for a, b in pairs_comparisons.keys():
    res_a = copy(decompositions[a][len(prog_names_dict[a])])
    res_a.prog_names = prog_names_dict[a]
    res_a.gene_coefs.columns = res_a.prog_names

    res_b = copy(decompositions[b][len(prog_names_dict[b])])
    res_b.prog_names = prog_names_dict[b]
    res_b.gene_coefs.columns = res_b.prog_names

    for index_a, index_b in pairs_comparisons[(a, b)]:
        comparator.compare_programs(res_a, index_a, res_b, index_b, comparison_dir,
                                    genes_symbols=adata.var['geneSymbol'], gp=gp)

#%% T0 and K12


res_a = copy(decompositions['00_All_early'][4])
res_a.prog_names = ['T0_AT2', 'T0_Lymphocyte', 'T0_Cell_Cycle', 'T0_Interferon_HPCS']
res_a.gene_coefs.columns = res_a.prog_names

res_b = copy(decompositions['04_K_12w_ND'][4])
res_b.prog_names = ["K12_HPCS", "K12_AT2", "K12_Cell_Cycle", "K12_AT1_ECM"]
res_b.gene_coefs.columns = res_b.prog_names

for index_a, index_b in [(0,1), (1,2), (2,2), (2,0), (3,0)]:
    comparator.compare_programs(res_a, index_a, res_b, index_b, comparison_dir,
                                genes_symbols=adata.var['geneSymbol'], gp=gp)


#%%

adata.raw = adata
sc.pp.normalize_total(adata, target_sum=1e6, exclude_highly_expressed=True)
sc.pp.log1p(adata)

#%%
cc_gene_symbles = ['Top2a', 'Cdkn3', 'Mki67', 'Rrm2', 'Lig1']
cc_gene_id = adata.var_names[adata.var.geneSymbol.isin(cc_gene_symbles)]

sc.pl.violin(adata, cc_gene_id)
sc.pl.stacked_violin(adata, cc_gene_id, groupby='timesimple')
sc.pl.dotplot(adata, cc_gene_id, groupby='timesimple')

# df = subset.obs.loc[:, ['S.Score', 'G2M.Score', 'Phase']]

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

#%% plotting utilizations clustermaps with new colors (cluster + mouseID:

for cat_a, cat_b in pairs:
    print(f'comparing {cat_a} and {cat_b}')

    adata_a = split_adatas[cat_a]
    adata_b = split_adatas[cat_b]

    comparator_dir = _utils.set_dir(results_dir.joinpath(
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

#%%

tmp = split_adatas['08_KP_30w_ND']
ct = pd.crosstab(tmp.obs.clusterK12, tmp.obs.mouseID)

#%% Trying to understand a split in cluster 4
cat_a = '05_K_30w_ND'
cat_b = '06_KP_12w_ND'

adata_a = split_adatas[cat_a]
adata_b = split_adatas[cat_b]

comparator_dir = _utils.set_dir(results_dir.joinpath(
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




#%% comparing cell cycle programs

a = '00_All_early'
b = '06_KP_12w_ND'

res_a = copy(decompositions[a][len(prog_names_dict[a])])
res_a.prog_names = prog_names_dict[a]
res_a.gene_coefs.columns = res_a.prog_names

res_b = copy(decompositions[b][len(prog_names_dict[b])])
res_b.prog_names = prog_names_dict[b]
res_b.gene_coefs.columns = res_b.prog_names

for index_a, index_b in [(2, 4)]:
    comparator.compare_programs(res_a, index_a, res_b, index_b, comparison_dir,
                                genes_symbols=adata.var['geneSymbol'], gp=gp)




