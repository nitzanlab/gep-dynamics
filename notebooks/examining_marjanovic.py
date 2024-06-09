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
    tmp.uns['clusterK12_colors_dict'] = dict(zip(tmp.obs['clusterK12'].cat.categories, tmp.uns['clusterK12_colors']))

    field_1 = 'clusterK12'

    tmp.obsm['row_colors'] = pd.concat([
        tmp.obs[field_1].map(tmp.uns[f'{field_1}_colors_dict']),
        ], axis=1)

    split_adatas[cat] = tmp

# %% Adding marker genes to the different decompositions:

color_obs_by = 'clusterK12'
decomposition_images = _utils.set_dir(results_dir.joinpath('decomposition_images'))

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

k_min, k_max = 2, 7

with warnings.catch_warnings():  # supress plotting warnings
    warnings.simplefilter(action='ignore', category=UserWarning)

    for cat in categories:
        tmp = split_adatas[cat]
        for k in range(k_min, k_max + 1):
            res = decompositions[cat][k]

            # # usages clustermap
            # un_sns = _utils.plot_usages_norm_clustermaps(
            #     tmp, normalized_usages=res.norm_usages, prog_names=res.prog_names,
            #     title=f'{cat}', show=False, sns_clustermap_params={
            #         'row_colors': tmp.obs[color_obs_by].map(tmp.uns[f'{color_obs_by}_colors_dict'])})
            # un_sns.savefig(decomposition_images.joinpath(f"{cat}_{k}_usages_norm.png"),
            #                dpi=180, bbox_inches='tight')
            # plt.close(un_sns.fig)
            #
            # # usages violin plot
            # _utils.plot_usages_norm_violin(
            #     tmp, color_obs_by, normalized_usages=res.norm_usages, prog_names=res.prog_names,
            #     save_path=decomposition_images.joinpath(
            #         f'{cat}_{k}_norm_usage_per_lineage.png'))

            # Marker genes heatmap
            heatmap_data = res.gene_coefs.loc[marker_genes_ID].T
            hm = sns.heatmap(heatmap_data, cmap='coolwarm', vmin=-2, vmax=2)
            plt.xticks(0.5 + np.arange(len(marker_genes_symbols)), marker_genes_symbols)

            plt.title(f'Marker genes coefficients for {res.name}')
            plt.tight_layout()

            hm.figure.savefig(decomposition_images.joinpath(f'{cat}_{k}_marker_genes.png'))
            plt.close()


# %% Sanky plot

reload(plotting)

plotting.pio.renderers.default = 'browser'
# plotting.pio.renderers.default = 'svg'

plotting.plot_sankey_for_nmf_results([
    decompositions['04_K_12w_ND'][4],
    decompositions['05_K_30w_ND'][4],
    decompositions['06_KP_12w_ND'][5],
    decompositions['07_KP_20w_ND'][5],
    decompositions['08_KP_30w_ND'][5]],
    gene_list_cutoff=101,
    cutoff=251, # cutoff for coefficient ranks in comparison
    display_threshold_counts=40)

#%%

pairs = [categories[[2,3]], categories[[2,4]], categories[[3,4]], categories[[3,5]], categories[[3,6]], categories[[4,5]], categories[[5,6]]]
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

    cmp = comparator.Comparator.load_from_file(comparison_dir.joinpath('comparator.npz'), adata_a, adata_b)
    cmp.plot_marker_genes_heatmaps(marker_genes_ID, marker_genes_symbols)

#%%
gp = _utils.MyGProfiler(organism='mmusculus', sources=['GO:BP', 'WP', 'REAC', 'KEGG'])

res_a = decompositions['04_K_12w_ND'][4]
res_b = decompositions['05_K_30w_ND'][4]
index_a = 2
index_b = 2

comparator.compare_programs(res_a, index_a, res_b, index_b,
                            results_dir.joinpath('programs_comparisons'),
                            genes_symbols=adata.var['geneSymbol'],
                            gp=gp)


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

#%% Club

marker_genes = ['Krt8', 'Hopx', 'Klf6', 'Aqp5', 'Sftpa1', 'Sftpb', 'Sftpc',
                'Mki67', 'Top2a', 'Rrm1', 'Rrm2',
                'Sox2', 'Scgb3a2', 'Foxj1', 'Dynlrb2', 'Hoxa5', 'Col5a2', ]

programs_list = [
    adata_a.varm['usage_coefs']['E12.p1'],
    adata_b.varm['usage_coefs']['E15.p3'],
    adata_c.varm['usage_coefs']['E17.p5']]

plot_marker_genes_heatmaps(programs_list, marker_genes, show=False,
                           title='Marker gene coefficients for Club cell programs',
                           save_file = zepp_results_dir.joinpath('marker_genes_heatmaps Club.png'))

#%% Ciliated

programs_list = [
    adata_a.varm['usage_coefs'],
    adata_b.varm['usage_coefs']['E15.p5'],
    adata_c.varm['usage_coefs']['E17.p4']]

plot_marker_genes_heatmaps(programs_list, marker_genes, show=True,
                           title='Marker gene coefficients for Ciliated programs',
                           save_file = zepp_results_dir.joinpath('marker_genes_heatmaps Ciliated.png')
                           )

#%% Alveolar

# monotonically increasing time
programs_list = [
    adata_a.varm['usage_coefs']['E12.p5'],
    adata_a.varm['usage_coefs']['E12.p3'],
    adata_b.varm['usage_coefs']['E15.p1'],
    adata_b.varm['usage_coefs']['E15.p0'],
    adata_c.varm['usage_coefs']['E17.p1'],
    adata_c.varm['usage_coefs']['E17.p0'],
    adata_c.varm['usage_coefs']['E17.p3'],
]
# # None monotonic
# programs_list = [
#     adata_a.varm['usage_coefs']['E12.p5'],
#     adata_b.varm['usage_coefs']['E15.p1'],
#     adata_c.varm['usage_coefs']['E17.p1'],
#     adata_c.varm['usage_coefs']['E17.p3'],
#     adata_c.varm['usage_coefs']['E17.p0'],
#     adata_b.varm['usage_coefs']['E15.p0'],
#     adata_a.varm['usage_coefs']['E12.p3'],
# ]

plot_marker_genes_heatmaps(programs_list, marker_genes, show=True,
                           title='Marker gene coefficients for Alveolar programs',
                           save_file = zepp_results_dir.joinpath('marker_genes_heatmaps Alveolar 2.png')
                           )

#%% Cell Cycle

programs_list = [
    adata_a.varm['usage_coefs']['E12.p2'],
    adata_b.varm['usage_coefs']['E15.p2'],
    adata_c.varm['usage_coefs']['E17.p2'],
    adata_b.varm['usage_coefs']['E15.p4'],
    adata_a.varm['usage_coefs']['E12.p4'],
]

plot_marker_genes_heatmaps(programs_list, marker_genes, show=True,
                           title='Marker gene coefficients for cell cycle programs',
                           save_file = zepp_results_dir.joinpath('marker_genes_heatmaps cell cycle.png')
                           )
