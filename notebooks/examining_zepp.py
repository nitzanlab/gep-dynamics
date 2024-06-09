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
# import matplotlib.pyplot as plt
import scanpy as sc

warnings.filterwarnings("ignore", category=FutureWarning)

sc.settings.n_jobs=-1

from gepdynamics import _utils, _constants, cnmf, pfnmf, comparator, plotting

_utils.cd_proj_home()
print(os.getcwd())

# %%

results_dir = _utils.set_dir('results')
zepp_results_dir = _utils.set_dir(results_dir.joinpath('zepp'))

decompositions = np.load(zepp_results_dir.joinpath('decompositions.npz'), allow_pickle=True)['obj'].item()

stages = ['E12', 'E15', 'E17', 'P3', 'P7']
pairs = [(stages[i], stages[i + 1]) for i in range(len(stages) -1)]

# %%

stage_a, stage_b = stages[:2]

comparison_dir = zepp_results_dir.joinpath(f"comparator_{stage_a}_{stage_b}")

adata_a = sc.read_h5ad(zepp_results_dir.joinpath("split_development_stage", f"{stage_a}.h5ad"))
adata_b = sc.read_h5ad(zepp_results_dir.joinpath("split_development_stage", f"{stage_b}.h5ad"))

# adding row colors
for tmp in [adata_a, adata_b]:
    tmp.obsm['row_colors'] = pd.concat([
        tmp.obs['celltype'].map(tmp.uns['celltype_colors_dict']),
        ], axis=1)

cmp = comparator.Comparator.load_from_file(comparison_dir.joinpath('comparator.npz'), adata_a, adata_b)
print(cmp)
cmp.print_errors()

# cmp.examine_adata_a_decomposition_on_jointly_hvgs(35, 3500)
# cmp.examine_adata_b_decompositions(3500, 35, 3500)
# cmp.plot_usages_violin('celltype', show=False)


# %%

for stage_a, stage_b in pairs:
    comparison_dir = zepp_results_dir.joinpath(f"comparator_{stage_a}_{stage_b}")

    adata_a = sc.read_h5ad(zepp_results_dir.joinpath("split_development_stage", f"{stage_a}.h5ad"))
    adata_b = sc.read_h5ad(zepp_results_dir.joinpath("split_development_stage", f"{stage_b}.h5ad"))

    # adding row colors
    for tmp in [adata_a, adata_b]:
        tmp.obsm['row_colors'] = pd.concat([
            tmp.obs['celltype'].map(tmp.uns['celltype_colors_dict']),
            ], axis=1)

    cmp = comparator.Comparator.load_from_file(comparison_dir.joinpath('comparator.npz'), adata_a, adata_b)
    print(cmp)

    # Where the work is done
    cmp.print_errors()


    # Close the loop
    del adata_a, adata_b


#%%

reload(plotting)

plotting.pio.renderers.default = 'browser'
# plotting.pio.renderers.default = 'svg'

res_a = decompositions['E12'][5]
res_b = decompositions['E15'][6]
res_c = decompositions['E17'][6]

res_a = decompositions['E15'][5]
res_b = decompositions['E17'][6]
res_c = decompositions['P3'][5]

plotting.plot_sankey_for_nmf_results(
    [decompositions['E12'][5],  decompositions['E15'][5], decompositions['E17'][6], decompositions['P3'][5]],
    gene_list_cutoff=101,
    cutoff=251, # cutoff for coefficient ranks in comparison
    display_threshold_counts=55)

#%% comparing programs pairs
from copy import copy
reload(comparator)

comparison_dir = zepp_results_dir.joinpath('programs_comparisons')

# create an instance of my gprofiler object for mus-musculus:
gp = _utils.MyGProfiler(organism='mmusculus', sources=['GO:BP', 'WP', 'REAC', 'KEGG'])

#%% E12 vs E15

res_a = copy(decompositions['E12'][5])
res_a.prog_names = ['E12_Mphase', 'E12_Club', 'E12_Alveolar', 'E12_AT2', 'E12_Sphase']
res_a.gene_coefs.columns = res_a.prog_names

res_b = copy(decompositions['E15'][5])
res_b.prog_names = ['E15_Club', 'E15_AT1', 'E15_AT2', 'E15_Mphase', 'E15_Sphase']
res_b.gene_coefs.columns = res_b.prog_names

for index_a, index_b in [(0,3), (4,4), (1,0), (1,1), (2,1), (2,2), (3,2)]:
    comparator.compare_programs(res_a, index_a, res_b, index_b, comparison_dir, gp=gp)

#%% E15 vs E17

res_a = copy(decompositions['E15'][5])
res_a.prog_names = ['E15_Club', 'E15_AT1', 'E15_AT2', 'E15_Mphase', 'E15_Sphase']
res_a.gene_coefs.columns = res_a.prog_names

res_b = copy(decompositions['E17'][6])
res_b.prog_names = ['E17_AT2', 'E17_AT1', 'E17_CellCycle', 'E17_Ciliated', 'E17_Club', 'E17_Progenitor']
res_b.gene_coefs.columns = res_b.prog_names

for index_a, index_b in [(3,2), (4,2), (0,3), (0,4), (1,1), (2,0), (2,5)]:
    comparator.compare_programs(res_a, index_a, res_b, index_b, comparison_dir, gp=gp)

#%% E17 vs P3

res_a = copy(decompositions['E17'][6])
res_a.prog_names = ['E17_AT2', 'E17_AT1', 'E17_CellCycle', 'E17_Ciliated', 'E17_Club', 'E17_Progenitor']
res_a.gene_coefs.columns = res_a.prog_names

res_b = copy(decompositions['P3'][5])
res_b.prog_names = ['P3_Ciliated', 'P3_Club', 'P3_AT1', 'P3_Progenitor', 'P3_AT2']
res_b.gene_coefs.columns = res_b.prog_names

for index_a, index_b in [(3,0), (4,1), (1,2), (0,4), (5,3)]:
    comparator.compare_programs(res_a, index_a, res_b, index_b, comparison_dir, gp=gp)

#%%

subset = sc.read_h5ad(zepp_results_dir.joinpath("epi_subset.h5ad"))
sc.pp.normalize_total(subset, target_sum=5e3, exclude_highly_expressed=True)
sc.pp.log1p(subset)
print(subset)

#catch user warnings
import warnings

#%%


sc.pl.violin(subset, ['Top2a', 'Cdkn3', 'Mki67', 'Rrm2', 'Lig1'])
sc.pl.stacked_violin(subset, ['Top2a', 'Cdkn3', 'Mki67', 'Rrm2'], groupby='development_stage')
sc.pl.dotplot(subset, ['Top2a', 'Cdkn3', 'Mki67', 'Rrm2'], groupby='development_stage')

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
