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
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

from sklearn.linear_model import LinearRegression

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
adata = sc.read_h5ad(zepp_results_dir.joinpath('epi_subset.h5ad'))
sc.pl.umap(adata, color=['development_stage', 'celltype'], show=False)
plt.savefig(zepp_results_dir.joinpath('umap_celltype_stage.png'))


# %%

adata_a = sc.read_h5ad(zepp_results_dir.joinpath("split_development_stage", f"{stages[0]}.h5ad"))
adata_b = sc.read_h5ad(zepp_results_dir.joinpath("split_development_stage", f"{stages[1]}.h5ad"))
adata_c = sc.read_h5ad(zepp_results_dir.joinpath("split_development_stage", f"{stages[2]}.h5ad"))
adata_d = sc.read_h5ad(zepp_results_dir.joinpath("split_development_stage", f"{stages[3]}.h5ad"))

# adding row colors
for tmp in [adata_a, adata_b, adata_c, adata_d]:
    tmp.obsm['row_colors'] = pd.concat([
        tmp.obs['celltype'].map(tmp.uns['celltype_colors_dict']),
        ], axis=1)

# %%
stage_a, stage_b = stages[:2]

comparison_dir = zepp_results_dir.joinpath(f"comparator_{stage_a}_{stage_b}")

cmp = comparator.Comparator.load_from_file(comparison_dir.joinpath('comparator.npz'), adata_a, adata_b)
print(cmp)
cmp.print_errors()

# cmp.examine_adata_a_decomposition_on_jointly_hvgs(35, 3500)
# cmp.examine_adata_b_decompositions(3500, 35, 3500)
# cmp.plot_usages_violin('celltype', show=False)


#%% results renaming preparation

cp_rn = comparator.NMFResultBase.copy_and_rename_programs

res_a = cp_rn(decompositions['E12'][5],
      ['E12_Mphase', 'E12_Proximal', 'E12_Alveolar', 'E12_AT2', 'E12_Sphase'])
res_b = cp_rn(decompositions['E15'][5],
          ['E15_Club_Ciliated', 'E15_AT1', 'E15_AT2', 'E15_Mphase', 'E15_Sphase'])
res_c = cp_rn(decompositions['E17'][6],
          ['E17_AT2', 'E17_AT1', 'E17_CellCycle', 'E17_Ciliated', 'E17_Club', 'E17_Progenitor'])
res_d = cp_rn(decompositions['P3'][5],
          ['P3_Ciliated', 'P3_Club', 'P3_AT1', 'P3_Progenitor', 'P3_AT2'])


#%% Sankey plot

reload(plotting)

plotting.pio.renderers.default = 'browser'
# plotting.pio.renderers.default = 'svg'

plotting.plot_sankey_for_nmf_results(
    [res_a, res_b, res_c, res_d],
    gene_list_cutoff=101,
    cutoff=251, # cutoff for coefficient ranks in comparison
    display_threshold_counts=55,
    show_unassigned_genes=False,
)

#%% comparing programs pairs

comparison_dir = zepp_results_dir.joinpath('programs_comparisons')

# create an instance of my gprofiler object for mus-musculus:
gp = _utils.MyGProfiler(organism='mmusculus', sources=['GO:BP', 'WP', 'REAC', 'KEGG'])

#%% E12 vs E15

for index_a, index_b in [(0,3), (4,4), (1,0), (1,1), (2,1), (2,2), (3,2)]:
    comparator.compare_programs(res_a, index_a, res_b, index_b, comparison_dir, gp=gp)

#%% E15 vs E17


for index_b, index_c in [(3,2), (4,2), (0,3), (0,4), (1,1), (2,0), (2,5)]:
    comparator.compare_programs(res_b, index_b, res_c, index_c, comparison_dir, gp=gp)

#%% E17 vs P3

for index_c, index_d in [(3,0), (4,1), (1,2), (0,4), (5,3)]:
    comparator.compare_programs(res_c, index_c, res_d, index_d, comparison_dir, gp=gp)


#%% marker genes heatmap
marker_genes = ["Sox2", "Tspan1", "Cyp2f2", "Scgb3a2", "Rsph1", "Foxj1",
                "Sox9", "Hopx", "Timp3", 'Aqp5', 'Sftpa1', 'Sftpb',
                "Mki67", "Cdkn3", "Rrm2", "Lig1"]

stage = 'E12'; k = 5; res = decompositions[stage][k]
col_order = res.gene_coefs.columns[[1,4,0,2,3]]

stage = 'E15'; k = 5; res = decompositions[stage][k]
col_order = res.gene_coefs.columns[[0,3,1,2,4]]

stage = 'E17'; k = 6; res = decompositions[stage][k]
col_order = res.gene_coefs.columns[[3,4,0,1,2,5]]

stage = 'P3'; k = 5; res = decompositions[stage][k]
col_order = res.gene_coefs.columns[[0,1,4,2,3]]


# Marker genes heatmap
hm = sns.heatmap(res.gene_coefs.loc[marker_genes, col_order], cmap='coolwarm', vmin=-2, vmax=2)
plt.tight_layout()
hm.figure.savefig(zepp_results_dir.joinpath(f'marker_genes_{stage}_{k}.png'))
plt.close()

#%% marker genes dynamics for proximal and cell cycle programs
reload(plotting)

marker_genes = ["Tspan1", "Ccnd2", "Igfbp2", "Igfbp5", "Anxa2",
                "Cyp2f2", "Scgb1a1", "Scgb3a2", "Cldn10", "Hp", "Krt15", "Nupr1",
                "Rsph1", "Foxj1", "Dynlrb2", "Fam183b", "Tm4sf1", "Tppp3",
                "Top2a", "Ube2c", "Ccnb1", "Cdc20", "Cdk1", "Cenpf",
                "Tyms", "Dek", "Gmnn", "Hells", "Rrm1",
                "Mki67", "Rrm2", "Lig1"]

marker_genes = [
"Anxa2",
"Igfbp5",
"Tspan1",
"Ccnd2",
"Foxj1",
"Rsph1",
"Fam183b",
"Dynlrb2",
"Tppp3",
"Tm4sf1",
"Krt15",
"Scgb3a2",
"Cldn10",
"Nupr1",
"Cyp2f2",
"Scgb1a1",
"Hp",
"Rrm1",
"Tyms",
"Dek",
"Gmnn",
"Lig1",
"Rrm2",
"Hells",
"Igfbp2",
"Cdk1",
"Top2a",
"Mki67",
"Ccnb1",
"Ube2c",
"Cenpf",
"Cdc20",
]

empty_column = pd.Series(np.zeros(res_a.gene_coefs.shape[0]), index=res_a.gene_coefs.index, name='empty')

programs_list = [
    res_a.gene_coefs['E12_Proximal'],
    res_b.gene_coefs['E15_Club_Ciliated'],
    res_c.gene_coefs['E17_Club'],
    res_d.gene_coefs['P3_Club'],
    res_c.gene_coefs['E17_Ciliated'],
    res_d.gene_coefs['P3_Ciliated'],
    empty_column,
    res_a.gene_coefs['E12_Mphase'],
    res_b.gene_coefs['E15_Mphase'],
    res_a.gene_coefs['E12_Sphase'],
    res_b.gene_coefs['E15_Sphase'],
    res_c.gene_coefs['E17_CellCycle'],
    ]


#%% log average TP5k expression for marker genes

for res, adata in zip([res_a, res_b, res_c, res_d], [adata_a, adata_b, adata_c, adata_d]):
    X = adata.X.toarray()
    X = X * 5000. / X.sum(axis=1, keepdims=True)
    # gene_importance = pd.DataFrame(np.log1p(X.T @ res.norm_usages),
    #                                index=res.gene_coefs.index, columns=res.gene_coefs.columns)
    nnls_reg = LinearRegression(positive=True, fit_intercept=False)

    data = nnls_reg.fit(res.norm_usages, X).coef_
    gene_importance = pd.DataFrame(np.log1p(data),
                                   index=res.gene_coefs.index, columns=res.gene_coefs.columns)
    res.gene_importance = gene_importance

    del nnls_reg

#%%

plotting.plot_marker_genes_heatmaps(programs_list, marker_genes, show=False,
                           title='Marker gene coefficients for cell programs',
                           save_file = zepp_results_dir.joinpath('marker_genes_dynamics_A2.png'))


#%%

# Marker genes heatmap
df = pd.concat(programs_list, axis=1)

hm = sns.heatmap(df.loc[marker_genes], cmap='BuGn', vmax=3)
plt.title('Marker genes log-TP5K dynamics')
plt.tight_layout()
hm.figure.savefig(zepp_results_dir.joinpath(f'marker_genes_tpm_dynamics_A2.png'))
plt.close()

#%%

cm = sns.clustermap(df.loc[marker_genes], cmap='BuGn', vmax=3, col_cluster=False,
                    yticklabels=True, cbar=False, metric='correlation')
cm.cax.set_visible(False)
cm.fig.suptitle('Marker genes log-TP5K dynamics', y=0.85, fontsize=18)

plt.tight_layout()
cm.figure.savefig(zepp_results_dir.joinpath(f'marker_genes_tpm_dynamics_cluster_A2.png'))
plt.close()


#%% projection of utilization from consecutive stages - Club

reload(plotting)
colors = plotting.PROJ_COLORS_LIST

sets = [(adata_a, res_a, 1, 'E12 Club', colors[0]),
        (adata_b, res_b, 0, 'E15 Club', colors[1]),
        (adata_c, res_c, 4, 'E17 Club', colors[2]),
        (adata_d, res_d, 1, 'P3 Club', colors[3]),]

title = 'Utilization levels of club program in consecutive stages'
filename = zepp_results_dir.joinpath('utilization_club_programs.png')

plotting.plot_joint_utilization_projection(sets, title, filename, show=False)

#%% projection of utilization from consecutive stages - ciliated

sets = [(adata_a, res_a, 1, 'E12 Club', colors[0]),
        (adata_b, res_b, 0, 'E15 Club', colors[1]),
        (adata_c, res_c, 3, 'E17 Ciliated', colors[2]),
        (adata_d, res_d, 0, 'P3 Ciliated', colors[3]),]

title = 'Utilization levels of ciliated program in consecutive stages'

filename = zepp_results_dir.joinpath('utilization_ciliated_programs.png')

plotting.plot_joint_utilization_projection(sets, title, filename, show=False)

#%% projection of utilization from consecutive stages - alveolar type 1


sets = [(adata_a, res_a, 2, 'E12 Alveolar', colors[0]),
        (adata_b, res_b, 1, 'E15 AT1', colors[1]),
        (adata_c, res_c, 1, 'E17 AT1', colors[2]),
        (adata_d, res_d, 2, 'P3 AT1', colors[3]),]

title = 'Utilization of AT1 cell program in consecutive stages'
filename = zepp_results_dir.joinpath('utilization_at1_programs.png')

plotting.plot_joint_utilization_projection(sets, title, filename, show=False)

#%% projection of utilization from consecutive stages - cell cycle m

reload(plotting)

sets = [(adata_a, res_a, 0, 'E12 Mphase', colors[0]),
        (adata_b, res_b, 3, 'E15 Mphase', colors[1]),
        (adata_c, res_c, 2, 'E17 CellCycle', colors[2]),]

title = 'Utilization of cell cycle programs in consecutive stages'
filename = zepp_results_dir.joinpath('utilization_mphase_programs.png')

plotting.plot_joint_utilization_projection(sets, title, filename, show=False)


#%% cell cycle visualization

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