#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import rankdata

import plotly.graph_objects as go
import plotly.io as pio
# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'


def get_rank_from_coefs(orig_coefs, gene_indices, cutoff):
    coefs = orig_coefs.iloc[gene_indices].copy()
    tmp_none = 1 - (coefs==0).all(axis=1).astype(int) # if all columns are zero set to 0 else 1
    coefs.loc[:,:] = rankdata(-coefs, axis=0)
    coefs[unassigned_column] = tmp_none * cutoff
    coefs[coefs > cutoff] = cutoff + 1
    return coefs

#%% load data
import scanpy as sc
from gepdynamics import _utils

results_dir = _utils.set_dir('results')
zepp_results_dir = _utils.set_dir(results_dir.joinpath('zepp'))

adata_a = sc.read_h5ad(zepp_results_dir.joinpath("split_development_stage", f"E12.h5ad"))
adata_b = sc.read_h5ad(zepp_results_dir.joinpath("split_development_stage", f"E15.h5ad"))
adata_c = sc.read_h5ad(zepp_results_dir.joinpath("split_development_stage", f"E17.h5ad"))

#%% plotly sankey

# Latest version: looking at genes that have high rank in one of the GEPs,
# and their coefficients are larger than one. Genes are passed between geps
# according to best rank


orig_coefs_a = adata_a.varm['usage_coefs'].copy()
orig_coefs_b = adata_b.varm['usage_coefs'].copy()
orig_coefs_c = adata_c.varm['usage_coefs'].copy()

gene_list_cutoff = 401
cutoff = 801 # cutoff for coefficient ranks in comparison
threshold_counts = 100
unassigned_column = 'unassigned'

# getting background genes
coefs = np.hstack([orig_coefs_a.values,
                   orig_coefs_b.values,
                   orig_coefs_c.values])
ranked_data = rankdata(-coefs, axis=0)
ranked_data[coefs<=0] = gene_list_cutoff
ranked_data[ranked_data >= gene_list_cutoff] = gene_list_cutoff
bg_genes = np.where(ranked_data.min(axis=1) < gene_list_cutoff)[0]

a_coefs = get_rank_from_coefs(orig_coefs_a, bg_genes, cutoff)
b_coefs = get_rank_from_coefs(orig_coefs_b, bg_genes, cutoff)
c_coefs = get_rank_from_coefs(orig_coefs_c, bg_genes, cutoff)

# find the column with the best value per row
a_min = a_coefs.idxmin(axis=1)
b_min = b_coefs.idxmin(axis=1)
c_min = c_coefs.idxmin(axis=1)

labels = [*a_coefs.columns, *b_coefs.columns, *c_coefs.columns]
source = [] # indices correspond to labels, eg A1, A2, A1, B1, ...
target = []
value = []
link_colors = []

# calculate a-b links
for i, col_a in enumerate(a_coefs.columns):
    for j, col_b in enumerate(b_coefs.columns):
        val = sum((a_min == col_a) & ( b_min == col_b))
        if val > threshold_counts:
            source.append(i)
            target.append(j + a_coefs.shape[1])
            value.append(val)
            link_colors.append('lightgrey')
            if col_a == unassigned_column:
                link_colors[-1] = 'lightblue'
            elif col_b == unassigned_column:
                link_colors[-1] = 'lightpink'

# calculate b-c links
for i, col_b in enumerate(b_coefs.columns):
    for j, col_c in enumerate(c_coefs.columns):
        val = sum((b_min == col_b) & ( c_min == col_c))
        if val > threshold_counts:
            source.append(i + a_coefs.shape[1])
            target.append(j + a_coefs.shape[1] + b_coefs.shape[1])
            value.append(val)
            link_colors.append('lightgrey')
            if col_b == unassigned_column:
                link_colors[-1] = 'lightpink'
            elif col_c == unassigned_column:
                link_colors[-1] = 'lightgreen'

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = labels,
      color = "blue"
    ),
    link = dict(
      source = source, target = target, value = value, color=link_colors))])

fig.update_layout(title_text=f"Sankey Diagram for top {cutoff-1} prominent gene coefficients. "
                             f"Background is {len(bg_genes)} highly ranked genes (top {gene_list_cutoff-1} per program)"
                             f", threshold={threshold_counts}", font_size=10)
fig.show()
