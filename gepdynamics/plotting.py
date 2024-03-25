#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Dict, Iterable, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import plotly.io as pio

from scipy.stats import rankdata
from scipy.cluster import hierarchy

import gepdynamics._utils as _utils

pio.renderers.default = 'browser'
# pio.renderers.default = 'svg'

UNASSIGNED_GENES_COLUMN = 'unassigned'

def get_rank_from_coefs(orig_coefs, gene_indices, cutoff):
    """ create a dataframe of ranks up to cutoff from a dataframe of coefficients
    with unassigned column for genes that are equal between all programs"""
    coefs = orig_coefs.iloc[gene_indices].copy()
    tmp_none = 1 - (coefs==0).all(axis=1).astype(int) # if all prog_names are zero set to 0 else 1
    coefs.loc[:,:] = rankdata(-coefs, axis=0)
    coefs[UNASSIGNED_GENES_COLUMN] = tmp_none * cutoff
    coefs[coefs > cutoff] = cutoff + 1
    return coefs


def load_data_for_lung_dev_sanky(zepp_results_dir: _utils.PathLike):
    """
    Load data for lung development dataset

    >>> from gepdynamics.plotting import load_data_for_lung_dev_sanky
    >>> adata_a, adata_b, adata_c = load_data_for_lung_dev_sanky(zepp_results_dir)

    """
    import scanpy as sc
    from gepdynamics import _utils

    zepp_results_dir = _utils.set_dir(zepp_results_dir)

    adata_a = sc.read_h5ad(zepp_results_dir.joinpath("split_development_stage", f"E12.h5ad"))
    adata_b = sc.read_h5ad(zepp_results_dir.joinpath("split_development_stage", f"E15.h5ad"))
    adata_c = sc.read_h5ad(zepp_results_dir.joinpath("split_development_stage", f"E17.h5ad"))

    return adata_a, adata_b, adata_c


def plot_sankey_for_lung_dev(nmf_res_a, nmf_res_b, nmf_res_c,
                             gene_list_cutoff=401,
                             cutoff=801, # cutoff for coefficient ranks in comparison
                             threshold_counts=100):
    """
    Create Sankey plot for lung development dataset

    # Latest version: looking at genes that have high rank in one of the GEPs,
    # and their coefficients are larger than one. Genes are passed between geps
    # according to best rank
    """

    orig_coefs_a = nmf_res_a.gene_coefs.copy()
    orig_coefs_b = nmf_res_b.gene_coefs.copy()
    orig_coefs_c = nmf_res_c.gene_coefs.copy()

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
    source = [] # indices correspond to labels, e.g. A1, A2, A1, B1, ...
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
                if col_a == UNASSIGNED_GENES_COLUMN:
                    link_colors[-1] = 'lightblue'
                elif col_b == UNASSIGNED_GENES_COLUMN:
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
                if col_b == UNASSIGNED_GENES_COLUMN:
                    link_colors[-1] = 'lightpink'
                elif col_c == UNASSIGNED_GENES_COLUMN:
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

    fig.update_layout(title_text=f"Sankey Diagram for top {gene_list_cutoff-1} prominent gene coefficients. "
                                 f"Background is {len(bg_genes)} highly ranked genes (top {cutoff-1} per program)"
                                 f", threshold={threshold_counts}", font_size=10)
    fig.show()


def plot_marker_genes_heatmaps(programs_list: List[pd.Series],
                               marker_genes: List[str],
                               title: str = None,
                               show: bool = False,
                               save_file: _utils.PathLike = None):
    """
    Plot heatmaps of the marker genes coefficients for each decomposition

    title example: 'Marker gene coefficients for Club cell programs'
    file example: 'marker_genes_heatmap.png'
    """

    # create dataframe from the list of series
    df = pd.concat(programs_list, axis=1)

    sns.heatmap(df.loc[marker_genes].T, cmap='coolwarm', vmin=-2, vmax=2)

    if title is None:
        title = 'Marker genes coefficients'
    plt.title(title)

    plt.tight_layout()

    if save_file is not None:
        plt.savefig(save_file)
    if show:
        plt.show()
    plt.close()


def get_ordered_adjacency_matrix(correlation_matrix: np.ndarray,
                                 prog_names: Iterable,
                                 ranks: Iterable,
                                 threshold=0.2,
                                 verbose: bool = False) -> pd.DataFrame:
    """
    Given a correlation matrix to base the adjacency matrix on, returns the
    adjacency matrix after filtering out edges with correlation below the threshold
    and keeping only edges between consecutive layers.
    """
    # adjacency matrix creation
    adjacency = pd.DataFrame(np.round((correlation_matrix), 2),
                             index=prog_names, columns=prog_names)

    # order
    linkage = hierarchy.linkage(
        adjacency, method='average', metric='euclidean')
    prog_order = hierarchy.leaves_list(
        hierarchy.optimal_leaf_ordering(linkage, adjacency))

    # keeping only edges between consecutive layers
    for i in range(len(ranks) - 2):
        adjacency.values[:np.sum(ranks[:i + 1]), np.sum(ranks[:i + 2]):] = 0
        adjacency.values[np.sum(ranks[:i + 2]):, :np.sum(ranks[:i + 1])] = 0

    np.fill_diagonal(adjacency.values, 0)
    adjacency.values[adjacency.values <= threshold] = 0

    if verbose:
        print(f'Number of edges={np.count_nonzero(adjacency)}')

    # ordering the nodes for display
    adjacency = adjacency.iloc[prog_order, prog_order]

    return adjacency


def plot_layered_correlation_flow_chart(layer_keys,
                                        adjacency_df: pd.DataFrame,
                                        prog_names_dict,
                                        title: str,
                                        # parameter layout that can be 'straight' or 'fan'
                                        layout_type: Literal['straight', 'fan'] = 'straight',
                                        plt_figure_kwargs: Dict = None,
                                        fig_title_kwargs: Dict = None) -> plt.Figure:
    """
    Plotting the flow chart of the correlation matrix between layers.
    """
    # setting figure arguments
    figure_kwargs = {'figsize': (14.4, 16.2), 'dpi': 100}
    if plt_figure_kwargs is not None: figure_kwargs.update(plt_figure_kwargs)

    title_kwargs = {'fontsize': 25, 'y': 0.95}
    if fig_title_kwargs is not None: title_kwargs.update(fig_title_kwargs)

    # mapping adata short name to layer number
    name_map = dict(zip(layer_keys, range(len(layer_keys))))

    # create the graph object
    G = nx.from_numpy_array(adjacency_df.values, create_using=nx.Graph)
    nx.relabel_nodes(G, lambda i: adjacency_df.index[i], copy=False)
    nx.set_node_attributes(
        G, {node: name_map[node.split('.')[0]] for node in G.nodes}, name='layer')

    # prepare graph for display
    layout = nx.multipartite_layout(G, subset_key='layer')

    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    edge_width = 15 * np.power(weights, 2)  # visual edge emphasis

    if len(layer_keys) > 2:
        for layer in {data['layer'] for key, data in G.nodes.data()}:
            nodes = [node for node in G.nodes if name_map[node.split('.')[0]] == layer]

            if layout_type == 'straight':
                angles = np.linspace(-np.pi / 4, np.pi / 4, len(nodes))
            elif layout_type == 'fan':
                rank = len(nodes)
                angles = np.linspace(-np.pi * rank / 72, np.pi * rank / 72, rank)
            else:
                raise NotImplementedError(f'layout type {layout_type} is not implemented')

            for i, node in enumerate(nodes):
                layout[node] = [layer + 2 * np.cos(angles[i]), np.sin(angles[i])]

    fig, ax = plt.subplots(1, 1, **figure_kwargs)
    nx.draw(G, layout, node_size=3000, with_labels=False, edge_color=weights,
            edge_vmin=0, edge_vmax=1., width=edge_width, ax=ax)

    cmp = plt.matplotlib.cm.ScalarMappable(plt.matplotlib.colors.Normalize(vmin=0, vmax=1))
    plt.colorbar(cmp, orientation='horizontal', cax=fig.add_subplot(18, 5, 86))

    # change color of layers
    for key in layer_keys:
        nx.draw_networkx_nodes(
            G, layout, node_size=2800, nodelist=prog_names_dict[key],
            # node_color=coloring_scheme[key],
            ax=ax)
    nx.draw_networkx_labels(G, layout, font_size=11, ax=ax)

    plt.suptitle(title, **title_kwargs)
    plt.tight_layout()

    return fig
