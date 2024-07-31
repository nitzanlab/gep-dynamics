#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Dict, Iterable, Literal, Optional

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


# colors for joint projections plots
mpl_colors = plt.matplotlib.colors

PROJ_COLORS_LIST = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow']
PROJ_COLORS_LIST = [tuple([0.75 * entry for entry in mpl_colors.to_rgb(color)]
                          ) for color in PROJ_COLORS_LIST]
PROJ_COLORS_LIST = [mpl_colors.LinearSegmentedColormap.from_list(
    f'{i}', ['lightgray', color]) for i, color in enumerate(PROJ_COLORS_LIST)]



def get_rank_from_coefs(orig_coefficients, gene_indices, cutoff):
    """ create a dataframe of ranks up to cutoff from a dataframe of coefficients
    with unassigned column for genes that are equal between all programs"""

    coefficients = orig_coefficients.iloc[gene_indices].copy()
    tmp_none = 1 - (coefficients==0).all(axis=1).astype(int) # if all prog_names are zero set to 0 else 1
    coefficients.loc[:,:] = rankdata(-coefficients, axis=0)
    coefficients.insert(0, UNASSIGNED_GENES_COLUMN, tmp_none * cutoff)
    coefficients[coefficients > cutoff] = cutoff + 1

    return coefficients


def calculate_background_genes(nmf_results_list: List['NMFResultBase'],
                                 gene_list_cutoff=401):
    """
    Calculate the background genes for joint best rank metric
    """

    # getting background genes, removing negative valued coefficients
    joint_coefficients = np.hstack([nmf_res.gene_coefs.values for nmf_res in nmf_results_list])
    ranked_data = rankdata(-joint_coefficients, axis=0)
    ranked_data[joint_coefficients<=0] = gene_list_cutoff
    ranked_data[ranked_data >= gene_list_cutoff] = gene_list_cutoff
    bg_genes = np.where(ranked_data.min(axis=1) < gene_list_cutoff)[0]

    return bg_genes


def plot_sankey_for_nmf_results(nmf_results_list: List['NMFResultBase'],
                                gene_list_cutoff=401,
                                cutoff=801, # cutoff for coefficient ranks in comparison
                                display_threshold_counts=100,
                                show_unassigned_genes=True):
    """
    Create Sankey plot for lung development dataset

    # Latest version: looking at genes that have high rank in one of the GEPs,
    # and their coefficients are larger than zero. Genes are passed between geps
    # according to best rank
    """
    if len(nmf_results_list) < 2:
        raise ValueError('Need at least two NMF results to compare')

    # Get the background list of genes
    bg_genes = calculate_background_genes(nmf_results_list, gene_list_cutoff)

    top_coefficients_lists = [get_rank_from_coefs(
        nmf_res.gene_coefs, bg_genes, cutoff) for nmf_res in nmf_results_list]

    if show_unassigned_genes:
        for i, coefficients in enumerate(top_coefficients_lists):
            coefficients.rename(columns={UNASSIGNED_GENES_COLUMN: UNASSIGNED_GENES_COLUMN + f'_{i}'}, inplace=True)

    # find the column with the best value per row
    best_rank_programs_lists = [coefficients.idxmin(axis=1) for coefficients in top_coefficients_lists]

    if not show_unassigned_genes:
        for i, coefficients in enumerate(top_coefficients_lists):
            coefficients.drop(columns=[UNASSIGNED_GENES_COLUMN], inplace=True)

    labels = [column for coefficients in top_coefficients_lists for column in coefficients.columns]
    source = [] # indices correspond to labels
    target = []
    value = []
    link_colors = []

    # calculate links for each pair of programs from adjacent points
    for k in range(len(nmf_results_list)-1):
        offset_source = sum([len(coefficients.columns) for coefficients in top_coefficients_lists[:k]])
        offset_target = sum([len(coefficients.columns) for coefficients in top_coefficients_lists[:k + 1]])
        for i, col_a in enumerate(top_coefficients_lists[k].columns):
            for j, col_b in enumerate(top_coefficients_lists[k+1].columns):
                val = sum((best_rank_programs_lists[k] == col_a) & (best_rank_programs_lists[k+1] == col_b))
                if val > display_threshold_counts:
                    source.append(i + offset_source)
                    target.append(j + offset_target)
                    value.append(val)
                    link_colors.append('lightgreen')
                    if col_a.startswith(UNASSIGNED_GENES_COLUMN) or col_b.startswith(UNASSIGNED_GENES_COLUMN):
                        link_colors[-1] = 'lightgrey'

    # x_positions should loop twice over
    x_positions = [[k / (len(nmf_results_list) - 1)] * top_coefficients_lists[k].shape[1] for k in range(len(nmf_results_list))]
    x_positions = [item for sublist in x_positions for item in sublist]

    # create a list of node colors that is blue for all except unassigned genes
    node_colors = []
    for i, label in enumerate(labels):
        if label.startswith(UNASSIGNED_GENES_COLUMN):
            node_colors.append('gray')
        else:
            node_colors.append('blue')

    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = labels,
            color = node_colors,
            # x=x_positions,
            # y=[0.1]*len(labels)
        ),
        link = dict(
          source = source, target = target, value = value, color=link_colors),
        # arrangement='snap',
    )])

    fig.update_layout(title_text=f"Sankey Diagram for top {gene_list_cutoff-1} prominent gene coefficients. "
                                 f"Background is {len(bg_genes)} highly ranked genes (top {cutoff-1} per program)"
                                 f", threshold={display_threshold_counts}", font_size=10)
    fig.show()


def plot_marker_genes_heatmaps(programs_list: List[pd.Series],
                               marker_genes: List[str],
                               marker_gene_names: Optional[List[str]] = None,
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

    sns.heatmap(df.loc[marker_genes], cmap='coolwarm', vmin=-2, vmax=2)

    if marker_gene_names is not None:
        plt.yticks(ticks=0.5 + np.arange(len(marker_gene_names)), labels=marker_gene_names, rotation=0)

    if title is None:
        title = 'Marker genes coefficients'
    plt.title(title)

    plt.tight_layout()

    if save_file is not None:
        plt.savefig(save_file, dpi=300)
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


def plot_joint_utilization_projection(sets, title, save_file, obsm_coordinates='X_umap', show=False):
    """
    Plot the projection of utilization from consecutive stages

    The sets parameter is a list of tuples, each consisted of the following:
    - adata_x: an AnnData object containing the data to be plotted
    - res_x: the NMFResult object containing the results of the decomposition
    - prog: the index of the program to be plotted
    - label: the label to be displayed in the legend
    - color: the color palette to be used for the scatter

    """

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Prepare a list to hold the legend elements
    legend_elements = []

    for adata_x, res_x, prog, label, cmap in sets:
        coordinates = adata_x.obsm[obsm_coordinates]
        plt.scatter(coordinates[:, 0], coordinates[:, 1],
                    c=res_x.norm_usages[:, prog], s=2, cmap=cmap)

        colors = [cmap(i) for i in [0.1, 0.45, 0.9]]
        Line2D = plt.matplotlib.lines.Line2D

        legend_elements.append(((*[
            Line2D([0], [0], marker='o', color=colors[i], linestyle='None', markersize=6, label=label) for i in range(3)],), label))

    # Create the custom legend
    legend = ax.legend([elem[0] for elem in legend_elements],
                       [elem[1] for elem in legend_elements],
                       handler_map={tuple: plt.matplotlib.legend_handler.HandlerTuple(ndivide=None)})

    # Add the legend to the plot
    ax.add_artist(legend)

    plt.xticks([])
    plt.xlabel(f'{obsm_coordinates[2:].upper()}1')

    plt.yticks([])
    plt.ylabel(f'{obsm_coordinates[2:].upper()}2')

    plt.title(title)

    plt.tight_layout()
    plt.savefig(save_file, dpi=300)

    if show:
        plt.show()

    plt.close()

