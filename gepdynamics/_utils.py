#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import typing
import warnings

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import scanpy as sc

from scipy.cluster import hierarchy
from scipy.sparse import csr_matrix
from scipy.stats import rankdata
from sklearn.metrics import jaccard_score
from gprofiler import GProfiler

from gepdynamics._constants import NON_NEG_CMAP, PROJECT_HOME_PATH

PathLike = typing.TypeVar('PathLike', str, bytes, os.PathLike)


# General utilities
def cd_proj_home():
    '''
    Change directory to the project home directory as defined in gepdynamics/_constants
    '''
    os.chdir(PROJECT_HOME_PATH[sys.platform])


def set_dir(path: PathLike) -> Path:
    '''Given a path to a directory, assert its existance or try to create it'''
    path = Path(path)
    if not os.path.isdir(path):
        print(f'Directory "{path}" does not exist.',
              f'trying to create it at {path.resolve()}', sep='\n')
        os.mkdir(path)
    return Path(path)


def fastols(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    '''fast, orthogonal least squares algorithm. Y=Xb'''
    pseudo_inverse = np.linalg.pinv(X)
    return np.dot(pseudo_inverse, Y)


def truncated_spearmans_correlation(data, truncation_level: int = 1000,
                                    smaller_is_better: bool = False, rowvar: bool = True):
    """

    Parameters
    ----------
    data : array_like
        Input data array.
    truncation_level : int, optional
        Maximum value allowed after truncation (default is 1000).
    smaller_is_better : bool, optional
        Specifies whether smaller values are considered better (default is False).
    rowvar : bool, optional
        Determines whether the row correlation is calculated (True) or the column correlation (False)
        (default is True).

    Returns
    -------
    correlation : ndarray
        The truncated Spearman's correlation matrix.

    Notes
    -----
    The truncated Spearman's correlation is calculated by first ranking the data.
    Ranks larger than the truncation level are set to the truncation level before
    calculating the correlation matrix.
    """
    if rowvar:  # transpose the data if the row correlation is desired
        data = data.T
    if smaller_is_better: # reverse the data if smaller values are considered better
        data = -data

    n_rows, n_cols = data.shape

    ranked_data = rankdata(data, axis=0)
    ranked_data[ranked_data > truncation_level] = truncation_level
    return np.corrcoef(ranked_data, rowvar=False)

def joint_hvg_across_stages(adata: sc.AnnData, obs_category_key: str, n_top_genes=5000):
    """
    Identifies joint highly variable genes across different stages/categories in single-cell RNA-seq data.
    Based on Seurat v3 normalized variance [Stuart19]

    Parameters
    ----------
    adata : sc.AnnData
        Scanpy object.
    obs_category_key : str
        Key of the observation category/column in `adata.obs` that represents stages/categories.
    n_top_genes : int, optional
        Number of top highly variable genes to select (default is 5000).

    Returns
    -------
    None

    Modifies
    --------
    Updates `adata.var` by adding two new columns:
    1. obs_category_key+'_max_var_norm' : numpy.ndarray
        Maximum normalized variance across stages/categories.
    2. 'joint_highly_variable' : numpy.ndarray
        Boolean indicating if a gene is one of the top highly variable genes.

    Examples
    --------
    >>> joint_hvg_across_stages(adata, obs_category_key='time_point')

    """
    # Calculate normalized variance of genes across all stages/categories
    hvg_all = sc.pp.highly_variable_genes(adata, flavor='seurat_v3',
                                          n_top_genes=n_top_genes, inplace=False).variances_norm

    #Calculate normalized variance of genes for each stage/category
    hvg_dfs = [sc.pp.highly_variable_genes(
        adata[adata.obs[obs_category_key] == cat], flavor='seurat_v3', n_top_genes=n_top_genes,
        inplace=False).variances_norm for cat in adata.obs[obs_category_key].cat.categories]

    # Stack the variances_norm of all stages/categories
    joint_variances_norm = np.vstack([hvg_all, *hvg_dfs])

    # Compute the maximum variance_norm across stages/categories
    maximal_variance_norm = joint_variances_norm.max(axis=0)

    # Add the maximal_variance_norm as a new column in adata.var
    adata.var[obs_category_key+'_max_var_norm'] = maximal_variance_norm

    # Identify the top highly variable genes based on rank
    adata.var['joint_highly_variable'] = adata.var[obs_category_key+'_max_var_norm'].rank(
        ascending=False, method='min') <= n_top_genes


def read_matlab_h5_sparse(filename: PathLike) -> csr_matrix:
    with h5py.File(filename, "r") as f:
        if set(f.keys()) != {'i', 'j', 'v'}:
            raise NotImplementedError("The h5 keys don't match the row, column, value format")
        if len(f['i'].shape) > 2 or (1 not in f['i'].shape):
            raise NotImplementedError("The sparse keys are not one dimensional")
        
        rows = np.array(f['j'], dtype=int).flatten() - 1
        cols = np.array(f['i'], dtype=int).flatten() - 1
        data = np.array(f['v']).flatten()
        
    return csr_matrix((data, (rows, cols)))


def df_jaccard_score(df: pd.DataFrame) -> np.ndarray:
    """
    Computes the Jaccard score between all pairs of boolean columns in a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): The input pandas DataFrame. Must contain only boolean columns.

    Returns:
        np.ndarray: A symmetric matrix of shape (n_cols, n_cols) where each entry (i, j) represents
            the Jaccard score between column i and column j.

    Raises:
        ValueError: If the input DataFrame contains non-boolean columns.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({
        ...     'A': [True, False, True, False],
        ...     'B': [True, True, False, False],
        ...     'C': [False, True, True, False],
        ...     'D': [False, False, True, True]
        ... })
        >>> df_jaccard_score(df)
        array([[1.        , 0.25      , 0.25      , 0.33333333],
               [0.25      , 1.        , 0.5       , 0.        ],
               [0.25      , 0.5       , 1.        , 0.5       ],
               [0.33333333, 0.        , 0.5       , 1.        ]])
    """
    if not all(df.dtypes == 'bool'):
        raise ValueError('Input DataFrame must contain only boolean columns.')
    
    n_cols = len(df.columns)
    scores = np.zeros((n_cols, n_cols))
    for i in range(n_cols):
        for j in range(n_cols):
            if i <= j:  # to avoid computing the same pairs twice
                scores[i, j] = jaccard_score(df.iloc[:, i], df.iloc[:, j])
                scores[j, i] = scores[i, j]  # since it's a symmetric matrix
    
    return scores


class MyGProfiler(GProfiler):
    '''
    Wrapper of the GProfiler class with special defaults

    Examples
    --------
    >>> gp = utils.MyGProfiler()
    >>> short_list = ['Nppa', 'Cox6a2', 'Atp5b', 'Atp2a2', 'Ndufa4',
                      'Ldb3', 'Atp5a1', 'Atp5g3', 'Uqcr11', 'Kcnk3',
                      'Ttn', 'Ryr2', 'Tmod1', 'Chchd10', 'Cox5a',
                      'Cox7a1', 'Cox7a2', 'Fabp3', 'Myom2', 'Myl7',
                      'Mdh1', 'Cox4i1', 'Sgcg', 'Cox6c', 'Cox8b',
                      'Cox7c', 'Mb', 'Myh6', 'Actn2', 'Corin']
    >>> enrichment = gp.profile(short_list, )
    >>> enrichment_background = gp.profile(short_list, background=all_genes)
    >>> orderd_background = gp.profile(ordered_genes, ordered=True,
                       background=ordered_genes)

    '''

    def __init__(self, user_agent: str = '', base_url: str = None,
                 return_dataframe: bool = True,
                 organism: str = 'hsapiens', #'mmusculus', 
                 sources: typing.List[str] = ['GO:BP', 'WP'],
                 user_threshold: float = 0.001,
                 ):
        super().__init__(user_agent, base_url, return_dataframe)

        self.default_profile_arguments = {
            'organism': organism,
            'user_threshold': user_threshold,
            'sources': sources
        }

    @staticmethod
    def calculate_enrichment(df: pd.DataFrame, ceil: int = 50):
        '''calculate enrichment (n/b)/(N/B) on gprodile results dataframe'''
        tmp = df.intersection_size * df.effective_domain_size \
            / (df.term_size * df.query_size)
        tmp[tmp > ceil] = ceil
        return tmp

    def profile(self, query: typing.Union[str, typing.List[str], typing.Dict[str, typing.List[str]]],
                df_reordered: bool=True, **kwargs) -> pd.DataFrame:
        """
        performs functional profiling of gene lists using various kinds of
        biological evidence. The tool performs statistical enrichment analysis
        to find over-representation of information such as GO terms etc.

        :param query: list of genes to profile. For running multiple queries at
            once, accepts a dictionary of lists as well.
        :param organism: Organism id for profiling. For full list see
            https://biit.cs.ut.ee/gprofiler/page/organism-list
        :param sources: List of annotation sources to include in analysis.
            Defaults to all known.
        :param user_threshold: Significance threshold for analysis.
        :param all_results: If True, return all analysis results regardless of
            statistical significance.
        :param ordered: If True, considers the order of input query to be
            significant. See
            https://biit.cs.ut.ee/gprofiler/page/docs#ordered_gene_lists
        :param no_evidences: If False, the results include lists of
            intersections and evidences for the intersections
        :param combined: If True, performs all queries and combines the results
            into a single table. NB! changes the output format.
        :param measure_underrepresentation: if True, performs test for
            significantly under-represented functional terms.
        :param no_iea: If True, excludes electronically annotated Gene Ontology
            terms before analysis.
        :param domain_scope: "known" for using all known genes as background,
            "annotated" to use all genes annotated for particular datasource.
        :param numeric_namespace: name for the numeric namespace to use if
            there are numeric values in the query.
        :param significance_threshold_method: method for multiple correction.
            "g_SCS"|"bonferroni"|"fdr".
            https://biit.cs.ut.ee/gprofiler/page/docs#significance_threhshold
        :param background: List of genes to use as a statistical background.
        :return: Dataframe of the results
        """
        args = {**deepcopy(self.default_profile_arguments), **kwargs}

        result_df = super().profile(query=query, **args)
        result_df['enrichment'] = self.calculate_enrichment(result_df)

        if df_reordered:
            return result_df[[
                'native', 'name', 'p_value', 'enrichment', 'description',
                'term_size', 'query_size', 'intersection_size',
                'effective_domain_size', 'source']]

        else:
            return result_df


# Basic  plotting functions
def floats_to_colors(floats_array: np.ndarray, cmap: str = 'coolwarm', vmin=None, vmax=None) -> np.ndarray:
    """
    Convert an array of floats to an array of hexadecimal colors using a specified colormap.

    Parameters:
    -----------
    floats_array: np.ndarray
        Array of floats to be converted to colors.

    cmap: str, optional (default: 'coolwarm')
        Name of the colormap to be used for the conversion. 
        Default is 'coolwarm'.
    
    vmin: float, optional (default: None)
        The minimum value for the normalization of the input values.
        If not specified, the minimum value of the input array will be used.

    vmax: float, optional (default: None)
        The maximum value for the normalization of the input values.
        If not specified, the maximum value of the input array will be used.
        
    Returns:
    --------
    np.ndarray
        Array of hexadecimal color codes corresponding to each float value in the input array.
    """    
    cmap = plt.matplotlib.cm.get_cmap(cmap)
    norm = plt.matplotlib.colors.Normalize(vmin=(vmin or floats_array.min()), vmax=(vmax or floats_array.max()))
    rgba_colors = cmap(norm(floats_array))
    
    return np.array([ plt.matplotlib.colors.rgb2hex(rgba_colors[i,:]) for i in range(rgba_colors.shape[0]) ])


def heatmap(matrix, ax=None, cmap=plt.cm.hot, param_dict=None, title=''):
    '''prints a simple heatmap.

    Parameters
    ----------
    matrix : ndarray
        the 2D array to print
    ax : plt.Axes
        The axes to draw to. if none is givin creates a new image
    cmaps_index : integer
        The colorscheme for the heatmap
    param_dict : dict
       Dictionary of kwargs to pass to ax.imshow

    Returns
    -------
    plt.Axes - the axes on which the heatmap is drawn

    Examples
    --------
    A = np.random.rand(4,5)
    ax = heatmap(A)
    '''
    if param_dict is None:
        param_dict = {}

    if ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(matrix, cmap=cmap,
                   interpolation='nearest', **param_dict)
    ax.figure.colorbar(im, ax=ax)

    ax.xaxis.tick_top()
    ax.set_xticks(ticks=np.arange(matrix.shape[1]))

    ax.set_yticks(ticks=np.arange(matrix.shape[0]))

    ax.set_title(title)

    return ax


def heatmap_with_numbers(
        matrix, ax=None, cmap=plt.cm.YlOrRd, title='', param_dict=None):
    '''prints a heatmap with cell values.

    Parameters
    ----------
    matrix : ndarray
        the 2D array to print
    ax : plt.Axes
        The axes to draw to. if none is givin creates a new image
    cmaps_index : integer
        The colorscheme for the heatmap
    param_dict : dict
       Dictionary of kwargs to pass to ax.imshow

    Returns
    -------
    plt.Axes - the axes on which the heatmap is drawn

    Examples
    --------
    A = np.random.rand(4,5)
    ax = heatmap_with_numbers(A)

    '''
    if param_dict is None:
        param_dict = {}

    if ax is None:
        fig, ax = plt.subplots(1)
    ax = heatmap(matrix=matrix, ax=ax,
                 cmap=cmap, param_dict=param_dict)

    ax.set_title(title)

    # Loop over data dimensions and create text annotations.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 0:
                val = 0
            else:
                val = np.round(matrix[i, j], decimals=2)
            ax.text(j, i, val, ha="center", va="center", color="k")

    return ax




# Scanpy AnnData objects related functions:
def _create_usages_norm_adata(adata):
    '''Given an adata with normalized usages of programs, create anndata'''
    var = pd.DataFrame(index=adata.varm['usage_coefs'].columns)

    with warnings.catch_warnings():  # supress 64 -> 32 float warning
        warnings.simplefilter(action='ignore', category=FutureWarning)
        return sc.AnnData(
            adata.obsm['usages_norm'].copy(), obs=adata.obs.copy(),
            var=var.copy(), uns=deepcopy(adata.uns), obsm=deepcopy(adata.obsm))


def get_single_row_color_from_adata(adata: sc.AnnData, col_index: int = -1):
    """
    Assuming the AnnData object has a "row_color" attribute in obsm,
    return a color for each row using the column index

    """
    if 'row_colors' not in adata.obsm_keys():
        return None

    if len(adata.obsm['row_colors'].shape) == 1:
        return adata.obsm['row_colors']
    elif isinstance(adata.obsm['row_colors'], pd.DataFrame):
        return adata.obsm['row_colors'].iloc[:, col_index]
    else:
        return adata.obsm['row_colors'][:, col_index]


def expand_adata_row_colors(adata: sc.AnnData, new_color_column):
    """
    For a given AnnData object, add a new color column to existing row colors
    If the object does not have row colors, return the new color column
    If the object has row colors, return the new color column appended to the existing row colors
    """
    if 'row_colors' not in adata.obsm_keys():
        return new_color_column
    elif isinstance(adata.obsm['row_colors'], pd.core.base.PandasObject):
        df = pd.concat([pd.DataFrame(adata.obsm['row_colors']).reset_index(drop=True),
                          pd.DataFrame(new_color_column).reset_index(drop=True)],
                         axis=1)
        return df.set_index(adata.obs_names)
    else:
        raise ValueError("Unsupported row colors type")


def plot_usages_norm_violin(
        adata: sc.AnnData,
        group_by_key: str,  # obs key
        save_path: typing.Union[PathLike, None]=None,
        close: bool=True
) -> sc.plotting._baseplot_class.BasePlot:
    u_data = _create_usages_norm_adata(adata)
    u_data.obs[group_by_key] = pd.Categorical(
        u_data.obs[group_by_key]).remove_unused_categories()
    _sname = u_data.uns["sname"]

    # Lineages hierarchical clustering
    sc.tl.dendrogram(u_data, group_by_key, cor_method='spearman', use_rep='X')

    # programs hierarchical clustering
    linkage = hierarchy.linkage(
        u_data.X.T, method='average', metric='correlation')
    prog_order = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(
        linkage, u_data.X.T))

    vp = sc.pl.stacked_violin(
        u_data, u_data.var_names[prog_order], groupby=group_by_key,
        return_fig=True, dendrogram=True)
    vp.fig_title = f'{u_data.uns["name"]} program usage per {group_by_key}'

    if save_path is not None:
        vp.savefig(save_path, dpi=180)
    if close:
        plt.close(vp.fig)

    return vp


# Scanpy code for creating color from categorical items: 
# pd.Categorical(adata.obs['timesimple'].map(
#    dict(zip(pd.Categorical(adata.obs['timesimple']).categories, adata.uns['timesimple_colors']))))
def plot_usages_norm_clustermaps(
    adata: sc.AnnData,
    metric='cosine',
    normalized_usages=None,
    columns=None,
    title=None,
    show: bool=False,
    sns_clustermap_params=None) -> sns.axisgrid._BaseGrid:
    '''
    Plots the normalized usages clustermaps, return sns figure object
    '''
    
    if normalized_usages is None:
        normalized_usages = adata.obsm['usages_norm']
    
    if columns is None:
        columns = adata.varm['usage_coefs'].columns
    
    if title is None:
        title = f'{adata.uns["name"]} programs normalized usages, k={normalized_usages.shape[1]}'

    default_clustermap_params = dict(
        row_colors=adata.obsm.get('row_colors', None),
        cmap=NON_NEG_CMAP,
        yticklabels=False,
        xticklabels=True,
        metric=metric,
        vmin=0, vmax=1,
        cbar_pos=(.02, .83, .05, .16))
    
    if sns_clustermap_params is None: sns_clustermap_params = dict()
    default_clustermap_params.update(sns_clustermap_params)
    
    # the entries are colored by a square root transform
    data = np.sqrt(normalized_usages)
    k = data.shape[1]
    
    un_sns = sns.clustermap(
        pd.DataFrame(data, index=adata.obs.index, columns=columns),
        **default_clustermap_params)
    
    # Tick labels are set to match the square root transformation
    with warnings.catch_warnings():  # supress FixedFormatter/FixedLocator warning        
        warnings.simplefilter(action='ignore', category=UserWarning)
        un_sns.ax_cbar.yaxis.set_ticklabels(
            np.round(np.power(np.linspace(0, 1, 6), 2), 2))

    un_sns.fig.suptitle(title, size=30, ha='center', y=1.05)

    if show:
        plt.show() # un_sns.fig.show()

    plt.close()

    return un_sns
