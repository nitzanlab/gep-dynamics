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


# Scanpy AnnData objects related functions:
def _create_usages_norm_adata(adata):
    '''Given an adata with normalized usages of programs, create anndata'''
    var = pd.DataFrame(index=adata.varm['usage_coefs'].columns)

    with warnings.catch_warnings():  # supress 64 -> 32 float warning
        warnings.simplefilter(action='ignore', category=FutureWarning)
        return sc.AnnData(
            adata.obsm['usages_norm'].copy(), obs=adata.obs.copy(),
            var=var.copy(), uns=deepcopy(adata.uns), obsm=deepcopy(adata.obsm))


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
    adata: sc.AnnData, metric='cosine', show: bool=False) -> sns.axisgrid._BaseGrid:
    '''
    Plots the normalized usages clustermaps, return sns figure object
    '''
    
    # the entries are colored by a square root transform
    data = np.sqrt(adata.obsm['usages_norm'])
    k = data.shape[1]
    
    un_sns = sns.clustermap(
        pd.DataFrame(data, index=adata.obs.index,
                     columns=adata.varm['usage_coefs'].columns),
        row_colors=adata.obsm.get('row_colors', None), cmap=NON_NEG_CMAP,
        yticklabels=False, metric=metric, vmin=0, vmax=1,
        cbar_pos=(.02, .83, .05, .16))  # (left, bottom, width, height)
    
    # Tick labels are set to match the square root transformation
    with warnings.catch_warnings():  # supress FixedFormatter/FixedLocator warning        
        warnings.simplefilter(action='ignore', category=UserWarning)
        un_sns.ax_cbar.yaxis.set_ticklabels(
            np.round(np.power(np.linspace(0, 1, 6), 2), 2))

    un_sns.fig.suptitle(
        f'{adata.uns["name"]} programs normalized usages, k={k}',
        size=30, ha='center', y=1.05)

    if show:
        display(un_sns.fig)

    plt.close()

    return un_sns



# Basic  plotting functions

def heatmap(matrix, ax=None, param_dict=None, title=''):
    '''prints a simple heatmap.

    Parameters
    ----------
    matrix : ndarray
        the 2D array to print
    ax : plt.Axes
        The axes to draw to. if none is givin creates a new image
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

    im = ax.imshow(matrix, cmap=plt.cm.YlOrRd,
                   interpolation='nearest', **param_dict)
    ax.figure.colorbar(im, ax=ax)

    ax.xaxis.tick_top()
    ax.set_xticks(ticks=np.arange(matrix.shape[1]))

    ax.set_yticks(ticks=np.arange(matrix.shape[0]))

    ax.set_title(title)

    return ax


def heatmap_with_numbers(
        matrix, ax=None, title='', param_dict={}):
    '''prints a heatmap with cell values.

    Parameters
    ----------
    matrix : ndarray
        the 2D array to print
    ax : plt.Axes
        The axes to draw to. if none is givin creates a new image
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
    ax = heatmap(matrix=matrix, ax=ax, param_dict=param_dict)

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