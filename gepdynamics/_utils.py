#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import typing
import warnings

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import scanpy as sc

from scipy.sparse import csr_matrix

from gepdynamics._constants import NON_NEG_CMAP 

PathLike = typing.TypeVar('PathLike', str, bytes, os.PathLike)


# General utilities
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