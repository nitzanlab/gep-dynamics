#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import typing

from pathlib import Path

import numpy as np
import h5py

from scipy.sparse import csr_matrix

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