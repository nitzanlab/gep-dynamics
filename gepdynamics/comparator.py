#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module implementing gene expression programs (GEPs) comparator between two
timepoints, examining timepoint a as the basis for timepoint b

input:
1. timepoint A data
2. timepoint A usages matrix (H in Matan’s terminology)
3. timepoint B data

pseudocode:
0. Identify A GEPs on timepoints A and B jointly highly variable genes
1. Decompose timepoint B data de-novo and using timepoint A GEP matrix with
    0,1,2,3 additional degrees of freedom.
2. Calculate gene coefficients for all the GEPs
3. Calculate the coefficients correlation of each A GEP with all the novel GEPs
4. Calculate the usage correlation of the de-novo and pfnmf GEPs
5. Input the correlation results to the fingerprint surrogate table
6. Classify GEPs to operators according to the fingerprint

output:
1. Classification of each GEP in timepoint A according to the operator
2. Fingerprint figure for each GEP in the timepoint A decomposition.

"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from sklearn.decomposition import _nmf as sknmf

import gepdynamics._utils as _utils
import gepdynamics.cnmf as cnmf
from gepdynamics._constants import NMFEngine, Stage, NUMBER_HVG, NMF_TOLERANCE


class Comparator(object):
    def __init__(self,
                 adata_a: sc.AnnData,
                 usages_matrix_a: np.ndarray,
                 adata_b: sc.AnnData,
                 results_dir: _utils.PathLike,
                 nmf_engine: NMFEngine = NMFEngine.sklearn,
                 beta_loss: str = 'kullback-leibler',
                 max_nmf_iter = 800,
                 max_added_rank: int = 3,
                 device: str = None
                 ):
        self.adata_a = adata_a
        self.usages_matrix_a = usages_matrix_a
        self.rank_a = self.usages_matrix_a.shape[1]
        self.adata_b = adata_b

        self.results_dir = results_dir
        self.beta_loss = beta_loss
        self.max_nmf_iter = max_nmf_iter
        self.nmf_engine = nmf_engine
        self.device = device
        self.max_added_rank = max_added_rank

        # Fields to be filled in later
        self.joint_hvgs = None
        self.geps_a = None

        self.stage = Stage.INITIALIZED

    def normalize_adata_for_decomposition(self, adata: sc.AnnData, method='variance') -> sc.AnnData:
        """
        Normalize adata for decomposition
        """
        if method == 'variance':
            return sc.pp.scale(adata[:, self.joint_hvgs].X.toarray(), zero_center=False)

    def extract_geps_on_jointly_hvgs(self, min_cells_per_gene: int = 5):
        """
        Identify A GEPs on timepoints A and B jointly highly variable genes
        """
        genes_subset = (self.adata_a.var.n_cells >= min_cells_per_gene) & (
            self.adata_b.var.n_cells >= min_cells_per_gene)
        hvg_a = sc.pp.highly_variable_genes(
            self.adata_a[:, genes_subset], flavor='seurat_v3', n_top_genes=100_000, inplace=False)
        hvg_b = sc.pp.highly_variable_genes(
            self.adata_b[:, genes_subset], flavor='seurat_v3', n_top_genes=100_000, inplace=False)
        joint_hvg_rank = hvg_a.highly_variable_rank + hvg_b.highly_variable_rank
        self.joint_hvgs = joint_hvg_rank.sort_values()[: NUMBER_HVG].index

        # Extracting GEPs on joint HVG, working in the transposed notation
        #  to get the programs: X_a.T ~ geps.T @ usages.T

        nmf_kwargs = {'H': self.usages_matrix_a.T.copy(),
                      'update_H': False,
                      'n_components': self.rank_a,
                      'tol': NMF_TOLERANCE,
                      'max_iter': self.max_nmf_iter,
                      'beta_loss': self.beta_loss,
                      'solver': 'mu'
                      }

        X_a = self.normalize_adata_for_decomposition(self.adata_a)

        if self.nmf_engine in (NMFEngine.torchnmf, NMFEngine.consensus_torch):
            W, H, n_iter = cnmf.nmf_torch(X_a.T, nmf_kwargs, device=self.device)
        else:
            W, H, n_iter = sknmf.non_negative_factorization(X_a.T, **nmf_kwargs)

        self.geps_a = W.T / np.linalg.norm(W.T, ord=2, axis=0, keepdims=True)

        self.stage = Stage.PREPARED


    def decompose_b(self):
        """
        Decompose timepoint B data de-novo and using timepoint A GEP matrix with 0,1,2,3 additional degrees of freedom.

        Parameters
        ----------

        # TODO: add normalization option
        """
        pass

        # set decomposition result attributes / class

    def plot_usages_clustermaps(self):
        """
        Plot the usages of the decomposed GEPs as clustermaps
        """
        pass

    def calculate_gene_coefficients(self):
        """
        Calculate gene coefficients for all of the newly decomposed GEPs
        """
        pass

        # set gene coefficients attributes / class

    def calculate_correlations(self):
        """
        Calculate the usage correlation of the de-novo and pfnmf geps
        """
        pass

        # set correlations attributes / class

    def plot_fingerprint(self):
        """
        Input the correlation results to the fingerprint surrogate table
        """
        pass






