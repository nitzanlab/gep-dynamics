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
import torch

from sklearn.decomposition import _nmf as sknmf

import gepdynamics._utils as _utils
import gepdynamics.cnmf as cnmf
from gepdynamics import pfnmf
from gepdynamics._constants import NMFEngine, Stage, NUMBER_HVG, NMF_TOLERANCE


class NMFResultBase(object):
    def __init__(self, name: str, loss_per_cell: np.ndarray, rank: int, algorithm: str):
        self.name = name
        self.loss_per_cell = loss_per_cell
        self.rank = rank
        self.algorithm = algorithm

        self.loss = np.sum(self.loss_per_cell)

        # Attributes added later on
        self.prog_names = None
        self.norm_usages = None
        self.prog_percentages = None
        self.prog_labels_1l = None
        self.prog_labels_2l = None


class NMFResult(NMFResultBase):
    def __init__(self, name, loss_per_cell, rank, W, H):
        super().__init__(name, loss_per_cell, rank, 'regular')
        self.W = W
        self.H = H


class PFNMFResult(NMFResultBase):
    def __init__(self, name, loss_per_cell, rank, W1, W2, H1, H2):
        super().__init__(name, loss_per_cell, rank, 'pfnmf')
        self.W1 = W1
        self.W2 = W2
        self.H1 = H1
        self.H2 = H2


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
                 device: str = None,
                 verbosity: int = 0
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
        self.verbosity = verbosity

        # Fields to be filled in later
        self.joint_hvgs = None
        self.geps_a = None
        self.fnmf_result = None
        self.pfnmf_results = []
        self.denovo_results = []

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
            W, H, n_iter = cnmf.nmf_torch(X_a.T, nmf_kwargs, device=self.device, verbose=(self.verbosity > 1))
        else:
            W, H, n_iter = sknmf.non_negative_factorization(X_a.T, **nmf_kwargs, verbose=(self.verbosity > 1))

        self.geps_a = W.T / np.linalg.norm(W.T, ord=2, axis=1, keepdims=True)

        self.stage = Stage.PREPARED

    def decompose_b(self, repeats: int = 1):
        """
        Decompose timepoint B data de-novo and using timepoint A GEP matrix with 0,1,2,3 additional degrees of freedom.

        Parameters
        ----------

        # TODO: add normalization option
        # TODO: add support for repeats > 1 in de-novo and fnmf
        """

        X_b = self.normalize_adata_for_decomposition(self.adata_b)

        if self.nmf_engine == NMFEngine.sklearn:
            self.decompose_b_fnmf(X_b)
            self.decompose_b_denovo(X_b)

        elif self.nmf_engine == NMFEngine.torchnmf:
            if 'torch' not in dir():
                import torch
            tens = torch.tensor(X_b).to(self.device)

            self.decompose_b_fnmf(X_b, tens)
            self.decompose_b_denovo(X_b, tens)

            del tens

        else:
            # not implementd
            raise NotImplementedError(f'nmf engine {self.nmf_engine} not implemented')

        self.decompose_b_pfnmf(X_b, repeats)

        self.stage = Stage.DECOMPOSED

    def decompose_b_fnmf(self, X_b: np.ndarray, tens: torch.Tensor=None):
        """
        Decompose timepoint B data using timepoint A GEPs and no additional GEPs

        Parameters
        ----------
        """

        # X_b ~ W @ geps_a
        nmf_kwargs = {
            'H': self.geps_a.copy(),
            'update_H': False,
            'n_components': self.rank_a,
            'tol': NMF_TOLERANCE,
            'max_iter': self.max_nmf_iter,
            'beta_loss': self.beta_loss,
            'solver': 'mu'}

        self.fnmf_result = self._run_nmf(
            X_b, nmf_kwargs, self.adata_a.uns['name'], tens, verbose=self.verbosity)

    def decompose_b_denovo(self, X_b: np.ndarray, tens: torch.Tensor = None):
        """
        Decompose timepoint B data de-novo

        Parameters
        ----------
        """
        self.denovo_results = []

        for added_rank in range(self.max_added_rank + 1):
            rank = self.rank_a + added_rank

            nmf_kwargs={
                'n_components': rank,
                'tol': NMF_TOLERANCE,
                'max_iter': self.max_nmf_iter,
                'beta_loss': self.beta_loss,
                'solver': 'mu'
               }

            self.denovo_results.append(self._run_nmf(
                X_b, nmf_kwargs, f'dn_{rank}', tens, verbose=self.verbosity))

    def decompose_b_pfnmf(self, X_b: np.ndarray, repeats: int = 1):
        """
        Decompose timepoint B data using timepoint A GEP matrix with 1,2,3 additional degrees of freedom.

        Parameters
        ----------

        # TODO: add multiple repeats and selection of the best
        # TODO: improve verbosity code to include what step we are in
        """
        # pfnmf is written for constant W_1, so we will transpose as needed:
        # X_b ~ W_1 @ geps_a + W_2 @ H_2  <--> X_b.T ~ geps_a.T @ W_1.T + H_2.T @ W_2.T

        self.pfnmf_results = []

        for added_rank in range(1, self.max_added_rank + 1):
            if self.verbosity:
                print(f"Working on added rank = {added_rank}")

            name = self.adata_a.uns['name'] + f'e{added_rank}'
            rank = self.rank_a + added_rank

            best_loss = np.infty

            for repeat in range(repeats):
                w1, h1, w2, h2, n_iter = pfnmf.pfnmf(
                    X_b.T, self.geps_a.T, rank_2=added_rank,
                    beta_loss=self.beta_loss, tol=NMF_TOLERANCE,
                    max_iter=self.max_nmf_iter, verbose=(self.verbosity > 1))

                loss_per_cell = pfnmf.calc_beta_divergence(
                    X_b.T, w1, w2, h1, h2, self.beta_loss, per_column=True)

                final_loss = np.sum(loss_per_cell)

                if final_loss <= best_loss:
                    best_loss = final_loss

                    best_result = PFNMFResult(
                        name, loss_per_cell, rank, w1, w2, h1, h2)

                if self.verbosity:
                    print(f"repeat {repeat}, after {n_iter} iterations reached"
                          f"error = {final_loss: .1f}")

            self.pfnmf_results.append(best_result)

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

    @staticmethod
    def _run_nmf(x, nmf_kwargs, name, tens: torch.Tensor = None, verbose: int = 0) -> NMFResult:
        if tens is not None:
            W, H, _ = cnmf.nmf_torch(x, nmf_kwargs, tens, verbose=(verbose > 0))
        else:
            W, H, _ = sknmf.non_negative_factorization(x, **nmf_kwargs, verbose=verbose)

        # X ~ W @ H, transpose for cells to be columns
        loss_per_cell = pfnmf.calc_beta_divergence(
            x.T, H.T, np.zeros((H.shape[1], 0)),
            W.T, np.zeros((0, W.shape[0])), per_column=True)

        return NMFResult(
            name=name,
            loss_per_cell=loss_per_cell,
            rank=W.shape[1],
            W=W,
            H=H)



