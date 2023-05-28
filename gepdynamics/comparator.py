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

import os

from copy import copy
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

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
        self.gene_coefs = None
        self.gsea_results = None

    def calculate_prog_labels(self):
        self.prog_percentages = self.norm_usages.sum(axis=0) * 100 / len(self.loss_per_cell)
        self.prog_labels_1l = [name + f' ({self.prog_percentages[i]: 0.1f}%)' for i, name in enumerate(self.prog_names)]
        self.prog_labels_2l = [name + f'\n({self.prog_percentages[i]: 0.1f}%)' for i, name in enumerate(self.prog_names)]

    @staticmethod
    def aggregate_results(results: list) -> Tuple[list, list, np.ndarray, list]:
        """
        Aggregate results from a list of NMFResult objects.

        Parameters
        ----------
        results : list
            A list of NMFResult objects.

        Returns
        -------
        Tuple[list, list, np.ndarray, list]
            A tuple containing the aggregated results:
            - joint_ranks: List of ranks from each NMFResult object.
            - joint_names: List of program names from all NMFResult objects.
            - joint_usages: Concatenated array of norm_usages from all NMFResult objects.
            - joint_labels: List of program labels from all NMFResult objects.
        """
        joint_ranks = [r.rank for r in results]
        joint_names = [name for r in results for name in r.prog_names]
        joint_labels = [label for r in results for label in r.prog_labels_1l]

        usages = [r.norm_usages for r in results]

        joint_usages = np.concatenate(usages, axis=1)

        return joint_ranks, joint_names, joint_usages, joint_labels

    def calculate_gene_coefs(self, z_scores: np.ndarray, gene_index: pd.Index):
        """
        Calculate gene coefficients for each program.

        Parameters
        ----------
        z_scores : np.ndarray
            Z-scores matrix of shape (n_genes, n_cells).
        gene_index : pd.Index
            Gene index.

        """
        self.gene_coefs = pd.DataFrame(
            _utils.fastols(self.norm_usages, z_scores).T,
            index=gene_index,
            columns=self.prog_names)


class NMFResult(NMFResultBase):
    def __init__(self, name, loss_per_cell, rank, W, H):
        super().__init__(name, loss_per_cell, rank, 'regular')
        self.W = W
        self.H = H

        self.norm_usages = W / W.sum(axis=1, keepdims=True)
        self.prog_names = [f'{name}.p{i}' for i in range(rank)]

        super().calculate_prog_labels()


class PFNMFResult(NMFResultBase):
    def __init__(self, name, loss_per_cell, rank, W1, W2, H1, H2):
        super().__init__(name, loss_per_cell, rank, 'pfnmf')
        self.W1 = W1
        self.W2 = W2
        self.H1 = H1
        self.H2 = H2

        usages = np.concatenate([W1, W2], axis=1)
        self.norm_usages = usages / usages.sum(axis=1, keepdims=True)

        self.prog_names = [f'{name}.p{i}' for i in range(W1.shape[1])]
        self.prog_names.extend([f'{name}.e{i}' for i in range(W2.shape[1])])

        super().calculate_prog_labels()


class Comparator(object):
    """
    Class implementing the comparison of two datasets by examening the applicability
    of GEPs in one timepoint to the decomposition of the second timepoint.

    # ToDo: create log file
    # ToDo: add colors
    """
    def __init__(self,
                 adata_a: sc.AnnData,
                 usages_matrix_a: np.ndarray,
                 adata_b: sc.AnnData,
                 results_dir: _utils.PathLike,
                 nmf_engine: NMFEngine = NMFEngine.sklearn,
                 beta_loss: str = 'kullback-leibler',
                 max_nmf_iter: int = 800,
                 max_added_rank: int = 3,
                 device: str = None,
                 verbosity: int = 0
                 ):
        """
        Initialize the Comparator class.

        Parameters
        ----------
        adata_a : sc.AnnData
            Annotated data for timepoint A.
        usages_matrix_a : np.ndarray
            Usages matrix for timepoint A (H in Matan's terminology).
        adata_b : sc.AnnData
            Annotated data for timepoint B.
        results_dir : _utils.PathLike
            Path to the directory where results will be stored.
        nmf_engine : NMFEngine, optional
            NMF engine to use for decomposition, by default NMFEngine.sklearn.
        beta_loss : str, optional
            Beta divergence loss function, by default 'kullback-leibler'.
        max_nmf_iter : int, optional
            Maximum number of iterations for NMF, by default 800.
        max_added_rank : int, optional
            Maximum additional rank for GEPs, by default 3.
        device : str, optional
            Device to use for torch-based NMF engine, by default None.
        verbosity : int, optional
            Level of verbosity for logging. Valid values are 0 (no logging),
            1 (minimal logging), and 2 (detailed logging). Defaults to 0.
        """
        self.adata_a = adata_a
        self.usages_matrix_a = usages_matrix_a
        self.rank_a = self.usages_matrix_a.shape[1]
        self.adata_b = adata_b
        self.a_sname = self.adata_a.uns['sname']
        self.b_sname = self.adata_b.uns['sname']

        self.results_dir = _utils.set_dir(results_dir)
        self.beta_loss = beta_loss
        self.max_nmf_iter = max_nmf_iter
        self.nmf_engine = nmf_engine
        self.device = device
        self.max_added_rank = max_added_rank
        self.verbosity = verbosity

        # Fields to be filled in later
        self.joint_hvgs = None
        self.geps_a = None
        self.a_result = None
        self.fnmf_result = None
        self.pfnmf_results = []
        self.denovo_results = []
        self._all_results = []

        self.fingerprints = [None] * self.rank_a

        self.stage = Stage.INITIALIZED

    def __repr__(self):
        """
        String representation of the Comparator object.
        """
        return f'Comparator(adata_a={self.a_sname}, adata_b={self.b_sname}) at stage {self.stage}'


    def save_to_file(self, filename: _utils.PathLike):
        """
        Save the Comparator object to a ".npz" file, excluding the adata fields.

        Parameters
        ----------
        filename : PathLike
            The path and filename to save the object to.

        Returns
        -------
        None
            This method does not return anything.

        Notes
        -----
        This method creates a copy of the Comparator object and sets the `adata_a` and `adata_b` fields to `None`.
        The copied object is then saved to the specified file using the `.npz` format.

        Examples
        --------
        >>> comp = Comparator()
        >>> comp.save_to_file('comp_object.npz')
        """
        copy_of_self = copy(self)
        copy_of_self.adata_a = None
        copy_of_self.adata_b = None

        # save the copied object
        np.savez(filename, obj=copy_of_self)

    @staticmethod
    def load_from_file(filename: _utils.PathLike, adata_a: sc.AnnData, adata_b: sc.AnnData) -> 'Comparator':
        """
        Load a Comparator object from a file.

        Parameters
        ----------
        filename : PathLike
            The path and filename of the file to load the object from.
        adata_a : scanpy.AnnData
            The `AnnData` object associated with the `adata_a` attribute of the loaded object.
        adata_b : scanpy.AnnData
            The `AnnData` object associated with the `adata_b` attribute of the loaded object.

        Returns
        -------
        Comparator
            The loaded Comparator object.

        Notes
        -----
        This method loads a Comparator object from a file saved in the `.npz` format.
        The loaded object is assigned the provided `adata_a` and `adata_b` attributes before returning.

        Examples
        --------
        >>> comp = Comparator.load_from_file('comp_object.npz', adata_a, adata_b)
        """
        # make sure the file exists
        assert os.path.exists(filename), f'File {filename} does not exist'

        new_instance = np.load(filename, allow_pickle=True)['obj'].item()

        assert new_instance.a_sname == adata_a.uns['sname'],\
            f"adata_a's sname ({adata_a.uns['sname']}) does not match the sname" \
            f" in the loaded object ({new_instance.a_sname})"

        assert new_instance.b_sname == adata_b.uns['sname'],\
            f"adata_b's sname ({adata_b.uns['sname']}) does not match the sname" \
            f" in the loaded object ({new_instance.b_sname})"

        new_instance.adata_a = adata_a
        new_instance.adata_b = adata_b
        return new_instance

    def _normalize_adata_for_decomposition(self, adata: sc.AnnData, method='variance') -> sc.AnnData:
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

        X_a = self._normalize_adata_for_decomposition(self.adata_a)

        if self.verbosity > 0:
            print(f'Extracting A GEPs on jointly highly variable genes')
        if self.nmf_engine in (NMFEngine.torchnmf, NMFEngine.consensus_torch):
            W, H, n_iter = cnmf.nmf_torch(X_a.T, nmf_kwargs, device=self.device, verbose=(self.verbosity > 1))
        else:
            W, H, n_iter = sknmf.non_negative_factorization(X_a.T, **nmf_kwargs, verbose=(self.verbosity > 1))

        self.geps_a = W.T / np.linalg.norm(W.T, ord=2, axis=1, keepdims=True)

        loss_per_cell = pfnmf.calc_beta_divergence(
            X_a.T, W, np.zeros((W.shape[0], 0)),
            H, np.zeros((0, H.shape[1])), per_column=True)

        self.a_result = NMFResult(
            name='A',
            loss_per_cell=loss_per_cell,
            rank=W.shape[1],
            W=H.T,
            H=W.T)

        # calculate gene coefs for each GEP
        z_scores = sc.pp.normalize_total(self.adata_a, target_sum=1e6,
                                         inplace=False)['X']
        z_scores = sc.pp.log1p(z_scores)
        z_scores = sc.pp.scale(z_scores, max_value=10)

        self.a_result.calculate_gene_coefs(z_scores, self.adata_a.var_names)

        self.stage = Stage.PREPARED

    def examine_adata_a_decomposition_on_jointly_hvgs(self, show: bool = False, save: bool = True):
        """
        Examine the decomposition of A on jointly HVGs

        old code:
        row_colors = pd.concat([pd.Series(k_12.obsm.get('row_colors'), name='cluster', index=k_12.obs.index),
                       pd.Series(_utils.floats_to_colors(loss_per_cell, cmap='RdYlGn_r', vmax=1000), name='residual', index=k_12.obs.index)], axis=1)

        un_sns = _utils.plot_usages_norm_clustermaps(k_12, normalized_usages=(H / np.sum(H, axis=0, keepdims=True)).T,
            title='K12 decomposition on HVG', show=True, sns_clustermap_params={'row_colors': row_colors})

        """

        if self.stage in (Stage.INITIALIZED):
            raise RuntimeError('Must extract GEPs on jointly HVGs first')

        dec_folder = _utils.set_dir(self.results_dir.joinpath('decompositions'))

        res = self.a_result
        title = f"{self.a_sname} normalized usages of " \
                f"original GEPs, k={res.rank}"

        row_colors = pd.concat([
            pd.Series(self.adata_a.obsm.get('row_colors'),
                      name='cluster', index=self.adata_a.obs.index),
            pd.Series(_utils.floats_to_colors(res.loss_per_cell,
                                              cmap='RdYlGn_r', vmax=1200),
                      name='residual', index=self.adata_a.obs.index)], axis=1)

        un_sns = _utils.plot_usages_norm_clustermaps(
            self.adata_a, normalized_usages=res.norm_usages,
            columns=res.prog_labels_2l, title=title, show=False,
            sns_clustermap_params={'row_colors': row_colors})

        if save:
            un_sns.savefig(dec_folder.joinpath(f'{res.name}_normalized_usages_clustermap.png'))
        if show:
            un_sns.fig.show()
        plt.close()

        plt.hist(res.loss_per_cell, bins=25, range=(0, min(2500, res.loss_per_cell.max())))
        plt.title(f'{res.name} loss per cell distribution')

        if save:
            plt.savefig(dec_folder.joinpath(f'{res.name}_loss_per_cell_distribution.png'))
        if show:
            plt.show()
        plt.close()


    def decompose_b(self, repeats: int = 1):
        """
        Decompose timepoint B data de-novo and using timepoint A GEP matrix with 0,1,2,3 additional degrees of freedom.

        Parameters
        ----------

        # TODO: add normalization option
        # TODO: add support for repeats > 1 in de-novo and fnmf
        """
        if self.stage == Stage.INITIALIZED:
            self.extract_geps_on_jointly_hvgs()

        X_b = self._normalize_adata_for_decomposition(self.adata_b)

        if self.nmf_engine == NMFEngine.sklearn:
            if self.verbosity > 0:
                print('Decomposing B using A GEPs and no additional GEPs')
            self._decompose_b_fnmf(X_b)

            if self.verbosity > 0:
                print('Decomposing B de-novo')
            self._decompose_b_denovo(X_b)

        elif self.nmf_engine == NMFEngine.torchnmf:
            if 'torch' not in dir():
                try:
                    import torch
                except ImportError:
                    raise ImportError("torch module is not installed. To use torchnmf engine please install torch.")
            tens = torch.tensor(X_b).to(self.device)

            if self.verbosity > 0:
                print('Decomposing B using A GEPs and no additional GEPs')
            self._decompose_b_fnmf(X_b, tens)

            if self.verbosity > 0:
                print('Decomposing B de-novo')
            self._decompose_b_denovo(X_b, tens)

            del tens

        else:
            # not implementd
            raise NotImplementedError(f'nmf engine {self.nmf_engine} not implemented')

        if self.verbosity > 0:
            print(f'Decomposing B using A GEPs and up to {self.max_added_rank} additional GEPs')
        self._decompose_b_pfnmf(X_b, repeats)

        self._all_results = self.denovo_results + [self.fnmf_result] + self.pfnmf_results

        self._calculate_gene_coefficients()

        self.stage = Stage.DECOMPOSED

    def _decompose_b_fnmf(self, X_b: np.ndarray, tens: 'torch.Tensor'=None):
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
            X_b, nmf_kwargs, self.a_sname, tens, verbose=self.verbosity)

    def _decompose_b_denovo(self, X_b: np.ndarray, tens: 'torch.Tensor' = None):
        """
        Decompose timepoint B data de-novo

        # ToDo: add support for repeats > 1 in de-novo
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

    def _decompose_b_pfnmf(self, X_b: np.ndarray, repeats: int = 1):
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

            name = self.a_sname + f'e{added_rank}'
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

                    # transposing back to original orientation
                    best_result = PFNMFResult(
                        name, loss_per_cell, rank, h1.T, h2.T, w1.T, w2.T)

                    if self.verbosity:
                        print(f"repeat {repeat}, after {n_iter} iterations reached "
                              f"error = {final_loss: .1f}")

            self.pfnmf_results.append(best_result)

    def _calculate_gene_coefficients(self):
        """
        Calculate gene coefficients for all the newly decomposed GEPs
        """
        z_scores = sc.pp.normalize_total(self.adata_b, target_sum=1e6, inplace=False)['X']
        z_scores = sc.pp.log1p(z_scores)
        z_scores = sc.pp.scale(z_scores, max_value=10)

        for res in self._all_results:
            res.calculate_gene_coefs(z_scores, self.adata_b.var_names)

    def print_errors(self):
        """
        Print the errors of the decompositions

        """
        if self.stage in (Stage.INITIALIZED, Stage.PREPARED):
            raise RuntimeError('Must decompose before printing errors')

        for result in self._all_results:
            print(f"{result.name} error = {result.loss: .1f}")

    def plot_loss_per_cell_histograms(self, bins: int = 25, max_hist_value: float = 2500, max_kde_value: float = 2000, show: bool = False, save: bool = True):
        """
        Plot the histogram of the loss per cell for each decomposition,
        violin plots of the loss per cell for the decomposition together,
        and the KDE of the loss per cell for the decomposition together.

        Parameters
        ----------
        bins : int, optional
            Number of bins for the histogram. Defaults to 20.
        max_hist_value: float, optional
            maximum value of x in the histograms. Defaults to 2500.
        max_kde_value: float, optional
            maximum value of x in the KDE comparison. Defaults to 2000.
        show : bool, optional
            Whether to display the plot. Defaults to False.
        save : bool, optional
            Whether to save the plot as an image. Defaults to True.
        """
        if self.stage in (Stage.INITIALIZED, Stage.PREPARED):
            raise RuntimeError('Must decompose before plotting error per cell histograms')

        dec_folder = _utils.set_dir(self.results_dir.joinpath('decompositions'))

        for result in self._all_results:
            plt.hist(result.loss_per_cell, bins=bins, range=(0, min(max_hist_value, result.loss_per_cell.max())))
            plt.title(f'{result.name} loss per cell distribution')

            if save:
                plt.savefig(dec_folder.joinpath(f'{result.name}_loss_per_cell_distribution.png'))
            if show:
                plt.show()
            plt.close()

        df = pd.concat([pd.Series(res.loss_per_cell, name=res.name) for res in self._all_results], axis=1)
        df = pd.melt(df, var_name='decomposition', value_name='loss per cell')
        sns.violinplot(x='decomposition', y='loss per cell', data=df)

        plt.title('loss per cell')
        plt.setp(plt.gca().get_xticklabels(), rotation=45)
        if save:
            plt.savefig(dec_folder.joinpath('loss_per_cell_violin.png'))
        if show:
            plt.show()
        plt.close()

        _utils.sns.kdeplot(df, hue="decomposition", x="loss per cell", clip=(0, max_kde_value))
        plt.tight_layout()
        if save:
            plt.savefig(dec_folder.joinpath('loss_per_cell_kde.png'))
        if show:
            plt.show()
        plt.close()

    def plot_usages_clustermaps(self, show: bool = False, save: bool = True):
        """
        Plot the usages of the decomposed GEPs as clustermaps
        """
        if self.stage in (Stage.INITIALIZED, Stage.PREPARED):
            raise RuntimeError('Must decompose before plotting usages clustermaps')

        dec_folder = _utils.set_dir(self.results_dir.joinpath('decompositions'))

        for res in self._all_results:
            name = res.name
            if 'dn' in name:
                title = f"{self.b_sname} normalized usages of " \
                        f" de-novo GEPs, k={res.rank}"
            elif res.rank == self.rank_a:
                title = f"{self.b_sname} normalized usages of " \
                        f"{self.a_sname} GEPs (rank={res.rank})"
            elif res.algorithm == 'pfnmf':
                title = f"{self.b_sname} normalized usages of " \
                        f"{self.a_sname} GEPs + " \
                        f"{res.rank - self.rank_a} novel GEPs"

            row_colors = pd.concat([
                pd.Series(self.adata_b.obsm.get('row_colors'),
                          name='cluster', index=self.adata_b.obs.index),
                pd.Series(_utils.floats_to_colors(res.loss_per_cell,
                                                  cmap='RdYlGn_r', vmax=1200),
                          name='residual', index=self.adata_b.obs.index)], axis=1)

            un_sns = _utils.plot_usages_norm_clustermaps(
                self.adata_b, normalized_usages=res.norm_usages,
                columns=res.prog_labels_2l, title=title, show=False,
                sns_clustermap_params={'row_colors': row_colors})

            if save:
                un_sns.savefig(dec_folder.joinpath(f'{res.name}_normalized_usages_clustermap.png'))
            if show:
                un_sns.fig.show()
            plt.close()

    def plot_decomposition_comparisons(self, show: bool = False,
                                       save: bool = True, max_cell_loss: float = 2000):
        """
        Plot comparisons of pfnmf and de-novo decompositions with similar ranks

        """
        if self.stage in (Stage.INITIALIZED, Stage.PREPARED):
            raise RuntimeError('Must decompose comparing decompositions')

        comp_folder = _utils.set_dir(self.results_dir.joinpath('comparisons'))

        for dn_res, pf_res in zip(self.denovo_results, [self.fnmf_result, *self.pfnmf_results]):
            title = f"{self.b_sname} normalized usages of de-novo " \
                    f"GEPs and {self.a_sname} GEPs + {pf_res.rank - self.rank_a} novel GEPs"

            joint_usages = np.concatenate([dn_res.norm_usages, pf_res.norm_usages], axis=1)
            joint_labels = dn_res.prog_labels_2l + pf_res.prog_labels_2l
            joint_colors = ['#2ca02c'] * dn_res.rank + ['#d62728'] * pf_res.rank

            un_sns = _utils.plot_usages_norm_clustermaps(
                self.adata_b, normalized_usages=joint_usages, columns=joint_labels,
                title=title, show=False, sns_clustermap_params={'col_colors': joint_colors})

            if save:
                un_sns.savefig(comp_folder.joinpath(f'{dn_res.name}_vs_{pf_res.name}_normalized_usages_clustermap.png'))
            if show:
                un_sns.fig.show()
            plt.close()

            plt.plot([0, max_cell_loss], [0, max_cell_loss], 'r-')
            plt.scatter(dn_res.loss_per_cell, pf_res.loss_per_cell, s=2)
            plt.xlabel(f'{dn_res.name} loss per cell')
            plt.ylabel(f'{pf_res.name} loss per cell')
            plt.title('Comparison of loss per cell using two decompositions')
            if save:
                plt.savefig(comp_folder.joinpath(f'{dn_res.name}_vs_{pf_res.name}_loss_per_cell.png'))
            if show:
                plt.show()
            plt.close()

    def run_gsea(self, n_top_genes: int = 1000, gprofiler_kwargs: Dict[str, Any] = None):
        """
        Run gene set enrichment analysis on all the decomposed GEPs
        """
        if self.stage in (Stage.INITIALIZED, Stage.PREPARED):
            raise RuntimeError('Must decompose before plotting error per cell histograms')

        if gprofiler_kwargs is None:
            gprofiler_kwargs = {}
        gp = _utils.MyGProfiler(**gprofiler_kwargs)

        # GSEA on A decomposition GEPs
        geneID_gt_2 = sc.pp.filter_genes(self.adata_a.X, min_cells=2)[0]
        background_geneSymbol = self.adata_a.var.loc[geneID_gt_2, 'geneSymbol'].to_list()

        self._run_gsea_on_nmf_result(self.a_result, background_geneSymbol, gp, self.adata_b.var, n_top_genes)

        # GSEA onB decompositions GEPs
        geneID_gt_2 = sc.pp.filter_genes(self.adata_b.X, min_cells=2)[0]
        background_geneSymbol = self.adata_b.var.loc[geneID_gt_2, 'geneSymbol'].to_list()

        for res in self._all_results:
            self._run_gsea_on_nmf_result(res, background_geneSymbol, gp, self.adata_b.var, n_top_genes)

    def _run_gsea_on_nmf_result(
            self, res: NMFResultBase, background_geneSymbol: List[str],
            gp: _utils.MyGProfiler, adata_var: pd.DataFrame, n_top_genes: int = 1000):
        if self.verbosity > 1:
            print(f'Running gene set enrichment analysis on {res.name} decomposition GEPs')

        gsea_dir = _utils.set_dir(self.results_dir.joinpath('programs_gsea'))

        res.gsea_results = [None] * res.rank

        for index in range(res.rank):
            ordered_genesID = res.gene_coefs.nlargest(
                columns=[res.prog_names[index]], n=n_top_genes).index
            ordered_genesSymbol = adata_var.loc[ordered_genesID, 'geneSymbol'].to_list()

            res.gsea_results[index] = gp.profile(
                ordered_genesSymbol, ordered=True, background=background_geneSymbol)

            res.gsea_results[index].to_csv(
                gsea_dir.joinpath(f"{res.prog_names[index]}.csv"))

    def calculate_fingerprints(self, truncation_level: int = 1000):
        """
        # ToDo: plot usages heatmaps (currently using csv to excel)

        """
        corr_a_vs_denovo = self._calculate_gene_coefs_tsc_a_denovo(truncation_level)
        corr_denovo_vs_pfnmf = self._calculate_corr_denovo_pfnmf(truncation_level)
        corr_a_vs_pfnmf = self._calculate_gene_coefs_tsc_a_pfnmf(truncation_level)

        for i in range(self.rank_a):
            df = corr_a_vs_denovo[i: i + 1].copy().reset_index(drop=True)
            for j in range(self.max_added_rank + 1):
                df.loc[j + 1] = corr_denovo_vs_pfnmf[j].loc[i]
            df.insert(1, 'A_coef_tsc', [np.nan, *corr_a_vs_pfnmf[:, i]])
            # df.loc[-1] = corr_a_vs_denovo.loc[i]

            self.fingerprints[i] = df.reset_index(drop=True)

        # save to file:
        csv_path = self.results_dir.joinpath(f"fingerprints_{self.a_sname}_{self.b_sname}.csv")
        separator = pd.DataFrame({'Separator': ['']})

        self.fingerprints[0].to_csv(csv_path, index=False)
        for i in range(1, self.rank_a):
            # separating line between the tables
            separator.to_csv(csv_path, mode='a', index=False, header=False)
            self.fingerprints[i].to_csv(csv_path, mode='a', index=False)

    def _calculate_gene_coefs_tsc_a_denovo(self, truncation_level: int = 1000):
        """
        Calculate the TSC between the gene coefs of timepoint A and de-novo coefs
        """
        dn_res = self.denovo_results[0]

        df = pd.DataFrame(columns=['pfnmf_gep', 'dn_coef_gep', 'dn_coef_tsc',
                                   'dn_usage_gep', 'dn_usage_corr']
                          ).astype({'dn_coef_tsc': float, 'dn_usage_corr': float})

        df.pfnmf_gep = self.a_result.prog_labels_1l

        tsc = _utils.truncated_spearmans_correlation(pd.concat(
            [self.a_result.gene_coefs, dn_res.gene_coefs], axis=1),
            truncation_level=truncation_level, rowvar=False)[:self.rank_a, self.rank_a:]

        argmax_geps = np.argmax(tsc, axis=1)
        df.dn_coef_gep = [dn_res.prog_labels_1l[i] for i in argmax_geps]
        df.dn_coef_tsc = tsc[np.arange(self.rank_a), argmax_geps]

        return df

    def _calculate_gene_coefs_tsc_a_pfnmf(self, truncation_level: int = 1000):
        """
        Calculate the TSC between the gene coefs of timepoint A and f/pfNMF coefs
        """
        output = np.zeros((self.max_added_rank + 1, self.rank_a))
        for i in range(self.rank_a):
            data = pd.concat([res.gene_coefs[res.prog_names[i]] for res in [
                self.a_result, self.fnmf_result, *self.pfnmf_results]], axis=1)
            output[:, i] = _utils.truncated_spearmans_correlation(
                data, truncation_level, rowvar=False)[1:, 0]

        return output

    def _calculate_corr_denovo_pfnmf(self, truncation_level: int = 1000):
        """
        Calculate the comparisons between de-novo and f/pfNMF GEPs.
        Performs GEP usages correlation  and TSC on the gene coefficients

        for each rank of decomposition outputs a dataframe with the columns:
        pfnmf_gep, dn_coef_gep, dn_coef_tsc, dn_usage_gep, dn_usage_corr
        """
        output = []
        for dn_res, pf_res in zip(self.denovo_results, [self.fnmf_result, *self.pfnmf_results]):
            df = pd.DataFrame(columns=['pfnmf_gep', 'dn_coef_gep', 'dn_coef_tsc',
                                       'dn_usage_gep', 'dn_usage_corr']
                              ).astype({'dn_coef_tsc': float, 'dn_usage_corr': float})
            df.pfnmf_gep = pf_res.prog_labels_1l[:self.rank_a]

            # gene coefficients TSC:
            tsc = _utils.truncated_spearmans_correlation(pd.concat(
                [pf_res.gene_coefs.iloc[:, :self.rank_a], dn_res.gene_coefs], axis=1),
                truncation_level=truncation_level, rowvar=False)[:self.rank_a, self.rank_a:]

            argmax_geps = np.argmax(tsc, axis=1)
            df.dn_coef_gep = [dn_res.prog_labels_1l[i] for i in argmax_geps]
            df.dn_coef_tsc = tsc[np.arange(self.rank_a), argmax_geps]

            # usage correlation:
            usage_corr = np.corrcoef(np.concatenate(
                [pf_res.norm_usages[:, :self.rank_a], dn_res.norm_usages], axis=1),
                rowvar=False)[:self.rank_a, self.rank_a:]

            argmax_geps = np.argmax(usage_corr, axis=1)
            df.dn_usage_gep = [dn_res.prog_labels_1l[i] for i in argmax_geps]
            df.dn_usage_corr = usage_corr[np.arange(self.rank_a), argmax_geps]

            output.append(df)
        return output

    def plot_fingerprint(self):
        """
        Input the correlation results to the fingerprint surrogate table
        """
        pass

    @staticmethod
    def _run_nmf(x, nmf_kwargs, name, tens: 'torch.Tensor' = None, verbose: int = 0) -> NMFResult:
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

