#!/usr/bin/env python
"""
Consensus non-negative matrix factorization (cNMF)
Adapted from (Kotliar, et al. 2019), expanding on v1.3 sourced at
https://github.com/dylkot/cNMF version 1.3
"""

import datetime
import errno
import itertools
import os
import uuid
import warnings
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import gridspec

from fastcluster import linkage
from scipy.cluster.hierarchy import leaves_list
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import non_negative_factorization
from sklearn.decomposition import _nmf as sknmf
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import euclidean_distances


def save_df_to_tsv(df: pd.DataFrame, filename):
    '''Given a dataframe, save in a tsv file the data with index and columns'''
    df.to_csv(filename, sep='\t')


def save_df_to_npz(df: pd.DataFrame, filename):
    '''Given a dataframe, save in npz format the data, index and columns'''
    np.savez_compressed(filename, data=df.values, index=df.index.values,
                        columns=df.columns.values)


def load_df_from_npz(filename):
    '''Load a data frame saved as npz with (data, index, columns)'''
    with np.load(filename, allow_pickle=True) as f:
        df = pd.DataFrame(**f)
    return df


def save_data_to_npz(data, filename, compress=False):
    if not compress:
        np.savez(filename, data=data)
    np.savez_compressed(filename, data=data)
    

def load_data_from_npz(filename):
    with np.load(filename, allow_pickle=True) as f:
        data = f['data']
    if data.ndim == 0:   # sparse matrices are saved as arry of object
        return data.reshape(-1)[0]
    return data
    

def check_dir_exists(path):
    """ Checks if directory already exists, and creates it if it doesn't """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def worker_filter(iterable, worker_index, total_workers):
    ''' Return a generator for the subset of iterable fitting worker index.
    i.e. (index - worker_index) % total_workers == 0'''
    return (p for i, p in enumerate(iterable) if 
            (i - worker_index) % total_workers == 0)


def fast_ols_all_cols(X, Y):
    '''fast, orthogonal least squares algorithm. Y=Xb'''
    pinv = np.linalg.pinv(X)
    return np.dot(pinv, Y)


def fast_ols_all_cols_df(X: pd.DataFrame, Y: pd.DataFrame):
    '''fast, orthogonal least squares algorithm. Y=Xb
    returning dataframe with rows from X's columns and coulmns from Y'''
    beta = fast_ols_all_cols(X, Y)
    beta = pd.DataFrame(beta, index=X.columns, columns=Y.columns)
    return beta


def var_sparse_matrix(X):
    '''calculate the variance of a sparse matrix'''
    mean = np.array(X.mean(axis=0)).reshape(-1)
    Xcopy = X.copy()
    Xcopy.data **= 2
    var = np.array(Xcopy.mean(axis=0)).reshape(-1) - (mean**2)
    return var


def _nmf_torch_translate_kwargs(X, nmf_kwargs):
    '''
    Translating keyword arguments to match torch-NMF requirements,
    taking in all possible sklearn.decomposition non_negative_factorization,
    returning torchnmf_NMF_kwargs & torchnmf_fit_kwargs dictionaries
    '''
    if 'torch' not in dir():
        import torch
    
    torchnmf_fit_kwargs = {}
    torchnmf_NMF_kwargs = {}
    
    orig_w = nmf_kwargs.get('W')
    orig_h = nmf_kwargs.get('H')
    orig_rank = nmf_kwargs.get('n_components')

    # Getting W, H.
    if (orig_w is None) and (orig_h is None):
        assert orig_rank, "NMF keywords must contain 'n_components' when W and H are not available"
        
        W, H = sknmf._initialize_nmf(X, orig_rank, init=nmf_kwargs.get('init'),
                                     random_state=nmf_kwargs.get('random_state'))
    
    elif (orig_w is not None) and (orig_h is not None):
        W = orig_w
        H = orig_h
    
    elif orig_w is None:  # We were given H
        H = orig_h
        
        assert (orig_rank is None) or (H.shape[0] == orig_rank), \
            f"'n_components' ({orig_rank}) is different from the first dimension of H ({H.shape[0]})"
        
        W, _ = sknmf._initialize_nmf(X, H.shape[0], init=nmf_kwargs.get('init'),
                                     random_state=nmf_kwargs.get('random_state'))
        
    else:   # we were given W
        W = orig_w
        
        assert (orig_rank is None) or (W.shape[1] == orig_rank), \
            f"'n_components' ({orig_rank}) is different from the second dimension of W ({W.shape[1]})"
        
        _, H = sknmf._initialize_nmf(X, W.shape[1], init=nmf_kwargs.get('init'),
                                     random_state=nmf_kwargs.get('random_state'))
    
    # torchnmf replaces the W and H, for that module V = H @ W.T
    torchnmf_NMF_kwargs['W'] = torch.tensor(H.transpose())
    torchnmf_NMF_kwargs['H'] = torch.tensor(W)
    
    torchnmf_NMF_kwargs['trainable_W'] = nmf_kwargs.get('update_H', True)

    # loss function:
    if nmf_kwargs.get('beta_loss') is not None:
        torchnmf_fit_kwargs['beta'] = sknmf._beta_loss_to_float(
            nmf_kwargs.get('beta_loss'))

    # tolerance:
    if nmf_kwargs.get('tol') is not None:
        torchnmf_fit_kwargs['tol'] = nmf_kwargs.get('tol')

    # maximal iterations:
    if nmf_kwargs.get('max_iter') is not None:
        torchnmf_fit_kwargs['max_iter'] = nmf_kwargs.get('max_iter')

    # verbrosety mode 
    torchnmf_fit_kwargs['verbose'] = nmf_kwargs.get('verbose', False)

    # working the arguments, regularization:
    if nmf_kwargs.get('alpha_W') is not None and nmf_kwargs['alpha_W'] > 0:
        raise NotImplementedError 

    return torchnmf_NMF_kwargs, torchnmf_fit_kwargs


def torch_to_np(tensor):
    return tensor.detach().cpu().numpy() 


def nmf_torch(X, nmf_kwargs, tens=None, verbose: bool=False, device=None):
    """
    Wrapper for torchnmf (GPU), assimilating the keywords to sklearn NMF.

    Parameters
    ----------
    X : np.ndarray or sparse.csr.csr_matrix
        Data to be factorized (used only for initiation).

    nmf_kwargs : dict
        Arguments in the format of `sklearn.decomposition.non_negative_factorization`.
        It must include 'n_components' with a value greater than or equal to 2.

    tens : torch.Tensor, optional
        Normalized counts to be factorized, already on the GPU device.
        If None, the device must be provided.

    verbose : bool, optional
        Whether to print the torchnmf progress.

    device : str or torch.device, optional
        Device to run the NMF on. If None, the `tens.device` attribute will be used.

    Returns
    -------
    W : ndarray of shape (n_samples, n_components)
        Solution to the non-negative factorization.

    H : ndarray of shape (n_components, n_features)
        Solution to the non-negative factorization.

    n_iter : int
        Actual number of iterations.
    """
    if 'torch' not in dir():
        import torch
    if 'torchnmf' not in dir():
        import torchnmf

    no_tensor = tens is None
    if no_tensor:
        tens = torch.tensor(X).to(device)
    device_type = tens.device.type
    
    NMF_args, NMF_fit_kwargs = _nmf_torch_translate_kwargs(X, nmf_kwargs)

    # torchnmf replaces the W and H, meaning V = H @ W.T
    ptm = torchnmf.nmf.NMF(**NMF_args)

    ptm.to(tens.device)

    n_iter = ptm.fit(tens, **NMF_fit_kwargs)
    
    if verbose:
        beta = NMF_fit_kwargs.get('beta') or 1
        loss = torchnmf.metrics.beta_div(ptm.forward(), tens, beta=beta)
        # print(f' ')
        print(f'beta {beta} loss = {loss}, # iterations was {n_iter}')
    
    W, H = torch_to_np(ptm.H), torch_to_np(ptm.W).transpose()
    
    # freeing gpu memory
    ptm.to('cpu')   
    
    del ptm

    if no_tensor:
        del tens

    if device_type == "cuda":
        torch.cuda.empty_cache()
    
    return W, H, n_iter


class cNMF():
    def __init__(self, output_dir=".", name=None):
        """
        Parameters
        ----------

        output_dir : path, optional (default=".")
            Output directory for analysis files.

        name : string, optional (default=None)
            A name for this analysis. Will be prefixed to all output files.
            If set to None, will be generated from date and random string.
        """
        self.output_dir = output_dir
        if name is None:
            now = datetime.datetime.now()
            rand_hash = uuid.uuid4().hex[:6]
            name = '%s_%s' % (now.strftime("%Y_%m_%d"), rand_hash)
        self.name = name
        self.paths = None
        self._initialize_dirs()

    def _initialize_dirs(self):
        '''Create names for different analysis files'''
        if self.paths is not None:
            return
    
        # Check that output directory exists, create it if needed.
        check_dir_exists(self.output_dir)
        
        run_dir = os.path.join(self.output_dir, self.name)
        check_dir_exists(run_dir)
        
        tmp_dir = os.path.join(run_dir, 'cnmf_tmp')
        check_dir_exists(tmp_dir)

        self.paths = {
            'data': os.path.join(
                tmp_dir, self.name + '.data.npz'),
            'nmf_replicate_parameters': os.path.join(
                tmp_dir, self.name + '.nmf_params.df.npz'),
            'nmf_run_parameters': os.path.join(
                tmp_dir, self.name + '.nmf_idvrun_params.yaml'),
            'nmf_genes_list': os.path.join(
                run_dir, self.name + '.overdispersed_genes.txt'),


            'iter_spectra': os.path.join(
                tmp_dir, self.name + '.spectra.k_%d.iter_%d.npz'),
            'iter_usages': os.path.join(
                tmp_dir, self.name + '.usages.k_%d.iter_%d.df.npz'),
            'merged_spectra': os.path.join(
                tmp_dir, self.name + '.spectra.k_%d.merged.df.npz'),

            'local_density_cache': os.path.join(
                tmp_dir,
                self.name + '.local_density_cache.k_%d_%d.merged.df.npz'),
            'consensus_spectra': os.path.join(
                tmp_dir,
                self.name + '.spectra.k_%d(%d).dt_%s.consensus.df.npz'),
            'consensus_spectra__txt': os.path.join(
                run_dir, self.name + '.spectra.k_%d(%d).dt_%s.consensus.txt'),
            'consensus_usages': os.path.join(
                tmp_dir,
                self.name + '.usages.k_%d(%d).dt_%s.consensus.df.npz'),
            'consensus_usages__txt': os.path.join(
                run_dir, self.name + '.usages.k_%d(%d).dt_%s.consensus.txt'),
            'consensus_stats': os.path.join(
                tmp_dir, self.name + '.stats.k_%d(%d).dt_%s.df.npz'),

            'clustering_plot': os.path.join(
                run_dir, self.name + '.clustering.k_%d(%d).dt_%s.png'),

            'k_selection_plot': os.path.join(
                run_dir, self.name + '.k_selection.png'),
            'k_selection_plot_dt': os.path.join(
                run_dir, self.name + '.k_selection.dt_%s.png'),
            'k_selection_stats': os.path.join(
                run_dir, self.name + '.k_selection_stats.df.npz'),
        }
        
    def _initialize_gpu(self, device='cuda'):
        if hasattr(self, 'device'):  # if environment was set for gpu usage
            return
        
        if not hasattr(self, 'X'):
            self.X = load_data_from_npz(self.paths['data'])
        
        # runtime imports
        global torch, torchnmf
        import torch
        assert torch.cuda.is_available()
        import torchnmf
        
        self.device = device

    def prepare(self, X: np.ndarray, components, n_iter=100,
                seed=None, new_nmf_kwargs: Dict={}):
        """
        Copy the input data, and setup files and folders for distributing 
        jobs over workers.


        Parameters
        ----------
        X : numpy ndarray
            Path to input data. Make sure

        components : list or numpy array
            Values of K to run NMF for
            
        n_iter : integer, optional (defailt=100)
            Number of iterations for factorization. If several ``k`` are
            specified, this many iterations will be run for each ``k``.

        seed : int or None, optional (default=None)
            Seed for sklearn random state.
            
        new_nmf_kwargs : Dictionary or None, optional (default={})
            Dictionary with keyword arguments to the NMF function.
            the defaults are:
            beta_loss='kullback-leibler',
            alpha_W=0.0,
            alpha_H='same',
            l1_ratio=0.0,
            solver='mu',
            tol=5e-5,
            max_iter=500

        """
        
        # save aside input adata.
        save_data_to_npz(X, self.paths['data'])

        # Check for any cells that have 0 counts of the overdispersed genes
        zerocells = X.sum(axis=1) == 0
        if zerocells.sum() > 0:
            print(f'Warning: {zerocells.sum()} cells have zero counts'
                  ' of overdispersed genes. E.g. %s'
                  'Consensus step may not run when this is the case')
        
        # setting parameters
        (replicate_params, run_params) = self.get_nmf_iter_params(
            ks=components, n_iter=n_iter,
            random_state_seed=seed, new_nmf_kwargs=new_nmf_kwargs)
        
        self.save_nmf_iter_params(replicate_params, run_params) 
        
    def get_nmf_iter_params(self, ks, n_iter=100, random_state_seed=None,
                            new_nmf_kwargs: Dict={}):
        """
        Create a DataFrame with parameters for NMF iterations.

        Parameters
        ----------
        ks : integer, or list-like.
            Number of topics (components) for factorization.
            Several values can be specified at the same time,
            which will be run independently.

        n_iter : integer, optional (defailt=100)
            Number of iterations for factorization. If several ``k`` are 
            specified, this many iterations will be run for each ``k``.

        random_state_seed : int or None, optional (default=None)
            Seed for sklearn random state.
            
        new_nmf_kwargs : dictionary, optional
            arguments for the sklearn nmf functions

        """
        if type(ks) is int:
            ks = [ks]

        # Remove any repeated k values, and order.
        k_list = sorted(set(list(ks)))

        n_runs = len(ks) * n_iter

        np.random.seed(seed=random_state_seed)
        nmf_seeds = np.random.randint(low=1, high=(2**32) - 1, size=n_runs)

        replicate_params = []
        for i, (k, r) in enumerate(itertools.product(k_list, range(n_iter))):
            replicate_params.append([k, r, nmf_seeds[i]])
        replicate_params = pd.DataFrame(
            replicate_params, columns=['n_components', 'iter', 'nmf_seed'])

        # TBD: consider adding regularization to NMF
        _nmf_kwargs = dict(
            beta_loss='kullback-leibler',
            alpha_W=0.0,
            alpha_H='same',
            l1_ratio=0.0,
            solver='mu',
            tol=5e-5,
            max_iter=500,
            init='random'
        )

        _nmf_kwargs.update(new_nmf_kwargs)

        return replicate_params, _nmf_kwargs

    def save_nmf_iter_params(self, replicate_params, run_params):
        self._initialize_dirs()
        save_df_to_npz(replicate_params,
                       self.paths['nmf_replicate_parameters'])
        with open(self.paths['nmf_run_parameters'], 'w') as F:
            yaml.dump(run_params, F)

    def _nmf(self, X, nmf_kwargs):
        """
        Parameters
        ----------
        X : np.ndarray or sparse.csr.csr_matrix,
            Normalized counts to be factorized.

        nmf_kwargs : dict,
            Arguments to be passed to ``non_negative_factorization``

        """
        with warnings.catch_warnings():  # supress convergence warning
            warnings.simplefilter(action='ignore', category=ConvergenceWarning)
            (usages, spectra, niter) = non_negative_factorization(
                X, **nmf_kwargs)

        return spectra, usages

    def factorize(self, worker_i=0, total_workers=1, gpu=False, device='cuda',
                  verbose=False):
        """
        Iteratively run NMF with prespecified parameters.

        Use the `worker_i` and `total_workers` parameters for parallelization.

        Generic kwargs for NMF are loaded from self.paths['nmf_run_parameters']
        
        random_state, n_components are both set by the prespecified
            self.paths['nmf_replicate_parameters'].


        Parameters
        ----------
        worker_i : int,
            The index of this worker

        total_workers : int,
            The total amount of workers set to factorize the matrix

        torch : bool, default False
            Wether to use torchNMF, if so, use device
        """
        
        run_params = load_df_from_npz(self.paths['nmf_replicate_parameters'])
        
        if not hasattr(self, 'X'):
            self.X = load_data_from_npz(self.paths['data'])
            
        _nmf_kwargs = yaml.load(
            open(self.paths['nmf_run_parameters']), Loader=yaml.FullLoader)

        jobs_for_this_worker = worker_filter(
            range(len(run_params)), worker_i, total_workers)
        
        if gpu:
            if not hasattr(self, 'device'):
                self._initialize_gpu(device)
            tens = torch.tensor(self.X).to(self.device)
            
        for idx in jobs_for_this_worker:
            p = run_params.iloc[idx, :]
            
            if verbose:
                print('[Worker %d]. Starting task %d.' % (worker_i, idx))
            
            _nmf_kwargs['random_state'] = p['nmf_seed']
            _nmf_kwargs['n_components'] = p['n_components']
            
            if gpu:
                usages, spectra, _ = nmf_torch(
                    self.X, _nmf_kwargs, tens, verbose)
            else:
                (spectra, usages) = self._nmf(self.X, _nmf_kwargs)
            
            save_data_to_npz(spectra, self.paths['iter_spectra'] % (
                p['n_components'], p['iter']))
            
            del spectra, usages
        
        if gpu:
            del tens
            torch.cuda.empty_cache()
        
    def combine(self, components=None):
        '''wrapper for combining results per different ``ks``'''
        self.run_params = load_df_from_npz(
            self.paths['nmf_replicate_parameters'])

        if type(components) is int:
            ks = [components]
        elif components is None:
            ks = sorted(set(self.run_params.n_components))
        else:
            ks = components

        for k in ks:
            self.combine_nmf(k)
    
    def combine_nmf(self, k, remove_individual_iterations=False):
        '''merging of nmf spectras for specific k'''
        print('Combining factorizations for k=%d.' % k)

        combined_spectra = None
        n_iter = sum(self.run_params.n_components == k)

        run_params_subset = self.run_params[
            self.run_params.n_components == k].sort_values('iter')
        spectra_labels = []

        for i, p in run_params_subset.iterrows():

            spectra = load_data_from_npz(
                self.paths['iter_spectra'] % (p['n_components'], p['iter']))
            if combined_spectra is None:
                combined_spectra = np.zeros(
                    (n_iter, k, spectra.shape[1]), dtype=spectra.dtype)
            combined_spectra[p['iter'], :, :] = spectra

            for t in range(k):
                spectra_labels.append('iter%d_topic%d' % (p['iter'], t + 1))

        combined_spectra = combined_spectra.reshape(
            -1, combined_spectra.shape[-1])
        combined_spectra = pd.DataFrame(
            combined_spectra, index=spectra_labels)

        save_df_to_npz(combined_spectra, self.paths['merged_spectra'] % k)
        return combined_spectra
    
    @staticmethod
    def convert_dt_to_str(density_threshold):
        '''Return a the string format of the density threshold float'''
        return str(density_threshold).replace('.', '_')
    
    def get_consensus_usages_spectra(
            self, k_source, k_clusters=None, density_threshold=0.5):
        '''
        Given the ``k`` and threshold, return the usages and spectra matrices
        '''
        if k_clusters is None:
            k_clusters = k_source
        
        density_threshold_str = self.convert_dt_to_str(density_threshold)
        
        usages = load_data_from_npz(self.paths['consensus_usages'] % (
            k_source, k_clusters, density_threshold_str))
        
        spectra = load_data_from_npz(self.paths['consensus_spectra'] % (
            k_source, k_clusters, density_threshold_str))
        
        return usages, spectra

    def consensus(self, k_source, k_clusters=None, density_threshold=0.5,
                  local_neighborhood_size=0.30, show_clustering=True,
                  close_clustergram_fig=False, consensus_method='median',
                  skip_density_and_return_after_stats=False,
                  nmf_refitting_iters: int=None, gpu=False, device='cuda',
                  verbose=False):
        '''
        Create the consensus usages and spectra
        

        Parameters
        ----------
        k_source : int,
            The ``k`` used in the selected factorization

        k_clusters : int, default None
            The number of clusters of spectras to use.
            If None uses k_clusters = k_source
            
        TODO - write all the parameters of the function
        
        '''
        if k_clusters is None:
            k_clusters = k_source
        merged_spectra = load_df_from_npz(
            self.paths['merged_spectra'] % k_source)
        
        if not hasattr(self, 'X'):
            self.X = load_data_from_npz(self.paths['data'])

        density_threshold_str = self.convert_dt_to_str(density_threshold)

        n_neighbors = int(local_neighborhood_size * (
            merged_spectra.shape[0] / k_source))

        # Rescale topics such to length of 1.
        l2_spectra = merged_spectra / np.linalg.norm(
            merged_spectra, 2, axis=1, keepdims=True)
        
        # remove entries that are close to zero and rescale
        EPSILON = np.finfo(l2_spectra.values.dtype).smallest_normal**0.5
        l2_spectra[l2_spectra < EPSILON] = 0
        l2_spectra = l2_spectra / np.linalg.norm(
            l2_spectra, 2, axis=1, keepdims=True)
        
        if show_clustering or not skip_density_and_return_after_stats:
            # Compute the local density matrix (if not previously cached)
            topics_dist = None  # The pairwise distances of spectra
            td_fn = self.paths['local_density_cache'] % (k_source, n_neighbors)
            if os.path.isfile(td_fn):
                local_density = load_df_from_npz(td_fn)
            else:
                #   first find the full distance matrix
                topics_dist = euclidean_distances(l2_spectra.values)
                #   partition based on the first n neighbors
                partitioning_order = np.argpartition(
                    topics_dist, n_neighbors + 1)[:, :n_neighbors + 1]
                #   find the mean over those n_neighbors 
                #   (excluding self, which has a distance of 0)
                distance_to_nearest_neighbors = topics_dist[np.arange(
                    topics_dist.shape[0])[:, None], partitioning_order]
                local_density = pd.DataFrame(
                    distance_to_nearest_neighbors.sum(1) / (n_neighbors),
                    columns=['local_density'], index=l2_spectra.index)
                save_df_to_npz(local_density, td_fn)
                del partitioning_order
                del distance_to_nearest_neighbors

            density_filter = local_density.iloc[:, 0] < density_threshold
            l2_spectra = l2_spectra.loc[density_filter, :]

        kmeans_model = KMeans(n_clusters=k_clusters, random_state=1,
                              n_init=max(10, k_clusters))
        kmeans_model.fit(l2_spectra)
        # yotamcon 2022-08-29 added reordering of clusters by balancing 
        #   both the cluster's average siluoette and the log of the size
        kmeans_clusters = pd.DataFrame(np.vstack(
            [(kmeans_model.labels_ + 1),
             silhouette_samples(l2_spectra, kmeans_model.labels_)]).T,
            index=l2_spectra.index, columns=['label', 'silhouette'])
        
        kmeans_clusters.label = kmeans_clusters.label.map(dict(zip(
            (kmeans_clusters.groupby('label').mean() * 
             np.log(kmeans_clusters.groupby('label').count())).sort_values(
                'silhouette', ascending=False).index,
            range(1, k_clusters + 1))))
        
        # Find representative usage for each gene across cluster
        if consensus_method == 'median':
            consensus_spectra = l2_spectra.groupby(kmeans_clusters.label).median()
        elif consensus_method == 'mean':
            consensus_spectra = l2_spectra.groupby(kmeans_clusters.label).mean()
        else:
            raise ValueError('consensus_method must be "median" or "mean"')

        consensus_spectra[consensus_spectra < EPSILON] = 0

        # Normalize median spectra to probability distributions.
        row_norm = np.linalg.norm(consensus_spectra, ord=1, axis=1)
        consensus_spectra = consensus_spectra / row_norm[:, None]

        # Compute the silhouette score
        stability = silhouette_score(
            l2_spectra.values, kmeans_clusters.label, metric='euclidean')

        # Obtain reconstructed count matrix by re-fitting usage,
        # and computing dot product: usage.dot(spectra)
        refit_nmf_kwargs = yaml.load(
            open(self.paths['nmf_run_parameters']), Loader=yaml.FullLoader)
        # yotamcon 2022-08-29 added optional nmf rounds (updating H), and 
        #  changed the error calculation to follow the cnmf running parameters.
        
        if verbose:
            print('Updating W based on consensus spectra')
        
        refit_nmf_kwargs.update(dict(
            n_components=k_clusters,
            H=consensus_spectra.values,
            update_H=False))
        
        if gpu:
            if not hasattr(self, 'device'):
                self._initialize_gpu(device)
            tens = torch.tensor(self.X).to(self.device)
            rf_usages, rf_spec, _ = nmf_torch(
                self.X, refit_nmf_kwargs, tens, verbose)
            del tens
            torch.cuda.empty_cache()
        else:
            _, rf_usages = self._nmf(self.X, nmf_kwargs=refit_nmf_kwargs)

        if nmf_refitting_iters is not None:            
            if verbose:
                print('Refitting W, H based on consensus')
                
            refit_nmf_kwargs.update(dict(
                H=consensus_spectra.values.copy(),
                W=rf_usages.copy(),
                init='custom',
                max_iter=nmf_refitting_iters,
                update_H=True
            ))
            
            if gpu:
                tens = torch.tensor(self.X).to(self.device)
                rf_usages, rf_spec, _ = nmf_torch(
                    self.X, refit_nmf_kwargs, tens, verbose)
                del tens
                torch.cuda.empty_cache()
            else:
                rf_spec, rf_usages = self._nmf(
                    self.X, nmf_kwargs=refit_nmf_kwargs)
            
            # L1 normalize the spectra
            rownorm = np.linalg.norm(rf_spec, ord=1, axis=1)
            consensus_spectra[:] = rf_spec / (rownorm[:, None])
            rf_usages = rf_usages * (rownorm[None, :])
        
        rf_usages = pd.DataFrame(
            rf_usages, columns=consensus_spectra.index)
        
        # rf_pred_norm_counts = rf_usages.dot(consensus_spectra)

        # Compute prediction error as the loss
        prediction_error = sknmf._beta_divergence(
            self.X, rf_usages, consensus_spectra,
            sknmf._beta_loss_to_float(refit_nmf_kwargs['beta_loss']))
        
        consensus_stats = pd.DataFrame(
            [k_source, k_clusters, density_threshold, stability,
             prediction_error], columns=['stats'],
            index=['k_source', 'k_clusters', 'local_density_threshold',
                   'stability', 'prediction_error'])

        if not skip_density_and_return_after_stats:
            save_df_to_npz(consensus_spectra, self.paths['consensus_spectra'] % (
                k_source, k_clusters, density_threshold_str))
            save_df_to_npz(rf_usages, self.paths['consensus_usages'] % (
                k_source, k_clusters, density_threshold_str))
            save_df_to_npz(consensus_stats, self.paths['consensus_stats'] % (
                k_source, k_clusters, density_threshold_str))
            save_df_to_tsv(consensus_spectra, self.paths['consensus_spectra__txt']
                           % (k_source, k_clusters, density_threshold_str))
            save_df_to_tsv(rf_usages, self.paths['consensus_usages__txt'] % (
                k_source, k_clusters, density_threshold_str))
        
        # TODO - split into a new function
        if show_clustering:           
            if topics_dist is None:
                # (l2_spectra was already filtered using the density filter)
                topics_dist = euclidean_distances(l2_spectra.values)
            else:
                # (but the previously computed topics_dist was not!)
                topics_dist = topics_dist[density_filter.values,
                                          :][:, density_filter.values]
    
            spectra_order = []
            for cl in range(1, k_clusters + 1):
                cl_filter = kmeans_clusters.label == cl
    
                if cl_filter.sum() > 1:
                    cl_dist = squareform(
                        topics_dist[cl_filter, :][:, cl_filter], checks=False)
                    # Rarely get floating point arithmetic issues
                    cl_dist[cl_dist < 0] = 0
                    cl_link = linkage(cl_dist, 'average')
                    cl_leaves_order = leaves_list(cl_link)
    
                    spectra_order += list(np.where(cl_filter)
                                          [0][cl_leaves_order])
                else:
                    # Corner case where a component only has one element
                    spectra_order += list(np.where(cl_filter)[0])
    
            width_ratios = [0.5, 9, 0.5, 4, 1]
            height_ratios = [0.5, 9]
            fig = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
            gs = gridspec.GridSpec(len(height_ratios), len(width_ratios), fig,
                                   0.01, 0.01, 0.98, 0.98,
                                   height_ratios=height_ratios,
                                   width_ratios=width_ratios,
                                   wspace=0, hspace=0)
    
            dist_ax = fig.add_subplot(
                gs[1, 1], xscale='linear', yscale='linear', xticks=[],
                yticks=[], xlabel='', ylabel='', frameon=True)
    
            D = topics_dist[spectra_order, :][:, spectra_order]
            dist_im = dist_ax.imshow(D, interpolation='none', rasterized=True,
                                     vmax=1.414, aspect='auto', cmap='viridis')
    
            left_ax = fig.add_subplot(gs[1, 0], xscale='linear',
                                      yscale='linear', xticks=[], yticks=[],
                                      xlabel='', ylabel='', frameon=True)
            left_ax.imshow(
                kmeans_clusters.label.values[spectra_order].reshape(-1, 1),
                interpolation='none', cmap='Spectral', aspect='auto',
                rasterized=True)
    
            top_ax = fig.add_subplot(
                gs[0, 1], xscale='linear', yscale='linear', xticks=[],
                yticks=[], xlabel='', ylabel='', frameon=True)
            top_ax.imshow(
                kmeans_clusters.label.values[spectra_order].reshape(1, -1),
                interpolation='none', cmap='Spectral', aspect='auto',
                rasterized=True)
    
            # histogram axis
            hist_gs = gridspec.GridSpecFromSubplotSpec(
                3, 1, subplot_spec=gs[1, 3], wspace=0, hspace=0)
    
            hist_ax = fig.add_subplot(
                hist_gs[0, 0], xscale='linear', yscale='linear', xlabel='',
                ylabel='', frameon=True, title='Local density histogram')
            hist_ax.hist(local_density.values, bins=np.linspace(0, 1.42, 72))
            hist_ax.yaxis.tick_right()

            hist_ax.set_xlim((-0.005, 1.435))
            xlim = hist_ax.get_xlim()
        
            ylim = hist_ax.get_ylim()
            if density_threshold < xlim[1]:
                hist_ax.axvline(density_threshold, linestyle='--', color='k')
                hist_ax.text(density_threshold + 0.02, ylim[1] * 0.95,
                             'filtering\nthreshold\n\n', va='top')
            hist_ax.set_xlim(xlim)
            hist_ax.set_xlabel(
                f'Mean distance to k={n_neighbors} nearest neighbors.\n'
                f'{sum(~density_filter)}/{len(density_filter)} '
                f'({100 * (~density_filter).mean(): .0f}%) spectra above '
                'threshold\nwere removed prior to clustering\n\n'
                f'{refit_nmf_kwargs["beta_loss"]} prediction error '
                f'{prediction_error:.3e}')
            
            # Add colorbar
            cbar_gs = gridspec.GridSpecFromSubplotSpec(
                8, 1, subplot_spec=hist_gs[1, 0], wspace=0, hspace=0)
            cbar_ax = fig.add_subplot(
                cbar_gs[4, 0], xscale='linear', yscale='linear', xlabel='',
                ylabel='', frameon=True, title='Euclidean Distance')
            # vmin = D.min().min()
            # vmax = D.max().max()
            fig.colorbar(dist_im, cax=cbar_ax,
                         # ticks=np.linspace(vmin, vmax, 3),
                         ticks=[0, 0.707, 1.414],
                         orientation='horizontal')
            cbar_ax.set_xticklabels(['$0$', r'$\sqrt{2}/2$', r'$\sqrt{2}$'])
            
            fig.savefig(
                self.paths['clustering_plot'] %
                (k_source, k_clusters, density_threshold_str), dpi=250)
            if close_clustergram_fig:
                pass
                plt.close(fig)
        return consensus_stats

    def k_selection_plot(self, close_fig=False, show_clustering=False, 
                         density_threshold=0.5, local_neighborhood_size=0.30,
                         nmf_refitting_iters=100, gpu=False, device='cuda',
                         verbose=False):
        '''
        Borrowed from Alexandrov Et Al. 2013 Deciphering Mutational Signatures
        publication in Cell Reports
        '''
        run_params = load_df_from_npz(self.paths['nmf_replicate_parameters'])
        stats = []
        for k in sorted(set(run_params.n_components)):
            try:
                stats.append(self.consensus(
                    k, density_threshold=density_threshold,
                    local_neighborhood_size=local_neighborhood_size,
                    skip_density_and_return_after_stats=True,
                    show_clustering=show_clustering,
                    close_clustergram_fig=True,
                    nmf_refitting_iters=nmf_refitting_iters,
                    gpu=gpu, device=device, verbose=verbose).stats)
            except Exception as e:
                print(f'Failed consensus for k={k} due to {type(e)}.\n{e}')
        
        if len(stats) == 0:
            print(f'Failed to create K-selection plot for threshold={density_threshold: .2f}')
            return
        
        stats = pd.DataFrame(stats)
        stats.reset_index(drop=True, inplace=True)

        save_df_to_npz(stats, self.paths['k_selection_stats'])

        fig = plt.figure(figsize=(6, 4))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        ax1.plot(stats.k_source, stats.stability, 'o-', color='b')
        ax1.set_ylabel('Stability', color='b', fontsize=15)
        for tl in ax1.get_yticklabels():
            tl.set_color('b')

        ax2.plot(stats.k_source, stats.prediction_error, 'o-', color='r')
        ax2.set_ylabel('Error', color='r', fontsize=15)
        for tl in ax2.get_yticklabels():
            tl.set_color('r')

        ax1.set_xlabel('Number of Components', fontsize=15)
        ax1.grid('on')
        
        density_str = self.convert_dt_to_str(density_threshold)
        ax1.set_title('k-selection assisting plot\n'
                      f'threshold={density_threshold:.2f},'
                      f' neighbors proportion={local_neighborhood_size}',
                      fontsize=12)
        
        plt.tight_layout()
        fig.savefig(self.paths['k_selection_plot_dt'] % density_str, dpi=250)
        if close_fig:
            plt.close(fig)
        else:
            return fig


def main():
    """
    Example commands:

        output_dir="./cnmf_test/"


        python cnmf.py prepare --output-dir $output_dir \
           --name test --counts ./cnmf_test/test_data.df.npz \
           -k 6 7 8 9 --n-iter 5

        python cnmf.py factorize  --name test --output-dir $output_dir

        THis can be parallelized as such:

        python cnmf.py factorize  --name test --output-dir $output_dir --total-workers 2 --worker-index WORKER_INDEX (where worker_index starts with 0)

        python cnmf.py combine  --name test --output-dir $output_dir

        python cnmf.py consensus  --name test --output-dir $output_dir

    """

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('command', type=str, choices=[
                        'prepare', 'factorize', 'combine', 'consensus', 'k_selection_plot'])
    parser.add_argument(
        '--name', type=str, help='[all] Name for analysis. All output will be placed in [output-dir]/[name]/...', nargs='?', default='cNMF')
    parser.add_argument('--output-dir', type=str,
                        help='[all] Output directory. All output will be placed in [output-dir]/[name]/...', nargs='?', default='.')
    parser.add_argument('-c', '--counts', type=str,
                        help='[prepare] Input (cell x gene) counts matrix as df.npz or tab delimited text file')
    parser.add_argument('-k', '--components', type=int,
                        help='[prepare] Numper of components (k) for matrix factorization. Several can be specified with "-k 8 9 10"', nargs='+')
    parser.add_argument('-n', '--n-iter', type=int,
                        help='[prepare] Numper of factorization replicates', default=100)
    parser.add_argument('--total-workers', type=int,
                        help='[all] Total number of workers to distribute jobs to', default=1)
    parser.add_argument(
        '--seed', type=int, help='[prepare] Seed for pseudorandom number generation', default=None)
    parser.add_argument('--genes-file', type=str,
                        help='[prepare] File containing a list of genes to include, one gene per line. Must match column labels of counts matrix.', default=None)
    parser.add_argument('--numgenes', type=int,
                        help='[prepare] Number of high variance genes to use for matrix factorization.', default=2000)
    parser.add_argument(
        '--tpm', type=str, help='[prepare] Pre-computed (cell x gene) TPM values as df.npz or tab separated txt file. If not provided TPM will be calculated automatically', default=None)
    parser.add_argument('--beta-loss', type=str, choices=['frobenius', 'kullback-leibler',
                        'itakura-saito'], help='[prepare] Loss function for NMF.', default='frobenius')
    parser.add_argument('--densify', dest='densify',
                        help='[prepare] Treat the input data as non-sparse', action='store_true', default=False) 
    parser.add_argument('--worker-index', type=int,
                        help='[factorize] Index of current worker (the first worker should have index 0)', default=0)
    parser.add_argument('--local-density-threshold', type=float,
                        help='[consensus] Threshold for the local density filtering. This string must convert to a float >0 and <=2', default=0.5)
    parser.add_argument('--local-neighborhood-size', type=float,
                        help='[consensus] Fraction of the number of replicates to use as nearest neighbors for local density filtering', default=0.30)
    parser.add_argument('--show-clustering', dest='show_clustering',
                        help='[consensus] Produce a clustergram figure summarizing the spectra clustering', action='store_true')

    args = parser.parse_args()

    cnmf_obj = cNMF(output_dir=args.output_dir, name=args.name)
    
    if args.command == 'prepare':
        cnmf_obj.prepare(args.counts, components=args.components, n_iter=args.n_iter, densify=args.densify,
                         tpm_fn=args.tpm, seed=args.seed, beta_loss=args.beta_loss,
                         num_highvar_genes=args.numgenes, genes_file=args.genes_file)

    elif args.command == 'factorize':
        cnmf_obj.factorize(worker_i=args.worker_index,
                           total_workers=args.total_workers)

    elif args.command == 'combine':
        cnmf_obj.combine(components=args.components)

    elif args.command == 'k_selection_plot':
        cnmf_obj.k_selection_plot(close_fig=True)

    elif args.command == 'consensus':
        run_params = load_df_from_npz(
            cnmf_obj.paths['nmf_replicate_parameters'])

        if type(args.components) is int:
            ks = [args.components]
        elif args.components is None:
            ks = sorted(set(run_params.n_components))
        else:
            ks = args.components

        for k in ks:
            cnmf_obj.consensus(
                k, args.local_density_threshold, args.local_neighborhood_size,
                args.show_clustering, close_clustergram_fig=True)


if __name__ == "__main__":
    main()