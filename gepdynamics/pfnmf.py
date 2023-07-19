## Partially Fixed Nonnegative Matrix Factorization
# Based on github.com/cwu307/NmfDrumToolbox by Chih-Wei Wu cwu307@gatech.edu
# IEEE "Drum transcription using partially fixed non-negative
# matrix factorization" DOI: 10.1109/EUSIPCO.2015.7362590
# and a later work "DRUM TRANSCRIPTION USING PARTIALLY FIXED NON-NEGATIVE
# MATRIX FACTORIZATION WITH TEMPLATE ADAPTATION"
#
# Original matlab signature:
# [WD, HD, WH, HH, err] = PfNmf(X, WD, HD, WH, HH, rh, sparsity)
# input:
#        X    = float, numFreqX*numFrames matrix, input magnitude spectrogram
#        WD   = float, numFreqD*rd matrix, drum dictionary
#        HD   = float, rd*numFrames matrix, drum activation matrix
#        WH   = float, numFreqH*rh matrix, harmonic dictionary
#        HH   = float, rh*numFrames matrix, harmonic activation matrix
#        rh   = int, rank of harmonic matrix
#        sparsity = float, sparsity coefficient
# output:
#        WD   = float, numFreqD*rd matrix, updated drum dictionary
#        HD   = float, rd*numFrames matrix, updated drum activation matrix
#        WH   = float, numFreqH*rh matrix, updated harmonic dictionary
#        HH   = float, rh*numFrames matrix, updated harmonic activation matrix
#        err  = error vector (numIter * 1)
# usage:
#        To randomly initialized different matrix, please give [] as input.
#        For example, [WD,HD,WH,HH,err] = PfNmf(X, WD, [], [], [], 0, 0)
#        is the basic NMF approach given only the drum template.
#        [WD,HD,WH,HH,err] = PfNmf(X, WD, [], [], [], 50, 0) is the
#        partially fixed NMF with 50 random intialized entries
#
# CW @ GTCMT 2015

import time
import warnings

from typing import Union

import numpy as np

from sklearn.decomposition import _nmf as sknmf
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state

PRINT_EVERY = 10


def calc_beta_divergence(X, w_1, w_2, h_1, h_2, beta_loss=1, square_root=False, per_column=False) -> Union[float, np.ndarray]:
    '''
    Calculate the beta divergence error between the original matrix X and the product of 
    the two non-negative matrices W and H, using the given beta loss parameter.

    Parameters:
    ----------
    X : array-like
        The original matrix to be factorized.
    w_1, w_2 : array-like
        The factor matrices to be multiplied to form W.
    h_1, h_2 : array-like
        The factor matrices to be multiplied to form H.
    beta_loss : float or {'frobenius', 'kullback-leibler', 'itakura-saito'}
        Parameter of the beta-divergence.
        If beta == 2, this is half the Frobenius *squared* norm.
        If beta == 1, this is the generalized Kullback-Leibler divergence.
        If beta == 0, this is the Itakura-Saito divergence.
        Else, this is the general beta-divergence.
    square_root : bool, default=False
        If True, return np.sqrt(2 * res)
        For beta == 2, it corresponds to the Frobenius norm.
    per_column : bool, optional (default=False)
        Whether to calculate the divergence error per column of X.

    Returns:
    -------
    Union[float, numpy.ndarray]
        If per_column is False, returns the overall beta divergence error between X and W*H.
        If per_column is True, returns an array of the beta divergence errors between each column
        of X and the corresponding column of W*H.
    '''

    if not per_column:
        return sknmf._beta_divergence(X, np.concatenate([w_1, w_2], axis=1),
                                      np.concatenate([h_1, h_2], axis=0),
                                      beta=beta_loss, square_root=square_root)
    else:
        result = np.full((X.shape[1]), np.inf)
        W = np.concatenate([w_1, w_2], axis=1)
        H = np.concatenate([h_1, h_2], axis=0)

        for col in range(X.shape[1]):
            result[col] = sknmf._beta_divergence(
                X[:, col: col + 1], W, H[:, col: col + 1], beta=beta_loss, square_root=square_root)

        return result


def pfnmf(X, w1, h1=None, w2=None, h2=None, rank_2: int = None,
          beta_loss=1., tol: float = 1e-4, max_iter: int = 200, verbose=False,
          random_state=None):
    """Return the partially fixed NMF solution for constant w1 matrix"""

    start_time = time.time()

    beta_loss = sknmf._beta_loss_to_float(beta_loss)
    if beta_loss not in [1., 2. ]:
        raise NotImplementedError("pfnmf is only supported for beta in [1, 2]")

    # Assert X has no zero rows or columns
    m_features, n_samples = X.shape
    if np.count_nonzero(X.sum(axis=0)) != n_samples:
        raise ValueError(
            f'X has an all-zeros sample')
    # if np.count_nonzero(X.sum(axis=1)) != m_features:
    #     raise ValueError(
    #         f'X has an all-zeros feature')
    epsilon = np.finfo(X.dtype).eps

    # Assert legal dimensions of w1 and w2 / rank_2 match
    if m_features != w1.shape[0]:
        raise ValueError(
            f'First dimension of w1 {w1.shape} does not match X {X.shape}')
    else:
        _, rank_1 = w1.shape

    if w2 is not None:
        if m_features != w2.shape[0]:
            raise ValueError(
                f'First dimension of w2 {w2.shape} does not match X {X.shape}')
        if rank_2 is not None:
            warnings.warn("Given w2 - ignoring rank_2 parameter")
        _, rank_2 = w2.shape
    elif rank_2 is None:
        raise ValueError("One of {w2, rank_2} must be provided")
    elif (rank_2 < 1) or (rank_2 % 1 != 0):
        raise ValueError("rank_2 must be a positive integer")

    # Initialize w2/h1/h2 matrices if needed
    if not (None not in [w2, h1, h2]):
        # Following sknmf "random" NMF initiation, adjusting k*E(H)*E(W) ~ E(X)
        avg_X = np.sqrt(X.mean() / (rank_1 + rank_2))
        rng = check_random_state(random_state)

        if w2 is None:
            w2 = avg_X * rng.standard_normal(size=(m_features, rank_2)).astype(
                X.dtype, copy=False)
            np.abs(w2, out=w2)
            w2 *= w1.mean() / w2.mean()  # adjusting E(w2) = E(w1)

        if h1 is None:
            h1 = avg_X * rng.standard_normal(size=(rank_1, n_samples)).astype(
                X.dtype, copy=False)
            np.abs(h1, out=h1)
            h1 *= avg_X / (w1.mean() * h1.mean())

        if h2 is None:
            h2 = avg_X * rng.standard_normal(size=(rank_2, n_samples)).astype(
                X.dtype, copy=False)
            np.abs(h2, out=h2)
            h2 *= avg_X / (w2.mean() * h2.mean())

    # Assert h1 and h2 dimensions
    if h1.shape[0] != rank_1:
        raise ValueError(
            f"h1 n rows ({h1.shape[0]}) don't match w1 n columns ({rank_1})")
    if h1.shape[1] != n_samples:
        raise ValueError(
            f'Second dimension of h1 {h1.shape} does not match X {X.shape}')
    if h2.shape[0] != rank_2:
        raise ValueError(
            f"h2 n rows ({h2.shape[0]}) don't match w2 / rank_2 ({rank_2})")
    if h2.shape[1] != n_samples:
        raise ValueError(
            f'Second dimension of h2 {h2.shape} does not match X {X.shape}')

    rep = np.ones((m_features, n_samples))

    def calc_error(w_1, w_2, h_1, h_2) -> float:
        '''Calculate the error according to the beta loss and X'''
        return calc_beta_divergence(X, w_1, w_2, h_1, h_2, beta_loss)

    error_at_init = calc_error(w1, w2, h1, h2)
    previous_error = error_at_init
    
    if verbose:
        print(f"Starting error is {error_at_init: .3e}")

    # start iterations
    for n_iter in range(1, max_iter + 1):
        # TODO: realize replacement of multiplying @ rep with sum

        approx = w1 @ h1 + w2 @ h2
        # update H
        if beta_loss == 1:   # KL
            approx[approx < epsilon] = epsilon # avoid division by zero
            x_over_approx = X / approx
            h1 = h1 * (w1.T @ x_over_approx) / (w1.T @ rep)
            h2 = h2 * (w2.T @ x_over_approx) / (w2.T @ rep)
        elif beta_loss == 2:  # Frobenius
            h1 = h1 * (w1.T @ X) / (w1.T @ approx)
            h2 = h2 * (w2.T @ X) / (w2.T @ approx)

        approx = w1 @ h1 + w2 @ h2
        # update W
        if beta_loss == 1:  # KL
            approx[approx < epsilon] = epsilon  # avoid devision by zero
            x_over_approx = X / approx
            w2 = w2 * (x_over_approx @ h2.T) / (rep @ h2.T)
            # if w1_update:
            #     w1 = w1 * (x_over_approx @ h1.T) / (rep @ h1.T)

        elif beta_loss == 2:  # Frobenius
            w2 = w2 * (X @ h2.T) / (approx @ h2.T)
            # if w1_update:
            #     w1 = w1 * (X @ h1.T) / (approx @ h1.T)

        error = calc_error(w1, w2, h1, h2)

        # Making sure the loss is non-increasing
        if previous_error < error:
            warnings.warn(
                f"Increasing loss event, starting loss was "
                f"{error_at_init: .3e}, previous loss was {previous_error: .4e}"
                f"and current loss is {error: .4e}",
                ConvergenceWarning)
            break

        if tol > 0 and n_iter % PRINT_EVERY == 0:
            if verbose:
                iter_time = time.time()
                print(f"Epoch {n_iter: 02d} reached after "
                f"{iter_time - start_time: .3f} seconds, error: {error: .4e}")

        if (previous_error - error) / error_at_init < tol:
            break
        previous_error = error

    # do not print if we have already printed in the convergence test
    if verbose and (tol == 0 or n_iter % PRINT_EVERY != 0):
        end_time = time.time()
        error = calc_error(w1, w2, h1, h2)
        print(f"Epoch {n_iter: 02d} reached after "
              f"{end_time - start_time: .3f} seconds, error: {error: .4e}")

    return w1, h1, w2, h2, n_iter

if __name__ == "__main__":
    # Basic test
    pfnmf(np.array([[1., 1], [1, 0]]), np.array([[0.], [0.5]]), rank_2=1,
          tol=0, max_iter=200, verbose=True)

