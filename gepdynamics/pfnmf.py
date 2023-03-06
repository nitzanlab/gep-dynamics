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

import numpy as np
from sklearn.decomposition import _nmf as sknmf

PRINT_EVERY = 10

def pfnmf(X, w1, h1=None, w2=None, h2=None, rh: int=None,
          beta_loss=1., tol: float=1e-4, max_iter: int=200, verbose=False):
    '''Return the partially fixed NMF solution for constant w1 matrix'''

    start_time = time.time()

    beta_loss = sknmf._beta_loss_to_float(beta_loss)
    if beta_loss not in [1., 2. ]:
        raise NotImplementedError("pfnmf is only supported for beta in [1, 2]")


    X += np.finfo(float).tiny  # make sure there's no zero frame
    numFreqX, numFrames = X.shape
    numFreqD, rd = w1.shape

    # initialization
    w1_update = 0
    h1_update = 0
    w2_update = 0
    h2_update = 0

    if w2 is not None:
        numFreqH, rh = w2.shape
    else:
        w2 = np.random.rand(numFreqD, rh)
        numFreqH, _ = w2.shape
        w2_update = 1

    if numFreqD != numFreqX:
        raise ValueError('Dimensionality of the w1 does not match X')
    elif numFreqH != numFreqX:
        raise ValueError('Dimensionality of the w2 does not match X')

    if h1 is not None:
        w1_update = 1
    else:
        h1 = np.random.rand(rd, numFrames)
        h1_update = 1

    if h2 is None:
        h2 = np.random.rand(rh, numFrames)
        h2_update = 1

    # normalize W / H matrix
    for i in range(rd):
        w1[:, i] = w1[:, i] / np.linalg.norm(w1[:, i], 1)

    for i in range(rh):
        w2[:, i] = w2[:, i] / np.linalg.norm(w2[:, i], 1)

    rep = np.ones((numFreqX, numFrames))

    def calc_error(w1, w2, h1, h2) -> float:
        '''Calculate the error according to the beta loss and X'''
        return sknmf._beta_divergence(X,
                                      np.concatenate([w1, w2], axis=1),
                                      np.concatenate([h1, h2], axis=0),
                                      beta=beta_loss, square_root=True)

    error_at_init = calc_error(w1, w2, h1, h2)
    previous_error = error_at_init

    # start iterations
    for n_iter in range(1, max_iter + 1):
        # TODO: realize replacement of multipling @ rep with sum
        # TODO: Fix the "h1_update" and friends - only allow one direction

        # update H
        if beta_loss == 1:   # KL
            x_over_approx = X / (w1 @ h1 + w2 @ h2)
            if h1_update:
                h1 = h1 * (w1.T @ (x_over_approx)) / (w1.T @ rep)
            if h2_update:
                h2 = h2 * (w2.T @ (x_over_approx)) / (w2.T @ rep)
        elif beta_loss == 2:  # Frobenius
            approx = w1 @ h1 + w2 @ h2
            if h1_update:
                h1 = h1 * (w1.T @ X) / (w1.T @ approx)
            if h2_update:
                h2 = h2 * (w2.T @ X) / (w2.T @ approx)

        # update W
        if beta_loss == 1:  # KL
            x_over_approx = X / (w1 @ h1 + w2 @ h2)
            if w1_update:
                w1 = w1 * (x_over_approx @ h1.T) / (rep @ h1.T)
            if w2_update:
                w2 = w2 * (x_over_approx @ h2.T) / (rep @ h2.T)
        elif beta_loss == 2:  # Frobenius
            approx = w1 @ h1 + w2 @ h2
            if w1_update:
                w1 = w1 * (X @ h1.T) / (approx @ h1.T)
            if w2_update:
                w2 = w2 * (X @ h2.T) / (approx @ h2.T)

        # normalize W matrix
        for i in range(rh):
            w2[:, i] = w2[:, i] / np.linalg.norm(w2[:, i], 1)
        for i in range(rd):
            w1[:, i] = w1[:, i] / np.linalg.norm(w1[:, i], 1)

        error = calc_error(w1, w2, h1, h2)

        # Making sure the loss is non-increasing
        if previous_error < error:
            raise Exception("Increasing loss event, starting loss was "
                            f"{error_at_init: .2e}, previous loss was "
                            f"{previous_error} and current loss is {error}")

        if tol > 0 and n_iter % PRINT_EVERY == 0:
            if verbose:
                iter_time = time.time()
                print(f"Epoch {n_iter: 02d} reached after "
                f"{iter_time - start_time: .3f} seconds, error: {error: .2e}")

        if (previous_error - error) / error_at_init < tol:
            break
        previous_error = error



    # do not print if we have already printed in the convergence test
    if verbose and (tol == 0 or n_iter % PRINT_EVERY != 0):
        end_time = time.time()
        error = calc_error(w1, w2, h1, h2)
        print(f"Epoch {n_iter: 02d} reached after "
              f"{end_time - start_time: .3f} seconds, error: {error: .2e}")

    return w1, h1, w2, h2, n_iter


if __name__ == "__main__":
    # Basic test
    pfnmf(np.array([[1., 1], [1, 0]]), np.array([[0.], [0.5]]), rh=1,
          tol=0, max_iter=200, verbose=True)