#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Copyright (c) 2014-2015, The University of Sheffield. 
This file is part of the seqhawkes software 
(see https://github.com/mlukasik/seqhawkes), 
and is free software, licenced under the GNU Library General Public License,
Version 2, June 1991 (in the distribution as file LICENSE).

Created on 16 Mar 2016

@author: mlukasik
'''

import numpy as np
from updates.likelihood_fullsummation import evaluate_nonapproximated_loglikelihood_nowords
from updates.mu_fullsum_nonapprox import mu_fullsum_grad
from updates.alpha_fullsum_nonapprox import alpha_fullsum_grad


def encode_all_params_nogammanoomega(mu, alpha):
    return np.hstack((mu, np.array(alpha).flatten()))


def decode_all_params_nogammanoomega(all_params, lenmu):
    mu = all_params[:lenmu]
    alpha = np.resize(all_params[lenmu:], (lenmu, lenmu))
    return (mu, alpha)


def log_fullsum_func(
    logparams,
    omega,
    gamma,
    node_vec,
    eventmemes,
    etimes,
    T,
    W,
    beta,
    kernel_evaluate,
    K_evaluate,
    lenmu,
    ):
    '''
    alpha should be passed as a list, not as a matrix! It is internally reshaped to be a matrix.
    '''

    return fullsum_func(
        np.exp(logparams),
        omega,
        gamma,
        node_vec,
        eventmemes,
        etimes,
        T,
        W,
        beta,
        kernel_evaluate,
        K_evaluate,
        lenmu,
        )


def fullsum_func(
    params,
    omega,
    gamma,
    node_vec,
    eventmemes,
    etimes,
    T,
    W,
    beta,
    kernel_evaluate,
    K_evaluate,
    lenmu,
    ):
    '''
    '''

    (mu, alpha) = decode_all_params_nogammanoomega(params, lenmu)
    ll = evaluate_nonapproximated_loglikelihood_nowords(
        node_vec,
        eventmemes,
        etimes,
        T,
        mu,
        gamma,
        alpha,
        omega,
        kernel_evaluate,
        K_evaluate,
        )

    return -ll


# =====

def log_fullsum_grad(
    logparams,
    omega,
    gamma,
    node_vec,
    eventmemes,
    etimes,
    T,
    W,
    beta,
    kernel_evaluate,
    K_evaluate,
    lenmu,
    derivative_kernel_evaluate,
    derivative_K_evaluate,
    ):
    '''
    alpha should be passed as a list, not as a matrix! It is internally reshaped to be a matrix.
    '''

    params = np.exp(logparams)
    return np.multiply(fullsum_grad(
        params,
        omega,
        gamma,
        node_vec,
        eventmemes,
        etimes,
        T,
        W,
        beta,
        kernel_evaluate,
        K_evaluate,
        lenmu,
        derivative_kernel_evaluate,
        derivative_K_evaluate,
        ), params)


def fullsum_grad(
    params,
    omega,
    gamma,
    node_vec,
    eventmemes,
    etimes,
    T,
    W,
    beta,
    kernel_evaluate,
    K_evaluate,
    lenmu,
    derivative_kernel_evaluate,
    derivative_K_evaluate,
    ):
    '''
    it actually returns negated gradient.
    '''

    (mu, alpha) = decode_all_params_nogammanoomega(params, lenmu)
    mugrad = mu_fullsum_grad(
        mu,
        node_vec,
        eventmemes,
        etimes,
        T,
        gamma,
        alpha,
        omega,
        W,
        beta,
        kernel_evaluate,
        K_evaluate,
        )
    alphagrad = alpha_fullsum_grad(
        alpha,
        node_vec,
        eventmemes,
        etimes,
        T,
        mu,
        gamma,
        omega,
        W,
        beta,
        kernel_evaluate,
        K_evaluate,
        )

    grad = encode_all_params_nogammanoomega(mugrad, alphagrad)
    return grad  # already flipped the sign before, so no need to put minus


# =====

def fullsum_funcgrad(
    updates,
    omega,
    gamma,
    node_vec,
    eventmemes,
    etimes,
    T,
    W,
    beta,
    kernel_evaluate,
    K_evaluate,
    lenmu,
    derivative_kernel_evaluate,
    derivative_K_evaluate,
    ):
    return (fullsum_func(
        updates,
        omega,
        gamma,
        node_vec,
        eventmemes,
        etimes,
        T,
        W,
        beta,
        kernel_evaluate,
        K_evaluate,
        lenmu,
        ), fullsum_grad(
        updates,
        omega,
        gamma,
        node_vec,
        eventmemes,
        etimes,
        T,
        W,
        beta,
        kernel_evaluate,
        K_evaluate,
        lenmu,
        derivative_kernel_evaluate,
        derivative_K_evaluate,
        ))


def log_fullsum_funcgrad(
    logupdates,
    omega,
    gamma,
    node_vec,
    eventmemes,
    etimes,
    T,
    W,
    beta,
    kernel_evaluate,
    K_evaluate,
    lenmu,
    derivative_kernel_evaluate,
    derivative_K_evaluate,
    ):
    return (log_fullsum_func(
        logupdates,
        omega,
        gamma,
        node_vec,
        eventmemes,
        etimes,
        T,
        W,
        beta,
        kernel_evaluate,
        K_evaluate,
        lenmu,
        ), log_fullsum_grad(
        logupdates,
        omega,
        gamma,
        node_vec,
        eventmemes,
        etimes,
        T,
        W,
        beta,
        kernel_evaluate,
        K_evaluate,
        lenmu,
        derivative_kernel_evaluate,
        derivative_K_evaluate,
        ))


