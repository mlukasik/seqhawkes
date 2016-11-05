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
from updates.likelihood_fullsummation import evaluate_nonapproximated_loglikelihood_wordsbyusers, \
    event_nonapproximated_logintensity
from itertools import izip


def logalpha_fullsum_func(
    logalpha,
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
    ):
    '''
    alpha should be passed as a list, not as a matrix! It is internally reshaped to be a matrix.
    '''

    return alpha_fullsum_func(
        np.exp(np.resize(logalpha, (len(mu), len(mu)))),
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


def alpha_fullsum_func(
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
    ):
    '''
    '''

#     res=0
#    for alphaij_sub in alphaij:
#    alpha[indexes[0]][indexes[1]]=alphaij_sub
#     res=evaluate_nonapproximated_loglikelihood_wordsbyusers(node_vec, eventmemes, etimes,
#                                                             T, mu, gamma, alpha, omega, W, beta,
#                                                             kernel_evaluate, K_evaluate)

    ll = 0

    alpha_summedrows = np.sum(alpha, axis=1)
    for (eventidx, (infected_u, eventmeme, etime1)) in \
        enumerate(izip(node_vec, eventmemes, etimes)):
        ll += event_nonapproximated_logintensity(
            infected_u,
            eventmeme,
            etime1,
            T,
            etimes[:eventidx],
            node_vec[:eventidx],
            eventmemes[:eventidx],
            mu,
            gamma,
            omega,
            alpha,
            kernel_evaluate,
            )
        ll -= alpha_summedrows[int(infected_u)] * K_evaluate(etime1, T,
                omega)

#         if -ll==np.Infinity:
#             event_nonapproximated_logintensity(infected_u, eventmeme, etime1, T,
#                                                etimes[:eventidx], node_vec[:eventidx], eventmemes[:eventidx],
#                                                mu, gamma, omega, alpha, kernel_evaluate)
#
#     ll-=T*np.sum(np.outer(mu, gamma))

    return -ll


# =====

def logalpha_fullsum_grad(
    logalpha,
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
    ):
    '''
    alpha should be passed as a list, not as a matrix! It is internally reshaped to be a matrix.
    '''

    alpha = np.exp(np.resize(logalpha, (len(mu), len(mu))))
    return np.multiply(alpha_fullsum_grad(
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
        ), alpha).flatten()


def alpha_fullsum_grad(
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
    ):
    '''
    it actually returns negated gradient.
    '''

    grad_alpha = np.zeros_like(alpha)

        # - SUM_i^R SUM_l^N I(i==y) I(l_i==x) K(T-t_l) #we can skip the SUM_i^R and I(i==y), because there is only one such term
        # so eventually we have: - SUM_l^N I(l_i==x) K(T-t_l)

    for (eventidx, (etime1, infected_u, eventmeme)) in \
        enumerate(izip(etimes, node_vec, eventmemes)):

#         if infected_u==indexes[0]:

        grad_alpha[infected_u, :] += -K_evaluate(etime1, T, omega)

    # + SUM_n^N I(i_n==y) (d intensity / d alpha_ij) / (intensity)
    # so eventually we have: - SUM_l^N I(l_i==x) K(T-t_l)
#     for eventidx, (etime1, infected_u, eventmeme) in enumerate(izip(etimes, node_vec, eventmemes)):

        denominator = np.exp(event_nonapproximated_logintensity(
            infected_u,
            eventmeme,
            etime1,
            T,
            etimes[:eventidx],
            node_vec[:eventidx],
            eventmemes[:eventidx],
            mu,
            gamma.flatten(),
            omega,
            alpha,
            kernel_evaluate,
            ))
        for (eventidx2, (etime2, infected_u2, eventmeme2)) in \
            enumerate(izip(etimes, node_vec, eventmemes)):
            if eventidx2 == eventidx:
                break
            if eventmeme == eventmeme2:
                grad_alpha[infected_u2, infected_u] += \
                    kernel_evaluate(etime1, etime2, omega) / denominator
    return -grad_alpha


# =====

def alpha_fullsum_funcgrad(
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
    ):
    '''
    '''

    print '[alpha_fullsum_funcgrad] alpha:', alpha
    return (alpha_fullsum_func(
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
        ), alpha_fullsum_grad(
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
        ))


def logalpha_fullsum_funcgrad(
    logalpha,
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
    ):
    '''
    '''

    print '[logalpha_fullsum_funcgrad] logalpha:', logalpha
    return (logalpha_fullsum_func(
        logalpha,
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
        ), logalpha_fullsum_grad(
        logalpha,
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
        ))


