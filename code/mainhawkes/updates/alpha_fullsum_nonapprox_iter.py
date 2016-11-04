#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 16 Mar 2016

@author: mlukasik
'''

import numpy as np
from updates.likelihood_fullsummation import evaluate_nonapproximated_loglikelihood_wordsbyusers, \
    event_nonapproximated_logintensity
from itertools import izip


def logalpha_fullsum_func(
    logalphaij,
    indexes,
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
    return alpha_fullsum_func(
        np.exp(logalphaij),
        indexes,
        np.exp(logalpha),
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
    alphaij,
    indexes,
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

    assert len(indexes) == 2
    try:
        alphaij[0]
    except:
        alphaij = [alphaij]
    res = 0
    for alphaij_sub in alphaij:
        alpha[indexes[0]][indexes[1]] = alphaij_sub
        res += -evaluate_nonapproximated_loglikelihood_wordsbyusers(
            node_vec,
            eventmemes,
            etimes,
            T,
            mu,
            gamma,
            alpha,
            omega,
            W,
            beta,
            kernel_evaluate,
            K_evaluate,
            )
    return res


# =====

def logalpha_fullsum_grad(
    logalphaij,
    indexes,
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
    alphaij = np.exp(logalphaij)
    return alpha_fullsum_grad(
        alphaij,
        indexes,
        np.exp(logalpha),
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
        ) * alphaij


def alpha_fullsum_grad(
    alphaij,
    indexes,
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

    assert len(indexes) == 2
    try:
        alphaij[0]
    except:
        alphaij = [alphaij]
    grad_alpha = 0
    for alphaij_sub in alphaij:
        alpha[indexes[0]][indexes[1]] = alphaij_sub

        # - SUM_i^R SUM_l^N I(i==y) I(l_i==x) K(T-t_l) #we can skip the SUM_i^R and I(i==y), because there is only one such term
        # so eventually we have: - SUM_l^N I(l_i==x) K(T-t_l)

        for (eventidx, (etime1, infected_u, eventmeme)) in \
            enumerate(izip(etimes, node_vec, eventmemes)):
            if infected_u == indexes[0]:
                grad_alpha += -K_evaluate(etime1, T, omega)

        # + SUM_n^N I(i_n==y) (d intensity / d alpha_ij) / (intensity)
        # so eventually we have: - SUM_l^N I(l_i==x) K(T-t_l)

        for (eventidx, (etime1, infected_u, eventmeme)) in \
            enumerate(izip(etimes, node_vec, eventmemes)):
            if infected_u == indexes[1]:
                denominator = event_nonapproximated_logintensity(
                    infected_u,
                    eventmeme,
                    etime1,
                    T,
                    etimes,
                    node_vec,
                    eventmemes,
                    mu,
                    gamma.flatten(),
                    omega,
                    alpha,
                    kernel_evaluate,
                    )
                for (eventidx2, (etime2, infected_u2, eventmeme2)) in \
                    enumerate(izip(etimes, node_vec, eventmemes)):
                    if eventidx2 == eventidx:
                        break
                    if eventmeme == eventmeme2 and infected_u2 \
                        == indexes[0]:
                        grad_alpha += kernel_evaluate(etime1, etime2,
                                omega) / denominator
    return -grad_alpha


# =====

def alpha_fullsum_funcgrad(
    alphaij,
    indexes,
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

    print '[alpha_fullsum_funcgrad] alphaij:', alphaij
    return (alpha_fullsum_func(
        alphaij,
        indexes,
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
        alphaij,
        indexes,
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
    logalphaij,
    indexes,
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

    print '[logalpha_fullsum_funcgrad] logalphaij:', logalphaij
    return (logalpha_fullsum_func(
        logalphaij,
        indexes,
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
        logalphaij,
        indexes,
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


