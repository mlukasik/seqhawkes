#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 13 Mar 2015

@author: michal
'''

import numpy as np
from updates.likelihood_fullsummation import evaluate_nonapproximated_loglikelihood_wordsbyusers, \
    event_nonapproximated_logintensity
from itertools import izip


def mu_normalizeconst(T, gamma):
    Z = T * np.sum(gamma)
    return Z


def mu_update(T, gamma, spontaneous_node_vec):
    return spontaneous_node_vec / mu_normalizeconst(T, gamma)


def gamma_normalizeconst(T, mu):
    Z = T * np.sum(mu)
    return Z


def gamma_update(T, spontaneous_meme_vec, mu):
    return spontaneous_meme_vec / gamma_normalizeconst(T, mu)


# gradient approach for the full summation

def gamma_fullsum_func(
    gamma,
    node_vec,
    eventmemes,
    etimes,
    T,
    mu,
    alpha,
    omega,
    W,
    beta,
    kernel_evaluate,
    K_evaluate,
    ):
    '''
    '''

    return -evaluate_nonapproximated_loglikelihood_wordsbyusers(
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


def gamma_fullsum_grad(
    gamma,
    node_vec,
    eventmemes,
    etimes,
    T,
    mu,
    alpha,
    omega,
    W,
    beta,
    kernel_evaluate,
    K_evaluate,
    ):
    '''
    '''

    # print "node_vec:", node_vec

    grad_gamma = np.array([-T * np.sum(mu) for _ in range(len(gamma))])
    for (eventidx, (etime1, infected_u, eventmeme)) in \
        enumerate(izip(etimes, node_vec, eventmemes)):
        grad_gamma[eventmeme] += mu[infected_u] \
            / event_nonapproximated_logintensity(
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
    return -grad_gamma


