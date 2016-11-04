#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 15 Mar 2016

@author: mlukasik
'''

import numpy as np
from updates.likelihood_fullsummation import evaluate_nonapproximated_loglikelihood_wordsbyusers, \
    event_nonapproximated_logintensity
from itertools import izip


def logmu_fullsum_func(
    logmu,
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
    ):
    return mu_fullsum_func(
        np.exp(logmu),
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


def mu_fullsum_func(
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
    ):
    '''
    '''

#     return -evaluate_nonapproximated_loglikelihood_wordsbyusers(node_vec, eventmemes, etimes,
#                                                                 T, mu, gamma, alpha, omega, W, beta,
#                                                                 kernel_evaluate, K_evaluate)

    ll = 0

##     alpha_summedrows=np.sum(alpha, axis=1)

    for (eventidx, (infected_u, eventmeme, etime1)) in \
        enumerate(izip(node_vec, eventmemes, etimes)):

#        ll+=gamma[eventmeme]*mu[infected_u]

        ll += event_nonapproximated_logintensity(  #                                                etimes, node_vec, eventmemes,
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

##         ll-=alpha_summedrows[int(infected_u)]*K_evaluate(etime1, T, omega)
##         if -ll==np.Infinity:
##             event_nonapproximated_logintensity(infected_u, eventmeme, etime1, T, etimes[:eventidx], node_vec[:eventidx],
##                                                eventmemes[:eventidx], mu, gamma, omega, alpha, kernel_evaluate)

    ll -= T * np.sum(np.outer(mu, gamma))
    return -ll


# =====

def logmu_fullsum_grad(
    logmu,
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
    ):
    mui = np.exp(logmu)
    subres = mu_fullsum_grad(
        mui,
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
    return np.multiply(subres, mui)


def mu_fullsum_grad(
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
    ):
    '''
    it actually returns negated gradient.
    '''

#     print "[mu_fullsum_grad] mu", mu
#    try:
#        mui[0]
#    except:
#        mui=[mui]

    gradres = np.ones_like(mu) * -T * np.sum(gamma)
    for (eventidx, (etime1, infected_u, eventmeme)) in \
        enumerate(izip(etimes, node_vec, eventmemes)):

#        gradres[infected_u]+=gamma[eventmeme]
#         #if infected_u==index:

        gradres[infected_u] += gamma[eventmeme] \
            / np.exp(event_nonapproximated_logintensity(  #                                                                                         etimes, node_vec, eventmemes,
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
            ))

#        gradres+=grad_mu

    return -gradres


# =====

def mu_fullsum_funcgrad(
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
    ):
    '''
    '''

    print '[mu_fullsum_funcgrad] mu:', mu
    return (mu_fullsum_func(
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
        ), mu_fullsum_grad(
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
        ))


def logmu_fullsum_funcgrad(
    logmu,
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
    ):
    '''
    '''

    print '[mu_fullsum_funcgrad] logmu:', logmu
    return (logmu_fullsum_func(
        logmu,
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
        ), logmu_fullsum_grad(
        logmu,
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
        ))


