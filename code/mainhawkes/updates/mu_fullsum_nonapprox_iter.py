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
    logmui,
    index,
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
        np.exp(logmui),
        index,
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
    mui,
    index,
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
    try:
        mui[0]
    except:
        mui = [mui]
    likelihood = 0
    for mui_sub in mui:
        mu[index] = mui_sub
        likelihood += \
            evaluate_nonapproximated_loglikelihood_wordsbyusers(
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
    return -likelihood


# =====

def logmu_fullsum_grad(
    logmui,
    index,
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
    mui = np.exp(logmui)
    return mu_fullsum_grad(
        mui,
        index,
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
        ) * mui


def mu_fullsum_grad(
    mui,
    index,
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

    try:
        mui[0]
    except:
        mui = [mui]
    gradres = 0
    for mui_sub in mui:
        mu[index] = mui_sub
        grad_mu = -T * np.sum(gamma)
        for (eventidx, (etime1, infected_u, eventmeme)) in \
            enumerate(izip(etimes, node_vec, eventmemes)):
            if infected_u == index:
                grad_mu += gamma[eventmeme] \
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

    #         except:
    #             pass

        gradres += grad_mu
    return -gradres


# =====

def mu_fullsum_funcgrad(
    mui,
    index,
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

    print '[mu_fullsum_funcgrad] mui:', mui
    return (mu_fullsum_func(
        mui,
        index,
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
        mui,
        index,
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
    logmui,
    index,
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

    print '[mu_fullsum_funcgrad] logmui:', logmui
    return (logmu_fullsum_func(
        logmui,
        index,
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
        logmui,
        index,
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


