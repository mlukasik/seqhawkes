#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Copyright (c) 2014-2015, The University of Sheffield. 
This file is part of the seqhawkes software 
(see https://github.com/mlukasik/seqhawkes), 
and is free software, licenced under the GNU Library General Public License,
Version 2, June 1991 (in the distribution as file LICENSE).

Created on 15 Mar 2016

@author: mlukasik
'''

import numpy as np
from updates.likelihood_fullsummation import event_nonapproximated_logintensity
from itertools import izip


def loggamma_fullsum_func(
    loggamma,
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
    return gamma_fullsum_func(
        np.exp(loggamma),
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
        )


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

    ll = 0
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
    ll -= T * np.sum(np.outer(mu, gamma))
    return -ll


# ==================================

def loggamma_fullsum_grad(
    loggamma,
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
    gammai = np.exp(loggamma)
    subres = gamma_fullsum_grad(
        gammai,
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
        )
    return np.multiply(subres, gammai)


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
    it actually returns negated gradient.
    '''

    gradres = np.ones_like(gamma) * -T * np.sum(mu)
    for (eventidx, (etime1, infected_u, eventmeme)) in \
        enumerate(izip(etimes, node_vec, eventmemes)):
        gradres[eventmeme] += mu[infected_u] \
            / np.exp(event_nonapproximated_logintensity(
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
    return -gradres


# =====

def gamma_fullsum_funcgrad(
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

    print '[mu_fullsum_funcgrad] gamma :', gamma
    return (gamma_fullsum_func(
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
        ), gamma_fullsum_grad(
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
        ))


def loggamma_fullsum_funcgrad(
    loggamma,
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

    print '[mu_fullsum_funcgrad] loggamma:', loggamma
    return (loggamma_fullsum_func(
        loggamma,
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
        ), loggamma_fullsum_grad(
        loggamma,
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
        ))


