#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Copyright (c) 2014-2015, The University of Sheffield. 
This file is part of the seqhawkes software 
(see https://github.com/mlukasik/seqhawkes), 
and is free software, licenced under the GNU Library General Public License,
Version 2, June 1991 (in the distribution as file LICENSE).

Created on 20 Jul 2015

@author: michal, srijith
'''

from itertools import izip
import numpy as np


# from updates.kernel_zha import kernel_evaluate, K_evaluate

def loglikelihood_words(eventmemes, W, beta):
    ll = 0
    for (eventidx, eventmeme) in enumerate(eventmemes):

#         print "W[eventidx, :]:", W[eventidx, :]
#         print "beta[eventmeme, :]:", beta[eventmeme, :]

        ll += np.dot(W[eventidx, :], np.log(beta[eventmeme, :]))
    return ll


def event_approximated_logintensity(
    infected_u,
    infecting_u,
    infected_e,
    infecting_e,
    eventmeme,
    etime1,
    T,
    etimes,
    infected_vec,
    mu,
    gamma,
    omega,
    alpha,
    kernel_evaluate,
    print_summed_elements=False,
    ):

    components = []
    ll = 0
    if infected_e == infecting_e:

        # self excited event

        if mu[int(infected_u)] * gamma[int(eventmeme)] > 0:
            ll += np.log(mu[int(infected_u)] * gamma[int(eventmeme)])
            components.append(np.log(mu[int(infected_u)]
                              * gamma[int(eventmeme)]))
    else:

        # influenced by another event

        etime2 = etimes[infected_vec == infecting_e]
        try:
            assert len(etime2) == 1
        except:
            print 'etime2:', etime2
            print 'infected_vec', infected_vec
            print 'infecting_e', infecting_e
            assert len(etime2) == 1
        etime2 = etime2[0]
        intensity_val = alpha[int(infecting_u), int(infected_u)] \
            * kernel_evaluate(etime1, etime2, omega)
        components.append(np.log(intensity_val))
        ll += np.log(intensity_val)
    if print_summed_elements:
        print '\t\t\t\t\t\t\t\t\t\t\t[event_approximated_logintensity] intensity=' \
            + '+ '.join(map(lambda x: '%10.6f' % x, components))
    return ll


def evaluate_pseudologlikelihood_wordsbyusers_meme(
    infecting_vec,
    infected_vec,
    node_vec,
    eventmemes,
    etimes,
    infecting_node_vec,
    T,
    mu,
    gamma,
    alpha,
    omega,
    W,
    beta,
    meme,
    kernel_evaluate,
    K_evaluate,
    ):

    ll = 0
    alpha_summedrows = np.sum(alpha, axis=1)

    indices = np.where(eventmemes == meme)[0]
    for eventidx in indices:
        (
            infected_u,
            infecting_u,
            infected_e,
            infecting_e,
            eventmeme,
            etime1,
            ) = (
            node_vec[eventidx],
            infecting_node_vec[eventidx],
            infected_vec[eventidx],
            infecting_vec[eventidx],
            eventmemes[eventidx],
            etimes[eventidx],
            )
        ll += event_approximated_logintensity(
            infected_u,
            infecting_u,
            infected_e,
            infecting_e,
            eventmeme,
            etime1,
            T,
            etimes,
            infected_vec,
            mu,
            gamma,
            omega,
            alpha,
            kernel_evaluate,
            )
        ll -= alpha_summedrows[int(infected_u)] * K_evaluate(etime1, T,
                omega)
        ll += np.dot(W[eventidx, :], np.log(beta[infected_u, :]))
    return ll


def yield_intensity_per_time(
    infecting_vec,
    infected_vec,
    node_vec,
    eventmemes,
    etimes,
    infecting_node_vec,
    T,
    mu,
    gamma,
    alpha,
    omega,
    kernel_evaluate,
    ):

    # alpha_summedrows=np.sum(alpha, axis=1)

    for (
        infected_u,
        infecting_u,
        infected_e,
        infecting_e,
        eventmeme,
        etime1,
        ) in izip(
        node_vec,
        infecting_node_vec,
        infected_vec,
        infecting_vec,
        eventmemes,
        etimes,
        ):
        yield (event_approximated_logintensity(
            infected_u,
            infecting_u,
            infected_e,
            infecting_e,
            eventmeme,
            etime1,
            T,
            etimes,
            infected_vec,
            mu,
            gamma,
            omega,
            alpha,
            kernel_evaluate,
            ), etime1)


def evaluate_approximated_loglikelihood_nowords(
    infecting_vec,
    infected_vec,
    node_vec,
    eventmemes,
    etimes,
    infecting_node_vec,
    T,
    mu,
    gamma,
    alpha,
    omega,
    kernel_evaluate,
    ):

    ll = 0
    for (intensity, etime1) in yield_intensity_per_time(
        infecting_vec,
        infected_vec,
        node_vec,
        eventmemes,
        etimes,
        infecting_node_vec,
        T,
        mu,
        gamma,
        alpha,
        omega,
        kernel_evaluate,
        ):
        ll += intensity

    ll -= T * np.sum(np.outer(mu, gamma))
    return ll


def evaluate_loglikelihood(
    infecting_vec,
    infected_vec,
    node_vec,
    eventmemes,
    etimes,
    infecting_node_vec,
    T,
    mu,
    gamma,
    alpha,
    omega,
    W,
    beta,
    kernel_evaluate,
    ):

    ll = evaluate_approximated_loglikelihood_nowords(
        infecting_vec,
        infected_vec,
        node_vec,
        eventmemes,
        etimes,
        infecting_node_vec,
        T,
        mu,
        gamma,
        alpha,
        omega,
        kernel_evaluate,
        )
    ll += loglikelihood_words(eventmemes, W, beta)
    return ll


def evaluate_approximated_loglikelihood_wordsbyusers(
    infecting_vec,
    infected_vec,
    node_vec,
    eventmemes,
    etimes,
    infecting_node_vec,
    T,
    mu,
    gamma,
    alpha,
    omega,
    W,
    beta,
    kernel_evaluate,
    ):
    '''
    Words generated by users rather than by memes;
    '''

    ll = evaluate_approximated_loglikelihood_nowords(
        infecting_vec,
        infected_vec,
        node_vec,
        eventmemes,
        etimes,
        infecting_node_vec,
        T,
        mu,
        gamma,
        alpha,
        omega,
        kernel_evaluate,
        )
    ll += loglikelihood_words(node_vec, W, beta)
    return ll


def evaluate_loglikelihood_regularized(
    infecting_vec,
    infected_vec,
    node_vec,
    eventmemes,
    etimes,
    infecting_node_vec,
    T,
    mu,
    gamma,
    alpha,
    omega,
    W,
    beta,
    kernel_evaluate,
    D,
    regularization_connections_set=[],
    ):

    ll = evaluate_loglikelihood(
        infecting_vec,
        infected_vec,
        node_vec,
        eventmemes,
        etimes,
        infecting_node_vec,
        T,
        mu,
        gamma,
        alpha,
        omega,
        W,
        beta,
        kernel_evaluate,
        )

    if len(regularization_connections_set) != 0:
        for i in range(0, D):
            for j in range(0, D):
                if (i, j) in regularization_connections_set:
                    continue
                else:
                    ll -= alpha[i, j] ** 2

    return ll


