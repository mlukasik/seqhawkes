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


def loglikelihood_words(eventmemes, W, beta):
    ll = 0
    for (eventidx, eventmeme) in enumerate(eventmemes):
        ll += np.dot(W[eventidx, :], np.log(beta[eventmeme, :]))
    return ll


def event_approximated_logintensity(
    infected_u,
    eventmeme,
    etime1,
    T,
    etimes,
    node_vec,
    eventmemes,
    mu,
    gamma,
    omega,
    alpha,
    kernel_evaluate,
    print_summed_elements=False,
    ):

    components = []
    added_component = False
    ll = 0
    if mu[int(infected_u)] * gamma[int(eventmeme)] > 0:
        ll += np.log(mu[int(infected_u)] * gamma[int(eventmeme)])
        components.append(np.log(mu[int(infected_u)]
                          * gamma[int(eventmeme)]))
        added_component = True

    for (eventidx, (etime2, node2, eventmeme2)) in \
        enumerate(izip(etimes, node_vec, eventmemes)):
        if etime2 < etime1:
            if eventmeme2 == eventmeme:
                intensity_val = alpha[int(node2), int(infected_u)] \
                    * kernel_evaluate(etime1, etime2, omega)
                if intensity_val > 0:
                    ll += np.log(intensity_val)
                    components.append(np.log(intensity_val))
                    added_component = True

    # If there are other components then 0, we can ignore zeros; however,
    # if there are no components whatsoever, then we get minus infinity

    if not added_component:
        ll = -np.Infinity
    if print_summed_elements:
        print '\t\t\t\t\t\t\t\t\t\t\t[event_approximated_logintensity] intensity=' \
            + '+ '.join(map(lambda x: '%10.6f' % x, components))
    return ll


def event_nonapproximated_logintensity(
    infected_u,
    eventmeme,
    etime1,
    T,
    etimes,
    node_vec,
    eventmemes,
    mu,
    gamma,
    omega,
    alpha,
    kernel_evaluate,
    print_summed_elements=False,
    ):
    '''
    Useful for prediction, but not for log likelihood computations used for 
    checking stopping criteria in updating - our updates are derived from the
    approximated intensity formulation, so it is not guaranteed to decrease
    if the non approximated intensity is used.
    '''

    components = []
    ll = 0
    ll += mu[int(infected_u)] * gamma[int(eventmeme)]
    components.append(mu[int(infected_u)] * gamma[int(eventmeme)])

#     if print_summed_elements:
#         print "etime2:",

    for (eventidx, (etime2, node2, eventmeme2)) in \
        enumerate(izip(etimes, node_vec, eventmemes)):
        if etime2 < etime1:
            if eventmeme2 == eventmeme:

#                 if print_summed_elements:
#                     print etime2,

                intensity_val = alpha[int(node2), int(infected_u)] \
                    * kernel_evaluate(etime1, etime2, omega)
                ll += intensity_val
                components.append(intensity_val)

#     if print_summed_elements:
#         print
    # print "[event_nonapproximated_logintensity] ll:", ll

    if print_summed_elements:
        print '\t\t\t\t\t\t\t\t\t\t\t[event_nonapproximated_logintensity] intensity=' \
            + '+ '.join(map(lambda x: '%10.6f' % x, components))
    return np.log(ll)


# def event_intensity_nonlog(infected_u, eventmeme, etime1, T,
#                      etimes, node_vec, eventmemes,
#                      mu, gamma, omega, alpha,
#                      kernel_evaluate):
#     ll=0
#     if mu[int(infected_u)]*gamma[int(eventmeme)] > 0:
#         ll+=mu[int(infected_u)]*gamma[int(eventmeme)]
#
#     for eventidx, (etime2, node2, eventmeme2) in enumerate(izip(etimes, node_vec, eventmemes)):
#         if etime2 <= etime1:
#             if eventmeme2==eventmeme:
#                 intensity_val=alpha[int(node2), int(infected_u)]*kernel_evaluate(etime1, etime2, omega)
#                 if intensity_val > 0:
#                     ll+=intensity_val
#     return ll

def evaluate_loglikelihood_nowords(
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
    event_logintensity_func,
    ):

    ll = 0
    alpha_summedrows = np.sum(alpha, axis=1)
    for (eventidx, (infected_u, eventmeme, etime1)) in \
        enumerate(izip(node_vec, eventmemes, etimes)):
        ll += event_logintensity_func(
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
        if -ll == np.Infinity:
            event_logintensity_func(
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
    return ll


# TODO: need to write a test for this!

def evaluate_approximated_loglikelihood_nowords(
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
    ):

    return evaluate_loglikelihood_nowords(
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
        event_approximated_logintensity,
        )


def evaluate_nonapproximated_loglikelihood_nowords(
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
    ):

    return evaluate_loglikelihood_nowords(
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
        event_nonapproximated_logintensity,
        )


def evaluate_approximated_loglikelihood_wordsbyusers(
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
    ):
    '''
    Words generated by users rather than by memes;
    '''

    ll = evaluate_approximated_loglikelihood_nowords(
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
    ll += loglikelihood_words(node_vec, W, beta)

#     print "[evaluate_approximated_loglikelihood_wordsbyusers] ll:", ll
#     print "node_vec:", list(node_vec)

    return ll


def evaluate_nonapproximated_loglikelihood_wordsbyusers(
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
    ):
    '''
    Words generated by users rather than by memes;
    '''

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
    ll += loglikelihood_words(node_vec, W, beta)
    return ll


