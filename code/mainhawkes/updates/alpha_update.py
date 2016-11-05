#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Copyright (c) 2014-2015, The University of Sheffield. 
This file is part of the seqhawkes software 
(see https://github.com/mlukasik/seqhawkes), 
and is free software, licenced under the GNU Library General Public License,
Version 2, June 1991 (in the distribution as file LICENSE).

Created on Tue Mar 24 21:03:44 2015

@author: srijith, michal, trevor
"""

import numpy as np


# from updates.kernel_zha import K_evaluate

def alphaI_update_lower(
    alphaI,
    node_vec,
    infecting_node_vec,
    omega,
    etimes,
    alphaS,
    uMat,
    T,
    R,
    K,
    K_evaluate,
    ):
    '''
    This is a fixed point approach, where we take gradient of the objective function and equate to zero, 
    re-arrange the terms and then have f(x) = x.
    
    TODO: get rid of the for loops.
    '''

    alphaInew = np.zeros((R, K / 2))

    alpha = np.dot(alphaI, alphaS.transpose())
    alpha[alpha == 0] = 1e-6
    alphabroad = alpha[infecting_node_vec, :][:, node_vec]
    sumalphaS = np.sum(alphaS, 0)
    kerneltimes = K_evaluate(etimes, T, omega)

    for k in range(K / 2):
        alphak = np.outer(alphaI[:, k], alphaS[:, k])
        for j in range(R):

#            print "alphaI update iteration " +  str(j) + " of " + str(R)

            alphakbroad = alphak[infecting_node_vec, :][:, node_vec]
            alphakj = alphakbroad[infecting_node_vec == j, :]
            alphaj = alphabroad[infecting_node_vec == j, :]
            vecnlMatj = uMat[infecting_node_vec == j, :]
            numer = np.sum(vecnlMatj * alphakj / alphaj)
            kerneltimesj = kerneltimes[node_vec == j]
            denom = np.sum(kerneltimesj * sumalphaS[k])
            alphaInew[j, k] = numer / denom

    return alphaInew


def alphaS_update_lower(
    alphaS,
    node_vec,
    infecting_node_vec,
    omega,
    etimes,
    alphaI,
    uMat,
    T,
    R,
    K,
    K_evaluate,
    ):

    alphaSnew = np.zeros((R, K / 2))

    alpha = np.dot(alphaI, alphaS.transpose())
    alpha[alpha == 0] = 1e-6
    alphabroad = alpha[infecting_node_vec, :][:, node_vec]
    kerneltimes = K_evaluate(etimes, T, omega)

    for k in range(K / 2):
        alphak = np.outer(alphaI[:, k], alphaS[:, k])
        for i in range(R):

#            print "alphaI update iteration " +  str(i) + " of " + str(R)

            alphakbroad = alphak[infecting_node_vec, :][:, node_vec]
            alphaki = alphakbroad[:, node_vec == i]
            alphai = alphabroad[:, node_vec == i]
            vecnlMati = uMat[:, node_vec == i]
            numer = np.sum(vecnlMati * alphaki / alphai)
            denom = np.sum(kerneltimes * alphaI[node_vec, k])
            alphaSnew[i, k] = numer / denom
    return alphaSnew


def alpha_update_linalg(  # , sparse=False):
    infections_mat,
    enode_mat,
    etimes,
    omega,
    T,
    K_evaluate,
    ):
    '''
    infections_mat - counts of infections between respective pair of users
    enode_mat - 1-hot representation of which node got infected for each event
    etimes - vector of floatof size N, provide time when the event has occurred
    omega - kernel parameter
    T - upper time limit
    '''

    kernel_times = K_evaluate(etimes, T, omega)

    # enode_mat = one_hot(infected_node, D)

    denominator = enode_mat.dot(kernel_times)

    # enode_mat prob needs to be transposed (can be computed this way)

    update = infections_mat / denominator[:, np.newaxis]

    update[np.isnan(update)] = 0

    return update


def alpha_update_l2_func(
    logalpha,
    infections_mat,
    enode_mat,
    etimes,
    omega,
    T,
    ):
    return alpha_update_l2_funcgrad(
        logalpha,
        infections_mat,
        enode_mat,
        etimes,
        omega,
        T,
        )[0]


def alpha_update_l2_grad(
    logalpha,
    infections_mat,
    enode_mat,
    etimes,
    omega,
    T,
    ):
    return alpha_update_l2_funcgrad(
        logalpha,
        infections_mat,
        enode_mat,
        etimes,
        omega,
        T,
        )[1]


def alpha_update_l2_funcgrad(  # , sparse=False):
    logalpha,
    infections_mat,
    enode_mat,
    etimes,
    omega,
    T,
    K_evaluate,
    ):
    '''
    infections_mat - counts of infections between respective pair of users
    enode_mat - 1-hot representation of which node got infected for each event
    etimes - vector of floatof size N, provide time when the event has occurred
    omega - kernel parameter
    T - upper time limit
    '''

    (R, N) = np.shape(enode_mat)

    alpha = np.exp(logalpha)

    alpha = np.reshape(alpha, (R, R))

    kernel_times = K_evaluate(etimes, T, omega)

    # enode_mat = one_hot(infected_node, D)

    denominator = enode_mat.dot(kernel_times)

    # enode_mat prob needs to be transposed (can be computed this way)

    kernelpart = denominator[:, np.newaxis]

    func_val1 = np.sum(infections_mat.multiply(np.log(alpha)))
    func_val2 = np.sum(kernelpart * alpha)
    func_val3 = np.sum(alpha ** 2)
    func_val = -func_val1 + func_val2 + func_val3

    grad_val1 = infections_mat / alpha
    grad_val2 = kernelpart
    grad_val3 = 2 * alpha
    grad_val = -grad_val1 + grad_val2 + grad_val3

    grad_val = grad_val * alpha
    grad_val = np.asarray(grad_val)
    grad_norm = np.sqrt(np.sum(grad_val ** 2))
    grad_val = grad_val.flatten()

    return (func_val, grad_val, grad_norm)


def alpha_update_l2(  # , sparse=False):
    connections_set,
    infections_mat,
    enode_mat,
    etimes,
    omega,
    T,
    K_evaluate,
    ):
    '''
    infections_mat - counts of infections between respective pair of users
    enode_mat - 1-hot representation of which node got infected for each event
    etimes - vector of floatof size N, provide time when the event has occurred
    omega - kernel parameter
    T - upper time limit
    '''

    kernel_times = K_evaluate(etimes, T, omega)

    # enode_mat = one_hot(infected_node, D)

    denominator = enode_mat.dot(kernel_times)

    # enode_mat prob needs to be transposed (can be computed this way)

    update = infections_mat / denominator[:, np.newaxis]

    (D, D) = np.shape(update)

    if len(connections_set) != 0:
        for i in range(0, D):
            for j in range(0, D):
                if (i, j) in connections_set:
                    continue
                else:
                    a = -2
                    b = -denominator[i]
                    c = infections_mat[i, j]
                    discriminant = b * b - 4 * a * c
                    root1 = (-b + np.sqrt(discriminant)) / (2 * a)
                    root2 = (-b - np.sqrt(discriminant)) / (2 * a)
                    rootmin = min(root1, root2)
                    rootmax = max(root1, root2)
                    if rootmax < 0:
                        root = 0
                    else:
                        if rootmin < 0:
                            root = np.abs(rootmax)
                        else:
                            root = np.abs(rootmin)
                    update[i, j] = root

    update[np.isnan(update)] = 0

    return update


def alpha_update(
    infections_mat,
    node_vec,
    etimes,
    omega,
    T,
    D,
    K_evaluate,
    ):
    '''
    infections_mat - counts of infections between respective pair of users
    etimes - vector of floatof size N, provide time when the event has occurred
    omega - kernel parameter
    T - upper time limit
    '''

    alpha = infections_mat  # np.zeros((D, D))
    kernel_times = K_evaluate(etimes, T, omega)
    for u in range(D):

        # alpha[u, :]=infections_mat[u,:]

        if (alpha[u, :] > 0).any():
            alpha[u, :] /= 1.0 * np.sum(kernel_times[node_vec == u])  # infecting_node==u])
    return alpha


