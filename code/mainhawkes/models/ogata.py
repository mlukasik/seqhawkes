#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Copyright (c) 2014-2015, The University of Sheffield. 
This file is part of the seqhawkes software 
(see https://github.com/mlukasik/seqhawkes), 
and is free software, licenced under the GNU Library General Public License,
Version 2, June 1991 (in the distribution as file LICENSE).

Created on 9 Nov 2015

@author: michal

Ogata thinning algorithm implementations.
'''

import numpy as np
import random
from itertools import izip


def ogata(
    tbegin,
    T,
    K,
    sumIntensities,
    ):
    '''
    Rewritten Ogata's thinning algorithm from the code for the paper: 
    Mixture of Mutually Exciting Processes for Viral Diffusion by Yang and Zha.
    (simhawkes.cpp, simMeme function).
    Need to modify to consider intensrepsity associated with a single meme
    K : denotes number of entitiies, D in case of users or M in case of memes
    '''

    iteration = 0
    t = tbegin  # rng.uniform(0, 1*T/2); // sample start time

    sampledtimes = []
    sampledentities = []

    while t < T:
        iteration += 1
        (_, sumI) = sumIntensities(t)
        t += np.random.exponential(1 / sumI, 1)
        if (iteration + 1) % 10000 == 0:
            print '(' + str(iteration + 1) + ',' + str(t) + ').'
        if t >= T:
            break
        (Is, sumIs) = sumIntensities(t)
        u = random.random()
        if u * sumI < sumIs:
            infected = np.random.choice(range(K), 1,
                    p=np.squeeze(np.asarray(Is / sumIs)))
            sampledtimes.append(float(t))
            sampledentities.append(int(infected))
    print 'total number of sampled points', iteration
    print 'number of accepted samples', len(sampledtimes)
    return (sampledtimes, sampledentities)


def ogata_single_iteration(
    tbegin,
    T,
    K,
    sumIntensities,
    ):
    '''
    Rewritten Ogata's thinning algorithm from Zha code: simhawkes.cpp, simMeme function.
    Need to modify to consider intensity associated with a single meme
    K : denotes number of entitiies, D in case of users or M in case of memes
    '''

    iteration = 0
    t = tbegin  # rng.uniform(0, 1*T/2); // sample start time

    sampledtime = None
    sampledentity = None

    while t < T:
        iteration += 1
        (_, sumI) = sumIntensities(t)
        t += np.random.exponential(1 / sumI, 1)
        if (iteration + 1) % 100 == 0:
            print '(' + str(iteration + 1) + ',' + str(t) + ').'
        if t >= T:
            break
        (Is, sumIs) = sumIntensities(t)
        u = random.random()
        if u * sumI < sumIs:
            infected = np.random.choice(range(K), 1,
                    p=np.squeeze(np.asarray(Is / sumIs)))
            sampledtime = float(t)
            sampledentity = int(infected)
            break
    return (sampledtime, sampledentity)


def ogata_single_iteration_inferringinfecting(
    tbegin,
    T,
    K,
    sumIntensities,
    ):
    '''
    Rewritten Ogata's thinning algorithm from Zha code: simhawkes.cpp, simMeme function.
    Need to modify to consider intensity associated with a single meme
    K : denotes number of entitiies, D in case of users or M in case of memes
    '''

    iteration = 0
    t = tbegin  # rng.uniform(0, 1*T/2); // sample start time

    sampledtime = None
    sampledentity = None
    infectingevent = None

    while t < T:
        iteration += 1
        (_, sumI, _, _, _) = sumIntensities(t)
        t += np.random.exponential(1 / sumI, 1)
        if (iteration + 1) % 100 == 0:
            print '(' + str(iteration + 1) + ',' + str(t) + ').'
        if t >= T:
            break
        (Is, sumIs, mugamma_term, kernelsummation_terms, eventids) = \
            sumIntensities(t)
        u = random.random()
        if u * sumI < sumIs:

            # sample infected user:

            infected = np.random.choice(range(K), 1,
                    p=np.squeeze(np.asarray(Is / sumIs)))

            # sample infecting user:
#             print "np.multiply(*list(kernelsummation_terms)):", np.multiply(*list(kernelsummation_terms))
#             print "mugamma_term[infected]:", mugamma_term[infected]

            weights = \
                np.hstack((np.multiply(*list(kernelsummation_terms))[infected][0],
                          mugamma_term[infected]))

#             print "len(eventids):", len(eventids), "range(len(eventids)+1):", range(len(eventids)+1)

            if len(eventids) == 0:
                infectingevent_idx = 0
            else:
                infectingevent_idx = \
                    np.random.choice(range(len(eventids) + 1), 1,
                        p=np.squeeze(np.asarray(weights
                        / np.sum(weights))))
            if infectingevent_idx == len(eventids):
                infectingevent = -1
            else:
                infectingevent = eventids[infectingevent_idx][0]

            sampledtime = float(t)
            sampledentity = int(infected)
            break
    return (sampledtime, sampledentity, infectingevent)


def ogata_consthazard(lambdafun, x, Tmax):
    lambdaval = lambdafun(x)
    lambdaup = np.max(lambdaval)

    # arrtimes = []

    T = 0
    while T < Tmax:
        intertime = np.random.exponential(1.0 / lambdaup)
        arr = T + intertime
        T = arr
        U = np.random.uniform(0, 1)
        if U < lambdafun(arr) / lambdaup:
            yield T


    # return arrtimes

# TODO: do average over runs: def average_runs_ogata_constanthazard(lambdafun, x, Tmax):

def ogata_reps_consthazard(
    lambdafun,
    x,
    Tmax,
    reps,
    ):

    # Tmax*reps guarantees that when one sequence finishes, all sequences finish, because it goes outside of Tmax interval!

    for arrivaltimes in izip(*[ogata_consthazard(lambdafun, x, Tmax
                             * reps) for _ in range(reps)]):
        arrivaltime = np.mean(arrivaltimes)
        if arrivaltime > Tmax:
            break
        yield arrivaltime


