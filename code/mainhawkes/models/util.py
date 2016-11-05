#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Copyright (c) 2014-2015, The University of Sheffield. 
This file is part of the seqhawkes software 
(see https://github.com/mlukasik/seqhawkes), 
and is free software, licenced under the GNU Library General Public License,
Version 2, June 1991 (in the distribution as file LICENSE).

Created on 3 Dec 2015

@author: michal

Utility functions for models.
'''

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dcmp(x, eps):
    '''
    From Zha code in common.h
    Utility re-written from c++ by Yang et al 2013
    '''

    if x < -eps:
        return -1
    else:
        return x > eps


def sample(d, s):
    '''
    Rewritten from simhawkes.cpp
    Utility re-written from c++ by Yang et al 2013
    '''

    u = np.random.uniform(0, s)
    i = 0
    while dcmp(u - d[i]) > 0 and i < len(d):
        i += 1
        u -= d[i]
    if i >= len(d):
        return len(d) - 1
    else:
        return i


def g(dt, omega):
    return omega * np.exp(-omega * dt)


def mintensity(
    t,
    d,
    m,
    mu,
    gamma,
    alpha,
    node_vec,
    etimes,
    ):
    res = mu[d] * gamma[m]
    for (ind, etime) in enumerate(etimes):

        # model.alpha contains Alpha matrix, where consecutive blocks of D values in model.alpha
        # (e.g. model.alpha[i*D1:(i+1)*D]) correspond to columns of values in Alpha (Alpha_{dot, i});
        # res+=self.alpha[d,self.node_vec[ind]]*self.g(t-etime); check if this or the next one is correct

        res += alpha[node_vec[ind], d] * g(t - etime)
    return res


def update_summary(
    var_up,
    var,
    start,
    end,
    ):
    diff = np.abs(var_up - var)
    reldiff = diff / var

    # filter out nan's

    try:
        reldiff = reldiff[~np.isnan(reldiff)]
    except:
        pass
    return (np.mean(diff), np.std(diff), np.mean(reldiff),
            np.std(reldiff), (end - start).microseconds)


