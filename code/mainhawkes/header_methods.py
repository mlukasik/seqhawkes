#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 15 Dec 2015

@author: michal

List of methods for experiments.

'''

import numpy as np
from models.hpseq_approxsummation import HPSeqFullSumApprox
from models.hpseq_baselines import HPSeqMajorityVote, HPSeqNB, \
    HPSeqBetaOnly
import models.hpseq_gradconstr_initfromapprox


def get_methods():
    ITERATIONS = 5
    model_constructors = [('MajorityClassifier', lambda etimes, \
                          node_vec, eventmemes, infected_vec, \
                          infecting_vec, W, T, V, D: HPSeqMajorityVote(
            etimes,
            node_vec,
            eventmemes,
            infected_vec,
            infecting_vec,
            W,
            T,
            V,
            D,
            [],
            0,
            verbose=False,
            )), ('NaiveBayes', lambda etimes, node_vec, eventmemes, \
                 infected_vec, infecting_vec, W, T, V, D: HPSeqNB(
            etimes,
            node_vec,
            eventmemes,
            infected_vec,
            infecting_vec,
            W,
            T,
            V,
            D,
            ['beta'],
            ITERATIONS,
            verbose=False,
            )), ('LanguageModel', lambda etimes, node_vec, eventmemes, \
                 infected_vec, infecting_vec, W, T, V, D: HPSeqBetaOnly(
            etimes,
            node_vec,
            eventmemes,
            infected_vec,
            infecting_vec,
            W,
            T,
            V,
            D,
            ['beta'],
            ITERATIONS,
            verbose=False,
            )), ('HPFullSummationApproxIntensityPredictionomega01',
                 lambda etimes, node_vec, eventmemes, infected_vec, \
                 infecting_vec, W, T, V, D: HPSeqFullSumApprox(
            etimes,
            node_vec,
            eventmemes,
            infected_vec,
            infecting_vec,
            W,
            T,
            V,
            D,
            ['mu', 'alpha', 'beta'],
            ITERATIONS,
            verbose=False,
            init_omega=0.1,
            init_gamma=np.array([1 for _ in xrange(len(set(eventmemes))
                                + 1)]),
            init_params_randomly=False,
            )), ('HPSeqFullSumGradConstrInitFromApproxomega01',
                 lambda etimes, node_vec, eventmemes, infected_vec, \
                 infecting_vec, W, T, V, D: \
                 models.hpseq_gradconstr_initfromapprox.HPSeqFullSumGradConstrInitFromApprox(
            etimes,
            node_vec,
            eventmemes,
            infected_vec,
            infecting_vec,
            W,
            T,
            V,
            D,
            ['beta', 'allnoomega'],
            1,
            verbose=False,
            init_omega=0.1,
            init_gamma=np.array([1 for _ in xrange(len(set(eventmemes))
                                + 1)]),
            init_params_randomly=False,
            ))]
    return model_constructors


def get_methodnames():
    return map(lambda x: x[0], get_methods())


