#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Copyright (c) 2014-2015, The University of Sheffield. 
This file is part of the seqhawkes software 
(see https://github.com/mlukasik/seqhawkes), 
and is free software, licenced under the GNU Library General Public License,
Version 2, June 1991 (in the distribution as file LICENSE).

Created on 11 Apr 2016

@author: michal

Evaluation metrics.
'''

import numpy as np
import sklearn.metrics
from collections import Counter
TMP_LABEL = -111222


def metric_fixed_testset_singlelabel(
    a,
    b,
    train_perc,
    max_train,
    metric,
    label,
    ):
    c = np.array(a)
    c[c != label] = TMP_LABEL
    c[c == label] = 1
    c[c == TMP_LABEL] = 0
    d = np.array(b)
    d[d != label] = TMP_LABEL
    d[d == label] = 1
    d[d == TMP_LABEL] = 0

    return metric_fixed_testset(c, d, train_perc, max_train, metric)


metric_varying_testset = lambda a, b, train_perc, max_train: \
    sklearn.metrics.accuracy_score(a, b)


def metric_fixed_testset(
    a,
    b,
    train_perc,
    max_train,
    metric,
    ):
    start_index = max_train - train_perc
    return metric(a[start_index:], b[start_index:])


# METRIC=metric_fixed_testset

metric_avgf1 = lambda a, b, train_perc, max_train: 100 \
    * np.mean([metric_fixed_testset_singlelabel(
        a,
        b,
        train_perc,
        max_train,
        sklearn.metrics.f1_score,
        LBL,
        ) for LBL in [0, 1, 2, 3]])
metric_avgf1_no14 = lambda a, b, train_perc, max_train: 100 \
    * np.mean([metric_fixed_testset_singlelabel(
        a,
        b,
        train_perc,
        max_train,
        sklearn.metrics.f1_score,
        LBL,
        ) for LBL in [0, 1, 2]])

metric_avgacc = lambda a, b, train_perc, max_train: 100 \
    * np.mean([metric_fixed_testset_singlelabel(
        a,
        b,
        train_perc,
        max_train,
        sklearn.metrics.accuracy_score,
        LBL,
        ) for LBL in [0, 1, 2, 3]])
metric_avgacc_no14 = lambda a, b, train_perc, max_train: 100 \
    * np.mean([metric_fixed_testset_singlelabel(
        a,
        b,
        train_perc,
        max_train,
        sklearn.metrics.accuracy_score,
        LBL,
        ) for LBL in [0, 1, 2]])

metric_f1_0 = lambda a, b, train_perc, max_train: 100 \
    * metric_fixed_testset_singlelabel(
        a,
        b,
        train_perc,
        max_train,
        sklearn.metrics.f1_score,
        0,
        )
metric_f1_1 = lambda a, b, train_perc, max_train: 100 \
    * metric_fixed_testset_singlelabel(
        a,
        b,
        train_perc,
        max_train,
        sklearn.metrics.f1_score,
        1,
        )
metric_f1_2 = lambda a, b, train_perc, max_train: 100 \
    * metric_fixed_testset_singlelabel(
        a,
        b,
        train_perc,
        max_train,
        sklearn.metrics.f1_score,
        2,
        )
metric_f1_3 = lambda a, b, train_perc, max_train: 100 \
    * metric_fixed_testset_singlelabel(
        a,
        b,
        train_perc,
        max_train,
        sklearn.metrics.f1_score,
        3,
        )

metric_avgf1_Qazv = lambda a, b, train_perc, max_train: 100 \
    * np.mean([metric_fixed_testset_singlelabel(
        a,
        b,
        train_perc,
        max_train,
        sklearn.metrics.f1_score,
        LBL,
        ) for LBL in [11, 12, 13, 14]])
metric_avgf1_no14_Qazv = lambda a, b, train_perc, max_train: 100 \
    * np.mean([metric_fixed_testset_singlelabel(
        a,
        b,
        train_perc,
        max_train,
        sklearn.metrics.f1_score,
        LBL,
        ) for LBL in [11, 12, 13]])

metric_avgf1_weighted_Qazv = lambda a, b, train_perc, max_train: 100.0 \
    * np.sum([len(filter(lambda x: x == LBL, b))
             * metric_fixed_testset_singlelabel(
        a,
        b,
        train_perc,
        max_train,
        sklearn.metrics.f1_score,
        LBL,
        ) for LBL in [11, 12, 13, 14]]) / len(b)

metric_avgf1_weighted = lambda a, b, train_perc, max_train: 100.0 \
    * np.sum([len(filter(lambda x: x == LBL, b))
             * metric_fixed_testset_singlelabel(
        a,
        b,
        train_perc,
        max_train,
        sklearn.metrics.f1_score,
        LBL,
        ) for LBL in [0, 1, 2, 3]]) / len(b)

metric_acc = lambda a, b, train_perc, max_train: 100 \
    * metric_fixed_testset(a, b, train_perc, max_train,
                           sklearn.metrics.accuracy_score)
METRICS = [
    metric_acc,
    metric_avgf1,
    metric_avgf1_no14,
    metric_f1_0,
    metric_f1_1,
    metric_f1_2,
    metric_f1_3,
    metric_avgf1_weighted,
    ]

METRICSNAMES = [
    'accuracy',
    'f1 avg11_12_13_14',
    'f1 avg11_12_13',
    'f1 11',
    'f1 12',
    'f1 13',
    'f1 14',
    'f1 weighted',
    ]
