#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 15 Dec 2015

@author: michal

Functions for analysis the experiment results.
'''

import numpy as np


def method_name_mapper(s):
    if s == 'MostFrequentSingle':
        s = 'MajorityLabel'
    else:
        s = s.replace('Pooled', 'Pooling')
        if 'Pooling' not in s:
            s = s + 'TargetRumourOnly'
    return s


def apply_metric_results_macro_average(results, metric,
        print_full_result=False):
    for method in results.keys():
        max_train = max(results[method].keys())
        for train_perc in sorted(results[method].keys()):
            samples = len(results[method][train_perc])
            if print_full_result:
                print ':'.join(map(str, [train_perc, method])) + ',' \
                    + ','.join(map(lambda x: '{:.2f}'.format(x),
                               [metric(a, b, train_perc=train_perc,
                               max_train=max_train) for (a, b) in
                               results[method][train_perc]]))
            metric_val = ' '.join(map(str, ['%.2f' % np.mean([metric(a,
                                  b, train_perc=train_perc,
                                  max_train=max_train) for (a, b) in
                                  results[method][train_perc]]), "\pm",
                                  '%.2f' % np.std([metric(a, b,
                                  train_perc=train_perc,
                                  max_train=max_train) for (a, b) in
                                  results[method][train_perc]])]))
            results[method][train_perc] = (metric_val, samples)


def apply_metric_results_micro_average(results, metric,
        print_full_result=False):
    for method in results.keys():
        max_train = max(results[method].keys())
        for train_perc in sorted(results[method].keys()):
            samples = len(results[method][train_perc])
            if print_full_result:
                print method, train_perc, " \pm ".join(map(lambda x: \
                        '{:.2f}'.format(x), [100 * metric(a, b,
                        train_perc=train_perc, max_train=max_train)
                        for (a, b) in results[method][train_perc]]))
            metric_val = ' '.join(map(str, ['%.2f'
                                  % metric(reduce(lambda x, y: x + y,
                                  [a for (a, b) in
                                  results[method][train_perc]]),
                                  reduce(lambda x, y: x + y, [b for (a,
                                  b) in results[method][train_perc]]),
                                  train_perc=train_perc,
                                  max_train=max_train), "\pm", '?']))
            results[method][train_perc] = (metric_val, samples)


def display_results_table(results, verbose=False):
    for method in results.keys():
        print 'method:', method
        for train_perc in sorted(results[method].keys()):
            if verbose:
                print train_perc, ':', results[method][train_perc][0], \
                    results[method][train_perc][1]
            else:
                print results[method][train_perc][0]


def display_results_figure(results, METRIC):
    import pylab as pb
    color = iter(pb.cm.rainbow(np.linspace(0, 1, len(results))))
    plots = []
    for method in results.keys():
        x = []
        y = []
        for train_perc in sorted(results[method].keys()):
            x.append(train_perc)
            y.append(results[method][train_perc][0])
        c = next(color)
        (pi, ) = pb.plot(x, y, color=c)
        plots.append(pi)
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('small')
    pb.legend(plots, map(method_name_mapper, results.keys()),
              prop=fontP, bbox_to_anchor=(0.6, .65))
    pb.xlabel('#Tweets from target rumour for training')
    pb.ylabel('Accuracy')
    pb.title(METRIC.__name__)
    pb.savefig('incrementing_training_size.png')


