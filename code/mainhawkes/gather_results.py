#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 15 Feb 2015

@author: michal

Gathering results from the result text files.
'''

import sys
from os import listdir
from os.path import join
from utils.postprocessing_stance import display_results_table, \
    apply_metric_results_micro_average, \
    apply_metric_results_macro_average
from models.metrics import METRICS, METRICSNAMES

print_full_result = False
verbose_table = True
if __name__ == '__main__':
    experiment_dir = sys.argv[1]
    setting = sys.argv[2]
    for (ind, METRIC) in enumerate(METRICS):
        results = {}
        method2foldresults = {}
        for fname_short in listdir(experiment_dir):
            fname = join(experiment_dir, fname_short)
            (fold, method, train_perc) = fname_short.split('_')
            train_perc = int(train_perc)
            fold = int(fold)

            ypred = None
            ytrue = None
            with open(fname) as f:
                lines = f.readlines()
                for l in lines:
                    if 'y_predicted' in l:
                        ypred = map(int, l.replace('y_predicted:', ''
                                    ).strip().replace('.0', ''
                                    ).split(','))
                    if 'y_true' in l:
                        ytrue = map(int, l.replace('y_true:', ''
                                    ).strip().replace('.0', ''
                                    ).split(','))

            if ypred == None or ytrue == None:
                print 'Problem with parsing:', fname
            else:
                results[method] = results.get(method, {})
                results[method][train_perc] = \
                    results[method].get(train_perc, []) + [(fold,
                        (ypred, ytrue))]
        for key1 in results.keys():
            for key2 in results[key1].keys():
                results[key1][key2] = sorted(results[key1][key2],
                        key=lambda x: x[0])
                results[key1][key2] = map(lambda x: x[1],
                        results[key1][key2])

        print '->metric:', METRICSNAMES[ind]
        if setting == 'micro':
            print 'micro average:'
            apply_metric_results_micro_average(results, METRIC,
                    print_full_result)
            display_results_table(results, verbose_table)
        if setting == 'macro':
            print 'macro average:'
            apply_metric_results_macro_average(results, METRIC,
                    print_full_result)
            display_results_table(results, verbose_table)
        if setting == 'finegrained':
            apply_metric_results_macro_average(results, METRIC,
                    print_full_result=True)
