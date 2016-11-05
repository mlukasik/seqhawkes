#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Copyright (c) 2014-2015, The University of Sheffield. 
This file is part of the seqhawkes software 
(see https://github.com/mlukasik/seqhawkes), 
and is free software, licenced under the GNU Library General Public License,
Version 2, June 1991 (in the distribution as file LICENSE).

Created on 4 May 2015

@author: michal, srijith

The main script launching complete stance classification experiment.
'''

import sys
from experiment.experiment_stance_classif import ExperimentStanceClassification
import sklearn.metrics
from header_methods import get_methods
from utils.qsub import LOCALLY
from experiment.utils import foldsplitter
from utils.utils import print_matrix, make_dir
from models.hp import HawkessProcessModel

if __name__ == '__main__':
    if LOCALLY():

        # useful for debugging on a local machine

        FOLDTORUN = 0
        methodname = 'MajorityClassifier'
        train_set_ratio = 0
        fname_data = '../../data/sydney.txt'
        DO_TRAIN = True
        DO_PLOT = False
    else:
        FOLDTORUN = int(sys.argv[1])
        methodname = sys.argv[2]
        train_set_ratio = int(sys.argv[3])
        fname_data = sys.argv[4]
        DO_TRAIN = True
        DO_PLOT = False

    metrics = [('ACCURACY', sklearn.metrics.accuracy_score)]

    model_constructors = get_methods()
    if methodname != None and methodname != 'ALL':

        # if we are interested in keeping only one method

        model_constructors = filter(lambda x: x[0] == methodname,
                                    model_constructors)

    exp = ExperimentStanceClassification(
        fname_data,
        model_constructors,
        train_set_ratio,
        metrics,
        foldsplitter,
        FOLDTORUN=FOLDTORUN,
        )
    exp.build_models(train=DO_TRAIN)

    for modelname in map(lambda x: x[0], model_constructors):
        for fold in exp.models[modelname].keys():
            if issubclass(type(exp.models[modelname][fold]),
                          HawkessProcessModel):
                print modelname, 'omega:', \
                    exp.models[modelname][fold].omega
                print modelname, 'beta:'
                print print_matrix(exp.models[modelname][fold].beta)
                print modelname, 'alpha:'
                print print_matrix(exp.models[modelname][fold].alpha)
                print modelname, 'mu:'
                print print_matrix(exp.models[modelname][fold].mu)

    if DO_PLOT:
        dirname = 'pictures_realtweetpred' + methodname + 'Train' \
            + str(DO_TRAIN)
        make_dir(dirname)
        exp.plot_intensities_wordlikelihoods(dirname)

    exp.evaluate()
    exp.summarize()
