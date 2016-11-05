#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Copyright (c) 2014-2015, The University of Sheffield. 
This file is part of the seqhawkes software 
(see https://github.com/mlukasik/seqhawkes), 
and is free software, licenced under the GNU Library General Public License,
Version 2, June 1991 (in the distribution as file LICENSE).

Created on 8 Dec 2015

@author: michal

Experiment class for stance classification.
'''

from experiment import Experiment


class ExperimentStanceClassification(Experiment):

    '''
    Experiment class for stance classification.
    '''

    def __init__(
        self,
        fname_data,
        model_constructors,
        train_fraction,
        metrics,
        foldsplitter,
        FOLDTORUN=-1,
        ):

        super(ExperimentStanceClassification, self).__init__(
            fname_data,
            model_constructors,
            0,
            usermetrics=metrics,
            normalize_time_per_meme=True,
            FOLDTORUN=FOLDTORUN,
            )
        self.train_fraction = train_fraction
        self.foldsplitter = foldsplitter

    def _split_train_test(self):
        '''
        Splitting data into train and test parts.
        '''

        for (trainindices, testindices) in \
            self.foldsplitter(self.eventmemes_all,
                              [self.train_fraction]):
            trainetimes = self.etimes_all[trainindices]
            traininfecting_vec = self.infecting_vec_all[trainindices]
            traininfected_vec = self.infected_vec_all[trainindices]
            trainnode_vec = self.node_vec_all[trainindices]
            traineventmemes = self.eventmemes_all[trainindices]
            trainW = self.W_all[trainindices, :]
            trainT = max(trainetimes)
            trainN = len(trainetimes)

            testtimes = self.etimes_all[testindices]
            testinfecting_vec = self.infecting_vec_all[testindices]
            testinfected_vec = self.infected_vec_all[testindices]
            testnode_vec = self.node_vec_all[testindices]
            testeventmemes = self.eventmemes_all[testindices]
            testW = self.W_all[testindices, :]
            testT = max(testtimes)
            testN = len(testtimes)

            yield ((
                trainN,
                trainetimes,
                traininfecting_vec,
                traininfected_vec,
                trainnode_vec,
                traineventmemes,
                trainW,
                trainT,
                ), (
                testN,
                testtimes,
                testinfecting_vec,
                testinfected_vec,
                testnode_vec,
                testeventmemes,
                testW,
                testT,
                ))

    def _split_train_test_loo(self):
        '''
        Leave One Meme Out data split setting.
        '''

        for meme in set(self.eventmemes_all):
            testindices = self.eventmemes_all == meme
            trainindices = self.eventmemes_all != meme

            trainetimes = self.etimes_all[trainindices]
            traininfecting_vec = self.infecting_vec_all[trainindices]
            traininfected_vec = self.infected_vec_all[trainindices]
            trainnode_vec = self.node_vec_all[trainindices]
            traineventmemes = self.eventmemes_all[trainindices]
            trainW = self.W_all[trainindices, :]
            trainT = max(trainetimes)
            trainN = len(trainetimes)

            testtimes = self.etimes_all[testindices]
            testinfecting_vec = self.infecting_vec_all[testindices]
            testinfected_vec = self.infected_vec_all[testindices]
            testnode_vec = self.node_vec_all[testindices]
            testeventmemes = self.eventmemes_all[testindices]
            testW = self.W_all[testindices, :]
            testT = max(testtimes)
            testN = len(testtimes)

            yield ((
                trainN,
                trainetimes,
                traininfecting_vec,
                traininfected_vec,
                trainnode_vec,
                traineventmemes,
                trainW,
                trainT,
                ), (
                testN,
                testtimes,
                testinfecting_vec,
                testinfected_vec,
                testnode_vec,
                testeventmemes,
                testW,
                testT,
                ))

    def evaluate(self):
        '''
        Evaluating models on stance classification experiments.
        '''

        for (modelname, model_results) in self.models.iteritems():
            self._init_results(modelname)
            for (foldind, ((
                trainN,
                trainetimes,
                traininfecting_vec,
                traininfected_vec,
                trainnode_vec,
                traineventmemes,
                trainW,
                trainT,
                ), (
                testN,
                testtimes,
                testinfecting_vec,
                testinfected_vec,
                testnode_vec,
                testeventmemes,
                testW,
                testT,
                ))) in enumerate(self._split_train_test()):

                # print "fold:", foldind

                if self.FOLDTORUN == -1 or self.FOLDTORUN == foldind:
                    print 'Evaluating model:', modelname, 'fold:', \
                        foldind
                    predictednode_vec = \
                        model_results[foldind].evaluate_stance(
                        testN,
                        testtimes,
                        testinfecting_vec,
                        testinfected_vec,
                        testeventmemes,
                        testW,
                        testT,
                        )
                    print '[experiment_stance_classification] predictednode_vec:', \
                        predictednode_vec
                    print 'y_predicted:' + ','.join(map(str,
                            predictednode_vec))
                    print 'y_true:' + ','.join(map(str, testnode_vec))
                    self._append_results_entities(modelname, foldind,
                            testnode_vec, predictednode_vec)


