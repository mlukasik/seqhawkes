#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Copyright (c) 2014-2015, The University of Sheffield. 
This file is part of the seqhawkes software 
(see https://github.com/mlukasik/seqhawkes), 
and is free software, licenced under the GNU Library General Public License,
Version 2, June 1991 (in the distribution as file LICENSE).

Created on 21 Dec 2015

@author: michal

Baselines which are special cases of the HP model.
'''

import numpy as np
from models.hpseq import HPSeq


class HPSeqMajorityVote(HPSeq):

    def _find_most_frequent(self):
        if not hasattr(self, 'most_frequent_node'):
            from collections import Counter
            c = Counter(self.node_vec)
            self.most_frequent_node = c.most_common(1)[0][0]
        return self.most_frequent_node

    def _intensityUserMeme(
        self,
        t,
        d,
        m,
        filterlatertimes=False,
        ):
        if d == self._find_most_frequent():
            return 1
        else:
            return 0

    def evaluate_stance(
        self,
        testN,
        testtimes,
        testinfecting_vec,
        testinfected_vec,
        testeventmemes,
        testW,
        testT,
        ):
        label = self._find_most_frequent()
        predictednode_vec = [label for _ in xrange(len(testtimes))]
        return predictednode_vec


class HPSeqNB(HPSeq):

    def evaluate_stance(
        self,
        testN,
        testtimes,
        testinfecting_vec,
        testinfected_vec,
        testeventmemes,
        testW,
        testT,
        ):
        predictednode_vec = []

        for next_event_index in xrange(len(testtimes)):
            print 'considering event', next_event_index
            words = testW[next_event_index, :]

            predictions = []
            for label in set(self.node_vec):
                loglikelihood_term = 0
                loglikelihood_term += np.dot(words,
                        np.log(self.beta[label, :]))
                loglikelihood_term += np.log(1.0 * np.sum(self.node_vec
                        == label) / len(self.node_vec))
                predictions.append((label, loglikelihood_term))
            predictednode_vec.append(max(predictions, key=lambda x: \
                    x[1])[0])
        return predictednode_vec


class HPSeqBetaOnly(HPSeq):

    def evaluate_stance(
        self,
        testN,
        testtimes,
        testinfecting_vec,
        testinfected_vec,
        testeventmemes,
        testW,
        testT,
        ):
        predictednode_vec = []

        for next_event_index in xrange(len(testtimes)):
            print 'considering event', next_event_index
            words = testW[next_event_index, :]

            predictions = []
            for label in set(self.node_vec):
                loglikelihood_term = 0
                loglikelihood_term += np.dot(words,
                        np.log(self.beta[label, :]))
                predictions.append((label, loglikelihood_term))
            predictednode_vec.append(max(predictions, key=lambda x: \
                    x[1])[0])
        return predictednode_vec


