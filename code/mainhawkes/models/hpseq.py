#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Copyright (c) 2014-2015, The University of Sheffield. 
This file is part of the seqhawkes software 
(see https://github.com/mlukasik/seqhawkes), 
and is free software, licenced under the GNU Library General Public License,
Version 2, June 1991 (in the distribution as file LICENSE).

Created on 7 Dec 2015

@author: michal

Hawkes Process model for sequence classification.
'''

import numpy as np
from models.hp import HawkessProcessModel
import updates.likelihood_observedu


def display_intensities_per_label(
    hpseq,
    testtimes,
    reslist,
    meme,
    intermediate_pnts=20,
    ):
    print '[display_intensities_per_label] displaying the intensity values'

    # set to -1 because we will only use it in the full summation case

    testinfected = -1
    testinfecting = -1
    for label in set(hpseq.node_vec):
        print 'label=', label
        for (prevtidx, tweett) in enumerate(testtimes[1:]):
            for intermt in (np.linspace(0, tweett
                            - testtimes[prevtidx],
                            num=intermediate_pnts)[::-1])[1:]:
                print tweett - intermt, hpseq._event_loglikelihood(
                    label,
                    testinfected,
                    testinfecting,
                    meme,
                    tweett - intermt,
                    hpseq.T,
                    hpseq.W[0, :],
                    print_summed_elements=False,
                    )[1], int(reslist[prevtidx] == label and tweett
                              - intermt == testtimes[prevtidx]
                              or reslist[prevtidx + 1] == label
                              and tweett - intermt
                              == testtimes[prevtidx + 1])


class HPSeq(HawkessProcessModel):

    '''
    Hawkes Process for Stance Classification
    
    The difference from the regular Hawkes is that beta is generating words conditioned on the user 
    (a label in the new interpretation) rather than on the meme.
    '''

    def __init__(
        self,
        etimes,
        node_vec,
        ememes,
        infected_vec,
        infecting_vec,
        W,
        T,
        V,
        D,
        updates,
        iterations,
        verbose,
        init_omega=0.1,
        init_gamma=[],
        init_params_randomly=False,
        kernel_evaluate=updates.kernel_zha.kernel_evaluate,
        K_evaluate=updates.kernel_zha.K_evaluate,
        M=None,
        ):

        super(HPSeq, self).__init__(
            etimes,
            node_vec,
            ememes,
            infected_vec,
            infecting_vec,
            W,
            T,
            V,
            D,
            updates,
            iterations,
            verbose,
            init_omega,
            init_params_randomly,
            kernel_evaluate,
            K_evaluate,
            M=M,
            )
        print '[HPSeq] self.M=', self.M, 'self.gamma.shape=', \
            self.gamma.shape, 'init_gamma=', init_gamma
        if init_gamma != []:
            self.gamma = init_gamma
        if init_params_randomly:
            self.beta = np.array([np.random.random() for _ in
                                 range(self.D * self.V)])
        else:
            self.beta = np.array([1. for _ in range(self.D * self.V)])
        self.beta = self.beta.reshape(self.D, self.V)
        self._normalize_beta_rowwise()

    def _run_one_time_updates(self):
        if 'beta' in self.updates:
            self.beta = self._beta_update()
        if 'betatfidf' in self.updates:
            self.beta = self._betatfidf_update()

    def _betatfidf_update(self):
        return self._generic_update(lambda : \
                                    self._beta_update_raw_tfidf(),
                                    'beta', self.beta)

    def _beta_update_raw(self):
        '''
        Run only once - it does not depend on other parameters.
        '''

        for nodeid in xrange(self.D):
            self.beta[nodeid] = self.W[self.node_vec == nodeid, :
                    ].sum(axis=0)

        # Laplace smoothing to avoid zeros!

        self.beta += 1
        self._normalize_beta_rowwise()
        return self.beta

    def _beta_update_raw_tfidf(self):
        '''
        Run only once - it does not depend on other parameters.
        '''

        for nodeid in xrange(self.D):
            self.beta[nodeid] = self.W[self.node_vec == nodeid, :
                    ].sum(axis=0)
        for nodeid in xrange(self.D):
            for wordid in xrange(self.beta.shape[1]):
                docs_cnt = np.sum(self.W[self.node_vec == nodeid,
                                  wordid] >= 1)
                docs_cnt += 1  # smooth by adding one
                self.beta[nodeid][wordid] *= 1 + np.log(self.W.shape[0]
                        * 1. / docs_cnt)  # 1+ because we still want to keep words which always occurr, but probably it never happens

        # Laplace smoothing to avoid zeros!

        self.beta += 1
        self._normalize_beta_rowwise()
        return self.beta

    def evaluate_likelihood(self):
        return updates.likelihood_observedu.evaluate_approximated_loglikelihood_wordsbyusers(
            self.infecting_vec,
            self.infected_vec,
            self.node_vec,
            self.eventmemes,
            self.etimes,
            self.infecting_node_vec,
            self.T,
            self.mu,
            self.gamma,
            self.alpha,
            self.omega,
            self.W,
            self.beta,
            self.kernel_evaluate,
            )

    def evaluate_pseudologlikelihood_wordsbyusers_meme(self, meme):
        return updates.likelihood_observedu.evaluate_pseudologlikelihood_wordsbyusers_meme(
            self.infecting_vec,
            self.infected_vec,
            self.node_vec,
            self.eventmemes,
            self.etimes,
            self.infecting_node_vec,
            self.T,
            self.mu,
            self.gamma,
            self.alpha,
            self.omega,
            self.W,
            self.beta,
            meme,
            self.kernel_evaluate,
            self.K_evaluate,
            )

    def _event_logintensity(
        self,
        label,
        infecting_u,
        infected_e,
        infecting_e,
        meme,
        t,
        T,
        print_summed_elements=False,
        ):
        return updates.likelihood_observedu.event_approximated_logintensity(
            label,
            infecting_u,
            infected_e,
            infecting_e,
            meme,
            t,
            T,
            self.etimes,
            self.infected_vec,
            self.mu,
            self.gamma,
            self.omega,
            self.alpha,
            self.kernel_evaluate,
            print_summed_elements,
            )

    def _iterate_updating_model(
        self,
        testN,
        testtimes,
        testinfecting_vec,
        testinfected_vec,
        testeventmemes,
        testW,
        testT,
        action,
        get_label,
        restore_original_training_data=True,
        final_action_before_restoring_data=False,
        ):

        print '[HPSeq._iterate_updating_model] START'
        meme = testeventmemes[0]

        (
            etimes,
            node_vec,
            eventmemes,
            infected_vec,
            infecting_vec,
            W,
            T,
            V,
            D,
            ) = (
            self.etimes,
            self.node_vec,
            self.eventmemes,
            self.infected_vec,
            self.infecting_vec,
            self.W,
            self.T,
            self.V,
            self.D,
            )
        reslist = []
        for next_event_index in xrange(len(testtimes)):
            if next_event_index % 10 == 0:
                print '[_iterate_updating_model] considering event', \
                    next_event_index
            t = testtimes[next_event_index]
            words = testW[next_event_index, :]

            if testinfected_vec[next_event_index] \
                == testinfecting_vec[next_event_index]:
                infecting_u = self.node_vec[next_event_index]
            else:
                infecting_u = \
                    self.eventid_to_node[testinfecting_vec[next_event_index]]
            res = action(
                infecting_u,
                testinfected_vec[next_event_index],
                testinfecting_vec[next_event_index],
                meme,
                t,
                t,
                words,
                )
            yield res
            reslist.append(res)

            self._set_training_data(  # np.vstack((self.W, words)),
                np.append(self.etimes, t),
                np.append(self.node_vec, get_label(
                    infecting_u,
                    testinfected_vec[next_event_index],
                    testinfecting_vec[next_event_index],
                    meme,
                    t,
                    t,
                    words,
                    res,
                    )),
                np.append(self.eventmemes, meme),
                np.append(self.infected_vec,
                          testinfected_vec[next_event_index]),
                np.append(self.infecting_vec,
                          testinfecting_vec[next_event_index]),
                self.W,
                t,
                self.V,
                self.D,
                )

        print 'final_action_before_restoring_data:', \
            final_action_before_restoring_data
        if final_action_before_restoring_data:
            print 'Passed condition: if final_action_before_restoring_data'
            final_action_before_restoring_data(self, testtimes,
                    reslist, meme)

        if restore_original_training_data:
            self._set_training_data(
                etimes,
                node_vec,
                eventmemes,
                infected_vec,
                infecting_vec,
                W,
                T,
                V,
                D,
                )

    def yield_intensity_per_time_update(
        self,
        testN,
        testtimes,
        testinfecting_vec,
        testinfected_vec,
        testeventmemes,
        testW,
        testT,
        label,
        ):
        for res in self._iterate_updating_model(
            testN,
            testtimes,
            testinfecting_vec,
            testinfected_vec,
            testeventmemes,
            testW,
            testT,
            action=lambda infecting_u, testinfected, testinfecting, \
                meme, t, T, words: (self._event_logintensity(
                    label,
                    infecting_u,
                    testinfected,
                    testinfecting,
                    meme,
                    t,
                    T,
                    ), t),
            get_label=lambda infecting_u, testinfected, testinfecting, \
                meme, t, T, words, res: self._evaluate_stance_action(
                    infecting_u,
                    testinfected,
                    testinfecting,
                    meme,
                    t,
                    T,
                    words,
                    ),
            ):
            yield res

    def yield_intensities_per_time_update(
        self,
        testN,
        testtimes,
        testinfecting_vec,
        testinfected_vec,
        testeventmemes,
        testW,
        testT,
        labels,
        restore_original_training_data=True,
        ):
        for res in self._iterate_updating_model(
            testN,
            testtimes,
            testinfecting_vec,
            testinfected_vec,
            testeventmemes,
            testW,
            testT,
            action=lambda infecting_u, testinfected, testinfecting, \
                meme, t, T, words: (tuple([self._event_logintensity(
                    label,
                    infecting_u,
                    testinfected,
                    testinfecting,
                    meme,
                    t,
                    T,
                    ) for label in labels]), t),
            get_label=lambda infecting_u, testinfected, testinfecting, \
                meme, t, T, words, res: self._evaluate_stance_action(
                    infecting_u,
                    testinfected,
                    testinfecting,
                    meme,
                    t,
                    T,
                    words,
                    ),
            restore_original_training_data=restore_original_training_data,
            ):
            yield res

    def _event_loglikelihood(
        self,
        label,
        testinfected,
        testinfecting,
        meme,
        t,
        T,
        words,
        print_summed_elements=True,
        ):
        loglikelihood_term = 0
        if testinfected == testinfecting:
            infecting_u = label
        else:
            infecting_u = self.eventid_to_node[testinfecting]
        logintensity = self._event_logintensity(
            label,
            infecting_u,
            testinfected,
            testinfecting,
            meme,
            t,
            T,
            print_summed_elements=print_summed_elements,
            )
        loglikelihood_term += logintensity
        text_term = np.dot(words, np.log(self.beta[label, :]))
        loglikelihood_term += text_term
        return (loglikelihood_term, logintensity, text_term)

    def _evaluate_stance_action_full_info(
        self,
        infecting_u,
        testinfected,
        testinfecting,
        meme,
        t,
        T,
        words,
        verbose=True,
        ):
        predictions = []
        if verbose:
            print '[HPSeq._evaluate_stance_action] considering tweet at t:', \
                t
        for label in set(self.node_vec):
            (loglikelihood_term, logintensity, text_term) = \
                self._event_loglikelihood(
                label,
                testinfected,
                testinfecting,
                meme,
                t,
                T,
                words,
                print_summed_elements=verbose,
                )
            predictions.append((label, loglikelihood_term))
            if verbose:
                print '[HPSeq._evaluate_stance_action] label:', label, \
                    'lambda=%8.2f lang=%8.2f all=%8.2f ' \
                    % (logintensity, text_term, loglikelihood_term)
        if verbose:
            print '[HPSeq._evaluate_stance_action] chosen label:', \
                max(predictions, key=lambda x: x[1])[0]
        return predictions

    def _evaluate_stance_action(
        self,
        infecting_u,
        testinfected,
        testinfecting,
        meme,
        t,
        T,
        words,
        ):
        predictions = self._evaluate_stance_action_full_info(
            infecting_u,
            testinfected,
            testinfecting,
            meme,
            t,
            T,
            words,
            )
        return max(predictions, key=lambda x: x[1])[0]

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
        return list(self._iterate_updating_model(
            testN,
            testtimes,
            testinfecting_vec,
            testinfected_vec,
            testeventmemes,
            testW,
            testT,
            action=self._evaluate_stance_action,
            get_label=lambda infecting_u, testinfected, testinfecting, \
                meme, t, T, words, res: res,
            final_action_before_restoring_data=display_intensities_per_label,
            ))


