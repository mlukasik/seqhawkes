#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Copyright (c) 2014-2015, The University of Sheffield. 
This file is part of the seqhawkes software 
(see https://github.com/mlukasik/seqhawkes), 
and is free software, licenced under the GNU Library General Public License,
Version 2, June 1991 (in the distribution as file LICENSE).

Created on 15 Dec 2015

@author: michal
'''

import numpy as np
import updates.likelihood_fullsummation
from models.hpseq import HPSeq
from updates.mu_gamma import mu_normalizeconst, gamma_normalizeconst
from updates.alpha_update import alpha_update_linalg
import scipy.sparse as sp


class HPSeqFullSumApproxExactPred(HPSeq):

    '''
    Hawkes Processes for Stance Classification with full summation in intensity function
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
        init_omega=1,
        init_gamma=[],
        init_params_randomly=False,
        kernel_evaluate=updates.kernel_zha.kernel_evaluate,
        K_evaluate=updates.kernel_zha.K_evaluate,
        M=None,
        ):

        super(HPSeqFullSumApproxExactPred, self).__init__(
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
            init_gamma,
            init_params_randomly,
            kernel_evaluate,
            K_evaluate,
            M=M,
            )
        print '[HPSeqFullSumApproxExactPred] self.M=', self.M, \
            'self.gamma.shape=', self.gamma.shape, 'init_gamma=', \
            init_gamma
        self.raw_meme_updates = np.zeros((self.M, 1))
        for meme in self.eventmemes:
            self.raw_meme_updates[int(meme)] += 1
        self.raw_node_updates = np.zeros((D, 1))
        for node in self.node_vec:
            self.raw_node_updates[int(node)] += 1

        print 'creating infections'
        self.infections_mat = sp.lil_matrix((D, D), dtype=np.int)
        for meme in set(self.eventmemes):
            observed_nodes_cnt = {}
            for node1 in node_vec[self.eventmemes == meme]:
                for (node_before, cnt) in \
                    observed_nodes_cnt.iteritems():
                    self.infections_mat[node_before, node1] += cnt
                observed_nodes_cnt[node1] = \
                    observed_nodes_cnt.get(node1, 0) + 1
        print 'end creating infections'

    def _gamma_update_core(self):
        return 1.0 * self.raw_meme_updates \
            / gamma_normalizeconst(self.T, self.mu)

    def _mu_update_core(self):
        return 1.0 * self.raw_node_updates / mu_normalizeconst(self.T,
                self.gamma)

    def _mu_update(self):
        return self._generic_update(lambda : \
                                    self._mu_update_core().flatten(),
                                    'mu', self.mu)

    def _gamma_update(self):
        return self._generic_update(lambda : \
                                    self._gamma_update_core().flatten(),
                                    'gamma', self.gamma)

    def _alpha_update(self):
        return self._generic_update(lambda : alpha_update_linalg(
                self.infections_mat,
                self.enode_mat_sparse,
                self.etimes,
                self.omega,
                self.T,
                self.K_evaluate,
                ), 'alpha', self.alpha)

    def evaluate_likelihood(self):
        return updates.likelihood_fullsummation.evaluate_approximated_loglikelihood_wordsbyusers(
            self.node_vec,
            self.eventmemes,
            self.etimes,
            self.T,
            self.mu,
            self.gamma,
            self.alpha,
            self.omega,
            self.W,
            self.beta,
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
        return updates.likelihood_fullsummation.event_nonapproximated_logintensity(
            label,
            meme,
            t,
            t,
            self.etimes,
            self.node_vec,
            self.eventmemes,
            self.mu,
            self.gamma,
            self.omega,
            self.alpha,
            self.kernel_evaluate,
            print_summed_elements,
            )


class HPSeqFullSumApprox(HPSeqFullSumApproxExactPred):

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
        return updates.likelihood_fullsummation.event_approximated_logintensity(
            label,
            meme,
            t,
            t,
            self.etimes,
            self.node_vec,
            self.eventmemes,
            self.mu,
            self.gamma,
            self.omega,
            self.alpha,
            self.kernel_evaluate,
            print_summed_elements,
            )


