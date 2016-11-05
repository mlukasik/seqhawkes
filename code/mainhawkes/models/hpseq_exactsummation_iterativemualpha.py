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
from updates.mu_gamma import gamma_normalizeconst, gamma_fullsum_func, \
    gamma_fullsum_grad
from scipy.optimize import minimize
from scipy.optimize import check_grad
from updates.mu_fullsum_nonapprox import mu_fullsum_func, \
    mu_fullsum_grad, mu_fullsum_funcgrad, logmu_fullsum_func, \
    logmu_fullsum_grad, logmu_fullsum_funcgrad
from models.hpseq_approxsummation import HPSeqFullSumApproxExactPred
import updates.alpha_fullsum_nonapprox_iter


class HPSeqFullSumGrad(HPSeqFullSumApproxExactPred):

    '''
    Gradient based optimization of the parameters.
    '''

    def _gamma_update_core(self):
        gamma = self.gamma
        err = check_grad(
            gamma_fullsum_func,
            gamma_fullsum_grad,
            gamma,
            self.node_vec,
            self.eventmemes,
            self.etimes,
            self.T,
            self.mu,
            self.alpha,
            self.omega,
            self.W,
            self.beta,
            self.kernel_evaluate,
            self.K_evaluate,
            )
        print 'gradient error ', err
        optout = minimize(
            gamma_fullsum_grad,
            gamma,
            (
                self.node_vec,
                self.eventmemes,
                self.etimes,
                self.T,
                self.mu,
                self.alpha,
                self.omega,
                self.W,
                self.beta,
                self.kernel_evaluate,
                self.K_evaluate,
                ),
            method='BFGS',
            jac=True,
            options={'disp': True},
            )
        return float(optout.x)

    def _mu_update_core(self):
        print '[HPSeqFullSumGrad] _mu_update_core'
        mu = self.mu
        for (index, mui) in enumerate(mu):
            print 'mui:', mui
            err = check_grad(
                mu_fullsum_func,
                mu_fullsum_grad,
                [mui],
                index,
                mu,
                self.node_vec,
                self.eventmemes,
                self.etimes,
                self.T,
                self.gamma,
                self.alpha,
                self.omega,
                self.W,
                self.beta,
                self.kernel_evaluate,
                self.K_evaluate,
                )
            print 'gradient error ', err
        new_mu = []
        for (index, mui) in enumerate(mu):
            optout = minimize(
                mu_fullsum_funcgrad,
                mui,
                (
                    index,
                    mu,
                    self.node_vec,
                    self.eventmemes,
                    self.etimes,
                    self.T,
                    self.gamma,
                    self.alpha,
                    self.omega,
                    self.W,
                    self.beta,
                    self.kernel_evaluate,
                    self.K_evaluate,
                    ),
                method='BFGS',
                jac=True,
                options={'disp': True},
                )
            new_mu.append(float(optout.x))
            mu[index] = float(optout.x)  # should we update the mu already?
        return np.array(new_mu)

    def _alpha_update_core(self):
        return 1.0 * self.raw_meme_updates \
            / gamma_normalizeconst(self.T, self.mu)

    def _mu_update(self):
        return self._generic_update(lambda : \
                                    self._mu_update_core().flatten(),
                                    'mu', self.mu)

    def _gamma_update(self):
        return self._generic_update(lambda : \
                                    self._gamma_update_core().flatten(),
                                    'gamma', self.gamma)

    def _alpha_update(self):
        return self._generic_update(lambda : self._alpha_update_core(),
                                    'alpha', self.alpha)

    def evaluate_likelihood(self):
        return updates.likelihood_fullsummation.evaluate_nonapproximated_loglikelihood_wordsbyusers(
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
            )


class HPSeqFullSumGradConstr(HPSeqFullSumGrad):

    '''
    Gradient based optimization of the parameters.
    '''

    def _mu_update_core(self):
        print '[HPSeqFullSumGradConstr] _mu_update_core'
        logmu = np.log(self.mu)
        for (index, logmui) in enumerate(logmu):
            print 'logmui:', logmui
            err = check_grad(
                logmu_fullsum_func,
                logmu_fullsum_grad,
                [logmui],
                index,
                logmu,
                self.node_vec,
                self.eventmemes,
                self.etimes,
                self.T,
                self.gamma,
                self.alpha,
                self.omega,
                self.W,
                self.beta,
                self.kernel_evaluate,
                self.K_evaluate,
                )
            print 'gradient error ', err
        new_logmu = []
        optout = minimize(
            logmu_fullsum_funcgrad,
            logmui,
            (
                index,
                logmu,
                self.node_vec,
                self.eventmemes,
                self.etimes,
                self.T,
                self.gamma,
                self.alpha,
                self.omega,
                self.W,
                self.beta,
                self.kernel_evaluate,
                self.K_evaluate,
                ),
            method='L-BFGS-B',
            jac=True,
            options={'disp': True},
            )
        new_mu = np.exp(np.array(new_logmu))
        return np.array(new_mu)

    def _alpha_update_core(self):
        print '[HPSeqFullSumGradConstr] _alpha_update_core'
        logalpha = np.log(self.alpha)
        for (index1, _) in enumerate(logalpha):
            for (index2, _) in enumerate(logalpha[index1]):
                logalphaij = logalpha[index1][index2]
                print 'logmalphaij:', logalphaij
                err = check_grad(
                    updates.alpha_fullsum_nonapprox_iter.logalpha_fullsum_func,
                    updates.alpha_fullsum_nonapprox_iter.logalpha_fullsum_grad,
                    [logalphaij],
                    (index1, index2),
                    logalpha,
                    self.node_vec,
                    self.eventmemes,
                    self.etimes,
                    self.T,
                    self.mu,
                    self.gamma,
                    self.omega,
                    self.W,
                    self.beta,
                    self.kernel_evaluate,
                    self.K_evaluate,
                    )
                print 'gradient error ', err

        new_logalpha = [[0 for (index2, _) in
                        enumerate(logalpha[index1])] for (index1, _) in
                        enumerate(logalpha)]

        for (index1, _) in enumerate(logalpha):
            for (index2, _) in enumerate(logalpha[index1]):
                optout = minimize(
                    updates.alpha_fullsum_nonapprox_iter.logalpha_fullsum_funcgrad,
                    logalpha[index1][index2],
                    (
                        (index1, index2),
                        logalpha,
                        self.node_vec,
                        self.eventmemes,
                        self.etimes,
                        self.T,
                        self.mu,
                        self.gamma,
                        self.omega,
                        self.W,
                        self.beta,
                        self.kernel_evaluate,
                        self.K_evaluate,
                        ),
                    method='L-BFGS-B',
                    jac=True,
                    options={'disp': True},
                    )
                new_logalpha[index1][index2] = float(optout.x)

        new_alpha = np.exp(np.matrix(new_logalpha))
        return np.array(new_alpha)

    def _mu_update(self):
        return self._generic_update(lambda : \
                                    self._mu_update_core().flatten(),
                                    'mu', self.mu)

    def _gamma_update(self):
        return self._generic_update(lambda : \
                                    self._gamma_update_core().flatten(),
                                    'gamma', self.gamma)

    def _alpha_update(self):
        return self._generic_update(lambda : self._alpha_update_core(),
                                    'alpha', self.alpha)

    def evaluate_likelihood(self):
        return updates.likelihood_fullsummation.evaluate_nonapproximated_loglikelihood_wordsbyusers(
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
            )


