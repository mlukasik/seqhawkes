#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 15 Dec 2015

@author: michal

Hawkes Process model for sequence classification 
- gradient based optimization of the parameters.
'''

import numpy as np
import updates.likelihood_fullsummation
from scipy.optimize import minimize
from scipy.optimize import check_grad
import updates.mu_fullsum_nonapprox
import updates.gamma_fullsum_nonapprox
import updates.alpha_fullsum_nonapprox
import updates.mualphaupdates_fullsum_nonapprox
from models.hpseq import HPSeq


class HPSeqFullSumGradConstr(HPSeq):

    '''
    Hawkes Process model for sequence classification 
    - gradient based optimization of the parameters.
    '''

    optim_options = {'disp': True}

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
        derivative_kernel_evaluate=updates.kernel_zha.derivative_kernel_evaluate,
        derivative_K_evaluate=updates.kernel_zha.derivative_K_evaluate,
        M=None,
        ):

        super(HPSeqFullSumGradConstr, self).__init__(
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
        self.derivative_kernel_evaluate = derivative_kernel_evaluate
        self.derivative_K_evaluate = derivative_K_evaluate

    def _mu_update_core(self):
        print '[HPSeqFullSumGradConstr] _mu_update_core'
        print 'self.spontaneous_node_vec:', self.spontaneous_node_vec
        logmu = np.log(self.mu)
        err = check_grad(
            updates.mu_fullsum_nonapprox.logmu_fullsum_func,
            updates.mu_fullsum_nonapprox.logmu_fullsum_grad,
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
        optout = minimize(
            updates.mu_fullsum_nonapprox.logmu_fullsum_funcgrad,
            logmu,
            (
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
            options=self.optim_options,
            )
        new_mu = np.exp(np.array(optout.x))
        return np.array(new_mu)

    def _gamma_update_core(self):
        print '[HPSeqFullSumGradConstr] _gamma_update_core'
        print 'self.spontaneous_node_vec:', self.spontaneous_node_vec
        loggamma = np.log(self.gamma)
        err = check_grad(
            updates.gamma_fullsum_nonapprox.loggamma_fullsum_func,
            updates.gamma_fullsum_nonapprox.loggamma_fullsum_grad,
            loggamma,
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

#                          epsilon=0.000000000000001)

        print 'gradient error ', err
        optout = minimize(
            updates.gamma_fullsum_nonapprox.loggamma_fullsum_funcgrad,
            loggamma,
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
            method='L-BFGS-B',
            jac=True,
            options=self.optim_options,
            )
        new_gamma = np.exp(np.array(optout.x))
        return np.array(new_gamma)

    def _alpha_update_core(self):
        print '[HPSeqFullSumGradConstr] _alpha_update_core'
        logalpha = np.log(self.alpha).flatten()
        err = check_grad(
            updates.alpha_fullsum_nonapprox.logalpha_fullsum_func,
            updates.alpha_fullsum_nonapprox.logalpha_fullsum_grad,
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

        optout = minimize(
            updates.alpha_fullsum_nonapprox.logalpha_fullsum_funcgrad,
            logalpha,
            (
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
            options=self.optim_options,
            )

        new_alpha = np.exp(optout.x)
        return np.reshape(new_alpha, (self.alpha.shape[0],
                          self.alpha.shape[1]))

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

    def _allnoomega_update(self, maxiter=-1):
        print '[HPSeqFullSumGradConstr] _allnoomega_update'
        logallparams = \
            np.log(updates.mualphaupdates_fullsum_nonapprox.encode_all_params_nogammanoomega(self.mu,
                   self.alpha))
        err = check_grad(
            lambda logallparams, omega, gamma, node_vec, eventmemes, \
                etimes, T, W, beta, kernel_evaluate, K_evaluate, lenmu, \
                derivative_kernel_evaluate, derivative_K_evaluate: \
                updates.mualphaupdates_fullsum_nonapprox.log_fullsum_func(
                    logallparams,
                    omega,
                    gamma,
                    node_vec,
                    eventmemes,
                    etimes,
                    T,
                    W,
                    beta,
                    kernel_evaluate,
                    K_evaluate,
                    lenmu,
                    ),
            updates.mualphaupdates_fullsum_nonapprox.log_fullsum_grad,
            logallparams,
            self.omega,
            self.gamma,
            self.node_vec,
            self.eventmemes,
            self.etimes,
            self.T,
            self.W,
            self.beta,
            self.kernel_evaluate,
            self.K_evaluate,
            len(self.mu),
            self.derivative_kernel_evaluate,
            self.derivative_K_evaluate,
            )
        print 'gradient error ', err

        options = self.optim_options
        if maxiter > 0:
            options['maxiter'] = maxiter
        optout = minimize(
            updates.mualphaupdates_fullsum_nonapprox.log_fullsum_funcgrad,
            logallparams,
            (
                self.omega,
                self.gamma,
                self.node_vec,
                self.eventmemes,
                self.etimes,
                self.T,
                self.W,
                self.beta,
                self.kernel_evaluate,
                self.K_evaluate,
                len(self.mu),
                self.derivative_kernel_evaluate,
                self.derivative_K_evaluate,
                ),
            method='L-BFGS-B',
            jac=True,
            options=options,
            )

        new_allparams = np.exp(optout.x)
        return new_allparams

    def _run_single_update_iteration(self,
            first_update_iteration=False, maxiter=-1):
        super(HPSeqFullSumGradConstr,
              self)._run_single_update_iteration(first_update_iteration=first_update_iteration)
        if 'allnoomega' in self.updates:
            all_params = self._allnoomega_update(maxiter=maxiter)
            (self.mu, self.alpha) = \
                updates.mualphaupdates_fullsum_nonapprox.decode_all_params_nogammanoomega(all_params,
                    len(self.mu))

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
        logintensity = self._event_logintensity(
            label,
            None,
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


