#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 11 Jul 2015

@author: michal, srijith

The basic Hawkes Process class.
'''

import numpy as np
from updates.mu_gamma import mu_update, gamma_update
from updates.alpha_update import alpha_update_linalg
from data.process import infecting_node, spontaneousnode_count, \
    spontaneousmeme_count, infections_count, one_hot_sparse
from updates.likelihood_observedu import evaluate_loglikelihood
from utils.base_plots import gpplot
from models.ogata import ogata, ogata_single_iteration, \
    ogata_single_iteration_inferringinfecting
import pylab as pb
import datetime
from models.util import update_summary
import updates.kernel_zha
import pickle


class HawkessProcessModel(object):

    '''
    A Hawkes Process Model.
    '''

    eps = 1e-8

    def __init__(
        self,
        etimes,
        node_vec,
        eventmemes,
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
        init_params_randomly=True,
        kernel_evaluate=updates.kernel_zha.kernel_evaluate,
        K_evaluate=updates.kernel_zha.K_evaluate,
        M=None,
        load_paramers_from_path=None,
        store_paramers=None,
        ):
        '''
        M - number of memes, can be passed here to overwrite the inferred M from the training data.
        '''

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
            M,
            )
        if init_params_randomly:
            self.mu = np.array([np.random.random() for _ in
                               range(self.D)])
            self.gamma = np.array([np.random.random() for _ in
                                  range(self.M)])
            self.alpha = np.array([np.random.random() for _ in
                                  range(self.D * self.D)])
            self.beta = np.array([np.random.random() for _ in
                                 range(self.M * self.V)])
        else:
            self.mu = np.array([1. for _ in range(self.D)])
            self.gamma = np.array([1. for _ in range(self.M)])
            self.alpha = np.array([1. for _ in range(self.D * self.D)])
            self.beta = np.array([1. for _ in range(self.M * self.V)])
        print '[HawkessProcessModel] self.M=', self.M, \
            'self.gamma.shape=', self.gamma.shape

        self.alpha = self.alpha.reshape(self.D, self.D)
        self.omega = init_omega

        self.beta = self.beta.reshape(self.M, self.V)
        self._normalize_beta_rowwise()

        self.updates = updates
        self.iterations = iterations
        self.verbose = verbose

        self.first_update_iteration = True

        self.kernel_evaluate = kernel_evaluate
        self.K_evaluate = K_evaluate

        if load_paramers_from_path is not None:
            self.load_parameters(load_paramers_from_path)

        # if store_paramers is not None:

        self.store_paramers = store_paramers

    def store_parameters(self, store_paramers_to_path):
        pickle.dump(dict((parameter_to_store, getattr(self,
                    parameter_to_store)) for parameter_to_store in
                    self.store_paramers[0]),
                    open(self.store_paramers[1], 'w'))

    def load_parameters(self, load_paramers_from_path):
        parameters = pickle.load(open(load_paramers_from_path, 'r'))
        for (paramname, paramvalue) in parameters.iteritems():
            setattr(self, paramname, paramvalue)
            print '[HP.load_parameters] loaded parameter', paramname, \
                paramvalue

    def _set_training_data(
        self,
        etimes,
        node_vec,
        eventmemes,
        infected_vec,
        infecting_vec,
        W,
        T,
        V,
        D,
        M=None,
        ):
        self._set_prediction_data(etimes, node_vec, eventmemes)
        self.infected_vec = infected_vec
        self.infecting_vec = infecting_vec
        self.W = W
        self.T = T
        self.V = V
        self.D = D
        if len(self.infected_vec) > 0:
            self.N = int(max(self.infected_vec)) + 1
        else:
            self.N = 0

        if M is not None:
            self.M = M
        elif len(self.eventmemes) > 0:
            self.M = int(max(self.eventmemes)) + 1  # because indexed from 0
        else:
            self.M = 0

        self.spontaneous_node_vec = \
            spontaneousnode_count(self.infecting_vec,
                                  self.infected_vec, self.node_vec,
                                  self.D)
        self.spontaneous_meme_vec = \
            spontaneousmeme_count(self.infecting_vec,
                                  self.infected_vec, self.eventmemes,
                                  self.M)

        self._set_infecting_node_vec_infections_mat()

        self.enode_mat_sparse = one_hot_sparse(self.node_vec, self.D)

    def _update_nodevec_data(self, node_vec):
        self.node_vec = node_vec
        self.spontaneous_node_vec = \
            spontaneousnode_count(self.infecting_vec,
                                  self.infected_vec, self.node_vec,
                                  self.D)
        self._set_infecting_node_vec_infections_mat()
        self.enode_mat_sparse = one_hot_sparse(self.node_vec, self.D)

    def _set_infecting_node_vec_infections_mat(self):
        (self.infecting_node_vec, self.eventid_to_node) = \
            infecting_node(self.infected_vec, self.infecting_vec,
                           self.node_vec)
        self.infections_mat = infections_count(self.infecting_node_vec,
                self.node_vec, self.infecting_vec, self.infected_vec,
                self.D)

    def _get_prediction_data(self):
        return (self.etimes, self.node_vec, self.eventmemes)

    def _set_prediction_data(
        self,
        etimes,
        node_vec,
        eventmemes,
        ):
        self.etimes = etimes
        self.node_vec = node_vec
        self.eventmemes = eventmemes

    def _generic_update(
        self,
        update_func,
        update_name,
        original_param,
        ):
        print 'Updating ' + update_name
        start = datetime.datetime.now()
        updated_param = update_func()
        end = datetime.datetime.now()
        print '\t'.join([
            'Variable',
            'meanabsdiff',
            'stdabsdiff',
            'meanreldiff',
            'stdreldiff',
            'time',
            ])
        summary = update_summary(updated_param, original_param, start,
                                 end)
        print update_name, '\t'.join(map(str, summary))
        timeO = (end - start).microseconds
        self.runtime[update_name] = self.runtime.get(update_name, []) \
            + [[self.N, timeO]]
        self.errors[update_name] = self.errors.get(update_name, []) \
            + [summary]
        return updated_param

    def _mu_update(self):
        return self._generic_update(lambda : mu_update(self.T,
                                    self.gamma,
                                    self.spontaneous_node_vec).flatten(),
                                    'mu', self.mu)

    def _gamma_update(self):
        return self._generic_update(lambda : gamma_update(self.T,
                                    self.spontaneous_meme_vec,
                                    self.mu).flatten(), 'gamma',
                                    self.gamma)

    def _alpha_update(self):
        return self._generic_update(lambda : alpha_update_linalg(
                self.infections_mat,
                self.enode_mat_sparse,
                self.etimes,
                self.omega,
                self.T,
                self.K_evaluate,
                ), 'alpha', self.alpha)

    def get_influential(self):
        (_, N) = np.shape(self.alpha)
        salpha = self.alpha
        infalpha = np.reshape(salpha[np.eye(N) == 0], (N, N - 1))
        return np.sum(infalpha, axis=1)

    def _normalize_beta_rowwise(self):
        row_sums = self.beta.sum(axis=1)
        self.beta = self.beta / row_sums[:, np.newaxis]

    def _beta_update_raw(self):
        '''
        Run only once - it does not depend on other parameters.
        '''

        for memeid in xrange(self.M):
            self.beta[memeid] = self.W[self.eventmemes == memeid, :
                    ].sum(axis=0)

        # Laplace smoothing to avoid zeros!

        self.beta += 1
        self._normalize_beta_rowwise()
        return self.beta

    def _beta_update(self):
        return self._generic_update(lambda : self._beta_update_raw(),
                                    'beta', self.beta)

    def _run_single_update_iteration(self,
            first_update_iteration=False):
        if self.first_update_iteration or first_update_iteration:
            self._run_one_time_updates()
            self.first_update_iteration = False
        if 'mu' in self.updates:
            self.mu = self._mu_update()
        if 'gamma' in self.updates:
            self.gamma = self._gamma_update()
        if 'omega' in self.updates:
            self.omega = self._omega_update()
        if 'alpha' in self.updates:
            self.alpha = self._alpha_update()

    def _run_one_time_updates(self):
        if 'beta' in self.updates:
            self.beta = self._beta_update()

    def evaluate_likelihood(self):
        return evaluate_loglikelihood(
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

    def _check_stop_condition(self):
        should_break = False
        if len(self.loglikelihoods) > 1:
            print 'log-likelihood:', self.loglikelihoods[-1]
            diff = self.loglikelihoods[-1] - self.loglikelihoods[-2]
            reldiff = np.abs(1. * diff / self.loglikelihoods[-2])
            print 'relative difference in loglikehood:', reldiff
            if reldiff < 1e-6:
                should_break = True
        if self.train_iteration >= self.iterations:
            should_break = True
        self.train_iteration += 1
        return should_break

    def train(self, first_update_iteration=False):
        '''
        The main training method.
        '''

        print '[HP.train] initial parameter values'
        print '[HP.train] self.alpha:', self.alpha
        print '[HP.train] self.mu:', self.mu
        print '[HP.train] self.gamma:', self.gamma
        print '[HP.train] self.omega:', self.omega

        print ','.join(map(str, [
            'Setting_',
            'T:',
            self.T,
            'D:',
            self.D,
            'M:',
            self.M,
            'V:',
            self.V,
            ]))

        self.loglikelihoods = []
        self.errors = {}
        self.runtime = {}
        self.train_iteration = 0
        while True:
            self.loglikelihoods.append(self.evaluate_likelihood())
            print 'Iteration (' + str(self.train_iteration) + ')', \
                self.loglikelihoods[-1]
            if self._check_stop_condition():
                break
            self._run_single_update_iteration(first_update_iteration=first_update_iteration)

        if self.verbose == True:
            print '[HP.train] self.alpha:', self.alpha
        print (self.errors, self.runtime)
        print '[HP.train] initial loglikelihood ', \
            self.loglikelihoods[0]
        print '[HP.train] final loglikelihood ', self.loglikelihoods[-1]

        if self.store_paramers is not None:
            print '[HP.train] storing parameters...'
            self.store_parameters(self.store_paramers)

        print '[HP.train] final parameter values'
        print '[HP.train] self.alpha:', self.alpha
        print '[HP.train] self.mu:', self.mu
        print '[HP.train] self.gamma:', self.gamma
        print '[HP.train] self.omega:', self.omega
        return (self.errors, self.runtime)

    def predict(
        self,
        tbegin,
        T,
        meme=None,
        single_point=False,
        ):
        if meme == None:
            (sampledtimes, sampledusers) = self.simAll(tbegin, T,
                    single_point)
        else:
            (sampledtimes, sampledusers) = self.simMeme(tbegin, T,
                    meme, single_point)
        return (sampledtimes, sampledusers)

    def predictUser(
        self,
        tbegin,
        T,
        user=None,
        single_point=False,
        ):
        if user == None:
            (sampledtimes, sampledmemes) = self.simUserAll(tbegin, T,
                    single_point)
        else:
            (sampledtimes, sampledmemes) = self.simUser(tbegin, T,
                    user, single_point)
        return (sampledtimes, sampledmemes)

    def plot_intensity_all(self, x, figure_path):
        sumI = np.array([self.sumIntensitiesAll(xi, self.node_vec,
                        self.etimes, True)[1] for xi in x])
        (f_mean, f_lower, f_upper) = (sumI, sumI, sumI)
        gpplot(x, f_mean, f_lower, f_upper)
        pb.xlabel('time')
        pb.ylabel('intensity lambda over all memes and users')
        pb.savefig(figure_path)

    def sumIntensitiesMeme(
        self,
        t,
        m,
        node_vec,
        etimes,
        filterlatertimes=True,
        ):
        if filterlatertimes:
            I = self.mu * self.gamma[m] \
                + np.dot(np.transpose(self.alpha[node_vec[etimes
                         < t].astype(int), :][:, range(self.D)]),
                         self.kernel_evaluate(t, etimes[etimes < t],
                         self.omega))
        else:
            I = self.mu * self.gamma[m] \
                + np.dot(np.transpose(self.alpha[node_vec.astype(int), :
                         ][:, range(self.D)]), self.kernel_evaluate(t,
                         etimes, self.omega))
        sumI = np.sum(I)
        return (I, sumI)

    def sumIntensitiesMemeMixtureModelUniform(
        self,
        t,
        m,
        filterlatertimes=True,
        ):
        '''
        Training assumes observed u; we can assume the intensity function is like a mixture model, so we can multiply each term
        by its weight.
        
        The weight here is 1/(n+1), where n is the number of training points.
        '''

        mugamma_term = self.mu * self.gamma[m]
        if filterlatertimes:
            kernelsummation_terms = \
                (np.transpose(self.alpha[self.node_vec[self.etimes
                 < t].astype(int), :][:, range(self.D)]),
                 self.kernel_evaluate(t, self.etimes[self.etimes < t],
                 self.omega))
        else:
            kernelsummation_terms = \
                (np.transpose(self.alpha[self.node_vec.astype(int), :][:
                 , range(self.D)]), self.kernel_evaluate(t,
                 self.etimes, self.omega))
        I = mugamma_term + np.dot(*list(kernelsummation_terms))
        sumI = np.sum(I)
        return (I, sumI, mugamma_term, kernelsummation_terms,
                self.infected_vec[self.etimes < t])

    def sumIntensitiesMemeMixtureModelWeighted(
        self,
        t,
        m,
        node_vec,
        etimes,
        filterlatertimes=True,
        ):
        '''
        Training assumes observed u; we can assume the intensity function is like a mixture model, so we can multiply each term
        by its weight.
        
        The weight here should be term/(sum of all terms)
        
        TODO: need to do some pointwise multiplications, the initial I is already the denominator for scaling coefficient. 
        '''

        if filterlatertimes:
            I = self.mu * self.gamma[m] \
                + np.dot(np.transpose(self.alpha[node_vec[etimes
                         < t].astype(int), :][:, range(self.D)]),
                         self.kernel_evaluate(t, etimes[etimes < t],
                         self.omega))
        else:
            I = self.mu * self.gamma[m] \
                + np.dot(np.transpose(self.alpha[node_vec.astype(int), :
                         ][:, range(self.D)]), self.kernel_evaluate(t,
                         etimes, self.omega))
        I = I / (1. * len(etimes[etimes < t]))
        sumI = np.sum(I)
        return (I, sumI)

    def sumIntensitiesAll(
        self,
        t,
        node_vec,
        etimes,
        filterlatertimes=False,
        ):
        if filterlatertimes:
            I = self.mu * np.sum(self.gamma) \
                + np.dot(np.transpose(self.alpha[node_vec[etimes
                         < t].astype(int), :][:, range(self.D)]),
                         self.kernel_evaluate(t, etimes[etimes < t],
                         self.omega))
        else:
            I = self.mu * np.sum(self.gamma) \
                + np.dot(np.transpose(self.alpha[node_vec.astype(int), :
                         ][:, range(self.D)]), self.kernel_evaluate(t,
                         etimes, self.omega))
        sumI = np.sum(I)
        return (I, sumI)

    def sumIntensitiesUser(
        self,
        t,
        d,
        filterlatertimes=False,
        ):
        I = np.zeros(self.M)
        for m in range(self.M):
            I[m] = self._intensityUserMeme(t, d, m, filterlatertimes)
        sumI = np.sum(I)
        return (I, sumI)

    def _intensityUserMeme(
        self,
        t,
        d,
        m,
        filterlatertimes=False,
        ):
        etimes = self.etimes[self.eventmemes == m]
        node_vec = self.node_vec[self.eventmemes == m]
        if filterlatertimes:
            return self.mu[d] * self.gamma[m] \
                + np.dot(np.transpose(self.alpha[node_vec[etimes
                         < t].astype(int), :][:, d]),
                         self.kernel_evaluate(t, etimes[etimes < t],
                         self.omega))
        else:
            return self.mu[d] * self.gamma[m] \
                + np.dot(np.transpose(self.alpha[node_vec.astype(int), :
                         ][:, d]), self.kernel_evaluate(t, etimes,
                         self.omega))

    def sumIntensitiesUserAll(self, t, filterlatertimes=False):
        I = np.zeros(self.M)
        for m in range(self.M):
            etimes = self.etimes[self.eventmemes == m]
            node_vec = self.node_vec[self.eventmemes == m]
            if filterlatertimes:
                I[m] = np.sum(self.mu * self.gamma[m]) \
                    + np.sum(np.dot(np.transpose(self.alpha[node_vec[etimes
                             < t].astype(int), :][:, range(self.D)]),
                             self.kernel_evaluate(t, etimes[etimes
                             < t], self.omega)))
            else:
                I[m] = np.sum(self.mu * self.gamma[m]) \
                    + np.sum(np.dot(np.transpose(self.alpha[node_vec.astype(int),
                             :][:, range(self.D)]),
                             self.kernel_evaluate(t, etimes,
                             self.omega)))
        sumI = np.sum(I)
        return (I, sumI)

    def simMeme(
        self,
        tbegin,
        T,
        m,
        single_point=False,
        ):
        '''
        Rewritten Ogata's thinning algorithm from Zha code: simhawkes.cpp, simMeme function.
        Needs to modify to consider intensity associated with a single meme
        '''

        etimes = self.etimes[self.eventmemes == m]
        node_vec = self.node_vec[self.eventmemes == m]
        if single_point:
            (sampledtimes, sampledusers) = \
                ogata_single_iteration(tbegin, T, self.D, lambda t: \
                    self.sumIntensitiesAll(t, node_vec, etimes))
        else:
            (sampledtimes, sampledusers) = ogata(tbegin, T, self.D,
                    lambda t: self.sumIntensitiesMeme(t, m, node_vec,
                    etimes))
        return (sampledtimes, sampledusers)

    def simUser(
        self,
        tbegin,
        T,
        d,
        single_point=False,
        ):
        '''
        Rewritten Ogata's thinning algorithm from Zha code: simhawkes.cpp, simMeme function.
        Needs to modify to consider intensity associated with a single user
        '''

        if single_point:
            (sampledtimes, sampledmemes) = \
                ogata_single_iteration(tbegin, T, self.M, lambda t: \
                    self.sumIntensitiesUserAll(t))
        else:
            (sampledtimes, sampledmemes) = ogata(tbegin, T, self.M,
                    lambda t: self.sumIntensitiesUser(t, d))

        return (sampledtimes, sampledmemes)

    def simAll(
        self,
        tbegin,
        T,
        single_point=False,
        ):
        '''
        Rewritten Ogata's thinning algorithm from Zha code: simhawkes.cpp, simMeme function.
        '''

        if single_point:
            (sampledtimes, sampledusers) = \
                ogata_single_iteration(tbegin, T, self.D, lambda t: \
                    self.sumIntensitiesAll(t, self.node_vec,
                    self.etimes))
        else:
            (sampledtimes, sampledusers) = ogata(tbegin, T, self.D,
                    lambda t: self.sumIntensitiesAll(t, self.node_vec,
                    self.etimes))
        return (sampledtimes, sampledusers)

    def simUserAll(
        self,
        tbegin,
        T,
        single_point=False,
        ):
        '''
        Rewritten Ogata's thinning algorithm from Zha code: simhawkes.cpp, simMeme function.
        '''

        if single_point:
            (sampledtimes, sampledmemes) = \
                ogata_single_iteration(tbegin, T, self.M,
                    self.sumIntensitiesUserAll)
        else:
            (sampledtimes, sampledmemes) = ogata(tbegin, T, self.M,
                    self.sumIntensitiesUserAll)
        return (sampledtimes, sampledmemes)

    def simMemeUpdatingModelGen(
        self,
        tbegin,
        Tmax,
        K,
        intensity_fun,
        meme,
        restore_original_training_data,
        ):
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
        sampledtime = tbegin
        while True:
            (sampledtime, sampleduser, infectingevent) = \
                ogata_single_iteration_inferringinfecting(sampledtime,
                    Tmax, K, intensity_fun)
            if sampledtime == None or sampledtime > Tmax:
                break

#             print "sampledtime, Tmax:", sampledtime, Tmax

            infectedevent = np.max(np.hstack((self.infected_vec,
                                   np.array([0])))) + 1
            if infectingevent == -1:
                infectingevent = infectedevent
            self._set_training_data(  # assigned next largest integer as a new event id
                                      # add -1 because it is not needed in prediction
                                      # np.vstack((self.W, words)),
                np.append(self.etimes, sampledtime),
                np.append(self.node_vec, sampleduser),
                np.append(self.eventmemes, meme),
                np.append(self.infected_vec, infectedevent),
                np.append(self.infecting_vec, infectingevent),
                self.W,
                sampledtime,
                self.V,
                self.D,
                )
            yield (sampledtime, sampleduser, infectingevent)
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

    def simMemeUpdatingModel(
        self,
        tbegin,
        T,
        K,
        intensity_fun,
        meme,
        restore_original_training_data,
        ):
        l = list(self.simMemeUpdatingModelGen(
            tbegin,
            T,
            K,
            intensity_fun,
            meme,
            restore_original_training_data,
            ))
        return (map(lambda x: x[0], l), map(lambda x: x[1], l),
                map(lambda x: x[2], l))


# class HawkessProcessModel(HawkessProcessModelKernel1):
#     def __init__(self, etimes, node_vec, eventmemes, infected_vec,
#                  infecting_vec, W, T, V, D, updates, iterations, verbose,
#                  init_omega=1, init_params_randomly=True):
#         super(HawkessProcessModelAlphaOnes, self).__init__(etimes, node_vec, eventmemes, infected_vec,
#                                                            infecting_vec, W, T, V, D, updates, iterations, verbose,
#                                                            init_omega=1, init_params_randomly)
#
#         self.kernel_evaluate=updates.kernel_zha.kernel_evaluate
#         self.K_evaluate=updates.kernel_zha.K_evaluate

class HawkessProcessModelAlphaOnes(HawkessProcessModel):

    def __init__(
        self,
        etimes,
        node_vec,
        eventmemes,
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
        init_params_randomly=True,
        kernel_evaluate=updates.kernel_zha.kernel_evaluate,
        K_evaluate=updates.kernel_zha.K_evaluate,
        M=None,
        load_paramers_from_path=None,
        store_paramers=None,
        ):

        super(HawkessProcessModelAlphaOnes, self).__init__(
            etimes,
            node_vec,
            eventmemes,
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
            M,
            load_paramers_from_path,
            store_paramers,
            )
        self.alpha = np.ones((self.D, self.D))
