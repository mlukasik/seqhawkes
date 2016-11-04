#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 15 Nov 2015

@author: michal

Experiment class that is base for stance classification.
'''

from data.load_zha import loadX
import numpy as np
import time


class Experiment(object):

    def __init__(
        self,
        fname_data,
        model_constructors,
        train_fraction,
        metrics=[],
        usermetrics=[],
        memerankmetrics=[],
        normalize_time_per_meme=False,
        FOLDTORUN=-1,
        ):

        self.train_fraction = train_fraction
        self._load_data(fname_data, normalize_time_per_meme)

        self.metrics = metrics
        self.usermetrics = usermetrics
        self.memerankmetrics = memerankmetrics
        self.model_constructors = model_constructors
        self.results = {}
        self.infresults = {}
        self.FOLDTORUN = FOLDTORUN

    def _load_data(self, fname_data, normalize_time_per_meme=False):
        (X, self.words_keys) = loadX(fname_data)

        self.N_all = X.shape[0]

        self.eventmemes_all = X[:, 0]
        self.infected_vec_all = X[:, 1]
        self.infecting_vec_all = X[:, 2]
        self.node_vec_all = X[:, 3]
        self.etimes_all = X[:, 4]
        if normalize_time_per_meme:
            for meme in set(self.eventmemes_all):
                etimes_meme = self.etimes_all[self.eventmemes_all
                        == meme]
                self.etimes_all[self.eventmemes_all == meme] = \
                    etimes_meme - np.min(etimes_meme)
        else:
            self.etimes_all = self.etimes_all - np.min(self.etimes_all)
        self.V = len(self.words_keys)
        self.W_all = X[:, 5:]
        assert self.V == self.W_all.shape[1]

        # sort the data, do we even need this?

        ordtimes = np.argsort(self.etimes_all)
        self.etimes_all = self.etimes_all[ordtimes]
        self.infecting_vec_all = self.infecting_vec_all[ordtimes]
        self.infected_vec_all = self.infected_vec_all[ordtimes]
        self.node_vec_all = self.node_vec_all[ordtimes]
        self.eventmemes_all = self.eventmemes_all[ordtimes]
        self.W_all = self.W_all[ordtimes, :]

        self.D = int(max(self.node_vec_all)) + 1

    def _split_train_test(self):
        trainN = int(self.N_all * self.train_fraction)
        trainetimes = self.etimes_all[0:trainN]
        traininfecting_vec = self.infecting_vec_all[0:trainN]
        traininfected_vec = self.infected_vec_all[0:trainN]
        trainnode_vec = self.node_vec_all[0:trainN]
        traineventmemes = self.eventmemes_all[0:trainN]
        trainW = self.W_all[0:trainN, :]
        trainT = max(trainetimes)

        testtimes = self.etimes_all[trainN:self.N_all]
        testinfecting_vec = self.infecting_vec_all[trainN:self.N_all]
        testinfected_vec = self.infected_vec_all[trainN:self.N_all]
        testnode_vec = self.node_vec_all[trainN:self.N_all]
        testeventmemes = self.eventmemes_all[trainN:self.N_all]
        testW = self.W_all[trainN:self.N_all, :]
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

    def build_models(self, train=True):
        self.models = {}
        self.traintime = {}
        for (foldind, ((
            _,
            trainetimes,
            traininfecting_vec,
            traininfected_vec,
            trainnode_vec,
            traineventmemes,
            trainW,
            trainT,
            ), (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            ))) in enumerate(self._split_train_test()):
            if self.FOLDTORUN == -1 or self.FOLDTORUN == foldind:
                for (modelname, model_constr) in \
                    self.model_constructors:
                    print 'Building model:', modelname
                    model = model_constr(
                        trainetimes,
                        trainnode_vec,
                        traineventmemes,
                        traininfected_vec,
                        traininfecting_vec,
                        trainW,
                        trainT,
                        self.V,
                        self.D,
                        )
                    time1 = time.time()
                    if train:
                        model.train()
                    time2 = time.time()
                    elapsedtime = time2 - time1
                    self.traintime[modelname] = \
                        self.traintime.get(modelname, {})
                    self.traintime[modelname][foldind] = elapsedtime
                    self.models[modelname] = self.models.get(modelname,
                            {})
                    self.models[modelname][foldind] = model

    def evaluate_influential(self, topN):
        for (foldind, _) in enumerate(self._split_train_test()):
            if self.FOLDTORUN == -1 or self.FOLDTORUN == foldind:
                for (modelname, model) in self.models.iteritems():
                    print 'Obtaining top influences for model:', \
                        modelname
                    infuserval = model[foldind].get_influential()
                    if topN != 0:
                        infusers = (np.argsort(infuserval)[::-1])[:topN]
                    else:
                        infusers = np.argsort(infuserval)[::-1]
                    self.infresults[modelname] = infusers

    def _init_results(self, modelname):
        self.results[modelname] = {}
        if self.metrics:
            for (metricname, _) in self.metrics:
                self.results[modelname][metricname] = {}
        if self.usermetrics:
            for (metricname, _) in self.usermetrics:
                self.results[modelname][metricname] = {}

    def _append_results(
        self,
        modelname,
        foldind,
        testtimes,
        sampledtimes,
        testentities,
        sampledentities,
        ):
        self._append_results_entities(modelname, foldind, testentities,
                sampledentities)
        self._append_results_times(modelname, foldind, testtimes,
                                   sampledtimes)

    def _append_results_entities(
        self,
        modelname,
        foldind,
        testentities,
        sampledentities,
        ):
        for (metricname, metric) in self.usermetrics:
            res = metric(testentities, sampledentities)
            self.results[modelname][metricname][foldind] = \
                self.results[modelname][metricname].get(foldind, [])
            self.results[modelname][metricname][foldind].append(res)

    def _append_results_times(
        self,
        modelname,
        foldind,
        testtimes,
        sampledtimes,
        ):
        for (metricname, metric) in self.metrics:
            res = metric(testtimes, sampledtimes)
            self.results[modelname][metricname][foldind] = \
                self.results[modelname][metricname].get(foldind, [])
            self.results[modelname][metricname][foldind].append(res)

    def evaluate_futurepoints_memesjoined(self):
        for (foldind, ((
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            trainT,
            ), (
            _,
            testtimes,
            _,
            _,
            testnode_vec,
            _,
            _,
            testT,
            ))) in enumerate(self._split_train_test()):
            if self.FOLDTORUN == -1 or self.FOLDTORUN == foldind:
                for (modelname, model) in self.models.iteritems():
                    self._init_results(modelname)
                    print 'Evaluating model:', modelname
                    (sampledtimes, sampledusers) = \
                        model[foldind].predict(trainT, testT)
                    print '#Predicted points:', len(sampledtimes), \
                        '#Test points ', len(testtimes)
                    predpoints = len(sampledtimes)
                    testpoints = len(testtimes)
                    rmse = np.sqrt((predpoints - testpoints) ** 2)
                    print 'RMSE number of future tweets on joint evaluation ', \
                        rmse
                    self._append_results(
                        modelname,
                        foldind,
                        testtimes,
                        sampledtimes,
                        testnode_vec,
                        sampledusers,
                        )

    def evaluate_futurepoints_separatedmemes(self):
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
            if self.FOLDTORUN == -1 or self.FOLDTORUN == foldind:
                for (modelname, model) in self.models.iteritems():
                    self._init_results(modelname)
                    print 'Evaluating model:', modelname
                    rmse = []
                    for meme in set(traineventmemes):
                        (sampledtimes, sampledusers) = \
                            model[foldind].predict(trainT, testT, meme)
                        testtimes_meme = testtimes[testeventmemes
                                == meme]
                        testnodes_meme = testnode_vec[testeventmemes
                                == meme]
                        print '#Predicted points:', len(sampledtimes), \
                            '#Test points ', len(testtimes)
                        predpoints = len(sampledtimes)
                        testpoints = len(testtimes)
                        rmse.append(np.sqrt((predpoints - testpoints)
                                    ** 2))
                        self._append_results(
                            modelname,
                            foldind,
                            testtimes_meme,
                            sampledtimes,
                            testnodes_meme,
                            sampledusers,
                            )
                    meanrmse = np.mean(rmse)
                    stdrmse = np.std(rmse)
                    print 'RMSE number of future tweets on memes ', \
                        meanrmse, '=/-', stdrmse

    def evaluate_futurepoints_separatedusers(self):
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
            if self.FOLDTORUN == -1 or self.FOLDTORUN == foldind:
                for (modelname, model) in self.models.iteritems():
                    print 'Evaluating model:', modelname
                    self._init_results(modelname)
                    rmse = []
                    for user in set(trainnode_vec):
                        (sampledtimes, sampledmemes) = \
                            model[foldind].predictUser(trainT, testT,
                                user)
                        try:
                            testtimes_user = testtimes[testnode_vec
                                    == user]
                        except:
                            pass
                        testmemes_user = testeventmemes[testnode_vec
                                == user]
                        print '#Predicted points:', len(sampledtimes), \
                            '#Test points ', len(testtimes_user)
                        predpoints = len(sampledtimes)
                        testpoints = len(testtimes_user)
                        rmse.append(np.sqrt((predpoints - testpoints)
                                    ** 2))
                        self._append_results(
                            modelname,
                            foldind,
                            testtimes_user,
                            sampledtimes,
                            testmemes_user,
                            sampledmemes,
                            )
                    meanrmse = np.mean(rmse)
                    stdrmse = np.std(rmse)
                    print 'RMSE number of future tweets by user ', \
                        meanrmse, '=/-', stdrmse

    def evaluate_futurepointssubintervals_separatedmemes(self,
            prediction_sub_interval_size):
        '''
        short-time prediction into future; then observing the gold data; repeat

        prediction_sub_interval_size - small subinterval which we observe at each time of prediction
        '''

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
            testtimes_vec,
            testinfecting_vec,
            testinfected_vec,
            testnode_vec,
            testeventmemes,
            testW,
            testT,
            ))) in enumerate(self._split_train_test()):
            if self.FOLDTORUN == -1 or self.FOLDTORUN == foldind:
                for (modelname, model) in self.models.iteritems():
                    self._init_results(modelname)
                    for meme in set(traineventmemes):
                        print 'Evaluating model:', modelname

                        testtimes = testtimes_vec[testeventmemes
                                == meme]
                        testnodes = testnode_vec[testeventmemes == meme]

                        # store original training data

                        (etimes, node_vec, eventmemes) = \
                            model._get_prediction_data()

                        (sampledtimes, sampledusers) = (np.array([]),
                                np.array([]))
                        start_time = self.T
                        end_time = self.T + prediction_sub_interval_size
                        while end_time <= self.testT:
                            print 'end_time:', end_time, 'self.testT', \
                                self.testT
                            (times, users) = \
                                model[foldind].predict(start_time,
                                    end_time, meme)

                            sampledtimes = np.hstack((sampledtimes,
                                    times))
                            sampledusers = np.hstack((sampledusers,
                                    users))

                            true_events_to_add = \
                                np.logical_and(end_time > testtimes,
                                    testtimes >= start_time)
                            times_to_add_true = \
                                testtimes[true_events_to_add]
                            users_to_add_true = \
                                testnodes[true_events_to_add]
                            model.etimes = np.hstack((model.etimes,
                                    times_to_add_true))
                            model.node_vec = np.hstack((model.node_vec,
                                    testnodes))
                            model.eventmemes = \
                                np.hstack((model.eventmemes, [meme
                                    for _ in
                                    xrange(len(times_to_add_true))]))

                            start_time = start_time \
                                + prediction_sub_interval_size
                            end_time = end_time \
                                + prediction_sub_interval_size

                        print '#Predicted points:', len(sampledtimes), \
                            '#Test points ', len(testtimes)
                        self._append_results(
                            modelname,
                            foldind,
                            testtimes,
                            sampledtimes,
                            testnodes,
                            sampledusers,
                            )

                        # restore original training data

                        model._set_prediction_data(etimes, node_vec,
                                eventmemes)

    def evaluate_nextpoint_separatedmemes(self):
        '''
        inter-arrival time: when is the next tweet
        
        We iteratively predict the next tweet, update with ground truth and move to the next tweet.
        Notice that this way we always predict a correct number of tweets.
        '''

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
            testtimes_vec,
            testinfecting_vec,
            testinfected_vec,
            testnode_vec,
            testeventmemes,
            testW,
            testT,
            ))) in enumerate(self._split_train_test()):
            if self.FOLDTORUN == -1 or self.FOLDTORUN == foldind:
                for (modelname, model) in self.models.iteritems():
                    self._init_results(modelname)
                    for meme in set(traineventmemes):
                        print 'Evaluating model:', modelname
                        testtimes = testtimes_vec[testeventmemes
                                == meme]
                        testnodes = testnode_vec[testeventmemes == meme]

                        # store original training data

                        (etimes, node_vec, eventmemes) = \
                            model[foldind]._get_prediction_data()

                        (sampledtimes, sampledusers) = (np.array([]),
                                np.array([]))
                        t = trainT
                        next_event_index = 0
                        while next_event_index < len(testtimes):
                            (t, user) = model[foldind].predict(t,
                                    testT, meme, single_point=True)
                            if t == None:
                                break
                            sampledtimes = np.append(sampledtimes, t)
                            sampledusers = np.append(sampledusers, user)

                            t = testtimes[next_event_index]
                            user = sampledusers[next_event_index]
                            next_event_index += 1
                            model[foldind].etimes = \
                                np.append(model[foldind].etimes, t)
                            model[foldind].node_vec = \
                                np.append(model[foldind].node_vec, user)
                            model[foldind].eventmemes = \
                                np.append(model[foldind].eventmemes,
                                    meme)

                        print '#Predicted points:', len(sampledtimes), \
                            '#Test points ', len(testtimes)
                        self._append_results(
                            modelname,
                            foldind,
                            testtimes,
                            sampledtimes,
                            testnodes,
                            sampledusers,
                            )

                        # restore original training data

                        model[foldind]._set_prediction_data(etimes,
                                node_vec, eventmemes)

    def evaluate_meme_ranking(self):
        '''
        which conversation will 'go viral', as a ranking problem
        '''

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
            testtimes_vec,
            testinfecting_vec,
            testinfected_vec,
            testnode_vec,
            testeventmemes,
            testW,
            testT,
            ))) in enumerate(self._split_train_test()):
            if self.FOLDTORUN == -1 or self.FOLDTORUN == foldind:
                for (modelname, model) in self.models.iteritems():
                    self.results[modelname] = \
                        self.results.get(modelname, {})

                    # self.results[modelname][foldind]={}

                    for (metricname, metric) in self.memerankmetrics:
                        self.results[modelname][metricname] = {}

                    memecount_sampled = []
                    memecount_true = []
                    for meme in set(traineventmemes):
                        print 'Evaluating model:', modelname, meme
                        (sampledtimes, sampledusers) = \
                            model[foldind].predict(trainT, testT, meme)
                        testtimes = testtimes_vec[testeventmemes
                                == meme]
                        testnodes = testnode_vec[testeventmemes == meme]
                        print '#Predicted points:', len(sampledtimes), \
                            '#Test points ', len(testtimes)
                        memecount_sampled.append(len(sampledtimes))
                        memecount_true.append(len(testtimes))

                    for (metricname, metric) in self.memerankmetrics:
                        print 'Evaluating meme rank:', \
                            memecount_sampled, memecount_true
                        res = metric(memecount_sampled, memecount_true)
                        if foldind \
                            in self.results[modelname][metricname]:
                            self.results[modelname][metricname][foldind].append(res)
                        else:
                            self.results[modelname][metricname][foldind] = \
                                res

    def summarize(self):
        print 'Result summary'
        import sys
        print >> sys.stderr, 'FOLDTORUN: ' + str(self.FOLDTORUN) + ':' \
            + str(self.results)
        self._summarize(map(lambda x: x[0], self.metrics
                        + self.usermetrics), lambda x: '%.10f' \
                        % np.mean(np.array(x.values()).flatten()) \
                        + '+/-' + '%.10f' \
                        % np.std(np.array(x.values()).flatten()))
        print 'Training time'
        for (methodname, times) in self.traintime.iteritems():
            print '\t'.join([methodname] + [str(times)])

    def summarize_memeranking(self):
        print 'Meme rank result summary'
        self._summarize(map(lambda x: x[0], self.memerankmetrics),
                        lambda x: '%.10f' \
                        % np.mean(np.array(x.values()).flatten()))

    def _summarize(self, metricnames, result_to_str):
        print 'self.results', self.results
        print '\t'.join(['method'] + metricnames)
        for (methodname, methodresults) in self.results.iteritems():
            print '\t'.join([methodname]
                            + [result_to_str(methodresults[metricname])
                            for metricname in metricnames])
        print


