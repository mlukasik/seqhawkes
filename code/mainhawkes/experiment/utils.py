#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 15 Dec 2015

@author: michal

Utility functions for experimenting functionality.
'''

import numpy as np


def foldsplitter(taskcolumn, train_set_sizes):
    '''
    For each task id (in passed taskcolumn) take rows from number 
    train_set_sizes up for testing, 
    and all other rows for training (so training consists of both other 
    task ids and of rows from the same task id
    up to number train_set_sizes-1.
    '''

    folds = sorted(list(set(taskcolumn)))
    for fold in folds:
        for train_set_size in train_set_sizes:
            testfold2train = taskcolumn == fold
            cnt = 0
            for (i, x) in enumerate(testfold2train):
                if testfold2train[i]:
                    cnt += 1
                    if cnt > train_set_size:
                        testfold2train[i] = False
            remaining_train = taskcolumn != fold
            x = np.logical_or.reduce([testfold2train, remaining_train])

            yield (x, np.logical_not(x))


def CVsplitter(taskcolumn, K):
    '''
    Divide tasks into roughly equal K sets, and do CV over such K sets.
    '''

    tasks = sorted(list(set(taskcolumn)))
    tasks_splitted = [[] for _ in range(K)]
    for (ind, task) in enumerate(tasks):
        tasks_splitted[ind % K].append(task)

    for fold in range(K):
        print 'fold:', fold, 'testtasks:', tasks_splitted[fold]
        test = np.logical_or.reduce([taskcolumn == taskid for taskid in
                                    tasks_splitted[fold]])

        yield (np.logical_not(test), test)


