#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Copyright (c) 2014-2015, The University of Sheffield. 
This file is part of the seqhawkes software 
(see https://github.com/mlukasik/seqhawkes), 
and is free software, licenced under the GNU Library General Public License,
Version 2, June 1991 (in the distribution as file LICENSE).

Created on 13 Mar 2015

@author: michal

General purpose utility functions.
'''

import time
import numpy as np
import getpass
import os
import pickle

# from cloud.serialization import cloudpickle

import socket


def initialize_seed_with_currtime():
    RANDOM_SEED = np.int(time.time())
    print '[initialize_seed_with_currtime] Initializing RANDOM_SEED to:', \
        RANDOM_SEED
    np.random.seed(RANDOM_SEED)


def make_dir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def current_time_str():
    return time.strftime('%d_%m_%Y_%H_%M_%S')


def store(model, path):
    pickle.dump(model, open(path, 'w'))


def load(path):
    return pickle.load(open(path, 'r'))


def model_output_path(store_models, methodname, foldind):
    return os.path.join(store_models, methodname + str(foldind)
                        + '.pick')


def print_matrix(mat):
    for row in mat:
        print row.tolist()


