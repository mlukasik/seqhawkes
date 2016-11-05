#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Copyright (c) 2014-2015, The University of Sheffield. 
This file is part of the seqhawkes software 
(see https://github.com/mlukasik/seqhawkes), 
and is free software, licenced under the GNU Library General Public License,
Version 2, June 1991 (in the distribution as file LICENSE).

Created on 20 Jul 2015

@author: michal

Functions related to kernel functionalities.
'''

import numpy as np


def kernel_evaluate(etimes1, etimes2, omega):
    return omega * np.exp(-omega * (etimes1 - etimes2))


def K_evaluate(etimes, T, omega):
    K = 1 - np.exp(-omega * (T - etimes))
    return K


def derivative_kernel_evaluate(etimes1, etimes2, omega):
    return np.exp(-omega * (etimes1 - etimes2)) - omega * (etimes1
            - etimes2) * np.exp(-omega * (etimes1 - etimes2))


def derivative_K_evaluate(etimes, T, omega):
    return -np.exp(-omega * (T - etimes)) * -(T - etimes)


