#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
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


