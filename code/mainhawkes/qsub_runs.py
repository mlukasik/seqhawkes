#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 23 Mar 2015

@author: michal

Submitting jobs for different folds. Useful when running experiments in 
parallel.
'''

import sys
from utils.utils import make_dir, current_time_str
import os
from utils.qsub import Qsub
from header_methods import get_methodnames
import socket


def create_res_catalogue(outcat_prefix, RESULTS_CAT, global_parameters):
    outcat = os.path.join(RESULTS_CAT, outcat_prefix
                          + '_'.join(global_parameters).replace('/', ''
                          ).replace('.', ''))
    make_dir(outcat)
    return outcat


if __name__ == '__main__':
    CVFOLDS = int(sys.argv[1])
    fname = sys.argv[2]
    train_set_ratios = sys.argv[3].split(',')
    experimenttype = sys.argv[4]
    if len(sys.argv) >= 6:
        outcat = sys.argv[5]
    else:
        outcat_prefix = current_time_str()
        if 'iceberg' in socket.gethostname():
            RESULTS_CAT = '/data/acp13ml/seqhawkes_results'
        elif 'yarra' in socket.gethostname():
            RESULTS_CAT = 'results'
        else:
            RESULTS_CAT = 'results'
        outcat = create_res_catalogue(outcat_prefix, RESULTS_CAT,
                [fname])

    methodnames = get_methodnames()

    qs = Qsub(
        CVFOLDS,
        methodnames,
        'main.py',
        [fname],
        outcat,
        train_set_ratios,
        experimenttype,
        )
    qs.run()
    qs.wait(0)
