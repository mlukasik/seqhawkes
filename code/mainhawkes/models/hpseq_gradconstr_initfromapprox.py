#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 15 Dec 2015

@author: michal

Initialize parameters from Approximate optimization
and then do gradient based optimization of the parameters.
'''

import numpy as np
from models.hpseq_gradconstr import HPSeqFullSumGradConstr
import updates
from models.hpseq_approxsummation import HPSeqFullSumApprox


class HPSeqFullSumGradConstrInitFromApprox(HPSeqFullSumGradConstr):

    '''
    Initialize parameters from Approximate optimization
    and then do gradient based optimization of the parameters.
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
        init_omega=0.1,
        init_gamma=[],
        init_params_randomly=False,
        kernel_evaluate=updates.kernel_zha.kernel_evaluate,
        K_evaluate=updates.kernel_zha.K_evaluate,
        derivative_kernel_evaluate=updates.kernel_zha.derivative_kernel_evaluate,
        derivative_K_evaluate=updates.kernel_zha.derivative_K_evaluate,
        M=None,
        ):

        hpseqapprox = HPSeqFullSumApprox(
            etimes,
            node_vec,
            ememes,
            infected_vec,
            infecting_vec,
            W,
            T,
            V,
            D,
            ['mu', 'alpha', 'beta'],
            iterations=5,
            verbose=False,
            init_omega=0.1,
            init_gamma=np.array([1 for _ in xrange(max(len(set(ememes))
                                + 1, M))]),
            init_params_randomly=False,
            M=M,
            )
        print '[HPSeqFullSumGradConstrInitFromApprox.__init__] START Training'
        hpseqapprox.train()
        print '[HPSeqFullSumGradConstrInitFromApprox.__init__] DONE Training'

        super(HPSeqFullSumGradConstrInitFromApprox, self).__init__(
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
            derivative_kernel_evaluate=derivative_kernel_evaluate,
            derivative_K_evaluate=derivative_K_evaluate,
            M=M,
            )

        self.mu = hpseqapprox.mu
        self.gamma = hpseqapprox.gamma
        self.alpha = hpseqapprox.alpha
        self.beta = hpseqapprox.beta


