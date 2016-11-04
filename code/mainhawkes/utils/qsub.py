#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 23 Mar 2015

@author: michal

Code for handling parallelization of experiments. 

Thanks to Zsolt Bitvai for the idea and the sample code.
'''

import socket
from subprocess import Popen, PIPE, STDOUT
import time
import os
import getpass


def ON_SERVER():
    return True #getpass.getuser() == 'acp13ml'


def LOCALLY():
    return not ON_SERVER()


class Qsub(object):

    '''
    Class for submitting jobs to a server. 
    '''

    QSUB_HEADER_EASY = '\n'.join([
        '#!/bin/bash',
        '#$ -m a',
        '#$ -M acp13ml@sheffield.ac.uk$ -l h_rt=6:00:00',
        '#$ -o result.log',
        '#$ -j y',
        '#$-l mem=10G',
        '#$-l rmem=10G',
        ])

    QSUB_HEADER_EASY_LONG = '\n'.join([
        '#!/bin/bash',
        '#$ -m a',
        '#$ -M acp13ml@sheffield.ac.uk$ -l h_rt=16:00:00',
        '#$ -o result.log',
        '#$ -j y',
        '#$-l mem=10G',
        '#$-l rmem=10G',
        ])

    QSUB_HEADER_HARD = '\n'.join([
        '#!/bin/bash',
        '#$ -m a',
        '#$ -M acp13ml@sheffield.ac.uk$ -l h_rt=12:00:00',
        '#$ -o result.log',
        '#$ -j y',
        '#$-l mem=40G',
        '#$-l rmem=30G',
        ])

    QSUB_HEADER_MOREHARD = '\n'.join([
        '#!/bin/bash',
        '#$ -m a',
        '#$ -M acp13ml@sheffield.ac.uk$ -l h_rt=12:00:00',
        '#$ -o result.log',
        '#$ -j y',
        '#$-l mem=60G',
        '#$-l rmem=40G',
        ])
    script_name = 'tmp.sh'
    SUBM_TIMEOUT = 0.02
    WAIT_TIMEOUT = 1
    WAIT_FREQ = 20

    def __init__(
        self,
        CVFOLDS,
        methodnames,
        pyscript,
        global_parameters,
        outcat,
        train_set_ratios,
        experimenttype,
        ):
        print '[Qsub.__init__]'
        self.cnt = 0
        self.CVFOLDS = CVFOLDS
        self.methodnames = methodnames
        self.pyscript = pyscript
        self.global_parameters = map(str, global_parameters)

        if 'iceberg' in socket.gethostname():
            self.SUB_CMD = 'qsub {}'
            self.MAX_WORKERS = 1999  # 350
            self.WORKER_CMD = 'Qstat'
            self.USERNAME = 'acp13ml'
            self.RUN_FOREGROUND = True
        elif 'yarra' in socket.gethostname():
            self.SUB_CMD = 'sh {} &'
            self.MAX_WORKERS = 30
            self.WORKER_CMD = 'ps'
            self.USERNAME = '\n'
            self.RUN_FOREGROUND = False
        else:
            print '[Qsub] Unknown host: just printing out the script content!'
            self.SUB_CMD = 'cat {}'
            self.MAX_WORKERS = 999999
            self.WORKER_CMD = 'ps'
            self.USERNAME = '\n'
            self.RUN_FOREGROUND = True
        self.outcat = outcat

        if experimenttype == 'easy':
            self.QSUB_HEADER = self.QSUB_HEADER_EASY
        elif experimenttype == 'hard':
            self.QSUB_HEADER = self.QSUB_HEADER_HARD
        elif experimenttype == 'easylong':
            self.QSUB_HEADER = self.QSUB_HEADER_EASY_LONG
        elif experimenttype == 'morehard':
            self.QSUB_HEADER = self.QSUB_HEADER_MOREHARD
        else:
            print 'Wrong experimenttype:', experimenttype, '!!!'

        self.train_set_ratios = train_set_ratios

    def iterate_settings(self, actions):
        for foldrun in xrange(self.CVFOLDS):
            for methodname in self.methodnames:
                for train_set_ratio in self.train_set_ratios:
                    for action in actions:
                        action(map(str, [foldrun, methodname,
                               train_set_ratio]))

    def inc_count(self, run_parameters):
        self.cnt += 1

    def create_qsub_script(self, run_parameters):

        # create script

        with open(self.script_name, 'w') as f:
            f.write(self.QSUB_HEADER + '\n')
            res_fname = os.path.join(self.outcat,
                    '_'.join(run_parameters))

            f.write(' '.join(['python', self.pyscript] + run_parameters
                    + self.global_parameters + ['>'] + [res_fname]))

    def submit_qsub_script(self, run_parameters):
        print self._execute(self.SUB_CMD.format(self.script_name))

    def print_progress_bar(self, run_parameters):
        print '/'.join(map(str, [self.cnt, self.all_terations]))

    def _execute(self, cmd):
        if self.RUN_FOREGROUND:
            outp = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE,
                         stderr=STDOUT).communicate()[0]
        else:
            outp = Popen(cmd.split(), stdin=PIPE, stderr=STDOUT)  # .communicate()[0]
        time.sleep(self.SUBM_TIMEOUT)
        return outp

    def get_running_workers(self):
        output = Popen(self.WORKER_CMD, shell=True, stdin=PIPE,
                       stdout=PIPE, stderr=STDOUT).communicate()[0]
        return output.count(self.USERNAME)

    def wait(self, max_workers=None):
        if max_workers == None:
            max_workers = self.MAX_WORKERS
        while self.get_running_workers() > max_workers:
            print 'max workers exceeded, wating...'
            time.sleep(self.WAIT_TIMEOUT)

    def wait_condition(self, run_parameters):
        if self.cnt % self.WAIT_FREQ == 0:
            self.wait()

    def run(self):
        print '[Qsub.run]'
        self.iterate_settings([self.inc_count])
        self.all_terations = self.cnt
        print 'Iterations to run:', self.all_terations

        self.cnt = 0
        self.iterate_settings([self.create_qsub_script,
                              self.submit_qsub_script, self.inc_count,
                              self.print_progress_bar,
                              self.wait_condition])


