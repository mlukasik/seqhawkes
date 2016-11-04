#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on 4 May 2015

@author: michal

Loading of data and converting to appropriate format.
'''

import numpy as np


def load(fname):
    '''
    Read and yield data records.
    '''

    with open(fname) as f:
        for l in f:
            l = l.split()
            memeid = int(l[0])
            eventid = int(l[1])
            infectingid = int(l[2])
            nodeid = int(l[3])
            time = float(l[4])
            wordslen = int(l[5])
            words = {}
            for wordidx in xrange(wordslen):
                word = l[6 + wordidx]
                word = map(int, word.split(':'))
                words[word[0]] = word[1]
            yield (
                memeid,
                eventid,
                infectingid,
                nodeid,
                time,
                words,
                )


def loadX(fname):
    '''
    Read data records into a data matrix.
    Also return vocabulary.
    '''

    events = []
    words_keys = set()
    for e in load(fname):
        events.append(e)
        words_keys = words_keys | set(e[5].keys())
    words_keys = sorted(list(words_keys))
    for (eidx, e) in enumerate(events):
        events[eidx] = list(e[:5]) + [e[5].get(word_key, 0)
                for word_key in words_keys]
    X = np.vstack(events)
    return (X, words_keys)


