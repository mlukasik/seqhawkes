#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Copyright (c) 2014-2015, The University of Sheffield. 
This file is part of the seqhawkes software 
(see https://github.com/mlukasik/seqhawkes), 
and is free software, licenced under the GNU Library General Public License,
Version 2, June 1991 (in the distribution as file LICENSE).

Created on 14 May 2015

@author: michal

Gathering raw data statistics useful for the HP model.
'''

import numpy as np
import scipy.sparse as sp
from itertools import izip


def spontaneousnode_count(
    infecting_vec,
    infected_vec,
    node_vec,
    D,
    ):
    '''
    Returns a vector with count of spontanous infections for nodes.
    
    Arguments:
    infecting_vec - vector of infecting event ids
    infected_vec - vector of event ids
    node_vec - vector of infected node ids
    D - number of nodes
    '''

    spontaneous_nodes = node_vec[infecting_vec == infected_vec]
    updates = np.zeros((D, 1))
    for node in spontaneous_nodes:
        updates[int(node)] += 1
    return updates


def spontaneousmeme_count(
    infecting_vec,
    infected_vec,
    eventmemes,
    M,
    ):
    '''
    Returns a vector with count of spontanous infections for memes.
    
    Arguments:
    infecting_vec - vector of infecting event ids
    infected_vec - vector of event ids
    eventmemes - vector of meme ids
    M - number of memes
    '''

    spontaneous_memes = eventmemes[infecting_vec == infected_vec]
    updates = np.zeros((M, 1))
    for meme in spontaneous_memes:
        updates[int(meme)] += 1
    return updates


def infecting_node(infected_vec, infecting_vec, node_vec):
    '''
    Returns a vector of nodes of infecting events.
    
    Arguments:
    infecting_vec - vector of infecting event ids
    infected_vec - vector of event ids
    node_vec - vector of infected node ids
    '''

    infecting_node_vec = []
    eventid_to_node = {}

    for (evid, inf_evid, nodeid) in izip(infected_vec, infecting_vec,
            node_vec):
        eventid_to_node[int(evid)] = nodeid
        infecting_node_vec.append(eventid_to_node[int(inf_evid)])
    infecting_node_vec = np.array(infecting_node_vec).flatten()
    return (infecting_node_vec, eventid_to_node)


def infections_count(
    infecting_node,
    infected_node,
    infecting_vec,
    infected_vec,
    D,
    ):
    '''
    For each pair of nodes counts infections between them.    
    
    infecting_node - vector, mapping events to nodes of events that infected 
        them
    infected_node - vector of integers, mapping events to nodes where they 
        occurred
    infecting_vec - vector, mapping events to ids of events that infected them
    infected_vec - vector of integers, mapping events to ids
    D - number of nodes
    
    returns: matrix of counts
    '''

    infections_mat = sp.lil_matrix((D, D), dtype=np.int)
    for (infected_u, infecting_u, infected_e, infecting_e) in \
        izip(infected_node, infecting_node, infected_vec,
             infecting_vec):
        if infected_e != infecting_e:
            infections_mat[infecting_u, infected_u] += 1
    return infections_mat


def one_hot_sparse(index_array, num_values):
    m = sp.lil_matrix((num_values, index_array.shape[0]), dtype=np.bool)
    for i in range(index_array.shape[0]):
        m[index_array[i], i] = 1
    return m.tocsr()


def one_hot(index_array, num_values):
    m = np.zeros(shape=(num_values, index_array.shape[0]),
                 dtype=np.bool)
    m[index_array.astype(int), range(index_array.shape[0])] = 1
    return m


