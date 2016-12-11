# Algorithms library for testing

import networkx as nx
import cvxpy as cvx
import numpy as np
import scipy as sp

import scipy.sparse.linalg as linalg

import util

def generate_graphs_with_constraints():
    raise NotImplementedError()

def brute_force(graph, constraints, k):
    raise NotImplementedError()

def min_cut(graph, constraints, k):
    raise NotImplementedError()

def sdp_partition(graph, constraints, k):
    raise NotImplementedError()

def flow_cut(graph, constraints, k=2):
    if k!=2: raise NotImplementedError()
    # form contraction mapping
    adjmat = nx.adjacency_matrix(graph).asfptype()
    degrees = np.array(list(float(graph.degree(v)) for v in graph))
    print degrees
    adjmat /= degrees[:,None]

    nodes_list = graph.nodes()
    N = len(nodes_list)
    ntoidx = {n:i for i,n in enumerate(nodes_list)}

    init_vector = np.zeros(N)

    for v, val in constraints.iteritems():
        ci = ntoidx[v]
        adjmat[ci, :] = util.unit_basis(N, ci)
        init_vector[ci] = val

    new_vec, prev_vec = None, init_vector
    for i in range(20):
        new_vec = adjmat.dot(prev_vec)
        print np.max(new_vec-prev_vec)
    print new_vec

