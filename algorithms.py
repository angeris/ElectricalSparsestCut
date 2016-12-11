# Algorithms library for testing

import networkx as nx
# import cvxpy as cvx
import numpy as np
import scipy as sp
import math
import util
import scipy.sparse.linalg as linalg

def generate_graphs_with_constraints(n = 100, k = 2, m = 2):
    if m < k:
        raise Exception('m (number of constraints) less than k (number of clusters)')
    G = nx.fast_gnp_random_graph(n, 10*math.pow(math.log(n),3)/n) #erdos renyi graph that is probably connected
    G = nx.convert_node_labels_to_integers(max(nx.connected_component_subgraphs(G), key=len)) #returns largest connected component
    constraints = {}
    for i,x in enumerate(np.random.choice(len(G), size=m)):
        if i < k:
            constraints[x] = i
        else:
            constraints[x] = np.random.randint(k)
    return G, constraints

def brute_force(graph, constraints, k):
    raise NotImplementedError()

def max_flow_cut(graph, constraints, k):
    if k!=2:
        raise Exception('Max flow only applicable for 2 partitions')
    if len(constraints.keys())!=2:
        raise Exception('Max flow only applicable with 2 constraints')

    graph_copy = graph.copy()
    keys = list(constraints.keys())
    print (keys)
    cut_edges = nx.algorithms.connectivity.minimum_st_edge_cut(graph, keys[0], keys[1])

    partition = {}
    graph_copy.remove_edges_from(cut_edges)
    for con in constraints:
        print('hi', con)
        for node in nx.algorithms.components.node_connected_component(graph_copy, con):
            print (node),
            partition[node] = constraints[con]

    return partition

def sdp_partition(graph, constraints, k):
    raise NotImplementedError()

def flow_cut(graph, constraints, k=2, max_iter=100, tol=1e-5, verbose=True):
    if k!=2: raise NotImplementedError()
    N = len(graph)

    # form contraction mapping
    condmat = nx.adjacency_matrix(graph).asfptype()
    np.reciprocal(condmat.data, out=condmat.data)
    c_mat = sp.sparse.diags(np.asarray(1./np.sum(condmat, 1)).flatten(), 0)
    condmat = c_mat*condmat

    nodes_list = graph.nodes()
    ntoidx = {n:i for i,n in enumerate(nodes_list)}

    init_vector = np.zeros(N)

    for v, val in constraints.iteritems():
        ci = ntoidx[v]
        condmat[ci, :] = util.unit_basis(N, ci)
        init_vector[ci] = val

    new_vec, prev_vec = None, init_vector.reshape(N,1)
    l_iter = 5*int(math.ceil(math.log10(max_iter)))

    for i in range(max_iter):
        new_vec = condmat*prev_vec
        curr_tol = np.max(new_vec-prev_vec)
        if (i+1)%l_iter==0 and verbose:
            print 'curr_norm : {} on iteration {}/{}'.format(curr_tol, i+1,
                                                             max_iter)
        if curr_tol < tol:
            break
        prev_vec = new_vec

