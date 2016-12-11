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

def flow_cut(graph, constraints, k=2, verbose=True):
    if k!=2: raise NotImplementedError()
    # form contraction mapping
    adjmat = nx.adjacency_matrix(graph).asfptype()
    degrees = np.array(list(float(graph.degree(v)) for v in graph))
    adjmat /= degrees[:,None]

    nodes_list = graph.nodes()
    N = len(nodes_list)
    ntoidx = {n:i for i,n in enumerate(nodes_list)}

    init_vector = np.zeros(N)

    for v, val in constraints.iteritems():
        ci = ntoidx[v]
        adjmat[ci, :] = util.unit_basis(N, ci)
        init_vector[ci] = val

    new_vec, prev_vec = None, init_vector.reshape(10,1)

    for i in range(100):
        new_vec = adjmat*prev_vec
        prev_vec = new_vec
        if i%10==0 and verbose:
            print 'curr_norm : {}'.format(np.max(new_vec - prev_vec))

    print new_vec

