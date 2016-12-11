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
    # G = nx.gnp_random_graph(n, math.pow(math.log(n),1.5)/n) #erdos renyi graph that is probably connected
    G = nx.connected_watts_strogatz_graph(n, k=5, p=np.log(n)/n, tries=100, seed=None)
    G = nx.convert_node_labels_to_integers(max(nx.connected_component_subgraphs(G), key=len)) #returns largest connected component
    constraints = {}
    for i,x in enumerate(np.random.choice(len(G), size=m)):
        if i < k:
            constraints[x] = i
        else:
            constraints[x] = np.random.randint(k)

    for (u, v) in G.edges():
        weight = np.random.rand()
        G.edge[u][v]['weight'] = weight
        G.edge[v][u]['weight'] = weight

    #make constraint edges full connected:
    # for con in constraints:
    #     for v in G.nodes():
    #         if con!=v:
    #             if np.random.rand() < 1:
    #                 G.add_edge(con, v,{'weight':1})
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
    cut_value, partial = nx.minimum_cut(graph_copy, keys[0], keys[1], capacity = 'weight')
    reachable, non_reachable = partial
    cutset = set()
    for u, nbrs in ((n, graph_copy[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)

    partition = {}
    graph_copy.remove_edges_from(cutset)
    for con in constraints:
        for node in nx.algorithms.components.node_connected_component(graph_copy, con):
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
