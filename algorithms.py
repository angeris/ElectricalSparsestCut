# Algorithms library for testing

import networkx as nx
# import cvxpy as cvx
import numpy as np
import scipy as sp
import math

def generate_graphs_with_constraints(n = 100, k = 2, m = 2):
    if m < k:
        raise Exception('m (number of constraints) less than k (number of clusters)')
    G = nx.gnp_random_graph(n, math.pow(math.log(n),1)/n) #erdos renyi graph that is probably connected
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
    cut_edges = nx.algorithms.connectivity.minimum_st_edge_cut(graph, keys[0], keys[1])
    partition = {}
    graph_copy.remove_edges_from(cut_edges)
    for con in constraints:
        for node in nx.algorithms.components.node_connected_component(graph_copy, con):
            partition[node] = constraints[con]

    return partition

def sdp_partition(graph, constraints, k):
    raise NotImplementedError()

def flow_cut(graph, constraints, k):
    raise NotImplementedError()
