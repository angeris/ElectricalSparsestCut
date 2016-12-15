import algorithms
import networkx as nx
# import cvxpy as cvx
import numpy as np
import scipy as sp
import math
import utilities

def test_graph_generation():
    G, constraints = algorithms.generate_graphs_with_constraints(n = 200, m = 10, k = 5)
    print(G)
    print ([len(x) for x in nx.connected_component_subgraphs(G)])
    print (constraints)

def test_max_flow_cut():
    G, constraints = algorithms.generate_graphs_with_constraints(n = 100, m = 2, k = 2)
    partitions = algorithms.max_flow_cut(G, constraints, 2)
    print (partitions)
    utilities.draw_partitions(G, partitions)

def test_sdp_partition():
    G, constraints = algorithms.generate_graphs_with_constraints(n = 100, m = 2, k = 2)
    partitions = algorithms.sdp_partition(G, constraints, 2)
    print (partitions)
    utilities.draw_partitions(G, partitions)
# test_graph_generation()
# test_max_flow_cut()
test_sdp_partition()

def test_all_algorithms():
    G, constraints = algorithms.generate_graphs_with_constraints(n = 100, m = 2, k = 2)
    partitions_maxflow = algorithms.max_flow_cut(G, constraints, 2)
    partitions_sdp = algorithms.sdp_partition(G, constraints, 2)
    partitions_flowcut = algorithms.flow_cut(G, constraints, 2)

    utilities.draw_partitions(G, partitions_maxflow)
    plt.savefig('maxflow.png')

    utilities.draw_partitions(G, partitions_sdp)
    plt.savefig('sdp.png')

    utilities.draw_partitions(G, partitions_flowcut)
    plt.savefig('flowcut.png')
