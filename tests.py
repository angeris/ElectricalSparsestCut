import algorithms
import networkx as nx
# import cvxpy as cvx
import numpy as np
import scipy as sp
import math
import utilities
import matplotlib.pyplot as plt
from copy import copy

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
# test_graph_generation() # test_max_flow_cut() test_sdp_partition()

def test_all_algorithms():
    G, constraints = algorithms.generate_graphs_with_constraints(n = 1000, m = 2, k = 2)
    print 'constraints are : {}'.format(constraints)
    partitions_maxflow = algorithms.max_flow_cut(G.copy(), copy(constraints), 2)
    # partitions_sdp = algorithms.sdp_partition(G.copy(), copy(constraints), 2)
    partitions_flowcut, voltages, idxarr = algorithms.flow_cut(G.copy(), copy(constraints), 2)

    # for v in G:
        # print 'set in maxflow {} | set in flowcut {}'.format(partitions_maxflow[v], partitions_flowcut[v])

    for v in idxarr:
        print 'voltage {} | set {}'.format(voltages[v], partitions_maxflow[v])

    # utilities.draw_partitions(G, partitions_maxflow)
    # plt.savefig('maxflow.png')
    # plt.close()

    # # utilities.draw_partitions(G, partitions_sdp)
    # # plt.savefig('sdp.png')
    # # plt.close()

    # utilities.draw_partitions(G, partitions_flowcut)
    # plt.savefig('flowcut.png')
    # plt.close()

test_all_algorithms()
