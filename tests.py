import algorithms
import networkx as nx
# import cvxpy as cvx
import numpy as np
import scipy as sp
import math
import utilities
import matplotlib.pyplot as plt
from copy import copy
import random
import csv
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

def get_approx_ratio_basecase():
    with open('cut_ratio_values_small.csv', 'a') as f:
        writer = csv.DictWriter(f, delimiter = ',', fieldnames = ['n', 'max_flow', 'electrical_alg', 'ratio'])
        # writer.writeheader()
        for i in range(1000000):
            n = np.random.randint(low = 10, high = 4000);
            print i,n,
            try:
                G, constraints = algorithms.generate_graphs_with_constraints(n = n, m = 2, k = 2)
                partitions_maxflow, maxflowcut = algorithms.max_flow_cut(G.copy(), copy(constraints), 2)
                partitions_flowcut, voltages, idxarr, flow_cut = algorithms.flow_cut(G.copy(), copy(constraints), 2)
                print flow_cut, maxflowcut, flow_cut/maxflowcut
                writer.writerow({'n' : n, 'max_flow' : maxflowcut, 'electrical_alg' : flow_cut, 'ratio' : flow_cut/maxflowcut})
            except Exception:
                pass
get_approx_ratio_basecase()

# test_graph_generation() # test_max_flow_cut() test_sdp_partition()
# test_all_algorithms()
