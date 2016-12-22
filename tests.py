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
import pandas as pd
import time

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

def test_CalinescuKarloffRabani():
    k = 4
    G, constraints = algorithms.generate_graphs_with_constraints(n = 100, m = k, k = k)
    partitions = algorithms.CalinescuKarloffRabani(G, constraints, k)
    print (partitions)
    # utilities.draw_partitions(G, partitions)

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

def higher_k_alltests():
    df = pd.read_csv('higher_k_cut_ratio_values_withtime.csv', index_col = False)
    done = df.pivot_table(index='n', columns='k', values='ratioCKRElectrical')
    with open('higher_k_cut_ratio_values_withtime.csv', 'a') as f:
        writer = csv.DictWriter(f, delimiter = ',', fieldnames = ['n', 'k', 'brute_force', 'CalinescuKarloffRabani', 'electrical_alg', 'ratio_BruteCKR', 'ratio_BruteElectrical', 'ratioCKRElectrical', 'brute_time', 'ckr_time', 'elec_time'])
        # writer.writeheader()
        # for i in range(1000000):
        #     n = np.random.randint(low = 10, high = 300);
        #     k = np.random.randint(low = 2, high = min(n/3+1, 15))
        for n in range(10, 300):
            for k in range(2, min(n/2+1, 21)):
                if n in done.index.values and k in done.columns and (not np.isnan(done.loc[n,k])):
                    print 'skipping {},{} because done'.format(n,k)
                    # continue
                else:
                    print n,k
                    try:
                        G, constraints = algorithms.generate_graphs_with_constraints(n = n, m = k, k = k)
                        # print k, len(constraints)
                        start = time.time()
                        partitions_CKR, CKRcut = algorithms.CalinescuKarloffRabani(G.copy(), copy(constraints), k)
                        ckr_time = time.time() - start
                        start = time.time()
                        partitions_flowcut, flow_cut = algorithms.voltage_cut_wrapper(G.copy(), copy(constraints), algorithms.both_greedy_and_random_cut, k, verbose = False)
                        elec_time = time.time() - start
                        dic = {'n' : n, 'k' : k, 'CalinescuKarloffRabani' : CKRcut, 'electrical_alg' : flow_cut, 'ratioCKRElectrical' : flow_cut/CKRcut, 'ckr_time' : ckr_time, 'elec_time':elec_time}
                        if n < 15 and k < 5:
                            start = time.time()
                            brutecut = algorithms.brute_force(G.copy(), copy(constraints), k)
                            brute_time = time.time() - start
                            dic['brute_force'] = brutecut
                            dic['ratio_BruteCKR'] = CKRcut/brutecut
                            dic['ratio_BruteElectrical']= flow_cut/brutecut
                            dic['brute_time']= brute_time

                        print dic

                        writer.writerow(dic)
                    except Exception:
                        print "exception"
                        pass

higher_k_alltests()
# test_CalinescuKarloffRabani();
# get_approx_ratio_basecase()

# test_graph_generation() # test_max_flow_cut() test_sdp_partition()
# test_all_algorithms()
