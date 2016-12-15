import algorithms as alg
import networkx as nx

A = nx.fast_gnp_random_graph(10, .3)

alg.flow_cut(A)
