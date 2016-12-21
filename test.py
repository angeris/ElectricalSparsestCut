import algorithms as alg
import networkx as nx

g = nx.barbell_graph(5, 5)
for u, v, w in g.edges(data=True):
    w['weight'] = 1

cut = alg.brute_force(g, {0:0, 10:1})
mcut = alg.max_flow_cut(g, {0:0, 10:1}, 2)
vcut = alg.voltage_cut_wrapper(g, {0:0, 10:1}, alg.greedy_cut)

print cut, mcut
