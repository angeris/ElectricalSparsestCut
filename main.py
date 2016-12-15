# Main running thigie

import networkx as nx
import algorithms as alg
import utilities as util

g = nx.barbell_graph(40, 40)

partitions = alg.flow_cut(g, {0:0, 119:1})

util.draw_partitions(g, partitions)
