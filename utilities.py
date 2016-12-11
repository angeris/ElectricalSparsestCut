import networkx as nx
# import cvxpy as cvx
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt

def draw_partitions(G, partitions):
    colors = [partitions[i] for i in sorted(partitions.keys())]
    nx.drawing.draw_networkx(G, node_color = colors)
    plt.show()
