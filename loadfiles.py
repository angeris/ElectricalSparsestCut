from scipy.sparse import dok_matrix
from collections import defaultdict

import os
import numpy as np
import scipy as sp
import scipy.linalg as linalg

def load_dict_network(dataset_dir='./dataset'):
    """Returns a dictionary of connections
    """
    all_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.egonet')]
    all_connections = defaultdict(set)
    for curr_f in all_files:
        f = open(curr_f)
        for line in f:
            f_split = line.strip().split(':')
            curr_key = int(f_split[0])
            all_friends = f_split[1].strip().split()
            for friend in all_friends:
                friend = int(friend)
                all_connections[curr_key].add(friend)
    return all_connections

def dict_to_sparse(d_connections):
    N = len(d_connections)
    to_idx = {p:i for i, p in enumerate(d_connections.keys())}
    dok = {(to_idx(curr_f),to_idx(neigh_f)):1 for neigh_f in fs for curr_f, fs in d_connections.iteritems()}
    adj_mat = dok_matrix((N,N))
    adj_mat.update(dok)
    return adj_mat
