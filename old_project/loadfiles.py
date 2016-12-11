from scipy.sparse import dok_matrix
from collections import defaultdict

import os
import numpy as np
import scipy as sp
import scipy.linalg as linalg

def load_dict_network(dataset_dir='./dataset'):
    """Returns a dictionary of connections and a set
    of all appearing people
    """
    all_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.egonet')]
    all_connections = defaultdict(set)
    all_people = set()
    for curr_f in all_files:
        f = open(curr_f)
        for line in f:
            f_split = line.strip().split(':')
            curr_key = int(f_split[0])
            all_people.add(curr_key)
            all_friends = f_split[1].strip().split()
            for friend in all_friends:
                friend = int(friend)
                all_connections[curr_key].add(friend)
                all_people.add(friend)
    return all_connections, all_people

def dict_to_sparse(d_connections, all_people):
    """Takes a dictionary of connections and turns it
    into a sparse matrix for later usage
    """
    N = len(all_people)
    to_idx = {p:i for i, p in enumerate(all_people)}
    dok = {}
    for p, s in d_connections.iteritems():
        for f in s:
            dok[(to_idx[p], to_idx[f])] = 1
            dok[(to_idx[f], to_idx[p])] = 1
    adj_mat = dok_matrix((N,N))
    adj_mat.update(dok)
    return adj_mat
