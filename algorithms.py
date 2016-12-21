# Algorithms library for testing

import networkx as nx
import cvxpy as cvx
import numpy as np
import scipy as sp
import math
import util
import scipy.sparse.linalg as linalg
from tqdm import tqdm
from itertools import product
from copy import copy

range = xrange

def e_boundary(graph, v_list, data='weight'):
    edges = graph.edges(v_list, data=data, default=1)
    return (e for e in edges if e[1] not in v_list)

def cut_weight(graph, v_list, data='weight'):
    edges = e_boundary(graph, v_list, data=data)
    return sum(w for u, v, w in edges)

def evaluate(graph, partitions, k):
    total_weight = 0
    partition_list = [set() for _ in range(k)]
    for k, v in partitions.iteritems():
        partition_list[v].add(k)
    for p in partition_list:
        total_weight += cut_weight(graph, p)
    return total_weight

def generate_graphs_with_constraints(n = 100, k = 2, m = 2):
    if m < k:
        raise Exception('m (number of constraints) less than k (number of clusters)')
    # G = nx.gnp_random_graph(n, math.pow(math.log(n),1.5)/n) #erdos renyi graph that is probably connected
    # G = nx.powerlaw_cluster_graph(n,int(np.log(n)), 3.0/int(np.log(n)) )
    G = nx.connected_watts_strogatz_graph(n, k=5, p=1.25*np.log(n)/n, tries=100, seed=None)
    G = nx.convert_node_labels_to_integers(max(nx.connected_component_subgraphs(G), key=len)) #returns largest connected component
    constraints = {}
    for i,x in enumerate(np.random.choice(len(G), size=m)):
        if i < k:
            constraints[x] = i
        else:
            constraints[x] = np.random.randint(k)

    for (u, v) in G.edges():
        weight = np.random.rand()
        G.edge[u][v]['weight'] = weight
        G.edge[v][u]['weight'] = weight

    #make constraint edges full connected:
    # for con in constraints:
    #     for v in G.nodes():
    #         if con!=v:
    #             if np.random.rand() < 1:
    #                 G.add_edge(con, v,{'weight':1})
    return G, constraints

def brute_force(graph, constraints, k):
    raise NotImplementedError()


# D = cvx.Variable((n,n))
# Dprime = cvx.Variable((n,n,n))
# obj = cvx.Minimize(cvx.sum_entries(cvx.mul_elemwise(adjmat,D)))
# consts = [D == D.T, D >= 0, D<=1]
# for i in range(n):
#     consts.append(D[i,i] == 0)
# for u in range(n):
#     for v in range(u+1, n):
#         for w in range(v+1, n):
#             conts.append(D[u,v] <= D[v,w] + D[w,u])
# for c1 in constraints.keys():
#     for c2 in constraints.keys():
#         if constraints[c1]!=constraints[c2]:
#             consts.append(D[c1,c2] = 1)
# for u in range(n):
#     s = 0
#     for c in constraints.keys():
#         s+= D[u, c]
#     consts.append(s == k-1)
def CalinescuKarloffRabani(graph, constraints, k):
    adjmat = nx.to_numpy_matrix(graph, weight = 'weight')
    n = nx.number_of_nodes(graph)
    X = []
    fun = 0
    consts = []
    for i in range(n):
        x = cvx.Variable(k)
        consts.extend([x >= 0, cvx.sum_entries(x) == 1])
        X.append(x)
    for c in constraints.keys():
        ei = np.zeros(k)
        ei[constraints[c]] = 1
        consts.append(X[c] == ei)
    for u in range(n):
        for v in range(u+1, n):
            fun+=adjmat[u,v]*cvx.norm(X[u] - X[v], 1)

    obj = cvx.Minimize(fun)

    prob = cvx.Problem(obj, consts)
    prob.solve()
    for i,x in enumerate(X):
        # print np.transpose(x.value)
        X[i] = np.transpose(np.asarray(x.value))

    #random cutting for now
    mincutweight = np.finfo(float).max
    best_partition = {}
    for i in range(10):
        partition = {}
        p = np.random.rand()
        korder = list(range(0, k-1))
        if np.random.rand() < .5:
            korder = list(reversed(korder))
        for u in range(n):
            partition[u] = k-1;
            for cluster in korder:
                if X[u][0,cluster] > p:
                    partition[u] = cluster
                    break
        cutweight = evaluate(graph, partition, k)
        # print i,p,cutweight
        if cutweight < mincutweight:
            mincutweight = cutweight
            best_partition = partition
    return best_partition
    #------------------
    # random_cut = sample_spherical(1, ndim = n)
    # partition = {}
    # signs = []
    # for i in range(n):
    #     signs.append(np.sign(np.dot(vecs[i,:], random_cut)))
    # for con in constraints:
    #     for i in range(n):
    #         if signs[con] == signs[i]:
    #             partition[i] = constraints[con]
    # return partition

def max_flow_cut(graph, constraints, k):
    if k!=2: raise Exception('Max flow only applicable for 2 partitions')
    if len(constraints.keys())!=2: raise Exception('Max flow only applicable with 2 constraints')

    graph_copy = graph.copy()
    keys = list(constraints.keys())
    cut_value, partial = nx.minimum_cut(graph_copy, keys[0], keys[1], capacity = 'weight')
    reachable, non_reachable = partial
    cutset = set()
    for u, nbrs in ((n, graph_copy[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)

    partition = {}
    graph_copy.remove_edges_from(cutset)
    for con in constraints:
        for node in nx.algorithms.components.node_connected_component(graph_copy, con):
            partition[node] = constraints[con]

    # cutweight = cut_weight(graph, {v for v,k in partition.iteritems() if k==0}, data='invweight')
    print 'max flow weight : {}'.format(evaluate(graph, partition, 2))
    return partition

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def sdp_partition(graph, constraints, k):
    if k!=2: raise Exception('SDP only applicable for 2 partitions')
    adjmat = nx.to_numpy_matrix(graph, weight = 'weight')
    n = nx.number_of_nodes(graph)
    Y = cvx.Semidef(n)
    obj = cvx.Maximize(cvx.sum_entries(cvx.mul_elemwise(np.tril(adjmat),(1-Y))))
    consts = [Y == Y.T]
    for i in range(n):
        consts.append(Y[i,i] == 1)
    for c1 in constraints.keys():
        for c2 in constraints.keys():
            if constraints[c1]!=constraints[c2]:
                consts.append(Y[c1,c2] <= math.cos(2*math.pi/k))
            else:
                consts.append(Y[c1, c2] == 1)
    prob = cvx.Problem(obj, consts)
    prob.solve(solver = 'SCS')
    vecs = Y.value
    random_cut = sample_spherical(1, ndim = n)
    partition = {}
    signs = []
    for i in range(n):
        signs.append(np.sign(np.dot(vecs[i,:], random_cut)))
    for con in constraints:
        for i in range(n):
            if signs[con] == signs[i]:
                partition[i] = constraints[con]
    return partition


def voltage_cut(graph, constraints, k=2, max_iter=10000, tol=1e-5, verbose=False):
    if k!=2: raise NotImplementedError()
    N = len(graph)

    # form mapping
    condmat = nx.adjacency_matrix(graph, weight='weight').asfptype()
    np.reciprocal(condmat.data, out=condmat.data)
    c_mat = sp.sparse.diags(np.asarray(1./np.sum(condmat, 1)).flatten(), 0)
    condmat = c_mat*condmat

    nodes_list = graph.nodes()
    ntoidx = {n:i for i,n in enumerate(nodes_list)}

    init_vector = np.zeros(N)

    for v, val in constraints.iteritems():
        ci = ntoidx[v]
        condmat[ci, :] = util.unit_basis(N, ci)
        init_vector[ci] = val

    new_vec, prev_vec = None, init_vector.reshape(N,1)
    l_iter = 5*int(math.ceil(math.log10(max_iter))) # Random heuristic I came up with for funsies

    for i in range(max_iter):
        new_vec = condmat*prev_vec
        curr_tol = np.max(new_vec-prev_vec)
        if verbose and (i+1)%l_iter==0:
            print 'curr_norm : {} on iteration {}/{}'.format(curr_tol, i+1,
                                                             max_iter)
        if curr_tol < tol:
            break
        prev_vec = new_vec

    final_vec = np.asarray(new_vec).flatten()
    idx_arr = np.argsort(final_vec)
    sor_arr = final_vec[idx_arr]
    # return {nodes_list[i]:0 if final_vec[i] < .5 else 1 for i in range(N)}

    min_cut, min_idx = None, None
    v_set = set(nodes_list)
    curr_set = set()

    for i in range(N-1):
        curr_vertex = nodes_list[idx_arr[i]]
        curr_set.add(curr_vertex)
        curr_cut = cut_weight(graph, curr_set, data='weight')
        if min_cut == None or min_cut > curr_cut:
            min_idx, min_cut = i, curr_cut

    # print 'minimum weight flowcut: {}'.format(min_cut)
    return {nodes_list[idx_arr[i]]:0 if i <= min_idx else 1 for i in range(N)}, final_vec, idx_arr, min_cut

def greedy_cut(q_arr, nodes_list):
    N, k = q_arr.shape
    d_idx = {(i,j):q_arr[i,j] for i,j in product(range(N), range(k))}
    sorted_idx = sorted(d_idx.keys(), key=lambda c: d_idx[c])
    partitions = {}

    for i, j in sorted_idx:
        if i in partitions:
            continue
        partitions[i] = j


    return partitions

def voltage_cut_wrapper(graph, constraints, cut_function, k=2, max_iter=10000, tol=1e-5, verbose=False):
    ''' Assumes that constraints[k] maps V->{0,1,...,k}. This may be fixed
    later if there's any question.
    '''
    N = len(graph)
    if len(constraints) < k:
        raise ValueError('not enough constraints')
    Q = np.zeros((N, k)) # Matrix of results

    # form mapping
    condmat = nx.adjacency_matrix(graph, weight='weight').asfptype()
    np.reciprocal(condmat.data, out=condmat.data)
    c_mat = sp.sparse.diags(np.asarray(1./np.sum(condmat, 1)).flatten(), 0)
    condmat = c_mat*condmat

    nodes_list = graph.nodes()
    ntoidx = {n:i for i,n in enumerate(nodes_list)}

    for v, val in constraints.iteritems():
        ci = ntoidx[v]
        condmat[ci, :] = 0
        Q[ci, val] = 1

    new_mat = None
    prev_mat = np.zeros((N,k))

    if verbose: print 'solving system of eq'
    total_range = tqdm(range(max_iter))
    for i in total_range:
        new_mat = condmat*prev_mat + Q
        curr_tol = np.max(new_mat-prev_mat)
        if verbose:
            total_range.set_description('Current norm : {}'.format(curr_tol))
        if curr_tol < tol:
            break
        prev_mat = new_mat

    Q_array = np.asarray(new_mat)

    # Passes a matrix of voltages (e.g. A[i,j] = i-th node and j-th constraint)
    # along with a list n[i] which maps indices to vertex labels
    partitions = cut_function(Q_array, nodes_list)
    print 'voltage cut weight : {}'.format(evaluate(graph, partitions, k))

def brute_force(graph, constraints, k=2):
    if len(constraints) < k:
        raise ValueError('not enough constraints')

    vertices = [v for v in graph.nodes() if v not in constraints]
    min_assignment = copy(constraints)
    min_evaluation = _brute_force(graph, vertices, 0,
                     copy(constraints), min_assignment, k)

    print 'min_weight assignment is : {}'.format(min_evaluation)
    return min_assignment

# Runtime is awful here at k^(|V|-k)
def _brute_force(graph, vertices, curr_vert_idx, curr_choice, curr_min, k):
    if curr_vert_idx >= len(vertices):
        return evaluate(graph, curr_choice, k)

    curr_vert = vertices[curr_vert_idx]
    min_eval = None

    for i in range(k):
        curr_choice[curr_vert] = i
        curr_eval = _brute_force(graph, vertices, curr_vert_idx+1,
                                 curr_choice, curr_min, k)
        if min_eval is None or curr_eval < min_eval:
            min_eval = curr_eval
            curr_min[curr_vert] = i

    return min_eval
