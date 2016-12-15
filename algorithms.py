# Algorithms library for testing

import networkx as nx
import cvxpy as cvx
import numpy as np
import scipy as sp
import math
import util
import scipy.sparse.linalg as linalg

def e_boundary(graph, v_list, data=None):
    edges = graph.edges(v_list, data=data, default=1)
    return (e for e in edges if e[1] not in v_list)

def cut_weight(graph, v_list, data=None):
    edges = e_boundary(graph, v_list, data=data)
    return sum(w for u, v, w in edges)

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
        G.edge[u][v]['invweight'] = 1/weight
        G.edge[v][u]['invweight'] = 1/weight

    #make constraint edges full connected:
    # for con in constraints:
    #     for v in G.nodes():
    #         if con!=v:
    #             if np.random.rand() < 1:
    #                 G.add_edge(con, v,{'weight':1})
    return G, constraints

def brute_force(graph, constraints, k):
    raise NotImplementedError()

def max_flow_cut(graph, constraints, k):
    if k!=2: raise Exception('Max flow only applicable for 2 partitions')
    if len(constraints.keys())!=2: raise Exception('Max flow only applicable with 2 constraints')

    graph_copy = graph.copy()
    keys = list(constraints.keys())
    cut_value, partial = nx.minimum_cut(graph_copy, keys[0], keys[1], capacity = 'invweight')
    reachable, non_reachable = partial
    cutset = set()
    for u, nbrs in ((n, graph_copy[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)

    partition = {}
    graph_copy.remove_edges_from(cutset)
    for con in constraints:
        for node in nx.algorithms.components.node_connected_component(graph_copy, con):
            partition[node] = constraints[con]

    print 'max flow weight : {}'.format(cut_weight(graph, {v for v,k in partition.iteritems() if k==0}, data='invweight'))
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


def flow_cut(graph, constraints, k=2, max_iter=10000, tol=1e-5, verbose=False):
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
        curr_cut = cut_weight(graph, curr_set, data='invweight')
        if min_cut == None or min_cut > curr_cut:
            min_idx, min_cut = i, curr_cut

    print 'minimum weight flowcut: {}'.format(min_cut)
    return {nodes_list[idx_arr[i]]:0 if i <= min_idx else 1 for i in range(N)}, final_vec, idx_arr
