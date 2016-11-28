import loadfiles
import numpy as np
from scipy.sparse import csgraph
import scipy
from sklearn import manifold
import test
import math

def find_random_point_in_unit_ball(h):
    point = np.random.rand((h))*2-1
    while np.linalg.norm(point,2) > 1:
        point = np.random.rand((h))
    return point
def get_vertex_weights(graph):
    weights = {}
    for edge in graph.keys():
        weights[edge[0]] = weights.get(edge[0],0) + graph[edge]
        weights[edge[1]] = weights.get(edge[1],0) + graph[edge]
    return weights

def calculate_subset_expansion(graph, n_vertices, subset): #calculate weighted subset expansion
    cut_val = 0

    subset_dict = {}
    for s in subset: #hash for easy graph expansion calculation
        subset_dict[s] = 0

    for vertex in subset:
        edges = graph.getcol(vertex)
        for e in edges.keys():
            if e[0] not in subset_dict.keys():
                cut_val += edges[e]

    return cut_val/len(subset)#max(len(subset), n_vertices - len(subset))

def spectral_embedding(graph, num_sets):
    embedding = manifold.spectral_embedding(graph, n_components=2*num_sets, eigen_solver=None, random_state=None, eigen_tol=0.0, norm_laplacian=True, drop_first=True)
    return embedding

def random_projection(embedding, num_sets, h=-1):
    if h==-1:
        h = int(math.ceil(math.log(num_sets, 2)))
    g = np.random.normal(size=(2*num_sets, h))

    projection = np.dot(embedding,g)/math.sqrt(h)
    return projection

def random_partitioning(projection, radius =.7):
    unfound_vertices = list(range(np.shape(projection)[0]))
    h = np.shape(projection)[1]
    sets = []

    #keep generating points in euclidean unit ball of size h
    #find all vertices within radius of the points (and not in any subset so far)
        #add to new subset
    #continue until all vertices assigned a subset
    while len(unfound_vertices)>0:
        point = find_random_point_in_unit_ball(h)
        new_set = []
        for v in unfound_vertices:
            if np.linalg.norm(point - projection[v,:]/np.linalg.norm(projection[v,:]), 2) < radius:
                new_set.append(v)
        if len(new_set)>0:
            for v in new_set:
                unfound_vertices.remove(v)
            sets.append(new_set)
    return sets
def merging(graph, projection, partitioning, num_sets):
    vertex_weights = get_vertex_weights(graph)
    partition_weights = [sum([vertex_weights[v]*np.linalg.norm(projection[v,:], 2) for v in s]) for s in partitioning]
    k_prime = int(math.ceil(num_sets*3/2.0))

    sorted_partitions = np.array(partition_weights).argsort()[::-1] #argument sort in descending order

    top_kprime_partition_indices = sorted_partitions[0:k_prime]
    top_kprime_partitions = [partitioning[ind] for ind in top_kprime_partition_indices]
    top_kprime_partition_weights = [partition_weights[ind] for ind in top_kprime_partition_indices]

    for i in range(k_prime, len(partitioning)):
        smallest_weight_index = np.argmin(top_kprime_partition_weights)
        top_kprime_partitions[smallest_weight_index].extend(partitioning[sorted_partitions[i]])
        top_kprime_partition_weights[smallest_weight_index] += partition_weights[sorted_partitions[i]]

    return top_kprime_partitions
def cheeger_sweep(graph, top_sets, projection, n_vertices, num_sets):
    #note: if needed, this function can be sped up significantly by modifying the expansion value last calculated instead redoing every time.

    sets_hat = []
    sets_hat_expansions = []
    #for each of the k_prime subsets, calculate the subset of it that has the best expansion
    for s in top_sets:
        #order the vertices in it by descending over ||F*(v)||, calculate expansion for each subsubset
        vertex_norms = [np.linalg.norm(projection[v,:], 2) for v in s]
        sorted_vertices = [s[ind] for ind in np.array(vertex_norms).argsort()[::-1]] #argument sort in descending order
        expansions = []
        for i in range(len(s)):
            expansions.append(calculate_subset_expansion(graph, n_vertices, sorted_vertices[0:i+1]))
        sets_hat.append(sorted_vertices[0:np.argmin(expansions)+1])
        sets_hat_expansions.append(np.min(expansions))

    #then return the k subsubsets with the best expansion (NOT A PARTITION?)
    best_expansions = [sets_hat[ind] for ind in np.array(sets_hat_expansions).argsort()[0:num_sets]]
    
    # print [sets_hat_expansions[ind] for ind in np.array(sets_hat_expansions).argsort()[0:num_sets]]
    # print sum([len(s) for s in best_expansions])
    return best_expansions
def higher_order_cheeger(graph, num_sets, n_vertices):
    embedding = spectral_embedding(graph, num_sets)
    projection = random_projection(embedding, num_sets)
    partitioning = random_partitioning(projection)
    top_sets = merging(graph, projection, partitioning, num_sets)
    final_sets = cheeger_sweep(graph, top_sets, projection, n_vertices, num_sets)
    print final_sets
def main_test():
    n_sets = 10
    n_vertices = 1000
    random_g = test.create_random_sparse_graph(n_vertices = n_vertices);

    higher_order_cheeger(random_g, n_sets,n_vertices)

main_test()
