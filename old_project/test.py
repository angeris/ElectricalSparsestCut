# Typical example of usage of helper functions

import scipy
from scipy.sparse import csgraph
import loadfiles
from sklearn import manifold

def test_loading():
    # Typical usage stuff
    d_net, people_set = loadfiles.load_dict_network()
    smat = loadfiles.dict_to_sparse(d_net, people_set)

    print smat

    lapl = csgraph.laplacian(smat)
    print csgraph.connected_components(smat, directed=False)

def create_random_sparse_graph(n_vertices = 100, density = .1):
    random_g = scipy.sparse.rand(n_vertices, n_vertices, density=density, format='dok', random_state=None)
    for edge in random_g.keys():#make matrix symmetric
        random_g[(edge[1], edge[0])] = random_g[edge]

    return random_g
