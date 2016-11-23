# Typical example of usage of helper functions

from scipy.sparse import csgraph
import loadfiles

# Typical usage stuff
d_net, people_set = loadfiles.load_dict_network()
smat = loadfiles.dict_to_sparse(d_net, people_set)

print smat

lapl = csgraph.laplacian(smat)
print csgraph.connected_components(smat, directed=False)
