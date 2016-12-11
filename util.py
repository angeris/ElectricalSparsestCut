import numpy as np

def unit_basis(n, i):
    ei = np.zeros(n)
    ei[i] = 1
    return ei
