import numpy as np
from mpc.system import System
from get_ideal_locality import get_ideal_locality

# TODO: don't forget to add comm adjustment for grid; via system.py?
# TODO: update the getLocality() function in system.py
# TODO: can speed up by using mldivide, sparse matrices, zero-col trimming

Nx   = 2
Nu   = 1
A    = np.array([[3,2],[1,0]])
B2   = np.array([[1],[0]])
T    = 3 # TODO: is this (T+1) compared to matlab?
eps  = 1e-8
d    = 2

sys = System(A=A, B2=B2, Nx=Nx, Nu=Nu)
loc = get_ideal_locality(sys, T)
