import numpy as np
from scipy.io import loadmat
from mpc.system import System
from get_ideal_locality import get_ideal_locality

# Load and create system from matlab data
testFile = '/home/lisa/Downloads/sandbox.mat'
data     = loadmat(testFile, squeeze_me=True)

sysParams = {k : data[k] for k in ('A','B2','Nx', 'Nu', 'AComm') if k in data}
sys       = System(**sysParams)
T         = 10

# TODO: can speed up by using mldivide, sparse matrices, zero-col trimming
loc = get_ideal_locality(sys, T)

