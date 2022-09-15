import time
import numpy as np
from scipy.io import loadmat
from mpc.system import System
from mpc.get_ideal_locality import get_ideal_locality

# Load and create system from matlab data
testFile = '/home/lisa/Downloads/sandbox.mat'
data     = loadmat(testFile, squeeze_me=True)

sysParams = {k : data[k] for k in ('A','B2','Nx', 'Nu', 'AComm') if k in data}
sys       = System(**sysParams)
T         = 10

t1  = time.perf_counter()
loc = get_ideal_locality(sys, T)
t2  = time.perf_counter()
print(f'Time elapsed: {t2-t1:.2f} sec')

sys.setLocality(loc)
