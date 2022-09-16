import sys
import numpy as np
from scipy.io import loadmat
from mpc.system import System
from mpc.get_ideal_locality import get_ideal_locality

# Load and create system from matlab data
testFile = sys.argv[1]
data     = loadmat(testFile, squeeze_me=True)

numSims = len(data['As'])
systems = [None] * numSims

# Load systems
for i in range(numSims):
    sysParams = {'A'     : data['As'][i],
                 'B2'    : data['B2s'][i],
                 'AComm' : data['AComms'][i],
                 'Nx'    : data['Nx'],
                 'Nu'    : data['Nu']
                }
    systems[i] = System(**sysParams)

# Get locality values
T = 15
for i in range(numSims):
    print(f'Sim {i+1} of {numSims}')
    loc = get_ideal_locality(systems[i], T)
    print(f'Locality: {loc}')
    systems[i].setLocality(loc)
