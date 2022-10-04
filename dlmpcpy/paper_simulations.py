import sys
import pickle
import numpy as np
from scipy.io import loadmat
from mpc.system import System
from mpc.get_ideal_locality import get_ideal_locality
from mpc.mpc_admm.admm import ADMM_GPU

# Expected usage: paper_simulations.py <.mat file> <.pkl file with outputs>
inputFileName  = sys.argv[1]
outputFileName = sys.argv[2]

data      = loadmat(inputFileName, squeeze_me=True)
numSims   = len(data['As'])

upper_bound_x = 20
lower_bound_x = -upper_bound_x

# TODO: not implemented (?)
upper_bound_u = 5
lower_bound_u = -upper_bound_u

Nx = data['Nx']
Nu = data['Nu']

Tsim = 40
T    = 15

admm_parameters = {'max_iterations'              : 10**4,
                   'primal_convergence_tolerance': 10**(-3),
                   'dual_convergence_tolerance'  : 10**(-3),
                   'rho'                         : 10
                  }

system_parameters = [None] * numSims
mpc_parameters    = [None] * numSims
xs                = [None] * numSims
us                = [None] * numSims
runtimes          = [None] * numSims

for i in range(numSims):
    print(f'Sim {i+1} of {numSims}')
    
    # TODO: incorporate randomized weights Q, R
    system_parameters[i] = {'A'     : data['As'][i],
                            'B2'    : data['B2s'][i],
                            'AComm' : data['AComms'][i],
                            'Nx'    : Nx,
                            'Nu'    : Nu,
                            'upper_bound_x' : upper_bound_x,
                            'lower_bound_x' : lower_bound_x,
                            'upper_bound_u' : upper_bound_u,
                            'lower_bound_u' : lower_bound_u
                           }
    system = System(**system_parameters[i])
    
    # Compute locality that preserves performance
    # TODO: this is a hack to save time (if we already know the result of get_ideal_locality)
    # loc                                = get_ideal_locality(system, T)
    # system_parameters[i]['locality_d'] = loc # So we have a record of it
    loc = 2
    system.setLocality(loc)

    x0 = np.random.rand(Nx,1)*2 - 1 # Uniform distribution over [-1, 1)
    mpc_parameters[i] = {'simulation_time'   : Tsim,
                         'time_horizon'      : T,
                         'initial_condition' : x0
                        }
    
    dlmpc = ADMM_GPU(system = system, **mpc_parameters[i])
    dlmpc.setParameters(**admm_parameters)
    
    xs[i], us[i] = dlmpc.run()
    runtimes[i]  = dlmpc.getRuntime()/Tsim
    
# Save data for postprocessing
outputFile = open(outputFileName, 'wb')
pickle.dump(admm_parameters, outputFile)
pickle.dump(system_parameters, outputFile)
pickle.dump(mpc_parameters, outputFile)
pickle.dump(xs, outputFile)
pickle.dump(us, outputFile)
pickle.dump(runtimes, outputFile)

# Code to load pickle file
'''
inputFile = open(inputFileName, 'rb')
admm_parameters   = pickle.load(inputFile)
system_parameters = pickle.load(inputFile)
mpc_parameters    = pickle.load(inputFile)
xs                = pickle.load(inputFile)
us                = pickle.load(inputFile)
runtimes          = pickle.load(inputFile)
'''




