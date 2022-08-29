import numpy as np

Nx         = 8
Tsim      = 10
T          = 5
x0         = np.array([[0.01200021], [0.41435228], [0.51271307], [0.18309696], [0.17909441], [0.3369922 ], [0.49142641], [0.47657257]]) #np.random.rand(Nx,1)
locality_d = 2

# --- PLANT DEFINITION --- #
def generate_plant(Nx):

    Nu = Nx

    A = np.zeros((Nx,Nx))
    j = 0
    for i in range(0,Nx,2):
        j += 1
        if j == 1:
            A[i:i+2,i+2:i+4] = [[0,0], [1,1]]
            A[i:i+2,i:i+2] = [[0,1], [-3,-3]]
        elif j == Nx/2:
            A[i:i+2,i:i+2] = [[0,1], [-3,-3]]
            A[i:i+2,i-2:i] = [[0,0], [1,1]]
        else:
            A[i:i+2,i+2:i+4] = [[0,0],[1,1]]
            A[i:i+2,i:i+2] = [[0,1],[-3,-3]]
            A[i:i+2,i-2:i] = [[0,0],[1,1]]
            
    B = np.eye(Nx)

    # Discretize
    Ts = .1
    A_discrete = (np.identity(Nx)+A*Ts)
    B_discrete = Ts*B
    
    return Nu, A_discrete, B_discrete

# --- SYSTEM PARAMETERS --- #
def get_system_parameters(Nx,locality_d):

    Nu, A_discrete, B_discrete = generate_plant(Nx)

    system_parameters = {
                        'Nx'            : Nx,
                        'Nu'            : Nu,
                        'A'             : A_discrete,
                        'B2'            : B_discrete,
                        'locality_d'    : locality_d,
                        'upper_bound_x' : 2,
                        'lower_bound_x' : -2,
                        'upper_bound_u' : 100,
                        'lower_bound_u' : -100
                        }
    
    return system_parameters

# --- MPC PARAMETERS --- #
def get_mpc_parameters(Tsim,T,x0):

    mpc_parameters = {
                    'simulation_time'   : Tsim,
                    'time_horizon'      : T,
                    'initial_condition' : x0
                    }
                    
    return mpc_parameters

# --- MPC COMPUTATION STRATEGY (OPTIONAL) --- #
''' 
Choose one of the following options. 
If left "None", the optimal GPU strategy would be used.

    - mpc_class = "Optimal_GPU_Strategy". Optimal GPU parallelization using maximum D, combining the kernels and exploiting local memory. 
    - mpc_class = "Combined_Kernels_GPU". GPU parallelization using maximum D and combining kernels but without exploiting local memory.
    - mpc_class = "Maximum_D_GPU". GPU parallelization using maximum D, going back to host after each ADMM iteration.
    - mpc_class = "Naive_parallelization_GPU". GPU implementation consisting on simply parallelizing the ADMM steps.
    - mpc_class = "Virtual_parallelization_CPU". CPU implementation of DLMPC, just as done in Matlab.
    - mpc_class = "SLSpy". CPU implementation using SLSpy solvers as opposed to ADMM.

'''
mpc_class = "Optimal_GPU_Strategy"

# --- ADMM PARAMETERS (OPTIONAL) --- #
def get_admm_parameters():

    admm_parameters = {
                    'max_iterations' :               10**4,
                    'primal_convergence_tolerance' : 10**(-4),
                    'dual_convergence_tolerance' :   10**(-3),
                    'rho' :                          10
                    }
    
    return admm_parameters
