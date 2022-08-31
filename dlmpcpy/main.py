if __name__ == '__main__':
    from mpc import *
    from user_defined_scenario import *
#else:
#    from .mpc import *
#    from .user_defined_scenario import *

import sys
import time

if __name__ == '__main__':

    # ----- System definition
    system_parameters = get_system_parameters(Nx,locality_d)
    system = System(**system_parameters)

    # ----- MPC scheme definition
    if mpc_class == None:
        mpc = ADMM_GPU

    elif mpc_class == "Optimal_GPU_Strategy": 
        mpc = ADMM_GPU
    elif mpc_class == "Combined_Kernels_GPU":
        mpc = ADMM_GPU_combinedkernels
    elif mpc_class == "Maximum_D_GPU":
        mpc = ADMM_GPU_variablelength
    elif mpc_class == "Naive_parallelization_GPU":
        mpc = ADMM_GPU_naive
    elif mpc_class == "Virtual_parallelization_CPU":
        mpc = ADMM_CPU
    elif mpc_class == "SLSpy":
        mpc = SLSpy

    mpc_parameters = get_mpc_parameters(Tsim,T,x0,Q)
    dlmpc = mpc(system = system, **mpc_parameters)

    # ----- ADMM parameter definition
    if (mpc_parameters is not None) and (mpc is not SLSpy) :
        admm_parameters = get_admm_parameters()
        dlmpc.setParameters(**admm_parameters)

    # ----- Run MPC
    x,u = dlmpc.run()

    # ----- OUTPUT
    print("The online average runtime per iteration  was %.3f seconds" %(dlmpc.getRuntime()/Tsim))


