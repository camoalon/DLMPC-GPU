from mpc import *
import sys
import time

from user_defined_scenario import *

if __name__ == '__main__':

    # ----- System definition
    system_parameters = get_system_parameters(Nx,locality_d)
    system = System(**system_parameters)

    # ---- MPC Scheme definition
    mpc_parameters = get_mpc_parameters(Tsim,T,x0)

    classes = [
        ADMM_GPU, ADMM_GPU_combinedkernels, ADMM_GPU_variablelength, ADMM_GPU_naive,
        ADMM_CPU, SLSpy
    ]

    for mpc_class in classes:
        dlmpc = mpc_class(
            system = system, 
            **mpc_parameters
        )

        # ----- ADMM parameter definition
        if (mpc_parameters is not None) and (mpc_class is not SLSpy) :
            admm_parameters = get_admm_parameters()
            dlmpc.setParameters(**admm_parameters)

        # ----- Run MPC
        dlmpc.run()

        # ----- OUTPUT
        print("The runtime of %s was %.3f seconds" % (
            dlmpc.getName(), dlmpc.getRuntime()
        ))
