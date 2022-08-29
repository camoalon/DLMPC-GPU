from mpc.system import *
from mpc.utils.plant_generators import *
import sys

if __name__ == '__main__':
    # System definition
    mpc_settings = {
        'Nx': 8,
        'Ny': 4
    }

    if len(sys.argv) > 1:
        if sys.argv[1] == 'dlmpc':
            system = System(**mpc_settings, locality_d = 2)
        else:
            system = LTI_System(**mpc_settings)
    else:
        system = LTI_System(**mpc_settings)
    
    generate_plant_dynamics ( system_model = system )

    print(system._Nx)
    print(system._Ny)
