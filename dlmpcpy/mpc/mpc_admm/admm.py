from ..mpc import MPC
from ..timer import *
import numpy as np
from .admm_methods.compilation      import *
from .admm_methods.localsize        import *
from .admm_methods.precomputation   import *
from .admm_methods.computation      import *
from .admm_methods.dynamics_proceed import *


class ADMM(MPC):
    '''
    ADMM general structure and parameters
    '''
    def __init__(self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.setName('ADMM')

        self._x0 = self._init_x
 
        self._phi = None
        self._psi = None
        self._lambda = None
 
        self._Phi = None
        self._Psi = None
        self._Lambda = None
 
        self._counter = 0
        self._yet_to_conv = True

        # helpers
        self._Nx_times_T = 0

        self._column_settings = (0,0,0,0)
        self._column_patch = None
        self._max_rows = None
 
        self._row_settings = (0,0,0,0)
        self._row_patch = None
        self._max_columns = None
 
        self._precomputation_runtime = 0
        self._optimization_runtime = 0


    def setParameters(self,
        max_iterations               = 10**4,    # Maximum number of ADMM iterations
        primal_convergence_tolerance = 10**(-4), # Primal convergence tolerance
        dual_convergence_tolerance   = 10**(-3), # Dual convergence tolerance
        rho                          = 10,       # ADMM multiplier - impacts convergence!
    ):
        self._max_iterations = max_iterations
        self._epsp  = primal_convergence_tolerance
        self._epsd  = dual_convergence_tolerance
        self._rho   = rho
 
 
    def setup(self):
        
        self._E, self._IZAZB = self._sys.getFeasibilityConstraints(self._T)
        
        self._column_patch, self._D_column = self._sys.getColumnPatches(self._T)
        self._row_patch, self._D_row = self._sys.getRowPatches(self._T)

        self._Nx_times_T = self._sys._Nx * self._T
        self._Nx_T_and_Nu_T_minus_1 = self._Nx_times_T + self._sys._Nu * (self._T - 1)

        default_timer.measurePrecomputationTime()

        self.precompile() 

        default_timer.measurePrecompilationTime()

        self.initialize()

        self._x.append(self._init_x)
        self._u = []

        default_timer.measurePrecomputationTime()

    def precompile(self):
        pass # Placeholder for children's class
    def initialize(self):
        pass # Placeholder for children's class

    def solve(self):

        self.precompute()

        self.preallocateGPU_Memory()
 
        default_timer.measurePrecomputationTime()
 
        # Perform ADMM iterations
        self._yet_to_conv = True

        while self._yet_to_conv:
            self._psi_old = self._psi
            self._Psi_old = self._Psi

            self.updatePhi() 
 
            self.updatePsi()
 
            self.updateLambda()
 
            self.updateYetToConv()
 
            # Maximal number of iterations
            self._counter += 1
            if self._counter > self._max_iterations:
               self.info('ADMM did not converge in ' + str(self._max_iterations) + ' iterations')
               break
          
        default_timer.measureOptimizationTime()

        #convergence_tracker(-1,-1) ## just to track convergence
       
    def precomputate(self):
        pass # Placeholder for children's class
    def preallocateGPU_Memory(self):
        pass # Placeholder for children's class
    def updatePhi(self):
        pass # Placeholder for children's class
    def updatePsi(self):
        pass # Placeholder for children's class
    def updateLambda(self):
        pass # Placeholder for children's class
    def updateYetToConv(self):
        pass # Placeholder for children's class

    def dynamicsProceed(self):

         x_next, u_next = self.computeDynamics() 

         self.setInitialCondition(x_next)

         self._x.append(x_next)
         self._u.append(u_next)

         default_timer.measureSimulationTime()

         return self._x, self._u

    def computeDynamics(self):
         pass # Placeholder for children's class


class ADMM_GPU(ADMM):
    '''
    Optimal GPU implementation
    '''
    def __init__(self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.setName('Optimal_GPU_Strategy')
      
    def precompile(self):
        generateContextAndQueue(self)
        #compilation_localmultithread_nogpuprecomp(self)
        compilation_localmultithread_gpuprecomp(self)

    def initialize(self):
        # Compute the row2col and col2row transformations before running ADMM 
        row_col_transformation_with_patches(self)

        # Precompute columns before starting ADMM (run with fixed D and in CPU)
        precomp_col_cpu_fixD(self) 

        # Initialize vectors psi and lambda before running ADMM
        init_vectors(self)

    def precompute(self):
        #precomp_row_cpu_fixD(self) # Precompute rows before MPC iteration (run with fixed D and in CPU)
        precomp_row_gpu(self)

    def preallocateGPU_Memory(self):
        preallocating_gpumemory_fixD_with_patches(self)
 
    def updatePhi(self):
        comp_row_col_lag_conv(self)  # This already includes Psi, Lambda and Convergence updates
 
    def computeDynamics(self):
        return dynamics_from_psi(self)

class ADMM_GPU_combinedkernels(ADMM):
    '''
    GPU implementation (combined kernels but not using local memory)
    '''
    def __init__(self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.setName('Combined_Kernels_GPU')
      
    def precompile(self):
        generateContextAndQueue(self)
        compilation_multithread_nogpuprecomp(self) 

    def initialize(self):
        # Compute the row2col and col2row transformations before running ADMM 
        row_col_transformation(self)

        # Precompute columns before starting ADMM (run with fixed D and in CPU)
        precomp_col_cpu_fixD(self) 

        # Initialize vectors psi and lambda before running ADMM
        init_vectors(self)

    def precompute(self):
        precomp_row_cpu_fixD(self) # Precompute rows before MPC iteration (run with fixed D and in CPU)

    def preallocateGPU_Memory(self):
        preallocating_gpumemory_fixD(self)
 
    def updatePhi(self):
        column_to_row(self)
        comp_row_gpu(self)

    def updatePsi(self):
        row_to_column(self)
        comp_col_lag_conv(self)
 
    def computeDynamics(self):
        return dynamics_from_phi(self)

class ADMM_GPU_variablelength(ADMM_GPU_combinedkernels):
    '''
    GPU implementation (going back to host in between kernels, using fixed D)
    '''
    def __init__(self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.setName('Maximum_D_GPU')
      
    def precompile(self):
        generateContextAndQueue(self)
        compilation_fixD_nogpuprecomp(self) 

    def updatePsi(self):
        row_to_column(self)
        comp_col_gpu(self)
 
    def updateLambda(self):
        comp_lag_gpu(self)
 
    def updateYetToConv(self):
        comp_conv_gpu(self)


class ADMM_GPU_naive(ADMM):
    '''
    Naive GPU implementation (going back to host in between kernels, using variable D)
    '''
    def __init__(self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.setName('Naive_parallelization_GPU')
      
    def precompile(self):
        generateContextAndQueue(self)
        compilation_varD(self) 

    def initialize(self):
        # Precompute columns before starting ADMM (run with variable D and in CPU)
        precomp_col_cpu_varD(self) 

        # Initialize matrices Psi and Lambda before running ADMM
        init_matrices(self)

    def precompute(self):
        # Precompute rows before MPC iteration (run with fixed D and in CPU)
        precomp_row_cpu_varD(self)

    def preallocateGPU_Memory(self):
        preallocating_gpumemory_varD(self)
 
    def updatePhi(self):
        comp_row_gpu_varD(self)

    def updatePsi(self):
        comp_col_gpu_varD(self)
 
    def updateLambda(self):
        comp_lag_gpu_varD(self)
 
    def updateYetToConv(self):
        comp_conv_gpu_varD(self)

    def computeDynamics(self):
        return dynamics_from_Phi(self)

class ADMM_CPU(ADMM):
    '''
    Naive CPU implementation  
    '''
    def __init__(self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.setName('Virtual_parallelization_CPU')
    
    def initialize(self):
        # Precompute columns before starting ADMM (run with variable D and in CPU)
        precomp_col_cpu_varD(self) 

        # Initialize matrices Psi and Lambda before running ADMM
        init_matrices(self)

    def precompute(self):
        precomp_row_cpu_varD(self)  # Precompute rows before MPC iteration (run with fixed D and in CPU)

    def updatePhi(self):
        comp_row_cpu(self)

    def updatePsi(self):
        comp_col_cpu(self)
 
    def updateLambda(self):
        comp_lag_cpu(self)
 
    def updateYetToConv(self):
        comp_conv_cpu(self)

    def computeDynamics(self):
       return dynamics_from_Phi(self)
