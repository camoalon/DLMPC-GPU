from ..mpc import MPC
from slspy import *
from .mpc_sls_objective_and_constraints import *
from ..timer import *

class SLSpy(MPC):
    '''
        SLSpy
    '''
    def __init__(self,
        sls = SLS,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.setName('SLSpy')

        self._sys._ignore_output = True

        self._sls = sls(
            system_model = self._sys,
            FIR_horizon = self._T
        )

        self._sls_approx_constraint = SLS_Cons_SLS_Approx(state_feedback=True)

        self._init_x = self._init_x.reshape(-1,1)
        self._next_x = self._init_x

    def setup(self):

        # Initialize the variables
        x_next = self._init_x
        self._sys.initialize(x0=self._init_x)

        self._x.append(self._init_x.flatten())
        self._u = []

        # Objective
        self._objective = MPC_SLS_Objective()

        # Locality constraints
        self._locality_constraints = MPC_SLS_Locality (system = self._sys, locality_d = self._sys._locality_d)

        # Upper and lower bounds
        self._bounding_constraints = MPC_SLS_StateBound(
            state_upper_bound = self._sys._state_upper_bound,
            state_lower_bound = self._sys._state_lower_bound
        )
        default_timer.measurePrecomputationTime()

    def solve(self):
 
         self._objective._x0 = self._next_x
         self._bounding_constraints._x0 = self._next_x
 
         self._sls << self._objective << self._sls_approx_constraint
         self._sls += self._locality_constraints
 
         default_timer.measurePrecomputationTime()
 
         controller = self._sls.synthesizeControllerModel()
 
         default_timer.measureOptimizationTime()
 
         self._controller = controller
 
 
    def dynamicsProceed(self):
         if self._controller is None: # indicates cvx failure
             self._x = np.nan
             self._u = np.nan
 
         else:
             # Compute the dynamics
             u_next = self._controller.getControl(y=self._next_x)
             self._sys.systemProgress(u=u_next)
             self._next_x = self._sys.getState()
 
             self._x.append(self._next_x.flatten())
             self._u.append(u_next.flatten())
 
             default_timer.measureSimulationTime()
 
         return self._x, self._u
