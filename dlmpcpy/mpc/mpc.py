from .base_class import *
from .timer import *
from slspy import *

class MPC(Base):
    '''
    Vanilla MPC
    '''
    def __init__(self,
        system            = None,   
        simulation_time   = None,
        time_horizon      = None,
        initial_condition = None,
        weigth            = None
    ):
        super().__init__()
        self.setName('MPC')

        self.setSystem(system)
        self.setSimulationTime(simulation_time)
        self.setFIR_Horizon(time_horizon)
        self.setInitialCondition(initial_condition)
        self.setOptimizationWeights(weight)

        self._x = []
        self._u = []

    def setSystem(self, system):
        self._sys = system

    def setSimulationTime(self, simulation_time):
        if simulation_time > 1:
            self._Tsim = simulation_time
        else:
            self._Tsim = 1
   
    def setFIR_Horizon(self, time_horizon):
        if time_horizon > 0:
            self._T = time_horizon
        else:
            self._T = 1

    def setInitialCondition(self, initial_condition):
        self._init_x = initial_condition

    def setOptimizationWeight(self, weight):
        self._Q = weight
 
    def run(self):
        
        self.setup()

        for t in range(self._Tsim):

            if t==1: # We do not measure time for warmstarts
                default_timer.startMeasuring()

            self.solve()
            self._x, self._u = self.dynamicsProceed()    

        return self._x, self._u

    def setup(self):
        pass # To be inherited and defined by children's class

    def solve(self):
        pass # To be inherited and defined by children's class

    def dynamicsProceed(self):
        pass # To be inherited and defined by children's class

    def getRuntime(self):
        return sum(default_timer.getMeasurements())
