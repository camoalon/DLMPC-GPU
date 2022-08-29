from .base_class import *

class Timer(Base):
    def __init__(self):
        super().__init__()
        self._precompilation_runtime = 0.0  # Precompiling
        self._precomputation_runtime = 0.0  # Precomputing
        self._optimization_runtime   = 0.0  # Optimization solving
        self._simulation_runtime     = 0.0  # Dynamics computation
        
    def getMeasurements(self):
        return [self._precompilation_runtime, self._precomputation_runtime, self._optimization_runtime, self._simulation_runtime]

    def startMeasuring(self):
        self._precompilation_runtime = 0.0  # Precompiling
        self._precomputation_runtime = 0.0  # Precomputing
        self._optimization_runtime   = 0.0  # Optimization solving
        self._simulation_runtime     = 0.0  # Dynamics computation
        super().startMeasuring()

    def measurePrecompilationTime(self):
        self._precompilation_runtime += self.returnRuntimeAndRestart()

    def measurePrecomputationTime(self):
        self._precomputation_runtime += self.returnRuntimeAndRestart()

    def measureOptimizationTime(self):
        self._optimization_runtime += self.returnRuntimeAndRestart()

    def measureSimulationTime(self): 
        self._simulation_runtime += self.returnRuntimeAndRestart()

default_timer = Timer()