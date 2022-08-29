from slspy import *

class SLS_Cons_SLS_Approx (SLS_Constraint):
    def __init__(self, state_feedback=False):
        self._state_feedback = state_feedback

    def addConstraints(self, sls, constraints=None):
        '''
        state-feedback constraints:
        [ zI-A, -B2 ][ Phi_x ] = I
                     [ Phi_u ]
        '''
        constraints = []

        Nx = sls._system_model._Nx
        Nu = sls._system_model._Nu

        # sls constraints
        # the below constraints work for output-feedback case as well because
        # sls._Phi_x = sls._Phi_xx and sls._Phi_u = sls._Phi_ux
        # Phi_x, Phi_u are in z^{-1} RH_{\inf}. Therefore, Phi_x[0] = 0, Phi_u = 0
        constraints += [ sls._Phi_x[0] == np.zeros([Nx,Nx]) ]
        constraints += [ sls._Phi_u[0] == np.zeros([Nu,Nx]) ]
        constraints += [ sls._Phi_x[1] == np.eye(Nx) ]
        # ignore the last constraint as the approximation
        #constraints += [ 
        #    (sls._system_model._A  @ sls._Phi_x[sls._FIR_horizon] +
        #     sls._system_model._B2 @ sls._Phi_u[sls._FIR_horizon] ) == np.zeros([Nx, Nx]) 
        #]
        for tau in range(1,sls._FIR_horizon):
            constraints += [
                sls._Phi_x[tau+1] == (
                    sls._system_model._A  @ sls._Phi_x[tau] +
                    sls._system_model._B2 @ sls._Phi_u[tau]
                )
            ]

        if not self._state_feedback:
            Ny = sls._system_model._Ny

            # Phi_xx, Phi_ux, and Phi_xy are in z^{-1} RH_{\inf}.
            # Phi_uy is in RH_{\inf} instead of z^{-1} RH_{\inf}.
            constraints += [ sls._Phi_xy[0] == np.zeros([Nx,Ny]) ]

            # output-feedback constraints
            constraints += [
                sls._Phi_xy[1] == sls._system_model._B2 @ sls._Phi_uy[0]
            ]
            # ignore the last constraint
            #constraints += [ 
            #    (sls._system_model._A  @ sls._Phi_xy[sls._FIR_horizon] +
            #     sls._system_model._B2 @ sls._Phi_uy[sls._FIR_horizon]) == np.zeros([Nx, Ny])
            #]
            #constraints += [ 
            #    (sls._Phi_xx[sls._FIR_horizon] @ sls._system_model._A  +
            #     sls._Phi_xy[sls._FIR_horizon] @ sls._system_model._C2 ) == np.zeros([Nx, Nx])
            #]
            constraints += [
                sls._Phi_ux[1] == sls._Phi_uy[0] @ sls._system_model._C2
            ]
            constraints += [
                (sls._Phi_ux[sls._FIR_horizon] @ sls._system_model._A  +
                 sls._Phi_uy[sls._FIR_horizon] @ sls._system_model._C2 ) == np.zeros([Nu, Nx])
            ]
            for tau in range(1,sls._FIR_horizon):
                constraints += [ 
                    sls._Phi_xy[tau+1] == (
                        sls._system_model._A  @ sls._Phi_xy[tau] +
                        sls._system_model._B2 @ sls._Phi_uy[tau]
                    )
                ]

                constraints += [
                    sls._Phi_xx[tau+1] == (
                        sls._Phi_xx[tau] @ sls._system_model._A  +
                        sls._Phi_xy[tau] @ sls._system_model._C2
                    )
                ]

                constraints += [
                    sls._Phi_ux[tau+1] == (
                        sls._Phi_ux[tau] @ sls._system_model._A  +
                        sls._Phi_uy[tau] @ sls._system_model._C2
                    )
                ]
        return constraints

class MPC_SLS_Objective (SLS_Objective):
    def __init__(self, x0 = None):
        self._x0 = x0

    def addObjectiveValue(self, sls, objective_value):
        self._objective_expression = 0
        Phi_x = sls._Phi_x
        Phi_u = sls._Phi_u
        for tau in range(len(Phi_x)):
            self._objective_expression += cp.sum_squares(Phi_x[tau] @ self._x0)
            self._objective_expression += cp.sum_squares(Phi_u[tau] @ self._x0)

        return objective_value + self._objective_expression

class MPC_SLS_Locality (SLS_Constraint):
    def __init__(self,
        system = None,
        locality_d = 2
    ):
        # compute the supports
        Nx = system._Nx
        Nu = system._Nu
        A = system._A
        B = system._B2
        d = locality_d

        Aux = np.identity(Nx)
        A_support = np.zeros((Nx,Nx))
        for i in range(Nx):
            A_support[i] = [int(x) for x in A[i]!=0]

        for i in range(d-1):
            Aux = np.matmul(A_support,Aux)
        locality_Phix = Aux!=0
        locality_Phiu = np.matmul(np.transpose(B),Aux)!=0

        # isn't the above equivalent to something simpler as follows?
        #self._support_x = (system._A != 0)
        #self._support_u = ...

        self._support_x = locality_Phix
        self._support_u = locality_Phiu

    def addConstraints(self, sls, constraints):
        Phi_x = sls._Phi_x
        Phi_u = sls._Phi_u

        for t in range(1,sls._FIR_horizon+1):
            # shutdown those not in the support
            for ix,iy in np.ndindex(self._support_x.shape):
                if self._support_x[ix,iy] == False:
                    constraints += [ Phi_x[t][ix,iy] == 0 ]
    
            for ix,iy in np.ndindex(self._support_u.shape):
                if self._support_u[ix,iy] == False:
                    constraints += [ Phi_u[t][ix,iy] == 0 ]

        return constraints

class MPC_SLS_StateBound (SLS_Constraint):
    def __init__(self,
        state_upper_bound = 1.2,                    # Upper bound on state
        state_lower_bound = -0.2,                   # Lower bound on state
        x0 = None
    ):
        self._ub = state_upper_bound
        self._lb = state_lower_bound
        self._x0 = x0
    
    def addConstraints(self, sls, constraints):
        Phi_x = sls._Phi_x

        for t in range(1,sls._FIR_horizon+1):
            # shutdown those not in the support
            constraints += [ Phi_x[t] @ self._x0 <= self._ub ] 
            constraints += [ Phi_x[t] @ self._x0 >= self._lb ]

        return constraints
