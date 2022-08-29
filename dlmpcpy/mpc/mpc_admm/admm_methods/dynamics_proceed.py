import numpy as np
import pyopencl as cl

def dynamics_vector2matrixPhi(self):
    # Transform vector phi into matrix Phi before computing x

    return None, None

def dynamics_from_psi(self):

    Nx  = self._sys._Nx
    Nu  = self._sys._Nu
    T   = self._T
    D   = self._D_row

    phi_dynamics = np.zeros((Nx+Nu)*D)
    phi_dynamics[0:Nx*D]= self._psi[self._row2col[Nx*D:2*Nx*D]]
    phi_dynamics[Nx*D:(Nx+Nu)*D] = self._psi[self._row2col[Nx*T*D:(Nx*T+Nu)*D]]
    
    # Create the input
    d_x0_maxD = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._x0_maxD.astype(np.float32))
    d_phi_dynamics = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=phi_dynamics.astype(np.float32))

    # Create the output
    self._result = np.zeros(Nx+Nu).astype(np.float32)
    d_result = cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY, self._result.nbytes)

    # SECTION 2 - PROGRAM EXECUTION
    self._kernel[2](self._queue, (Nx+Nu,),  None, Nx+Nu, D, d_x0_maxD, d_phi_dynamics, d_result)

    # SECTION 1.2 - MEMORY COPYING: device to host
    cl.enqueue_copy(self._queue,self._result,d_result)

    x = self._result[0:Nx]
    u = self._result[Nx:Nx+Nu]

    return x,u

# NEED TO PASTE HERE DYNAMICS FROM PSI (OLD) AND RENAME THIS!

def dynamics_from_phi(self):
    # Compute the dynamics (x,u) given the vector phi
    # For shorter notation    
    dimension = (self._Nx_T_and_Nu_T_minus_1)

    # Recover the matrix
    phi_row = np.zeros(dimension*self._D_row)
    #for i in range(len(self._col2row)):
    #    phi_row[self._col2row[i]] = self._phi[i]
    phi_row[self._col2row] = self._phi[0:len(self._col2row)]

    x = []
    base = self._sys._Nx*self._D_row
    for columns in self._row_patch[self._sys._Nx:2*self._sys._Nx]:
        x.append(np.matmul(phi_row[range(base,base+len(columns))],self._x0[columns]))
        base += self._D_row
    x = np.array(x)

    u = []
    base = self._Nx_times_T*self._D_row
    for columns in self._row_patch[self._Nx_times_T:self._Nx_times_T+self._sys._Nu]:
        u.append(np.matmul(phi_row[range(base,base+len(columns))],self._x0[columns]))
        base += self._D_row
    u = np.array(u)

    return x,u
    
def dynamics_from_Phi(self):
    # Compute the dynamics (x,u) given the matrix Phi

    # Simulate the dynamics
    x = np.matmul(self._Phi[self._sys._Nx:2*self._sys._Nx,:],self._x0)
    u = np.matmul(self._Phi[self._Nx_times_T:self._Nx_times_T+self._sys._Nu,:],self._x0)

    return x,u
