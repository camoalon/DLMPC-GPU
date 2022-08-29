import numpy as np
import pyopencl as cl

'''
In this script we define the different functions that we have to deal with the precomputations according to different computation strategies: CPU, GPU, variable or fixed D, etc. All of these functions fill in the appropriate parameters (row_matrix, row_vector, row_scalar, or column_matrix and column_vector) into mpc_parameters.
'''

# -----------------------------------  Perform precomputation in GPU ----------------------------------- #

# -------- Row-wise precomputation :

def precomp_row_gpu(self): # Fills in row_matrix, row_vector and row_scalar in mpc_parameters

    # For shorter notation
    D   = self._D_row
    DIM = self._Nx_T_and_Nu_T_minus_1
    DIM_rows = self._Nx_times_T

    ################# --------------- THIS IS STILL DONE IN CPU ------------- #################

    xi = [0] * DIM
    row_patch = self._row_patch # ---- NEED TO CHANGE THIS TO BE OF MAX D LENGTH
    # Computations for maximum length D
    for row in range(DIM):
        columns = row_patch[row] 
        if len(columns)<D :
            xi[row] = np.concatenate((self._x0[columns],np.vstack([0]*(D-len(columns)))),axis=0) # local initial condition
        else :
            xi[row] = self._x0[columns] # local initial condition
    self._x0_maxD = np.array(xi).flatten()

    ###########################################################################################

    # Create the input
    d_x0_maxD = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._x0_maxD.astype(np.float32))

    # Create the output
    self._row_matrix = np.zeros(D*D*DIM).astype(np.float32)
    self._row_vector = np.zeros(D*DIM).astype(np.float32)
    self._row_scalar = np.zeros(DIM).astype(np.float32)

    d_matrix = cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY, self._row_matrix.nbytes)
    d_vector = cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY, self._row_vector.nbytes)
    d_scalar = cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY, self._row_scalar.nbytes)

    # SECTION 2 - PROGRAM EXECUTION
    self._kernel[0](self._queue, (DIM,),  None, DIM, DIM_rows, self._rho, D, d_x0_maxD, d_matrix, d_vector, d_scalar)

    # SECTION 1.2 - MEMORY COPYING: device to host
    # Read back the results from the compute device
    cl.enqueue_copy(self._queue,self._row_matrix,d_matrix)
    cl.enqueue_copy(self._queue,self._row_vector,d_vector)
    cl.enqueue_copy(self._queue,self._row_scalar,d_scalar)

# -------- Column-wise precomputation :

def precomp_col_gpu(self): # Fills in col_matrix and col_vector in mpc_parameters
    self.error('precomp_col_gpu not yet implemented')

# -----------------------------------  Perform precomputation in CPU with fixed D ----------------------------------- #

# -------- Row-wise precomputation :

def precomp_row_cpu_fixD(self): # Fills in row_matrix, row_vector and row_scalar in mpc_parameters
    # For shorter notation
    Nx  = self._sys._Nx
    Nu  = self._sys._Nu
    rho = self._rho
    x0  = self._x0
    T   = self._T
    D   = self._D_row

    n = [0]* self._Nx_T_and_Nu_T_minus_1
    xi = [0] * self._Nx_T_and_Nu_T_minus_1
    const_matrix = [np.zeros((D,D))] * self._Nx_T_and_Nu_T_minus_1
    const_vector = [np.zeros(D)] * self._Nx_T_and_Nu_T_minus_1
    const_scalar = [np.zeros(1)] * self._Nx_T_and_Nu_T_minus_1

    for row in range(self._Nx_T_and_Nu_T_minus_1):
        columns = self._row_patch[row] # column patch

        n[row] = len(columns) # local dimension
        xi[row] = x0[columns].reshape(-1,1) # local initial condition

        aux_mat = np.zeros((D,D))
        aux_mat[0:n[row],0:n[row]] = np.linalg.pinv(2*np.multiply(xi[row],np.transpose(xi[row])) + rho*np.identity(n[row]))

        const_matrix[row] = aux_mat
        

        if  row < self._Nx_times_T: # Because only the state is subject to constraints
            aux_vec = np.matmul(aux_mat[0:n[row],0:n[row]],xi[row]).flatten()
            const_vector[row] = list(aux_vec) + [0]*(D-n[row])
            const_scalar[row] = np.matmul(np.transpose(xi[row]),aux_vec).flatten()

    # This saves GPU precomputation - vectorize everything
    vector = np.concatenate(const_vector)
    scalar = np.concatenate(const_scalar)
    matrix = []
    for row in range(self._Nx_T_and_Nu_T_minus_1):
        matrix = np.append(matrix,const_matrix[row].flatten())

    self._row_matrix = matrix
    self._row_vector = vector
    self._row_scalar = scalar

# -------- Column-wise precomputation :

def precomp_col_cpu_fixD(self): # Fills in col_matrix and col_vector in mpc_parameters
    # For shorter notation
    Nx = self._sys._Nx
    Nu = self._sys._Nu
    T  = self._T
    D  = self._D_column
    E  = self._E
    IZA_ZB = self._IZAZB

    n = [0] * (Nx)
    const_matrix = [np.zeros((D,D))] * (Nx)
    const_vector = [np.zeros(D)] * (Nx)

    for column in range(Nx):
        rows = self._column_patch[column] # row patch

        n[column] = len(rows) # local dimension

        M = IZA_ZB[:,rows] # FIXME: right now we are keeping the rows that are all zeros, we might want to remove that in the future
        b = E[:,column]

        M_aux = np.matmul(M.T,np.linalg.pinv(np.matmul(M,M.T)))

        aux_mat = np.zeros((D,D))
        aux_mat[0:n[column],0:n[column]] = np.matmul(M_aux,M)
        const_matrix[column] = aux_mat

        aux_vec = np.matmul(M_aux,b)
        const_vector[column] = list(aux_vec) + [0]*(D-n[column])


    # This saves GPU precomputation
    vector = np.concatenate(const_vector)
    matrix = []
    for column in range(Nx):
        matrix = np.append(matrix,const_matrix[column].flatten())

    self._col_matrix = matrix
    self._col_vector = vector

# -----------------------------------  Perform precomputation in CPU with variable D ----------------------------------- #

# -------- Row-wise precomputation :

def precomp_row_cpu_varD(self): # Fills in row_matrix, row_vector and row_scalar in mpc_parameters
    # For shorter notation
    Nx        = self._sys._Nx
    Nu        = self._sys._Nu
    T         = self._T
    rho       = self._rho
    x0        = self._x0
    row_patch = self._row_patch

    n = [0] * (self._Nx_T_and_Nu_T_minus_1)
    xi = [0] * (self._Nx_T_and_Nu_T_minus_1)
    const_matrix = [0]*(self._Nx_T_and_Nu_T_minus_1)
    const_vector = [0]*(self._Nx_times_T)
    const_scalar = [0]*(self._Nx_times_T)
    
    for row in range(self._Nx_T_and_Nu_T_minus_1):
        columns = row_patch[row] # column patch

        n[row] = len(columns) # local dimension
        xi[row] = x0[columns].reshape(-1,1) # local initial condition

        const_matrix[row] = np.linalg.pinv(2*np.multiply(xi[row],np.transpose(xi[row])) + rho*np.identity(n[row]))

        if  row < self._Nx_times_T: # Because only the state is subject to constraints
            const_vector[row] = np.matmul(const_matrix[row],xi[row]).flatten()
            const_scalar[row] = np.matmul(np.transpose(xi[row]),const_vector[row]).flatten()

    self._row_matrix = const_matrix
    self._row_vector = const_vector
    self._row_scalar = const_scalar

# -------- Column-wise computation :

def precomp_col_cpu_varD(self): # Fills in col_matrix and col_vector in mpc_parameters
    # For shorter notation
    Nx            = self._sys._Nx
    Nu            = self._sys._Nu
    T             = self._T
    column_patch  = self._column_patch
    E             = self._E
    IZA_ZB        = self._IZAZB

    const_matrix = [0] * (Nx)
    const_vector = [0] * (Nx)

    for column in range(Nx):
        rows = column_patch[column] # row patch

        M = IZA_ZB[:,rows] #FIXME: right now we are keeping the rows that are all zeros, we might want to remove that in the future
        b = E[:,column]
        
        aux_matrix = np.matmul(M.T,np.linalg.pinv(np.matmul(M,M.T)))
        const_matrix[column] = np.matmul(aux_matrix,M)
        const_vector[column] = np.matmul(aux_matrix,b)
        
    self._col_matrix = const_matrix
    self._col_vector = const_vector
