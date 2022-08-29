import pyopencl as cl
import numpy as np

from .helper import convergence_tracker

'''
In this script we define the different functions to compute the ADMM steps that we have given the different computation options. Most of them are run in GPU using a precompiled kernel
'''

# -----------------------------------  CPU ----------------------------------- #

# -------- Row-wise computation :
def comp_row_cpu(self):
    # For shorter notation
    Nx           = self._sys._Nx
    Nu           = self._sys._Nu
    upper_bound  = self._sys._state_upper_bound
    lower_bound  = self._sys._state_lower_bound
    row_patch    = self._row_patch
    const_scalar = self._row_scalar
    const_matrix = self._row_matrix
    const_vector = self._row_vector

    self._Phi = np.zeros((self._Nx_T_and_Nu_T_minus_1,Nx))

    for row in range(self._Nx_T_and_Nu_T_minus_1):
        #columns = row_patch['Corresponding columns'][row] # column patch
        columns = self._row_patch[row]

        a = self._Psi[row,columns]-self._Lambda[row,columns]

        self._Phi[row,columns] = self._rho*np.matmul(a,const_matrix[row])

        if  row < self._Nx_times_T: # Because only the state is subject to constraints
            lambda1 = 0
            lambda2 = 0

            criterion = self._rho*np.matmul(a,const_vector[row])

            if criterion - upper_bound > 0:
                lambda1 = (criterion - upper_bound)/const_scalar[row]
            elif - criterion + lower_bound > 0:
                lambda2 = (- criterion + lower_bound)/const_scalar[row]

            self._Phi[row,columns] -= (lambda1-lambda2)*const_vector[row].T

# -------- Column-wise computation :
def comp_col_cpu(self):
    # For shorter notation
    Nx           = self._sys._Nx
    Nu           = self._sys._Nu
    T            = self._T
    column_patch = self._column_patch
    const_matrix = self._col_matrix
    const_vector = self._col_vector

    self._Psi = np.zeros((self._Nx_T_and_Nu_T_minus_1,Nx))

    for column in range(Nx):
        #rows = column_patch['Corresponding rows'][column] # row patch
        rows = column_patch[column]

        a = self._Phi[rows,column] + self._Lambda[rows,column]

        self._Psi[rows,column] = const_vector[column] + a - np.matmul(const_matrix[column],a)

# -------- Lagrange multiplier computation :
def comp_lag_cpu(self):
    Nx = self._sys._Nx
    Nu = self._sys._Nu
    T  = self._T

    for row in range(self._Nx_T_and_Nu_T_minus_1):
        for column in range(Nx):
            self._Lambda[row,column] += self._Phi[row,column]
            self._Lambda[row,column] -= self._Psi[row,column]

# -------- Convergence criterion computation :
def comp_conv_cpu(self):
    # For shorter notation
    Nx           = self._sys._Nx
    Nu           = self._sys._Nu
    T            = self._T
    column_patch = self._column_patch
    
    conv1_helper = 0 ## just to track convergence
    conv2_helper = 0 ## just to track convergence

    self._yet_to_conv = False
    for column in range(Nx):
        #rows = column_patch['Corresponding rows'][column] # row patch
        rows = column_patch[column] # row patch
        
        conv1 = np.linalg.norm(self._Phi[rows,column]-self._Psi[rows,column])
        conv2 = np.linalg.norm(self._Psi[rows,column]-self._Psi_old[rows,column])

        if conv1 > self._epsp or conv2 > self._epsd:
            self._yet_to_conv = True
            
        conv1_helper += conv1 ## just to track convergence
        conv2_helper += conv2 ## just to track convergence
            
    convergence_tracker(conv1_helper,conv2_helper) ## just to track convergence

# -----------------------------------  Naive GPU (variable D) ----------------------------------- #

# -------- Row-wise computation :
def comp_row_gpu_varD(self):
    for row in range(self._Nx_T_and_Nu_T_minus_1):
        #columns = self._row_patch['Corresponding columns'][row]
        columns = self._row_patch[row]
        self._psi[self._ini_pos_row[row]:self._fin_pos_row[row]] = self._Psi[row,columns]
        self._lambda[self._ini_pos_row[row]:self._fin_pos_row[row]] = self._Lambda[row,columns]

    #self._psi = self._Psi.flatten()
    #self._lambda = self._Lambda.flatten()

    # Create the input
    d_psi    = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._psi.astype(np.float32))
    d_lambda = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._lambda.astype(np.float32))

    # Create the output
    d_phi = cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY, self._phi.nbytes)

    self._kernel[0](self._queue, (self._Nx_T_and_Nu_T_minus_1,), None,
        self._Nx_times_T, # state dimension
        self._rho, self._sys._state_lower_bound, self._sys._state_upper_bound, 
        self._d_ini_pos_row, self._d_fin_pos_row, self._d_ini_pos_mat_row, self._d_fin_pos_mat_row,
        self._d_matrix, self._d_vector, self._d_scalar,
        d_psi, d_lambda, d_phi
    )

    cl.enqueue_copy(self._queue, self._phi, d_phi)

    # Postcomputation
    self._Phi = np.zeros((self._Nx_T_and_Nu_T_minus_1,self._sys._Nx)).astype(np.float32)
    for row in range(self._Nx_T_and_Nu_T_minus_1):
        #columns = self._row_patch['Corresponding columns'][row]
        columns = self._row_patch[row]
        self._Phi[row,columns] = self._phi[self._ini_pos_row[row]:self._fin_pos_row[row]]

# -------- Column-wise computation :
def comp_col_gpu_varD(self):
    for column in range(self._sys._Nx):
        #rows = self._column_patch['Corresponding rows'][column]
        rows = self._column_patch[column]
        self._phi[self._ini_pos_col[column]:self._fin_pos_col[column]] = self._Phi[rows,column]
        self._lambda[self._ini_pos_col[column]:self._fin_pos_col[column]] = self._Lambda[rows,column]

    # Precompute (should be moved out from here)
    d_phi = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._phi.astype(np.float32))
    d_lambda = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._lambda.astype(np.float32))

    # Create the output
    d_psi = cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY, self._psi.nbytes)

    self._kernel[1](self._queue, (self._sys._Nx,), None,
        self._d_ini_pos_col, self._d_fin_pos_col, self._d_ini_pos_mat_col, self._d_fin_pos_mat_col,
        self._d_matrix_col, self._d_vector_col,
        d_phi, d_lambda, d_psi
    )

    cl.enqueue_copy(self._queue, self._psi, d_psi)

    self._Psi = np.zeros((self._Nx_T_and_Nu_T_minus_1,self._sys._Nx)).astype(np.float32)
    for column in range(self._sys._Nx):
        #rows = self._column_patch['Corresponding rows'][column]
        rows = self._column_patch[column]
        self._Psi[rows,column] = self._psi[self._ini_pos_col[column]:self._fin_pos_col[column]]


# -------- Lagrange multiplier computation :
def comp_lag_gpu_varD(self): # = comp_lag_gpu_varD
    self._phi = self._Phi.T.flatten()
    self._psi = self._Psi.T.flatten()
    self._lambda = self._Lambda.T.flatten()

    # Create the input
    d_phi = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._phi.astype(np.float32))
    d_psi = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._psi.astype(np.float32))
    d_lambda_old = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._lambda.astype(np.float32))

    # Create the output
    d_lambda = cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY, self._lambda.nbytes)

    self._kernel[2](self._queue, (len(self._phi),), None,
        d_phi, d_psi, d_lambda_old, d_lambda
    )

    cl.enqueue_copy(self._queue, self._lambda, d_lambda)

    self._Lambda = np.reshape(self._lambda, (self._sys._Nx, self._Nx_T_and_Nu_T_minus_1)).T.astype(np.float32)

# -------- Convergence criterion computation :
def comp_conv_gpu_varD(self):
    # not really variable length...?
    self._psi_old = self._Psi_old.T.flatten()

    # Create the input
    d_phi = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._phi.astype(np.float32))
    d_psi = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._psi.astype(np.float32))
    d_psi_old = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._psi_old.astype(np.float32))

    # Create the output
    yet_to_converge = np.empty(self._sys._Nx).astype(np.bool)
    d_yet_to_converge = cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY, yet_to_converge.nbytes)

    self._kernel[3](self._queue, (self._sys._Nx,), None,
        self._Nx_T_and_Nu_T_minus_1, self._epsp, self._epsd,
        d_phi, d_psi, d_psi_old, d_yet_to_converge
    )

    cl.enqueue_copy(self._queue, yet_to_converge, d_yet_to_converge)

    self._yet_to_conv = True in yet_to_converge

# -----------------------------------  GPU with fixed D ----------------------------------- #

# -------- Row-wise computation :
def comp_row_gpu(self):
    # SECTION 1.1 - MEMORY COPYING: host to device
    #cl.enqueue_copy(self._queue, self._d_psi_row, self._psi_row.astype(np.float32))
    #cl.enqueue_copy(self._queue, self._d_lambda_row, self._lambda_row.astype(np.float32))
    d_psi_row    = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._psi_row.astype(np.float32))
    d_lambda_row  = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._lambda_row.astype(np.float32))

    # Create the output
    d_phi_row = cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY, self._phi_row.nbytes)

    # SECTION 2 - PROGRAM EXECUTION

    self._kernel[0](self._queue, (self._Nx_T_and_Nu_T_minus_1,), None,
        self._D_row, self._rho, self._sys._state_lower_bound, self._sys._state_upper_bound, 
        self._d_matrix, self._d_vector, self._d_scalar,
        #self._d_psi_row, self._d_lambda_row, self._d_phi_row
        d_psi_row, d_lambda_row, d_phi_row
    )

    # SECTION 1.2 - MEMORY COPYING: device to host

    # Read back the results from the compute device
    #cl.enqueue_copy(self._queue, self._phi_row, self._d_phi_row)
    cl.enqueue_copy(self._queue, self._phi_row, d_phi_row)

# -------- Column-wise computation :
def comp_col_gpu(self):
    # Create the input
    d_phi = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._phi.astype(np.float32))
    d_lambda = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._lambda.astype(np.float32))

    # Create the output
    d_psi = cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY, self._psi.nbytes)

    self._kernel[1](self._queue,(self._sys._Nx,), None,
        self._D_column,
        self._d_matrix_col, self._d_vector_col,
        d_phi, d_lambda, d_psi
    )

    cl.enqueue_copy(self._queue, self._psi, d_psi)

# -------- Lagrange multiplier computation :
def comp_lag_gpu(self):
    # Create the input
    d_phi = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._phi.astype(np.float32))
    d_psi = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._psi.astype(np.float32))
    d_lambda_old = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._lambda.astype(np.float32))

    # Create the output
    d_lambda = cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY, self._lambda.nbytes)

    self._kernel[2](self._queue, (len(self._phi),), None,
        d_phi, d_psi, d_lambda_old, d_lambda
    )

    cl.enqueue_copy(self._queue, self._lambda, d_lambda)

# -------- Convergence criterion computation :
def comp_conv_gpu(self):
    DIM_col = self._sys._Nx

    # Create the input
    d_phi = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._phi.astype(np.float32))
    d_psi = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._psi.astype(np.float32))
    d_psi_old = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._psi_old.astype(np.float32))

    # Create the output
    yet_to_converge = np.empty(DIM_col).astype(np.bool)
    d_yet_to_converge = cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY, yet_to_converge.nbytes)

    self._kernel[3](self._queue, (DIM_col,), None,
        self._D_column, self._epsp, self._epsd,
        d_phi, d_psi, d_psi_old, d_yet_to_converge
    )

    cl.enqueue_copy(self._queue, yet_to_converge, d_yet_to_converge)

    self._yet_to_conv = True in yet_to_converge

# -----------------------------------  GPU with Column+Lagrange+Convergence lumped  ----------------------------------- #

# -------- Row-wise computation is the same as in the previous case (comp_row_gpu)

# -------- Column-wise + Lagrange + Convergence computation :
def comp_col_lag_conv(self):
    D_col = self._D_column
    eps_p = self._epsp
    eps_d = self._epsd

    DIM_col = self._sys._Nx
    DIM_lag = len(self._phi)

    # SECTION 1.1 - MEMORY COPYING: host to device
    '''
    # swap the references of old results and placeholder
    cl.enqueue_copy(self._queue, self._d_phi, self._phi.astype(np.float32))

    # equivalent to cl.enqueue_copy(self._queue, self._d_psi_old, self._psi.astype(np.float32))
    tmp = self._d_psi
    self._d_psi = self._d_psi_old
    self._d_psi_old = tmp
    
    # equivalent to cl.enqueue_copy(self._queue, self._d_lambda_old, self._lambda.astype(np.float32))
    tmp = self._d_lambda
    self._d_lambda = self._d_lambda_old
    self._d_lambda_old = tmp

    yet_to_converge = np.empty(DIM_col).astype(np.bool)
    cl.enqueue_copy(self._queue, self._d_yet_to_converge, yet_to_converge)
    '''
    # Create the input
    d_phi = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._phi.astype(np.float32))
    d_lambda_old = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._lambda.astype(np.float32))
    d_psi_old = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._psi.astype(np.float32))

    # Create the output
    d_psi = cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY, self._psi.nbytes)
    d_lambda = cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY, self._lambda.nbytes)

    yet_to_converge = np.empty(DIM_col).astype(np.bool)
    d_yet_to_converge = cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY, yet_to_converge.nbytes)

    # SECTION 2 - PROGRAM EXECUTION
    self._kernel[1](self._queue, (DIM_col,), None, 
        D_col, DIM_lag, eps_p, eps_d,
        self._d_matrix_col, self._d_vector_col, 
        #self._d_phi, self._d_lambda_old, self._d_psi_old, self._d_lambda, self._d_psi, self._d_yet_to_converge
        d_phi, d_lambda_old, d_psi_old, d_lambda, d_psi, d_yet_to_converge
    )

    # SECTION 1.2 - MEMORY COPYING: device to host

    # Read back the results from the compute device
    #cl.enqueue_copy(self._queue, self._psi, self._d_psi)
    #cl.enqueue_copy(self._queue, self._lambda, self._d_lambda)
    #cl.enqueue_copy(self._queue, yet_to_converge, self._d_yet_to_converge)
    cl.enqueue_copy(self._queue, self._psi, d_psi)
    cl.enqueue_copy(self._queue, self._lambda, d_lambda)
    cl.enqueue_copy(self._queue, yet_to_converge, d_yet_to_converge)

    self._yet_to_conv = True in yet_to_converge

# -----------------------------------  GPU with whole ADMM step lumped (using local memory)   ----------------------------------- #
    
# -------- Row-wise + Column-wise + Lagrange + Convergence computation :
def comp_row_col_lag_conv(self):

    # SECTION 1.1 - MEMORY COPYING: host to device
    d_psi_old    = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._psi.astype(np.float32))
    d_lambda_old  = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._lambda.astype(np.float32))

    # Create the output
    d_psi = cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY, self._psi.nbytes)
    d_lambda = cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY, self._lambda.nbytes)
    yet_to_converge = np.empty(self._sys._Nx).astype(np.bool)
    d_yet_to_converge = cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY, yet_to_converge.nbytes)

    # SECTION 2 - PROGRAM EXECUTION
    globalrange = (self._sys._Nx,self._D_column) 
    localrange  = (1,self._D_column)
    localmem    = cl.LocalMemory(np.dtype(np.float32).itemsize * self._D_column)

    self._kernel[1](self._queue, globalrange, localrange,
        self._D_row, self._D_column,
        self._rho, self._sys._state_lower_bound, self._sys._state_upper_bound, self._epsp, self._epsd,
        self._d_matrix, self._d_vector, self._d_scalar, self._d_matrix_col, self._d_vector_col,
        self._d_row2col, self._d_patch_of_rows, self._d_patch_of_columns,
        d_psi_old, d_lambda_old,
        localmem,
        d_psi, d_lambda, d_yet_to_converge
        )

    # SECTION 1.2 - MEMORY COPYING: device to host
    # Read back the results from the compute device
    cl.enqueue_copy(self._queue, self._psi, d_psi)
    cl.enqueue_copy(self._queue, self._lambda, d_lambda)
    cl.enqueue_copy(self._queue, yet_to_converge, d_yet_to_converge)

    self._yet_to_conv = True in yet_to_converge

