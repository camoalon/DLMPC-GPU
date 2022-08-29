import numpy as np
import pyopencl as cl
from .c_kernels import *
'''
In this script we define the different kernels that we have given the different computation options. We precompile them and save the queue, context and kernel to mpc_parameters
'''

def generateContextAndQueue(self):
    # Create a compute context
    self._context = cl.create_some_context()
    # Create a command queue
    self._queue = cl.CommandQueue(self._context)

# -----------------------------------  Naive GPU (variable D) ----------------------------------- #

def compilation_varD(self): # Fills in queue, context and kernel into mpc_parameters
    program = cl.Program(self._context, 
        kernel_row_varD_comp + kernel_col_varD_comp + kernel_lag_comp + kernel_conv_comp
    ).build()

    row_varD_comp = program.row_varD_comp
    row_varD_comp.set_scalar_arg_dtypes(row_varD_comp_arg_type)

    col_varD_comp = program.col_varD_comp
    col_varD_comp.set_scalar_arg_dtypes(col_varD_comp_arg_type)

    lag_comp = program.lag_comp
    lag_comp.set_scalar_arg_dtypes(lag_comp_arg_type)

    conv_comp = program.conv_comp
    conv_comp.set_scalar_arg_dtypes(conv_comp_arg_type)

    self._kernel = [row_varD_comp, col_varD_comp, lag_comp, conv_comp]

# -----------------------------------  GPU with fixed D ----------------------------------- #

def compilation_fixD_nogpuprecomp(self): # Fills in queue, context and kernel into mpc_parameters
    # Create the compute program from the source buffer and build it
    program = cl.Program(self._context, 
        kernel_row_comp + kernel_col_comp + kernel_lag_comp + kernel_conv_comp
    ).build()
    
    row_comp = program.row_comp
    row_comp.set_scalar_arg_dtypes(row_comp_arg_type)

    col_comp = program.col_comp
    col_comp.set_scalar_arg_dtypes(col_comp_arg_type)

    lag_comp = program.lag_comp
    lag_comp.set_scalar_arg_dtypes(lag_comp_arg_type)

    conv_comp = program.conv_comp
    conv_comp.set_scalar_arg_dtypes(conv_comp_arg_type)

    self._kernel = [row_comp, col_comp, lag_comp, conv_comp]

# -----------------------------------  GPU with fixed D + GPU precomputations ----------------------------------- #

def compilation_fixD_gpuprecomp(self): # Fills in queue, context and kernel into mpc_parameters
    self.error('compilation_fixD_gpuprecomp not yet implemented')

# -----------------------------------  GPU with Column+Lagrange+Convergence lumped ----------------------------------- #

def compilation_multithread_nogpuprecomp(self): # Fills in queue, context and kernel into mpc_parameters
    # Create the compute program from the source buffer and build it
    program = cl.Program(self._context, 
        kernel_row_comp + kernel_col_lag_conv_comp
    ).build()

    row_comp = program.row_comp
    row_comp.set_scalar_arg_dtypes(row_comp_arg_type)

    col_lag_conv_comp = program.col_lag_conv_comp
    col_lag_conv_comp.set_scalar_arg_dtypes(col_lag_conv_comp_arg_type)

    self._kernel = [row_comp, col_lag_conv_comp]

# -----------------------------------  GPU with Column+Lagrange+Convergence lumped + precomputations ----------------------------------- #

def compilation_multithread_gpuprecomp(self): # Fills in queue, context and kernel into mpc_parameters
    self.error('compilation_multithread_gpuprecomp not yet implemented')

# -----------------------------------  GPU with whole ADMM step lumped (using local memory) ----------------------------------- #

def compilation_localmultithread_nogpuprecomp(self): # Fills in queue, context and kernel into mpc_parameters
    # Create the compute program from the source buffer and build it
    program = cl.Program(self._context,
        kernel_row_col_lag_conv_comp
    ).build()

    row_col_lag_conv_comp = program.row_col_lag_conv_comp
    row_col_lag_conv_comp.set_scalar_arg_dtypes(row_col_lag_conv_comp_arg_type)

    self._kernel = [row_col_lag_conv_comp]

# --------------------------------  GPU with whole ADMM step lumped (using local memory) + precomputations -------------------------------- #

def compilation_localmultithread_gpuprecomp(self): # Fills in queue, context and kernel into mpc_parameters
    # Create the compute program from the source buffer and build it
    program = cl.Program(self._context, 
            kernel_row_precomp + kernel_row_col_lag_conv_comp + kernel_dynamics_from_psi).build()

    row_precomp = program.row_precomp
    row_precomp.set_scalar_arg_dtypes(row_precomp_arg_type)

    row_col_lag_conv_comp = program.row_col_lag_conv_comp
    row_col_lag_conv_comp.set_scalar_arg_dtypes(row_col_lag_conv_comp_arg_type)

    dynamics_from_psi = program.dynamics_from_psi
    dynamics_from_psi.set_scalar_arg_dtypes(dynamics_from_psi_arg_type)

    self._kernel = [row_precomp,row_col_lag_conv_comp,dynamics_from_psi]

# ---------------------------------------------------- #

def preallocating_gpumemory_varD(self):
    # We need to transpose the constant matrix, so that multiplication works out nicely
    # Row transformations
    matrix = []
    for row in range(self._Nx_T_and_Nu_T_minus_1):
        matrix = np.append(matrix,self._row_matrix[row].T.flatten())
    vector = np.concatenate(self._row_vector)
    scalar = np.concatenate(self._row_scalar)

    self._d_matrix = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix.astype(np.float32))
    self._d_vector = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=vector.astype(np.float32))
    self._d_scalar = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=scalar.astype(np.float32))

    row_patch = self._row_patch

    ini_pos = []
    fin_pos = []
    next_pos = 0
    ini_pos_mat = []
    fin_pos_mat = []
    next_pos_mat = 0
    for row in range(self._Nx_T_and_Nu_T_minus_1):
        #columns = len(row_patch['Corresponding columns'][row])
        columns = len(row_patch[row])
        ini_pos = np.append(ini_pos,next_pos)
        ini_pos_mat = np.append(ini_pos_mat,next_pos_mat)
        next_pos += columns
        next_pos_mat += (columns**2)
        fin_pos = np.append(fin_pos,next_pos)
        fin_pos_mat = np.append(fin_pos_mat,next_pos_mat)
    ini_pos = ini_pos.astype(np.int32)
    fin_pos = fin_pos.astype(np.int32)
    ini_pos_mat = ini_pos_mat.astype(np.int32)
    fin_pos_mat = fin_pos_mat.astype(np.int32)

    self._ini_pos_row = ini_pos
    self._fin_pos_row = fin_pos
    #self._ini_pos_mat_row = ini_pos_mat
    #self._fin_pos_mat_row = fin_pos_mat

    self._d_ini_pos_row = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ini_pos.astype(np.int32))
    self._d_fin_pos_row = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=fin_pos.astype(np.int32))
    self._d_ini_pos_mat_row = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ini_pos_mat.astype(np.int32))
    self._d_fin_pos_mat_row = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=fin_pos_mat.astype(np.int32))

    # Column transformations
    matrix_col = []
    for column in range(self._sys._Nx):
        matrix_col = np.append(matrix_col,self._col_matrix[column].flatten())
    vector_col = np.concatenate(self._col_vector)

    self._d_matrix_col = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix_col.astype(np.float32))
    self._d_vector_col = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=vector_col.astype(np.float32))

    column_patch = self._column_patch

    ini_pos = []
    fin_pos = []
    next_pos = 0
    ini_pos_mat = []
    fin_pos_mat = []
    next_pos_mat = 0
    for column in range(self._sys._Nx):
        #rows= len(column_patch['Corresponding rows'][column])
        rows= len(column_patch[column])
        ini_pos = np.append(ini_pos,next_pos)
        ini_pos_mat = np.append(ini_pos_mat,next_pos_mat)
        next_pos += rows
        next_pos_mat += (rows**2)
        fin_pos = np.append(fin_pos,next_pos)
        fin_pos_mat = np.append(fin_pos_mat,next_pos_mat)
    ini_pos = ini_pos.astype(np.int32)
    fin_pos = fin_pos.astype(np.int32)
    ini_pos_mat = ini_pos_mat.astype(np.int32)
    fin_pos_mat = fin_pos_mat.astype(np.int32)

    self._ini_pos_col = ini_pos
    self._fin_pos_col = fin_pos
    #self._ini_pos_mat_col = ini_pos_mat
    #self._fin_pos_mat_col = fin_pos_mat

    self._d_ini_pos_col = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ini_pos.astype(np.int32))
    self._d_fin_pos_col = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=fin_pos.astype(np.int32))
    self._d_ini_pos_mat_col = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ini_pos_mat.astype(np.int32))
    self._d_fin_pos_mat_col = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=fin_pos_mat.astype(np.int32))


    # Create the buffers
    self._Phi = np.empty((self._Nx_T_and_Nu_T_minus_1,self._sys._Nx)).astype(np.float32)
    self._phi = self._Phi.flatten()

    self._Psi = np.zeros(self._Phi.shape).astype(np.float32)
    self._psi = self._Psi.flatten()

    self._Lambda = np.zeros(self._Phi.shape).astype(np.float32)
    self._lambda = self._Lambda.flatten()

def preallocating_gpumemory_fixD(self):
    self._d_matrix = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._row_matrix.astype(np.float32))
    self._d_vector = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._row_vector.astype(np.float32))
    self._d_scalar = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._row_scalar.astype(np.float32))
    self._d_matrix_col = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._col_matrix.astype(np.float32))
    self._d_vector_col = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._col_vector.astype(np.float32))

    # Create the buffers
    buffer_row_dim = self._Nx_T_and_Nu_T_minus_1*self._D_column
    self._phi_row = np.empty(buffer_row_dim).astype(np.float32)

    buffer_dim = self._sys._Nx * self._D_column
    self._psi = np.zeros(buffer_dim).astype(np.float32)
    self._lambda = np.zeros(buffer_dim).astype(np.float32)

def preallocating_gpumemory_fixD_with_patches(self):
    preallocating_gpumemory_fixD(self)
    
    self._d_row2col = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._row2col.astype(np.int32))
    self._d_patch_of_rows = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._patch_of_rows.astype(np.int32))
    self._d_patch_of_columns = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._patch_of_columns.astype(np.int32))
