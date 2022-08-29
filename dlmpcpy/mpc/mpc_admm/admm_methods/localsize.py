import numpy as np

'''
In this script we define the different functions that we have to deal with how we define the local size (either patches of variable size or worst-case D). We have three main kinds of functions in this script:

    1) Function to inialize the ADMM variables Psi and Lambda: can be initialized as matrices (variable D) or vectors (worst-case D).
    2) Function to switch between a row vector and a column vector (only relevant when using worst-case D)
    3) Function to compute x given Phi (matrix) or phi (vector)
'''

# -----------------------------------  Worst-case D ----------------------------------- #

# -------- 1) Initialize vectors :

def init_vectors(self): # Returns psi and lambda of the appropriate dimension (vectors)
    dimension = self._sys._Nx * self._D_column

    self._psi = np.zeros(dimension)
    self._lambda = np.zeros(dimension)
    
# -------- 2) Switch between rows and columns :

def row_col_transformation(self): # Fills in col2row and row2col in mpc_parameters

    # For shorter notation
    Nx       = self._sys._Nx
    D_row    = self._D_row
    D_column = self._D_column

    col2row_total = Nx* D_column
    row2col_total = self._Nx_T_and_Nu_T_minus_1 * D_row

    #FIXME: the default coordinate should point to some empty node, now we point it to the last entry, but it would break if the last entry is not empty
    col2row = np.array([row2col_total-1]*col2row_total).astype(np.int32)
    col2row_count = np.zeros(Nx).astype(np.int32)

    row2col = np.array([col2row_total-1]*row2col_total).astype(np.int32)
    row2col_count = np.zeros(self._Nx_T_and_Nu_T_minus_1).astype(np.int32)

    base = 0
    for i in range(self._Nx_T_and_Nu_T_minus_1):
        columns = self._row_patch[i]
        col2row[columns*D_column+col2row_count[columns]] = np.array(range(len(columns))) + base
        col2row_count[columns] += 1
        base += D_row

    base = 0
    for j in range(Nx):
        rows = self._column_patch[j]
        row2col[rows*D_row+row2col_count[rows]] = np.array(range(len(rows))) + base
        row2col_count[rows] += 1
        base += D_column

    # Save to mpc_params
    self._col2row = col2row 
    self._row2col = row2col 

def row_col_transformation_with_patches(self): # Fills in col2row and row2col in mpc_parameters
    ''' this is only used by scenario 6 (everything lumped)'''

    Nx       = self._sys._Nx
    D_row    = self._D_row
    D_column = self._D_column

    col2row_total = Nx* D_column
    row2col_total = self._Nx_T_and_Nu_T_minus_1 * D_row

    #FIXME: the default coordinate should point to some empty node, now we point it to the last entry, but it would break if the last entry is not empty
    col2row = np.array([row2col_total-1]*col2row_total).astype(np.int32)
    col2row_count = np.zeros(Nx).astype(np.int32)

    row2col = np.array([col2row_total-1]*row2col_total).astype(np.int32)
    row2col_count = np.zeros(self._Nx_T_and_Nu_T_minus_1).astype(np.int32)

    patch_of_columns = np.empty(row2col_total).astype(np.int32)
    patch_of_rows = np.empty(col2row_total).astype(np.int32)

    base = 0
    for i in range(self._Nx_T_and_Nu_T_minus_1):
        base_next = base + D_row
        columns = self._row_patch[i]

        fill_dummy_index = 0
        while fill_dummy_index in columns:
            # try next
            fill_dummy_index += 1
        patch_of_columns[base:base_next] = fill_dummy_index

        column_indices = np.array(range(len(columns))) + base

        col2row[columns*D_column+col2row_count[columns]] = column_indices
        patch_of_columns[column_indices] = columns

        col2row_count[columns] += 1
        base = base_next

    base = 0
    for j in range(Nx):
        base_next = base + D_column
        rows = self._column_patch[j]

        fill_dummy_index = 0
        while fill_dummy_index in rows:
            # try next
            fill_dummy_index += 1
        patch_of_rows[base:base_next] = fill_dummy_index

        row_indices = np.array(range(len(rows))) + base

        row2col[rows*D_row+row2col_count[rows]] = row_indices
        patch_of_rows[row_indices] = rows

        row2col_count[rows] += 1
        base = base_next

    # Save to mpc_params
    self._col2row = col2row
    self._row2col = row2col

    self._patch_of_rows = patch_of_rows
    self._patch_of_columns = patch_of_columns

def column_to_row(self): # Perform change from column vector to row vector
    dimension = (self._Nx_T_and_Nu_T_minus_1)*self._D_column

    self._psi_row = np.zeros(dimension)
    self._lambda_row = np.zeros(dimension)

    self._psi_row[self._col2row] = self._psi[0:len(self._col2row)]
    self._lambda_row[self._col2row] = self._lambda[0:len(self._col2row)]

def row_to_column(self): # Perform change from row vector to column vector
    self._phi = np.zeros(len(self._psi))

    self._phi[self._row2col] = self._phi_row[0:len(self._row2col)]
    
# -----------------------------------  Variable  D ----------------------------------- #


# -------- 1) Initialize matrices :

def init_matrices(self): # Returns Psi and Lambda of the appropriate dimension (matrices)
    # For shorter notation
    dimension = (self._Nx_T_and_Nu_T_minus_1,self._sys._Nx)
    self._Psi = np.zeros(dimension)
    self._Lambda = np.zeros(dimension)
