import numpy as np

# ============================================================
#  Row precomputations
# ============================================================

row_precomp_arg_type = [np.uint32, np.uint32, np.float32, np.uint32, None, None, None, None]

kernel_row_precomp = ''' 
    __kernel void row_precomp(
        const int DIM,
        const int DIM_rows,
        const float rho,
        const int D,
        __global float * x0_maxD,
        __global float * matrix,
        __global float * vector,
        __global float * scalar
    )

    {   float m[2048];
        float b[2048];
        float ainv[2048];
        float ainvb[2048];
        float bainvb[2048];
        float dbainvb[2048];
        int i;
        int j;
        int k;
        int l;
        int row = get_global_id(0);

        if (row < DIM){

            for (i = row*(D*D); i<(row+1)*(D*D); i++){
                j = i / D;
                k = i % D;
                if (j == k + D*row){
                    m[j*D+k] = 2*x0_maxD[j]*x0_maxD[k+D*row]+rho;
                }
                else{
                    m[j*D+k] = 2*x0_maxD[j]*x0_maxD[k+D*row];
                }
            }

        // COMPUTE THE INVERSE

        ainv[D*D*row] = 1/m[D*D*row];

        for (i=0; i<D; i++){

            if (i>1){
                for (j=row*(D*D); j<(row+1)*(D*D); j++){
                    ainv[j] = matrix[j];
                }
            }

            for (j=0; j<i; j++){  
                b[j+D*row] = m[D*(D*row+i)+j];
                ainvb[j+D*row] = 0;
            }

            bainvb[row] = 0;
            
            for (j=0; j<i*i; j++){ 
                k = j / i + D*row;
                l = j % i + D*row;
                ainvb[k] = ainvb[k] + ainv[row*D*D+j]*b[l];
                bainvb[row] = bainvb[row] + b[k]*ainv[row*D*D+j]*b[l];
            }

            dbainvb[row] = 1/(m[D*D*row+D*i+i]-bainvb[row]);

            for (j=0; j<i*i; j++){ 
                k = j / i;
                l = j % i;
                matrix[D*D*row+k*(i+1)+l] = ainv[j+D*D*row]+ dbainvb[row]*ainvb[k+D*row]*ainvb[D*row+l];
            }

            matrix[D*D*row+(i+1)*(i+1)-1] = dbainvb[row];
            for (j=0; j<i; j++){
                matrix[D*D*row+i*(i+1)+j] = -dbainvb[row]*ainvb[j+D*row];
                matrix[D*D*row+j*(i+1)+i] = -dbainvb[row]*ainvb[j+D*row];
            }
        }

        for (i=0; i<D; i++){
            if (matrix[D*D*row+i*D+i] == 1/rho){
                matrix[D*D*row+i*D+i] = 0.0;
            }
        }

        // CONSTRAINTS
        for (k=0; k<D*D; k++){ 
            i = k / D;
            j = k % D;
            vector[i+D*row] = 0.0;
        }
        scalar[row] = 0.0;


        //FIXME: we should also handle input constraints
        if (row < DIM_rows){ //Because only the state is subject to constraints
         
           for (k=0; k<D*D; k++){ 
                i = k / D;
                j = k % D;
                vector[i+D*row] += matrix[D*D*row+i*D+j]*x0_maxD[j+row*D];
            }

            for (k=0; k<D; k++){ 
                scalar[row] += x0_maxD[k+row*D]*vector[k+D*row];
            }
         }

        }
    }


    '''


# ============================================================
#  Row computations
# ============================================================

row_comp_arg_type = [
    np.uint32, np.float32, np.float32, np.float32,
    None, None, None, None, None, None
]
kernel_row_comp = '''
__kernel void row_comp(
    const int D_row,
    const float rho,
    const float lower_bound,
    const float upper_bound,
    __global float* matrix_row,
    __global float* vector_row,
    __global float* scalar_row,
    __global float* psi,
    __global float* lambda,
    __global float* phi
){
    float a_row[1024];
    float lambda1;
    float lambda2;
    float criterion;
    int n;
    int i_row;
    int j_row;
    int k_row;
    int row = get_global_id(0);
    
    int size1 = row*D_row;
    int size2 = size1*D_row;

    lambda1 = 0.0;
    lambda2 = 0.0;
    criterion = 0.0;
    for (k_row = size1, n = 0; n < D_row; ++k_row, ++n){
        phi[k_row] = 0;
        a_row[n] = psi[k_row]-lambda[k_row];

        criterion += a_row[n]*vector_row[k_row];
    }
    criterion *= rho;

    if (criterion - upper_bound > 0.0){
        lambda1 = (criterion - upper_bound)/scalar_row[row];
    }
    else if (- criterion + lower_bound > 0.0){
        lambda2 = (- criterion + lower_bound)/scalar_row[row];
    }
    for (k_row = size1; k_row<size1+D_row; ++k_row){
        phi[k_row] = - (lambda1-lambda2)*vector_row[k_row];
    }

    for (k_row = size2; k_row<size2+D_row*D_row; ++k_row){
        i_row = (k_row-size2) / D_row;
        j_row = (k_row-size2) % D_row;
        phi[size1+i_row] += rho*matrix_row[k_row]*a_row[j_row];
    }
}
'''

row_varD_comp_arg_type = [
    np.uint32, np.float32, np.float32, np.float32,
    None, None, None, None,
    None, None, None, None, None, None
]
kernel_row_varD_comp = '''
__kernel void row_varD_comp(
    const int state_dimension,
    
    const float rho,
    const float lower_bound,
    const float upper_bound,

    __global int* ini_pos,
    __global int* fin_pos,
    __global int* ini_pos_mat,
    __global int* fin_pos_mat,

    __global float* matrix_row,
    __global float* vector_row,
    __global float* scalar_row,
    __global float* psi,
    __global float* lambda,
    __global float* phi
){
    float a[1024];
    float lambda1;
    float lambda2;
    float criterion;
    int i;
    int j;
    int k;
    int n;
    int dim = 0;
    int row = get_global_id(0);
    
    lambda1 = 0.0;
    lambda2 = 0.0;
    criterion = 0.0;

    for (k = ini_pos[row], n = 0; k < fin_pos[row]; ++k, ++n){
        phi[k] = 0;
        a[n] = psi[k]-lambda[k];
        
        criterion += a[n]*vector_row[k];
        ++dim;
    }
    criterion *= rho;

    if (row < state_dimension) {
        // only the states are subjected to constraints

        if (criterion - upper_bound > 0.0){
            lambda1 = (criterion - upper_bound)/scalar_row[row];
        }
        else if (- criterion + lower_bound > 0.0){
            lambda2 = (- criterion + lower_bound)/scalar_row[row];
        }

        for (k = ini_pos[row]; k < fin_pos[row]; ++k){
            phi[k] = - (lambda1-lambda2)*vector_row[k];
        }
    }

    for (k = ini_pos_mat[row]; k < fin_pos_mat[row]; ++k){
        i = (k-ini_pos_mat[row]) / dim;
        j = (k-ini_pos_mat[row]) % dim;
        phi[ini_pos[row]+i] += rho*matrix_row[k]*a[j];
    }
}
'''

# ============================================================
#  Column computations
# ============================================================

col_comp_arg_type = [
    np.uint32,
    None, None, None, None, None
]
kernel_col_comp = '''
__kernel void col_comp(
    const int D,
    __global float* matrix,
    __global float* vector,
    __global float* phi,
    __global float* lambda,
    __global float* psi
){
    float a[1024];
    int i;
    int j;
    int k;
    int n;
    int dim;
    int column = get_global_id(0);

    int size1 = column*D;
    int size2 = size1*D;

    dim = 0;
    for (k = size1, n = 0; n < D; ++k, ++n){
        psi[k] = 0;
        a[n] = phi[k] + lambda[k];
        psi[k] = a[n] + vector[k];

        ++dim;
    }
    
    for (k = size2; k < size2 + D*D; ++k){
        i = (k-size2) / dim;
        j = (k-size2) % dim;
        psi[size1+i] += - matrix[k] * a[j];
    }
}
'''

col_varD_comp_arg_type = [
    None, None, None, None,
    None, None, None, None, None
]
kernel_col_varD_comp = '''
__kernel void col_varD_comp(
    __global int* ini_pos,
    __global int* fin_pos,
    __global int* ini_pos_mat,
    __global int* fin_pos_mat,

    __global float* matrix,
    __global float* vector,
    __global float* phi,
    __global float* lambda,
    __global float* psi
){
    float a[1024];
    int i;
    int j;
    int k;
    int n;
    int dim = 0;
    int column = get_global_id(0);

    for (k = ini_pos[column], n = 0; k < fin_pos[column]; ++k, n++){
        psi[k] = 0;
        a[n] = phi[k]+lambda[k];
        psi[k] = a[n] + vector[k];

        ++ dim;
    }

    for (k = ini_pos_mat[column]; k < fin_pos_mat[column]; ++k){
        i = (k-ini_pos_mat[column]) / dim;
        j = (k-ini_pos_mat[column]) % dim;
    
        psi[ini_pos[column]+i] += - matrix[k]*a[j];
    }
}
'''

# ============================================================
#  Lambda computations
# ============================================================

lag_comp_arg_type = [
    None, None, None, None
]
kernel_lag_comp = '''
__kernel void lag_comp(
    __global float* phi,
    __global float* psi,
    __global float* lambda_old,
    __global float* lambda
){
    int i = get_global_id(0);
    lambda[i] = phi[i] - psi[i] + lambda_old[i];
}
'''

# ============================================================
#  Convergence criterion
# ============================================================

conv_comp_arg_type = [
    np.uint32, np.float32, np.float32,
    None, None, None, None
]
kernel_conv_comp = '''
__kernel void conv_comp(
    const int D_col,
    const float eps_p,
    const float eps_d,
    __global float* phi,
    __global float* psi,
    __global float* psi_old,
    __global bool* yet_to_conv
){
    float conv1, conv2;
    int l_col;
    int column = get_global_id(0);

    int size1 = column*D_col;
    float tmp;

    conv1 = 0; conv2 = 0; yet_to_conv[column] = false;
    for (l_col = size1; l_col<size1+D_col; ++l_col){ // CONVERGENCE CRITERION
        tmp = psi[l_col]-psi_old[l_col];
        conv2 += tmp*tmp;
        tmp = phi[l_col]-psi[l_col];
        conv1 += tmp*tmp;
    }

    if (conv1 > eps_p || conv2 > eps_d){
        yet_to_conv[column] = true;
    }
}
'''

# ============================================================
#  Column + lambda computation + convergence criterion
# ============================================================
col_lag_conv_comp_arg_type = [
    np.uint32, np.uint32, np.float32, np.float32,
    None, None, None, None, None, None, None, None
]
kernel_col_lag_conv_comp = '''
__kernel void col_lag_conv_comp(
    const int D_col,
    const int DIM_lag,
    const float eps_p,
    const float eps_d,
    __global float* matrix_col,
    __global float* vector_col,
    __global float* phi,
    __global float* lambda,
    __global float* psi_old,
    __global float* lambda_new,
    __global float* psi,
    __global bool* yet_to_conv
){
    float a_col[1024];
    float conv1;
    float conv2;
    int n;
    int i_col;
    int j_col;
    int k_col;
    int l_col;
    int column = get_global_id(0);

    int size1 = column*D_col;
    int size2 = size1*D_col;
    float tmp;

    for (k_col = size1, n = 0; n < D_col; ++k_col, ++n){
        psi[k_col] = 0;
        a_col[n] = phi[k_col]+lambda[k_col];
        psi[k_col] = a_col[n] + vector_col[k_col];
    }

    for (k_col = size2; k_col<size2 + D_col*D_col; ++k_col){
        i_col = (k_col-size2) / D_col;
        j_col = (k_col-size2) % D_col;
        psi[size1+i_col] -= matrix_col[k_col]*a_col[j_col];
    }

    conv1 = 0; conv2 = 0; yet_to_conv[column] = false;
    for (l_col = size1; l_col<size1+D_col; ++l_col){ // CONVERGENCE CRITERION + LAGRANGE
        tmp = psi[l_col]-psi_old[l_col];
        conv2 += tmp*tmp;
        tmp = phi[l_col]-psi[l_col];
        conv1 += tmp*tmp;
        lambda_new[l_col] = tmp + lambda[l_col];
    }

    if (conv1 > eps_p || conv2 > eps_d){
        yet_to_conv[column] = true;
    }
}
'''

# ============================================================
#  Row + column + lambda computation + convergence criterion
# ============================================================


row_col_lag_conv_comp_arg_type = [ 
    np.uint32, np.uint32, 
    np.float32, np.float32, np.float32, np.float32, np.float32,
    None, None, None, None, None, 
    None, None, None,
    None, None,
    None,
    None, None, None
]


kernel_row_col_lag_conv_comp = ''' 
__kernel void row_col_lag_conv_comp(
    
    const int D_row,
    const int D_col,

    const float rho,
    const float lower_bound,
    const float upper_bound,
    const float eps_p,
    const float eps_d,

    __global float* matrix_row,
    __global float* vector_row,
    __global float* scalar_row,
    __global float* matrix_col,
    __global float* vector_col,

    __global int* row2col,
    __global int* patch_of_rows,
    __global int* patch_of_columns,
    
    __global float* psi_old,
    __global float* lambda_old,
    
    __local float* phi,

    __global float* psi,
    __global float* lambda,
    __global bool* yet_to_conv
){
    
    float a_row[1024];
    float a_col[1024];
    float lambda1;
    float lambda2;
    float criterion;
    float phi_row[1024];

    float conv1; 
    float conv2;

    float tmp;

    bool relevant_row;
    
    int i, j, k, n;
    int row;
    int size1, size2, size3, size4;

    int column = get_global_id(0);
    int local_row = get_local_id(1);
    
    // ROW COMPUTATIONS

    size3 = column*D_col;  
    size4 = size3*D_col;

    row = patch_of_rows[local_row+size3];

    size1 = row*D_row;
    size2 = size1*D_row;

    lambda1 = 0.0;
    lambda2 = 0.0;
    criterion = 0.0;
 

    for (k = 0; k < D_row; ++k){
        phi_row[k] = 0;
        a_row[k] = psi_old[row2col[k+size1]]-lambda_old[row2col[k+size1]];
        criterion += a_row[k]*vector_row[k+size1];
    }
    
    criterion *= rho;

    if (criterion - upper_bound > 0.0){
        lambda1 = (criterion - upper_bound)/scalar_row[row];
    }
    else if (- criterion + lower_bound > 0.0){
        lambda2 = (- criterion + lower_bound)/scalar_row[row];
    }
    
    for (k = 0; k<D_row; ++k){
        phi_row[k] = - (lambda1-lambda2)*vector_row[k+size1];
    }

    for (k = 0; k<D_row*D_row; ++k){
        i = k / D_row;
        j = k % D_row;
        phi_row[i] += rho*matrix_row[k+size2]*a_row[j];
    }


    // FORM COLUMNS
    
    relevant_row = false;
    for (k = 0; k < D_row; ++k){
        if (patch_of_columns[k+size1] == column){
            phi[local_row] = phi_row[k];
            relevant_row = true;
            break;
        }
    }
    if (relevant_row == false){
        phi[local_row] = 0;
    }           

    barrier(CLK_LOCAL_MEM_FENCE); // TO MAKE SURE phi IS COMPUTED BEFORE WE PROCEED

    if (local_row>0){   
        return;
    }

    // COLUMN COMPUTATIONS

    for (k = size3, n = 0; n < D_col; ++k, ++n){
        psi[k] = 0;
        a_col[n] = phi[n]+lambda_old[k];
        psi[k] = a_col[n] + vector_col[k];
    }

    for (k = size4, n = 0; n < D_col*D_col; ++k, ++n){
        i = n / D_col;
        j = n % D_col;
        psi[size3+i] -= matrix_col[k]*a_col[j];
    }

    // LAGRANGE AND CONVERGENCE CRITERION

    conv1 = 0; conv2 = 0; yet_to_conv[column] = false;
    for (k = size3, n = 0; n < D_col; ++k, ++n){ 
        tmp = psi[k]-psi_old[k];
        conv2 += tmp*tmp;
        tmp = phi[n]-psi[k];
        conv1 += tmp*tmp;
        lambda[k] = tmp + lambda_old[k];
    }

    if (conv1 > eps_p || conv2 > eps_d){
        yet_to_conv[column] = true;
    }
}
'''

# ============================================================
# Dynamics
# ============================================================

dynamics_from_psi_arg_type = [np.uint32, np.uint32,  None, None, None]
 
kernel_dynamics_from_psi = '''
          __kernel void dynamics_from_psi(
          const int DIM,
          const int D,
          __global float * x0_maxD,
          __global float * phi,
          __global float * result
      )
      
      {   int i;
          int row = get_global_id(0);
         
          if (row < DIM){
              result[row]=0;
              for (i=row*D; i<(row+1)*D; i++){
                   result[row] += phi[i]*x0_maxD[i];
              }
          }
  
      } '''

'''

     ####### PSEUDOKERNEL

     self._result = np.zeros(Nx+Nu)
     for row in range(Nx+Nu):

         if row < Nx:
             columns = (Nx+row)*D
         else:
             columns = (Nx*(T-1)+row)*D

         #for j in range(columns, columns+D):
             #self._result[row] += self._psi[self._row2col[j]]*self._x0_maxD[j]

         self._result[row] = np.matmul(self._psi[self._row2col[columns:columns+D]],self._x0_maxD[columns:columns+D])

         #for i in range(row*D, (row+1)*D):
         #   self._result[row] += phi_dynamics[i]*self._x0_maxD[i]

     self._result = np.array(self._result)

     x = self._result[0:Nx]
     u = self._result[Nx:Nx+Nu]

'''
