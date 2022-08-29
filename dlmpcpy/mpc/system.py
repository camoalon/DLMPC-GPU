import numpy as np
import scipy
from slspy import *

class System (LTI_System):

    def __init__(self, 
                A = None,
                B2 = None,
                locality_d = None,
                upper_bound_x = None,
                lower_bound_x = None,
                upper_bound_u = None,
                lower_bound_u = None,
                ** kwargs):

        LTI_System.__init__(self, **kwargs)

        self._A = A
        self._B2 = B2 
        self._state_upper_bound = upper_bound_x
        self._state_lower_bound = lower_bound_x
        self._input_upper_bound = upper_bound_u
        self._input_lower_bound = lower_bound_u

        self.getLocality(locality_d)

    def getLocality(self,locality_d):
            
        self._locality_Phix = None
        self._locality_Phiu = None

        if locality_d is None:
            # DO FUNCTION TO COMPUTE LOCALITY
            pass
        else:
            self._locality_d = locality_d 

    def getLocalityRegion(self):

        if self._locality_Phix is not None:
            return

        Nx = self._Nx
        Nu = self._Nu
        A  = self._A
        B  = self._B2
        d  = self._locality_d
 
        Aux = np.identity(Nx)
        A_support = np.zeros((Nx,Nx))
        for i in range(Nx):
            A_support[i] = [int(x) for x in A[i]!=0]
 
        for i in range(d-1):
            Aux = np.matmul(A_support,Aux)
        self._locality_Phix = Aux!=0
        self._locality_Phiu = np.matmul(np.transpose(B),Aux)!=0
        
    def getColumnPatches(self,T):
        # For shoter notation
        Nx = self._Nx
        Nu = self._Nu
        d  = self._locality_d
 
        self.getLocalityRegion()
        locality_Phix = self._locality_Phix
        locality_Phiu = self._locality_Phiu

        column_patch = []
        
        max_rows = 0
        
        for i in range(Nx):
            set_rows = []
            
            for j in range(Nx):
                if locality_Phix[j,i] == True:
                    this_row = [j+Nx*t for t in range(T)]
                    set_rows = np.concatenate((set_rows, this_row), axis=0)
            for j in range(Nu):
                if locality_Phiu[j,i] == True:
                    this_row = [j+Nx*T+Nu*t for t in range(T-1)]
                    set_rows = np.concatenate((set_rows, this_row), axis=0)
                    
            set_rows = [int(x) for x in set_rows]
            
            if len(set_rows)>max_rows: # Keep track of the maximum number of rows (useful in GPU)
                max_rows = len(set_rows)
            
            column_patch.append(np.sort(set_rows))
            
        column_patch = np.array(column_patch,dtype=object)

        return column_patch, max_rows

    def getRowPatches(self,T):
        # For shoter notation
        Nx = self._Nx
        Nu = self._Nu
        d  = self._locality_d
     
        self.getLocalityRegion()
        locality_Phix = self._locality_Phix
        locality_Phiu = self._locality_Phiu

        row_patch = []
        
        max_columns = 0
        
        for i in range(Nx*T+Nu*(T-1)):
            set_columns= []
            
            for j in range(Nx):
                if i<Nx*T and locality_Phix[np.fmod(i,Nx),j]:
                    this_column = j
                    set_columns = np.append(set_columns, this_column)
                elif i>Nx*T-1 and locality_Phiu[np.remainder(i-Nx*T,Nu),j]:
                    this_column = j
                    set_columns = np.append(set_columns, this_column)
                    
            set_columns = [int(x) for x in set_columns]
            
            if len(set_columns)>max_columns: # Keep track of the maximum number of columns (useful in GPU)
                max_columns = len(set_columns)
            
            row_patch.append(np.sort(set_columns))
        
        row_patch = np.array(row_patch,dtype=object)
        
        return row_patch, max_columns    


    def getFeasibilityConstraints(self,T):

        Nx = self._Nx
        Nu = self._Nu
        A  = self._A
        B  = self._B2

        # Construct the RHS of the feasibility constraints
        E = np.concatenate((np.identity(Nx),np.zeros((Nx*(T-1),Nx))),axis=0)
 
        # Construct the Z_{AB} matrix
        I = np.kron(np.identity(T),np.identity(Nx))
 
        Z = np.kron(np.identity(T-1),np.identity(Nx))
        Z = np.concatenate((Z,np.zeros((Nx*(T-1),Nx))),axis=1)
        Z = np.concatenate((np.zeros((Nx,Nx*(T))),Z),axis=0)
 
        A_blk = []
        B_blk = []
        for i in range(T):
             A_blk = scipy.linalg.block_diag(A,A_blk)
             B_blk = scipy.linalg.block_diag(B,B_blk)
        A_blk = A_blk[:-1]
        B_blk = B_blk[:-1]
 
        IZA_ZB = np.concatenate((I-np.matmul(Z,A_blk), np.matmul(-Z,B_blk)),axis=1)
        IZA_ZB = IZA_ZB[:,:-Nu]
 
        return E,IZA_ZB
