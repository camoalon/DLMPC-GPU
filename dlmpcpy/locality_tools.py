import numpy as np
from mpc.system import System

# TODO: functionalize, update getLocality in system.py
# Note: locality conventions for python are same as MATLAB; (1 = only self)

Nx   = 2
Nu   = 1
A    = np.array([[3,2],[1,0]])
B2   = np.array([[1],[0]])
T    = 3
eps  = 1e-8
d    = 2

sys = System(A=A, B2=B2, Nx=Nx, Nu=Nu)

# TODO: don't forget to add locality adjustment for grid
#       idea: let pass in a commMtx instead of using A ~= 0
# TODO: can speed up by using mldivide, sparse matrices, zero-col trimming

sys._locality_d = d
x0        = np.ones([sys._Nx, 1])
nPhi      = sys._Nx*T + sys._Nu*(T-1);
(IO, ZAB) = sys.getFeasibilityConstraints(T)
h         = np.reshape(IO.T, [IO.size, 1])

sys._locality_Phix = None
sys._locality_Phiu = None
sys.getLocalityRegion()

phiSupp = np.empty([0, Nx])
for i in range(T):
    phiSupp = np.concatenate((phiSupp, sys._locality_Phix))
for i in range(T-1):
    phiSupp = np.concatenate((phiSupp, sys._locality_Phiu))

phiSuppFlat = np.reshape(phiSupp.T, [phiSupp.size, 1])
suppIdx     = np.nonzero(phiSuppFlat)[0]
suppSize    = len(suppIdx)

mtx = np.empty([nPhi-Nx, 0])
for i in range(sys._Nx):
    idxStart = i*nPhi
    idxEnd   = (i+1)*nPhi-1
    myIdx = suppIdx[suppIdx >= idxStart]
    myIdx = myIdx[myIdx <= idxEnd] - i*nPhi
    nCols = len(myIdx)
    
    Hi = ZAB[:,myIdx]
    hi = h[i*Nx*T:(i+1)*Nx*T]
    
    # Python has no mldivide; linalg.lstsq uses pinverse anyway, so calculate it now
    Hip     = np.linalg.pinv(Hi)
    testSol = Hip @ hi 

    # Note: using max instead of euclidean norm to be less dimension-dependent
    if max(abs(Hi @ testSol - hi))[0] > eps: # No solution exists
        break
    
    IHHi = np.eye(nCols) - Hip @ Hi
    xi   = x0[i] * np.eye(nPhi)
    mtx  = np.concatenate((mtx, xi[Nx:, myIdx] @ IHHi), axis=1)

# TODO: can check mtx.size == 0 at end


#maxLoc = Nx
#for locality in range(1, maxLoc+1): # convention, 1 = no neighbors
#    print(f'Checking locality size {locality}')
#    # TODO: get mtx
#    rankRatio = np.linalg.matrix_rank(mtx) / sys._Nu*(T-1)
#    print(f'\tRank ratio: {rankRatio:.2f')    
#    if rankRatio == 1:
#        break
# print(f'Final locality: {locality}')
