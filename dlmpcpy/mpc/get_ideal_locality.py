import numpy as np

def get_ideal_locality(sys, T, eps=1e-8):
    x0        = np.ones([sys._Nx, 1]) # Values don't matter as along as all nonzero
    nPhi      = sys._Nx*T + sys._Nu*(T-1)
    (IO, ZAB) = sys.getFeasibilityConstraints(T)
    h         = np.reshape(IO.T, [IO.size, 1])

    Nx = sys._Nx
    for locality in range(2, Nx+1): # locality=1 means no neighbors
        print(f'Checking locality size {locality}')
        sys._locality_d    = locality
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
                mtx = 0 # Will report rank = 0
                break
            
            IHHi = np.eye(nCols) - Hip @ Hi
            xi   = x0[i] * np.eye(nPhi)
            mtx  = np.concatenate((mtx, xi[Nx:, myIdx] @ IHHi), axis=1)

        rankRatio = np.linalg.matrix_rank(mtx) / sys._Nu / (T-1)
        print(f'\tRank ratio: {rankRatio:.2f}')    
        if rankRatio == 1:
            break
    
    return locality
