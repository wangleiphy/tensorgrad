import torch    
from .adlib import EigenSolver
symeig = EigenSolver.apply

def renormalize(*tensors):
    # T(up,left,down,right), u=up, l=left, d=down, r=right
    # C(d,r), EL(u,r,d), EU(l,d,r)

    C, E, T, chi = tensors

    dimT, dimE = T.shape[0], E.shape[0]
    D_new = min(dimE*dimT, chi)
    
    # step 1: contruct the density matrix Rho
    Rho = torch.tensordot(C,E,([1],[0]))        # C(ef)*EU(fga)=Rho(ega)
    Rho = torch.tensordot(Rho,E,([0],[0]))      # Rho(ega)*EL(ehc)=Rho(gahc)
    Rho = torch.tensordot(Rho,T,([0,2],[0,1]))  # Rho(gahc)*T(ghdb)=Rho(acdb)
    Rho = Rho.permute(0,3,1,2).contiguous().view(dimE*dimT, dimE*dimT)  # Rho(acdb)->Rho(ab;cd)

    Rho = Rho+Rho.t()
    Rho = Rho/Rho.norm()

    if (not torch.isfinite(Rho).all()):
        print ('Rho is not finite!!')
    
    # step 2: Get Isometry P
    """
    U, S, V = svd(Rho)
    truncation_error = S[D_new:].sum()/S.sum()
    P = U[:, :D_new] # projection operator
    """

    S, U = symeig(Rho)
    sorted, indices = torch.sort(S.abs(), descending=True)
    truncation_error = sorted[D_new:].sum()/sorted.sum() 
    S = S[indices][:D_new]
    P = U[:, indices][:, :D_new] # projection operator

    # step 3: renormalize C and E
    C = (P.t() @ Rho @ P) #C(D_new, D_new)
    
    ## EL(u,r,d)
    P = P.view(dimE,dimT,D_new)
    E = torch.tensordot(E, P, ([0],[0]))  # EL(def)P(dga)=E(efga)
    E = torch.tensordot(E, T, ([0,2],[1,0]))  # E(efga)T(gehb)=E(fahb)
    E = torch.tensordot(E, P, ([0,2],[0,1]))  # E(fahb)P(fhc)=E(abc)
    
    # step 4: symmetrize C and E
    C = 0.5*(C+C.t())
    E = 0.5*(E + E.permute(2, 1, 0))

    return C/C.norm(), E, S.abs()/S.abs().max(), truncation_error

