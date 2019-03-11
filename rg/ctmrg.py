import torch
from torch.utils.checkpoint import checkpoint

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


def CTMRG(T, chi, max_iter, use_checkpoint=False):
    # T(up, left, down, right)

    threshold = 1E-12 if T.dtype is torch.float64 else 1E-6 # ctmrg convergence threshold

    # C(down, right), E(up,right,down)
    C = T.sum((0,1))  #
    E = T.sum(1).permute(0,2,1)

    truncation_error = 0.0
    sold = torch.zeros(chi, dtype=T.dtype, device=T.device)
    diff = 1E1
    for n in range(max_iter):
        tensors = C, E, T, torch.tensor(chi)
        if use_checkpoint: # use checkpoint to save memory
            C, E, s, error = checkpoint(renormalize, *tensors)
        else:
            C, E, s, error = renormalize(*tensors)

        Enorm = E.norm()
        E = E/Enorm
        truncation_error += error.item()
        if (s.numel() == sold.numel()):
            diff = (s-sold).norm().item()
            #print( s, sold )
        #print( 'n: %d, Enorm: %g, error: %e, diff: %e' % (n, Enorm, error.item(), diff) )
        if (diff < threshold):
            break
        sold = s
    print ('ctmrg iterations, diff, error', n, diff, truncation_error/n)

    return C, E

if __name__=='__main__':
    import time
    torch.manual_seed(42)
    D = 6
    chi = 80
    max_iter = 100
    device = 'cpu'

    # T(u,l,d,r)
    T = torch.randn(D, D, D, D, dtype=torch.float64, device=device, requires_grad=True)

    T = (T + T.permute(0, 3, 2, 1))/2.      # left-right symmetry
    T = (T + T.permute(2, 1, 0, 3))/2.      # up-down symmetry
    T = (T + T.permute(3, 2, 1, 0))/2.      # skew-diagonal symmetry
    T = (T + T.permute(1, 0, 3, 2))/2.      # digonal symmetry
    T = T/T.norm()

    C, E = CTMRG(T, chi, max_iter, use_checkpoint=True)
    C, E = CTMRG(T, chi, max_iter, use_checkpoint=False)
    print( 'diffC = ', torch.dist( C, C.t() ) )
    print( 'diffE = ', torch.dist( E, E.permute(2,1,0) ) )

