import torch
from adlib import EigenSolver
symeig = EigenSolver.apply
from args import args

def renormalize(*tensors):
    T, C, E = tensors

    D, d = E.shape[0], E.shape[1]
    # M = torch.einsum('ab,eca,bdg,cdfh->efgh', (C, E, E, T)).contiguous().view(D*d, D*d)
    M = torch.tensordot(E,C,dims=1)  # E(eca)*C(ab)=M(ecb)
    M = torch.tensordot(M,E,dims=1)  # M(ecb)*E(bdg)=M(ecdg)
    M = torch.tensordot(M,T,dims=([1,2],[1,0]))  # M(ecdg)*T(dcfh)=M(egfh)
    M = M.permute(0,2,1,3).contiguous().view(D*d, D*d)  # M(egfh)->M(ef;gh)

    M = (M+M.t())/2.
    M = M/M.norm()

    D_new = min(D*d, args.chi)
    if (not torch.isfinite(M).all()):
        print ('M is not finite!!')
    
    #U, S, V = svd(M)
    #truncation_error = S[D_new:].sum()/S.sum()
    #P = U[:, :D_new] # projection operator

    #S, U = torch.symeig(M, eigenvectors=True)
    S, U = symeig(M)
    sorted, indices = torch.sort(S.abs(), descending=True)
    truncation_error = sorted[D_new:].sum()/sorted.sum() 
    S = S[indices][:D_new]
    P = U[:, indices][:, :D_new] # projection operator

    C = (P.t() @ M @ P) #(D, D)
    C = (C+C.t())/2.
    
    ## EL(u,r,d)
    P = P.view(D,d,D_new)
    E = torch.tensordot(E, P, ([0],[0]))  # E(dhf)P(dea)=E(hfea)
    E = torch.tensordot(E, T, ([0,2],[1,0]))  # E(hfea)T(ehgb)=E(fagb)
    E = torch.tensordot(E, P, ([0,2],[0,1]))  # E(fagb)P(fgc)=E(abc)
    
    #ET = torch.einsum('ldr,adbc->labrc', (E, T)).contiguous().view(D*d, d, D*d)
    #ET = torch.tensordot(E, T, dims=([1], [1]))
    #ET = ET.permute(0, 2, 3, 1, 4).contiguous().view(D*d, d, D*d)
    #E = torch.einsum('li,ldr,rj->idj', (P, ET, P)) #(D_new, d, D_new)

    E = (E + E.permute(2, 1, 0))/2.

    return C/C.norm(), E/E.norm(), S/S.max(), truncation_error
