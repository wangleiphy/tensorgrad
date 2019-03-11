import torch
from torch.utils.checkpoint import checkpoint

from .adlib import SVD
svd = SVD.apply

def renormalize(*args):
    T, chi, epsilon = args

    D = T.shape[0]
    Ma = T.view(D**2, D**2)
    Mb = T.permute(1, 2, 0, 3).contiguous().view(D**2, D**2)

    Ua, Sa, Va = svd(Ma)
    Ub, Sb, Vb = svd(Mb)

    D_new = min(min(D**2, chi), min((Sa>epsilon).sum().item(), (Sb>epsilon).sum().item()))

    S1 = (Ua[:, :D_new]* torch.sqrt(Sa[:D_new])).view(D, D, D_new)
    S3 = (Va[:, :D_new]* torch.sqrt(Sa[:D_new])).view(D, D, D_new)
    S2 = (Ub[:, :D_new]* torch.sqrt(Sb[:D_new])).view(D, D, D_new)
    S4 = (Vb[:, :D_new]* torch.sqrt(Sb[:D_new])).view(D, D, D_new)

    return torch.einsum('xwu,yxl,yzd,wzr->uldr', (S2, S3, S4, S1))

def TRG(T, chi, no_iter, use_checkpoint=False,  epsilon=1E-15):

    lnZ = 0.0
    for n in range(no_iter):
        maxval = T.abs().max()
        T = T/maxval
        lnZ += 2**(no_iter-n)*torch.log(maxval)

        args = T, torch.tensor(chi), torch.tensor(epsilon, dtype=T.dtype, device=T.device)
        if use_checkpoint: # use checkpoint to save memory
            T = checkpoint(renormalize, *args)
        else:
            T = renormalize(*args)

    trace = 0.0
    for x in range(T.shape[0]):
        for y in range(T.shape[1]):
            trace += T[x, y, x, y]
    lnZ += torch.log(trace)

    return lnZ/2**no_iter

