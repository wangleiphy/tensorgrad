import torch
from .adlib import SVD 
svd = SVD.apply

def TRG(T, chi, no_iter,  epsilon=1E-15):

    D = T.shape[0]
    lnZ = 0.0
    for n in range(no_iter):
        
        #print(n, " ", T.max(), " ", T.min())
        maxval = T.abs().max()
        T = T/maxval 
        lnZ += 2**(no_iter-n)*torch.log(maxval)

        Ma = T.permute(2, 1, 0, 3).contiguous().view(D**2, D**2)
        Mb = T.permute(3, 2, 1, 0).contiguous().view(D**2, D**2)

        Ua, Sa, Va = svd(Ma)
        Ub, Sb, Vb = svd(Mb)

        D_new = min(min(D**2, chi), min((Sa>epsilon).sum().item(), (Sb>epsilon).sum().item()))
    
        S1 = (Ua[:, :D_new]* torch.sqrt(Sa[:D_new])).view(D, D, D_new)
        S3 = (Va[:, :D_new]* torch.sqrt(Sa[:D_new])).view(D, D, D_new)
        S2 = (Ub[:, :D_new]* torch.sqrt(Sb[:D_new])).view(D, D, D_new)
        S4 = (Vb[:, :D_new]* torch.sqrt(Sb[:D_new])).view(D, D, D_new)

        T_new = torch.einsum('war,abu,bgl,gwd->ruld', (S1, S2, S3, S4))

        D = D_new
        T = T_new

    trace = 0.0
    for i in range(D):
        trace += T[i, i, i, i]
    lnZ += torch.log(trace)

    return lnZ 

