import torch
from adlib import SVD
svd = SVD.apply

def TRG(K, Dcut, no_iter, device='cpu', epsilon=1E-15):
    D = 2

    lam = [torch.cosh(K)*np.sqrt(2), torch.sinh(K)*np.sqrt(2)]
    T = []
    for i in range(D):
        for j in range(D):
            for k in range(D):
                for l in range(D):
                    if ((i+j-k-l)%2==0):
                        T.append(torch.sqrt(lam[i]*lam[j]*lam[k]*lam[l]))
                    else:
                        T.append(torch.tensor(0.0, dtype=K.dtype, device=K.device))
    T = torch.stack(T).view(D, D, D, D)
    
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

        D_new = min(min(D**2, Dcut), min((Sa>epsilon).sum().item(), (Sb>epsilon).sum().item()))
    
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

if __name__=="__main__":
    import numpy as np 
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-float32", action='store_true', help="use float32")
    parser.add_argument("-cuda", type=int, default=-1, help="use GPU")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cuda<0 else "cuda:"+str(args.cuda))
    dtype = torch.float32 if args.float32 else torch.float64

    Dcut = 24
    n = 20

    for K in np.linspace(0.4, 0.5, 11):
        beta = torch.tensor(K, dtype=dtype, device=device).requires_grad_()
        lnZ = TRG(beta, Dcut, n, device=device)
        dlnZ, = torch.autograd.grad(lnZ, beta,create_graph=True) #  En = -d lnZ / d beta
        dlnZ2, = torch.autograd.grad(dlnZ, beta) # Cv = beta^2 * d^2 lnZ / d beta^2
        print (K, lnZ.item()/2**n, -dlnZ.item()/2**n, dlnZ2.item()*beta.item()**2/2**n)
