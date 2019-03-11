import torch
import numpy as np

from rg import TRG

def build_tensor(K):
    lam = [torch.cosh(K)*2, torch.sinh(K)*2]
    T = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    if ((i+j+k+l)%2==0):
                        T.append(torch.sqrt(lam[i]*lam[j]*lam[k]*lam[l])/2.)
                    else:
                        T.append(torch.tensor(0.0, dtype=K.dtype, device=K.device))
    T = torch.stack(T).view(2, 2, 2, 2)
    return T

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-chi", type=int, default=24, help="chi")
    parser.add_argument("-Niter", type=int, default=20, help="Niter")
    parser.add_argument("-use_checkpoint", action='store_true', help="use checkpoint")
    parser.add_argument("-float32", action='store_true', help="use float32")
    parser.add_argument("-cuda", type=int, default=-1, help="GPU #")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cuda<0 else "cuda:"+str(args.cuda))
    dtype = torch.float32 if args.float32 else torch.float64

    for K in np.linspace(0.4, 0.5, 51):
        beta = torch.tensor(K, dtype=dtype, device=device).requires_grad_()
        T = build_tensor(beta)
        lnZ = TRG(T, args.chi, args.Niter, use_checkpoint=args.use_checkpoint)
        dlnZ, = torch.autograd.grad(lnZ, beta,create_graph=True) #  En = -d lnZ / d beta
        dlnZ2, = torch.autograd.grad(dlnZ, beta) # Cv = beta^2 * d^2 lnZ / d beta^2
        print (K, lnZ.item(), -dlnZ.item(), dlnZ2.item()*beta.item()**2)
