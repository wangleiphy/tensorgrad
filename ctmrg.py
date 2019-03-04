import torch
from renormalize import renormalize
from torch.utils import checkpoint

def ctmrg(T, chi, max_iter, use_checkpoint=False):

    threshold = 1E-12 if T.dtype is torch.float64 else 1E-6 # ctmrg convergence threshold  

    C = T.sum((0,1))
    E = T.sum(1)
    
    truncation_error = 0.0
    diff = 1E10
    sold = torch.zeros(chi, dtype=T.dtype, device=T.device)
    for n in range(max_iter):
        tensors = T, C, E
        if use_checkpoint: # use checkpoint to save memory 
            C, E, s, error = checkpoint(*tensors)
        else:
            C, E, s, error = renormalize(*tensors) 

        truncation_error = max(truncation_error, error.item())
        if (s.numel() == sold.numel()):
            diff = (s-sold).norm().item()
            #print( 'n: %d, error: %e, diff: %e' % (n, error.item(), diff) ) 

        if (diff < threshold):
            break
        sold = s
    print ('ctmrg iterations, diff, error', n, diff, truncation_error/n) 
    
    return C, E

if __name__=='__main__':
    import time 
    D = 64
    chi = 150
    device = 'cuda:0'
    T = torch.randn(D, D, D, D, dtype=torch.float64, device=device, requires_grad=True)
    T = (T + T.permute(3, 1, 2, 0))/2.             
    T = (T + T.permute(0, 2, 1, 3))/2. 
    T = (T + T.permute(2, 3, 0, 1))/2. 
    T = (T + T.permute(1, 0, 3, 2))/2. 
    T = T/T.norm()

    C = torch.randn(chi, chi, dtype=torch.float64, device=device, requires_grad=True)
    C = (C+C.t())/2.
    E = torch.randn(chi, D, chi, dtype=torch.float64, device=device, requires_grad=True)
    E = (E + E.permute(2, 1, 0))/2.
    args = C, E, T, torch.tensor(chi)
    checkpoint(renormalize, *args)
