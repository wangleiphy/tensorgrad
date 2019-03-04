import torch
from .renormalize import renormalize
from torch.utils.checkpoint import checkpoint

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

