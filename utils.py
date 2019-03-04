import re
import torch 

def kronecker_product(t1, t2):
    """
    Computes the Kronecker product between two tensors.
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_height, t2_width, 1)
          .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2

def symmetrize(A):
    '''
    A(phy, up, left, down, right)
    left-right, up-down, diagonal symmetrize 
    '''
    Asymm = (A + A.permute(0, 1, 4, 3, 2))/2.           # left-right symmetry            
    Asymm = (Asymm + Asymm.permute(0, 3, 2, 1, 4))/2.   # up-down symmetry
    Asymm = (Asymm + Asymm.permute(0, 4, 3, 2, 1))/2.   # skew-diagonal symmetry
    Asymm = (Asymm + Asymm.permute(0, 2, 1, 4, 3))/2.   # diagonal symmetry

    return Asymm/Asymm.norm()

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    #print(model.state_dict().keys())
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, args, model):
    print( 'load old model from %s ' % checkpoint_path )
    print( 'Dold = ', re.search('_D([0-9]*)_', checkpoint_path).group(1) )
    Dold = int(re.search('_D([0-9]*)_', checkpoint_path).group(1))
    
    d, D = args.d, args.D
    dtype, device = model.A.dtype, model.A.device

    if (Dold != D):
        B = torch.rand( d, Dold, Dold, Dold, Dold, dtype=dtype, device=device)
        model.A = torch.nn.Parameter(B)

    state = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])

    if (Dold != D):
        Aold = model.A.data
        B = 1E-2*torch.rand( d, D, D, D, D, dtype=dtype, device=device)
        B[:, :Dold, :Dold, :Dold, :Dold] = Aold.reshape(d, Dold, Dold, Dold, Dold)
        model.A = torch.nn.Parameter(B)

def mem_report():
    import gc 
    import psutil 
    import sys 
    import os
    import numpy as np 
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print (type(obj), obj.size(), np.prod(obj.size())*8/2**30)
    
    print(sys.version)
    print (psutil.virtual_memory())
    pid = os.getpid()
    py = psutil.Process(pid)
    print ('memory GB:', py.memory_info()[0]/2**30)


if __name__=='__main__':
    import torch 
    A = torch.arange(4).view(2,2)
    B = torch.arange(9).view(3,3)
    print (A)
    print (B)
    print (kronecker_product(A, B))

    A = torch.randn(2,4,4,4,4) 
    print ('A', A)
    Asymm = symmetrize(A)
    print ('A', A)
    print ('Asymm', Asymm)

