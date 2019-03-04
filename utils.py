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
    A(phy,u,l,d,r)
    left-right, up-down, diagonal symmetrize 
    '''
    Asymm = (A + A.permute(0, 1, 4, 3, 2))/2. 
    Asymm = (Asymm + Asymm.permute(0, 3, 2, 1, 4))/2. 
    Asymm = (Asymm + Asymm.permute(0, 2, 1, 4, 3))/2. 
    Asymm = (Asymm + Asymm.permute(0, 4, 3, 2, 1))/2. 

    return Asymm/Asymm.norm()

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    #print(model.state_dict().keys())
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    Dold = state['state_dict']['A'].shape[1]
    if (Dold != model.D):
        A = 0.01*torch.randn(model.d, model.D, model.D, model.D, model.D, dtype=model.A.dtype, device=model.A.device) # some pertubation 
        A[:, :Dold, :Dold, :Dold, :Dold] = state['state_dict']['A']
        state['state_dict']['A'] = A
    else: # since we changed D, have to reinitialize optimizer 
        optimizer.load_state_dict(state['optimizer'])
    #print (state['state_dict'])
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

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

