import torch 
from ctmrg import ctmrg 
from measure import get_obs
from utils import symmetrize
from args import args

class iPEPS(torch.nn.Module):
    def __init__(self, args, dtype=torch.float64, device='cpu', use_checkpoint=False):
        super(iPEPS, self).__init__()

        B = torch.rand(args.d, args.D, args.D, args.D, args.D, dtype=dtype, device=device)
        B = B/B.norm()
        self.A = torch.nn.Parameter(B)
        
    def forward(self, H, Mpx, Mpy, Mpz):

        Asymm = symmetrize(self.A)

        d, D = Asymm.shape[0], Asymm.shape[1]
        T = (Asymm.view(d, -1).t()@Asymm.view(d, -1)).view(D, D, D, D, D, D, D, D).permute(0,4, 1,5, 2,6, 3,7).contiguous().view(D**2, D**2, D**2, D**2)
        T = T/T.norm()

        C, E = ctmrg(T, args.chi, args.Maxiter, args.use_checkpoint) 
        loss, Mx, My, Mz = get_obs(Asymm, H, Mpx, Mpy, Mpz, C, E)

        return loss, Mx, My, Mz 
