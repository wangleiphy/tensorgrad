import torch 
import time
from rg import CTMRG
from measure import get_obs
from utils import symmetrize
from args import args

class iPEPS(torch.nn.Module):
    def __init__(self, args, dtype=torch.float64, device='cpu', use_checkpoint=False):
        super(iPEPS, self).__init__()
        self.d = args.d
        self.D = args.D
        self.chi = args.chi
        self.Niter = args.Niter
        self.use_checkpoint = use_checkpoint 
        
        d, D = self.d, self.D
        # B(phy, up, left, down, right)
        
        # Note: if we initialize B by torch.randn, the eigenvalues of the density matrix Rho in ctmrg 
        # will exactly two-fold degenerate due to symmetrization of B. This will cause the energy higher
        # than the true ground state energy, but eventually the variational energy will converge to the correct value.
        # Thus we usually use torch.rand to initialize B to avoid the unphysical problem.

        B = torch.rand(d, D, D, D, D, dtype=dtype, device=device)
        B = B/B.norm()
        self.A = torch.nn.Parameter(B)
        
    def forward(self, H, Mpx, Mpy, Mpz, chi):
        # Asymm(phy, up, left, down, right), T(up, left, down, right)
        
        d, D, chi, Niter = self.d, self.D, self.chi, self.Niter
        Asymm = symmetrize(self.A)

        T = (Asymm.view(d, -1).t()@Asymm.view(d, -1)).contiguous().view(D, D, D, D, D, D, D, D)
        T = T.permute(0,4, 1,5, 2,6, 3,7).contiguous().view(D**2, D**2, D**2, D**2)
        T = T/T.norm()

        C, E = CTMRG(T, chi, Niter, self.use_checkpoint) 
        loss, Mx, My, Mz = get_obs(Asymm, H, Mpx, Mpy, Mpz, C, E)

        return loss, Mx, My, Mz 
