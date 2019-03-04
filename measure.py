import torch

def get_obs(Asymm, H, Sx, Sy, Sz, C, E ):
    # A(phy,u,l,d,r), C(d,r), E(u,r,d)
    
    Da = Asymm.size()
    Td = torch.einsum('mefgh,nabcd->eafbgchdmn',(Asymm,Asymm)).contiguous().view(Da[1]**2, Da[2]**2, Da[3]**2, Da[4]**2, Da[0], Da[0])
    #print( torch.dist( Td, Td.permute(0,3,2,1,4,5) ) )    # test left-right reflection symmetry of Td

    CE = torch.tensordot(C,E,([1],[0]))         # C(1d)E(dga)->CE(1ga)
    EL = torch.tensordot(E,CE,([2],[0]))        # E(2e1)CE(1ga)->EL(2ega)  use E(2e1) == E(1e2) 
    EL = torch.tensordot(EL,Td,([1,2],[1,0]))   # EL(2ega)T(gehbmn)->EL(2ahbmn)
    EL = torch.tensordot(EL,CE,([0,2],[0,1]))   # EL(2ahbmn)CE(2hc)->EL(abmnc), use CE(2hc) == CE(1ga) 
    Rho = torch.tensordot(EL,EL,([0,1,4],[0,1,4])).permute(0,2,1,3).contiguous().view(Da[0]**2,Da[0]**2)
    
    # print( (Rho-Rho.t()).norm() )
    Rho = 0.5*(Rho + Rho.t())
    
    Tnorm = Rho.trace()
    Energy = torch.mm(Rho,H).trace()/Tnorm
    Mx = torch.mm(Rho,Sx).trace()/Tnorm
    My = torch.mm(Rho,Sy).trace()/Tnorm
    Mz = torch.mm(Rho,Sz).trace()/Tnorm
   
    #print("Tnorm = %g, Energy = %g " % (Tnorm.item(), Energy.item()) )

    return Energy, Mx, My, Mz

