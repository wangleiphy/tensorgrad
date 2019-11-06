'''
PyTorch has its own implementation of backward function for symmetric eigensolver https://github.com/pytorch/pytorch/blob/291746f11047361100102577ce7d1cfa1833be50/tools/autograd/templates/Functions.cpp#L1660
However, it assumes a triangular adjoint.  
We reimplement it to return a symmetric adjoint
'''

import numpy as np 
import torch

class EigenSolver(torch.autograd.Function):
    @staticmethod
    def forward(self, A):
        w, v = torch.symeig(A, eigenvectors=True)

        self.save_for_backward(w, v)
        return w, v

    @staticmethod
    def backward(self, dw, dv):
        w, v = self.saved_tensors
        dtype, device = w.dtype, w.device
        N = v.shape[0]

        F = w - w[:,None]
        F.diagonal().fill_(np.inf)
        # safe inverse
        msk = (torch.abs(F) < 1e-20)
        F[msk] += 1e-20
        F = 1./F  

        vt = v.t()
        vdv = vt@dv

        return v@(torch.diag(dw) + F*(vdv-vdv.t())/2) @vt

