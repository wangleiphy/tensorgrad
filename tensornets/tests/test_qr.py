import numpy as np
import torch
import sys, os
testdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(testdir+"/..")

from adlib.qr import QR

def test_qr():
    M, N = 4, 6
    torch.manual_seed(2)
    A = torch.randn(M, N, dtype=torch.float64)
    A.requires_grad=True
    assert(torch.autograd.gradcheck(QR.apply, A, eps=1e-4, atol=1e-2))
