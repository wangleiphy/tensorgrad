import numpy as np
import torch
import sys, os
testdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(testdir+"/..")

from adlib.svd import SVD

def test_svd():
    M, N = 20, 16
    torch.manual_seed(2)
    input = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    assert(torch.autograd.gradcheck(SVD.apply, input, eps=1e-6, atol=1e-4))
