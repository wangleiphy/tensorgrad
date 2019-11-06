import numpy as np
import torch
import sys, os
testdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(testdir+"/..")

from adlib.power import FixedPoint

def test_backward():
    N = 4 
    torch.manual_seed(2)
    A = torch.rand(N, N, dtype=torch.float64, requires_grad=True)
    x0 = torch.rand(N, dtype=torch.float64)
    x0 = x0/x0.norm()
    tol = 1E-10
    
    input = A, x0, tol
    assert(torch.autograd.gradcheck(FixedPoint.apply, input, eps=1E-6, atol=tol))

def test_forward():
    torch.manual_seed(42)
    N = 100
    tol = 1E-8
    dtype = torch.float64
    A = torch.randn(N, N, dtype=dtype)
    A = A+A.t()

    w, v = torch.symeig(A, eigenvectors=True)
    idx = torch.argmax(w.abs())

    v_exact = v[:, idx] 
    v_exact = v_exact[0].sign() * v_exact
    
    x0 = torch.rand(N, dtype=dtype)
    x0 = x0/x0.norm()
    x = FixedPoint.apply(A, x0, tol)

    assert(torch.allclose(v_exact, x, rtol=tol, atol=tol))
