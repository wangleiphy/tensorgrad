import numpy as np
import torch
import sys, os
testdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(testdir+"/..")

from adlib.eigh import EigenSolver

def test_eigs():
    M = 2
    torch.manual_seed(42)
    A = torch.rand(M, M, dtype=torch.float64)
    A = torch.nn.Parameter(A+A.t())
    assert(torch.autograd.gradcheck(EigenSolver.apply, A, eps=1e-6, atol=1e-4))
