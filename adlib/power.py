import torch 
from torch.utils.checkpoint import detach_variable

def step(A, x):
    y = A@x 
    y = y[0].sign() * y 
    return y/y.norm() 

class FixedPoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, x0, tol):
        x, x_prev = step(A, x0), x0
        while torch.dist(x, x_prev) > tol:
            x, x_prev = step(A, x), x
        ctx.save_for_backward(A, x)
        ctx.tol = tol
        return x 

    @staticmethod
    def backward(ctx, grad):
        A, x = detach_variable(ctx.saved_tensors)
        dA = grad
        while True:
            with torch.enable_grad():
                grad = torch.autograd.grad(step(A, x), x, grad_outputs=grad)[0]
            if (torch.norm(grad) > ctx.tol):
                dA = dA + grad
            else:
                break
        with torch.enable_grad():
            dA = torch.autograd.grad(step(A, x), A, grad_outputs=dA)[0]
        return dA, None, None

def test_backward():
    N = 4 
    torch.manual_seed(2)
    A = torch.rand(N, N, dtype=torch.float64, requires_grad=True)
    x0 = torch.rand(N, dtype=torch.float64)
    x0 = x0/x0.norm()
    tol = 1E-10
    
    input = A, x0, tol
    assert(torch.autograd.gradcheck(FixedPoint.apply, input, eps=1E-6, atol=tol))

    print("Backward Test Pass!")

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
    print("Forward Test Pass!")

if __name__=='__main__':
    test_forward()
    test_backward()
