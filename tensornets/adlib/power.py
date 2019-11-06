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

