import torch

class QR(torch.autograd.Function):
    @staticmethod
    def forward(self, A):
        Q, R = torch.qr(A)
        self.save_for_backward(A, Q, R)
        return Q, R

    @staticmethod
    def backward(self, dq, dr):
        A, q, r = self.saved_tensors
        if r.shape[0] == r.shape[1]:
            return _simple_qr_backward(q, r, dq ,dr)
        M, N = r.shape
        B = A[:,M:]
        dU = dr[:,:M]
        dD = dr[:,M:]
        U = r[:,:M]
        da = _simple_qr_backward(q, U, dq+B@dD.t(), dU)
        db = q@dD
        return torch.cat([da, db], 1)

def _simple_qr_backward(q, r, dq, dr):
    if r.shape[-2] != r.shape[-1]:
        raise NotImplementedError("QrGrad not implemented when ncols > nrows "
                          "or full_matrices is true and ncols != nrows.")

    qdq = q.t() @ dq
    qdq_ = qdq - qdq.t()
    rdr = r @ dr.t()
    rdr_ = rdr - rdr.t()
    tril = torch.tril(qdq_ + rdr_)

    def _TriangularSolve(x, r):
        """Equiv to x @ torch.inverse(r).t() if r is upper-tri."""
        res = torch.triangular_solve(x.t(), r, upper=True, transpose=False)[0].t()
        return res

    grad_a = q @ (dr + _TriangularSolve(tril, r))
    grad_b = _TriangularSolve(dq - q @ qdq, r)
    return grad_a + grad_b

