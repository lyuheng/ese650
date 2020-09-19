import torch
import numpy as np

class HW0Solution:
    def __init__(self):
        pass

    # Simulate the system Ax+b for one timestep using the given initial conditions
    # A = NxN numpy array
    # b = 1D N numpy array
    # init_cond = N x 1 x M numpy array (M different initial conditions)
    # return = N x 1 x M numpy array (M different initial conditions)
    def sim_systems(self, A, b, init_cond):
        init_cond = np.squeeze(init_cond, axis=1)  # N*M
        x = A.dot(init_cond)  # N*N  N*M = N*M
        x = x.T + b   # M*N + N = M*N
        return np.expand_dims(x.T, axis=1)

    # Compute the partial derivatives (gradients) of a multi-dimensional function.
    # dx = Scalar value showing distance between discrete samples
    # y = N-dimentional numpy array of sample points at regular intervals.
    #   -For the 2D case, y is N x N
    # returns = gradient (as torch tensor) at every point in y.
    #   -For the 2D case, return 2 x (N-1) x (N-1).  The first 2 channels are the x and y components
    def compute_derivative(self, dx, y):
        N = len(y.shape)
        out = None
        if N == 1:
            M = y.shape[0]
            out = np.zeros(M-1)
            out = (y[1:] - y[:-1])/dx 
            out = torch.Tensor(out).unsqueeze(0)  # (1, M-1)  (2,M-1,M-1) (3,M-1,M-1,M-1)
        else:
            M = y.shape[0]
            dim = [M-1 for _ in range(N)]
            dim.insert(0, N)
            out = np.zeros(dim)
            out = torch.Tensor(out)  
        return out
