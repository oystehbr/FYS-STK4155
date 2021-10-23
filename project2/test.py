import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad as egrad

def c(x, y, z):
    wow = np.mean(x[0].T @ x[1])
    return wow**2 + y + z

cdef = egrad(c)

print(cdef(np.array([[2.0,21.0], [3.0,1.0]]), 1123, 124))
