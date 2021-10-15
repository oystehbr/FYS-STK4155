import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad as egrad

def c(x, y, z):
    wow = x[0]**2 + x[1]
    return wow**2 + y + z

cdef = egrad(c)

print(cdef([2.0,1.0], 2, 3))

#