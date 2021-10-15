import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad as egrad 

def f(x):
    return x[0]**2 + 3*x[1]**2
def df(x) :
    return np.cos(2*np.pi*x + x**2)*(2*np.pi + 2*x)

x = np.linspace(-1,1,101)

f_grad = egrad(f)

print(f_grad([1.0, 2.0]))

