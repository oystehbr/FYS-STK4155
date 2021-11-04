import numpy as np

y = np.array([[1], [2]])
y_hat = np.array([[3], [5]])
m = y.shape[0]
[a], [b] = list(zip(y, y_hat))[0]
print(a)
print(m)