

import numpy as np


a = np.array([[1, 3], [7, 4], [2, 4]])

print(a.shape)
print(a)

print(a.max(axis=0))
print(a/a.max(axis=0))
