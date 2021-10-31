import numpy as np

a = np.array([-1, -2, 3])
print(np.where(a >= 0, a, 0))
print('--------------------------------')
