import numpy as np


a = np.array(range(10))
a1 = a[0:2]
a2 = a[7:9]

print(a1)
print(a2)
testing = np.concatenate((a1, a2))
print(testing)
