import numpy as np

a = np.array([[1, 2, 3], [1, 4, 5], [1, 6, 7]])
b = np.array([[1, 2, 3], [1, 4, 5], [1, 6, 7]])
X = np.array([[2, 2, 2], [3, 3, 3]])

alt = a[:,1:]
noe = a[:,0]

sum = alt + noe[:, np.newaxis]

print(noe)
print(alt)
print(sum)

