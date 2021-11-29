import numpy as np

a = np.array([[1, 2, 3], [1, 4, 5], [1, 6, 7]])
b = np.array([[1], [2], [3]])

sum = np.concatenate((a, b), axis=1)

print(sum)
np.random.shuffle(sum)
print(sum)
