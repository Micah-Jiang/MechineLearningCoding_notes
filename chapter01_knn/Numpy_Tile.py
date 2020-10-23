import numpy as np

a = np.array([1, 2, 3, 4])

x1 = np.tile(a, 3)

x2 = np.tile(a, (2, 1))

print(x1)

print("分割线-------------------\n")

print(x2)