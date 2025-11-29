import numpy as np

# 2D array: shape (2, 3)
A = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

# 1D array: shape (3,)
b = np.array([1, 2, 3])

print("A shape:", A.shape)
print("b shape:", b.shape)

# Broadcasting addition
C = A + b

print("\nResult of A + b:\n", C)
