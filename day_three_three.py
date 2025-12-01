import numpy as np


# 1. INPUT MATRIX (4 samples, 3 features)
X = np.array([
    [1.0, 2.0, 3.0],   # sample 1
    [4.0, 5.0, 6.0],   # sample 2
    [7.0, 8.0, 9.0],   # sample 3
    [1.5, 2.5, 3.5]    # sample 4
]).T   # shape becomes (3,4)

print("X shape:", X.shape)
print("X:\n", X, "\n")


# 2. WEIGHTS (5 neurons, 3 inputs)
W = np.array([
    [0.1, -0.2,  0.3],
    [0.4,  0.5, -0.6],
    [-0.7, 0.8, 0.9],
    [1.0, -1.1, 1.2],
    [-0.3, 0.2, 0.1]
])


# 3. BIAS (5 neurons, 1 each)
b = np.array([[1], [2], [3], [4], [5]])


# 4. Z = np.dot(W, X) + b
Z = np.dot(W, X) + b   # shape = (5,4)
print("Z = np.dot(W, X) + b:\n", Z, "\n")

# -----------------------
# 5. ACTIVATION FUNCTIONS
# -----------------------

# ReLU
relu = np.maximum(0, Z)
print("ReLU(Z):\n", relu, "\n")

# Sigmoid
sigmoid = 1 / (1 + np.exp(-Z))
print("Sigmoid(Z):\n", sigmoid, "\n")

# Softmax (per sample → axis=0)
exp_vals = np.exp(Z)
softmax = exp_vals / np.sum(exp_vals, axis=0, keepdims=True)
print("Softmax(Z):\n", softmax)
