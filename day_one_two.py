import numpy as np

X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

w = np.array([4, 2, 1])
b = 0.5

output = np.dot(X,w) + b
print(output)

"""
In this case, you cannot do the wX, because, the deep leaarning's neuron layers need row-wise multiplication to be done.
X = [[x11 x12 x13]
     [x21 x22 x23]
     [x31 x32 x33]]

w = [w1 w2 w3]

X·w = [x11*w1 + x12*w2 + x13*w3,
       x21*w1 + x22*w2 + x23*w3,
       x31*w1 + x32*w2 + x33*w3]  # row-wise

w·X = [w1*x11 + w2*x21 + w3*x31,
       w1*x12 + w2*x22 + w3*x32,
       w1*x13 + w2*x23 + w3*x33]  # column-wise

"""