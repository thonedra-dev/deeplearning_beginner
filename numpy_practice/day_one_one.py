import numpy as np

arr = np.array([1, 2, 3, 4, 5])
lis = [ 1, 2, 3, 4, 5]


print(arr*2)
print(lis[1] * 2)    #Because, list data type cannot be done all at once as array. It need to explicitly selected or, we have to do it as a loop.

#As, you know, the machine learning need a lot of cols and rows to do maths at the same time, vectorization, we deeply need array.