import numpy as np

# 1️⃣ 1D array to 2D
arr1 = np.arange(1,13)
print("Q1 - (3,4):\n", arr1.reshape(3,4))
print("Q2 - (4,3):\n", arr1.reshape(4,3))
print("Q3 - Flattened:\n", arr1.reshape(3,4).flatten())

# 2️⃣ 2D array
arr2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("Q4 - Transposed:\n", arr2.T)
print("Q5 - Row vector (1,9):\n", arr2.reshape(1,9))
print("Q6 - Column vector (9,1):\n", arr2.reshape(9,1))
print()

# 3️⃣ 3D array
arr3 = np.arange(1,25).reshape(2,3,4)
print(arr3)
print()
print()

print("Q7 - Swap axes 0 & 1:\n", np.swapaxes(arr3,0,2)) #In here, 0 means sheet and 2 means col, so, we changed the format of sheet to column and column to sheet.
print("Q8 - Flattened 1D:\n", arr3.flatten())
