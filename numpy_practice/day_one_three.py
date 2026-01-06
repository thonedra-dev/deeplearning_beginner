import numpy as np

arr1 = np.array([10,20,30,40,50,60,70,80])
arr2 = np.array([[1,2,3],
                 [4,5,6],
                 [7,8,9]])

arr3 = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120]
])


# 1D slicing

print(arr1[1:7]) 
print()

# 2D slicing
print(arr2[0:, 0:3])   # The overall matric

print(arr3[0, 0:])     # The first row only

print(arr3[-1, 0:])    # The last row only

print(arr3[0: ,1])     # The second column only

print(arr3[0:2, 2:4])  # First two rows, last two cols

print(arr3[1:3, 1:3])  

print(arr3[::-1])



