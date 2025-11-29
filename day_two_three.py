import numpy as np

arr1 = np.arange(1,7).reshape(2,3)
print("Original Array: \n", arr1)

swap1_1 = np.swapaxes(arr1,0,1)
print("Swaped Array: \n", swap1_1)


swap1_2 = np.swapaxes(arr1,1,0)
print("Swaped Array: \n", swap1_2)


###################################
arr2 = np.array([   
                [10, 20],
                [30, 40],
                [50, 60]
                ])

print("Original Array 2: \n", arr2)

swap2_1 = np.swapaxes(arr2, 0, 1)
print("Swapped Array: \n", swap2_1)
swap2_2 = np.swapaxes(arr2, 1, 0)
print("Swapped Array: \n", swap2_2)