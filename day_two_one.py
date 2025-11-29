import numpy as np

print("----- 1D Array -----")
arr1 = np.arange(1,6)
print("arange():", arr1, "Shape:", arr1.shape)

arr2 = np.array([1,2,3,4,5])
print("array():", arr2, "Shape:", arr2.shape)
print()




print("----- 2D Array -----")
arr1_2d = np.arange(1,10).reshape(3,3)
print("arange():\n", arr1_2d, "Shape:", arr1_2d.shape)

arr2_2d = np.array([ [1,2,3],[4,5,6],[7,8,9] ])
print("arange():\n", arr2_2d, "Shape:", arr2_2d.shape)
print()



print("\n----- 3D Array -----")
arr1_3d = np.arange(1,13).reshape(2,2,3)
print("arange():\n", arr1_3d, "Shape:", arr1_3d.shape)
print()
arr2_3d = np.array([
    [
        [1,2,3],
        [4,5,6]
    ],
    [
        [7,8,9],
        [10,11,12]
    ]
])
print("arange():\n\n", arr2_3d, "Shape:", arr2_3d.shape)



