import numpy as np
arr = np.array([
    [[1, 2, 3],
     [7, 5, 6]],
    
    [[7, 10, 9],
     [10, 11, 12]]
])

print("Original Array:\n", arr, "\n")

print("Sum of all elements in the array : ", np.sum(arr))
print("Largest of all elements in the array : ", np.max(arr))
print("The average value within the array: ", np.mean(arr))
print()
print()

print(np.sum(arr, axis=1)) # This will add the rows in a single sheet.
print(np.sum(arr, axis=2)) # This will add the cols in a single row.
print(np.sum(arr, axis=0)) # This will add every rows and cols from the different sheets into a single sheet.
# The concept behind is that, 
# sheet => row => col,
# So, rows in a sheet
# So, cols in a row
# So, rows and cols of a sheet.

print(np.max(arr, axis=1)) # This will give the maximum row from each sheet.
print(np.max(arr, axis=2)) # This will give the maximum col values frome each row from each sheet.
print(np.max(arr, axis=0)) # This will simply give the maximum sheet with largest values.