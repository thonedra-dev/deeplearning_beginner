import numpy as np
# 3D array slicing
# 3D array: 2 “sheets”, 3 rows, 4 columns
arr = np.array([
    [[1,2,3,4], [5,6,7,8], [9,10,11,12]],
    [[13,14,15,16], [17,18,19,20], [21,22,23,24]]
])

# 1️⃣ Print the entire first sheet
print("First sheet:\n", arr[0,0:])  #In here, you can just use [0].

# 2️⃣ Print the second row of the second sheet
print("Second row of second sheet:\n", arr[1,1,0:])

# 3️⃣ Print the last column of all sheets
print("Last column of all sheets:\n", arr[0:,0:,-1])

# 4️⃣ Print rows 1 and 3 of first sheet, columns 1 and 3
print("Rows 0 & 2, Columns 1 & 3 of first sheet:\n", arr[0,[0,2]][:,[1,3]])