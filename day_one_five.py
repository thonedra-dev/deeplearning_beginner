import numpy as np

# -----------------------------------------
# 1️⃣ Case 1: One row only in secondary array (cols match) → broadcasting works
# ✅ Broadcasting works here
A = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9],
    [10,11,12]
])

B = np.array([[100,200,300]])  # 1 row, 3 cols
print("Case 1 - Result:\n", A + B)


# -----------------------------------------
# 2️⃣ Case 2: One column only in secondary array (rows match) → broadcasting works
# ✅ Broadcasting works here
C = np.array([
    [1,2,3,4,5],
    [6,7,8,9,10],
    [11,12,13,14,15],
    [16,17,18,19,20]
])

D = np.array([
    [100],
    [200],
    [300],
    [400]
])  # 4 rows, 1 col

print("\nCase 2 - Result:\n", C + D)



# When two arrays have the same number of rows, and the smaller array has only 1 column, 
             # its column stretches across the larger array for mathematical operations.
# When two arrays have the same number of columns, and the smaller array has only 1 row, 
             # its row stretches down across the larger array for operations.
# Any other situation → broadcasting cannot happen.