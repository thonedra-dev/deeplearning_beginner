import numpy as np

a = np.array([1,2,3,4])

a_row = a.reshape(1,4)  # 1 row, 4 columns
a_col = a.reshape(4,1)  # 4 rows, 1 column

print(a_row)
print(a_col)