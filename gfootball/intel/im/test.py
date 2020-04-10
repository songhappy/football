import numpy as np

x = [[1,2,3],[4,5,6]]
y = [[0,0,0]]
arrx = np.array(x, dtype=float)
arry = np.array(y, dtype=float)
arr = np.concatenate([arrx, arry])
print(arr)