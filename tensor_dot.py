import numpy as np

x, y, z = 2, 2, 3

A = np.matrix('1 2; 3 4')
B = np.matrix('1 2')

print(f"Matrix A: {A}")
print(f"Matrix B: {B}")

# B_reshaped = B[:, :, np.newaxis]

# C = np.einsum('ijk,ij->ij', A, B)
C = A @ B.T

print(f"Matrix C: {C}")

