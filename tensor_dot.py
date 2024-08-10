import numpy as np

x, y, z = 2, 2, 3

A = np.random.rand(x,y,z)
B = np.random.rand(x,y)

print(f"Matrix A: {A}")
print(f"Matrix B: {B}")

B_reshaped = B[:, :, np.newaxis]

C = np.einsum('ijk,ij->ij', A, B)

print(f"Matrix C: {C}")

