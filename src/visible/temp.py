import numpy as np

from src.ik.utils import vector_norm

a = np.array([0.2629, 0.4576, 0.1724]) - np.array([0.0523, 0.6130, 0.1966])
print(np.linalg.norm(a))
b = vector_norm(a)
print(b)
a = np.array([0.2629, 0.4576, 0.1724]) - np.array([0.3559, 0.2392, 0.2515])
print(np.linalg.norm(a))
b = vector_norm(a)
print(b)

