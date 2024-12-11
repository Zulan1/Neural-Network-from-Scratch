import numpy as np

size = 2

X = np.random.randint(0, 10,(size, size, size))
input_len = size
output_len = 1
W = np.random.randint(0, 10,(input_len, output_len))

out = np.dot(X, W)
print(f"{X.shape=}, {W.shape=}, {X=}, {W=}")
print(f"{out.shape=}, {out=}")