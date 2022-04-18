import numpy as np
import matplotlib.pyplot as plt

def scatter_dataset(X, Y):
    Yr = np.reshape(Y, [len(Y)])
    Xno = X[np.nonzero(Yr == -1)]
    Xyes = X[np.nonzero(Yr == 1)]
    plt.scatter(Xyes[:, 0], Xyes[:, 1], marker='.', c='blue', s=3)
    plt.scatter(Xno[:, 0], Xno[:, 1], marker='o', c='red', s=3)

data = np.array([[1, 0, 1],[-1,0,-1]])
sep_lin = np.genfromtxt('solutions/q3_4/boundary_a.txt', delimiter=',')
sep_tr  = np.genfromtxt('solutions/q3_4/boundary_b.txt', delimiter=',')
print(sep_lin)
plt.figure(figsize=(6,6))
scatter_dataset(data[:, :2], data[:, 2:])
plt.plot(sep_lin[:, 0], sep_lin[:, 1], c='black', label='linear')
plt.plot(sep_tr[:, 0], sep_tr[:, 1], c='purple', label='transformed')
plt.legend()
plt.savefig('q3.png')