import numpy as np
import matplotlib.pyplot as plt

def scatter_dataset(X, Y):
    Yr = np.reshape(Y, [len(Y)])
    Xno = X[np.nonzero(Yr == -1)]
    Xyes = X[np.nonzero(Yr == 1)]
    plt.scatter(Xyes[:, 1], Xyes[:, 2], marker='.', c='blue', s=3)
    plt.scatter(Xno[:, 1], Xno[:, 2], marker='o', c='red', s=3)

data = np.genfromtxt('f_test.txt', delimiter=',')

sep = np.genfromtxt('solutions/q1_2/q2_a_sep.txt', delimiter=',')
ein = np.genfromtxt('solutions/q1_2/q2_a.txt', delimiter=',')

plt.figure(figsize=(10,10))
scatter_dataset(data[:, 0:3], data[:, 3:4])
plt.title(f'Classification Ein={0.01}, Eout={0.02434}')
plt.scatter(sep[:, 0], sep[:, 1], marker='.', c='black', s=3)
plt.savefig('plots/q2_a_sep.png')
plt.close()

plt.figure(figsize=(10,10))
plt.yscale('log')
plt.xscale('log')
plt.plot(ein)
plt.savefig('plots/q2_a_ein.png')
plt.close()


sep = np.genfromtxt('solutions/q1_2/q2_b_sep.txt', delimiter=',')
ein = np.genfromtxt('solutions/q1_2/q2_b.txt', delimiter=',')

plt.figure(figsize=(10,10))
scatter_dataset(data[:, 0:3], data[:, 3:4])
plt.scatter(sep[:, 0], sep[:, 1], marker='.', c='black', s=3)
plt.title(f'Classification Ein={0.01}, Eout={0.0220049}')
plt.savefig('plots/q2_b_sep.png')
plt.close()

plt.figure(figsize=(10,10))
plt.yscale('log')
plt.xscale('log')
plt.plot(ein)
plt.savefig('plots/q2_b_ein_aug.png')
plt.close()


sep = np.genfromtxt('solutions/q1_2/q2_c_sep.txt', delimiter=',')
ein = np.genfromtxt('solutions/q1_2/q2_c_ein.txt', delimiter=',')
eval = np.genfromtxt('solutions/q1_2/q2_c_eval.txt', delimiter=',')

plt.figure(figsize=(10,10))
scatter_dataset(data[:, 0:3], data[:, 3:4])
plt.scatter(sep[:, 0], sep[:, 1], marker='.', c='black', s=3)
plt.title(f'Classification Ein={0.01333}, Eout={0.0227}')
plt.savefig('plots/q2_c_sep.png')
plt.close()

plt.figure(figsize=(10,10))
plt.yscale('log')
plt.xscale('log')
plt.plot(ein, c='red', label='E_in')
plt.plot(eval, c='blue', label='E*')
plt.legend()
plt.savefig('plots/q2_c_ein_eval.png')
plt.close()


sep_lin = np.genfromtxt('solutions/q3_4/boundary_a.txt')
sep_tr  = np.genfromtxt('solutions/q3_4/boundary_b.txt')


