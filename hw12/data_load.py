import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

train_path = '../digits_data/ZipDigits.train'
test_path = '../digits_data/ZipDigits.test'

def load_data(data_path):
    with open(data_path, 'r') as f:
        def line_to_entry(l):
            d = [np.float128(e) for e in l.strip().split(' ')]
            m = np.zeros([16, 16], dtype=np.float128)
            for idx, val in enumerate(d[1:]):
                m[idx//16][idx%16] = val
            return (int(d[0]), m)
        data = [line_to_entry(l) for l in f]
        return data

def avg_intensity(number):
    return sum(sum(r for r in l if r >= 0) for l in number[1]) / 256

def symmetry(number):
    ret = 0.0
    for i in range(16):
        for j in range(16):
            ret += np.abs(number[1][i][j] - number[1][i][15-j]) + \
                np.abs(number[1][i][j] - number[1][15-i][j])
    return -1 * ret/1024

def compute_features(data):
    D = []
    for num in data:
        Xi = np.array([1, avg_intensity(num), symmetry(num)], dtype=np.float128)
        yi = np.array([1 if num[0] == 1 else -1], dtype=np.float128)
        D.append((Xi, yi))
    X, Y = np.stack([d[0] for d in D]), np.stack([d[1] for d in D])
    return (X, Y)

def rand_split(X, Y, K):
    """
    return X_train, Y_train, X_test, Y_test
    """
    D = np.append(X, Y, axis=1)
    np.random.shuffle(D)
    return D[:K, :-1], D[:K, -1:], D[K:, :-1], D[K:, -1:]

def write_to_file(X, Y, f_train, f_test, K=300):
    X_train, Y_train, X_test, Y_test = rand_split(X, Y, K)
    D_train = np.append(X_train, Y_train, axis=1)
    np.savetxt(f_train, D_train, delimiter=',')
    D_test = np.append(X_test, Y_test, axis=1)
    np.savetxt(f_test, D_test, delimiter=',')


if __name__ == '__main__':
    write_to_file(*compute_features(load_data(test_path) + load_data(train_path)), 'f_train.txt', 'f_test.txt')