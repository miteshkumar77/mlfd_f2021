import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as img
from functools import lru_cache
from math import pow

train_path = '../digits_data/ZipDigits.train'
test_path = '../digits_data/ZipDigits.test'

def load_data(data_path):
    with open(data_path, 'r') as f:
        def line_to_entry(l):
            d = [float(e) for e in l.strip().split(' ')]
            m = np.zeros([16, 16])
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
        Xi = np.array([1, avg_intensity(num), symmetry(num)])
        yi = 1 if num[0] == 1 else -1
        D.append((Xi, yi))
    return D

def normalize(D):
    mins = [x for x in D[0][0]]
    maxs = [x for x in D[0][0]]
    print(mins, maxs)
    for row, _ in D:
        for idx, val in enumerate(row):
            mins[idx] = min(mins[idx], val)
            maxs[idx] = max(maxs[idx], val)
    nval = lambda x, minv, maxv : 1 if maxv == minv else (((x - minv)/(maxv - minv)) * 2 - 1)
    nrow = lambda d : np.fromiter((nval(x, mins[idx], maxs[idx]) for idx, x in enumerate(d)), dtype=np.float16)
    return [(nrow(d[0]), d[1]) for d in D]


def rand_split(D, K):
    """
    return D_train, D_test
    """
    random.shuffle(D)
    return D[:K], D[K:]

@lru_cache(maxsize=None)
def legendre_poly(k):
    if k <= 1:
        return lambda x : pow(x, k)
    return lambda x : ((2 * k - 1)/k) * x * legendre_poly(k-1)(x) - ((k-1)/k) * legendre_poly(k-2)(x)

def transform(x, phi=None):
    if phi is None:
        return x
    return np.fromiter((phi_i(x) for phi_i in phi), dtype=np.float64)

def get_2D_legendre_transformer(Q):
    transformer = []
    def creator(p1, p2):
        return lambda x : legendre_poly(p1)(x[1]) * legendre_poly(p2)(x[2])
    for order in range(0, Q+1):
        for idx in range(order+1):
            transformer.append(creator(order - idx, idx))
    return transformer

def lin_reg(D):
    X = np.zeros([len(D), len(D[0][0])])
    Y = np.zeros([len(D), 1])
    i = 0
    for d in D:
        j = 0
        for q in d[0]:
            X[i][j] = q
            j += 1
        Y[i][0] = d[1]
        i += 1
    
    XT = np.transpose(X)   
    XTX = np.matmul(XT, X)
    XTXi = np.linalg.inv(XTX)
    Xp = np.matmul(XTXi, XT)
    return np.array(np.transpose(np.matmul(Xp, Y))[0])

def regularized_lin_reg(D, reg):
    Z, Y = split_Z_y(D)
    ZT = np.transpose(Z)
    ZTZ = np.matmul(ZT, Z)
    regI = reg * np.identity(ZTZ.shape[0])
    ZTY = np.matmul(ZT, Y)
    return np.array(np.transpose(np.matmul(np.linalg.inv(ZTZ + regI), ZTY)))




def show_transform_separator(w, transformer, N=500):
    x1 = np.linspace(-1, 1, N)
    x2 = np.linspace(-1, 1, N)
    xx1, xx2 = np.meshgrid(x1, x2)
    yy = np.zeros(xx1.shape)
    for i in range(N):
        for j in range(N):
            yy[i][j] = np.sign(np.dot(w, transform(np.array([1, xx1[i][j], xx2[i][j]]), transformer)))
    
    outx1 = []
    outx2 = []
    dx = [0, 0, 1, 1]
    dy = [0, 1, 0, 1]

    for i in range(N-1):
        for j in range(N-1):
            seen = set()
            for ip, jp in zip(dx, dy):
                ni = i + ip
                nj = j + jp
                if ni >= 0 and ni < N and nj >= 0 and nj < N:
                    seen.add(yy[ni][nj])
                    if len(seen) > 1:
                        break
            if len(seen) > 1:
                x1avg = (xx1[i][j] + xx1[i][j+1])/2
                x2avg = (xx2[i][j] + xx2[i+1][j])/2
                outx1.append(x1avg)
                outx2.append(x2avg)
    plt.scatter(outx1, outx2, color='black', label='separator', s=5)
    


def show_data_points(D):
    ones = [[d[0][1] for d in D if d[1] == 1],
            [d[0][2] for d in D if d[1] == 1]]
    others = [[d[0][1] for d in D if d[1] == -1],
             [d[0][2] for d in D if d[1] == -1]]

    plt.scatter(ones[0], ones[1], marker = 'o', color='blue', label='ones', s=5)
    plt.scatter(others[0], others[1], marker = 'x', color='red', label='fives', s=5)


def split_Z_y(d_train):
    Z = np.zeros([len(d_train), len(d_train[0][0])])
    Y = np.zeros([len(d_train), 1])
    i = 0
    for d in d_train:
        j = 0
        for q in d[0]:
            Z[i][j] = q
            j += 1
        Y[i][0] = d[1]
        i += 1
    return (Z, Y)

def calc_H_reg(Z, reg):
    ZT = np.transpose(Z)
    ZTZ = np.matmul(ZT, Z)
    RegI = reg * np.identity(ZTZ.shape[0])
    return np.matmul(Z, np.matmul(np.linalg.inv(ZTZ + RegI), ZT))
    

def reg_cv_error(transformed_d_train, reg):
    Z, Y = split_Z_y(transformed_d_train)
    H_reg = calc_H_reg(Z, reg)
    Y_pred = np.matmul(H_reg, Y)
    N = len(Y)
    return sum(pow((Y_pred[n] - Y[n])/(1 - H_reg[n][n]), 2) for n in range(N)) / N


def show_reg_cv_range(d_train, regs):
    reg_cv_errors = [reg_cv_error(d_train, reg) for reg in regs]
    min_idx = reg_cv_errors.index(min(reg_cv_errors))
    plt.plot(regs, reg_cv_errors, color='red', label='cross validation error')
    plt.legend()
    print(f'reg cv error range, min_error={reg_cv_errors[min_idx]}, reg={regs[min_idx]}')
    return regs[min_idx]


def classification_error(D, wt):
    N = len(D)
    component = lambda xi, yi: 0.0 if np.sign(np.dot(wt, xi)) == np.sign(yi) else 1.0
    return (1.0 / N) * sum(component(*d) for d in D)

def regression_error(D, wt):
    N = len(D)
    component = lambda xi, yi: pow(2, np.dot(wt, xi) - yi)
    return (1.0 / N) * sum(component(*d) for d in D)

def reg_test_error(d_train, d_test, reg):
    w_reg = regularized_lin_reg(d_train, reg)
    return regression_error(d_test, w_reg)

def show_test_range(d_train, d_test, regs):
    reg_test_errors = [reg_test_error(d_train, d_test, reg) for reg in regs]
    min_idx = reg_test_errors.index(min(reg_test_errors))
    plt.plot(regs, reg_test_errors, color='blue', label='test error')
    plt.legend()
    print(f'test error range, min_error={reg_test_errors[min_idx]}, reg={regs[min_idx]}')


if __name__ == '__main__':
    d_train, d_test = rand_split(normalize(compute_features(load_data(test_path) + load_data(train_path))), 300)
    print(f"|d_train| = {len(d_train)}, |d_test| = {len(d_test)}")

    transformer = get_2D_legendre_transformer(8)
    transformed_d_train = [(transform(d[0], transformer), d[1]) for d in d_train]
    transformed_d_test = [(transform(d[0], transformer), d[1]) for d in d_test]

    plt.figure(figsize=(10,10))
    regs = np.arange(start=0.01, stop=2, step=0.01)
    best_reg = show_reg_cv_range(transformed_d_train, regs)
    plt.savefig('4_1.png')
    plt.close()

    plt.figure(figsize=(10,10))
    show_test_range(transformed_d_train, transformed_d_test, regs)
    plt.savefig('4_2.png')
    plt.close()

    plt.figure(figsize=(10,10))
    best_reg = show_reg_cv_range(transformed_d_train, regs)
    show_test_range(transformed_d_train, transformed_d_test, regs)
    plt.savefig('4.png')
    plt.close()
    

    w_no_reg = lin_reg(transformed_d_train)
    w_reg_2 = regularized_lin_reg(transformed_d_train, reg=2.0)
    w_best_reg = regularized_lin_reg(transformed_d_train, reg=best_reg)

    plt.figure(figsize=(10,10))
    show_data_points(d_test)
    show_transform_separator(w_best_reg, transformer)
    plt.title(f'reg={best_reg}, classification error={classification_error(transformed_d_test , w_best_reg)}')
    plt.savefig('5.png')

    plt.figure(figsize=(10,10))
    show_data_points(d_test)
    show_transform_separator(w_no_reg, transformer)
    plt.title(f'classification error = {classification_error(transformed_d_test, w_no_reg)}')
    plt.savefig('2.png')
    plt.close()
    
    plt.figure(figsize=(10,10))
    show_data_points(d_test)
    show_transform_separator(w_reg_2, transformer)
    plt.title(f'classification error = {classification_error(transformed_d_test, w_reg_2)}')
    plt.savefig('3.png')
    plt.close()