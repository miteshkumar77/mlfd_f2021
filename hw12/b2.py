import numpy as np
import q1
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
        yi = np.array([1 if num[0] == 1 else -1])
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

def scatter_dataset(X, Y):
    Yr = np.reshape(Y, [len(Y)])
    Xno = X[np.nonzero(Yr == -1)]
    Xyes = X[np.nonzero(Yr == 1)]
    plt.scatter(Xyes[:, 1], Xyes[:, 2], marker='.', c='blue', s=3)
    plt.scatter(Xno[:, 1], Xno[:, 2], marker='o', c='red', s=3)

def show_2D_decision_boundary(X, Y, classify, res, transformer=None):
    x1_lo = np.min(X[:, :1]) - 0.3
    x1_hi = np.max(X[:, :1]) + 0.3 
    x2_lo = np.min(X[:, -1:]) - 0.3
    x2_hi = np.max(X[:, -1:]) + 0.3

    x1, x2 = np.linspace(x1_lo, x1_hi, res), np.linspace(x2_lo, x2_hi, res)
    xx1, xx2 = np.meshgrid(x1, x2)

    if transformer is None:
        z = np.fromiter((classify(q1.augment_1(np.reshape(x, [2, 1]))) for x in np.c_[xx1.ravel(), xx2.ravel()]), dtype=np.float128)
    else:
        z = np.fromiter((classify(q1.augment_1(np.reshape(x, [2, 1]))) for x in transformer(np.c_[xx1.ravel(), xx2.ravel()])), dtype=np.float128)

    zz = np.reshape(z, xx1.shape)
    cmap_light = ListedColormap(["orange", "cyan"])

    plt.contourf(xx1, xx2, zz, cmap=cmap_light)
    scatter_dataset(X, Y)

def compute_scale(X, mult):
    return mult * np.max(np.power(np.linalg.norm(X, axis=1), 2))

def classification_error(X, Y, classifier):
    return np.mean(np.power(np.fromiter((classifier(x, False) for x in X), dtype=np.float128) - Y, 2))

def lin_reg(X, Y):
    XT = np.transpose(X)
    wlin = np.matmul(np.matmul(np.linalg.inv(np.matmul(XT, X)), XT), Y)
    return wlin

def variable_learning_rate_gd_nn(W, T, D, outfn, n0, alpha, beta, iterations):
    Wt = np.copy(W)
    Eins = np.zeros([iterations+1])
    Eins[0], G = q1.error_gradient_nn(Wt, T, D, outfn)
    for t in range(1, iterations+1):
        Wtt = [None] + [wt - n0 * g for wt, g in zip(Wt[1:], G[1:])]       
        Eint = q1.error(Wtt, T, D)
        if Eint <= Eins[t-1]:
            Eins[t] = Eint
            Wt = Wtt
            _, G = q1.error_gradient_nn(Wt, T, D, outfn)
            n0 *= alpha
        else:
            Eins[t] = Eins[t-1]
            n0 *= beta
    return (Wt, Eins)

def plot_ein_iters(Eins):
    x = np.linspace(1, len(Eins)+1, len(Eins))
    plt.plot(x, Eins, marker='.', color='green')

def make_nn_classifier(W, T):
    def classify(x):
        _, X = q1.fprop_nn(W, x, T)
        return np.sign(X[-1][1])
    return classify

if __name__ == '__main__':

    # D = q1.augment_data(np.array([[1, 2, 1]]))
    X_train, Y_train, X_test, Y_test = rand_split(*compute_features(load_data(test_path) + load_data(train_path)), 300)
    D_train = np.append(X_train, Y_train, axis=1)
    D_test = np.append(X_test, Y_test, axis=1)
    W = q1.create_nn([2, 10, 1], scale=compute_scale(X_train, 0.00001))
    print(W)
    T = q1.transitions_nn(len(W), 'identity')
    Wlearned, Eins = variable_learning_rate_gd_nn(W, T, D_train, 'identity', 0.00003, 1.01, 0.99, 10000)

    plt.figure(figsize=[10,10])
    plot_ein_iters(Eins[1:])
    plt.savefig('2a_1.png')
    plt.close()

    plt.figure(figsize=[10,10])
    show_2D_decision_boundary(X_test, Y_test, make_nn_classifier(Wlearned, T), 500)
    plt.savefig('2a_2.png')
    plt.close()

    # Wlearned = variable_learning_rate_gd_nn(W, T, D, 'identity', 0.001, 1.05, 0.5, 100)
    # print(W)
    # print(Wlearned)























