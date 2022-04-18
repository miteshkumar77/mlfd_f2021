import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import centering

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
    for row, _ in D:
        for idx, val in enumerate(row):
            mins[idx] = min(mins[idx], val)
            maxs[idx] = max(maxs[idx], val)
    nval = lambda x, minv, maxv : 1 if maxv == minv else (((x - minv)/(maxv - minv)) * 2 - 1)
    nrow = lambda d : np.fromiter((nval(x, mins[idx], maxs[idx]) for idx, x in enumerate(d)), dtype=np.float64)
    X = np.stack([nrow(d[0]) for d in D])
    Y = np.reshape(np.fromiter((d[1] for d in D), dtype=np.float64), [len(X), 1])
    return X, Y


def rand_split(X, Y, K):
    """
    return X_train, Y_train, X_test, Y_test
    """
    D = np.append(X, Y, axis=1)
    np.random.shuffle(D)
    return D[:K, :-1], D[:K, -1:], D[K:, :-1], D[K:, -1:]

def knn(X, Y, t, k, sgn=True):
    t = np.mean(Y[np.argpartition(np.linalg.norm(X-t,axis=1), k)[:k+1]])
    if sgn:
        return np.sign(t)
    return t

def leave_one_out(X, Y, k):
    D = np.append(X, Y, axis=1)
    return np.append(D[:k, :-1], D[k+1:, :-1], axis=0), np.append(D[:k, -1:], D[k+1:, -1:], axis=0)

def knn_ecv_leave_one_out(X, Y, k):
    gen = (((knn(*leave_one_out(X, Y, i), X[i], k, sgn=False) - Y[i])**2) for i in range(len(X)))
    return np.mean(np.fromiter(gen, dtype=np.float64))

def scatter_dataset(X, Y):
    Yr = np.reshape(Y, [len(Y)])
    Xno = X[np.nonzero(Yr == -1)]
    Xyes = X[np.nonzero(Yr == 1)]

    plt.scatter(Xyes[:, :1], Xyes[:, -1:], marker='.', c='blue', s=3)
    plt.scatter(Xno[:, :1], Xno[:, -1:], marker='o', c='red', s=3)

def show_knn_plot(X, Y, stop):
    Ks = np.arange(start=1, stop=stop, step=1, dtype=int)
    Ecvs = np.fromiter((knn_ecv_leave_one_out(X, Y, k) for k in Ks), dtype=np.float64)
    best = np.argmin(Ecvs)
    bestK = Ks[best]
    bestEcv = Ecvs[best]
    plt.title(f'bestK: {bestK}, bestEcv: {bestEcv}')
    plt.xlabel('k')
    plt.ylabel('Ecv')
    plt.plot(Ks, Ecvs)
    return bestK, bestEcv



def show_2D_decision_boundary(X, Y, classify, res, transformer=None):
    x1_lo = np.min(X[:, :1]) - 0.3
    x1_hi = np.max(X[:, :1]) + 0.3 
    x2_lo = np.min(X[:, -1:]) - 0.3
    x2_hi = np.max(X[:, -1:]) + 0.3

    x1, x2 = np.linspace(x1_lo, x1_hi, res), np.linspace(x2_lo, x2_hi, res)
    xx1, xx2 = np.meshgrid(x1, x2)

    if transformer is None:
        z = np.fromiter((classify(x, sgn=True) for x in np.c_[xx1.ravel(), xx2.ravel()]), dtype=np.float64)
    else:
        z = np.fromiter((classify(x, sgn=True) for x in transformer(np.c_[xx1.ravel(), xx2.ravel()])), dtype=np.float64)

    zz = np.reshape(z, xx1.shape)
    cmap_light = ListedColormap(["orange", "cyan"])

    plt.contourf(xx1, xx2, zz, cmap=cmap_light)
    scatter_dataset(X, Y)

def make_knn_classifier(X, Y, k):
    return lambda x , sgn=True: knn(X, Y, x, k, sgn)

def make_rbf_ntwk_classifier(w):
    return lambda x, sgn=True : np.sign(np.dot(np.transpose(w), x)) if sgn else np.dot(np.transpose(w), x)

def classification_error(X, Y, classifier):
    return np.mean(np.power(np.fromiter((classifier(x, False) for x in X), dtype=np.float64) - Y, 2))

def rbf_centers(X, k, iters=1):
    P = centering.partition(X, k)
    for _ in range(1, iters):
        P = centering.improve_partition(P)
    return np.stack([p[0] for p in P])


def gk(z):
    return np.power(np.e, np.power(z, 2)/-2)

def rbf_transform(X, r, centers):
    return np.column_stack([np.ones([len(X)])] +
        [gk(np.linalg.norm(X - c, axis=1)/r) for c in centers])

def lin_reg(X, Y):
    XT = np.transpose(X)
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(XT, X)), XT), Y)

def show_rbf_ntwk_plot(X, Y, stop):
    Ks = np.arange(start=1, stop=stop, step=1)

def rbf_ecv_leave_one_out(X, Y, k):
    Z = rbf_transform(X, 2/np.sqrt(k), rbf_centers(X, k, 10))
    def gen():
        for i in range(len(Y)):
            Xi, Yi = leave_one_out(X, Y, i)
            centers_i = rbf_centers(Xi, k, 10)
            Zi = rbf_transform(Xi, 2/np.sqrt(k), centers_i)
            classifier = make_rbf_ntwk_classifier(lin_reg(Zi, Yi))
            yield np.power((classifier(Z[i], sgn=False)-Y[i]), 2)
    # gen = ((make_rbf_ntwk_classifier(lin_reg(*leave_one_out(Z, Y, i)))(Z[i], sgn=False) - Y[i]) ** 2 for i in range(len(Y)))
    return np.mean(np.fromiter((i for i in gen()), dtype=np.float64))

def show_rbf_plot(X, Y, stop):
    Ks = np.arange(start=1, stop=stop, step=1, dtype=int)
    Ecvs = np.fromiter((rbf_ecv_leave_one_out(X, Y, k) for k in Ks), dtype=np.float64)
    print(Ecvs)
    best = np.argmin(Ecvs)
    bestK = Ks[best]
    bestEcv = Ecvs[best]
    plt.title(f'bestK: {bestK}, bestEcv: {bestEcv}')
    plt.xlabel('k')
    plt.ylabel('Ecv')
    plt.plot(Ks, Ecvs)
    return bestK, bestEcv

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = rand_split(*normalize(compute_features(load_data(test_path) + load_data(train_path))), 300)
    X_train = X_train[:, 1:]
    X_test = X_test[:, 1:]
    print(f"|d_train| = {len(Y_train)}, |d_test| = {len(Y_test)}")

    plt.figure(figsize=(8,8))
    bestK, bestEcv = show_knn_plot(X_train, Y_train, 100)
    print(f'bestK: {bestK}, bestEcv: {bestEcv}')
    plt.savefig('1a.png')
    plt.close()

    bestKClassifier = make_knn_classifier(X_train, Y_train, bestK)
    plt.figure(figsize=(8,8))
    show_2D_decision_boundary(X_test, Y_test, bestKClassifier, 500)
    plt.title(f'{bestK}-nn decision boundary, Ecv: {bestEcv}, '
        f'Ein: {classification_error(X_train, Y_train, bestKClassifier)} '
        f'Etest: {classification_error(X_test, Y_test, bestKClassifier)}', loc='center', wrap=True)
    plt.savefig(f'1b.png')
    plt.close()

    # centers = [rbf_centers(X_train, k, 10) for k in range(1, 30)]
    # Z_trains = [rbf_transform(X_train, 2/np.sqrt(len(c)), c) for c in centers]

    plt.figure(figsize=(8,8))
    bestK, bestEcv = show_rbf_plot(X_train, Y_train, 30)
    print(f'bestK: {bestK}, bestEcv: {bestEcv}')
    plt.savefig('2a.png')
    plt.close()

    bestCenters = rbf_centers(X_train, bestK, 10)
    bestZ_train = rbf_transform(X_train, 2/np.sqrt(bestK), bestCenters)
    bestRbfW = lin_reg(bestZ_train, Y_train)
    bestKClassifier = make_rbf_ntwk_classifier(bestRbfW)
    bestR = 2/np.sqrt(bestK)
    bestKTransformer = lambda D : rbf_transform(D, bestR, bestCenters)
    plt.figure(figsize=(8,8))
    show_2D_decision_boundary(X_test, Y_test, bestKClassifier, 500, transformer=bestKTransformer)
    plt.title(f'{bestK}-rbf-ntwk decision boundary, Ecv: {bestEcv}, '
        f'Ein: {classification_error(bestZ_train, Y_train, bestKClassifier)} '
        f'Etest: {classification_error(bestKTransformer(X_test), Y_test, bestKClassifier)}', loc='center', wrap=True)
    plt.savefig(f'2b.png')
    plt.close()

    