from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

def dist(x, xp):
    return np.linalg.norm(xp - x)

def knn(D, x, k):
    D.sort(key=lambda el : dist(x, el[0]))
    return np.sign(sum(d[1] for d in D[:k]))

def scatter_dataset(D):
    x1yes = np.fromiter((d[0][0] for d in D if d[1] > 0), dtype=np.float64)
    x2yes = np.fromiter((d[0][1] for d in D if d[1] > 0), dtype=np.float64)
    x1no  = np.fromiter((d[0][0] for d in D if d[1] < 0), dtype=np.float64)
    x2no  = np.fromiter((d[0][1] for d in D if d[1] < 0), dtype=np.float64)
    plt.scatter(x1yes, x2yes, marker='.', c='blue')
    plt.scatter(x1no, x2no, marker='x', c='red')

def show_2D_decision_boundary(D, classify, res):
    x1_lo = min(d[0][0] for d in D) - 3
    x1_hi = max(d[0][0] for d in D) + 3
    x2_lo = min(d[0][1] for d in D) - 3
    x2_hi = max(d[0][1] for d in D) + 3

    x1, x2 = np.linspace(x1_lo, x1_hi, res), np.linspace(x2_lo, x2_hi, res)
    xx1, xx2 = np.meshgrid(x1, x2)

    z = np.fromiter((classify(x) for x in np.c_[xx1.ravel(), xx2.ravel()]), dtype=np.float64)
    zz = np.reshape(z, xx1.shape)
    cmap_light = ListedColormap(["orange", "cyan"])

    plt.contourf(xx1, xx2, zz, cmap=cmap_light)
    scatter_dataset(D)

def transform(d, T):
    return np.fromiter((t(d) for t in T), dtype=np.float64)

def make_knn_classifier(D, k, transformer=None):
    if transformer == None:
        return (lambda x : knn(D, x, k))
    else:
        return (lambda x : knn(D, transform(x, transformer), k))

if __name__ == '__main__':
    D = [
        (np.array([1, 0]), -1),
        (np.array([0, 1]), -1),
        (np.array([0, -1]), -1),
        (np.array([-1, 0]), -1),
        (np.array([0, 2]), 1),
        (np.array([0, -2]), 1),
        (np.array([-2, 0]), 1)
    ]

    _transformer = [
        (lambda x : np.linalg.norm(x)),
        (lambda x : (np.sign(x[1]) * np.pi/2 if x[0] == 0.0 else np.arctan(x[1]/x[0])))
    ]
    Dt = [(transform(d[0], _transformer), d[1]) for d in D]


    # print(knn(D, np.array([0, 2]), 1))


    plt.figure(figsize=[6,6])
    show_2D_decision_boundary(D, make_knn_classifier(D, 1), 500)
    plt.title('1-nn')
    plt.savefig('p1_1nn.png')
    plt.close()

    plt.figure(figsize=[6,6])
    show_2D_decision_boundary(D, make_knn_classifier(D, 3), 500)
    plt.title('3-nn')
    plt.savefig('p1_3nn.png')
    plt.close()

    plt.figure(figsize=[6,6])
    show_2D_decision_boundary(D, make_knn_classifier(Dt, 1, _transformer), 500)
    plt.title('transformed 1-nn')
    plt.savefig('p1_transformed_1nn.png')
    plt.close()

    plt.figure(figsize=[6,6])
    show_2D_decision_boundary(D, make_knn_classifier(Dt, 3, _transformer), 500)
    plt.title('transformed 3-nn')
    plt.savefig('p1_transformed_3nn.png')
    plt.close()

