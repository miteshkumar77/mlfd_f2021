import cvxopt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def svm_predict(kf, b_str, a_str, sv_x, sv_y, x):
    return b_str + np.sum(a_str * sv_y * kf(sv_x, x))

def fit_sm_svm(X, Y, kf, C):
    m, _ = X.shape

    K = np.zeros(shape=[m,m])
    for i in range(m):
        for j in range(m):
            K[i, j] = kf(X[i], X[j])

    P = cvxopt.matrix(np.outer(Y, Y) * K) 
    q = cvxopt.matrix(-1 * np.ones(m))
    G = cvxopt.matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
    h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt.matrix(Y.reshape(1, m))
    b = cvxopt.matrix(0.0)
    cvxopt.solvers.options['show_progress'] = False

    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    l_multipliers = np.ravel(solution['x'])

    sv_ind = np.nonzero(l_multipliers > 1e-4)[0]
    sv_multipliers = l_multipliers[sv_ind]
    sv_xs = X[sv_ind]
    sv_ys = Y[sv_ind]

    dec_sv_ind = np.nonzero(sv_multipliers < C)[0]
    if len(dec_sv_ind) == 0:
        print(C)
        print(C-1e-4)
        print(sv_multipliers)
    dec_sv_ind = dec_sv_ind[-1]
    dec_sv_x = X[dec_sv_ind]
    dec_sv_y = Y[dec_sv_ind]

    b_str = dec_sv_y - svm_predict(kf, 0.0, sv_multipliers, sv_xs, sv_ys, dec_sv_x)


    def predict(x):
        return svm_predict(kf, b_str, sv_multipliers, sv_xs, sv_ys, x)


    def classify(x):
        return np.sign(svm_predict(kf, b_str, sv_multipliers, sv_xs, sv_ys, x))
    return classify, predict, sv_ind


def scatter_dataset(X, Y):
    Yr = np.reshape(Y, [len(Y)])
    Xno = X[np.nonzero(Yr == -1)]
    Xyes = X[np.nonzero(Yr == 1)]

    plt.scatter(Xyes[:, :1], Xyes[:, -1:], marker='.', c='blue', s=3)
    plt.scatter(Xno[:, :1], Xno[:, -1:], marker='o', c='red', s=3)

def classification_error(X, Y, classify):
    Z = np.fromiter((classify(x) for x in X), dtype=np.float64)
    return np.mean(np.logical_xor(Z > 0, Y > 0))

def show_2D_decision_boundary(X, Y, classify, res):
    x1_lo = np.min(X[:, 0]) - 0.3
    x1_hi = np.max(X[:, 0]) + 0.3 
    x2_lo = np.min(X[:, 1]) - 0.3
    x2_hi = np.max(X[:, 1]) + 0.3
    

    x1, x2 = np.linspace(x1_lo, x1_hi, res), np.linspace(x2_lo, x2_hi, res)
    xx1, xx2 = np.meshgrid(x1, x2)

    z = np.fromiter((classify(x) for x in np.c_[xx1.ravel(), xx2.ravel()]), dtype=np.float64)

    zz = np.reshape(z, xx1.shape)
    cmap_light = ListedColormap(["orange", "cyan"])

    plt.contourf(xx1, xx2, zz, cmap=cmap_light)
    scatter_dataset(X, Y)

def cv_error(C, d_train, kf):
    m, _ = d_train.shape
    _, predictor, sv_inds = fit_sm_svm(d_train[:, :2], d_train[:, 2], kf, C)
    sv_inds = set(sv_inds)
    print(len(sv_inds))
    def it():
        for i in range(m):
            if i in sv_inds:
                Xcv = np.concatenate((d_train[:i, :2], d_train[i+1:, :2]), axis=0)
                Ycv = np.concatenate((d_train[:i, 2], d_train[i+1:, 2]), axis=0)
                tmp_classifier, tmp_predictor, _ = fit_sm_svm(Xcv, Ycv, kf, C)
                yield int(tmp_classifier(d_train[i, :2]) != np.sign(d_train[i, 2]))
                # yield np.power(tmp_predictor(d_train[i, :2]) - np.sign(d_train[i, 2]), 2)
            else:
                yield int(classifier(d_train[i, :2]) != np.sign(d_train[i, 2]))
                # yield np.power(predictor(d_train[i, :2]) - np.sign(d_train[i, 2]), 2)
    return np.mean(np.fromiter(it(), dtype=np.float64))

def cv_best(Cs, d_train, kf):
    def it():
        for C in Cs:
            r = cv_error(C, d_train, kf)
            yield r
    err_cvs = np.fromiter(it(), dtype=np.float64)
    minidx = np.argmin(err_cvs)
    return (err_cvs, Cs[minidx], err_cvs[minidx])

if __name__ == '__main__':
    d_test = np.genfromtxt('../../f_test.txt', delimiter=',')[:, 1:]
    d_train = np.genfromtxt('../../f_train.txt', delimiter=',')[:, 1:]

    def q4_kernel(x1, x2):
        return np.power(1 + np.matmul(x1, x2), 8)

    smallC = 1 
    largeC = 10000
    # classifier = fit_soft_margin_svm(q4_kernel, d_train[:, :2], d_train[:, 2], 0.1)
    classifier, _, _ = fit_sm_svm(d_train[:, :2], d_train[:, 2], q4_kernel, smallC)
    plt.figure(figsize=[10,10])
    show_2D_decision_boundary(d_train[:, :2], d_train[:, 2], classifier, 400)
    plt.title(f'C = {smallC}')
    plt.savefig('4_a_small.png')
    plt.close()

    classifier, _, _ = fit_sm_svm(d_train[:, :2], d_train[:, 2], q4_kernel, largeC)
    plt.figure(figsize=[10,10])
    show_2D_decision_boundary(d_train[:, :2], d_train[:, 2], classifier, 400)
    plt.title(f'C = {largeC}')
    plt.savefig('4_a_large.png')
    plt.close()

    Cs = np.power(1.7, np.arange(0, 20, 1))
    err_cvs, C_best, ecv_best = cv_best(Cs, d_train, q4_kernel)
    np.savetxt('err_cvs.csv', err_cvs, delimiter=',')
    # err_cvs = np.genfromtxt('err_cvs.csv', delimiter=',')
    # C_best = Cs[np.argmin(err_cvs)]
    # ecv_best = np.min(err_cvs)

    C_worst = Cs[np.argmax(err_cvs)]
    plt.figure(figsize=[10,10])
    plt.plot(Cs, err_cvs, marker='x')
    plt.title(f'Ecv vs C, bestC={C_best}, bestEcv={ecv_best}')
    plt.savefig('4_c_ecv.png')
    plt.close()

    plt.figure(figsize=[10,10])
    classifier, _, _ = fit_sm_svm(d_train[:, :2], d_train[:, 2], q4_kernel, C_best)
    show_2D_decision_boundary(d_train[:, :2], d_train[:, 2], classifier, 400)
    plt.title(f'C={C_best}, Eout={classification_error(d_test[:, :2], d_test[:, 2], classifier)}')
    plt.savefig('4_cv_best.png')
    plt.close()

    plt.figure(figsize=[10,10])
    classifier, _, _ = fit_sm_svm(d_train[:, :2], d_train[:, 2], q4_kernel, C_worst)
    show_2D_decision_boundary(d_train[:, :2], d_train[:, 2], classifier, 400)
    plt.title(f'C={C_worst}, Eout={classification_error(d_test[:, :2], d_test[:, 2], classifier)}')
    plt.savefig('4_cv_worst.png')
    plt.close()
