import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

def hard_margin_qp_solve(K, X, Y):
    m = X.shape[0]
    Q = np.zeros(shape=(m, m))
    for i in range(0, m):
        for j in range(0, m):
            Q[i][j] = Y[i] * Y[j] * K(X[i, :], X[j, :])
    P = cvxopt_matrix(Q)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(-np.eye(m))
    h = cvxopt_matrix(np.zeros(shape=(m, 1)))
    A = cvxopt_matrix(Y.reshape(1, m))
    b = cvxopt_matrix(np.zeros(1))


    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solvers.options['abstol'] = 1e-10
    cvxopt_solvers.options['reltol'] = 1e-10
    cvxopt_solvers.options['feastol'] = 1e-10

    #Run solver
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas

def hard_margin_svm(K, X, Y):
    alphas = hard_margin_qp_solve(K, X, Y).reshape(2)
    print(alphas)
    s = np.nonzero(alphas > 1e-6)[0]

    def it():
        for n in s:
            yield Y[n] * alphas[n] * K(X[n, :], X[s[0], :])
    
    b_str = Y[s[0]] - np.sum(np.fromiter(it(), dtype=np.float64))
    return b_str

def compute_kernel(kf, X, Y):
    m = X.shape[0]
    K = np.zeros(shape=[m,m])
    for i in range(0, m):
        for j in range(0, m):
            K[i, j] = kf(X[i, :], X[j, :])
    return K
    
def fit_soft_margin_svm(kf, X, Y, C):
    K = compute_kernel(kf, X, Y)
    m = X.shape[0]
    P = cvxopt_matrix(np.outer(Y, Y) * K)
    q = cvxopt_matrix(-1 * np.ones((m, 1)))
    G = cvxopt_matrix(np.concatenate((-1 * np.eye(m), np.eye(m)), axis=0))
    h = cvxopt_matrix(np.concatenate((np.zeros(shape=(m, 1)), np.full([m, 1], C)), axis=0))
    A = cvxopt_matrix(Y.reshape(1, m))
    b = cvxopt_matrix(np.zeros(1))

    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    a = np.array(sol['x'])
    sv_ind = np.nonzero(a > 1e-5)[0]

    a_str = a[sv_ind]
    sv_x = X[sv_ind]
    sv_y = Y[sv_ind]
    # sv_y_p = np.nonzero(sv_y > 0)[0]

    # def bit():
    #     for i in sv_ind:
    #         yield 1 - np.sum(a_str * sv_y * K[sv_ind, i])
    # b_str = 1 - np.sum(a_str * sv_y * K[sv_ind, sv_y_p[0]])
    # b_str = np.mean(np.fromiter(bit(), dtype=np.float64))
    def bit():
        for n in sv_ind:
            yield Y[n] - np.sum(a_str * sv_y * K[n, sv_ind])
    b_str = np.mean(np.fromiter(bit(), dtype=np.float64))

    return (b_str, a_str, sv_y, sv_x)
#     def bit():
#         for i in sv_ind:
#             print(Y[i])
#             print(np.sum(a_str * sv_y * K[i, sv_ind]))
#             yield Y[i] - np.sum(a_str * sv_y * K[i, sv_ind])
#     b = np.mean(np.fromiter(bit(), dtype=np.float64))

#     return (a, a_str, sv_ind, sv_x, sv_y, K, b)

def classify(b_str, a_str, sv_y, sv_x, kf, x):
    return np.sign(np.sum(a_str * sv_y * kf(sv_x, x)) + b_str)

# def classify(a, a_strs, sv_inds, sv_xs, sv_ys, K, b, kf, x):
#     def sm():
#         for a_str, sv_y, sv_x in zip(a_strs, sv_ys, sv_xs):
#             yield a_str * sv_y * kf(x, sv_x)
#     return np.sign(np.mean(np.fromiter(sm(), dtype=np.float64)) + b)


def make_classifier(b_str, a_str, sv_y, sv_x, kf):
    def classifier(x):
        return classify(b_str, a_str, sv_y, sv_x, kf, x)
    return classifier

def scatter_dataset(X, Y):
    Yr = np.reshape(Y, [len(Y)])
    Xno = X[np.nonzero(Yr == -1)]
    Xyes = X[np.nonzero(Yr == 1)]

    plt.scatter(Xyes[:, :1], Xyes[:, -1:], marker='.', c='blue', s=3)
    plt.scatter(Xno[:, :1], Xno[:, -1:], marker='o', c='red', s=3)

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
    
if __name__ == '__main__':
    # def q1_kernel(x1, x2):
    #     def t(x):
    #         z = np.zeros(shape=[2,1])
    #         z[0] = x[0] * x[0] * x[0] - x[1]
    #         z[1] = x[0] * x[1]
    #         return z
    #     return np.dot(np.transpose(t(x1)), t(x2))
    
    # D = np.array([[1, 0, 1], [-1, 0, -1]]).astype('float')
    # print(hard_margin_svm(q1_kernel, D[:, :2], D[:, 2]))

    d_test = np.genfromtxt('../../f_test.txt', delimiter=',')[:, 1:]
    d_train = np.genfromtxt('../../f_train.txt', delimiter=',')[:, 1:]

    def q4_kernel(x1, x2):
        return np.power(1 + np.matmul(x1, x2), 8)

    svm_parts = fit_soft_margin_svm(q4_kernel, d_train[:, :2], d_train[:, 2], 10000)
    classifier = make_classifier(*svm_parts, q4_kernel)
    plt.figure(figsize=[10,10])
    show_2D_decision_boundary(d_train[:, :2], d_train[:, 2], classifier, 200)
    plt.savefig('4_a.png')

    # print(classify(*svm_parts, q4_kernel, d_train[0, :2]))