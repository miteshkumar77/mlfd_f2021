# Logistic regression for classification using gradient descent
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
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
        data = [l for l in data if l[0] == 5 or l[0] == 1]

        return data

def avg_intensity(number):
    return sum(sum(r for r in l if r >= 0) for l in number[1]) / 256.0

def symmetry(number):
    ret = 0.0
    for i in range(16):
        for j in range(16):
            ret += np.abs(number[1][i][j] - number[1][i][15-j]) + \
                np.abs(number[1][i][j] - number[1][15-i][j])
    return 1 - ret / 1024.0

def compute_features(data):
    D = []
    for num in data:
        Xi = np.array([1, avg_intensity(num), symmetry(num)])
        yi = 1 if num[0] == 1 else -1
        D.append((Xi, yi))
    return D

def random_weights(d):
    return np.random.standard_normal(d+1)

def mean_logistic_error_gradient(D, wt):
    N = len(D)
    component = lambda xi, yi: yi * xi / (1 + np.exp(yi * np.dot(wt, xi)))
    return (-1 / N) * sum(component(*d) for d in D)

def mean_logistic_error(D, wt):
    N = len(D)
    component = lambda xi, yi: np.log(1 + np.exp(yi * np.dot(wt, xi)))
    return (1 / N) * sum(component(*d) for d in D)

def classification_error(D, wt):
    N = len(D)
    component = lambda xi, yi: 0.0 if np.sign(np.dot(wt, xi)) == np.sign(yi) else 1.0
    return (1 / N) * sum(component(*d) for d in D)

def logistic_regression(F, maxN=1000, maxErrorDiff=0.001, maxError=0.1, eta=0.1):
    d = len(F[0][0]) - 1
    wt = random_weights(d)
    errorDiff = np.inf
    error = mean_logistic_error(F, wt)
    for it in range(maxN):
        vt = -1 * mean_logistic_error_gradient(F, wt)
        wt = wt + eta * vt
        error2 = classification_error(F, wt)
        errorDiff = abs(error2 - error)
        error = error2
        if errorDiff < maxErrorDiff and error < maxError:
            print(errorDiff, error, it)
            break
    return wt

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


# given an input set of Weights W = [w_0, ..., w_d]
# and an input set of inputs X_i = [1, x_1, ..., x_d]
# calculate y_i using W^T . X_i
def eval_y(W, X_i):
    return np.sign(np.dot(W, X_i))

def pocket_algorithm(F, w0, maxN=1000):
    wp = w0.copy()
    wt = w0.copy()
    ep = classification_error(F, wp)
    idxs = [i for i in range(len(F))]
    for _ in range(maxN):
        np.random.shuffle(idxs)
        for i in idxs:
            X_t = F[i][0]
            y_t = F[i][1]
            if eval_y(wt, X_t) * y_t < 0:
                wt += y_t * X_t
                et = classification_error(F, wt)
                if et < ep:
                    ep = et
                    wp = wt.copy()
                break
    
    return wp
        
def get_2d_weight_equation(w):
    w0, w1, w2 = w[0], w[1], w[2]
    return lambda x1 : [(-1 * w1/w2) * x1 - (w0/w2)]

def get_cubic_weight_equation(w):
    return lambda x1 : np.fromiter((r for r in np.roots(
        [w[9], w[8] * x1 + w[5], w[7]*x1*x1 + w[4]*x1 + w[2], w[6]*x1*x1*x1 + w[3]*x1*x1+w[1]*x1+w[0]]
    ) if np.isreal(r)), dtype=np.float64)
def plot_weights(w, label, phi=None, eqn=get_2d_weight_equation):
    f = eqn(w)
    x = []
    y = []
    for xs in np.linspace(0, 1, 1000):
        xt = transform(xs, phi)
        for ys in f(xt):
            x.append(xt)
            y.append(ys)
    plt.scatter(np.array(x), np.array(y), label=label, s=0.1)

def plot_2d_weights_against_data(w_linreg, w_pocket, D, feature_labels, file, title):
    ones = [[d[0][1] for d in D if d[1] == 1],
            [d[0][2] for d in D if d[1] == 1]]
    fives = [[d[0][1] for d in D if d[1] == -1],
             [d[0][2] for d in D if d[1] == -1]]

    plt.figure(figsize=(10,10))
    plt.scatter(ones[0], ones[1], marker = 'o', color='blue', label='ones', s=0.5)
    plt.scatter(fives[0], fives[1], marker = 'x', color='red', label='fives', s=0.5)
    plot_weights(w_linreg, 'linear regression')
    plot_weights(w_pocket, 'pocket algorithm improvement')
    plt.xlabel(feature_labels[0])
    plt.ylabel(feature_labels[1])
    plt.title(f"{title}, Error = {classification_error(D, w_linreg)}, Improved Error = {classification_error(D, w_pocket)}", wrap=True)
    plt.legend()
    plt.savefig(file)
    plt.close()

def plot_2d_weights_against_transformed_data(w_linreg, w_pocket, D, T, feature_labels, file, title, eqn):
    ones = [[d[0][1] for d in D if d[1] == 1],
            [d[0][2] for d in D if d[1] == 1]]
    fives = [[d[0][1] for d in D if d[1] == -1],
             [d[0][2] for d in D if d[1] == -1]]

    plt.figure(figsize=(10,10))
    plt.scatter(ones[0], ones[1], marker = 'o', color='blue', label='ones', s=0.5)
    plt.scatter(fives[0], fives[1], marker = 'x', color='red', label='fives', s=0.5)
    plot_weights(w_linreg, 'linear regression', eqn=eqn)
    plot_weights(w_pocket, 'pocket algorithm improvement', eqn=eqn)
    plt.xlabel(feature_labels[0])
    plt.ylabel(feature_labels[1])
    plt.title(f"{title}, Error = {classification_error(T, w_linreg)}, Improved Error = {classification_error(T, w_pocket)}", wrap=True)
    plt.legend()
    plt.savefig(file)
    plt.close()


def get_2D_polynomial_transformer(Q):
    transformer = []
    def creator(p1, p2):
        return lambda x : np.power(x[1], p1) * np.power(x[2], p2)
    for order in range(0, Q+1):
        for idx in range(order+1):
            transformer.append(creator(order - idx, idx))
    return transformer

def transform(x, phi=None):
    if phi is None:
        return x
    return np.fromiter((phi_i(x) for phi_i in phi), dtype=np.float64)


def make_reg_plots():

    feature_labels = ['intensity', 'symmetry']
    F_train = compute_features(load_data(train_path))
    F_test = compute_features(load_data(test_path))
    print(f"N_train={len(F_train)}, N_test={len(F_test)}")
    w_linreg = lin_reg(F_train)
    w_pocket = pocket_algorithm(F_train, np.array(w_linreg[:]), maxN=10000)

    plot_2d_weights_against_data(w_linreg, w_pocket, F_train, feature_labels, 'p1a_train.png', 'Pocket Algorithm Against Training Data')
    plot_2d_weights_against_data(w_linreg, w_pocket, F_test, feature_labels, 'p1a_test.png', 'Pocket Algorithm Against Test Data')

def make_transform_plots():
    feature_labels = ['intensity', 'symmetry']
    transformer = get_2D_polynomial_transformer(3)
    F_train = compute_features(load_data(train_path))
    F_t_train = [(transform(d[0], transformer), d[1]) for d in F_train]
    F_test = compute_features(load_data(test_path))
    F_t_test = [(transform(d[0], transformer), d[1]) for d in F_test]
    w_linreg = lin_reg(F_t_train)
    w_pocket = pocket_algorithm(F_t_train, w_linreg.copy(), maxN=10000)

    plot_2d_weights_against_transformed_data(w_linreg, w_pocket, F_train, F_t_train, feature_labels, 'p1d_train.png', 'Pocket Algorithm Against 3rd Order Transform Training Data', eqn=get_cubic_weight_equation)
    plot_2d_weights_against_transformed_data(w_linreg, w_pocket, F_test, F_t_test, feature_labels, 'p1d_test.png', 'Pocket Algorithm Against 3rd Order Transform Test Data', eqn=get_cubic_weight_equation)


    

if __name__ == '__main__':
    make_reg_plots()
    make_transform_plots()


