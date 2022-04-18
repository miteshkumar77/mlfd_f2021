import numpy as np
from numba import jit

def create_nn(d, scale=1, dfl=None) :
    W = [None]
    if dfl is None:
        for l in range(1, len(d)):
            W.append(np.random.normal(0, scale, size=[d[l-1]+1, d[l]]))
    else:
        for l in range(1, len(d)):
            W.append(np.full(shape=[d[l-1]+1, d[l]], fill_value=dfl, dtype=np.float128))

    return W 

def augment_data(D):
    return np.append(np.ones(shape=[D.shape[0], 1]), D, axis=1)

def augment_1(x):
    return np.append(np.ones(shape=[1,1]), x, axis=0)

def deaugment_1(x):
    return x[1:]

def fprop_nn(W, x, T):
    X = [np.copy(x)]
    S = [None]
    for l in range(1, len(W)) :
        S.append(np.matmul(np.transpose(W[l]), X[l-1]))
        X.append(augment_1(T[l](S[l])))
    return (S, X)

def bprop_nn(W, X, y, outfn):
    L = len(W)-1
    D = [None for _ in W]
    if outfn == 'tanh':
        D[L] = (1/2) * (deaugment_1(X[L]) - y) * (1 - np.power(deaugment_1(X[L]), 2))
    elif outfn == 'identity':
        D[L] = (1/2) * (deaugment_1(X[L]) - y)
    else:
        raise ValueError("outfn=['identity', 'tanh']")
    for l in range(L-1, 0, -1):
        Tpl = deaugment_1(1 - np.power(X[l], 2))
        D[l] = np.multiply(Tpl, deaugment_1(np.matmul(W[l+1], D[l+1])))

    return D


# return T, such that T[l] is the transformation
# for the input to node l
def transitions_nn(L, outfn):
    Tm = [None] + [np.tanh for _ in range(1, L)]
    if outfn == 'identity':
        Tm.append(lambda x : x)
    elif outfn == 'tanh':
        Tm.append(np.tanh)
    elif outfn == 'sign':
        Tm.append(np.sign)
    else:
        raise ValueError("outfn=['identity', 'tanh', 'sign']")
    return Tm

def error_gradient_nn(W, T, D, outfn):
    Ein = np.zeros([1,1])
    G = [None if w is None else np.zeros(shape=w.shape) for w in W]
    L = len(W)-1
    N = len(D)
    for d in D:
        x = d[:-1]
        y = d[-1:]
        x = np.reshape(x, [len(x), 1])
        y = np.reshape(y, [1, 1])

        _, X = fprop_nn(W, x, T)
        D = bprop_nn(W, X, y, outfn)
        Ein += np.power((X[L][1:] - y), 2)
        for l in range(1, L+1):
            G[l] += np.matmul(X[l-1], np.transpose(D[l]))/N
    
    return (Ein/N, G)

def error(W, T, D):
    L = len(W)-1

    def lins():
        for i in range(len(D)):
            x = np.transpose(D[i:i+1, :-1])
            y = D[i:i+1, -1:]
            _, X = fprop_nn(W, x, T)
            yield np.power((X[L][1:] - y), 2)
    
    return np.mean(np.fromiter(lins(), dtype=np.float128))

def numerical_gradient_nn(W, T, D, delta):
    L = len(W)-1
    G = [None if w is None else np.zeros(shape=w.shape) for w in W]
    E0 = error(W, T, D)
    W = np.copy(W)
    for l in range(1, L+1):
        for r in range(len(W[l])):
            for c in range(len(W[l][r])):
                w0 = W[l][r][c]
                W[l][r][c] += delta
                G[l][r][c] = (error(W, T, D) - E0)/delta
                W[l][r][c] = w0
    return G

if __name__ == '__main__':
    W = create_nn([2, 2, 1], dfl=0.25)
    T = transitions_nn(len(W), 'tanh')
    D = augment_data(np.array([[1, 2, 1]]))
    Ein, G = error_gradient_nn(W, T, D, 'tanh')
    print(Ein)
    print(G)
    print(numerical_gradient_nn(W, T, D, 0.0001))

