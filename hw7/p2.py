import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, gf, w0, eta, N):
    wt = w0.copy()
    values = [f(wt)]
    for _ in range(N):
        wt = wt + -1 * eta * gf(wt)
        values.append(f(wt))
    return (wt, np.array(values))

def make_progress_plot(f, gf, w0, eta, N, file):
    (res, values) = gradient_descent(f, gf, w0.copy(), eta, 50)
    x = np.fromiter((i for i in range(0, N+1)), dtype=np.int8)
    plt.figure(figsize=(8,8))
    plt.scatter(x, values, s=4, color='blue')
    plt.title(f"eta: {eta}, initial weights: {w0}, final weights: {res}, final value: {values[-1]}", wrap=True)
    plt.xlabel(f'Iterations (0 ... {N})')
    plt.ylabel('value')
    plt.savefig(file)
    plt.close()

if __name__ == '__main__':
    f = lambda w : w[0] * w[0] + 2 * w[1] * w[1] + 2 * np.sin(2 * np.pi * w[0]) * np.sin(2 * np.pi * w[1])
    gf = (lambda w : np.array([
        2 * w[0] + 4 * np.pi * np.cos(2 * np.pi * w[0]) * np.sin(2 * np.pi * w[1]),
        4 * w[1] + 4 * np.pi * np.sin(2 * np.pi * w[0]) * np.cos(2 * np.pi * w[1])
    ]))
    w0 = np.array([0.1, 0.1])
    N = 50

    make_progress_plot(f, gf, w0, 0.01, 50, "p2a_1.png")
    make_progress_plot(f, gf, w0, 0.1, 50, "p2a_2.png")
    w0s = [np.array([0.1, 0.1]), np.array([1,1]), np.array([-0.5,-0.5]), np.array([-1,-1])]
    print("start location, end location, end value")
    for w in w0s:
        (res, values) = gradient_descent(f, gf, w.copy(), 0.01, 50)
        print(f"{w}, {res}, {values[-1]}")
