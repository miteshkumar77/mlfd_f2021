from math import comb, ceil, floor, e as ex
import numpy as np
import matplotlib.pyplot as plt

def calc(e, u, N):
    l = max(0, ceil(N * (u - e)))
    r = min(N, floor(N * (u + e)))
    c = lambda x: comb(N, x) * pow(u, x) * pow(1 - u, N - x)
    a = sum(c(i) for i in range(l, r+1))
    return 1 - pow(a, 2)

def bound(e, N):
    h = 2 * pow(ex, -2 * N * e * e)
    return h + h - h * h


x = np.linspace(0, 1, 1000)
y = np.array([calc(e, 0.5, 6) for e in x])
yb = np.array([bound(e, 6) for e in x])
plt.plot(x, y, label="analytical")
plt.plot(x, yb, label="hoeffding bound")
plt.xlabel("e")
plt.ylabel("P[|v-u| > e]")
plt.legend()
plt.show()