import numpy as np
import matplotlib.pyplot as plt
np.random.seed(12341234)
def gen_line():
    x1, x2 = np.random.uniform(-1.0, 1.0, 2)
    return (x1 + x2, -1 * x1 * x2)

am = 0.0
bm = 0.0
iters1 = 1000
for i in range(iters1):
    a, b = gen_line()
    am += a
    bm += b

am /= float(iters1) 
bm /= float(iters1)
gm = lambda x : am * x + bm

print("g_mean(x) = {} x + {}".format(am, bm))

iters2 = 1000
bias = 0.0
for i in range(iters2):
    x = np.random.uniform(-1.0, 1.0, 1)
    dev = gm(x) - x*x
    bias += dev * dev

bias /= float(iters2)

var = 0.0
iters3 = 1000
for i in range(iters3):
    ad, bd = gen_line()
    gd = lambda x : ad * x + bd
    x = np.random.uniform(-1.0, 1.0, 1)
    dev = gd(x) - gm(x)
    var += dev * dev

var /= iters3

eout = 0.0
iters4 = 1000
for i in range(iters4):
    ad, bd = gen_line()
    gd = lambda x : ad * x + bd
    x = np.random.uniform(-1.0, 1.0, 1)
    dev = gd(x) - x * x
    eout += dev * dev

eout /= iters4

print("bias: {}, var: {}, eout: {}, bias+var: {}".format(bias, var, eout, bias+var))

xvals = np.linspace(-1, 1, 10000)
gmvals = [gm(xval) for xval in xvals]
fvals = [xval * xval for xval in xvals]

plt.plot(xvals, gmvals)
plt.plot(xvals, fvals)
plt.show()