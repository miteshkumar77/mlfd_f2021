import p1_4 as pla
import p3_x
import numpy as np
import matplotlib.pyplot as plt


def calc_steps(sep):
    D = p3_x.generate_linearly_separable_data(sep, N=20000)
    _, iters = pla.learn(D)
    return iters

seps = np.arange(start=0.2, step=0.2, stop=5.1)
iters = np.array([calc_steps(sep) for sep in seps])

plt.scatter(x=seps, y=iters)
plt.xlabel("sep")
plt.ylabel("iterations")
plt.savefig('p3_2.png')