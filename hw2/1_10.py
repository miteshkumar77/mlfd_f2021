import numpy as np
import matplotlib.pyplot as plt
import math

def flip_coins(n):
    return np.array([np.random.randint(0, 2, 10) for _ in range(n)])

N = 1000
def avg(c):
    return sum(c)/len(c)

def calc_cs(N):
    flipped = flip_coins(N)
    c_1 = flipped[0]
    c_rand = flipped[np.random.randint(0, N)]
    c_min = min(flipped, key=lambda x: sum(x))
    return [avg(c) for c in (c_1, c_rand, c_min)]

def make_hists(N, data):
    fig, axes = plt.subplots(1, 3)
    
    axes[0].set_ylabel('v_1')
    axes[1].set_ylabel('v_rand')
    axes[2].set_ylabel('v_min')
    # h_x = np.arange(0, 1, 10)
    h_x = np.linspace(0, 1, 1000)
    h_y = [math.e ** (-2 * abs(x-0.5) * abs(x-0.5) * 1000) for x in h_x]
    axes[0].plot(h_x, h_y)
    axes[1].plot(h_x, h_y)
    axes[2].plot(h_x, h_y)
    plt.show()

def make_epsilon(N, data):


data = [calc_cs(N) for _ in range(100000)]

make_hists(1000, data)


