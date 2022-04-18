import numpy as np



def sample_semi(r1, r2, t1, t2, dx, dy):
    R = np.random.uniform(0, 1)
    T = np.random.uniform(t1, t2)

    Rtr = np.sqrt((r2*r2 - r1*r1) * R + r1*r1)
    return np.array([1, Rtr * np.cos(T) + dx, Rtr * np.sin(T) + dy])

def generate_linearly_separable_data(sep, N=2000):
    D = []
    for _ in range(N):
        if np.random.uniform(0, 1) >= 0.5:
            D.append((sample_semi(10, 15, 0, np.pi, 0, sep), -1))
        else:
            D.append((sample_semi(10, 15, np.pi, 2 * np.pi, 12.5, 0), 1))
    return D