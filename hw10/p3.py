import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import math

def gen_unit_square(N):
    return np.random.random([N,2])

def gen_rand_gaussians(N, Gs):
    intervals = np.concatenate([np.array([0]), np.sort(np.random.randint(0, N, len(Gs)-1)), np.array([N])])
    def gen(i):
        return np.random.multivariate_normal(Gs[i], 0.1 * np.identity(Gs[i].shape[0]), intervals[i+1] - intervals[i])
    return np.concatenate([gen(i) for i in range(len(Gs))], axis=0)

def min_dist(C, P):
    return np.min(np.linalg.norm(C - P, axis=1))

def max_dist(C, P):
    return np.max(np.linalg.norm(C - P, axis=1))

def calcdist(P1, P2):
    return math.sqrt(math.pow(P1[0] - P2[0], 2) + math.pow(P1[1]-P2[1], 2))

def min_idx_slow(C, P):
    dist = calcdist(C[0], P)
    idx = 0
    for i in range(1, len(C)):
        d = calcdist(C[i], P)
        if d < dist:
            dist = d
            idx = i
    return idx

def min_idx(C, P):
    return np.argmin(np.linalg.norm(C -  P, axis=1))

def furthest_idx(D, C):
    # D: N x 2
    # C: M x 2
    if len(C) == 0:
        return np.random.randint(0, len(D))
    else:
        return np.argmax(np.apply_along_axis((lambda P : min_dist(C, P)), 1, D))

def separate(D, centers):
    M = len(centers)
    partitions = [np.ndarray([0, 2]) for _ in range(M)]
    for P in D:
        bucket = min_idx(centers, P)
        partitions[bucket] = np.append(partitions[bucket], [P], axis=0)
    tmp =  [(np.mean(partitions[bucket], axis=0), partitions[bucket]) for bucket in range(M)]
    return [(t[0], max_dist(t[1], t[0]), t[1]) for t in tmp]

def partition(D, M):
    centers = np.ndarray([0, 2])
    while(len(centers) < M):
        furthest = furthest_idx(D, centers)
        centers = np.append(centers, [D[furthest]], axis=0)
        D[len(D)-1], D[furthest] = D[furthest], D[len(D)-1]
        D = D[0:len(D)-1]
    
    D = np.append(D, centers, axis=0)
    return separate(D, centers)

def improve_partition(P):
    D = np.concatenate([p[2] for p in P], axis=0)
    Centers = np.concatenate([[p[0]] for p in P], axis=0)
    return separate(D, Centers)

def show_partition(P):
    colors = cm.rainbow(np.linspace(0, 1, len(P) + 1))
    for y, c in zip(P, colors[1:]):
        plt.scatter(y[2][ : , 0], y[2][ : , 1], color=c, s=2)
        plt.scatter([y[0][0]], [y[0][1]], color=colors[0], s=35, marker='x')
        crc = plt.Circle((y[0][0], y[0][1]), y[1], fill=False)
        plt.gcf().gca().add_artist(crc)

def branch_and_bound_nn(Partition, Pt):
    Centers = np.concatenate([[p[0]] for p in Partition], axis=0)
    closest_center_part = min_idx(Centers, Pt)
    Ans = Partition[closest_center_part][2][min_idx(Partition[closest_center_part][2], Pt)]
    AnsDist = np.linalg.norm(Ans - Pt)
    CheckDist = AnsDist
    for i in range(len(Partition)):
        if i == closest_center_part:
            continue
        if CheckDist > np.linalg.norm(Pt - Partition[i][0]) - Partition[i][1]:
            CAns = Partition[i][2][min_idx(Partition[i][2], Pt)]
            CDist = np.linalg.norm(CAns - Pt)
            if CDist < AnsDist:
                AnsDist = CDist
                Ans = CAns
    return Ans

def benchmark_nn(D, Partition, N):
    test_pts = np.random.random((N, 2))
    start_bf = time.time()
    for i in range(N):
        min_idx(D, test_pts[i])
    end_bf = time.time()

    start_bd = time.time()
    for i in range(N):
        branch_and_bound_nn(Partition, test_pts[i])
    end_bd = time.time()
    return (end_bf - start_bf, end_bd - start_bd)

def bnb_test(D, Partition, N):
    test_pts = np.random.random((N, 2))
    
    for i in range(N):
        # assert(all(D[min_idx(D, test_pts[i])] == branch_and_bound_nn(Partition, test_pts[i])))
        bf = D[min_idx(D, test_pts[i])]
        bb = branch_and_bound_nn(Partition, test_pts[i])
        if not all(np.equal(bf, bb)):
            print(bf)
            print("")
            print(bb)

if __name__ == '__main__':
    # N x 2
    D = gen_unit_square(100000)
    P = partition(D, 10)
    plt.figure(figsize=(10,10))
    show_partition(P)
    plt.title("initial partition of 10000 points")
    plt.savefig('3a1_initial.png')
    plt.close()

    for _ in range(20):
        P = improve_partition(P)
    
    plt.figure(figsize=(10,10))
    show_partition(P)
    plt.title("20 iterations of improvement on 10000 points")
    plt.savefig('3a1_improved.png')
    plt.close()

    bench = benchmark_nn(D, P, 10000)
    print("part a:")
    print(f"brute force time: {bench[0]}, branch and bound time: {bench[1]}")


    Gs = gen_unit_square(10)
    D2 = gen_rand_gaussians(100000, Gs)
    P2 = partition(D2, 10)

    plt.figure(figsize=(10,10))
    show_partition(P2)
    plt.title("initial partition of 10000 gaussian points")
    plt.savefig('3b1_initial.png')
    plt.close()

    for _ in range(20):
        P2 = improve_partition(P2)
    
    plt.figure(figsize=(10,10))
    show_partition(P2)
    plt.title("20 iterations of improvement on 10000 points")
    plt.savefig('3b1_improved.png')
    plt.close()

    bench = benchmark_nn(D2, P2, 10000)
    print("part b:")
    print(f"brute force time: {bench[0]}, branch and bound time: {bench[1]}")

   