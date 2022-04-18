
def calc(g, tests):
    r = [0 for _ in range(4)]
    for t in tests:
        m = 0
        for i in range(3):
            if t[i] == g[i]:
                m += 1
        r[m] += 1
    return r

if __name__ == "__main__":

    tests = [[0,0,0],
            [0,0,1],
            [0,1,0],
            [0,1,1],
            [1,0,0],
            [1,0,1],
            [1,1,0],
            [1,1,1]]

    g1 = [1,1,1]
    g2 = [0,0,0]
    g3 = [0,0,1]
    g4 = [1,1,0]
    print(calc(g1, tests))
    print(calc(g2, tests))
    print(calc(g3, tests))
    print(calc(g4, tests))

