import numpy as np
import matplotlib.pyplot as plt
# generates weights W = [w_0, w_1, ..., w_d]
# where each W_i is randomly distributed between -1 and 1
def gen_line(d):
    return np.random.uniform(-10, 10, d+1)


# given an input set of Weights W = [w_0, ..., w_d]
# and an input set of inputs X_i = [1, x_1, ..., x_d]
# calculate y_i using W^T . X_i
def eval_y(W, X_i):
    return np.sign(np.dot(W, X_i))

# generates a single input value X_i and its calculated
# output value y_i given a set of weights.
def gen_input(W):
    X_i = np.append(np.array([1]), np.random.uniform(-10000, 10000, len(W)-1))
    return (X_i, eval_y(W, X_i))

# generates n length d dimensional Data D that 
# is linearly separable by weights W
# D_i: X_i = [1, x_1, ..., x_d], y_i = {+1, -1}
def gen_linearly_separable_data(W, n):
    return [gen_input(W) for _ in range(n)]

# for linearly separable d dimensional input data D
# where D_i: X_i: [1, x_1, ..., x_d], y_i: {+1, -1}
# output weights that are able to separate the data.
# i.e. learn from the data
def learn(D):
    idxs = np.array([i for i in range(len(D))])
    done = False
    d = D[0][0].size - 1
    W = gen_line(d)
    iters = 0
    while not done:
        done = True
        np.random.shuffle(idxs)
        for i in idxs:
            X_t = D[i][0]
            y_t = D[i][1]
            if eval_y(W, X_t) != y_t:
                done = False
                W += y_t * X_t
                break
        if not done:
            iters += 1
    return (W, iters)

def create_2D_plots(n, generator=None, fname=None):
    d = 2
    w_gen = gen_line(2)
    print("generation weights: {}".format(str(w_gen)))
    m = -1 * w_gen[1]/w_gen[2]
    b = -1 * w_gen[0]/w_gen[2]
    print("line: x_2 = {} * x_1 + {}".format(m, b))

    D = generator() if generator != None else gen_linearly_separable_data(w_gen, n)

    print("linearly separable data D: {}".format(str(D)))

    learned_W, iters = learn(D)
    learned_m = -1 * learned_W[1]/learned_W[2]
    learned_b = -1 * learned_W[0]/learned_W[2]
    print("Learning took {} iterations...".format(iters))
    print("weights learned: {}".format(str(learned_W)))
    print("line learned: x_2 = {} * x_1 + {}".format(learned_m, learned_b))

    lower_lin_space = int(min([X_i[1] for X_i, _ in D]))
    upper_lin_space = int(max([X_i[1] for X_i, _ in D]))

    print(lower_lin_space, upper_lin_space)

    xPos = np.array([X_i[1] for X_i, y_i in D if y_i > 0])
    yPos = np.array([X_i[2] for X_i, y_i in D if y_i > 0])
    xNeg = np.array([X_i[1] for X_i, y_i in D if y_i < 0])
    yNeg = np.array([X_i[2] for X_i, y_i in D if y_i < 0])

    plt.scatter(xPos, yPos, c='blue', label='+')
    plt.scatter(xNeg, yNeg, c='red', label='-')

    x = np.linspace(lower_lin_space, upper_lin_space, upper_lin_space - lower_lin_space + 1)
    y_gen = m * x + b
    y_learned = learned_m * x + learned_b

    gen_label = "generated: x_2 = {} * x_1 + {}".format(m, b)
    learned_label = "learned: x_2 = {} * x_1 + {}".format(learned_m, learned_b)

    plt.plot(x, y_gen, c='green', label=gen_label)
    plt.plot(x, y_learned, c='purple', label=learned_label)
    plt.title('Graph of perceptron demonstration: N = {}, Iterations to learn = {}'.format(n, iters))
    plt.xlabel('x_1', color='#1C2833')
    plt.ylabel('x_2', color='#1C2833')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='best')
    plt.grid()
    if fname != None:
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.show()
    plt.clf()

create_2D_plots(20, '1_4_b.png')

create_2D_plots(20, '1_4_c.png')

create_2D_plots(100, '1_4_d.png')

create_2D_plots(1000, '1_4_e.png')