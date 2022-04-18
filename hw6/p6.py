import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
data_path = '../digits_data/ZipDigits.train'

def save_num(number, file):
    plt.imshow(number[1], cmap=plt.cm.binary)
    plt.title(str(number[0]))
    plt.savefig(file)

def avg_intensity(number):
    return sum(sum(r for r in l if r >= 0) for l in number[1]) / 256.0

def symmetry(number):
    ret = 0.0
    for i in range(16):
        for j in range(16):
            ret += np.abs(number[1][i][j] - number[1][i][15-j]) + \
                np.abs(number[1][i][j] - number[1][15-i][j])
    return 1 - ret / 1024.0

with open(data_path, 'r') as f:
    def line_to_entry(l):
        d = [float(e) for e in l.strip().split(' ')]
        m = np.zeros([16, 16])
        for idx, val in enumerate(d[1:]):
            m[idx//16][idx%16] = val

        return (int(d[0]), m)
    
    data = [line_to_entry(l) for l in f]
    data = [l for l in data if l[0] == 5 or l[0] == 1]
    for entry in data:
        if entry[0] == 5:
            save_num(entry, '5.png')
            break

    for entry in data:
        if entry[0] == 1:
            save_num(entry, '1.png')
            break
    
    plt.close()
    
    ones_intensity = []
    ones_symmetry = []
    fives_intensity = []
    fives_symmetry = []
    
    for num in data:
        i, s = avg_intensity(num), symmetry(num)
        if num[0] == 1:
            ones_intensity.append(i)
            ones_symmetry.append(s)
        else:
            fives_intensity.append(i)
            fives_symmetry.append(s)
    
    plt.scatter(ones_intensity, ones_symmetry, marker='o', color='blue', label='ones')
    plt.scatter(fives_intensity, fives_symmetry, marker='x', color='red', label='fives')
    plt.xlabel('average intensity')
    plt.ylabel('symmetry')
    plt.legend()
    plt.savefig('p6.png')