import numpy as np
import matplotlib.pyplot as plt
import p1_4 as pla
import p3_x


# pla.create_2D_plots(20000, lambda : p3_x.generate_linearly_separable_data(5), "p3_1a.png")

  
pts = [p3_x.sample_semi(.1, 10, 0, 2*np.pi, 0, 0) for _ in range(20000)]
x = [pt[1] for pt in pts]
y = [pt[2] for pt in pts]

plt.scatter(x, y, s=0.4)
plt.show()