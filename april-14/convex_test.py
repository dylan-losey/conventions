from environment import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


s_0 = np.array([[0],[0]])


R = np.linspace(-10.0, 10.0, 21)
H = np.linspace(0.5, 2.0, 21)
X, Y = np.meshgrid(R, H)
Z = np.zeros(np.shape(X))

xs, ys, zs = None, None, np.Inf
for idx in range(len(R)):
    for jdx in range(len(H)):
        Q = cost_Q(s_0, R[idx], H[jdx])
        if Q < zs:
            xs, ys, zs = R[idx], H[jdx], Q
        Z[jdx, idx] = Q

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.plot([xs, xs], [ys, ys], [zs-10.0, zs], 'ko-')
plt.xlabel("R")
plt.ylabel("H")
print(xs, ys, zs)
plt.show()
