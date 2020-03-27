from environment import Params, Human, Robot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


human = Human()
robot = Robot()

control = human.control_opt
s_star = [1, 0]
human.task(s_star)

f1 = np.arange(-2.0, 2.0, 0.1)
f2 = np.arange(-2.0, 2.0, 0.1)
X, Y = np.meshgrid(f1, f2)
Z = np.zeros(np.shape(X))

xs, ys, zs = None, None, np.Inf
for idx in range(len(f1)):
    for jdx in range(len(f2)):
        F = np.array([[f1[idx], f2[jdx]]])
        robot.convention(F)
        human.convention(F)
        xi_s, xi_z = robot.rollout(control)
        J = human.cost(xi_s, xi_z)
        if J < zs:
            xs, ys, zs = f1[idx], f2[jdx], J
        Z[idx, jdx] = J

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.plot([ys, ys], [xs, xs], [0, zs], 'ko-')
plt.show()
