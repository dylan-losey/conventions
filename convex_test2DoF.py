from environment2DoF import Params, Human, Robot, query, grad_F, grad_g
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math


params = Params()
human = Human()
robot = Robot()

control = human.control_fixed
s_star = [1, -1, 0, 0]
human.task(s_star)

g = np.linspace(-math.pi/2, math.pi/2, 21)
F = params.F_0
Z = np.zeros(np.shape(g))

xs, zs = None, np.Inf
for idx in range(len(g)):
    g_theta = g[idx]
    robot.convention(F, g_theta)
    xi_s, xi_z = robot.rollout(control)

    # # comment out - just for visualization
    # robot.plot_position(xi_s)
    # robot.plot_input(xi_z)
    # plt.show()

    J = human.cost(xi_s, xi_z)
    if J < zs:
        xs, zs = g_theta, J
    Z[idx] = J


plt.plot(g, Z)
plt.plot(xs, zs, 'ko-')
plt.show()


# # sanity check on the gradient procedure
# g = 0.5
# F = np.random.random((2,4)) * -0.5
# J = query(human, robot, control, F, g)
# print(J, g, F)
# while True:
#     dJdg = grad_g(human, robot, control, F, g, params.delta)
#     dJdF = grad_F(human, robot, control, F, g, params.delta)
#     g_next = g - params.alpha * dJdg
#     F_next = F - params.alpha * dJdF
#     J_next = query(human, robot, control, F_next, g_next)
#     print(J_next, g_next, F_next)
#     if abs(J_next - J) < 0.01:
#         break
#     g, F, J = g_next, F_next, J_next
