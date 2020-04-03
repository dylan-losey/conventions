from environment2DoF import Params, Human, Robot, query, grad_F, grad_g
import numpy as np
import matplotlib.pyplot as plt
import math


params = Params()
human = Human()
robot = Robot()

control = human.control_adapt3
s_star = [1, -1, 0, 0]
gh = 0.0
magh = 1.0
human.task(s_star)
human.convention(gh, magh)

g = np.random.random() - 0.5
F = np.random.random((2,4)) * -0.5
robot.convention(F, g)

alpha = 0.001
J_prev = np.Inf

for count in range(1000):

    # rollout trajectory with human and robot
    xi_s, xi_z = robot.rollout(control)
    J = query(human, robot, control, F, g)

    # robot updates convention
    dJdg = grad_g(human, robot, control, F, g, params.delta)
    dJdF = grad_F(human, robot, control, F, g, params.delta)
    g_next = g - alpha * dJdg
    F_next = F - alpha * dJdF
    if np.linalg.norm(F_next) > 1.0:
        F_next /= np.linalg.norm(F_next)

    # human updates convention
    gh_next = gh * 0.9 + 0.5 * (1.0 - g - gh)
    magh_next = np.linalg.norm(human.s_star - xi_s[params.n_steps - 1])
    # magh_next = magh

    # record and update both human and robot
    print(count, J, g_next, gh_next, magh_next, F_next)
    if abs(J- J_prev) < 1e-4:
        break
    J_prev, g, gh, magh, F = J, g_next, gh_next, magh_next, F_next
    robot.convention(F, g)
    human.convention(gh, magh)


xi_s, xi_z = robot.rollout(control)
robot.plot_position(xi_s)
robot.plot_input(xi_z)
plt.show()
