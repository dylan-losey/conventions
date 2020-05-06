from environment import Params, Human, Robot, query, grad_F
import numpy as np
import matplotlib.pyplot as plt
import math


params = Params()
human = Human()
robot = Robot()

control = human.control_react
s_star = [1,0]
magh = 1.0
human.task(s_star)
human.update(magh)

F = np.random.random((1,2)) * -0.5
robot.convention(F)

alpha = 0.01
J_prev = np.Inf

for count in range(1000):

    # rollout trajectory with human and robot
    xi_s, xi_z = robot.rollout(control)
    J = query(human, robot, control, F)

    # robot updates convention
    dJdF = grad_F(human, robot, control, F, params.delta)
    F_next = F - alpha * dJdF
    if np.linalg.norm(F_next) > 1.0:
        F_next /= np.linalg.norm(F_next)

    # human updates convention
    magh_next = 0.5 + (np.random.random() - 0.5) * 0.99**count
    # magh_next = magh + 0.1*(np.linalg.norm(human.s_star - xi_s[params.n_steps - 1]) - 0.5)
    # magh_next = magh

    # record and update both human and robot
    print(count, J, magh_next, F_next)
    if abs(J- J_prev) < 1e-5:
        break
    J_prev, magh, F = J, magh_next, F_next
    robot.convention(F)
    human.update(magh)

    alpha *= 1.0


xi_s, xi_z = robot.rollout(control)
robot.plot_position(xi_s)
robot.plot_input(xi_z)
plt.show()
