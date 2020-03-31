from environment import Params, Human, Robot, query
import numpy as np
import matplotlib.pyplot as plt
import copy


n_tasks = 5

s_star = np.ones((2, n_tasks))
s_star[1,:] = np.linspace(-1, 1, n_tasks)

params = Params()
human = Human()
robot = Robot()

control = human.control_react

F_star, J_0, J_star = [], [], []

for idx in range(n_tasks):

    s_curr = np.reshape(s_star[:,idx], (2,1))
    F = copy.deepcopy(params.F_0)
    human.task(s_curr)
    xi_s, xi_z = robot.rollout(control)
    error = np.linalg.norm(s_curr - xi_s[20])
    J_0.append(query(human, robot, control, F))

    count = 0
    while(True):

        F1p = F + np.array([[params.delta, 0]])
        F1n = F - np.array([[params.delta, 0]])
        F2p = F + np.array([[0, params.delta]])
        F2n = F - np.array([[0, params.delta]])

        J1p = query(human, robot, control, F1p)
        J1n = query(human, robot, control, F1n)
        J2p = query(human, robot, control, F2p)
        J2n = query(human, robot, control, F2n)

        F -= params.alpha * np.array([[J1p - J1n, J2p - J2n]])

        J_0.append(query(human, robot, control, F))
        xi_s, xi_z = robot.rollout(control)
        error = np.linalg.norm(s_curr - xi_s[20])
        human.react(error)
        count += 1

        if abs(J1p - J1n) + abs(J2p - J2n) < 1e-5:
            print("I just converged for task # " + str(idx))
            break

    F_star.append(list(F[0]))
    J_star.append(query(human, robot, control, F))


F_star = np.asarray(F_star)
plt.plot(s_star[1,:], F_star)
plt.xlabel('Task Number')
plt.ylabel('Optimal Convention')
plt.legend(['F_1', 'F_2'])
plt.show()

plt.plot(s_star[1,:], J_0)
plt.plot(s_star[1,:], J_star)
plt.xlabel('Task Number')
plt.ylabel('Cost')
plt.legend(['Initial', 'Optimal'])
plt.show()
