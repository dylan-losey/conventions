from environment import Params, Human, Robot, query
import numpy as np
import matplotlib.pyplot as plt
import copy


n_tasks = 10

s_star = np.zeros((2, n_tasks))
s_star[0, :] = 2.0 * np.random.rand(n_tasks) - 1.0
s_star[1, :] = np.random.rand(n_tasks) + 1.0

params = Params()
human = Human()
robot = Robot()

control = human.control_opt

F = copy.deepcopy(params.F_0)
J_0, J_star = 0, 0
for idx in range(n_tasks):
    s_curr = np.reshape(s_star[:,idx], (2,1))
    human.task(s_curr)
    J_0 += query(human, robot, control, F) / n_tasks

while(True):

    F1p = F + np.array([[params.delta, 0]])
    F1n = F - np.array([[params.delta, 0]])
    F2p = F + np.array([[0, params.delta]])
    F2n = F - np.array([[0, params.delta]])

    J1p, J1n, J2p, J2n = 0, 0, 0, 0
    for idx in range(n_tasks):
        s_curr = np.reshape(s_star[:,idx], (2,1))
        human.task(s_curr)
        J1p += query(human, robot, control, F1p) / n_tasks
        J1n += query(human, robot, control, F1n) / n_tasks
        J2p += query(human, robot, control, F2p) / n_tasks
        J2n += query(human, robot, control, F2n) / n_tasks

    F -= params.alpha * np.array([[J1p - J1n, J2p - J2n]])

    print(F)
    if abs(J1p - J1n) + abs(J2p - J2n) < 1e-5:
        break

for idx in range(n_tasks):
    s_curr = np.reshape(s_star[:,idx], (2,1))
    human.task(s_curr)
    J_star += query(human, robot, control, F) / n_tasks

plt.bar([1, 2], [J_0, J_star])
plt.xticks([1, 2], ['Initial', 'Optimal'])
plt.ylabel('Average Task Cost')
plt.show()
