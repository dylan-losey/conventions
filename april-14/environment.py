import numpy as np
import matplotlib.pyplot as plt
import copy


mass = 1.0
timestep = 0.05
n_steps = 21
A = np.array([[1, timestep], [0, 1]])
B = np.array([[0], [timestep / mass]])
Qcost = np.array([[10, 0], [0, 1]])
Rcost = 1.0
n_tasks = 5
omega = np.zeros((2,n_tasks))
omega[0,:] = np.linspace(0.2,1,n_tasks)


def dynamics(s_star, s, R, H):
    z = np.array([[H, 0.1]]) @ (s_star - s)
    a = np.array([[R, 0.0]]) @ s + z
    s_next = A@s + B@a
    return s_next, z

def rollout(s_star, s_0, R, H):
    xi_s, xi_z, s = {}, {}, copy.deepcopy(s_0)
    for idx in range(n_steps - 1):
        xi_s[idx] = copy.deepcopy(s)
        s, z = dynamics(s_star, s, R, H)
        xi_z[idx] = z
    xi_s[n_steps - 1] = s
    return xi_s, xi_z

def cost_J(s_star, s_0, R, H):
    xi_s, xi_z = rollout(s_star, s_0, R, H)
    effort, error = 0, 0
    for idx in xi_s:
        if idx in xi_z:
            effort += float(Rcost * xi_z[idx]**2)
        e = s_star - xi_s[idx]
        error += float(np.transpose(e) @ Qcost @ e)
    return effort + error

def cost_Q(s_0, R, H):
    expected_cost = 0
    for idx in range(n_tasks):
        s_star = np.array([[omega[0,idx]],[omega[1,idx]]])
        expected_cost += cost_J(s_star, s_0, R, H)
    return expected_cost / n_tasks

def plot_position(xi_s):
	time = np.linspace(0, 1, n_steps)
	position = []
	for key in xi_s:
	    position.append(float(xi_s[key][0]))
	plt.plot(time, position)

def plot_input(xi_z):
	time = np.linspace(0, 1, n_steps)
	input = []
	for key in xi_z:
	    input.append(float(xi_z[key]))
	plt.plot(time[:-1], input)
