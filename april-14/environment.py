import numpy as np
import matplotlib.pyplot as plt
import copy


class Simple(object):

    def __init__(self):
        self.timestep = 0.05
        self.n_steps = 21
        self.A = np.array([[1, self.timestep], [0, 1]])
        self.B = np.array([[0], [self.timestep]])
        self.Qcost = np.array([[10, 0], [0, 1]])
        self.Rcost = 1.0
        self.n_tasks = 5
        self.omega = np.zeros((2,self.n_tasks))
        self.omega[0,:] = np.linspace(0.2,1,self.n_tasks)

    def dynamics(self, s_star, s, R, H):
        z = np.array([[H, 0.1]]) @ (s_star - s)
        a = np.array([[R, 0.0]]) @ s + z
        s_next = self.A@s + self.B@a
        return s_next, z

    def cost_J(self, s_star, s_0, R, H):
        xi_s, xi_z, s = {}, {}, copy.deepcopy(s_0)
        effort, error = 0, 0
        for idx in range(self.n_steps - 1):
            xi_s[idx], e = s, s_star - s
            s, z = self.dynamics(s_star, s, R, H)
            error += float(np.transpose(e) @ self.Qcost @ e)
            effort += float(self.Rcost * z**2)
            xi_z[idx] = z
        xi_s[self.n_steps - 1], e = s, s_star - s
        error += float(np.transpose(e) @ self.Qcost @ e)
        return error + effort, xi_s, xi_z

    def cost_Q(self, s_0, R, H):
        expected_cost = 0
        for idx in range(self.n_tasks):
            s_star = np.array([[self.omega[0,idx]],[self.omega[1,idx]]])
            J, _, _ = self.cost_J(s_star, s_0, R, H)
            expected_cost += J / self.n_tasks
        return expected_cost

    def plot_position(self, xi_s):
    	time = np.linspace(0, 1, self.n_steps)
    	position = []
    	for key in xi_s:
    	    position.append(float(xi_s[key][0]))
    	plt.plot(time, position)

    def plot_input(self, xi_z):
    	time = np.linspace(0, 1, self.n_steps)
    	input = []
    	for key in xi_z:
    	    input.append(float(xi_z[key]))
    	plt.plot(time[:-1], input)
