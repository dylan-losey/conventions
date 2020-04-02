import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
import copy
import math



def query(human, robot, control, F, g):
    robot.convention(F, g)
    xi_s, xi_z = robot.rollout(control)
    return human.cost(xi_s, xi_z)

def rotZ(q):
    return np.array([[math.cos(q),-math.sin(q)],[math.sin(q),math.cos(q)]])



class Params(object):

    def __init__(self):

        mass = 0.5
        damper = 1.0
        timestep = 0.1
        n_steps = 21
        s_0 = [0.0, 0.0, 0.0, 0.0]
        g_0 = 0.0
        delta = 0.01
        alpha = 0.1
        offset = -0.5

        self.n_steps = n_steps
        self.timestep = timestep
        self.F_0 = np.array([[0,0,0,0],[0,0,0,0]])
        self.K_fixed = np.array([[1,0,0.1,0],[0,1,0,0.1]])
        self.s_0 = np.reshape(np.asarray(s_0), (4, 1))
        self.G_0 = rotZ(g_0)
        self.A = np.array([[1,0,timestep,0],[0,1,0,timestep],[0,0,1-timestep*damper/mass,0],[0,0,0,1-timestep*damper/mass]])
        self.B = np.array([[0,0],[0,0],[timestep/mass,0],[0,timestep/mass]]) @ rotZ(offset)
        self.Q = np.array([[10,0,0,0],[0,10,0,0],[0,0,1,0],[0,0,0,1]])
        self.R = np.array([[1.0, 0],[0, 1.0]])
        self.delta = delta
        self.alpha = alpha


class Human(object):

    def __init__(self):
        params = Params()
        self.Q = params.Q
        self.R = params.R
        self.K_fixed = params.K_fixed
        self.s_star = None

    def task(self, s_star):
    	self.s_star = np.reshape(np.asarray(s_star), (4, 1))

    def cost(self, xi_s, xi_z):
    	effort, error = 0, 0
    	for idx in xi_s:
    		if idx in xi_z:
    			effort += float(np.transpose(xi_z[idx]) @ self.R @ xi_z[idx])
    		e = self.s_star - xi_s[idx]
    		error += float(np.transpose(e) @ self.Q @ e)
    	return effort + error

    def control_fixed(self, s):
    	return self.K_fixed @ (self.s_star - s)


class Robot(object):

    def __init__(self):
        params = Params()
        self.A = params.A
        self.B = params.B
        self.F = params.F_0
        self.G = params.G_0
        self.s_0 = params.s_0
        self.n_steps = params.n_steps

    def convention(self, F, g):
        self.F = copy.deepcopy(F)
        self.G = rotZ(g)

    def dynamics(self, s, z):
        a = self.F @ s + self.G @ z
        s_next = self.A @ s + self.B @ a
        return s_next

    def rollout(self, control):
        xi_s, xi_z, s = {}, {}, copy.deepcopy(self.s_0)
        for idx in range(self.n_steps - 1):
            z = control(s)
            s_next = self.dynamics(s, z)
            xi_z[idx], xi_s[idx], s = z, copy.deepcopy(s), s_next
        xi_s[self.n_steps - 1] = s
        return xi_s, xi_z

    def plot_position(self, xi_s):
        time = np.linspace(0, 1, self.n_steps)
        position = []
        for key in xi_s:
            position.append([xi_s[key][0,0], xi_s[key][1,0]])
        plt.plot(time, position)

    def plot_input(self, xi_z):
        time = np.linspace(0, 1, self.n_steps)
        input = []
        for key in xi_z:
            input.append([xi_z[key][0,0], xi_z[key][1,0]])
        plt.plot(time[:-1], input)
