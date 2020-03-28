import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
import copy


def query(human, robot, control, F):
    robot.convention(F)
    human.convention(F)
    xi_s, xi_z = robot.rollout(control)
    return human.cost(xi_s, xi_z)


class Params(object):

	def __init__(self):

		mass = 1.0
		damper = 1.0
		timestep = 0.1
		n_steps = 21
		s_0 = [0.0, 0.0]
		F_0 = [0.0, 0.0]
		delta = 0.01
		alpha = 0.1
		K_pro = [1.0, 0.1]

		self.n_steps = n_steps
		self.s_0 = np.reshape(np.asarray(s_0), (2, 1))
		self.F_0 = np.reshape(np.asarray(F_0), (1, 2))
		self.K_pro = np.reshape(np.asarray(K_pro), (1, 2))
		self.A = np.array([[1, timestep], [0, 1 - timestep * damper/mass]])
		self.B = np.array([[0], [timestep / mass]])
		self.Q = np.array([[10, 0], [0, 1]])
		self.R = 1.0
		self.delta = delta
		self.alpha = alpha


class Human(object):

	def __init__(self):
		params = Params()
		self.A = params.A
		self.B = params.B
		self.Q = params.Q
		self.R = params.R
		self.Abar = params.A + params.B @ params.F_0
		self.K_pro = params.K_pro
		self.K_opt = self.dare()
		self.s_star = None

	def task(self, s_star):
		self.s_star = np.reshape(np.asarray(s_star), (2, 1))

	def convention(self, F):
		self.Abar = self.A + self.B @ F
		self.K_opt = self.dare()

	def cost(self, xi_s, xi_z):
		effort, error = 0, 0
		for idx in xi_s:
			if idx in xi_z:
				effort += float(self.R * xi_z[idx]**2)
			e = self.s_star - xi_s[idx]
			error += float(np.transpose(e) @ self.Q @ e)
		return effort + error

	def dare(self):
		P = la.solve_discrete_are(self.Abar, self.B, self.Q, self.R)
		denominator = 1.0 / (self.R + self.B[1]**2 * P[1, 1])
		numerator = np.transpose(self.B) @ P @ self.Abar
		return denominator * numerator

	def control_pro(self, s):
		return self.K_pro @ (self.s_star - s)

	def control_opt(self, s):
		return self.K_opt @ (self.s_star - s)


class Robot(object):

	def __init__(self):
		params = Params()
		self.A = params.A
		self.B = params.B
		self.F = params.F_0
		self.n_steps = params.n_steps
		self.s_0 = params.s_0

	def convention(self, F):
		self.F = copy.deepcopy(F)

	def dynamics(self, s, z):
		return (self.A + self.B @ self.F) @ s + self.B @ z

	def rollout(self, control):
		xi_s, xi_z, s = {}, {}, self.s_0
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
		    position.append(float(xi_s[key][0]))
		plt.plot(time, position)

	def plot_input(self, xi_z):
		time = np.linspace(0, 1, self.n_steps)
		input = []
		for key in xi_z:
		    input.append(float(xi_z[key]))
		plt.plot(time[:-1], input)
