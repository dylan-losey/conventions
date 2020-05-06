import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle
import copy
import numpy as np
import matplotlib.pyplot as plt


# constrain the human to only apply inputs for a certain timespan

class Convention(nn.Module):

    def __init__(self):
        super(Convention, self).__init__()

        self.timestep = 0.05
        self.n_steps = 41
        self.A = torch.tensor([[1, self.timestep], [0, 1]])
        self.B = torch.tensor([[0], [self.timestep]])
        self.Qcost = torch.tensor([[10.0, 0.0], [0.0, 1.0]])
        self.Rcost = 1.0
        self.n_tasks = 5
        self.omega = np.zeros((2,self.n_tasks))
        self.omega[0,:] = np.linspace(0.2,1,self.n_tasks)
        self.s_0 = torch.tensor([[0.0],[0.0]])

        self.alpha = None
        self.fch1 = nn.Linear(4,4)
        self.fch2 = nn.Linear(4,1)
        self.fcr1 = nn.Linear(3,3)
        self.fcr2 = nn.Linear(3,1)

        self.cost = []
        self.diverg = []

    def dynamics(self, s_star, s):
        s_star_list = [s_star[0,0], s_star[1,0]]
        s_list = [s[0,0], s[1,0]]
        z = self.human(s_star_list, s_list)
        a = self.robot(s_list, z)
        s_next = self.A@s + self.B*a
        return s_next, z

    def human_initial(self, s_star, s):
        e = torch.tensor([[s_star[0,0]-s[0,0]],[s_star[1,0]-s[1,0]]])
        z = torch.tensor([[1.0, 0.1]]) @ e
        return z[0]

    def human(self, s_star, s):
        x = torch.FloatTensor(s_star + s)
        h1 = torch.tanh(self.fch1(x))
        h2 = torch.tanh(self.fch2(h1))
        return h2

    def robot(self, s, z):
        x = torch.cat((torch.FloatTensor(s), z), dim=0)
        h1 = torch.tanh(self.fcr1(x))
        return self.fcr2(h1)

    def cost_J(self, s_star, s_0):
        s = copy.deepcopy(s_0)
        effort, error, normal = 0, 0, 0
        for idx in range(self.n_steps - 1):
            e = s_star - s
            zt = self.human_initial(s_star, s)
            s, z = self.dynamics(s_star, s)
            error += torch.transpose(e, 0, 1) @ self.Qcost @ e
            if abs(z) > 0.01:
                effort += 1
            # effort += self.Rcost * z**2
            normal += (zt - z)**2
        e = s_star - s
        error += torch.transpose(e, 0, 1) @ self.Qcost @ e
        return error, effort, normal

    def cost_Q(self):
        expected_cost, expected_diverg = 0.0, 0.0
        for idx in range(self.n_tasks):
            s_star = torch.tensor([[self.omega[0,idx]],[self.omega[1,idx]]])
            error, effort, normal = self.cost_J(s_star, self.s_0)
            expected_cost += (error + effort) / self.n_tasks
            expected_diverg += normal / self.n_tasks
        expected_total = expected_cost + self.alpha * expected_diverg
        self.cost.append(expected_cost.item())
        self.diverg.append(expected_diverg.item())
        return expected_total

    def rollout(self, s_star):
        s = copy.deepcopy(self.s_0)
        xi_s, xi_z = np.zeros((2,self.n_steps)), np.zeros((1,self.n_steps-1))
        for idx in range(self.n_steps - 1):
            xi_s[0,idx] = s[0,0]
            xi_s[1,idx] = s[1,0]
            s, z = self.dynamics(s_star, s)
            xi_z[0,idx] = z[0]
        xi_s[0,self.n_steps-1] = s[0,0]
        xi_s[1,self.n_steps-1] = s[1,0]
        return xi_s, xi_z

    def rollout_J(self, s_star, s_0):
        s = copy.deepcopy(s_0)
        xi_s, xi_z = np.zeros((2,self.n_steps)), np.zeros((1,self.n_steps-1))
        for idx in range(self.n_steps - 1):
            xi_s[0,idx] = s[0,0]
            xi_s[1,idx] = s[1,0]
            s, z = self.dynamics(s_star, s)
            xi_z[0,idx] = z[0]
        xi_s[0,self.n_steps-1] = s[0,0]
        xi_s[1,self.n_steps-1] = s[1,0]
        return xi_s, xi_z

    def traj_Q(self):
        time = np.linspace(0,1,self.n_steps)
        for idx in range(self.n_tasks):
            s_star = torch.tensor([[self.omega[0,idx]],[self.omega[1,idx]]])
            xi_s, xi_z = self.rollout_J(s_star, self.s_0)
            plt.plot(1,s_star[0,0], 'kx')
            plt.plot(time, xi_s[0,:])
            plt.plot(time[:-1], xi_z[0,:])
        plt.show()
