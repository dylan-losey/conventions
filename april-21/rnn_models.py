import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
import copy
import numpy as np
import sys
import os



class RNNAE(nn.Module):

    def __init__(self):
        super(RNNAE, self).__init__()

        self.n_steps = 10
        self.n_tasks = 5
        self.omega = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0])

        self.h_start = None
        self.h_length = None
        self.h_go1 = None
        self.h_go2 = None

        self.hidden_size = 10
        self.input_size = 2
        self.output_size = 1

        self.i2o_1 = nn.Linear(self.hidden_size, 2*self.hidden_size)
        self.i2o_2 = nn.Linear(2*self.hidden_size, 2*self.hidden_size)
        self.i2o_3 = nn.Linear(2*self.hidden_size, self.output_size)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(self.input_size, self.hidden_size)


    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))


    # def human(self, s_star, t):
    #     noise = torch.randn(1) * 0.0
    #     ah = torch.tensor(0.0)
    #     if t >= self.h_start and t < (self.h_start + self.h_length):
    #         ah = (s_star - 0.6) / 0.4
    #         if s_star == 0.4 or s_star == 0.8:
    #             noise = torch.randn(1) * 0.1
    #     return ah.view(1) + noise


    def human(self, s_star, t):
        noise = torch.randn(1) * 0.05
        ah = torch.tensor(0.0)
        if s_star == 0.4 or s_star == 0.8:
            if t in self.h_go1:
                ah = (s_star - 0.6) / 0.2
        else:
            if t in self.h_go2:
                ah = (s_star - 0.6) / 0.4
        return ah.view(1) + noise


    def robot(self, input, hidden):
        output, hidden = self.lstm(input.view(1, 1, -1), hidden)
        h1 = self.relu(self.i2o_1(output[0,0,:]))
        h2 = self.relu(self.i2o_2(h1))
        return self.i2o_3(h2), hidden


    def rollout(self, s_star):
        error = 0.0
        s = torch.tensor(0.0).view(1)
        xi_s, xi_ah, xi_ar = [], [], []
        xi_s.append(s.item())
        hidden = self.init_hidden()
        for t in range(self.n_steps):
            ah = self.human(s_star, t)
            context = torch.cat((s, ah), 0)
            ar, hidden = self.robot(context, hidden)
            s = s + ar
            xi_s.append(s.item())
            xi_ah.append(ah.item())
            xi_ar.append(ar.item())
            error += (s - s_star)**2
        return error, xi_s, xi_ah, xi_ar


    def traj(self, idx):
        s_star = self.omega[idx]
        _, xi_s, xi_ah, xi_ar = self.rollout(s_star)
        return xi_s, xi_ah, xi_ar


    def loss(self):
        Q = 0.0
        self.h_start = np.random.randint(0, self.n_steps)
        self.h_length = 1 + np.random.randint(0, 5)
        self.h_go1 = np.random.randint(0, 4, 4)
        self.h_go2 = np.random.randint(4, 8, 4)
        for s_star in self.omega:
            error, _, _, _ = self.rollout(s_star)
            Q += error
        return Q


EPOCH = 10000
LR = 0.01
LR_STEP_SIZE = 2000
LR_GAMMA = 0.1
SAVENAME = "models/test-rnn-2.pt"


def main():

    model = RNNAE()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    for idx in range(EPOCH):

        optimizer.zero_grad()
        loss = model.loss()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(idx, loss.item())
        torch.save(model.state_dict(), SAVENAME)


if __name__ == "__main__":
    main()
