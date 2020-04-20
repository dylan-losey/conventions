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


class Human(object):

    def __init__(self):
        self.name = "human_2"

    def human(self, s_star, t):
        ah = 0.0
        if s_star == 0.4 or s_star == 0.8:
            if t == 0:
                ah = (s_star - 0.6) / 0.2
        elif s_star == 0.2 or s_star == 1.0:
            if t == 4:
                ah = (s_star - 0.6) / 0.4
        ah = torch.tensor(ah)
        return ah.view(1)


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.Human = Human()

        self.n_steps = 10
        self.n_tasks = 5
        self.omega = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0])

        self.h_input_size = 2
        self.h_output_size = 1
        self.h_i2o_1 = nn.Linear(self.h_input_size, 2*self.h_input_size)
        self.h_i2o_2 = nn.Linear(2*self.h_input_size, 2*self.h_input_size)
        self.h_i2o_3 = nn.Linear(2*self.h_input_size, self.h_output_size)


    def H(self, input):
        h1 = torch.tanh(self.h_i2o_1(input))
        h2 = torch.tanh(self.h_i2o_2(h1))
        return self.h_i2o_3(h2)


    def rollout(self, s_star):
        error = 0.0
        xi_ah = []
        for t in range(self.n_steps):
            h_context = torch.cat((s_star.view(1), torch.tensor(float(t)).view(1)), 0)
            ah_star = self.Human.human(s_star, t)
            ah = self.H(h_context)
            error += (ah - ah_star)**2
            xi_ah.append(ah.item())
        return error, xi_ah


    def loss(self):
        Q = 0.0
        for s_star in self.omega:
            error, xi_ah = self.rollout(s_star)
            Q += error
        return Q


EPOCH = 10000
LR = 0.1
LR_STEP_SIZE = 1000
LR_GAMMA = 0.1
SAVENAME = "models/test-human-2.pt"


def main():

    model = MLP()
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
