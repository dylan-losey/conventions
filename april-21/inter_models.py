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
from rnn_models import RNNAE
import sys



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


class Robot(object):

    def __init__(self):
        self.model = RNNAE()
        modelname = "models/test-rnn-1.pt"
        model_dict = torch.load(modelname, map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval

    def robot(self, input, hidden):
        return self.model.robot(input, hidden)

    def init_hidden(self):
        return self.model.init_hidden()


class TEAM(nn.Module):

    def __init__(self):
        super(TEAM, self).__init__()

        self.Human = Human()
        self.Robot = Robot()

        self.wc = None
        self.wh = None
        self.wr = None

        self.n_steps = 10
        self.n_tasks = 5
        self.omega = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0])

        self.etask = None
        self.ehuman = None
        self.erobot = None

        self.h_input_size = 2
        self.h_output_size = 1
        self.h_i2o_1 = nn.Linear(self.h_input_size, 2*self.h_input_size)
        self.h_i2o_2 = nn.Linear(2*self.h_input_size, 2*self.h_input_size)
        self.h_i2o_3 = nn.Linear(2*self.h_input_size, self.h_output_size)

        model_dict = torch.load("models/test-human-2.pt", map_location='cpu')
        self.load_state_dict(model_dict)

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


    def H(self, input):
        h1 = torch.tanh(self.h_i2o_1(input))
        h2 = torch.tanh(self.h_i2o_2(h1))
        return self.h_i2o_3(h2)


    def R(self, input, hidden):
        output, hidden = self.lstm(input.view(1, 1, -1), hidden)
        h1 = self.relu(self.i2o_1(output[0,0,:]))
        h2 = self.relu(self.i2o_2(h1))
        return self.i2o_3(h2), hidden


    def rollout(self, s_star):
        etask, ehuman, erobot = 0.0, 0.0, 0.0
        s = torch.tensor(0.0).view(1)
        xi_s, xi_ah, xi_ar = [], [], []
        xi_s.append(s.item())
        hidden_star = self.Robot.init_hidden()
        hidden = self.init_hidden()
        for t in range(self.n_steps):
            h_context = torch.cat((s_star.view(1), torch.tensor(float(t)).view(1)), 0)
            ah_star = self.Human.human(s_star, t)
            ah = self.H(h_context)
            r_context = torch.cat((s, ah), 0)
            ar_star, hidden_star = self.Robot.robot(r_context, hidden_star)
            ar, hidden = self.R(r_context, hidden)
            s = s + ar
            xi_s.append(s.item())
            xi_ah.append(ah.item())
            xi_ar.append(ar.item())
            etask += (s - s_star)**2
            ehuman += (ah - ah_star)**2
            erobot += (ar - ar_star)**2
        return etask, ehuman, erobot, xi_s, xi_ah, xi_ar


    def traj(self, idx):
        s_star = self.omega[idx]
        _, _, _, xi_s, xi_ah, xi_ar = self.rollout(s_star)
        return xi_s, xi_ah, xi_ar


    def loss(self):
        setask, sehuman, serobot = 0.0, 0.0, 0.0
        for s_star in self.omega:
            etask, ehuman, erobot, _, _, _ = self.rollout(s_star)
            setask += etask
            sehuman += ehuman
            serobot += erobot
        self.etask = setask
        self.ehuman = sehuman
        self.erobot = serobot
        Q = self.wc * setask + self.wh * sehuman + self.wr * serobot
        return Q


EPOCH = 10000
LR = 0.01
LR_STEP_SIZE = 1000
LR_GAMMA = 0.1
SAVENAME = "models/team-1.pt"


def main():

    wc = float(sys.argv[1])
    wh = float(sys.argv[2])
    wr = float(sys.argv[3])

    model = TEAM()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    model.wc = wc
    model.wh = wh
    model.wr = wr

    for idx in range(EPOCH):

        optimizer.zero_grad()
        loss = model.loss()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(idx, loss.item(), model.etask.item(), model.ehuman.item(), model.erobot.item())
        torch.save(model.state_dict(), SAVENAME)


if __name__ == "__main__":
    main()
