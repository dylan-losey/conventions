import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from clone import MLP
from respond import R_MLP
from ideal import STAR_MLP
from selfplay import MLP_MLP
import pickle



class I_MLP(nn.Module):

    def __init__(self):
        super(I_MLP, self).__init__()

        self.name = "models/inf_model.pt"
        self.n_steps = 10

        self.fc_1 = nn.Linear(4, 4)
        self.fc_2 = nn.Linear(4, 2)
        # model_dict = torch.load("models/h_model.pt", map_location='cpu')
        # self.load_state_dict(model_dict)

        self.rc_1 = nn.Linear(4, 8)
        self.rc_2 = nn.Linear(8, 8)
        self.rc_3 = nn.Linear(8, 2)

        # curr human
        self.human = MLP()
        model_dict = torch.load("models/h_model.pt", map_location='cpu')
        self.human.load_state_dict(model_dict)
        self.human.eval

    def prediction(self, x):
        h1 = self.fc_1(x)
        return torch.tanh(self.fc_2(h1))

    def policy(self, x):
        h1 = torch.tanh(self.rc_1(x))
        h2 = torch.tanh(self.rc_2(h1))
        return self.rc_3(h2)

    def rollout(self, s_star, s_0):
        error = 0.0
        s = torch.FloatTensor(s_0)
        for t in range(self.n_steps):
            x = torch.cat((s, s_star), 0)
            ah_hat = self.human.prediction(x).detach()
            ah = self.prediction(x)
            context = torch.cat((s, ah), 0)
            ar = self.policy(context)
            s = s + 0.1 * ar
            error += torch.norm(s - s_star)**2
            error += torch.tanh(ah[0]) + torch.tanh(ah[1])
            error += torch.norm(ah)
            error += 1.0 * torch.norm(ah - ah_hat)
        return error

    def loss(self):
        Q = 0.0
        g1 = torch.FloatTensor([1.0, 0.0])
        g2 = torch.FloatTensor([0.0, 1.0])
        for round in range(10):
            s_0 = np.random.random(2)
            for s_star in [g1, g2]:
                error = self.rollout(s_star, s_0)
                Q += error
        return Q


def main():

    EPOCH = 1000
    LR = 0.01
    LR_STEP_SIZE = 300
    LR_GAMMA = 0.1
    savename = "models/inf_model.pt"

    model = I_MLP()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    for idx in range(EPOCH):
        optimizer.zero_grad()
        loss = model.loss()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(idx, loss.item())
        torch.save(model.state_dict(), savename)


if __name__ == "__main__":
    main()
